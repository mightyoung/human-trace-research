# -*- coding: utf-8 -*-
"""S3：Q1 同机位长程关联（遮挡断链后接续）—— 几何证据层 + 受约束全局拼接。

┌─ 背景（本脚本实测，非推测）───────────────────────────────────────────────┐
│ 现有生产口径 `tools/track_stitching.py`（→ `stitch_clean2` 实体）=            │
│   **归一化坐标的投影残差** `cos≥0.80 ∧ pe_adj≤0.08` + **mutual-best**        │
│   间隔视界：普通 30s / 静止对 120s。                                        │
│ 实测其在 splits_v5 val 同机位对上 P 0.639 / R 0.885 / F1 0.742              │
│ → 短间隔已经很稳（且 cos≥0.80 在这批对上**不 binding**）。                   │
└──────────────────────────────────────────────────────────────────────────┘

S3 的两条**可量化增量**：
  A. **几何证据层**：新增**与间隔无关**的尺度连续性 `d_h`（|h_A−h_B|/h_A，框高归一化）。
     生产门叠加 `d_h≤0.5` → 同序列 A/B：**P 0.639→0.676，R 不变，F1 0.742→0.767**（FP −15%）。
  B. **拼接从 mutual-best 升级为受约束路径覆盖**：生产用 mutual-best（互为最优）导致
     "一人多段 / 一对多竞争"时**结构性漏接**；改用 1-后继/1-前驱的全局路径覆盖
     （dt≥0 ⇒ 时间互斥自动满足；仍禁分支 ⇒ 保住"防链式误连"），在**同一证据标准**下回收漏接。

三条**负结果**（写进报告，避免后人重走）：
  * 方向一致性 `d_dir` / 速度连续性 `d_spd` 在本数据**无效**（0.2s 采样 + 检测抖动主导）；
  * 把视界从 30s 拉到 300s（"离开后回到原处"位置先验）在 val 上**精度崩塌**
    （长间隔子集 P 0.071）→ **长间隔同机位重认不可解**，生产 30s 视界已近最优；
  * `e_ext`（匀速外推残差）只在短间隔有效，长间隔对两类都爆炸。

用法：
  python tools/gil_frag_link.py                 # 校准 + 对账（只读）
  python tools/gil_frag_link.py --apply         # 写 GIL：edge(kind=stitch,tier=kin) + 重建 frag
  python tools/gil_frag_link.py --long-gap      # 打开长间隔扩展（默认关；仅用于复现负结果）
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
import track_stitching as TS  # noqa: E402  复用生产 pe 口径
from gil import GILStore  # noqa: E402

DET_DIR = os.path.join(ROOT, "output_crosscam", "cleanset_v2")
CAMS = ["305", "316", "318", "320", "323", "343"]
RES = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
FPS = 25.0
K_VEL = 4             # 尾/头速度采样步数（×0.2s ≈ 0.8s 基线）
PROD_COS, PROD_PE = 0.80, 0.08      # 生产门
GAP_MAX, GAP_STILL = 750, 3000      # 生产间隔视界（帧）：30s / 120s（静止对）
T_DH = 0.50           # ★ S3 新增：尺度连续性门
DT_WIN = 300.0        # 候选枚举上限（含长间隔，用于复现负结果）


def load_traj(cam):
    tr = collections.defaultdict(list)
    with open(os.path.join(DET_DIR, f"{cam}_detections.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for tid, b in zip(d["track_ids"], d["boxes"]):
                x1, y1, x2, y2 = b
                tr[int(tid)].append((fr, (x1 + x2) * 0.5, y2, y2 - y1))
    out = {}
    for tid, s in tr.items():
        s.sort()
        a = np.asarray(s, np.float32)
        out[tid] = dict(T=a[:, 0] / FPS, U=a[:, 1], V=a[:, 2], H=a[:, 3])
    return out


def tail_state(tr):
    T, U, V, H = tr["T"], tr["U"], tr["V"], tr["H"]
    n = len(T)
    k = min(K_VEL, n - 1)
    if n >= 2 and T[-1] - T[-1 - k] > 1e-6:
        dt = T[-1] - T[-1 - k]
        vu, vv = (U[-1] - U[-1 - k]) / dt, (V[-1] - V[-1 - k]) / dt
    else:
        vu = vv = 0.0
    return dict(u=float(U[-1]), v=float(V[-1]), vu=float(vu), vv=float(vv),
                h=float(np.median(H[-min(3, n):])), has_vel=(n >= 2))


def head_state(tr):
    T, U, V, H = tr["T"], tr["U"], tr["V"], tr["H"]
    n = len(T)
    k = min(K_VEL, n - 1)
    if n >= 2 and T[k] - T[0] > 1e-6:
        dt = T[k] - T[0]
        vu, vv = (U[k] - U[0]) / dt, (V[k] - V[0]) / dt
    else:
        vu = vv = 0.0
    return dict(u=float(U[0]), v=float(V[0]), vu=float(vu), vv=float(vv),
                h=float(np.median(H[:min(3, n)])), has_vel=(n >= 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/frag_link_v1.json")
    ap.add_argument("--splits", default="output_crosscam/analysis/splits_v5.json")
    ap.add_argument("--dt-win", type=float, default=DT_WIN)
    ap.add_argument("--long-gap", action="store_true", help="打开长间隔扩展（默认关；负结果）")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    st = GILStore(os.path.join(ROOT, a.store), create=False)
    vb = st.vectors(load=True)
    rows = list(st.con.execute(
        "SELECT id,cam,tid,t_first,t_last,u,v,h_med,entity_id FROM obs "
        "WHERE tombstone=0 ORDER BY id"))
    oid = np.array([r["id"] for r in rows])
    id2row = {int(x): i for i, x in enumerate(oid)}
    N = len(rows)
    E = np.zeros((N, st.dim), np.float32)
    for b in vb.blocks():
        Eb, ids, i2r = vb._cache[b]
        for k, i in i2r.items():
            if int(k) in id2row:
                E[id2row[int(k)]] = Eb[i]
    S = E @ E.T

    TRAJ = {c: load_traj(c) for c in CAMS}
    TAIL = {int(r["id"]): (tail_state(TRAJ[r["cam"]][int(r["tid"])])
                           if int(r["tid"]) in TRAJ[r["cam"]] else None) for r in rows}
    HEAD = {int(r["id"]): (head_state(TRAJ[r["cam"]][int(r["tid"])])
                           if int(r["tid"]) in TRAJ[r["cam"]] else None) for r in rows}
    # 生产摘要（归一化坐标 + 原始帧）
    PROD = {}
    for cam in CAMS:
        raw = collections.defaultdict(list)
        with open(os.path.join(DET_DIR, f"{cam}_detections.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                fr = d["frame"]
                for tid, b in zip(d["track_ids"], d["boxes"]):
                    raw[int(tid)].append((fr, b))
        W, H = RES[cam]
        for tid, obs in raw.items():
            obs.sort()
            PROD[(cam, tid)] = TS.track_summary(obs, W, H, 15)
    print(f"[s3] 节点 {N}；尾/头状态齐备 {sum(1 for v in TAIL.values() if v)}")

    def prod_feats(cam, ta, tb):
        sa, sb = PROD[(cam, ta)], PROD[(cam, tb)]
        gap = sb["f0"] - sa["f1"]
        if gap < 0:
            return None
        v_ok = all(abs(v) < 0.002 for v in (sa["v1"][0], sa["v1"][1], sb["v0"][0], sb["v0"][1]))
        dr = TS.dist(sa["p1"], sb["p0"])
        still = v_ok and dr < 0.05
        if gap > (GAP_STILL if still else GAP_MAX):
            return None
        g = max(gap, 0)
        pb = (sa["p1"][0] + sa["v1"][0] * g, sa["p1"][1] + sa["v1"][1] * g)
        ea, eb = TS.edge_dist(sa["p1"]) < 0.10, TS.edge_dist(sb["p0"]) < 0.10
        pf = TS.dist(pb, sb["p0"]) if g > 0 else dr
        pfa = dr if (TS.out_of_frame(pb) and ea) else pf
        pa = (sb["p0"][0] - sb["v0"][0] * g, sb["p0"][1] - sb["v0"][1] * g)
        pbm = TS.dist(pa, sa["p1"]) if g > 0 else dr
        pba = dr if (TS.out_of_frame(pa) and eb) else pbm
        return dict(pe_adj=float(min(pfa, pba)), still=bool(still), gap_s=float(gap / FPS))

    # ---------------- 候选对 ----------------
    cand = []
    for cam in CAMS:
        idx = [i for i, r in enumerate(rows) if r["cam"] == cam]
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                if TAIL[int(oid[i])] is None or HEAD[int(oid[j])] is None:
                    continue
                ra, rb = rows[i], rows[j]
                if ra["t_first"] > rb["t_first"]:
                    i, j, ra, rb = j, i, rb, ra
                dt = rb["t_first"] - ra["t_last"]
                if dt < 0 or dt > a.dt_win:
                    continue
                A, B = TAIL[int(oid[i])], HEAD[int(oid[j])]
                # 生产 pe
                pp = prod_feats(cam, int(ra["tid"]), int(rb["tid"]))
                # S3 几何（框高归一化）
                hA = max(A["h"], 1e-6)
                d_disp = float(np.hypot(B["u"] - A["u"], B["v"] - A["v"]) / hA)
                d_h = float(abs(A["h"] - B["h"]) / hA)
                na, nb = np.hypot(A["vu"], A["vv"]), np.hypot(B["vu"], B["vv"])
                d_dir = None
                if na > 1e-6 and nb > 1e-6:
                    c = (A["vu"] * B["vu"] + A["vv"] * B["vv"]) / (na * nb)
                    d_dir = float((1.0 - np.clip(c, -1, 1)) / 2.0)
                f = dict(dt=float(dt), cos=float(S[i, j]), cam=cam,
                         d_disp=d_disp, d_h=d_h, d_dir=d_dir,
                         d_spd=float(abs(na - nb) / (0.5 * (na + nb) + 1e-6)),
                         pe_adj=(pp["pe_adj"] if pp else None),
                         still=(pp["still"] if pp else None),
                         in_horizon=pp is not None)
                cand.append((cam, i, j, dt, f))
    n_hz = sum(1 for *_, f in cand if f["in_horizon"])
    print(f"[s3] 候选对 {len(cand)}（dt≤{a.dt_win:.0f}s）；生产视界内 {n_hz}"
          f"（{100*n_hz/max(len(cand),1):.0f}%）")

    # ---------------- 规则 ----------------
    def G_prod(f):                       # 生产门（视界内）
        return f["in_horizon"] and f["cos"] >= PROD_COS and f["pe_adj"] <= PROD_PE

    def G_s3(f):                         # ★ S3 门 = 生产门 + 尺度连续性
        return G_prod(f) and f["d_h"] is not None and f["d_h"] <= T_DH

    def G_ext(f):                        # 长间隔扩展（负结果；默认关）
        if G_s3(f):
            return True
        return (f["dt"] > 30.0 and f["cos"] >= 0.90 and f["d_h"] <= 0.50
                and f["d_disp"] <= 0.05)

    def link_rule(f):
        return (G_ext(f) if a.long_gap else G_s3(f))

    # ---------------- val 校准 / A-B ----------------
    key2i = {(r["cam"], int(r["tid"])): i for i, r in enumerate(rows)}
    fbp = {}
    for cam, i, j, dt, f in cand:
        fbp[frozenset(((cam, int(rows[i]["tid"])), (cam, int(rows[j]["tid"]))))] = (i, j, f)
    sp = json.load(open(os.path.join(ROOT, a.splits), encoding="utf-8"))["val"]

    def collect(pairs):
        out = []
        for c, x, y in pairs:
            it = fbp.get(frozenset(((c, int(x)), (c, int(y)))))
            if it:
                out.append(it)
        return out

    VP, VN = collect(sp["pos_pairs"]), collect(sp["neg_pairs"])

    def prf(fn, P_, N_):
        tp = sum(1 for i, j, f in P_ if fn(f))
        fp = sum(1 for i, j, f in N_ if fn(f))
        fn_ = len(P_) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn_) if tp + fn_ else 0.0
        return dict(tp=tp, fp=fp, fn=fn_, precision=p, recall=r,
                    f1=2 * p * r / (p + r) if p + r else 0.0,
                    n_pos=len(P_), n_neg=len(N_))

    W_ = [(i, j, f) for i, j, f in VP if f["in_horizon"]]
    X_ = [(i, j, f) for i, j, f in VN if f["in_horizon"]]
    LP = [(i, j, f) for i, j, f in VP if not f["in_horizon"]]
    LN = [(i, j, f) for i, j, f in VN if not f["in_horizon"]]
    B_prod = prf(G_prod, W_, X_)
    B_s3 = prf(G_s3, W_, X_)
    B_pc = prf(G_s3, VP, VN)             # 含视界外（pn 全量口径）

    print(f"\n=== val 同序列 A/B（视界内可评 pos {len(W_)} / neg {len(X_)}）===")
    print(f"  R_prod 生产门  cos≥{PROD_COS} ∧ pe≤{PROD_PE}                "
          f"P {B_prod['precision']:.3f} R {B_prod['recall']:.3f} F1 {B_prod['f1']:.3f}"
          f"  (TP {B_prod['tp']} FP {B_prod['fp']} FN {B_prod['fn']})")
    print(f"  R_s3  +尺度 d_h≤{T_DH}                        "
          f"P {B_s3['precision']:.3f} R {B_s3['recall']:.3f} F1 {B_s3['f1']:.3f}"
          f"  (TP {B_s3['tp']} FP {B_s3['fp']} FN {B_s3['fn']})   "
          f"→ 精度 {B_s3['precision']-B_prod['precision']:+.3f} / 召回 "
          f"{B_s3['recall']-B_prod['recall']:+.3f}")

    print(f"\n=== 负结果（写进报告）===")
    print(f"  · 视界外（dt>30s 且非静止对，生产结构性不可达）：pos {len(LP)} / neg {len(LN)}")
    if LP:
        r = prf(G_ext, LP, LN)
        print(f"      长间隔位置先验（d_disp≤0.05 ∧ cos≥0.90）：P {r['precision']:.3f} "
              f"R {r['recall']:.3f} → **不可用**")
    for nm, key in (("d_dir 方向一致性", "d_dir"), ("d_spd 速度连续性", "d_spd")):
        pv = sorted(f[key] for i, j, f in W_ if f[key] is not None)
        nv = sorted(f[key] for i, j, f in X_ if f[key] is not None)
        if pv and nv:
            print(f"  · {nm}: pos p50 {pv[len(pv)//2]:.3f} vs neg p50 {nv[len(nv)//2]:.3f} "
                  f"→ **不可分**")
    if B_pc["f1"] < B_s3["f1"]:
        print(f"  · 长间隔扩展（--long-gap）全量口径 F1 {B_pc['f1']:.3f} < 视界内 {B_s3['f1']:.3f}")

    # ---------------- 拼接：mutual-best vs 受约束路径覆盖 ----------------
    def build(rule, mode):
        c = [(cam, i, j, f) for cam, i, j, dt, f in cand if rule(f)]
        c.sort(key=lambda x: -(x[3]["cos"] - 2.0 * (x[3]["pe_adj"] or 0.0)))
        succ, pred, links = {}, {}, []
        if mode == "mb":                 # 生产：互为最优
            bs, bp = {}, {}
            for cam, i, j, f in c:
                bs.setdefault(i, (cam, i, j, f))
                bp.setdefault(j, (cam, i, j, f))
            for i, (cam, i2, j, f) in bs.items():
                if j in bp and bp[j][1] == i:
                    links.append((cam, i, j, f))
        else:                            # S3：受约束路径覆盖（1-后继/1-前驱）
            for cam, i, j, f in c:
                if i in succ or j in pred:
                    continue
                succ[i], pred[j] = j, i
                links.append((cam, i, j, f))
        return links

    def ents(links, seed_entities=True):
        par = {}
        def find(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        if seed_entities:               # 先按现有实体成员预合并
            byent = collections.defaultdict(list)
            for r in rows:
                if r["entity_id"] is not None:
                    byent[r["entity_id"]].append(id2row[int(r["id"])])
            for ms in byent.values():
                for q in ms[1:]:
                    ri, rj = find(ms[0]), find(q)
                    if ri != rj:
                        par[ri] = rj
        n_merge = 0
        for cam, i, j, f in links:
            ri, rj = find(i), find(j)
            if ri != rj:
                par[ri] = rj
                n_merge += 1
        comp = collections.defaultdict(list)
        for i in list(par):
            comp[find(i)].append(i)
        # 未被任何实体/边涉及的孤立 obs
        for i in range(N):
            if i not in par:
                comp[("solo", i)] = [i]
        return comp, n_merge

    L_mb = build(G_prod, "mb")          # ① 生产：门 + mutual-best
    L_mb_s3 = build(G_s3, "mb")         # ② +尺度门 d_h（仍 mutual-best）
    L_pc = build(link_rule, "pc")       # ③ +受约束路径覆盖
    # ★ 同口径对比：只用本次 clean2 重算的链接构实体（**不** seed 现有实体，
    #   因为现有 stitch_clean2 的口径不同——其 stitch_{cam}.csv 在 clean v1（未掩膜）轨迹集上算，
    #   316/323 与 clean2 重算的 pe 中位差 0.088 且多出 764/1471 对候选；318/320 才逐位一致）
    comp_mb, _ = ents(L_mb, seed_entities=False)
    comp_mb_s3, _ = ents(L_mb_s3, seed_entities=False)
    comp, n_merge_s3 = ents(L_pc, seed_entities=False)
    eo = {int(r["id"]): r["entity_id"] for r in rows}
    n_cross = sum(1 for cam, i, j, f in L_pc
                  if eo[int(oid[i])] is not None and eo[int(oid[j])] is not None
                  and eo[int(oid[i])] != eo[int(oid[j])])
    n_agree = sum(1 for cam, i, j, f in L_pc
                  if eo[int(oid[i])] is not None and eo[int(oid[i])] == eo[int(oid[j])])
    n_conflict = 0
    conflict_list = []
    for cam, i, j, dt, f in cand:
        if eo[int(oid[i])] is None or eo[int(oid[i])] != eo[int(oid[j])]:
            continue
        if dt <= 30 and f["d_disp"] > 1.5 * dt + 0.3:
            n_conflict += 1
            conflict_list.append(dict(cam=cam, a=f"{cam}|{rows[i]['tid']}", b=f"{cam}|{rows[j]['tid']}",
                                      dt=round(f["dt"], 2), d_disp=round(f["d_disp"], 3),
                                      cos=round(f["cos"], 4)))
    conflict_list.sort(key=lambda x: -x["d_disp"])

    # d_h 门扫描（显示工作点非单点）
    dh_scan = []
    for th in np.round(np.arange(0.10, 1.21, 0.10), 3):
        r = prf(lambda f, th=th: G_prod(f) and f["d_h"] is not None and f["d_h"] <= th, W_, X_)
        dh_scan.append([float(th), round(r["precision"], 4), round(r["recall"], 4), round(r["f1"], 4)])

    print(f"\n=== 同机位拼接三步消融（**同一 clean2 口径**）===")
    print(f"  ① 生产 门+mulual-best      边 {len(L_mb):4d} → 实体 {len(comp_mb):4d}"
          f"（多段 {sum(1 for v in comp_mb.values() if len(v) > 1)}）")
    print(f"  ② +尺度门 d_h≤{T_DH}           边 {len(L_mb_s3):4d} → 实体 {len(comp_mb_s3):4d}"
          f"（多段 {sum(1 for v in comp_mb_s3.values() if len(v) > 1)}）")
    print(f"  ③ +受约束路径覆盖          边 {len(L_pc):4d} → 实体 {len(comp):4d}"
          f"（多段 {sum(1 for v in comp.values() if len(v) > 1)}）")
    print(f"  ★ 合计：实体 {len(comp_mb)} → {len(comp)}（−{len(comp_mb)-len(comp)}，"
          f"{100*(len(comp_mb)-len(comp))/max(len(comp_mb),1):.0f}%）")
    print(f"  ⚠ 现有 stitch_clean2 = 411 实体（口径=clean v1 CSV + 人工修正，**不可直接相减**）；"
          f"本次边与其一致 {n_agree} / 跨其边界 {n_cross}")
    print(f"  运动学矛盾（现同实体 ∧ dt≤30s ∧ 位移>1.5·dt+0.3）：{n_conflict} 条")
    print(f"  d_h 门扫描 F1：{[(x[0], x[3]) for x in dh_scan]}")

    out = dict(
        params=dict(dt_win=a.dt_win, k_vel=K_VEL, t_dh=T_DH,
                    prod=dict(cos=PROD_COS, pe=PROD_PE, gap_max_frames=GAP_MAX,
                              gap_still_frames=GAP_STILL),
                    long_gap_enabled=bool(a.long_gap),
                    note="S3 门 = 生产门 cos/pe + 尺度连续性 d_h；拼接 = 受约束路径覆盖"),
        n_obs=N, n_cand=len(cand), n_in_horizon=n_hz,
        val=dict(prod=B_prod, s3=B_s3, n_pos_horizon=len(W_), n_neg_horizon=len(X_),
                 n_pos_out=len(LP), n_neg_out=len(LN),
                 out_of_horizon_res=(prf(G_ext, LP, LN) if LP else None)),
        stitch=dict(step1_mb_prod=dict(links=len(L_mb), entities=len(comp_mb)),
                    step2_mb_dh=dict(links=len(L_mb_s3), entities=len(comp_mb_s3)),
                    step3_pc=dict(links=len(L_pc), entities=len(comp)),
                    entities_mutual_best=len(comp_mb), entities_path_cover=len(comp),
                    multi_frag_mutual_best=sum(1 for v in comp_mb.values() if len(v) > 1),
                    multi_frag_path_cover=sum(1 for v in comp.values() if len(v) > 1),
                    merges_path_cover=n_merge_s3,
                    agree_with_stitch_clean2=n_agree, cross_stitch_clean2=n_cross,
                    conflicts_with_stitch_clean2=n_conflict,
                    baseline_note="stitch_clean2=411 实体，口径为 clean v1 CSV + 人工修正，"
                                  "与本次 clean2 重算不可直接相减",
                    dh_scan=dh_scan),
        conflicts_with_current_entity=conflict_list[:200],
        links=[dict(cam=cam, a=f"{cam}|{rows[i]['tid']}", b=f"{cam}|{rows[j]['tid']}",
                    dt=round(f["dt"], 2), cos=round(f["cos"], 4),
                    pe_adj=(round(f["pe_adj"], 4) if f["pe_adj"] is not None else None),
                    d_h=round(f["d_h"], 4), d_disp=round(f["d_disp"], 4),
                    long_gap=bool(f["dt"] > 30),
                    same_entity=bool(eo[int(oid[i])] == eo[int(oid[j])]))
               for cam, i, j, f in L_pc],
    )
    json.dump(out, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[out] {a.out}")

    if a.apply:
        # ★ 只写**增量证据层**：edge(kind=stitch_kin) + frag（S3 精化链）
        #   独立 kind → 不覆盖 migrate 生成的 stitch(intra) 边，保住版本链；
        #   绝不改 `entity` 表——它是 relay/C0 的消费口径，且 splits_v5 冻结了其 md5。
        st.con.execute("DELETE FROM edge WHERE kind='stitch_kin'")
        for cam, i, j, f in L_pc:
            st.add_edge("stitch_kin", "obs", int(oid[i]), "obs", int(oid[j]),
                        dt=round(f["dt"], 3), score=round(f["cos"], 6), status="active",
                        tier="kin",
                        evidence=dict(cos=round(f["cos"], 5),
                                      pe_adj=(round(f["pe_adj"], 5) if f["pe_adj"] is not None else None),
                                      d_h=round(f["d_h"], 4), d_disp=round(f["d_disp"], 4),
                                      long_gap=bool(f["dt"] > 30), src="gil_frag_link_v1"))
        st.con.execute("DELETE FROM frag WHERE src LIKE 'migrate:%' OR src LIKE 'frag_link_v1:%'")
        n_frag = 0
        for v in comp.values():
            v2 = sorted(v, key=lambda q: rows[q]["t_first"])
            st.add_frag(rows[v2[0]]["cam"], [int(oid[x]) for x in v2],
                        rows[v2[0]]["t_first"], rows[v2[-1]]["t_last"], src="frag_link_v1:kin")
            n_frag += 1
        inl = {int(oid[x]) for v in comp.values() for x in v}
        for r in rows:
            if int(r["id"]) not in inl:
                st.add_frag(r["cam"], [int(r["id"])], r["t_first"], r["t_last"],
                            src="frag_link_v1:singleton")
                n_frag += 1
        st.meta_set(frag_link=dict(
            version="frag_link_v1", method="生产门(cos/pe) + 尺度连续性 d_h + 受约束路径覆盖",
            t_dh=T_DH, dt_win=a.dt_win, n_edges=len(L_pc), n_frag=n_frag,
            entity_untouched=True,
            note="frag=同机位片段链（S3 接续单位）；entity 表保持不变（生产口径）"),
            )
        st.commit()
        mf = st.snapshot()
        print(f"[gil] stitch_kin 边 {len(L_pc)}；frag 重建 {n_frag}（entity 未改动）")
        print(f"[gil] manifest -> {mf}")
        print("[gil] stats:", json.dumps(st.stats(), ensure_ascii=False))
    st.close()


if __name__ == "__main__":
    main()
