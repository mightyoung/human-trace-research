# -*- coding: utf-8 -*-
"""S2：gperson = 「时间互斥 + 门户转移先验 + 外观」约束下的层次聚类。

动机：relay 图的 mutual-best **逐对独立打分**，从不检查对端能否同时成立 →
      实测至少 **37 条**跨机位边在逻辑上不可能为真
      （见 output_crosscam/analysis/truth/ADJUDICATION-2026-10-14.md）。

节点 = obs（= 片段/tid，546 个）。三层：

  ① **硬约束 · 时间互斥**：同一人不可能同时出现在两处 → 时间重叠者绝不同人（不可合并）。
     ★ 唯一豁免：**重叠视野机位对 318×320**（同区域两视角，同一人可同时出现；
       实测 8 对人工 GT **全部时间重叠**）。
  ② **同机位**（碎片接续，Q1）：无 dt 要求；证据 = 外观 cos；阈值 **T_same** 在 val 校准。
  ③ **跨机位**（Q2/Q3）：非重叠视野对要求 `0 ≤ dt ≤ DT_MAX`（先离开再到达）；
     证据 = `cos + λ·ln w_dt(dt)`；阈值 **T_cross** 取 relay v2 冻结的 `sim_thr=0.943`
     （不碰 test/人工真值，避免污染）。

方法：**约束凝聚聚类**——只在"合并后簇内仍无互斥对"时合并；默认 `max` 连接度
      （门户交接是"存在性"主张：只要存在一对时间兼容的强证据即应连接）+ **规模上限 8**
      （防单链串接；实测该组合在人工真值上 P 1.000 / R 0.538，且无 >8 成员的簇）。
不变量：任何 gperson 内不得存在互斥对（脚本内硬断言）。

输出：GIL `gperson`/`membership` 落库（`--apply`）+ `analysis/gil/gperson_v1.json`
用法：DET_VIEW=clean2 python tools/gil_gperson.py [--apply] [--no-overlap]
"""
import argparse
import heapq
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
from gil import GILStore  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
CAMS = ["305", "316", "318", "320", "323", "343"]
OVERLAP_PAIRS = {frozenset(("318", "320"))}
DT_MAX = 30.0
T_CROSS_FROZEN = 0.943          # = relay v2 params.sim_thr（已有冻结工作点）
GT_318_320 = [(1, 1), (14, 3), (37, 11), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28)]


def w_dt_fn():
    d = json.load(open(os.path.join(ANA, "relay_v2", "dist_v1.json"), encoding="utf-8"))
    G = np.asarray(d["grid_x"], float); F = np.asarray(d["grid_f"], float)
    FMAX = float(F.max()); lam = float(d.get("lambda_default", 0.01))

    def w(dt):
        x = float(np.clip(dt, G[0], G[-1]))
        i = max(0, min(int(np.searchsorted(G, x) - 1), len(G) - 2))
        t = (x - G[i]) / (G[i + 1] - G[i])
        return float(max(F[i] * (1 - t) + F[i + 1] * t, 1e-9) / FMAX)
    return w, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/gperson_v1.json")
    ap.add_argument("--splits", default="output_crosscam/analysis/splits_v5.json")
    ap.add_argument("--t-cross", type=float, default=T_CROSS_FROZEN)
    ap.add_argument("--t-same", type=float, default=None, help="默认在 val 上校准")
    ap.add_argument("--no-overlap", action="store_true",
                    help="不合并重叠视野机位对（318×320 交由 C0-A 同步映射支路）")
    ap.add_argument("--linkage", default="max", choices=["avg", "max", "complete"],
                    help="簇间连接度：max（默认；存在性=门户交接语义）/ complete（最严，无串接）"
                         "/ avg（UPGMA）")
    ap.add_argument("--mutual", action="store_true",
                    help="互最佳守卫：仅当彼此是对方最佳伙伴时才合并（防单链链式串接；"
                         "与项目既有 mutual-knn 纪律一致）")
    ap.add_argument("--max-size", type=int, default=8,
                    help="簇成员数上限（默认 8；防单链串接 → 实测最佳工作点）")
    ap.add_argument("--apply", action="store_true", help="写入 GIL（gperson/membership）")
    # ---------- S6-b A/B 开关（默认关闭，不影响既有口径）----------
    ap.add_argument("--drop-reflections", action="store_true",
                    help="把倒影 obs（flags.reflection>0 或 flags.wall_mirror>0）视为"
                         "「非独立观测」——它们只是某真人观测的镜像副本，不作为独立身份节点参与聚类；"
                         "其归属由 observed 关系在输出中复原（信息不丢）")
    ap.add_argument("--dump-nodes", action="store_true",
                    help="输出每个 obs 的簇号（供 A/B 逐节点对账）")
    # ---------- S3 链传导（默认关闭，不影响既有口径）----------
    ap.add_argument("--s3-links", default=None,
                    help="S3 链 json（frag_link_v1.json）：把已采纳的同机位碎片链注入候选"
                         "（高优先 margin=0.5）→ 把 S3 的 frag 划分传导到 gperson 聚类；"
                         "其余约束（时间互斥/规模上限/跨机位门）不变")
    # ---------- 人工裁决覆盖（S6-b，默认关闭，主库零改动）----------
    ap.add_argument("--verdicts", default=None,
                    help="人工裁决台账（human_adjudication_v1.json）：① revoke_reflection 从倒影"
                         "排除集合中剔除（视为真人）；② 搭桥 keep/split 施加强制合并/禁止约束。"
                         "★ 只在内存中生效，不改 GIL flags/annotation")
    # ---------- 人脸 must-link / cannot-link 约束（默认关闭）----------
    ap.add_argument("--constraints", default=None,
                    help="人脸复核约束 json（face_mustlink_v1.json）：must_link 簇内 entity 的 obs "
                         "强制合并（margin≥0.65）；cannot_link 两 entity 之间强制禁止。★ 内存生效")
    a = ap.parse_args()

    st = GILStore(ROOT + "/" + a.store, create=False)
    vb = st.vectors(load=True)

    # ---------- 1) 节点 ----------
    rows_all = list(st.con.execute("SELECT id,cam,tid,t_first,t_last,flags FROM obs "
                                   "WHERE tombstone=0 ORDER BY id"))
    # 倒影节点：flags.reflection>0（S5 地板倒影）或 flags.wall_mirror>0（S6 墙镜倒影）
    # 物理语义：倒影不是独立观测，而是真人观测的镜像副本 → 不能作为独立身份证据。
    refl_src, refl_rows = {}, []
    for r in rows_all:
        f = json.loads(r["flags"]) if r["flags"] else {}
        if (f.get("reflection") or 0) > 0 or (f.get("wall_mirror") or 0) > 0:
            refl_rows.append(r)
            src_tid = f.get("wall_mirror_src") if f.get("wall_mirror") else f.get("explained_by")
            refl_src[r["id"]] = (r["cam"], src_tid)

    # ---------- 1b) 人工裁决覆盖（S6-b）：revoke_reflection 恢复为真人 ----------
    verdicts, vstat = None, {}
    if a.verdicts:
        vp_ = a.verdicts if os.path.isabs(a.verdicts) else os.path.join(ROOT, a.verdicts)
        verdicts = json.load(open(vp_, encoding="utf-8"))
        rev = set()
        for it in verdicts.get("items", []):
            if it.get("group") == "reflection_entity" and it.get("action") == "revoke_reflection":
                rev.update(it.get("members", []))
        drop_ids = {r["id"] for r in refl_rows if f"{r['cam']}|{r['tid']}" in rev}
        refl_rows = [r for r in refl_rows if r["id"] not in drop_ids]
        for q in drop_ids:
            refl_src.pop(q, None)
        vstat["revoked_obs"] = len(drop_ids)
        print(f"[s2] ★ 人工裁决覆盖：撤销倒影 {len(drop_ids)} 条（{sorted(rev)}）"
              f" → 实际排除 {len(refl_rows)}")

    if a.drop_reflections:
        rows = [r for r in rows_all if r["id"] not in refl_src]
        print(f"[s2] ★ A/B：排除倒影节点 {len(refl_rows)} 个（{len(rows_all)} → {len(rows)}）")
    else:
        rows = rows_all
    N = len(rows)
    oid = np.array([r["id"] for r in rows])
    cam = np.array([r["cam"] for r in rows])
    t0 = np.array([r["t_first"] if r["t_first"] is not None else np.nan for r in rows], float)
    t1 = np.array([r["t_last"] if r["t_last"] is not None else np.nan for r in rows], float)
    id2row = {int(x): i for i, x in enumerate(oid)}
    E = np.zeros((N, st.dim), np.float32)
    for b in vb.blocks():
        Eb, ids, i2r = vb._cache[b]
        for k, i in i2r.items():
            if int(k) in id2row:
                E[id2row[int(k)]] = Eb[i]
    S = E @ E.T
    print(f"[s2] 节点 {N}（向量齐备）")

    # ---------- 2) 硬约束：时间互斥（豁免 318×320）----------
    ov = (t0[:, None] <= t1[None, :]) & (t0[None, :] <= t1[:, None])
    exempt = np.zeros((N, N), bool)
    for i in range(N):
        for j in range(N):
            if frozenset((cam[i], cam[j])) in OVERLAP_PAIRS:
                exempt[i, j] = True
    incompat = ov & ~exempt
    lower = np.tril(np.ones((N, N), bool), -1)
    incompat &= ~np.eye(N, dtype=bool)
    print(f"[s2] 区间相交对 {int((ov.sum()-N)/2)}；豁免(318×320) {int((exempt&ov).sum()/2)}"
          f" → 互斥对 {int((incompat&lower).sum())}")

    # ---------- 3) 候选/打分 ----------
    w, lam = w_dt_fn()
    same = cam[:, None] == cam[None, :]
    is_ov_pair = np.zeros((N, N), bool)
    for p in OVERLAP_PAIRS:
        x, y = sorted(p)
        is_ov_pair |= ((cam[:, None] == x) & (cam[None, :] == y)) | \
                      ((cam[:, None] == y) & (cam[None, :] == x))
    # 方向无关的转移时间：**早者离开 → 晚者进入**（按时间排序，不按索引；保证 margin 对称）
    d_ij = t0[None, :] - t1[:, None]           # i 更早
    d_ji = t0[:, None] - t1[None, :]           # j 更早
    dt = np.where(t0[:, None] <= t0[None, :], d_ij, d_ji)
    wv = np.vectorize(lambda x: w(x) if np.isfinite(x) else 0.0)(dt)
    cross_ok = (~same) & lower & (~incompat) & (((dt >= 0) & (dt <= DT_MAX)) | is_ov_pair)

    # ---------- 4) val 校准 T_same ----------
    key2row = {(r["cam"], r["tid"]): i for i, r in enumerate(rows)}
    sp = json.load(open(ROOT + "/" + a.splits, encoding="utf-8"))
    vp = [p for p in sp["val"]["pos_pairs"] if (p[0], p[1]) in key2row and (p[0], p[2]) in key2row]
    vn = [p for p in sp["val"]["neg_pairs"] if (p[0], p[1]) in key2row and (p[0], p[2]) in key2row]

    def stat(T, pairs):
        tp = sum(1 for c, i, j in vp if S[key2row[(c, i)], key2row[(c, j)]] >= T)
        fp = sum(1 for c, i, j in vn if S[key2row[(c, i)], key2row[(c, j)]] >= T
                 and not incompat[key2row[(c, i)], key2row[(c, j)]])
        fn = len(vp) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return (2 * p * r / (p + r) if p + r else 0.0), p, r, tp, fp

    grid = np.round(np.arange(0.80, 0.9951, 0.005), 3)
    cal = [(float(T),) + stat(float(T), vp) for T in grid]
    cal.sort(key=lambda x: -x[1])
    T_same = a.t_same if a.t_same is not None else cal[0][0]
    bs = stat(T_same, vp)
    print(f"[s2] val 校准 T_same={T_same:.3f}  F1={bs[0]:.4f} P={bs[1]:.3f} R={bs[2]:.3f}"
          f" (pos {len(vp)} / neg {len(vn)} 中剔除互斥 {sum(1 for c,i,j in vn if incompat[key2row[(c,i)],key2row[(c,j)]])})")
    print(f"     ★ 诚实标注：同机位碎片接续**外观不可判**（pos p50 {np.median([S[key2row[(c,i)],key2row[(c,j)]] for c,i,j in vp]):.4f}"
          f" vs neg p50 {np.median([S[key2row[(c,i)],key2row[(c,j)]] for c,i,j in vn]):.4f}）→ 属 S3 职责")
    print(f"     T_same 扫描前 6: {[(f'{x[0]:.3f}', round(x[1],3)) for x in cal[:6]]}")

    # ---------- 5) 阈值矩阵 + 候选 ----------
    T_cross = a.t_cross
    thr = np.where(same, T_same, T_cross)
    # ★ 重叠视野对（318×320）**不施加 dt 先验**（同时可见是正常的，dt 无意义）
    sc_cross = np.where(is_ov_pair, S, S + lam * np.log(wv + 1e-9))
    mraw = np.where(same, S - T_same, sc_cross - T_cross)
    # ---------- 5b) S3 链传导：链对注入候选（高优先）----------
    s3_mask = None
    n_s3_pairs = 0
    if a.s3_links:
        p = a.s3_links if os.path.isabs(a.s3_links) else os.path.join(ROOT, a.s3_links)
        fl = json.load(open(p, encoding="utf-8"))["links"]
        s3_mask = np.zeros((N, N), bool)
        n_miss = 0
        for lk in fl:
            ca = lk["cam"]
            ta = int(lk["a"].split("|")[1])
            tb = int(lk["b"].split("|")[1])
            i = key2row.get((ca, ta))
            j = key2row.get((ca, tb))
            if i is None or j is None:
                n_miss += 1
                continue
            s3_mask[i, j] = s3_mask[j, i] = True
        n_s3_pairs = int(np.triu(s3_mask, 1).sum())
        mraw = np.where(s3_mask, np.maximum(mraw, 0.5), mraw)
        print(f"[s2] ★ S3 链传导：注入 {n_s3_pairs} 对（缺失 {n_miss}；margin≥0.5 高优先）")
    # 候选：按上三角枚举一次；但 margin 必须**对称**（否则互最佳守卫会读到无效方向）
    same_ok = (S >= T_same)
    if s3_mask is not None:
        same_ok = same_ok | s3_mask
    cand_half = ((same & same_ok) | cross_ok) & lower & (~incompat) & (mraw >= 0)
    cand = cand_half | cand_half.T
    if a.no_overlap:
        cand &= ~is_ov_pair

    # ---------- 5c) 人工裁决覆盖：搭桥 keep/split 约束 ----------
    n_keep = n_split = 0
    if verdicts:
        def _rk(k):
            c_, t_ = k.split("|")
            return key2row.get((c_, int(t_)))

        for it in verdicts.get("items", []):
            if it.get("group") != "bridge_merge" or it.get("action") not in ("keep", "split"):
                continue
            bcl = it.get("b_clusters") or {}
            labs = list(bcl.keys())
            for xi in range(len(labs)):
                for yi in range(xi + 1, len(labs)):
                    for x in bcl[labs[xi]]:
                        for y in bcl[labs[yi]]:
                            i, j = _rk(x), _rk(y)
                            if i is None or j is None or i == j:
                                continue
                            if it["action"] == "keep":
                                if incompat[i, j]:
                                    continue
                                cand[i, j] = cand[j, i] = True
                                mraw[i, j] = mraw[j, i] = max(mraw[i, j], 0.65)
                                n_keep += 1
                            else:
                                cand[i, j] = cand[j, i] = False
                                n_split += 1
        vstat["force_keep"] = n_keep
        vstat["force_split"] = n_split
        print(f"[s2] ★ 人工裁决覆盖：搭桥 强制合并 {n_keep} 对 / 强制禁止 {n_split} 对")

    # ---------- 5d) 人脸 must-link / cannot-link 约束（entity 级）----------
    if a.constraints:
        cp = a.constraints if os.path.isabs(a.constraints) else os.path.join(ROOT, a.constraints)
        cj = json.load(open(cp, encoding="utf-8"))
        ent2obs = defaultdict(list)
        for r in st.con.execute("SELECT id,entity_id FROM obs "
                                "WHERE entity_id IS NOT NULL AND tombstone=0"):
            ent2obs[r["entity_id"]].append(r["id"])
        oid2r = {int(k): i for k, i in id2row.items()}

        def _rows(eids):
            out = []
            for e in eids:
                for ov in ent2obs.get(e, []):
                    i = oid2r.get(int(ov))
                    if i is not None:
                        out.append(i)
            return sorted(set(out))

        n_ml = n_cl = 0
        for c in cj.get("must_link", []):
            rs = _rows(c["entity_ids"])
            for x in range(len(rs)):
                for y in range(x + 1, len(rs)):
                    i, j = rs[x], rs[y]
                    if not incompat[i, j]:
                        cand[i, j] = cand[j, i] = True
                        mraw[i, j] = mraw[j, i] = max(mraw[i, j], 0.65)
                        n_ml += 1
        for c in cj.get("cannot_link", []):
            ra, rb = _rows([c["entity_a"]]), _rows([c["entity_b"]])
            for i in ra:
                for j in rb:
                    if i != j:
                        cand[i, j] = cand[j, i] = False
                        mraw[i, j] = mraw[j, i] = -9.0
                        n_cl += 1
        vstat["must_link"] = n_ml
        vstat["cannot_link"] = n_cl
        print(f"[s2] ★ 人脸约束：must-link 强制合并 {n_ml} 对 / cannot-link 禁止 {n_cl} 对"
              f"（{len(cj.get('must_link', []))} 簇 + {len(cj.get('cannot_link', []))} 对）")

    margin = np.where(cand, mraw, -9.0)
    np.fill_diagonal(margin, -9.0)
    print(f"[s2] 候选对 {int(cand.sum())}（同机位 {int((cand & same).sum())} / "
          f"跨机位 {int((cand & ~same).sum())}，其中重叠视野 {int((cand & is_ov_pair).sum())}）")

    # ---------- 6) 约束凝聚聚类 ----------
    AGG = {"avg": lambda arr: float(arr.mean()),
           "max": lambda arr: float(arr.max()),
           "complete": lambda arr: float(arr.min())}[a.linkage]

    def link_of(cx, cy):
        if incompat[np.ix_(cl[cx], cl[cy])].any():
            return None
        return AGG(margin[np.ix_(cl[cx], cl[cy])])

    def best_partner(cx):
        bb = None
        for cz in cl:
            if cz == cx:
                continue
            v = link_of(cx, cz)
            if v is None:
                continue
            if bb is None or v > bb[0]:
                bb = (v, cz)
        return bb

    cl = {i: [i] for i in range(N)}
    owner = np.arange(N)
    heap = [(-float(margin[x, y]), int(x), int(y)) for x, y in zip(*np.where(cand))]
    heapq.heapify(heap)
    n_merge = n_skip = n_skip_s3 = 0
    while heap:
        negv, x, y = heapq.heappop(heap)
        cx, cy = int(owner[x]), int(owner[y])
        if cx == cy or cx not in cl or cy not in cl:
            continue
        mx, my = cl[cx], cl[cy]
        if incompat[np.ix_(mx, my)].any():
            continue
        m = AGG(margin[np.ix_(mx, my)])
        if m < 0:
            continue
        if (-negv) > m + 1e-9:                     # 堆内值过期 → 回填
            heapq.heappush(heap, (-m, mx[0], my[0]))
            continue
        if a.mutual:                               # ★ 互最佳守卫
            ba, bb = best_partner(cx), best_partner(cy)
            if not (ba and bb and ba[1] == cy and bb[1] == cx):
                n_skip += 1
                continue
        if a.max_size and len(mx) + len(my) > a.max_size:   # ★ 规模上限（防串接）
            n_skip += 1
            if s3_mask is not None and s3_mask[np.ix_(mx, my)].any():
                n_skip_s3 += 1
            continue
        cl[cx] = mx + my
        del cl[cy]
        for q in my:
            owner[q] = cx
        n_merge += 1
        for cz, mz in list(cl.items()):
            if cz == cx or incompat[np.ix_(cl[cx], mz)].any():
                continue
            mm = AGG(margin[np.ix_(cl[cx], mz)])
            if mm >= 0:
                heapq.heappush(heap, (-mm, cl[cx][0], mz[0]))
    print(f"[s2] 合并 {n_merge} 次（互最佳守卫跳过 {n_skip}）→ gperson {len(cl)} 个"
          f"（linkage={a.linkage}{'+mutual' if a.mutual else ''}）")

    # ---------- 7) 不变量 ----------
    bad = []
    for cid, ms in cl.items():
        if len(ms) > 1:
            sub = incompat[np.ix_(ms, ms)].copy()
            np.fill_diagonal(sub, False)
            if sub.any():
                bad.append(cid)
    print(f"[s2] 不变量（簇内无时间互斥对）：违规 {len(bad)}  {'✅' if not bad else '❌'}")

    # ---------- 7b) S3 传导统计 ----------
    s3_transfer = None
    if s3_mask is not None:
        idx = np.where(np.triu(s3_mask, 1))
        n_p = len(idx[0])
        n_same_cl = sum(1 for i, j in zip(*idx) if owner[i] == owner[j])
        s3_transfer = dict(n_pairs=n_p, n_same_cluster=n_same_cl,
                           rate=round(n_same_cl / max(n_p, 1), 4),
                           n_blocked_by_sizecap=n_skip_s3)
        print(f"[s3] 链传导结果：{n_same_cl}/{n_p} 对最终同簇"
              f"（{100*n_same_cl/max(n_p,1):.1f}%；规模上限阻断 {n_skip_s3} 对）")

    # ---------- 8) 评价 ----------
    gid = np.zeros(N, int)
    labels = {}
    for k, (cid, ms) in enumerate(sorted(cl.items(), key=lambda kv: min(kv[1]))):
        for q in ms:
            gid[q] = k
        labels[cid] = f"g{k+1:04d}"
    cid2idx = {cid: k for k, (cid, _) in enumerate(sorted(cl.items(), key=lambda kv: min(kv[1])))}
    sizes = Counter(gid.tolist())
    ncams = Counter(len({cam[q] for q in ms}) for ms in cl.values())

    gt_hit = sum(1 for x, y in GT_318_320 if ("318", x) in key2row and ("320", y) in key2row
                 and gid[key2row[("318", x)]] == gid[key2row[("320", y)]])
    gt_cl = {gid[key2row[("318", x)]] for x, _ in GT_318_320 if ("318", x) in key2row}
    print(f"[eval] 318×320 GT：8 对中 {gt_hit} 对同簇；7 人 → {len(gt_cl)} 簇"
          f"（{gt_hit}/8 覆盖率，簇数应=7）")
    print(f"[eval] 簇：最大 {max(sizes.values())} 个成员；跨 ≥2 机位 {sum(v for k,v in ncams.items() if k>=2)} 个 "
          f"（分布 {dict(sorted(ncams.items()))}）")

    # 人工真值（跨机位已判行）
    tr = {}
    for src in (os.path.join(ROOT, "webapp/backend/data/truth/seed_labels.json"),
                os.path.join(ANA, "truth/remote/labels.json")):
        if os.path.exists(src):
            d = json.load(open(src, encoding="utf-8")) or {}
            tr.update(d.get("pair", d) if isinstance(d, dict) else {})
    ent_members = defaultdict(list)
    for r in st.con.execute("SELECT id,entity_id FROM obs WHERE entity_id IS NOT NULL"):
        ent_members[r["entity_id"]].append(r["id"])

    def ent_gid(c, e_):
        r = st.con.execute("SELECT id FROM entity WHERE cam=? AND eid=?", (c, e_)).fetchone()
        if not r:
            return None
        ms = [id2row[q] for q in ent_members.get(r["id"], []) if q in id2row]
        return Counter(int(gid[q]) for q in ms).most_common(1)[0][0] if ms else None

    tp = fp = fn = unk = 0
    for pid, v in tr.items():
        if not isinstance(v, dict) or v.get("v") not in (0, 1) or "__" not in pid:
            continue
        # 命名空间防御：S6B_P_/PRT_ 前缀剥离后计入（标准 pair 格式）
        pid = pid[6:] if pid.startswith("S6B_P_") else (
            pid[4:] if pid.startswith("PRT_") else pid)
        if pid.count("__") != 1 or pid.count("_E_") != 2:
            continue
        pa, pb = pid.split("__")
        ca, ea = pa.split("_E_"); cb, eb = pb.split("_E_")
        ga, gb = ent_gid(ca, ea), ent_gid(cb, eb)
        if ga is None or gb is None:
            unk += 1
            continue
        pred = int(ga == gb)
        if v["v"] == 1:
            tp += pred; fn += 1 - pred
        else:
            fp += pred
    P = tp / (tp + fp) if tp + fp else None
    R = tp / (tp + fn) if tp + fn else None
    F = (2 * P * R / (P + R)) if P and R else None
    print(f"[eval] 人工真值（跨机位已判 {tp+fp+fn} 条，另有 {unk} 条无归属）："
          f"TP {tp} FP {fp} FN {fn} → "
          + (f"P {P:.3f} R {R:.3f} F1 {F:.3f}" if P and R else "样本不足"))
    # 与 relay v2 边的一致性
    edges = json.load(open(os.path.join(ANA, "relay_v2", "edges_v2_gated.json"), encoding="utf-8"))["edges"]
    ag = dis = 0
    for e in edges:
        if e["status"] not in ("active", "audit", "gate"):
            continue
        c1, e1 = e["a"].split(":"); c2, e2 = e["b"].split(":")
        g1, g2 = ent_gid(c1, e1), ent_gid(c2, e2)
        if g1 is None or g2 is None:
            continue
        if g1 == g2:
            ag += 1
        else:
            dis += 1
    print(f"[eval] 与系统 {ag+dis} 条边的一致性：同簇 {ag} / 跨簇 {dis}"
          f"（跨簇 = 系统连接被 gperson 判为不同人）")

    # ---------- 8b) A/B：倒影归属复原 + 逐节点簇号 ----------
    # 倒影节点被排除出节点集，但其「被哪个真人观测解释」的关系完整保留 → 信息不丢。
    refl_attach = []
    for r in refl_rows:
        cam_, src_tid = refl_src[r["id"]]
        srcoid = st.con.execute("SELECT id FROM obs WHERE cam=? AND tid=? AND tombstone=0",
                                (cam_, src_tid)).fetchone()
        gi = id2row.get(int(srcoid[0])) if srcoid else None
        refl_attach.append(dict(
            refl=f"{r['cam']}|{r['tid']}", source=f"{cam_}|{src_tid}",
            src_gperson=(None if gi is None else f"g{int(gid[gi])+1:04d}"),
            src_in_nodes=bool(gi is not None)))
    node2gid = ({f"{rows[i]['cam']}|{rows[i]['tid']}": int(gid[i]) for i in range(N)}
                if a.dump_nodes else None)

    out = dict(
        params=dict(T_same=T_same, T_cross=T_cross, dt_max=DT_MAX, lam=lam,
                    overlap_pairs=[sorted(x) for x in OVERLAP_PAIRS],
                    no_overlap=bool(a.no_overlap), linkage=a.linkage, mutual=bool(a.mutual),
                    max_size=int(a.max_size),
                    method=f"constrained agglomerative / {a.linkage} linkage"
                           + (f" + size_cap={a.max_size}" if a.max_size else ""),
                    T_same_source="splits_v5 val 同机位 pos/neg 校准",
                    T_cross_source="relay_v2 params.sim_thr（冻结，未用真值）",
                    note="跨机位阈值来源为既有冻结工作点；同机位阈值 val 校准；"
                         "硬约束=时间互斥（豁免重叠视野对 318×320）"),
        calib=dict(f1=bs[0], precision=bs[1], recall=bs[2], tp=bs[3], fp=bs[4],
                   n_val_pos=len(vp), n_val_neg=len(vn),
                   grid=[[float(x[0]), round(x[1], 4)] for x in cal[:12]],
                   caveat="同机位碎片接续外观不可判 → 属 S3（LDP 长程关联）职责"),
        n_nodes=N, n_gperson=len(cl), n_merge=n_merge, invariant_violations=len(bad),
        drop_reflections=bool(a.drop_reflections),
        s3_links=(a.s3_links if a.s3_links else None),
        s3_transfer=s3_transfer,
        verdicts=(os.path.basename(a.verdicts) if a.verdicts else None),
        verdict_effect=(vstat or None),
        constraints=(os.path.basename(a.constraints) if a.constraints else None),
        reflections_excluded=len(refl_rows) if a.drop_reflections else 0,
        reflections_seen=len(refl_rows),
        reflection_attach=refl_attach,
        node2gid=node2gid,
        size_dist={str(k): v for k, v in sorted(Counter(sizes.values()).items())},
        cams_per_cluster={str(k): v for k, v in sorted(ncams.items())},
        gt_318_320=dict(pair_hit=gt_hit, n_pairs=len(GT_318_320), n_clusters=len(gt_cl)),
        human_truth=dict(tp=tp, fp=fp, fn=fn, precision=P, recall=R, f1=F, unresolved=unk),
        relay_consistency=dict(same_cluster=ag, cross_cluster=dis),
        clusters=[dict(gid=labels[k], n=len(ms), cams=sorted({cam[q] for q in ms}),
                       obs=sorted(int(oid[q]) for q in ms),
                       t0=float(np.nanmin(t0[ms])), t1=float(np.nanmax(t1[ms])))
                  for k, ms in sorted(cl.items(), key=lambda kv: -len(kv[1]))[:60]],
    )
    json.dump(out, open(ROOT + "/" + a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {a.out}")

    if a.apply:
        # ★ 保留非人聚类（如 gREFLECTION: state='non_person'）——重建 gperson 不应抹掉它
        st.con.execute("DELETE FROM gperson WHERE state != 'non_person'")
        src_tag = f"gperson_v2:T_same={T_same}"
        if a.s3_links:
            src_tag += "+s3links"
        if a.drop_reflections:
            src_tag += "+droprefl"
        src_tag += f"+cap{a.max_size}"
        st.con.execute("DELETE FROM membership WHERE node_kind='obs' AND "
                       "(src LIKE 'gperson_v1%' OR src LIKE 'gperson_v2%')")
        for k, ms in cl.items():
            g = st.add_gperson(labels[k], state="candidate")
            for q in ms:
                sc = float(np.mean([S[q, p2] for p2 in ms if p2 != q])) if len(ms) > 1 else None
                st.add_membership(g, "obs", int(oid[q]), score=sc, prior=None,
                                  decision="auto", src=src_tag)
        st.con.execute(
            "UPDATE gperson SET first_t=(SELECT MIN(o.t_first) FROM membership m "
            "JOIN obs o ON o.id=m.node_id WHERE m.gperson_id=gperson.id), "
            "last_t=(SELECT MAX(o.t_last) FROM membership m JOIN obs o ON o.id=m.node_id "
            "WHERE m.gperson_id=gperson.id)")
        st.commit()
        print(f"[gil] 已写入 gperson {len(cl)} / membership {sum(len(v) for v in cl.values())}")
        print("[gil] stats:", json.dumps(st.stats(), ensure_ascii=False))
    st.close()


if __name__ == "__main__":
    main()
