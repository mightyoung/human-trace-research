#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EVAL-3c：H1 门控自训练的**长数据流版**（35 分钟连续流，8 轮，同 session bootstrap）。

与 EVAL-3b 的关系（预注册，先于看数确定）：
  3b：cradio 空间、1012 val 15min 四等分（4 轮）、选择器训练于 1012-train。
  3c：DINOv3-B 空间、long35 **35 分钟连续流**（8 轮 × 250s），时长 >2× 3b；
      选择器在 **long35 自己的前 10 分钟**上无监督 bootstrap（部署形态：先建库再流式）。

为什么 bootstrap 用同 session 而不是跨 session：
  实测 d3b 选择器（1012-train 构建）对另一天的 long35 覆盖仅 **33%**（同 session 为 85%）。
  同段重叠校验（long35 前 10min ≡ synced-10min）：d3b 同人 cos 中位 **0.9993**、随机对 p99 0.966
  → 判别信息充分，低覆盖是**跨日外观漂移**所致，不是模型失效。
  → 长流实验用同 session bootstrap（否则两臂都学不到东西）；跨 session 覆盖崩塌单独报告。

时间结构（帧 → 秒 = T0[cam] + frame/25；T0 沿用项目口径，long35 与 10min 同源同步）：
  REF    [0, 15000)           bootstrap 段（前 10min；用于**构建选择器**，不参与流式轮次）
  STREAM [15000, 45000)       8 轮 × 3750 帧（每轮 ~150s）→ 逐轮累积更新
  TAIL   [45000, 52500)       留出段（从不参与更新；全部指标在此评测）

臂（同一 TAIL 上评测；轮内分类用上一轮更新后的选择器 = 真自训练环）：
  A1_frozen       冻结（= bootstrap 选择器，不更新）
  R2_naive        朴素流式：接受即学
  R2_gated        margin 门控（per-cam p30，生产口径）
  RR_p20/p15/p10  更松的 margin 门（流式）
  W_warmup        （可选，默认关）REF 段再适配一次——长流实验中 REF 已用于 bootstrap，通常不需要

指标（与 eval3b 同构）：unknown / ab_same / retention（TAIL↔REF mutual-NN≥0.90 同簇比例）
/ viol（TAIL 内跨机位时间重叠>2s 却同簇，318×320 豁免）/ churn / drift / hijack
+ 每轮低于 margin 门的成员数（n_below_gate）= 门控实际拦下的"可疑学习"

只读实验：不写任何生产文件；产物 output_crosscam_local/d3b_ab/eval3c_long_stream.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector import IdentitySelector, _norm  # noqa: E402

ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "output_crosscam_local", "d3b_ab")
SEL = os.path.join(BASE, "selector_d3b_longref.npz")
EMB = os.path.join(BASE, "long35")
DET = os.path.join(ROOT, "output_crosscam", "long35")
OUT = BASE

CAMS = ["305", "318", "320", "323"]
T0 = {"305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0}
FPS = 25.0
EXEMPT = {("318", "320")}
MIN_MEMBERS = 3
# eval3b 的 0.95 在 cradio 下"不生效"（assign thr 0.975 > 0.95）→ 等价于"全部 known 都参与学习"。
# d3b 的 assign thr 更低（0.87–0.93），故此处用 0.0 复刻同一语义。
SIM_ACCEPT = 0.0
REF_END = 15000
STREAM_END = 45000
N_ROUNDS = 8


# ---------------- 数据 ----------------
def load_cam(cam):
    seg = f"{cam}_long35"
    p = os.path.join(EMB, f"{cam}_emb_long35_d3b.npz")
    pa = os.path.join(EMB, f"{cam}_emb_long35_d3b_a.npz")
    pb = os.path.join(EMB, f"{cam}_emb_long35_d3b_b.npz")
    z, za, zb = np.load(p), np.load(pa), np.load(pb)
    ks = [k for k in z.files if k in za.files and k in zb.files]
    va = np.asarray([za[k] for k in ks], np.float32)
    vb = np.asarray([zb[k] for k in ks], np.float32)
    v = _norm(va + vb, axis=1)
    # 帧区间
    acc = defaultdict(lambda: [10 ** 9, -1])
    with open(os.path.join(DET, f"{cam}_long35_detections.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for t in d.get("track_ids", []) or []:
                if t is None or int(t) < 0:
                    continue
                a = acc[int(t)]
                a[0] = min(a[0], d["frame"])
                a[1] = max(a[1], d["frame"])
    keys, vas, vbs, vs, ts, mids = [], [], [], [], [], []
    import re
    for i, k in enumerate(ks):
        tid = int(re.search(r"(\d+)$", k).group(1))
        if tid not in acc:
            continue
        f0, f1 = acc[tid]
        keys.append(f"{seg}#{tid}")
        vas.append(va[i]); vbs.append(vb[i]); vs.append(v[i])
        ts.append((T0[cam] + f0 / FPS, T0[cam] + f1 / FPS))
        mids.append((f0 + f1) / 2.0)
    return dict(keys=keys, cams=[cam] * len(keys), Va=np.asarray(vas), Vb=np.asarray(vbs),
                V=np.asarray(vs), t=ts, midf=np.asarray(mids))


def load_all():
    parts = [load_cam(c) for c in CAMS]
    out = {}
    for k in ("keys", "cams", "Va", "Vb", "V", "t", "midf"):
        if k in ("keys", "cams", "t"):
            out[k] = sum((p[k] for p in parts), [])
        else:
            out[k] = np.concatenate([p[k] for p in parts], 0)
    return out


def subset(data, idx):
    idx = list(idx)
    return dict(keys=[data["keys"][i] for i in idx], cams=[data["cams"][i] for i in idx],
                Va=data["Va"][idx], Vb=data["Vb"][idx], V=data["V"][idx],
                t=[data["t"][i] for i in idx], midf=data["midf"][idx])


# ---------------- 指标 ----------------
def overlap_pairs(meta):
    """同簇且跨机位时间重叠 >2s 的对（错误代理；318×320 豁免）。"""
    viol = []
    n = len(meta)
    for i in range(n):
        ki, ci, a0, a1, gi = meta[i]
        if gi < 0:
            continue
        for j in range(i + 1, n):
            kj, cj, b0, b1, gj = meta[j]
            if gj < 0 or gi != gj or ci == cj:
                continue
            if tuple(sorted((ci, cj))) in EXEMPT:
                continue
            if min(a1, b1) - max(a0, b0) > 2.0:
                viol.append((ki, kj))
    return viol


def mutual_pairs(A, B, thr=0.90):
    S = A["V"] @ B["V"].T
    bi = S.argmax(1)
    bj = S.argmax(0)
    out = []
    for i in range(len(A["V"])):
        j = int(bi[i])
        if int(bj[j]) == i and S[i, j] >= thr:
            out.append((i, j, float(S[i, j])))
    return out


def clone(sel, P, C):
    return IdentitySelector(protos=P, thr_unknown=sel.thr_unknown, proto_cluster=C,
                            thr_unknown_per_cam=dict(sel.thr_unknown_per_cam or {}),
                            thr_loose=sel.thr_loose,
                            thr_loose_per_cam=dict(sel.thr_loose_per_cam or {}),
                            margin_prec=sel.margin_prec,
                            margin_prec_per_cam=dict(sel.margin_prec_per_cam or {}),
                            margin_ret=sel.margin_ret, meta=dict(sel.meta))


def predict_margin(sel, V, cams, m_arr=None):
    """selector.predict 的显式 margin 变体（同一打分路径）。m_arr=None → 不过滤。"""
    v = _norm(np.atleast_2d(V.astype(np.float32)), axis=1)
    S = v @ sel.protos.T
    sc = sel._cluster_scores(S)
    idx = sc.argmax(1)
    best = sc[np.arange(len(idx)), idx]
    thr = sel._thr_for(cams, False)
    srt = np.sort(sc, axis=1)
    mg = srt[:, -1] - srt[:, -2]
    ids = np.where(best >= thr, idx, -1)
    if m_arr is not None:
        ids = np.where(best >= thr, np.where(mg >= m_arr, idx, -1), -1)
    return ids, best, mg


def mutex_filter(idx_accept, cams, ts):
    drop = set()
    for ii in range(len(idx_accept)):
        for jj in range(ii + 1, len(idx_accept)):
            i, j = idx_accept[ii], idx_accept[jj]
            if cams[i] == cams[j] or tuple(sorted((cams[i], cams[j]))) in EXEMPT:
                continue
            if min(ts[i][1], ts[j][1]) - max(ts[i][0], ts[j][0]) > 2.0:
                drop.add(i); drop.add(j)
    return [i for i in idx_accept if i not in drop], len(drop)


def main():
    global SIM_ACCEPT, MIN_MEMBERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--margins", default="p30,p20,p15,p10")
    ap.add_argument("--json", default=os.path.join(OUT, "eval3c_long_stream.json"))
    ap.add_argument("--warmup", default="none", choices=["none", "gated", "auto"],
                    help="none=直接流式（默认；选择器已用 REF bootstrap）；"
                         "gated=REF 再门控适配一次；auto=按 REF 覆盖率自动决定")
    ap.add_argument("--selector", default=SEL)
    ap.add_argument("--sim-accept", type=float, default=SIM_ACCEPT,
                    help="接受成员的最低相似度（默认 0=与 eval3b 等价：全部 known 参与学习）")
    ap.add_argument("--min-members", type=int, default=MIN_MEMBERS,
                    help="每簇每轮最少成员数才更新（默认 3；探索性压力测试可设 1）")
    ap.add_argument("--ref-end", type=int, default=REF_END)
    ap.add_argument("--stream-end", type=int, default=STREAM_END)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    args = ap.parse_args()

    SIM_ACCEPT = float(args.sim_accept)
    MIN_MEMBERS = int(args.min_members)

    sel = IdentitySelector.load(args.selector)
    print(f"[selector] d3b: protos={sel.protos.shape[0]} clusters={sel.n_clusters} "
          f"F1(train cooc)={sel.meta.get('f1_cooc_train')}", flush=True)
    data = load_all()
    print(f"[data] long35 4 cams: {len(data['keys'])} tracks "
          f"(frames {data['midf'].min():.0f}..{data['midf'].max():.0f})", flush=True)

    midf = data["midf"]
    ref_idx = np.where(midf < args.ref_end)[0]
    tail_idx = np.where(midf >= args.stream_end)[0]
    stream_idx = np.where((midf >= args.ref_end) & (midf < args.stream_end))[0]
    REF = subset(data, ref_idx)
    TAIL = subset(data, tail_idx)
    print(f"[split] REF={len(ref_idx)} STREAM={len(stream_idx)} TAIL={len(tail_idx)}", flush=True)

    pairs = mutual_pairs(TAIL, REF, 0.90)
    print(f"[pairs] TAIL↔REF mutual-NN>=0.90: n={len(pairs)}", flush=True)

    # 初始覆盖率（在 REF 上量，非评测集 → 允许用于决定 warm-up）
    ids_ref, sims_ref = sel.predict(REF["V"], cams=REF["cams"])
    cov_ref = float((ids_ref >= 0).mean())
    ids_tail, _ = sel.predict(TAIL["V"], cams=TAIL["cams"])
    cov_tail0 = float((ids_tail >= 0).mean())
    print(f"[coverage] frozen on REF={cov_ref:.3f} TAIL={cov_tail0:.3f}", flush=True)
    warmup = args.warmup
    if warmup == "auto":
        warmup = "gated" if cov_ref < 0.55 else "none"
    print(f"[design] warmup={warmup}（auto 规则：REF 覆盖 <0.55 则先用 REF 门控适配）", flush=True)

    def evaluate(tag, sel_arm):
        ids_t, sims_t = sel_arm.predict(TAIL["V"], cams=TAIL["cams"])
        ids_a, _ = sel_arm.predict(TAIL["Va"], cams=TAIL["cams"])
        ids_b, _ = sel_arm.predict(TAIL["Vb"], cams=TAIL["cams"])
        both = (ids_a != -1) & (ids_b != -1)
        ab_same = float(((ids_a == ids_b) & both).sum() / max(both.sum(), 1))
        ids_r, _ = sel_arm.predict(REF["V"], cams=REF["cams"])
        ret_same = sum(1 for i, j, _ in pairs if ids_t[i] != -1 and ids_t[i] == ids_r[j])
        ret_both_unk = sum(1 for i, j, _ in pairs if ids_t[i] == -1 and ids_r[j] == -1)
        meta = [(TAIL["keys"][i], TAIL["cams"][i], TAIL["t"][i][0], TAIL["t"][i][1],
                 int(ids_t[i])) for i in range(len(ids_t))]
        viol = overlap_pairs(meta)
        return dict(tag=tag, unknown=round(float((ids_t == -1).mean()), 4),
                    ab_same=round(ab_same, 4), ret_same=ret_same, ret_n=len(pairs),
                    ret_both_unknown=ret_both_unk, viol=len(viol), n_tail=len(ids_t),
                    known=int((ids_t >= 0).sum()), ids=ids_t.tolist())

    arms = {}
    base = evaluate("A1_frozen", sel)
    arms["A1_frozen"] = base
    base_ids = np.asarray(base["ids"])

    # ---- 可选 warm-up（两臂共享起点）----
    start_sel = sel
    warm = None
    if warmup == "gated":
        ids_w, sims_w = sel.predict(REF["V"], cams=REF["cams"], margin="prec")
        members = defaultdict(list)
        for i, (cid, s) in enumerate(zip(ids_w, sims_w)):
            if cid >= 0 and s >= SIM_ACCEPT:
                members[int(cid)].append(i)
        P, C = [sel.protos], [sel.proto_cluster]
        n_upd, n_drop = 0, 0
        for c, lidx in sorted(members.items()):
            kept, nd = mutex_filter(lidx, REF["cams"], REF["t"])
            n_drop += nd
            if len(kept) < MIN_MEMBERS:
                continue
            mu = REF["V"][kept].mean(0)
            mu = mu / (np.linalg.norm(mu) + 1e-9)
            P.append(mu[None, :]); C.append(np.array([c])); n_upd += 1
        start_sel = clone(sel, np.concatenate(P, 0), np.concatenate(C, 0))
        warm = dict(n_ref=len(REF["keys"]), n_confident=int(sum(len(v) for v in members.values())),
                    clusters_updated=n_upd, mutex_dropped=n_drop)
        ev_w = evaluate("W_warmup", start_sel)
        arms["W_warmup"] = ev_w
        print(f"[warmup] conf={warm['n_confident']} upd={n_upd} drop={n_drop} → "
              f"tail unknown={ev_w['unknown']} ab={ev_w['ab_same']} "
              f"ret={ev_w['ret_same']}/{ev_w['ret_n']} viol={ev_w['viol']}", flush=True)
        warm["tail_after"] = {k: v for k, v in ev_w.items() if k != "ids"}
        base_ids = np.asarray(arms["W_warmup"]["ids"])   # 流式 churn 相对"共享起点"计
        print("[churn-ref] 以 W_warmup 后状态为 churn 基线", flush=True)

    scan_pc = sel.meta.get("margin_scan_per_cam") or {}
    margins = {}
    for c, v in (sel.margin_prec_per_cam or {}).items():
        margins[c] = dict(scan_pc.get(c) or {"p30": float(v)})
        margins[c].setdefault("p30", float(v))
    print(f"[margins] {json.dumps(margins)}", flush=True)

    def run_stream(tag, gated, margin_key):
        cur = start_sel
        P, C = [cur.protos], [cur.proto_cluster]
        traj = []
        bounds = np.linspace(args.ref_end, args.stream_end, args.rounds + 1)
        for r in range(args.rounds):
            lo, hi = bounds[r], bounds[r + 1]
            idx = [i for i, mf in enumerate(midf) if lo <= mf < hi]
            if not idx:
                traj.append(dict(round=r + 1, n_chunk=0, skipped=True))
                continue
            ch = subset(data, idx)
            m_arr = np.asarray([margins[c][margin_key] for c in ch["cams"]], np.float32)
            ids_v, sims_v, mg_v = predict_margin(cur, ch["V"], ch["cams"], None)
            members = defaultdict(list)
            below_gate = []
            for i, (cid, s) in enumerate(zip(ids_v, sims_v)):
                if cid < 0 or s < SIM_ACCEPT:
                    continue
                low = mg_v[i] < m_arr[i]
                if low:
                    below_gate.append((i, int(cid), float(s), float(mg_v[i]), float(m_arr[i])))
                if low and gated:
                    continue          # 门控：低于 margin 门的成员不参与学习
                members[int(cid)].append(i)
            n_drop = 0
            if gated:
                for c, lidx in list(members.items()):
                    kept, nd = mutex_filter(lidx, ch["cams"], ch["t"])
                    members[c] = kept
                    n_drop += nd
            n_acc = int(sum(len(v) for v in members.values()))
            added, drifts, hijacks = [], [], []
            for c, lidx in sorted(members.items()):
                if len(lidx) < MIN_MEMBERS:
                    continue
                mu = ch["V"][lidx].mean(0)
                mu = mu / (np.linalg.norm(mu) + 1e-9)
                cols = np.where(cur.proto_cluster == c)[0]
                cent = _norm(cur.protos[cols].mean(0, keepdims=True))[0]
                drifts.append(float(np.dot(mu, cent)))
                # hijack：与其他簇质心的最大 cos
                other = [k for k in range(cur.n_clusters) if k != c]
                cents = []
                for k in other:
                    cc = np.where(cur.proto_cluster == k)[0]
                    if cc.size:
                        cents.append(_norm(cur.protos[cc].mean(0, keepdims=True))[0])
                if cents:
                    hijacks.append(float(np.max(np.asarray(cents) @ mu)))
                P.append(mu[None, :]); C.append(np.array([c])); added.append(c)
            cur = clone(start_sel, np.concatenate(P, 0), np.concatenate(C, 0))
            ev = evaluate(f"{tag}_r{r+1}", cur)
            ids = np.asarray(ev["ids"])
            churn = dict(
                switched=int(((base_ids != -1) & (ids != -1) & (base_ids != ids)).sum()),
                gained=int(((base_ids == -1) & (ids != -1)).sum()),
                lost=int(((base_ids != -1) & (ids == -1)).sum()))
            traj.append(dict(round=r + 1, t_lo_min=round(float(lo / FPS / 60), 2),
                             t_hi_min=round(float(hi / FPS / 60), 2),
                             n_chunk=len(idx), n_confident=n_acc, mutex_dropped=n_drop,
                             n_below_gate=len(below_gate),
                             below_gate=[dict(key=ch["keys"][i], cluster=c, sim=round(s, 4),
                                              margin=round(mg, 6), gate=round(g, 6))
                                         for i, c, s, mg, g in
                                         sorted(below_gate, key=lambda x: x[3] - x[4])[:10]],
                             clusters_updated=len(added), updated_ids=added,
                             drift_med=round(float(np.median(drifts)), 4) if drifts else None,
                             drift_min=round(float(np.min(drifts)), 4) if drifts else None,
                             hijack_max=round(float(np.max(hijacks)), 4) if hijacks else None,
                             proto_total=int(cur.protos.shape[0]),
                             unknown=ev["unknown"], ab_same=ev["ab_same"],
                             ret_same=ev["ret_same"], ret_n=ev["ret_n"],
                             ret_both_unknown=ev["ret_both_unknown"], viol=ev["viol"],
                             known=ev["known"], churn=churn,
                             tail_ids=(ev["ids"] if tag in ("R2_naive", "R2_gated") else None)))
            print(f"[{tag}] r{r+1} n={len(idx)} conf={n_acc} low={len(below_gate)} "
                  f"upd={len(added)} unk={ev['unknown']} ab={ev['ab_same']} "
                  f"ret={ev['ret_same']}/{ev['ret_n']} viol={ev['viol']} "
                  f"drift={traj[-1]['drift_med']} hijack={traj[-1]['hijack_max']} "
                  f"churn={churn}", flush=True)
        final = evaluate(tag, cur)
        return final, traj

    r_naive, traj_naive = run_stream("R2_naive", gated=False, margin_key="p30")
    arms["R2_naive"] = r_naive
    r_g, traj_g = run_stream("R2_gated", gated=True, margin_key="p30")
    arms["R2_gated"] = r_g
    scans = {}
    for mk in [x for x in args.margins.split(",") if x and x != "p30"]:
        r_s, traj_s = run_stream(f"RR_{mk}", gated=True, margin_key=mk)
        arms[f"RR_{mk}"] = r_s
        scans[mk] = traj_s

    out = dict(version="eval3c/1", selector=SEL, space="dinov3b-768",
               design=dict(ref=[0, args.ref_end], stream=[args.ref_end, args.stream_end],
                           tail=[args.stream_end, None], rounds=args.rounds,
                           warmup=warmup, sim_accept=SIM_ACCEPT, min_members=MIN_MEMBERS,
                           t0=T0, fps=FPS, cams=CAMS),
               n=dict(total=len(data["keys"]), ref=len(ref_idx),
                      stream=len(stream_idx), tail=len(tail_idx), pairs=len(pairs)),
               coverage=dict(ref=cov_ref, tail_frozen=cov_tail0),
               margins_used=margins,
               warmup=warm,
               arms={k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in arms.items()},
               rounds=dict(naive=traj_naive, gated=traj_g, scans=scans),
               note="只读实验；长流=long35 35min 连续流，选择器构建于 1012-train（跨 session）")
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n===== 汇总 =====")
    print(f"{'arm':>12} {'unknown':>8} {'ab_same':>8} {'ret':>9} {'ret_unk':>8} {'viol':>5}")
    for tag, a in arms.items():
        print(f"{tag:>12} {a['unknown']:>8} {a['ab_same']:>8} "
              f"{a['ret_same']:>4}/{a['ret_n']:<4} {a['ret_both_unknown']:>8} {a['viol']:>5}")
    print("saved", args.json)


if __name__ == "__main__":
    main()
