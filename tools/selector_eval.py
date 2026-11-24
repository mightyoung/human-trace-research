#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务一评估：选择器在验证/测试集上的表现（无标签协议）。

协议：
  ① 分片一致性（核心）：同轨迹前/后半向量（_a/_b）经选择器分配 → 同簇率 vs 随机对碰撞率
  ② 跨集匹配：测试集向量 → 训练集原型的 sim 分布；top 匹配对导出（供人工目视）
  ③ 三档对照：selector(簇原型) vs 惰性最近邻(全部 train 向量) vs 随机基线

产物：metrics json + matches csv（top 跨集匹配）
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector, _norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tail_digits(s):
    num = ""
    for ch in reversed(s):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_split(emb_dir, segs):
    keys, va, vb = [], [], []
    for seg in segs:
        pa = os.path.join(emb_dir, f"{seg}_emb_a.npz")
        pb = os.path.join(emb_dir, f"{seg}_emb_b.npz")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"[skip] {seg} (missing a/b)")
            continue
        za, zb = np.load(pa), np.load(pb)
        for k in za.files:
            if k not in zb.files:
                continue
            keys.append((seg, tail_digits(k)))
            va.append(za[k])
            vb.append(zb[k])
    return keys, np.asarray(va, np.float32), np.asarray(vb, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--segs", required=True)
    ap.add_argument("--train-segs", required=True, help="训练段（用于惰性最近邻对照）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eval_segs = [s for s in a.segs.split(",") if s]
    train_segs = [s for s in a.train_segs.split(",") if s]

    sel = IdentitySelector.load(a.selector)
    K = len(sel.protos)
    print(f"[sel] loaded: {K} prototypes, thr_unknown={sel.thr_unknown}")

    keys, Va, Vb = load_split(a.emb_dir, eval_segs)
    N = len(keys)
    print(f"[eval] {N} tracks from {len(eval_segs)} segs")

    res = {"n_eval_tracks": N, "n_protos": K}

    # ① 分片一致性（known-restricted 主口径 + 全口径诊断）
    SA = sel.sims_all(Va)
    SB = sel.sims_all(Vb)
    idxa = SA.argmax(1)
    idxb = SB.argmax(1)
    sa_best = SA[np.arange(N), idxa]
    sb_best = SB[np.arange(N), idxb]
    ia = np.where(sa_best >= sel.thr_unknown, idxa, -1)
    ib = np.where(sb_best >= sel.thr_unknown, idxb, -1)
    kn = (ia >= 0) & (ib >= 0)
    cov = float(kn.mean())
    same_all = float((ia == ib).mean())
    same_known = float((ia[kn] == ib[kn]).mean()) if kn.sum() > 0 else None
    both_unk = float(((ia < 0) & (ib < 0)).mean())
    rng = np.random.default_rng(3)
    l = rng.integers(0, N, 30000)
    r = rng.integers(0, N, 30000)
    keep = l != r
    il, ir = ia[l[keep]], ib[r[keep]]
    coll_all = float((il == ir).mean())
    kn2 = (il >= 0) & (ir >= 0)
    coll_known = float((il[kn2] == ir[kn2]).mean()) if kn2.sum() > 0 else None
    print(f"[consistency] thr={sel.thr_unknown:.3f} coverage={cov:.3f} both_unknown={both_unk:.3f} | "
          f"same: all={same_all:.3f} known={round(same_known,3) if same_known is not None else None} | "
          f"coll: all={coll_all:.3f} known={round(coll_known,3) if coll_known is not None else None}")
    res["consistency"] = {"same_all": same_all, "same_known": same_known,
                          "collision_all": coll_all, "collision_known": coll_known,
                          "coverage": cov, "both_unknown": both_unk}

    # ①b thr 诊断曲线（工作点选择须在 train 上做；此处仅供分析展示）
    curve = []
    for t in [round(float(x), 3) for x in np.arange(0.88, 0.996, 0.01)]:
        ia_t = np.where(sa_best >= t, idxa, -1)
        ib_t = np.where(sb_best >= t, idxb, -1)
        kn_t = (ia_t >= 0) & (ib_t >= 0)
        cov_t = float(kn_t.mean())
        same_t = float((ia_t[kn_t] == ib_t[kn_t]).mean()) if kn_t.sum() > 0 else None
        il_t, ir_t = ia_t[l[keep]], ib_t[r[keep]]
        kn2_t = (il_t >= 0) & (ir_t >= 0)
        coll_t = float((il_t[kn2_t] == ir_t[kn2_t]).mean()) if kn2_t.sum() > 0 else None
        curve.append({"thr": float(t), "coverage": cov_t,
                      "same_known": same_t, "collision_known": coll_t})
    res["thr_curve"] = curve
    print("[thr-curve]")
    for c in curve:
        print(f"  thr={c['thr']:.3f} cov={c['coverage']:.3f} "
              f"same_known={c['same_known'] if c['same_known'] is None else round(c['same_known'],3)} "
              f"coll_known={c['collision_known'] if c['collision_known'] is None else round(c['collision_known'],3)}")

    # ② 跨集匹配（test → train 惰性最近邻）
    tk, Tva, Tvb = load_split(a.emb_dir, train_segs)
    T = _norm((Tva + Tvb) / 2.0, axis=1)
    Q = _norm((Va + Vb) / 2.0, axis=1)
    S = Q @ T.T                                   # [N_eval, N_train]
    idx = S.argmax(1)
    sims = S[np.arange(len(idx)), idx]
    res["cross_set"] = {
        "top1_sim_mean": float(sims.mean()),
        "top1_sim_p50": float(np.percentile(sims, 50)),
        "top1_sim_p90": float(np.percentile(sims, 90)),
        "frac_sim_gt_0.5": float((sims > 0.5).mean()),
        "frac_sim_gt_0.6": float((sims > 0.6).mean()),
        "frac_sim_gt_0.7": float((sims > 0.7).mean()),
    }
    print(f"[cross-set] top1 sim mean={sims.mean():.3f} p90={np.percentile(sims,90):.3f} "
          f">0.6: {(sims>0.6).mean():.3f} >0.7: {(sims>0.7).mean():.3f}")

    # top 匹配导出
    order = np.argsort(-sims)
    out_csv = a.out.replace(".json", "") + "_matches.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "query_seg", "query_tid", "match_seg", "match_tid", "sim"])
        for rank, oi in enumerate(order[:100], 1):
            w.writerow([rank, keys[oi][0], keys[oi][1], tk[idx[oi]][0],
                        tk[idx[oi]][1], f"{sims[oi]:.4f}"])
    print(f"[out] top100 matches -> {out_csv}")

    # ③ 随机基线
    r0 = np.random.default_rng(5).integers(0, len(T), N)
    base = float(np.mean((Q * T[r0]).sum(1)))
    res["random_baseline_top1_sim"] = base
    print(f"[baseline] random pair sim mean={base:.3f}")

    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] metrics -> {a.out}")


if __name__ == "__main__":
    main()
