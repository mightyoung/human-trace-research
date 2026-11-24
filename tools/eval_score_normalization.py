#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R3 评分层实验：query 侧分数归一化 / 竞争簇校正 / 多原型，能否推高 same≥0.90 下的覆盖。

全部为 NumPy 后处理，不训练、不新增标注、不重跑编码。
每个变体独立扫描 theta，按生产口径 self-cal（same>=0.90 且 evaluable>=100 中取覆盖最大）。

变体：
  base            生产配置（每簇单原型，per-identity z 校准，全局阈值）
  tnorm           查询级 T-norm：z 再按该 query 在所有簇上的均值/标准差归一（argmax 不变，只改接受）
  asnormK         自适应 cohort：每个候选簇用「除它以外分数最高的 K 个簇」做归一（会改排名）
  snormK          对称：0.5*(z + asnormK)
  compL          竞争簇校正：z_k - L * mean(top3 其它簇 z)（会改排名）
  percam          每簇按机位建原型（>=2 条才建），评分取最大
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/r3_score_norm.json")
GRID = np.round(np.arange(-2.0, 8.01, 0.05), 3)


def build():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    cams = np.array([k[0].split("_")[0] for k in ktr])
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g = Vtr.mean(0)
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    return ktr, kte, Vtr, Va_te, Vb_te, cams, lab, kept, K, mu_g, mu_q


def main():
    ktr, kte, Vtr, Va_te, Vb_te, cams, lab, kept, K, mu_g, mu_q = build()

    # 单原型（生产）
    p_single, pcl_single, _ = S.build_gallery(Vtr, lab, cams, kept, percam=False)
    p_single = _norm(p_single - mu_g)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, p_single, pcl_single, K, mu=mu_g), lab, kept)

    def calibrated(X):
        Xa = _norm(_norm(X) - mu_q)
        Sx = S.cluster_scores(Xa, p_single, pcl_single, K)
        return (Sx - m_hat) / s_hat

    Za, Zb = calibrated(Va_te), calibrated(Vb_te)

    # 多原型（每机位）
    p_pc, pcl_pc, _ = S.build_gallery(Vtr, lab, cams, kept, percam=True)
    p_pc = _norm(p_pc - mu_g)
    m2, s2, _ = S.fit_ident_stats(S.score_rows(Vtr, p_pc, pcl_pc, K, mu=mu_g), lab, kept)

    def calibrated_pc(X):
        Xa = _norm(_norm(X) - mu_q)
        Sx = S.cluster_scores(Xa, p_pc, pcl_pc, K)
        return (Sx - m2) / s2

    Pa, Pb = calibrated_pc(Va_te), calibrated_pc(Vb_te)

    def tnorm(Z):
        mu = Z.mean(1, keepdims=True)
        sd = Z.std(1, keepdims=True) + 1e-6
        return (Z - mu) / sd

    def asnorm(Z, k):
        out = np.empty_like(Z)
        for i in range(Z.shape[0]):
            s = Z[i]
            for c in range(Z.shape[1]):
                others = np.delete(s, c)
                kk = min(k, others.size)
                coh = np.sort(others)[-kk:]
                out[i, c] = (s[c] - coh.mean()) / (coh.std() + 1e-6)
        return out

    def comp(Z, lam, topn=3, keep_self=True):
        out = np.empty_like(Z)
        for i in range(Z.shape[0]):
            s = Z[i]
            for c in range(Z.shape[1]):
                others = np.delete(s, c)
                out[i, c] = s[c] - lam * np.sort(others)[-topn:].mean()
        return out

    variants = {
        "base": (Za, Zb),
        "tnorm": (tnorm(Za), tnorm(Zb)),
        "asnorm16": (asnorm(Za, 16), asnorm(Zb, 16)),
        "asnorm32": (asnorm(Za, 32), asnorm(Zb, 32)),
        "asnorm64": (asnorm(Za, 64), asnorm(Zb, 64)),
        "snorm32": (0.5 * (Za + asnorm(Za, 32)), 0.5 * (Zb + asnorm(Zb, 32))),
        "comp0.2": (comp(Za, 0.2), comp(Zb, 0.2)),
        "comp0.3": (comp(Za, 0.3), comp(Zb, 0.3)),
        "comp0.5": (comp(Za, 0.5), comp(Zb, 0.5)),
        "percam": (Pa, Pb),
    }

    cooc = []
    keys_ok = set(kte)
    for seg in S.TEST_SEGS:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        cooc += [((seg, u), (seg, v)) for u, v in cooc_pairs(p, keys_ok, seg, 5)]

    res = {}
    for name, (Sa, Sb) in variants.items():
        rows = S.sweep(Sa, Sb, kte, cooc, GRID)
        ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
        best = max(ok, key=lambda r: r["coverage"]) if ok else None
        # 同覆盖 0.568 附近的工作点（最近）
        near = min(rows, key=lambda r: abs(r["coverage"] - 0.568))
        res[name] = {"self_cal": best, "at_cov_0.568": near}
        b = best or {}
        print(f"[{name:9s}] thr={b.get('thr')} cov={b.get('coverage', float('nan')):.3f} "
              f"same={b.get('same', float('nan')):.3f} fm={b.get('false_merge', float('nan'))*100:.2f}% "
              f"eval={b.get('evaluable')} | @cov.568: same={near['same']:.3f} "
              f"fm={near['false_merge']*100:.2f}%", flush=True)

    json.dump({"n_test": len(kte), "K": K, "n_cooc": len(cooc), "grid": [float(x) for x in GRID],
               "variants": res}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
