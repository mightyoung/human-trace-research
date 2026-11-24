#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R3 修正版：归一化分数 + 绝对余弦下限（双门）。

上一轮发现：T-norm 的 +9pp 主要来自 33 条「相对有峰但绝对余弦仅 0.50」的轨迹，
其中 26 条低于 base 已接受组的 p10（0.534）。本脚本加一个绝对证据门：
两片的 top-1 原始余弦都必须 >= floor，才允许按归一化分数接受。

扫描 floor 与 theta，输出 same>=0.90 下的最大覆盖。
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
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/r3_dual_gate.json")
GRID = np.round(np.arange(-2.0, 8.01, 0.05), 3)


def main():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g, mu_q = Vtr.mean(0), _norm((Va_te + Vb_te) / 2.0).mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)

    def parts(X):
        Xa = _norm(_norm(X) - mu_q)
        raw = S.cluster_scores(Xa, protos, pcl, K)
        z = (raw - m_hat) / s_hat
        return raw, z

    Ra, Za = parts(Va_te)
    Rb, Zb = parts(Vb_te)
    Ta = (Za - Za.mean(1, keepdims=True)) / (Za.std(1, keepdims=True) + 1e-6)
    Tb = (Zb - Zb.mean(1, keepdims=True)) / (Zb.std(1, keepdims=True) + 1e-6)

    cooc = []
    keys_ok = set(kte)
    for seg in S.TEST_SEGS:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        cooc += [((seg, u), (seg, v)) for u, v in cooc_pairs(p, keys_ok, seg, 5)]

    absmin = np.minimum(Ra.max(1), Rb.max(1))

    # 把绝对下限编码成「这两条轨迹永远不被接受」：给它们的分数填 -inf
    res = {}
    for floor in (None, 0.50, 0.55, 0.60, 0.65):
        if floor is None:
            Sa, Sb, tag = Ta, Tb, "tnorm"
        else:
            kill = absmin < floor
            Sa, Sb, tag = Ta.copy(), Tb.copy(), f"tnorm+abs{floor:.2f}"
            Sa[kill] = -1e9
            Sb[kill] = -1e9
        rows = S.sweep(Sa, Sb, kte, cooc, GRID)
        ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
        best = max(ok, key=lambda r: r["coverage"]) if ok else None
        res[tag] = best
        b = best or {}
        print(f"[{tag:16s}] thr={b.get('thr')} cov={b.get('coverage', float('nan')):.3f} "
              f"same={b.get('same', float('nan')):.3f} fm={b.get('false_merge', float('nan'))*100:.2f}% "
              f"eval={b.get('evaluable')} merged={int(round(b.get('false_merge', 0)*b.get('evaluable', 0)))}")

    json.dump({"n_test": len(kte), "K": K, "results": res},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
