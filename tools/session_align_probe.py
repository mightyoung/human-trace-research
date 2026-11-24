#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会话级嵌入对齐的快速验证（对应调研报告 L1-#1）。

问题：1012 train（上午）建库，test（下午）原型分从 0.929 掉到 0.810，覆盖 0.89→0.27。
文献（PEA, ICLR 2026）主张域漂移 = 平移/缩放/旋转，可用无标签统计量校正。

本脚本对比三种表征方式（阈值统一扫描，取"误合并≤2% 下覆盖最大"的工作点）：
  raw  : 原始余弦（现状）
  mean : 各自减去本会话均值后重归一（平移校正）
  zca  : 各自用本会话协方差白化后再重归一（平移+缩放+旋转校正）
注意两侧统计量各自独立估计：train 用 train 嵌入，test 用 test 嵌入，互不泄露标签。

指标：覆盖（a/b 双侧接受）、稳定（条件一致）、安全（共现误合并）。
用法：python tools/session_align_probe.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector, _norm  # noqa: E402
from eval_selector_1012test import load_pair, cooc_pairs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "output_crosscam_local/1012_osnet_emb")
SEL = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/selector.npz")
TRAIN_SEGS = ["305_train_529", "318_train_525", "320_train_514", "323_train_510"]
TEST_SEGS = ["305_test_544", "318_test_534", "320_test_518", "323_test_527"]


def load(segs):
    keys, Va, Vb = [], [], []
    for s in segs:
        ks, a, b = load_pair(EMB, s)
        keys += ks
        Va.append(a)
        Vb.append(b)
    return keys, np.concatenate(Va), np.concatenate(Vb)


def zca_fit(X):
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    # 收缩到对角，抑制小样本奇异
    d = np.diag(np.diag(C))
    C = 0.8 * C + 0.2 * d + 1e-3 * np.eye(C.shape[0], dtype=np.float32)
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-8)
    W = (V / np.sqrt(w)) @ V.T
    return mu.astype(np.float32), W.astype(np.float32)


def transform(X, mode, stat):
    if mode == "raw":
        return _norm(X)
    mu = stat["mu"]
    if mode == "mean":
        return _norm(X - mu)
    mu, W = stat["mu"], stat["W"]
    return _norm((X - mu) @ W)


def evaluate(protos, Va, Vb, keys, cooc, thr_grid):
    kidx = {k: i for i, k in enumerate(keys)}
    best = None
    rows = []
    for thr in thr_grid:
        sa = (Va @ protos.T).max(1)
        sb = (Vb @ protos.T).max(1)
        ia = (Va @ protos.T).argmax(1)
        ib = (Vb @ protos.T).argmax(1)
        acc_a, acc_b = sa >= thr, sb >= thr
        both = acc_a & acc_b
        same = float((ia[both] == ib[both]).mean()) if both.any() else float("nan")
        ev = mg = 0
        for u, v in cooc:
            iu, iv = kidx[u], kidx[v]
            if (acc_a[iu] and acc_b[iv]) or (acc_a[iv] and acc_b[iu]):
                ev += 1
                if (acc_a[iu] and acc_b[iv] and ia[iu] == ib[iv]) or \
                   (acc_a[iv] and acc_b[iu] and ia[iv] == ib[iu]):
                    mg += 1
        fm = mg / ev if ev else float("nan")
        row = {"thr": round(float(thr), 3), "coverage": float(both.mean()),
               "same_cond": same, "false_merge": fm, "evaluable": ev}
        rows.append(row)
        if ev >= 100 and not np.isnan(fm) and fm <= 0.02:
            if best is None or row["coverage"] > best["coverage"]:
                best = row
    return rows, best


def main():
    sel = IdentitySelector.load(SEL)
    protos_raw = sel.protos.astype(np.float32)

    ktr, Vtr_a, Vtr_b = load(TRAIN_SEGS)
    kte, Vte_a, Vte_b = load(TEST_SEGS)
    print(f"train {len(ktr)} tracks / test {len(kte)} tracks / protos {len(protos_raw)}")

    cooc = []
    for s in TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte), s, 5):
            cooc.append(((s, u), (s, v)))
    print(f"cooc pairs {len(cooc)}")

    thr_grid = np.round(np.arange(0.10, 0.99, 0.01), 3)
    out = {"train_tracks": len(ktr), "test_tracks": len(kte), "cooc_pairs": len(cooc),
           "variants": {}}

    curves = {}
    for mode in ("raw", "mean", "zca"):
        if mode == "raw":
            st_tr = {"mu": np.zeros_like(protos_raw[0])}
            st_te = st_tr
        elif mode == "mean":
            st_tr = {"mu": Vtr_a.mean(0)}
            st_te = {"mu": Vte_a.mean(0)}
        else:
            st_tr = dict(zip(("mu", "W"), zca_fit(Vtr_a)))
            st_te = dict(zip(("mu", "W"), zca_fit(Vte_a)))

        P = transform(protos_raw, mode, st_tr)
        A = transform(Vte_a, mode, st_te)
        B = transform(Vte_b, mode, st_te)
        rows, best = evaluate(P, A, B, kte, cooc, thr_grid)
        out["variants"][mode] = {"best_fm_le_2pct": best, "sweep": rows}
        curves[mode] = rows
        b = best or {}
        print(f"[{mode:4s}] best@fm<=2%: thr={b.get('thr')} cov={b.get('coverage', float('nan')):.3f} "
              f"same={b.get('same_cond', float('nan')):.3f} fm={b.get('false_merge', float('nan')):.4f} "
              f"(eval={b.get('evaluable')})")

    # ---- 同覆盖率对照（消除工作点差异）----
    print("\n== 同覆盖率对照 ==")
    print(f"{'cov':>5} " + " ".join(f"{m:>18s}" for m in curves))
    matched = {}
    for target in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        line = f"{target:5.2f} "
        matched[str(target)] = {}
        for m, rows in curves.items():
            cand = [r for r in rows if r["coverage"] >= target and r["evaluable"] >= 100
                    and not np.isnan(r["false_merge"])]
            if not cand:
                line += f"{'n/a':>18s} "
                continue
            r = min(cand, key=lambda x: x["coverage"])
            matched[str(target)][m] = r
            line += f"{r['same_cond']:.3f}/{r['false_merge']*100:5.2f}%  "
        print(line + "  (same/fm)")
    out["matched_coverage"] = matched

    dst = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/session_align_probe.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {dst}")


if __name__ == "__main__":
    main()
