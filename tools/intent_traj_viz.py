#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图预测可视化：抽样轨迹画"历史窗(蓝) → 未来窗实际(类色) + 预测标签"对比格图。

用法：python tools/intent_traj_viz.py --npz <test.npz> --ckpt <best.pt> --out <jpg> [--per-class 6]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intent_model import IntentBERT, CLASS_NAMES  # noqa: E402

CLS_COLOR = {0: "#d62728", 1: "#2ca02c", 2: "#ff7f0e", 3: "#9467bd", 4: "#8c564b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=5)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(a.npz, allow_pickle=True)
    X, y = z["X"], z["y"]
    Xf = z["Xf"] if "Xf" in z.files else None
    meta = z["meta"] if "meta" in z.files else None
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    mu, sd = ck["mu"], ck["sd_norm"]
    cfg = ck["cfg"]
    model = IntentBERT(n_feat=cfg["n_feat"], d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                       n_heads=cfg["n_heads"], dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
                       max_len=cfg["max_len"])
    model.load_state_dict(ck["sd"])
    model.eval()
    Xn = (X - mu) / sd
    with torch.no_grad():
        pred = model(torch.from_numpy(Xn.astype(np.float32))).argmax(-1).numpy()

    # 按类抽样（尽量正确+错误混合）
    rng = np.random.default_rng(1)
    picks = []
    for c in range(5):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        correct = idx[pred[idx] == c]
        wrong = idx[pred[idx] != c]
        n_c = min(a.per_class, len(idx))
        take = []
        take += list(rng.choice(correct, min(n_c, len(correct)), replace=False)) if len(correct) else []
        remain = n_c - len(take)
        if remain > 0 and len(wrong):
            take += list(rng.choice(wrong, min(remain, len(wrong)), replace=False))
        picks += take

    n = len(picks)
    ncol = min(6, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow), squeeze=False)
    for k, oi in enumerate(picks):
        ax = axes[k // ncol][k % ncol]
        pts = X[oi][:, :2]           # u, v（归一化）
        ax.plot(pts[:, 0], pts[:, 1], "-o", ms=2.5, lw=1.4, color="#1f77b4")
        if Xf is not None:
            fu = Xf[oi]
            seg_u = np.concatenate([[pts[-1, 0]], fu[:, 0]])
            seg_v = np.concatenate([[pts[-1, 1]], fu[:, 1]])
        else:
            seg_u, seg_v = pts[-1:, 0], pts[-1:, 1]
        ax.plot(seg_u, seg_v, "-o", ms=3, lw=1.8, color=CLS_COLOR[int(y[oi])])
        ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=30, marker="s", color="#1f77b4", zorder=5)
        ax.set_title(f"true={CLASS_NAMES[int(y[oi])]}\npred={CLASS_NAMES[int(pred[oi])]}",
                     fontsize=7.5, color=("black" if pred[oi] == y[oi] else "#b30000"))
        ax.invert_yaxis()
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Intent samples (blue=history 2.4s, colored=future 1s, red title=wrong)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    acc = float((pred == y).mean())
    print(f"[viz] {n} samples, overall acc={acc:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
