#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 D7 的 curve.csv 画成图 2 的两个面板。

清单要的是「精度—覆盖」曲线，但 D4 真值未到位，精度轴暂时缺。
本脚本用两个无标签代理替代，并在图上标注：

  左：覆盖 coverage  vs  两片同簇率 same   （same 高 = 判定更自洽）
  右：覆盖 coverage  vs  误合并率 fm       （fm 低 = 更少把不同人并一起）

精度必须等 D4 标完再补第三张，不能用这两张代替。
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "deliverable/data-request-v2/results")

# name -> (颜色, 线宽, 是否画工作点)
STYLE = {
    "A_baseline":                 ("#6b7280", 3.0, True),
    "C_dmax0.30":                 ("#2563eb", 1.8, True),
    "C_dmax0.34":                 ("#93c5fd", 1.4, False),
    "D_split0.20":                ("#059669", 1.8, True),
    "D_split0.20_veto0.30":       ("#dc2626", 3.0, True),
    "G_member_complete":          ("#7c3aed", 1.6, True),
    "G_global0.20_veto0.30":      ("#ea580c", 1.6, True),
}
LABEL = {
    "A_baseline":            "A  baseline (no veto)",
    "C_dmax0.30":            "C  d_max veto 0.30 (production)",
    "C_dmax0.34":            "C  d_max veto 0.34 (mild)",
    "D_split0.20":           "D  cross-cam split 0.20",
    "D_split0.20_veto0.30":  "D  split 0.20 + veto 0.30",
    "G_member_complete":     "G  member-level complete",
    "G_global0.20_veto0.30": "G  global 0.20 + veto",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT)
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.dir, "curve.csv"), encoding="utf-8-sig")))
    import json
    best = json.load(open(os.path.join(a.dir, "best.json"), encoding="utf-8"))

    dark = a.theme == "dark"
    fg = "#e5e7eb" if dark else "#111827"
    bg = "#111827" if dark else "#ffffff"
    grid = "#374151" if dark else "#e5e7eb"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=150)
    fig.patch.set_facecolor(bg)

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(True, color=grid, lw=0.7, alpha=0.8)
        ax.set_xlabel("coverage  (fraction of 229 queries with both halves above θ)",
                      color=fg, fontsize=10)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=fg, labelsize=9)

    for name, (c, lw, mark) in STYLE.items():
        sub = [r for r in rows if r["config"] == name]
        if not sub:
            continue
        sub.sort(key=lambda r: float(r["coverage"]))
        x = [float(r["coverage"]) for r in sub]
        ys = [float(r["same"]) if r["same"] else None for r in sub]
        yf = [float(r["fm"]) * 100 if r["fm"] else None for r in sub]

        ax = axes[0]
        xs = [xx for xx, yy in zip(x, ys) if yy is not None]
        yy = [v for v in ys if v is not None]
        ax.plot(xs, yy, color=c, lw=lw, label=LABEL.get(name, name), solid_capstyle="round")

        ax = axes[1]
        xs = [xx for xx, yy in zip(x, yf) if yy is not None]
        yy = [v for v in yf if v is not None]
        ax.plot(xs, yy, color=c, lw=lw, label=LABEL.get(name, name), solid_capstyle="round")

        if mark and name in best:
            b = best[name]
            axes[0].scatter([b["coverage"]], [b["same"]], s=90, color=c,
                            edgecolor=bg, linewidth=1.6, zorder=5)
            if b.get("false_merge") is not None:
                axes[1].scatter([b["coverage"]], [b["false_merge"] * 100], s=90, color=c,
                                edgecolor=bg, linewidth=1.6, zorder=5)
            axes[0].annotate(f"θ={b['thr']:.2f}\nK={b['K']}", (b["coverage"], b["same"]),
                             textcoords="offset points", xytext=(7, -14),
                             fontsize=7.5, color=c)

    axes[0].axhline(0.90, color="#dc2626", ls="--", lw=1.0, alpha=0.75)
    axes[0].text(0.30, 0.9025, "self-cal floor  same ≥ 0.90", color="#dc2626", fontsize=8)
    axes[0].set_ylabel("a/b same-cluster rate", color=fg, fontsize=10)
    axes[0].set_title("Coverage vs. a/b consistency  (higher-left is better)",
                      color=fg, fontsize=11, pad=10)

    axes[1].axhline(2.0, color="#dc2626", ls="--", lw=1.0, alpha=0.75)
    axes[1].text(0.30, 2.03, "fm budget 2%", color="#dc2626", fontsize=8)
    axes[1].set_ylabel("false-merge rate  (%)", color=fg, fontsize=10)
    axes[1].set_title("Coverage vs. false merge  (higher-left is worse)",
                      color=fg, fontsize=11, pad=10)

    for ax in axes:
        ax.set_xlim(0.28, 0.78)

    leg = axes[0].legend(loc="lower right", fontsize=8, framealpha=0.95)
    leg.get_frame().set_facecolor(bg)
    leg.get_frame().set_edgecolor(grid)
    for t in leg.get_texts():
        t.set_color(fg)

    fig.suptitle("D7 — Fig. 2 proxy  |  precision axis pending D4 ground truth  "
                 "(markers = self-cal work points)", color=fg, fontsize=10, y=0.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(a.dir, "fig2_proxy.png")
    fig.savefig(out, facecolor=bg)
    print("[out]", out)


if __name__ == "__main__":
    main()
