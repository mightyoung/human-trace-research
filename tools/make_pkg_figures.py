#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「脱敏版」指标图（研究包用）。

原 `fig_multicam_timeline.png` 的子图标题内嵌场景名（办公区/茶水间/…），
PNG 无法用正则脱敏 → 从 `multicam_overview.json` **重新绘制**，
标题改为匿名代号 `<SITE-X>`，仅保留机位号与曲线。

用法: python tools/make_pkg_figures.py --out research_package/04_试验数据/09_图表
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "output_crosscam", "analysis", "multicam_overview.json")

# ★ 场景匿名映射（与 build_research_package.REDACT 保持一致，仅用于图内标题）
SITE = {"305": "SITE-A", "316": "SITE-B", "318": "SITE-C",
        "320": "SITE-D", "323": "SITE-E", "343": "SITE-F"}
ORDER = ["305", "316", "318", "320", "323", "343"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output_crosscam",
                                                  "analysis", "_pkg_fig"))
    ap.add_argument("--bin-sec", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(SRC, encoding="utf-8") as f:
        d = json.load(f)

    fig, axes = plt.subplots(3, 2, figsize=(11, 7.2))
    for ax, cam in zip(axes.ravel(), ORDER):
        c = d[cam]
        y = c.get("active_per_30s") or []
        x = [i * args.bin_sec / 60.0 for i in range(len(y))]
        ax.fill_between(x, y, alpha=0.45)
        ax.plot(x, y, lw=1.0)
        ax.set_title(f"{SITE[cam]} ({cam})", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("active tracks", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time within clip (min)", fontsize=8)
    fig.suptitle("Per-camera active-track timeline (6 cameras, 30 s bins, "
                 "synchronised 10-min slice; site labels anonymised)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(args.out, "fig_multicam_timeline__desensitized.png")
    fig.savefig(out, dpi=130)
    print(f"[fig] -> {out}")


if __name__ == "__main__":
    main()
