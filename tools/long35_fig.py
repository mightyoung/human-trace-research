# -*- coding: utf-8 -*-
"""长窗总览图：三路并发曲线 + 轨迹时长分布 + 时间三段稳定性（数据齐后运行）。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis", "long35")

fp = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
matplotlib.rcParams["axes.unicode_minus"] = False
COLORS = {"316": "#1f77b4", "318": "#2ca02c", "320": "#d62728"}
NAMES = {"316": "316 走廊（流动型）", "318": "318 办公区（驻留型）", "320": "320 球机（稀疏型）"}


def main():
    d = json.load(open(os.path.join(ANA, "analysis_long35.json"), encoding="utf-8"))
    cams = [c for c in ["316", "318", "320"] if c in d]

    fig = plt.figure(figsize=(16, 9), dpi=105)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22)

    # (1) 并发 mean 曲线
    ax1 = fig.add_subplot(gs[0, :])
    for cam in cams:
        c = d[cam]["concurrency"]["bucket_60s"]
        t = np.array(c["t"]) / 60
        ax1.plot(t, c["mean"], lw=2, color=COLORS[cam], label=f"{NAMES[cam]} · 均值")
        ax1.plot(t, c["max"], lw=1, ls="--", alpha=0.45, color=COLORS[cam])
    ax1.set_xlabel("时间（分钟，自 08:55:00）", fontproperties=fp)
    ax1.set_ylabel("同时目标数", fontproperties=fp)
    ax1.set_title("三机位并发曲线（35 分钟长窗；虚线=桶内峰值）", fontproperties=fp, fontsize=13)
    ax1.legend(prop=fp, ncol=3, loc="upper left")
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, 35)

    # (2) 轨迹时长结构（堆叠柱）
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(cams))
    labels = []
    for i, cam in enumerate(cams):
        st = d[cam]["stat"]
        seg = st["seg"]
        total = st["n_tracks"]
        # 堆叠：<2s / 2-10 / 10-60 / >60
        vals = [seg["lt2s"], seg["lt10s"], seg["lt60s"], seg["ge60s"]]
        bottom = 0
        for v, col, lab in zip(vals, ["#c6dbef", "#6baed6", "#2171b5", "#08306b"],
                               ["<2s", "2–10s", "10–60s", ">60s"]):
            ax2.bar(i, v / total * 100, bottom=bottom / total * 100, color=col,
                    label=lab if i == 0 else None, width=0.6)
            bottom += v
        labels.append(f"{cam}\n({total} 轨)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontproperties=fp)
    ax2.set_ylabel("轨迹占比 %", fontproperties=fp)
    ax2.set_title("轨迹时长结构", fontproperties=fp, fontsize=13)
    ax2.legend(prop=fp, loc="upper right")
    ax2.grid(alpha=0.2, axis="y")

    # (3) 时间三段：并发 vs 碎片率
    ax3 = fig.add_subplot(gs[1, 1])
    w = 0.25
    for i, cam in enumerate(cams):
        tsg = d[cam]["time_segments"]
        xs = np.arange(3) + (i - 1) * w
        frag_rate = [s["n_frag"] / max(s["n_active"], 1) for s in tsg]
        ax3.bar(xs, frag_rate, width=w, color=COLORS[cam], alpha=0.85,
                label=NAMES[cam])
        for x_, fr in zip(xs, frag_rate):
            ax3.text(x_, fr + 0.005, f"{fr:.2f}", ha="center", fontsize=9,
                     fontproperties=fp)
    ax3.set_xticks(np.arange(3))
    ax3.set_xticklabels(["前 1/3", "中 1/3", "后 1/3"], fontproperties=fp)
    ax3.set_ylabel("碎片重连 / 活跃轨迹", fontproperties=fp)
    ax3.set_title("时间三段稳定性：碎片率（按活跃归一化）", fontproperties=fp, fontsize=13)
    ax3.legend(prop=fp, fontsize=9)
    ax3.grid(alpha=0.2, axis="y")

    out = os.path.join(ANA, "long35_overview.png")
    plt.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
