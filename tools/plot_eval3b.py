#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval3b 轨迹图：读 self_train_v2.json → 输出自训练轨迹三联图。

产物：output_crosscam/analysis/eval3/self_train_v2.png
用法：python tools/plot_eval3b.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "output_crosscam", "analysis", "eval3", "self_train_v2.json")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "eval3", "self_train_v2.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    d = json.load(open(J, encoding="utf-8"))
    rounds = d["rounds"]
    rscan = d.get("rounds_scan", {})
    base_ab = d["arms"]["A1_frozen"]["ab_same"]
    naive = rounds["naive"]
    gated = rounds["gated"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=140)

    # (a) ab_same 轨迹
    ax = axes[0]
    xs = np.arange(1, 5)
    ax.axhline(base_ab, color="#888", ls="--", lw=1, label=f"冻结基线 {base_ab:.4f}")
    ax.plot(xs, [r["ab_same"] for r in naive], "o-", color="#d62728", label="朴素流式")
    ax.plot(xs, [r["ab_same"] for r in gated], "s-", color="#2ca02c", label="门控流式(p30)")
    for mk, color in (("p20", "#1f77b4"), ("p15", "#9467bd"), ("p10", "#ff7f0e")):
        if mk in rscan:
            ax.plot(xs, [r["ab_same"] for r in rscan[mk]], "--", alpha=0.7,
                    color=color, label=f"门控 {mk}")
    ax.set_title("(a) 分片一致率 ab_same（逐轮）")
    ax.set_xlabel("更新轮次")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, lw=0.5)

    # (b) test 换簇 churn
    ax = axes[1]
    w = 0.35
    ax.bar(xs - w / 2, [r["churn"]["switched"] for r in naive], w,
           color="#d62728", label="朴素 换簇数")
    ax.bar(xs + w / 2, [r["churn"]["switched"] for r in gated], w,
           color="#2ca02c", label="门控 换簇数")
    ax.set_title("(b) test 身份换簇 churn（越低越稳）")
    ax.set_xlabel("更新轮次")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.5)

    # (c) 接受成员/更新簇 对比
    ax = axes[2]
    ax.plot(xs, [r["n_confident"] for r in naive], "o-", color="#d62728", label="朴素 接受成员")
    ax.plot(xs, [r["n_confident"] for r in gated], "s-", color="#2ca02c", label="门控 接受成员")
    ax.plot(xs, [r["clusters_updated"] for r in naive], "o--", alpha=0.6,
            color="#d62728", label="朴素 更新簇数")
    ax.plot(xs, [r["clusters_updated"] for r in gated], "s--", alpha=0.6,
            color="#2ca02c", label="门控 更新簇数")
    ax.set_title("(c) 每轮接受成员 / 更新簇")
    ax.set_xlabel("更新轮次")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, lw=0.5)

    fig.suptitle("EVAL-3b 流式门控自训练：朴素 vs 门控（val 四时段逐轮累积）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
