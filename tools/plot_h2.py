#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""H2 对照图：读 h2_portal_vs_camera_v1.json → 四联图。

(a) E1 纯转移 TP-FP 曲线  (b) E2 系统 TP-FP 曲线
(c) 冷启动 TP@FP<=2 vs 拟合样本量 N  (d) 探索性：同源组内排序正确率
产物：output_crosscam/analysis/relay_v2/h2_portal_vs_camera_v1.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "output_crosscam", "analysis", "relay_v2", "h2_portal_vs_camera_v1.json")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "relay_v2", "h2_portal_vs_camera_v1.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

COL = dict(cam="#4C78A8", b1="#F58518", b2="#E45756")
NAME = dict(cam="相机级 (cam)", b1="门户级·C0门 (b1)", b2="门户级·v2簇 (b2)")


def step_panel(ax, met, title, xmax, ymax):
    for m in ("cam", "b1", "b2"):
        cur = met[m]["curve"]
        fp = np.array([c[0] for c in cur], float)
        tp = np.array([c[1] for c in cur], float)
        ax.step(fp, tp, where="post", lw=1.8, color=COL[m],
                label=f"{NAME[m]}  AUC={met[m]['auc']:.3f}")
        b2 = met[m]["tp_at_fp"]["2"]
        ax.plot([2], [b2], "o", ms=7, mfc="none", mec=COL[m], mew=1.8)
    ax.axvline(2, color="#999", ls="--", lw=1)
    ax.set_xlim(-0.3, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("误接数 FP（被判同人但实为异人的对）")
    ax.set_ylabel("有效接续数 TP（被接受的真同人对）")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")


def main():
    d = json.load(open(J, encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), dpi=140)

    # (a)(b) TP-FP 曲线
    n1 = d["metrics"]["E1"]["cam"]
    step_panel(axes[0][0], d["metrics"]["E1"],
               f"(a) E1 纯转移：TP-FP 曲线（pos={n1['n_pos']} / neg={n1['n_neg']}）", 12, 12)
    n2 = d["metrics"]["E2"]["cam"]
    step_panel(axes[0][1], d["metrics"]["E2"],
               f"(b) E2 系统口径（base=sim+dt，只换转移项；pos={n2['n_pos']} / neg={n2['n_neg']}）", 12, 26)

    # (c) 冷启动
    ax = axes[1][0]
    cold = d["cold_start"]["E1"]
    for m in ("cam", "b1", "b2"):
        xs = [p["N"] for p in cold[m]]
        ys = [p["tp2"] for p in cold[m]]
        ax.plot(xs, ys, "o-", color=COL[m], lw=1.8, ms=5, label=NAME[m])
    ax.set_xlabel("已观察的通行事件数 N（拟合集按时序累积）")
    ax.set_ylabel("TP @ FP≤2")
    ax.set_title("(c) 冷启动：固定误接预算下的有效接续 vs 事件数（E1）", fontsize=11)
    ax.set_ylim(-0.3, 6.2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    ax.annotate("E1 下 TP 规模过小（≤5），\n曲线不可分辨 → '所需事件数' 结论不成立",
                xy=(0.02, 0.62), xycoords="axes fraction", fontsize=8.5,
                color="#444", bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#ccc"))

    # (d) 探索性补充：同源组内排序
    ax = axes[1][1]
    sup = d["supplement_within_a"]
    groups = [("E1", "E1 纯转移"), ("E1_343", "E1 · 343源"), ("E2", "E2 系统"), ("E2_343", "E2 · 343源")]
    width = 0.24
    xs = np.arange(len(groups))
    for i, m in enumerate(("cam", "b1", "b2")):
        vals = [sup[g][m]["acc"] for g, _ in groups]
        ns = [sup[g][m]["comparisons"] for g, _ in groups]
        bars = ax.bar(xs + (i - 1) * width, vals, width * 0.92, color=COL[m],
                      alpha=0.9, label=NAME[m])
        for x, v, n in zip(xs + (i - 1) * width, vals, ns):
            ax.text(x, v + 0.02, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[1] for g in groups], fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("同源组内 pos 排在 neg 之前 的比例")
    ax.set_title("(d) 探索性补充：同源(a端)候选排序正确率（未预注册）", fontsize=11)
    ax.axhline(0.5, color="#999", ls=":", lw=1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", ncol=3)

    fig.suptitle("H2 对照：门户级转移建模 vs 纯相机级图（冻结集 frozen_eval_v1，sha f02de206…；拟合剔除全部 109 判定行）",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
