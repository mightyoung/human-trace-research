"""E21 中文稿配图（英文标注，避免字体问题）。数据全部取自 docs/e21-results-2026-11-09.md。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_A, C_C, C_ACC = "#4C72B0", "#DD8452", "#C44E52"
GREY = "#555555"


# ---------- 图 1：三级阈值的作用域 ----------
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, text, fc="#EEF2F8", ec=C_A):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     lw=1.0, color=GREY))

    box(1, 17, 15, 12, "tracklets\n(per camera)")
    box(21, 17, 17, 12, "within-camera\nmerge", fc="#FBEEE4", ec=C_C)
    box(43, 17, 17, 12, "cross-camera\nmerge", fc="#FBEEE4", ec=C_C)
    box(64, 17, 13, 12, "identity\nclusters")
    box(83, 17, 14, 12, "retrieval")

    for x1, x2 in ((16.6, 20.4), (38.6, 42.4), (60.6, 63.4), (77.6, 82.4)):
        arrow(x1, 23, x2, 23)

    ax.text(29.5, 32.5, r"$\theta_{\rm in}$ (0.15)", ha="center", fontsize=9, color=C_C, weight="bold")
    ax.text(29.5, 12.5, "same-camera\nfalse merges\nONLY from here", ha="center", va="top",
            fontsize=7.3, color=C_C)
    ax.text(51.5, 32.5, r"$\theta_{\rm out}$ (0.25)", ha="center", fontsize=9, color=C_C, weight="bold")
    ax.text(51.5, 12.5, "cross-camera\nfalse merges\nONLY from here", ha="center", va="top",
            fontsize=7.3, color=C_C)
    ax.text(90.0, 32.5, r"$\theta$ (global)", ha="center", fontsize=9, color="#333333", weight="bold")
    ax.text(90.0, 12.5, "accept / reject\n= coverage", ha="center", va="top", fontsize=7.3, color="#333333")

    ax.text(50, 42, "Three thresholds, three disjoint error channels",
            ha="center", fontsize=9.5, weight="bold")
    fig.savefig(os.path.join(OUT, "fig1_threshold_scopes.png"))
    plt.close(fig)


# ---------- 图 2：删簇数随 thr_out 塌缩 ----------
def fig2():
    days = ["10-22", "10-23", "10-26"]
    v25 = [3, 9, 9]
    v20 = [1, 2, 1]
    x = np.arange(len(days)); w = 0.33
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    b1 = ax.bar(x - w / 2, v25, w, label=r"$\theta_{\rm out}=0.25$ (production)", color=C_A, ec="white")
    b2 = ax.bar(x + w / 2, v20, w, label=r"$\theta_{\rm out}=0.20$", color=C_C, ec="white")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.18, "%d" % h, ha="center", fontsize=8.4)
    for xi, a, b in zip(x, v25, v20):
        ax.text(xi, a + 1.35, "−%d%%" % round(100 * (1 - b / a)), ha="center",
                fontsize=8.6, color=C_ACC, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_ylabel(r"clusters removed by veto ($d_{\max}=0.27$)")
    ax.set_ylim(0, 11.6)
    ax.set_title("Veto headroom is manufactured by the cross-camera threshold",
                 fontsize=9.3, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.savefig(os.path.join(OUT, "fig4_veto_collapse.png"))
    plt.close(fig)


# ---------- 图 3：fm 随 theta 的形状 ----------
def fig3():
    th = [0.50, 0.90, 2.10, 3.30, 4.50]
    data = {"10-22": [12.1, 10.1, 11.2, 14.7, 18.1],
            "10-23": [22.5, 23.7, 28.9, 40.7, 55.8],
            "10-26": [4.3, 4.5, 6.6, 9.9, 19.4]}
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for (k, v), c, m in zip(data.items(), (C_A, C_C, "#55A868"), ("o", "s", "^")):
        ax.plot(th, v, marker=m, ms=4.5, lw=1.5, color=c, label=k)
    ax.set_xlabel(r"$\theta$ (z-score threshold; higher = stricter)")
    ax.set_ylabel("false-merge rate (%)")
    ax.set_ylim(0, 60)
    ax.set_title("Stricter matching does not mean fewer false merges", fontsize=9.3, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(2.55, 47.5, "monotone increase\nin 2 of 3 days", fontsize=7.8, color=GREY)
    ax.plot([3.0, 3.3], [45.0, 41.5], lw=0.9, color="#9A9A9A")
    fig.savefig(os.path.join(OUT, "fig3_fm_vs_theta.png"))
    plt.close(fig)


# ---------- 图 4：评估集漂移导致的翻转 ----------
def fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # 左：A 完全包含于 C
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10); ax1.axis("off")
    ax1.add_patch(plt.Circle((5.0, 5.4), 3.3, fc="#FBEEE4", ec=C_C, lw=1.2, ls="--"))
    ax1.add_patch(plt.Circle((3.9, 5.4), 1.75, fc="#DCE6F2", ec=C_A, lw=1.2))
    ax1.text(3.9, 5.4, "A\n299", ha="center", va="center", fontsize=8.6,
             color="#2F4E7A", weight="bold")
    ax1.text(7.35, 5.4, "C only\n196", ha="center", va="center", fontsize=8.4, color="#A25A2C")
    ax1.text(5.0, 1.1, r"A $\subset$ C  (A-only = 0)", ha="center", fontsize=8.4)
    ax1.set_title("Evaluable co-occurrence pairs", fontsize=9.2, pad=6)

    # 右：两种口径下的 fm
    labels = ["A\n(own)", "C\n(own)", "A\n(common)", "C\n(common)"]
    vals = [7.36, 6.06, 7.36, 7.69]
    bars = ax2.bar(range(4), vals, 0.58, color=[C_A, C_C, C_A, C_C], ec="white")
    for i, (b, a) in enumerate(zip(bars, [0.40, 0.40, 1.0, 1.0])):
        b.set_alpha(a)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.12, "%.2f" % v, ha="center", fontsize=8.2)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=7.8)
    ax2.set_ylabel("false-merge rate (%)"); ax2.set_ylim(0, 9.6)
    ax2.set_title("Same data, opposite verdict", fontsize=9.2, pad=6)
    ax2.axvline(1.5, color="#AAAAAA", lw=0.8, ls=":")
    ax2.text(0.5, 8.75, "C looks better", ha="center", fontsize=7.9, color="#666666")
    ax2.text(2.5, 8.75, "C is worse", ha="center", fontsize=7.9, color=C_ACC, weight="bold")
    fig.savefig(os.path.join(OUT, "fig2_eval_set_drift.png"))
    plt.close(fig)


for f in (fig1, fig2, fig3, fig4):
    f()
print("saved to", OUT)
for n in sorted(os.listdir(OUT)):
    print("  ", n, os.path.getsize(os.path.join(OUT, n)), "bytes")
