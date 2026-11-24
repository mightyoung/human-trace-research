#!/usr/bin/env python
"""多后端 ReID 对比可视化：读 reid_compare/result.json 出图。

输出: output_crosscam/analysis/reid_compare/compare_fig.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "output_crosscam/analysis/reid_compare/result.json"
OUT = IN.parent / "compare_fig.png"

data = json.loads(IN.read_text(encoding="utf-8"))
res = data["results"]

# 后端显示顺序（固定，便于阅读）
ORDER = [k for k in ["dinov2", "dinov3s", "dinov3s@256", "dinov3b",
                     "dinov3b@256", "cradio224", "cradio512"] if k in res]
LABELS = {
    "dinov2": "DINOv2-S/14\n224",
    "dinov3s": "DINOv3-S/16\n224",
    "dinov3s@256": "DINOv3-S/16\n256",
    "dinov3b": "DINOv3-B/16\n224",
    "dinov3b@256": "DINOv3-B/16\n256",
    "cradio224": "C-RADIOv4-SO\n431M, 224",
    "cradio512": "C-RADIOv4-SO\n431M, 512",
}

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8),
                         gridspec_kw={"width_ratios": [2.35, 1.0]})

# ---- 左：分数分布散点 ----
ax = axes[0]
rng = np.random.default_rng(0)
for xi, k in enumerate(ORDER):
    r = res[k]
    pv = [x[2] for x in r["pos"] if x[2] is not None]
    nv = [x[2] for x in r["neg"] if x[2] is not None]
    jit = rng.uniform(-0.07, 0.07, len(pv))
    ax.scatter(np.full(len(pv), xi) + jit, pv, s=42, c="#1f77b4",
               marker="o", alpha=0.85, label="确认同人 (8)" if xi == 0 else None,
               zorder=3, edgecolors="white", linewidths=0.6)
    jit = rng.uniform(-0.07, 0.07, len(nv))
    ax.scatter(np.full(len(nv), xi) + jit, nv, s=34, c="#d62728",
               marker="x", alpha=0.75, label="确认异人 (22)" if xi == 0 else None,
               zorder=3)
    # min(pos) / max(neg) 连线示意
    ax.plot([xi - 0.22, xi + 0.22], [min(pv)] * 2, c="#1f77b4", lw=1.2,
            alpha=0.9, zorder=2)
    ax.plot([xi - 0.22, xi + 0.22], [max(nv)] * 2, c="#d62728", lw=1.2,
            alpha=0.9, zorder=2)
    ax.text(xi, 0.845, f"AUC={r['auc']:.3f}", ha="center", fontsize=10.5,
            fontweight="bold")

ax.axhline(0.7, color="gray", ls="--", lw=1, alpha=0.7)
ax.text(len(ORDER) - 0.45, 0.705, "阈值 0.7", fontsize=9, color="gray")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([LABELS[k] for k in ORDER], fontsize=9.5)
ax.set_ylim(0.25, 0.88)
ax.set_ylabel("余弦相似度（轨迹均值嵌入）")
ax.set_title("同一批 30 个判读对 · 各后端外观判别力", fontsize=12)
ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)
ax.grid(axis="y", alpha=0.25)

# ---- 右：AUC 条形 + 速度 ----
ax2 = axes[1]
aucs = [res[k]["auc"] for k in ORDER]
colors = ["#8c8c8c", "#5aa1e3", "#1f77b4", "#e07b39", "#b04a42"][:len(ORDER)]
bars = ax2.barh(range(len(ORDER)), aucs, color=colors, alpha=0.85)
ax2.set_yticks(range(len(ORDER)))
ax2.set_yticklabels([LABELS[k].replace("\n", " ") for k in ORDER],
                    fontsize=9.5)
ax2.set_xlim(0.5, 1.0)
ax2.invert_yaxis()
for i, k in enumerate(ORDER):
    r = res[k]
    ax2.text(r["auc"] + 0.005, i,
             f"{r['auc']:.3f}  ({r['ms_per_img']:.0f} ms/图)",
             va="center", fontsize=9.5)
ax2.set_xlabel("AUC（8 正 vs 22 负）")
ax2.set_title("判别力与 CPU 推理开销", fontsize=12)
ax2.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"[out] {OUT}")
