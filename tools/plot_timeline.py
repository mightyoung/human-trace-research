#!/usr/bin/env python
"""从检测结果 JSONL 生成人数时间序列曲线（raw 检测 vs ByteTrack 跟踪后）。"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in ["Microsoft YaHei", "SimHei", "DengXian"]:
    try:
        font_manager.findfont(font_manager.FontProperties(family=f), fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = pathlib.Path("D:/dev/human-count/output_tracked")
STRIDE = 5
FPS = 25.0


def load(stem):
    recs = [json.loads(l) for l in (OUT / f"{stem}_detections.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    t = [r["frame"] / FPS for r in recs]
    trk = [r["count"] for r in recs]
    raw = [r.get("raw_count", r["count"]) for r in recs]
    return t, trk, raw


fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), dpi=130)
for ax, stem in zip(axes, ["clip01", "clip02"]):
    t, trk, raw = load(stem)
    ax.plot(t, raw, color="#9ca3af", lw=1.4, ls="--", label=f"原始检测 (均值 {np.mean(raw):.1f})")
    ax.plot(t, trk, color="#dc2626", lw=1.9, label=f"ByteTrack 跟踪 (均值 {np.mean(trk):.1f})")
    ax.fill_between(t, trk, alpha=0.12, color="#dc2626")
    ax.set_title(
        f"{stem}  人数随时间变化   std: raw={np.std(raw):.2f} → tracked={np.std(trk):.2f}",
        fontsize=11,
    )
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("人数")
    ax.set_ylim(0, max(14, max(raw) + 2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

fig.suptitle(
    'GroundingDINO + ByteTrack 人数检测 — 2× NVIDIA L20  (prompt: "person", conf≥0.30, stride=5)',
    fontsize=12.5,
)
fig.tight_layout()
p = OUT / "count_timeline.png"
fig.savefig(p, bbox_inches="tight")
print("saved", p)
