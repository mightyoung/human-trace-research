#!/usr/bin/env python
"""去重前/后人数曲线对比（raw vs deduped vs tracked）。"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in ["Microsoft YaHei", "SimHei", "DengXian"]:
    try:
        font_manager.findfont(font_manager.FontProperties(family=f), fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [f]; break
    except Exception: continue
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = pathlib.Path("D:/dev/human-count/output_dedup")
FPS = 25.0
fig, axes = plt.subplots(2, 1, figsize=(11.5, 8), dpi=130)
for ax, stem in zip(axes, ["clip01", "clip02"]):
    recs = [json.loads(l) for l in (OUT/f"{stem}_detections.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    t = [r["frame"]/FPS for r in recs]
    raw = [r["raw_count"] for r in recs]
    ded = [r["deduped_count"] for r in recs]
    trk = [r["count"] for r in recs]
    ax.plot(t, raw, color="#9ca3af", lw=1.3, ls="--", label=f"原始检测 均值{np.mean(raw):.2f} std{np.std(raw):.2f}")
    ax.plot(t, ded, color="#059669", lw=1.6, label=f"去重后  均值{np.mean(ded):.2f} std{np.std(ded):.2f}")
    ax.plot(t, trk, color="#dc2626", lw=1.9, label=f"+跟踪   均值{np.mean(trk):.2f} std{np.std(trk):.2f}")
    ax.set_title(f"{stem}  人数对比：原始 → Containment-NMS 去重 → ByteTrack 跟踪", fontsize=11)
    ax.set_xlabel("时间 (秒)"); ax.set_ylabel("人数")
    ax.set_ylim(0, max(14, max(raw)+2)); ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
fig.suptitle("重复框去除效果 — Containment-NMS (阈值 0.85, 保留大框) + ByteTrack", fontsize=12.5)
fig.tight_layout()
p = OUT/"count_timeline.png"; fig.savefig(p, bbox_inches="tight"); print("saved", p)
