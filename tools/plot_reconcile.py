# -*- coding: utf-8 -*-
"""对账可视化：读 reconcile JSON（6 路）-> 输出 PNG 总览（2×2）。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"D:/human-count/output_crosscam_local"
CAMS = ["305", "316", "318", "320", "323", "343"]

det = json.load(open(f"{BASE}/reconcile_detect_all.json", encoding="utf-8"))
emb = json.load(open(f"{BASE}/reconcile_embed_all.json", encoding="utf-8"))

fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)

# (a) IoU 直方图（6 路合计）
ax = axes[0][0]
labels = ["<0.5", "0.5-0.7", "0.7-0.9", "0.9-0.95", "0.95-0.99", ">=0.99"]
total = None
for cam in CAMS:
    h = det.get(cam, {}).get("iou_hist")
    if not h:
        continue
    c = np.array(h["counts"], dtype=float)
    total = c if total is None else total + c
if total is not None and total.sum() > 0:
    pct = 100 * total / total.sum()
    ax.bar(labels, pct, color="#4C78A8")
    for i, v in enumerate(pct):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_title("(a) 检测框 IoU 分布（6 路合计，逐帧贪心配对）")
ax.set_ylabel("占比 %")

# (b) 305 逐帧 count 对比
ax = axes[0][1]
cs = det.get("305", {}).get("count_series", {})
if cs:
    l = np.array(cs["local"]); r = np.array(cs["remote"])
    x = np.arange(len(l))
    ax.plot(x, r, lw=0.9, label="远程 CUDA", color="#4C78A8", alpha=0.85)
    ax.plot(x, l, lw=0.9, label="本地 DML", color="#E45756", alpha=0.85)
    ax.legend(fontsize=8)
ax.set_title("(b) 305 逐帧人数（tracked count）")
ax.set_xlabel("处理帧序号（stride=5）")

# (c) 逐路指标
ax = axes[1][0]
w = 0.35
x = np.arange(len(CAMS))
iou_mean = [det.get(c, {}).get("iou_mean", 0) for c in CAMS]
cnt_within1 = [det.get(c, {}).get("count_within1", 0) for c in CAMS]
ax.bar(x - w / 2, iou_mean, w, label="IoU mean", color="#54A24B")
ax.bar(x + w / 2, cnt_within1, w, label="count |d|<=1 占比", color="#B279A2")
ax.set_xticks(x); ax.set_xticklabels(CAMS)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=8)
ax.set_title("(c) 检测对账：IoU / 计数一致性（逐路）")

# (d) 嵌入 cos 分布（内容级匹配，6 路合并）
ax = axes[1][1]
coss = []
for cam in CAMS:
    for m in emb.get(cam, {}).get("matched_all", []):
        coss.append(m["cos"])
if coss:
    ax.hist(coss, bins=40, range=(0, 1), color="#F58518")
    ax.set_title(f"(d) 嵌入 cos 分布（内容级匹配，n={len(coss)}）")
    ax.set_xlabel("cos(本地嵌入, 远程嵌入)")
else:
    ax.text(0.5, 0.5, "(无数据)", ha="center")

fig.suptitle("本地链路 vs 远程历史产物 对账总览（2026-10-17）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{BASE}/reconcile_overview.png"
fig.savefig(out)
print("saved", out)
