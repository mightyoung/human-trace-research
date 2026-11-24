# -*- coding: utf-8 -*-
"""规模化实验汇总图：AUC 对比 / 接力边与链 / 跨机位检索命中。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

A = os.path.join(ROOT, "output_crosscam", "analysis")
BS = os.path.join(A, "bridge_scale")
RL = os.path.join(A, "relay")

eval_p = os.path.join(BS, "eval_results.json")
if not os.path.exists(eval_p):
    raise SystemExit(f"缺 {eval_p}")
ev = json.load(open(eval_p, encoding="utf-8"))


def relay_stat(tag):
    p = os.path.join(RL, f"relay_mutual_global_{tag}.json")
    c = os.path.join(RL, f"relay_chains_{tag}.json")
    s = os.path.join(RL, f"relay_scored_{tag}.json")
    n_thr = n_mut = n_chain = n_multi = 0
    if os.path.exists(s):
        d = json.load(open(s, encoding="utf-8"))
        n_thr = sum(1 for x in d["candidates"] if x["sim"] >= d["sim_thr"])
    if os.path.exists(p):
        n_mut = json.load(open(p, encoding="utf-8"))["n"]
    if os.path.exists(c):
        d = json.load(open(c, encoding="utf-8"))
        n_chain = d["n_chains"]
        n_multi = len([x for x in d["chains"] if len(x) >= 3])
    return n_thr, n_mut, n_chain, n_multi


demo_p = os.path.join(A, "global_reid", "demo_results.json")
demo = json.load(open(demo_p, encoding="utf-8")) if os.path.exists(demo_p) else []

fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)

# ---- P1: AUC 对比 ----
ax = fig.add_subplot(gs[0, 0:2])
protos = [("protoA_30pair", "协议A 30对\n(跨机位)"),
          ("B_same", "同机位链接\n(58正/3负)"),
          ("B_cross", "跨机位\n(8正/24负)"),
          ("B_all", "全量\n(66正/27负)")]
x = range(len(protos))
w = 0.36
vals_d = [ev["d3b"][k].get("auc") or 0 for k, _ in protos]
vals_c = [ev["cradio_scale"][k].get("auc") or 0 for k, _ in protos]
ax.bar([i - w / 2 for i in x], vals_d, w, label="DINOv3-B", color="#8c8c8c",
       edgecolor="black", linewidth=0.5)
ax.bar([i + w / 2 for i in x], vals_c, w, label="C-RADIOv4×Gemma2 (大样本W)",
       color="#ff7f0e", edgecolor="black", linewidth=0.5)
for i, (a, b) in enumerate(zip(vals_d, vals_c)):
    ax.text(i - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels([t for _, t in protos], fontsize=9.5)
ax.set_ylim(0.8, 1.02)
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
ax.set_title("识别准确率：两种嵌入在四组评估上的 AUC（留出跨机位轨）", fontsize=11.5)
ax.grid(alpha=0.25, axis="y")
ax.legend(fontsize=9.5, loc="lower right")

# ---- P2: 接力路径 ----
ax = fig.add_subplot(gs[0, 2])
rd = relay_stat("d3b")
rc = relay_stat("cradio")
labels = ["sim≥thr", "mutual 边", "链", "多跳链"]
xd = range(len(labels))
ax.bar([i - 0.2 for i in xd], rd, 0.4, label="DINOv3-B", color="#8c8c8c",
       edgecolor="black", linewidth=0.5)
ax.bar([i + 0.2 for i in xd], rc, 0.4, label="C-RADIOv4 桥接", color="#ff7f0e",
       edgecolor="black", linewidth=0.5)
for i, (a, b) in enumerate(zip(rd, rc)):
    ax.text(i - 0.2, a + 6, str(a), ha="center", fontsize=9)
    ax.text(i + 0.2, b + 6, str(b), ha="center", fontsize=9)
ax.set_xticks(list(xd))
ax.set_xticklabels(labels, fontsize=9.)
ax.set_title("路径跟踪：接力候选→边→链\n（各自校准阈值）", fontsize=11)
ax.grid(alpha=0.25, axis="y")
ax.legend(fontsize=8.5)

# ---- P3: 跨机位检索命中 ----
ax = fig.add_subplot(gs[1, 0])
names, r1_full, r1_320, r5_320 = [], [], [], []
for d in demo:
    pairs = d["pairs"]
    n = len(pairs)
    r1f = sum(1 for r in pairs if r.get("rank") == 1)
    r1s = sum(1 for r in pairs if r.get("rank_320") == 1)
    r5s = sum(1 for r in pairs if r.get("rank_320") and r["rank_320"] <= 5)
    names.append(d["model"])
    r1_full.append(r1f)
    r1_320.append(r1s)
    r5_320.append(r5s)
xx = range(len(names))
w2 = 0.25
ax.bar([i - w2 for i in xx], r1_full, w2, label="全库 rank-1 (8对)", color="#d62728",
       edgecolor="black", linewidth=0.5)
ax.bar(list(xx), r1_320, w2, label="320库 rank-1", color="#2ca02c",
       edgecolor="black", linewidth=0.5)
ax.bar([i + w2 for i in xx], r5_320, w2, label="320库 rank≤5", color="#98df8a",
       edgecolor="black", linewidth=0.5)
for i in xx:
    ax.text(i - w2, r1_full[i] + 0.1, str(r1_full[i]), ha="center", fontsize=9)
    ax.text(i, r1_320[i] + 0.1, str(r1_320[i]), ha="center", fontsize=9)
    ax.text(i + w2, r5_320[i] + 0.1, str(r5_320[i]), ha="center", fontsize=9)
ax.set_xticks(list(xx))
ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylim(0, 8.8)
ax.set_title("全局 ReID 检索（8 个已知配对查询）", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.25, axis="y")

# ---- P4: 规模与成本表 ----
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
if demo:
    est = demo[0]["estimate"]
    lat = demo[0]["latency_ms"]
    rows = [
        ["嵌入存储 / 轨", f"{est['bytes_per_track']:.0f} B (768d fp16)"],
        ["轨道吞吐", f"{est['tracks_per_hour_6cam']} 轨/时（6机位）"],
        ["存储 / 天（6机位）", f"{est['mb_per_day_6cam']:.0f} MB"],
        ["存储 / 年（6机位）", f"{est['gb_per_year_6cam']:.1f} GB"],
        ["存储 / 年（50机位）", f"{est['gb_per_year_50cam']:.0f} GB"],
        ["暴力检索延迟", f"{lat:.1f} ms / 727轨"],
        ["检索规模上限(内存)", "≤200万轨 / 64GB"],
    ]
    tb = ax.table(cellText=rows, loc="center", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.7)
ax.set_title("缓存层容量与延迟（实测）", fontsize=11)

# ---- P5: 融合口径注记 ----
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
txt = (
    "口径说明\n"
    "─────────────\n"
    "· W（1152→768 ridge）在 6 机位全部轨道上训练\n"
    "  （≈5300 图 / 137 万 token 对），\n"
    "  排除 30对+跨机位标签涉及的 36 轨 → 留出\n"
    "· 协议A/跨机位为留出泛化指标；\n"
    "  同机位 58 链接为域内运行指标\n"
    "· 接力检索引擎：时间门 + 全局 mutual-best\n"
    "· 阈值按各自分数带校准\n"
    "  （DINOv3-B: 0.65；C-RADIOv4 桥接: 0.93）"
)
ax.text(0.02, 0.98, txt, fontsize=10.5, va="top")

fig.suptitle("C-RADIOv4 × EmbeddingGemma 2 桥接规模化实验（六机位全轨迹）",
             fontsize=13.5, y=0.995)
out = os.path.join(BS, "scale_summary.png")
os.makedirs(BS, exist_ok=True)
plt.savefig(out, dpi=125, bbox_inches="tight")
print("saved", out)
