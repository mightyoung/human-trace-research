"""出图：D4 真值下的精度—覆盖前沿，与 d_max 分档错误率。"""
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
RUN = os.path.join(DELIV, "results")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


prec = {r["config"]: r for r in rd(os.path.join(RUN, "precision.csv"))}

groups = {
    "A baseline": ("A_baseline", "#111111", "o", 90),
    "C d_max veto": ([c for c in prec if c.startswith("C_dmax")], "#d95f02", "^", 34),
    "D split": ([c for c in prec if c.startswith("D_split") and "veto" not in c], "#1b9e77", "s", 34),
    "D split+veto": ([c for c in prec if c.startswith("D_split") and "veto" in c], "#7570b3", "D", 34),
    "G ablation": ([c for c in prec if c.startswith("G_")], "#666666", "x", 40),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

ax = axes[0]
for name, (cfgs, color, marker, size) in groups.items():
    if isinstance(cfgs, str):
        cfgs = [cfgs]
    xs = [float(prec[c]["precision"]) for c in cfgs if prec[c]["precision"]]
    ys = [int(prec[c]["C_over_229"]) for c in cfgs if prec[c]["precision"]]
    ax.scatter(xs, ys, s=size, c=color, marker=marker, alpha=0.75,
               label=name, edgecolors="white", linewidths=0.5)

for c in ["A_baseline", "C_dmax0.30", "D_split0.20", "D_split0.20_veto0.30"]:
    r = prec[c]
    ax.annotate(c.replace("_", " "), (float(r["precision"]), int(r["C_over_229"])),
                textcoords="offset points", xytext=(6, 5), fontsize=8)
ax.set_xlabel("precision = C/(C+W)  [per-query truth]")
ax.set_ylabel("C / 229  (correct answers per query)")
ax.set_title("(a) Precision–correctness frontier (strict output)")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
geo = {}
for row in rd(os.path.join(DELIV, "ledger", "clusters_geom.csv")):
    if row["config"] == "baseline":
        geo[int(row["cluster_idx"])] = row
lab2gid = {int(r["cluster_lab"]): g for g, r in geo.items()}
loc2gid = {int(r["cluster_local"]): lab2gid[int(r["cluster_lab"])]
           for r in rd(os.path.join(RUN, "members", "A_baseline.csv"))}
per = collections.defaultdict(lambda: {"C": 0, "W": 0, "U": 0})
for row in rd(os.path.join(RUN, "verdicts", "A_baseline.csv")):
    per[loc2gid[int(row["cluster_local"])]][row["verdict"]] += 1

buckets = [(-1, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 9)]
labels, rates, ns = [], [], []
for lo, hi in buckets:
    s = w = 0
    for gid, c in per.items():
        d = float(geo[gid]["d_max"])
        if lo <= d < hi:
            s += c["C"] + c["W"]
            w += c["W"]
    labels.append(f"[{lo:.2f},{hi:.2f})" if lo > 0 else f"<{hi:.2f}")
    rates.append(w / s if s else 0)
    ns.append(s)

bars = ax.bar(labels, rates, color=["#5aae61", "#a6dba0", "#fdae61", "#d7191c", "#4393c3"])
for b, n, r in zip(bars, ns, rates):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
            f"n={n}", ha="center", fontsize=8)
ax.axvline(3.5, color="#333333", ls="--", lw=1.2)
ax.text(3.55, max(rates) * 0.9, "veto 0.30\ndeletes this side", fontsize=8, color="#333333")
ax.set_ylabel("error rate among accepted queries")
ax.set_xlabel("d_max bucket")
ax.set_title("(b) Error rate by cluster d_max")
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
op = os.path.join(RUN, "fig_d4_truth.png")
plt.savefig(op, bbox_inches="tight")
print(f"[out] {op}")
print("分档:", list(zip(labels, ns, [round(r, 3) for r in rates])))
