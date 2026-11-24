"""核对报告 §1.1：旧簇判定 × 新真值判定的交叉表。"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
RUN = os.path.join(DELIV, "results")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


geo = {}
for row in rd(os.path.join(DELIV, "ledger", "clusters_geom.csv")):
    if row["config"] == "baseline":
        geo[int(row["cluster_idx"])] = row
lab2gid = {}
for gid, row in geo.items():
    lab2gid[int(row["cluster_lab"])] = gid
loc2gid = {}
for row in rd(os.path.join(RUN, "members", "A_baseline.csv")):
    loc2gid[int(row["cluster_local"])] = lab2gid[int(row["cluster_lab"])]

tab = collections.defaultdict(lambda: collections.Counter())
for row in rd(os.path.join(RUN, "verdicts", "A_baseline.csv")):
    gid = loc2gid[int(row["cluster_local"])]
    tab[geo[gid]["purity_verdict"] or "(未审)"][row["verdict"]] += 1

print(f"{'旧簇判定':<10}{'查询':>5}{'C':>4}{'W':>4}{'U':>4}{'实际精度':>9}")
for k in sorted(tab):
    c = tab[k]
    n = c["C"] + c["W"]
    print(f"{k:<10}{sum(c.values()):>5}{c['C']:>4}{c['W']:>4}{c['U']:>4}"
          f"{(c['C']/n if n else float('nan')):>9.3f}")

# 旧方法会算成"错"的总数
old_wrong = tab["bad"]["C"] + tab["bad"]["W"] + tab["unk"]["C"] + tab["unk"]["W"]
print(f"\n旧方法（bad/unk 一律算错）：错 {old_wrong}，对 {tab['ok']['C']} → 精度 {tab['ok']['C']/138:.3f}")
print(f"新方法（逐条真值）：错 {sum(t['W'] for t in tab.values())}，对 {sum(t['C'] for t in tab.values())}")
