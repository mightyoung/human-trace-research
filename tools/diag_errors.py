"""列出 baseline 的全部错判逐条，并核对 veto 删簇的账。"""
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
lab2gid = {int(r["cluster_lab"]): g for g, r in geo.items()}
loc2gid = {int(r["cluster_local"]): lab2gid[int(r["cluster_lab"])]
           for r in rd(os.path.join(RUN, "members", "A_baseline.csv"))}

d4 = {r["query_id"]: r for r in rd(os.path.join(DELIV, "gt", "gt_query_gt_FILLED.csv"))}
v = rd(os.path.join(RUN, "verdicts", "A_baseline.csv"))

q = {r["query_id"]: r for r in rd(os.path.join(RUN, "queries", "A_baseline.csv"))}
print(f"{'查询':<18}{'真值':<9}{'簇主':<9}{'已标/身份':<12}{'簇':>5}{'d_max':>7}  {'box':<8}{'minScore':>8}")
for row in v:
    if row["verdict"] != "W":
        continue
    gid = loc2gid[int(row["cluster_local"])]
    score = min(float(q[row["query_id"]]["score_a"]), float(q[row["query_id"]]["score_b"]))
    print(f"{row['query_id']:<18}{row['q_pid']:<9}{row['c_pid']:<9}"
          f"{row['cluster_n_lab'] + '/' + row['cluster_n_ids']:<12}{gid:>5}"
          f"{float(geo[gid]['d_max']):>7.3f}  {d4[row['query_id']]['box_type']:<8}{score:>8.2f}")

# veto 账
mb = {int(r["cluster_local"]): int(r["cluster_lab"]) for r in rd(os.path.join(RUN, "members", "A_baseline.csv"))}
mv = {int(r["cluster_local"]): int(r["cluster_lab"]) for r in rd(os.path.join(RUN, "members", "C_dmax0.30.csv"))}
gone = set(mb.values()) - set(mv.values())
lab2loc = {v2: k for k, v2 in mb.items()}
locs = {lab2loc[g] for g in gone if g in lab2loc}
stat = collections.defaultdict(lambda: collections.Counter())
for row in v:
    cl = int(row["cluster_local"])
    if cl in locs:
        stat[cl][row["verdict"]] += 1
tot = collections.Counter()
for c in stat:
    tot += stat[c]
print(f"\nveto 0.30 删 {len(gone)} 簇；其中有查询的 {len(stat)} 个，服务 {sum(tot.values())} 条，"
      f"判对 {tot['C']}、判错 {tot['W']}")
for cl in sorted(stat, key=lambda c: -stat[c]["C"]):
    row = next(r for r in v if int(r["cluster_local"]) == cl)
    print(f"  簇{loc2gid[cl]:>5} 服务{sum(stat[cl].values()):>3} 对{stat[cl]['C']:>3} 错{stat[cl]['W']:>2} "
          f"主体={row['c_pid']}")
