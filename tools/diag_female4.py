"""定位错判集中度与 female4 簇的构成。"""
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

v = rd(os.path.join(RUN, "verdicts", "A_baseline.csv"))
d3 = {row["track_id"]: row for row in rd(os.path.join(DELIV, "gt", "gt_gallery_members_FILLED.csv"))}

cnt = collections.Counter()
for row in v:
    if row["verdict"] == "W":
        cnt[loc2gid[int(row["cluster_local"])]] += 1
print("错判按簇:", cnt.most_common())

gid = cnt.most_common(1)[0][0]
memids = set()
for row in rd(os.path.join(RUN, "members", "A_baseline.csv")):
    if loc2gid[int(row["cluster_local"])] == gid:
        memids.add(row["track_id"])

print(f"\n=== 簇 {gid} 全部成员（D3 标注）===")
for tid, row in sorted(d3.items()):
    if tid in memids:
        print(f"  {tid:<20} pid={row['person_id']:<9} box={row['box_type']:<8} "
              f"n={row['n_frames']:>4} switch={(row['id_switch'].strip() or '-'):<3} note={row['note']}")
print(f"\n该簇 d_max={geo[gid]['d_max']}  n={geo[gid]['n']}  cams={geo[gid]['cams']}")
print(f"簇内未在 D3 出现的成员: {len(memids) - sum(1 for t in memids if t in d3)}")

print("\n=== 5 条 female4→male7 错判 ===")
for row in v:
    if row["verdict"] == "W" and row["c_pid"] == "female4":
        print(f"  {row['query_id']:<18} 真值={row['q_pid']:<8} 簇主={row['c_pid']:<8} cam={row['cam']}")
print("\n若该簇 female4 实为 male7：C 112→117, W 12→7, 精度 0.903→%.3f" % (117 / 124))
