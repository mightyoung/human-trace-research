"""box_type 口径的精度（用修正后的真值）。"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
RUN = os.path.join(DELIV, "results")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


d4 = {r["query_id"]: r for r in rd(os.path.join(DELIV, "gt", "gt_query_gt_FILLED.csv"))}
v = rd(os.path.join(RUN, "verdicts", "A_baseline.csv"))
t = collections.defaultdict(lambda: collections.Counter())
for row in v:
    t[d4[row["query_id"]]["box_type"]][row["verdict"]] += 1
print(f"{'类型':<9}{'接受':>5}{'C':>4}{'W':>4}{'U':>4}{'精度':>7}")
for k in sorted(t):
    c = t[k]
    n = c["C"] + c["W"]
    print(f"{k:<9}{sum(c.values()):>5}{c['C']:>4}{c['W']:>4}{c['U']:>4}{(c['C']/n if n else 0):>7.3f}")
print()
print(f"图库身份集合: {len({r['person_id'].strip() for r in rd(os.path.join(DELIV, 'gt', 'gt_gallery_members_FILLED.csv')) if r['person_id'].strip()})}")
print("female4 是否还在:", any(r["person_id"].strip() == "female4"
                            for r in rd(os.path.join(DELIV, "gt", "gt_gallery_members_FILLED.csv"))))
