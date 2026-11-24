"""可回答上限：查询的真值人物是否在图库中出现过。"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
GT = os.path.join(DELIV, "gt")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


g = {}
for r in rd(os.path.join(GT, "gt_gallery_members_FILLED.csv")):
    pid = r["person_id"].strip()
    if pid and pid.upper() != "UNK":
        g.setdefault(pid, set()).add(r["track_id"])
gal_pids = set(g)

q = [r for r in rd(os.path.join(GT, "gt_query_gt_FILLED.csv")) if r["evaluable"] == "1"]
have = [r for r in q if r["person_id"].strip()]
ans = [r for r in have if r["person_id"].strip() in gal_pids]
miss = [r for r in have if r["person_id"].strip() not in gal_pids]

print(f"可评估 {len(q)} 条｜有人工身份 {len(have)}｜未标 {len(q)-len(have)}")
print(f"真值人物在图库中出现过: {len(ans)}")
print(f"图库外新人（结构上必错）: {len(miss)} → "
      f"{collections.Counter(r['person_id'].strip() for r in miss).most_common()}")
print(f"\n可回答上限 = {len(ans)}/229 = {len(ans)/229:.3f}")
print(f"baseline 正确 112 → 占可回答的 {112/len(ans):.3f}，占全部 229 的 {112/229:.3f}")

print("\n各身份在图库中的轨迹数（前 12）:")
for pid in sorted(g, key=lambda p: -len(g[p]))[:12]:
    print(f"  {pid:<9} {len(g[pid]):>3} 条")
