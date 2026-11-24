# -*- coding: utf-8 -*-
"""深入分析 S3 传导缺口：Top1 分量构成 + entity 拆分检查（只读）。"""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
from gil import GILStore  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")

st = GILStore(ROOT + "/output_crosscam/analysis/gil/store_v1", create=False)

# obs → gperson（当前 store）
obs2g = {}
for r in st.con.execute("SELECT node_id, gperson_id FROM membership WHERE node_kind='obs'"):
    obs2g[int(r[0])] = int(r[1])

# obs → entity（obs 表 entity_id 列）
obs2e = {}
for r in st.con.execute("SELECT id, entity_id FROM obs WHERE tombstone=0"):
    if r[1] is not None:
        obs2e[int(r[0])] = int(r[1])

# obs → cam/tid
id2key = {}
for r in st.con.execute("SELECT id, cam, tid FROM obs WHERE tombstone=0"):
    id2key[int(r[0])] = (r[1], int(r[2]))

d = json.load(open(os.path.join(ANA, "gil", "s3_transfer_audit.json"), encoding="utf-8"))
top1 = d["components"][0]["gids"]
top1_set = set(top1)
print(f"Top1 分量: {len(top1)} 个 gperson")

# Top1 分量的 obs 构成
obs_in = [o for o, g in obs2g.items() if g in top1_set]
cams_top1 = Counter(id2key[o][0] for o in obs_in if o in id2key)
print("Top1 分量 obs 机位数:", dict(cams_top1), "总计", len(obs_in))

# Top1 分量中的链的构成
cross = d["cross_links"]
top1_links = [r for r in cross if set(r["gid_a"]) & top1_set or set(r["gid_b"]) & top1_set]
print("Top1 分量涉及链:", len(top1_links))
print("链机位:", Counter(r["cam"] for r in top1_links))
print("链 long_gap:", Counter(r.get("long_gap") for r in top1_links))

# Top1 分量的时间跨度与机位混合：看每个 gperson 的机位
gid2cams = defaultdict(set)
gid2t = {}
for o in obs_in:
    if o in id2key:
        gid2cams[obs2g[o]].add(id2key[o][0])
for g, cams in sorted(gid2cams.items())[:100]:
    pass
multi_cam = {g: sorted(c) for g, c in gid2cams.items() if len(c) > 1}
print("Top1 内多机位 gperson:", len(multi_cam), "例:", dict(list(multi_cam.items())[:8]))

# ★ entity 拆分检查：全库范围，多少 entity 的 obs 被分到 >1 个 gperson
ent2g = defaultdict(set)
ent_obs = defaultdict(list)
for o, e in obs2e.items():
    if o in obs2g:
        ent2g[e].add(obs2g[o])
        ent_obs[e].append(o)
split_ents = {e: sorted(g) for e, g in ent2g.items() if len(g) > 1}
print()
print(f"★ entity 被 gperson 拆分的个数: {len(split_ents)} / {len(ent2g)}")
split_sizes = Counter(len(ent_obs[e]) for e in split_ents)
print("  拆分 entity 的 obs 数分布:", dict(sorted(split_sizes.items())))
# 其中哪些 entity 的成员还在同一 gperson（对齐的）
print("  例:", dict(list(split_ents.items())[:10]))
