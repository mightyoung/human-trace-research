# -*- coding: utf-8 -*-
"""对三对波动 entity 做成员级追踪：哪些 obs 的簇归属变了、被哪条链拉走。"""
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
from gil import GILStore  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
VERSIONS = ["base", "cap8", "cap10", "cap12", "combo"]

st = GILStore(ROOT + "/output_crosscam/analysis/gil/store_v1", create=False)

obs_info = {}
for r in st.con.execute("SELECT id,cam,tid,t_first,t_last,entity_id FROM obs WHERE tombstone=0"):
    obs_info[int(r["id"])] = dict(cam=r["cam"], tid=int(r["tid"]),
                                  t0=r["t_first"], t1=r["t_last"], eid=r["entity_id"])
eid2name = {}
for r in st.con.execute("SELECT id,cam,eid FROM entity"):
    eid2name[int(r["id"])] = f"{r['cam']}_E_{r['eid']}"

# node2gid
vers = {}
for v in VERSIONS:
    vers[v] = json.load(open(os.path.join(ANA, "gil", f"nodes_{v}.json"), encoding="utf-8"))["node2gid"]

# S3 链的 key→链 列表（查"谁连接了这些 obs"）
fl = json.load(open(os.path.join(ANA, "gil", "frag_link_v1.json"), encoding="utf-8"))["links"]
link_of = defaultdict(list)
for lk in fl:
    link_of[lk["a"]].append((lk["b"], lk["dt"], lk["cos"], lk.get("long_gap"), lk.get("same_entity")))
    link_of[lk["b"]].append((lk["a"], lk["dt"], lk["cos"], lk.get("long_gap"), lk.get("same_entity")))

targets = ["343_E_343-E009", "316_E_316-E025", "343_E_343-E010", "316_E_316-E014",
           "343_E_343-E042", "316_E_316-E046"]
# entity key → obs
want = {}
for oid, inf in obs_info.items():
    k = eid2name.get(inf["eid"])
    if k in targets:
        want.setdefault(k, []).append(oid)

for t in targets:
    print(f"\n===== {t} =====")
    for oid in sorted(want.get(t, []), key=lambda o: obs_info[o]["t0"] or 0):
        inf = obs_info[oid]
        key = f"{inf['cam']}|{inf['tid']}"
        gs = " ".join(f"{v}={vers[v].get(key,'-')}" for v in VERSIONS)
        lks = link_of.get(key, [])
        lk_s = "; ".join(f"→{b}(dt={dt},cos={c}{',LG' if lg else ''}{',SE' if se else ''})"
                         for b, dt, c, lg, se in lks[:6])
        print(f"  {key:<10} t=[{inf['t0']:.1f},{inf['t1']:.1f}]  {gs}")
        if lk_s:
            print(f"            链: {lk_s}")
