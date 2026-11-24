# -*- coding: utf-8 -*-
"""S3 传导质量综合审计（只读）：entity 一致性 / val 同机位对 / relay 边变化。"""
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
VERSION_FILES = {"base": "nodes_base.json", "cap12": "nodes_cap12.json", "v2": "gperson_v2.json"}
VERSIONS = list(VERSION_FILES)

st = GILStore(ROOT + "/output_crosscam/analysis/gil/store_v1", create=False)
obs_key, obs_eid = {}, {}
for r in st.con.execute("SELECT id,cam,tid,entity_id FROM obs WHERE tombstone=0"):
    obs_key[int(r["id"])] = f"{r['cam']}|{int(r['tid'])}"
    obs_eid[int(r["id"])] = r["entity_id"]
eid2name = {}
for r in st.con.execute("SELECT id,cam,eid FROM entity"):
    eid2name[int(r["id"])] = (r["cam"], r["eid"])

vers = {}
for v in VERSIONS:
    vers[v] = json.load(open(os.path.join(ANA, "gil", VERSION_FILES[v]), encoding="utf-8"))["node2gid"]

# ---- 1) entity 一致性：一个 entity 的 obs 落在多少个簇 ----
print("== 1) entity 一致性（一个 entity 的 obs 跨簇数）==")
ent_obs = defaultdict(list)
for oid, eid in obs_eid.items():
    if eid is not None:
        ent_obs[eid].append(oid)
for v in VERSIONS:
    g = vers[v]
    n_split = 0
    n_multi = 0
    split_list = []
    for eid, obss in ent_obs.items():
        votes = [g[obs_key[o]] for o in obss if obs_key[o] in g]
        ncl = len(set(votes))
        if len(obss) > 1:
            n_multi += 1
        if ncl > 1:
            n_split += 1
            if len(obss) >= 4:
                split_list.append((eid2name.get(eid), len(obss), ncl))
    print(f"  [{v}] 多obs entity={n_multi} 被拆分={n_split} 拆分率={n_split/max(n_multi,1):.3f}")
    if v != "base" and split_list:
        for x in sorted(split_list, key=lambda z: -z[1])[:8]:
            print(f"      {x[0]} obs={x[1]} 跨簇={x[2]}")

# ---- 2) val 同机位对同簇率 ----
print("\n== 2) splits_v5 val 同机位对（S3 工作点来源；同簇率对照）==")
sp = json.load(open(os.path.join(ANA, "splits_v5.json"), encoding="utf-8"))["val"]
key2 = {(r[0], r[1]) for r in [("x", "y")]}  # placeholder
pos, neg = sp["pos_pairs"], sp["neg_pairs"]
key_of = {}
for oid, k in obs_key.items():
    c, t = k.split("|")
    key_of[(c, int(t))] = oid


def rate(g):
    tp = fp = 0
    npos = nneg = 0
    for c, i, j in pos:
        oi, oj = key_of.get((c, int(i))), key_of.get((c, int(j)))
        if oi is None or oj is None:
            continue
        va, vb = g.get(obs_key[oi]), g.get(obs_key[oj])
        if va is None or vb is None:      # ★ 节点被排除（如倒影）→ 不可评
            continue
        npos += 1
        if va == vb:
            tp += 1
    for c, i, j in neg:
        oi, oj = key_of.get((c, int(i))), key_of.get((c, int(j)))
        if oi is None or oj is None:
            continue
        va, vb = g.get(obs_key[oi]), g.get(obs_key[oj])
        if va is None or vb is None:
            continue
        nneg += 1
        if va == vb:
            fp += 1
    P = tp / (tp + fp) if tp + fp else None
    R = tp / npos if npos else None
    F = (2 * P * R / (P + R)) if P and R else None
    return dict(tp=tp, fp=fp, n_pos=npos, n_neg=nneg, P=P, R=R, F1=F)


for v in VERSIONS:
    r = rate(vers[v])
    print(f"  [{v}] pos同簇 {r['tp']}/{r['n_pos']}  neg同簇 {r['fp']}/{r['n_neg']} "
          f"→ P={r['P']:.3f} R={r['R']:.3f} F1={r['F1']:.3f}" if r["P"] and r["R"]
          else f"  [{v}] {r}")

# ---- 3) relay 196 边同簇变化 ----
print("\n== 3) relay v2 边同簇变化 ==")
edges = json.load(open(os.path.join(ANA, "relay_v2", "edges_v2_gated.json"), encoding="utf-8"))["edges"]


def ent_oid(c, e_):
    r = st.con.execute("SELECT id FROM entity WHERE cam=? AND eid=?", (c, e_)).fetchone()
    if not r:
        return []
    return ent_obs.get(int(r[0]), [])


def edge_state(g, e):
    c1, e1 = e["a"].split(":"); c2, e2 = e["b"].split(":")
    o1, o2 = ent_oid(c1, e1), ent_oid(c2, e2)
    if not o1 or not o2:
        return None
    l1 = Counter(g.get(obs_key[o]) for o in o1 if obs_key[o] in g).most_common(1)
    l2 = Counter(g.get(obs_key[o]) for o in o2 if obs_key[o] in g).most_common(1)
    if not l1 or not l2:
        return None
    return int(l1[0][0] == l2[0][0])


chg = []
n_na = 0
for e in edges:
    if e["status"] not in ("active", "audit", "gate"):
        continue
    sb = edge_state(vers["base"], e)
    s12 = edge_state(vers["cap12"], e)
    sc = edge_state(vers["v2"], e)
    if sb is None or sc is None:
        n_na += 1
        continue
    if sb != sc:
        chg.append((e["a"], e["b"], e["status"], sb, s12, sc))
print(f"  base→v2 变化边 {len(chg)} 条（同簇→跨簇 或 反向；不可评 {n_na} 条）")
for a, b, s, x, y, z in chg[:25]:
    print(f"    {a:<14} {b:<14} {s:<7} base={x} cap12={y} v2={z}")
