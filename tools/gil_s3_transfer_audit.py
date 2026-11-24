# -*- coding: utf-8 -*-
"""S3 链 → gperson 传导缺口诊断（只读，不改库）。

背景：S3（gil_frag_link.py）把同机位碎片接续精化为 268 条 stitch_kin 边 + frag 重建，
      但 gperson（S2 产物）仍消费旧口径 → EVAL-1 发现「101 条已采纳链未传导」。
本脚本对账：268 条链的两端 obs 在当前 gperson 中是否同簇，输出缺口清单。

用法：python tools/gil_s3_transfer_audit.py
产物：analysis/gil/s3_transfer_audit.json
"""
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
OUT = os.path.join(ANA, "gil", "s3_transfer_audit.json")


def main():
    st = GILStore(ROOT + "/output_crosscam/analysis/gil/store_v1", create=False)

    # obs → gperson 集合
    mem = defaultdict(set)
    for r in st.con.execute("SELECT node_id, gperson_id FROM membership WHERE node_kind='obs'"):
        mem[int(r[0])].add(int(r[1]))

    # obs id → (cam, tid) 反查
    id2key = {}
    for r in st.con.execute("SELECT id, cam, tid FROM obs WHERE tombstone=0"):
        id2key[int(r[0])] = (r[1], int(r[2]))

    d = json.load(open(os.path.join(ANA, "gil", "frag_link_v1.json"), encoding="utf-8"))
    links = d["links"]

    rows = []
    stats = defaultdict(int)
    for L in links:
        cam = L["cam"]
        ta = int(L["a"].split("|")[1])
        tb = int(L["b"].split("|")[1])
        ia = st.obs_id_of(cam, ta)
        ib = st.obs_id_of(cam, tb)
        rec = dict(L)
        rec["obs_a"], rec["obs_b"] = ia, ib
        if ia is None or ib is None:
            stats["missing_obs"] += 1
            rec["status"] = "missing_obs"
        else:
            ga, gb = mem.get(ia, set()), mem.get(ib, set())
            rec["gid_a"], rec["gid_b"] = sorted(ga), sorted(gb)
            if not ga or not gb:
                stats["no_gperson"] += 1
                rec["status"] = "no_gperson"
            elif ga & gb:
                stats["transferred"] += 1
                rec["status"] = "same_gperson"
            else:
                stats["not_transferred"] += 1
                rec["status"] = "cross_gperson"
        rows.append(rec)

    # 未传导链的分组：按 (gid_a, gid_b) 对统计；并做"缺口连通分量"分析
    cross = [r for r in rows if r["status"] == "cross_gperson"]
    pair_count = defaultdict(int)
    for r in cross:
        ga = tuple(r["gid_a"]) if r["gid_a"] else ()
        gb = tuple(r["gid_b"]) if r["gid_b"] else ()
        pair_count[(ga, gb)] += 1

    # 缺口图：把 gperson 视为节点，未传导链为边 → 连通分量 = 潜在合并组
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    gset = set()
    for r in cross:
        for g in r["gid_a"] + r["gid_b"]:
            gset.add(g)
    for r in cross:
        for g in r["gid_a"]:
            for h in r["gid_b"]:
                union(g, h)

    comp = defaultdict(list)
    for g in gset:
        comp[find(g)].append(g)
    comps = sorted([sorted(v) for v in comp.values()], key=len, reverse=True)
    n_multi = sum(1 for c in comps if len(c) > 1)

    # 传导后大小估算：每个连通分量成员 gperson 的规模合计
    sizes = defaultdict(int)
    for oid, gs in mem.items():
        for g in gs:
            sizes[g] += 1

    comp_info = []
    for c in comps:
        n_members = sum(sizes.get(g, 0) for g in c)
        n_links = sum(1 for r in cross
                      if set(r["gid_a"]) & set(c) or set(r["gid_b"]) & set(c))
        comp_info.append(dict(gids=c, n_gperson=len(c), n_obs=sum(sizes.get(g, 0) for g in c),
                              n_links_involved=n_links))

    out = dict(
        n_links=len(links),
        stats=dict(stats),
        n_cross_pairs=len(cross),
        n_gap_components=len(comps),
        n_multi_gid_components=n_multi,
        components=comp_info[:80],
        cross_links=cross,
    )
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("== S3→gperson 传导对账 ==")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"未传导链 {len(cross)} 条，涉及 gperson 节点 {len(gset)} 个，"
          f"缺口连通分量 {len(comps)} 个（多gid {n_multi}）")
    print("Top 缺口分量（按 obs 数）:")
    for c in sorted(comp_info, key=lambda x: -x["n_obs"])[:12]:
        print(f"  gids={c['gids']} obs={c['n_obs']} links={c['n_links_involved']}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
