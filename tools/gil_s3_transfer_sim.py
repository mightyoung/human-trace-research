# -*- coding: utf-8 -*-
"""S3 链 → gperson 传导模拟（只读，不落库；多子集对照）。

对每个链子集：
  1) gperson 级并查集合并（链连接两端 obs 的 gperson）
  2) 检查合并簇内时间互斥违规（318×320 豁免）
  3) 对账人工真值 P/R、318×320 GT
输出：analysis/gil/s3_transfer_sim.json
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
from gil import GILStore  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
OVERLAP_PAIRS = {frozenset(("318", "320"))}
GT_318_320 = [(1, 1), (14, 3), (37, 11), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28)]


def union_find():
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union, parent


def main():
    st = GILStore(ROOT + "/output_crosscam/analysis/gil/store_v1", create=False)

    # ---- 库数据 ----
    rows = list(st.con.execute("SELECT id,cam,tid,t_first,t_last,flags FROM obs WHERE tombstone=0 ORDER BY id"))
    oid = [int(r["id"]) for r in rows]
    cam = {int(r["id"]): r["cam"] for r in rows}
    t0 = {int(r["id"]): r["t_first"] for r in rows}
    t1 = {int(r["id"]): r["t_last"] for r in rows}
    id2g = {}
    for r in st.con.execute("SELECT node_id, gperson_id FROM membership WHERE node_kind='obs'"):
        id2g[int(r[0])] = int(r[1])
    gids = sorted(set(id2g.values()))
    n_g = len(gids)
    print(f"obs={len(rows)}  gperson={n_g}")

    gsize = Counter(id2g.values())

    # entity → obs 成员（用于真值对账）
    ent_members = defaultdict(list)
    for r in st.con.execute("SELECT id,entity_id FROM obs WHERE entity_id IS NOT NULL"):
        ent_members[int(r["entity_id"])].append(int(r["id"]))

    def ent_obs(c, e_):
        r = st.con.execute("SELECT id FROM entity WHERE cam=? AND eid=?", (c, e_)).fetchone()
        return ent_members.get(int(r[0]), []) if r else []

    # 人工真值
    tr = {}
    for src in (os.path.join(ROOT, "webapp/backend/data/truth/seed_labels.json"),
                os.path.join(ANA, "truth/remote/labels.json")):
        if os.path.exists(src):
            d = json.load(open(src, encoding="utf-8")) or {}
            tr.update(d.get("pair", d) if isinstance(d, dict) else {})

    def eval_pairs(gid_of_obs):
        """gid_of_obs: obs_id -> 簇号（任意可哈希）。返回 P/R。"""
        def c_of(c, e_):
            ms = ent_obs(c, e_)
            if not ms:
                return None
            vals = [gid_of_obs[o] for o in ms if o in gid_of_obs]
            return Counter(vals).most_common(1)[0][0] if vals else None
        tp = fp = fn = unk = 0
        for pid, v in tr.items():
            if not isinstance(v, dict) or v.get("v") not in (0, 1) or "__" not in pid:
                continue
            # 命名空间防御：S6B_P_/PRT_ 前缀剥离后计入；非法格式键跳过
            pid = pid[6:] if pid.startswith("S6B_P_") else (
                pid[4:] if pid.startswith("PRT_") else pid)
            if pid.count("__") != 1 or pid.count("_E_") != 2:
                continue
            pa, pb = pid.split("__")
            ca, ea = pa.split("_E_"); cb, eb = pb.split("_E_")
            ga, gb = c_of(ca, ea), c_of(cb, eb)
            if ga is None or gb is None:
                unk += 1
                continue
            pred = int(ga == gb)
            if v["v"] == 1:
                tp += pred; fn += 1 - pred
            else:
                fp += pred
        P = tp / (tp + fp) if tp + fp else None
        R = tp / (tp + fn) if tp + fn else None
        return dict(tp=tp, fp=fp, fn=fn, unk=unk, P=P, R=R)

    # ---- 链子集 ----
    fl = json.load(open(os.path.join(ANA, "gil", "frag_link_v1.json"), encoding="utf-8"))["links"]

    def key2oid(k):
        c, t = k.split("|")
        return st.obs_id_of(c, int(t))

    subsets = {
        "all268": [l for l in fl],
        "no_longgap": [l for l in fl if not l.get("long_gap")],
        "no_longgap_no_sameent": [l for l in fl if not l.get("long_gap") and not l.get("same_entity")],
        "strict": [l for l in fl if (not l.get("long_gap")) and (not l.get("same_entity"))
                   and (l.get("cos") or 0) >= 0.8 and (l.get("pe_adj") is not None and l["pe_adj"] <= 0.08)],
    }

    results = {}
    for name, links in subsets.items():
        find, union, parent = union_find()
        # 兜底：先给每个 gperson 建自环
        for g in gids:
            union(("g", g), ("g", g))
        n_link_union = n_missing = 0
        for l in links:
            oa, ob = key2oid(l["a"]), key2oid(l["b"])
            ga, gb = id2g.get(oa), id2g.get(ob)
            if ga is None or gb is None:
                n_missing += 1
                continue
            if find(("g", ga)) != find(("g", gb)):
                union(("g", ga), ("g", gb))
                n_link_union += 1
        # 簇：成员列表
        comp = defaultdict(list)
        for g in gids:
            comp[find(("g", g))].append(g)
        comps = list(comp.values())
        n_cl = len(comps)

        # 违规检查 + 大小（全信任合并的上界口径：合并后簇内若存在时间互斥对 → 该合并不可行）
        viol = 0
        viol_detail = []
        obs_of_g = defaultdict(list)
        for o, gg in id2g.items():
            obs_of_g[gg].append(o)
        for cs in comps:
            if len(cs) < 2:
                continue
            obss = [o for g in cs for o in obs_of_g.get(g, [])]
            vc = 0
            for i in range(len(obss)):
                for j in range(i + 1, len(obss)):
                    ai, aj = obss[i], obss[j]
                    if frozenset((cam[ai], cam[aj])) in OVERLAP_PAIRS:
                        continue
                    if None in (t0[ai], t1[ai], t0[aj], t1[aj]):
                        continue
                    if t0[ai] <= t1[aj] and t0[aj] <= t1[ai]:
                        vc += 1
            if vc:
                viol += 1
                viol_detail.append((cs[:8], vc))
        # 对账
        gid_of_obs = {o: find(("g", gg)) for o, gg in id2g.items()}
        pr = eval_pairs(gid_of_obs)
        gt_hit = 0
        for x, y in GT_318_320:
            oa, ob = st.obs_id_of("318", x), st.obs_id_of("320", y)
            if oa in gid_of_obs and ob in gid_of_obs and gid_of_obs[oa] == gid_of_obs[ob]:
                gt_hit += 1
        big = sorted([len(cs) for cs in comps], reverse=True)[:8]
        results[name] = dict(n_links=len(links), n_union=n_link_union, n_missing=n_missing,
                             n_clusters=n_cl, top_sizes=big, violations=viol,
                             pr=pr, gt_hit=gt_hit)
        print(f"[{name}] links={len(links)} union={n_link_union} miss={n_missing} "
              f"→ 簇 {n_cl}（原 {n_g}） 最大{big} 违规={viol} "
              f"P={pr['P']} R={pr['R']} GT={gt_hit}/8")

    import time as _time
    results["_note"] = ("生成于 " + _time.strftime("%Y-%m-%d %H:%M") +
                        "；语义=从『现行库』gperson 态出发的 links 边际评估（S3 传导已落库）；"
                        "历史『方案模拟』数字（如 all268 P=0.625）见 docs/gil-s3-transfer-report.md")
    json.dump(results, open(os.path.join(ANA, "gil", "s3_transfer_sim.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ analysis/gil/s3_transfer_sim.json")


if __name__ == "__main__":
    main()
