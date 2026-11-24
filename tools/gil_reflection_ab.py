# -*- coding: utf-8 -*-
"""S6-b：倒影「打 flag 降权（不删框）」的同序列 A/B。

问题：S6 判定出 11 条墙镜倒影（+1 条地板倒影）**存活于生产口径 clean2**，它们以
      "真人"身份参与了下游全部层。设计选择是「**打 flag 不删框**」——
      倒影框保留用于**解释**（它必须被某真人观测解释），但**不作为独立身份证据**。
      这个选择真的改善下游吗？代价是什么？

方法（严格同序列）：
  A 臂 = 现状：倒影 obs 作为普通节点参与 gperson 聚类。
  B 臂 = 处置：倒影 obs 视为「非独立观测」排除出节点集（归属关系保留在
         reflection_attach 中，信息不丢）。
  ★ 关键纪律：两臂必须**冻结同一工作点**（T_same=0.975）。
     否则 val 校准集也会因节点减少而改变（实测 A 的 neg 435 → B 的 365，
     T_same 漂到 0.960 → 264 变 204 —— 那是阈值漂移，不是倒影的功劳）。

度量（不用"召回提升"这种会被评测集变化污染的指标）：
  ① 人数：n_gperson；
  ② 零代价：人工真值在**两臂共同可评集**上的 P/R/F1（逐对出表）；
  ③ 结构：倒影污染簇在 B 中的裂解情况（幽灵成员是否归位到真人的簇）；
  ④ relay 精度：被污染边是否落在 active 集（active 才是精度承诺的对象）。

用法：DET_VIEW=clean2 python tools/gil_reflection_ab.py
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ANA = os.path.join(ROOT, "output_crosscam", "analysis")


def load_clusters(path):
    d = json.load(open(path, encoding="utf-8"))
    n2g = d["node2gid"]
    cl = collections.defaultdict(set)
    for k, g in n2g.items():
        cl[g].add(k)
    return d, cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="output_crosscam/analysis/gil/gperson_ab_A.json")
    ap.add_argument("--b", default="output_crosscam/analysis/gil/gperson_ab_B.json")
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--relay", default="output_crosscam/analysis/relay_v2/edges_v2_gated.json")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/reflection_ab_v1.json")
    a = ap.parse_args()

    A, clA = load_clusters(ROOT + "/" + a.a)
    B, clB = load_clusters(ROOT + "/" + a.b)
    # 臂标签：A 的 gid 是 0 基索引；统一成 g0001 形式
    def lab(g):
        return f"g{g + 1:04d}"

    comm = set(A["node2gid"]) & set(B["node2gid"])
    print(f"[ab] 共同节点 {len(comm)}；A 簇 {len(clA)}（节点 {len(A['node2gid'])}）"
          f"；B 簇 {len(clB)}（节点 {len(B['node2gid'])}）")

    # ---------- ① 人数 & 不变量 ----------
    print("\n=== ① 人数 ===")
    print(f"  A（现状）            gperson {A['n_gperson']}  合并 {A['n_merge']}  违规 {A['invariant_violations']}")
    print(f"  B（倒影降权）        gperson {B['n_gperson']}  合并 {B['n_merge']}  违规 {B['invariant_violations']}")
    print(f"  → 人数 {A['n_gperson']} → {B['n_gperson']}（{B['n_gperson']-A['n_gperson']:+d}）")

    # ---------- ② 人工真值：共同可评集 ----------
    sp_lab = {}
    for src in (os.path.join(ROOT, "webapp/backend/data/truth/seed_labels.json"),
                os.path.join(ANA, "truth/remote/labels.json")):
        if os.path.exists(src):
            d = json.load(open(src, encoding="utf-8")) or {}
            sp_lab.update(d.get("pair", d) if isinstance(d, dict) else {})   # 与 gil_gperson 同序
    truth = {k: v for k, v in (sp_lab or {}).items()
             if isinstance(v, dict) and v.get("v") in (0, 1) and "__" in k}

    import sqlite3
    con = sqlite3.connect(os.path.join(ROOT, a.store, "gil.sqlite"))
    con.row_factory = sqlite3.Row
    ent_members = collections.defaultdict(list)
    for r in con.execute("SELECT id,entity_id FROM obs WHERE entity_id IS NOT NULL"):
        ent_members[r["entity_id"]].append(r["id"])
    eid2cam = {}
    for r in con.execute("SELECT id,cam,eid FROM entity"):
        eid2cam[r["id"]] = (r["cam"], r["eid"])
    id2key = {r["id"]: f"{r['cam']}|{r['tid']}" for r in con.execute(
        "SELECT id,cam,tid FROM obs WHERE tombstone=0")}

    def ent_key(cam, eid):
        for eid_, (c, e) in eid2cam.items():
            if c == cam and str(e) == str(eid):
                ms = [id2key[q] for q in ent_members.get(eid_, []) if q in id2key]
                return ms
        return []

    # key -> cluster label（两臂）
    ndA = {k: lab(g) for k, g in A["node2gid"].items()}
    ndB = {k: lab(g) for k, g in B["node2gid"].items()}

    def pred(arm, keys):
        s = {arm[k] for k in keys if k in arm}
        return (len(s) == 1) if s else None       # 全部落在唯一簇 = 判同人

    rows = []
    for pid, v in truth.items():
        pa, pb = pid.split("__")
        ca, ea = pa.split("_E_"); cb, eb = pb.split("_E_")
        ka, kb = ent_key(ca, ea), ent_key(cb, eb)
        if not ka or not kb:
            continue
        y = int(v["v"])
        pa_ = pred(ndA, ka + kb)
        pb_ = pred(ndB, ka + kb)
        rows.append(dict(pair=pid, y=y, A=(None if pa_ is None else int(pa_)),
                         B=(None if pb_ is None else int(pb_)),
                         common_eval=(pa_ is not None and pb_ is not None)))

    def prf(arm_i):
        tp = sum(1 for r in rows if r["y"] == 1 and r[arm_i] == 1)
        fp = sum(1 for r in rows if r["y"] == 0 and r[arm_i] == 1)
        fn = sum(1 for r in rows if r["y"] == 1 and r[arm_i] == 0)
        p = tp / (tp + fp) if tp + fp else None
        rr = tp / (tp + fn) if tp + fn else None
        f = (2 * p * rr / (p + rr)) if p and rr else None
        return dict(tp=tp, fp=fp, fn=fn,
                    precision=(None if p is None else round(p, 4)),
                    recall=(None if rr is None else round(rr, 4)),
                    f1=(None if f is None else round(f, 4)))

    co = [r for r in rows if r["common_eval"]]
    print("\n=== ② 人工真值（共同可评集，逐对同源）===")
    print(f"  共同可评对 {len(co)}（总标注 {len(rows)}）")
    print(f"  A（现状）      {prf('A')}")
    print(f"  B（倒影降权）  {prf('B')}")
    print("  逐对：", "  ".join(f"{r['pair']}:y{r['y']}|A{r['A']}|B{r['B']}" for r in co))
    changed = [r for r in rows if r["A"] != r["B"]]
    print(f"  判定发生变化的对：{len(changed)}")
    for r in changed:
        tag = "★由不同人转为同人（倒影曾把它拆开）" if (r["A"] == 0 and r["B"] == 1) else (
              "⚠由同人转为不同人" if (r["A"] == 1 and r["B"] == 0) else "评测集变化(不可评)")
        print(f"    {r['pair']}  y={r['y']}  A={r['A']} B={r['B']}  {tag}")

    # ---------- ③ 倒影污染簇的裂解 ----------
    print("\n=== ③ 倒影污染簇在 B 臂的归属 ===")
    refl_keys = [r["refl"] for r in A["reflection_attach"]]
    # A 中与某倒影同簇的节点集合 = 污染簇
    contaminated = collections.defaultdict(set)
    for rk in refl_keys:
        g = A["node2gid"].get(rk)
        if g is None:
            continue
        contaminated[g] |= clA[g]
    # 哪些簇消失 / 缩小
    vanish, shrink = [], []
    for g, members in contaminated.items():
        mem_B = collections.defaultdict(list)
        for k in members:
            gb = B["node2gid"].get(k)
            if gb is not None:
                mem_B[gb].append(k)
        n_refl = sum(1 for k in members if k in refl_keys)
        if not mem_B:
            vanish.append((g, members, n_refl))
        elif len(mem_B) > 1 or any(len(v) < len(members) for v in mem_B.values()):
            shrink.append((g, members, mem_B, n_refl))
    print(f"  受污染簇（A 中含倒影的簇）{len(contaminated)} 个；"
          f"B 中裂解/缩小 {len(shrink)}；完全消失 {len(vanish)}")
    struct, spurious = [], []
    for g, members, mem_B, n_refl in sorted(shrink):
        real = sorted(k for k in members if k not in refl_keys)
        # ★ 倒影导致的错误合并：真人成员在 B 中被拆到 >1 个簇
        real_cl = {B["node2gid"].get(k) for k in real} - {None}
        if len(real_cl) > 1:
            spurious.append(dict(a_cluster=lab(g), n_reflection=n_refl,
                                 real_members=real,
                                 b_clusters={lab(gb): sorted(k for k in ks if k not in refl_keys)
                                             for gb, ks in mem_B.items()
                                             if any(k not in refl_keys for k in ks)}))
        print(f"  -- A 的 {lab(g)}：{len(members)} 成员（倒影 {n_refl}，真人 {len(real)}）"
              f" → B 裂为 {len(mem_B)} 簇 --")
        for gb, ks in sorted(mem_B.items(), key=lambda kv: -len(kv[1])):
            print(f"       {lab(gb)}: {sorted(ks)}")
        struct.append(dict(a_cluster=lab(g), n_members=len(members), n_reflection=n_refl,
                           b_clusters={lab(gb): sorted(ks) for gb, ks in mem_B.items()}))
    print(f"  ★ 倒影导致的**错误合并**（真人被倒影桥接连成一簇，B 中拆开）：{len(spurious)} 个簇")
    for s in spurious:
        print(f"    A 的 {s['a_cluster']}（倒影 {s['n_reflection']}）真人 {s['real_members']}"
              f" → B: {s['b_clusters']}")

    # ---------- ④ 倒影归属复原：倒影 → 源的簇 ----------
    print("\n=== ④ 倒影归属复原（倒影由其「源真人」的簇解释）===")
    attach = []
    for r in A["reflection_attach"]:
        rk, sk = r["refl"], r["source"]
        g_src = ndB.get(sk) or ndA.get(sk)
        g_refl_A = ndA.get(rk)
        print(f"  {rk} ← 源 {sk}：源在 B 属于 {g_src}；倒影在 A 曾被放进 {g_refl_A}"
              f"  {'✅ 归位' if g_src and g_refl_A and g_src != g_refl_A else '（同簇/不可评）'}")
        attach.append(dict(**r, src_cluster_b=g_src, refl_cluster_a=g_refl_A,
                           relocated=bool(g_src and g_refl_A and g_src != g_refl_A)))

    # ---------- ⑤ relay 边：端点错位（倒影实体）→ 重定向到源 + 与身份层对账 ----------
    # 语义：倒影不是独立的人，而是「某真人观测的镜像副本」→ 指向倒影实体的边
    #       **不是必然为假**，而是**端点错位**：等价于「指向其源实体的边」。
    #       是否正确，要看重定向后**两端是否被身份层判为同一人**。
    rl = json.load(open(ROOT + "/" + a.relay, encoding="utf-8"))["edges"]
    refl_set = set(refl_keys)
    key2ent = {}
    for r in con.execute("SELECT id,cam,tid,entity_id FROM obs WHERE tombstone=0"):
        key2ent[f"{r['cam']}|{r['tid']}"] = r["entity_id"]
    eid2id = {r["eid"]: r["id"] for r in con.execute("SELECT id,cam,eid FROM entity")}
    eid2members = {}
    for e_, (c, eid) in eid2cam.items():
        eid2members[eid] = sorted(ent_key(c, eid))
    # 实体 → B 臂簇（成员一致才算确定）
    ent2gB = {}
    for eid, ms in eid2members.items():
        gs = {ndB[k] for k in ms if k in ndB}
        ent2gB[eid] = (gs.pop() if len(gs) == 1 else (None if len(gs) > 1 else None))
    src_of = dict((x["refl"], x["source"]) for x in A["reflection_attach"])

    touched = []
    ent_by_id = {r["id"]: r["eid"] for r in con.execute("SELECT id,cam,eid FROM entity")}
    for e in rl:
        info = {}
        for side in ("a", "b"):
            c, eid = e[side].split(":")
            ms = eid2members.get(eid, [])
            hit = [k for k in ms if k in refl_set]
            if hit:
                info[side] = dict(endpoint=eid, members=ms, hit=hit,
                                  pure=bool(ms) and len(hit) == len(ms))
        if not info:
            continue
        # 倒影端 → 源 obs → 源所在实体 → 该实体在 B 臂的簇
        hit_all = sorted({k for v in info.values() for k in v["hit"]})
        src_keys = [src_of[k] for k in hit_all]
        src_eids = {key2ent[k] for k in src_keys} - {None}
        g_srcs = {ent2gB.get(ent_by_id[x]) for x in src_eids} - {None}
        other = "b" if "a" in info else "a"
        o_cam, o_eid = e[other].split(":")
        g_neigh = ent2gB.get(o_eid)
        verdict = ("corroborated" if (g_neigh and g_srcs and g_neigh in g_srcs)
                   else ("contradicted" if (g_neigh and g_srcs) else "unresolved"))
        touched.append(dict(
            edge=f"{e['a']}->{e['b']}", status=e["status"], score=e.get("score"),
            mirror_sides={s: dict(endpoint=v["endpoint"], members=v["members"],
                                  hit=v["hit"], pure_mirror=v["pure"]) for s, v in info.items()},
            redirect={s: f"{v['endpoint']}→源实体({sorted(src_eids)})" for s, v in info.items()},
            neighbor_endpoint=f"{o_cam}:{o_eid}", g_neighbor=g_neigh,
            g_source=sorted(g_srcs), verdict=verdict))
    st = collections.Counter(t["status"] for t in touched)
    stv = collections.Counter(t["verdict"] for t in touched)
    print("\n=== ⑤ relay 边：端点错位（倒影实体）→ 重定向 + 身份层对账 ===")
    print(f"  端点含倒影实体的 relay 边：{len(touched)}  状态 {dict(st)}")
    print(f"  ★ 重定向到源实体后与 B 臂身份层对账：{dict(stv)}")
    print("    corroborated = 源与邻居被判同一人（边被证实）；"
          "contradicted = 源与邻居被判不同人（边与身份层冲突）")
    for t in sorted(touched, key=lambda x: (x["verdict"] != "contradicted", x["status"] != "active")):
        print(f"    [{t['status']:6s} {t['score']}] {t['edge']:30s} "
              f"邻居 {t['neighbor_endpoint']}({t['g_neighbor']}) vs 源 {t['g_source']} → {t['verdict']}")

    # ---------- ④b 倒影实体（身份级幽灵）：倒影与源不同实体 = 纯倒影身份 ----------
    key2oid = {r["id"]: f"{r['cam']}|{r['tid']}" for r in con.execute(
        "SELECT id,cam,tid FROM obs WHERE tombstone=0")}
    ent_mem = collections.defaultdict(list)
    for r in con.execute("SELECT id,cam,tid,entity_id FROM obs WHERE tombstone=0"):
        ent_mem[r["entity_id"]].append(f"{r['cam']}|{r['tid']}")
    ent_eid = {r["id"]: f"{r['cam']}-{r['eid']}" for r in con.execute("SELECT id,cam,eid FROM entity")}
    refl_ent = {}
    for rk in refl_keys:
        cam, tid = rk.split("|")
        row = con.execute("SELECT entity_id FROM obs WHERE cam=? AND tid=? AND tombstone=0",
                          (cam, int(tid))).fetchone()
        if not row:
            continue
        e = row[0]
        src = dict((x["refl"], x["source"]) for x in A["reflection_attach"])[rk]
        has_src = src in ent_mem[e]
        refl_ent.setdefault(e, dict(entity=ent_eid.get(e, str(e)), members=sorted(ent_mem[e]),
                                    n_reflection=0, colocated_source=False))
        refl_ent[e]["n_reflection"] += 1
        refl_ent[e]["colocated_source"] |= has_src
    print("\n=== ④b 倒影实体（身份级幽灵）===")
    print(f"  含倒影的实体 {len(refl_ent)} 个；其中「源也在同实体」的 {sum(1 for v in refl_ent.values() if v['colocated_source'])}")
    for e, v in sorted(refl_ent.items()):
        print(f"    E{v['entity']:<12} 成员 {v['members']}  倒影 {v['n_reflection']}"
              f"  源同实体={v['colocated_source']}")

    out = dict(
        version="reflection_ab_v1",
        design="倒影「打 flag 降权（不删框）」的同序列 A/B；两臂冻结同一 T_same",
        arms=dict(
            A=dict(label="现状：倒影作普通节点", n_nodes=A["n_nodes"], n_gperson=A["n_gperson"],
                   n_merge=A["n_merge"], violations=A["invariant_violations"],
                   human_truth=prf("A"), calib_T_same=A["params"]["T_same"]),
            B=dict(label="倒影降权：倒影非独立观测", n_nodes=B["n_nodes"], n_gperson=B["n_gperson"],
                   n_merge=B["n_merge"], violations=B["invariant_violations"],
                   human_truth=prf("B"), calib_T_same=B["params"]["T_same"]),
        ),
        n_gperson_delta=B["n_gperson"] - A["n_gperson"],
        common_eval=dict(n=len(co), per_pair=[dict(pair=r["pair"], y=r["y"], A=r["A"], B=r["B"])
                                              for r in co],
                         changed=[dict(pair=r["pair"], y=r["y"], A=r["A"], B=r["B"]) for r in changed]),
        contaminated_clusters=[dict(a_cluster=lab(g), n_members=len(m),
                                   n_reflection=sum(1 for k in m if k in refl_keys))
                               for g, m in sorted(contaminated.items())],
        reflection_entities=[dict(entity=v["entity"], members=v["members"],
                                  n_reflection=v["n_reflection"],
                                  colocated_source=bool(v["colocated_source"]))
                             for _, v in sorted(refl_ent.items())],
        structure_change=struct,
        spurious_merges=spurious,
        reflection_attach=[dict(**r, src_cluster_b=ndB.get(r["source"]), relocated=bool(
            ndB.get(r["source"]) and ndA.get(r["refl"]) and ndB.get(r["source"]) != ndA.get(r["refl"])))
            for r in A["reflection_attach"]],
        relay_contamination=dict(
            n_edges=len(touched), by_status=dict(st), by_verdict=dict(stv),
            n_active=st.get("active", 0),
            n_pure_mirror_endpoint=sum(1 for t in touched
                                       if any(v["pure_mirror"] for v in t["mirror_sides"].values())),
            rows=touched,
            note="端点=实体（edges_v2_gated 的 cam:eid）。倒影实体不是独立的人，而是某真人观测的镜像副本 → "
                 "该边**不是必然为假**，而是**端点错位**（语义等价于指向其源实体的边）。"
                 "本表把倒影端重定向到源实体，再与 B 臂身份层对账：corroborated=源与邻居被判同一人；"
                 "contradicted=源与邻居被判不同人（边与身份层冲突 → 候选假链）。"
                 "★ 旧 reflection_impact_v1 用 obs id 比对 entity id（整数撞号）→ 旧的"
                 "『relay 6 条且全 audit』为误数，已由本表取代。"),
        caveat=[
            "① 评测集缩小时「召回提升」可能是假象 → 本表用共同可评集，并逐对列出判定变化。",
            "② 两臂必须冻结同一 T_same，否则差异来自阈值漂移（实测未冻结时 264→204 是漂移所致）。",
            "③ relay 污染必须按 entity 层级比对，不可按 obs 级 id（旧 reflection_impact_v1 曾撞号）。",
            "④ 『contradicted』≠ 边为假：gperson 召回仅 0.54，未合并也可能只是身份层漏接。"
            "该清单是**两个子系统的分歧清单**，供人工定向裁决。",
            "⑤ 『倒影搭桥导致的合并被拆开』也需真值裁决：拆开可能对（桥错），也可能错"
            "（两人确为同一人、桥恰好连对）。当前无真值覆盖这 3 处。",
        ],
        pending_adjudication=dict(
            bridge_merges=[dict(a_cluster=s["a_cluster"], real_members=s["real_members"],
                                b_clusters=s["b_clusters"]) for s in spurious],
            conflicted_relay_edges=[dict(edge=t["edge"], status=t["status"], score=t["score"],
                                         g_neighbor=t["g_neighbor"], g_source=t["g_source"])
                                    for t in touched if t["verdict"] == "contradicted"],
            note="人工裁决入口：这两份清单可在控制台 truth 模块逐条判定（P 优先：conflicted 边）。"),
    )
    json.dump(out, open(ROOT + "/" + a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {a.out}")
    con.close()


if __name__ == "__main__":
    main()
