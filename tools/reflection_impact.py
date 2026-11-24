#!/usr/bin/env python
"""S6 下游影响：量化「墙镜倒影未被识别」对 relay/stitch/visit/gperson 的污染。

输入：analysis/gil/wallmirror_assign_v1.json（S6 判定）+ GIL store
输出：analysis/gil/reflection_impact_v1.json
"""
import collections
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output_crosscam/analysis/gil")


def main():
    con = sqlite3.connect(os.path.join(OUT, "store_v1/gil.sqlite"))
    con.row_factory = sqlite3.Row
    d = json.load(open(os.path.join(OUT, "wallmirror_assign_v1.json"), encoding="utf-8"))

    wall_key = set()
    for cam in d:
        if not isinstance(d[cam], dict) or "mirror_rows" not in d[cam]:
            continue
        for x in d[cam]["mirror_rows"]:
            wall_key.add((cam, x["tid"]))
    key2oid = {(r["cam"], r["tid"]): r["id"]
               for r in con.execute("SELECT id,cam,tid FROM obs WHERE tombstone=0")}
    wall_ids = {key2oid[k] for k in wall_key if k in key2oid}
    # 地板倒影（S5）
    floor_ids = {r["id"] for r in con.execute("SELECT id,flags FROM obs WHERE tombstone=0")
                 if r["flags"] and json.loads(r["flags"]).get("reflection", 0) > 0}
    oid2info = {r["id"]: dict(r) for r in
                con.execute("SELECT id,cam,tid,t_first,t_last,entity_id FROM obs WHERE tombstone=0")}

    # 边污染
    # ★ 修正（2026-10-14）：GIL `edge` 表的 a_id/b_id 依 a_kind/b_kind 而异——
    #   relay/relay_gate 等是 **entity 级**（a_id = entity.id），stitch/stitch_kin 是 **obs 级**。
    #   旧版把二者一律当作 obs id 比对 → 与 entity id 整数撞号，得到「relay 6 条」的假数。
    #   现按 kind 正确解析：entity 级 → 取该实体全部成员 obs；obs 级 → 直接比对。
    ent_members = collections.defaultdict(list)
    for r in con.execute("SELECT id,entity_id FROM obs WHERE tombstone=0 AND entity_id IS NOT NULL"):
        ent_members[r["entity_id"]].append(r["id"])
    edge_hit = {}
    for k in ["relay", "relay_v5", "relay_gate", "stitch", "stitch_kin"]:
        rows = []
        for r in con.execute("SELECT a_kind,a_id,b_kind,b_id,dt,score,status FROM edge WHERE kind=?",
                             (k,)):
            for kind, eid in ((r["a_kind"], r["a_id"]), (r["b_kind"], r["b_id"])):
                hits = ([oid for oid in ent_members.get(eid, []) if oid in wall_ids]
                        if kind == "entity" else ([eid] if eid in wall_ids else []))
                if hits:
                    prov = (kind == "entity"
                            and len(hits) == len(ent_members.get(eid, [])))   # 端点实体成员全是倒影
                    rows.append(dict(obs=",".join(f"{oid2info[o]['cam']}#{oid2info[o]['tid']}"
                                                  for o in hits),
                                     endpoint_kind=kind, status=r["status"], score=r["score"],
                                     dt=r["dt"], mirror_only_endpoint=prov))
        edge_hit[k] = dict(n=len(rows),
                           n_mirror_only_endpoint=sum(1 for x in rows if x["mirror_only_endpoint"]),
                           rows=rows)

    # visit 污染
    vrows = [dict(id=r["id"], cam=r["cam"], kind=r["kind"], t=r["t"],
                  obs=f"{oid2info[r['obs_id']]['cam']}#{oid2info[r['obs_id']]['tid']}",
                  gperson_id=r["gperson_id"])
             for r in con.execute("SELECT id,cam,kind,t,obs_id,gperson_id FROM visit")
             if r["obs_id"] in wall_ids]

    # gperson 污染
    g2o = collections.defaultdict(list)
    for r in con.execute("SELECT gperson_id,node_id,score FROM membership WHERE node_kind='obs'"):
        g2o[r["gperson_id"]].append((r["node_id"], r["score"]))
    gph = []
    for g, mem in g2o.items():
        nw = sum(1 for o, _ in mem if o in wall_ids)
        if not nw:
            continue
        cams = collections.Counter(oid2info[o]["cam"] for o, _ in mem if o in oid2info)
        t0 = min(oid2info[o]["t_first"] for o, _ in mem if o in oid2info)
        t1 = max(oid2info[o]["t_last"] for o, _ in mem if o in oid2info)
        gph.append(dict(gperson_id=g, n_members=len(mem), n_wall_reflection=nw,
                        ratio=round(nw / len(mem), 3), cams=dict(cams),
                        t_span=[round(t0, 1), round(t1, 1)],
                        verdict="ghost" if nw >= len(mem) * 0.5 else "contaminated"))
    gph.sort(key=lambda x: -x["ratio"])

    out = dict(version="reflection_impact_v1",
               source="wallmirror_assign_v1（S6 墙镜判定）+ GIL",
               n_wall_reflection_in_gil=len(wall_ids),
               n_floor_reflection_in_gil=len(floor_ids),
               note="S5 地板倒影在 clean2 已基本被剔除（仅 1 条残留）；本表量化的是"
                    "S6 新发现的**墙镜倒影**（从未被掩膜/判据处理）造成的污染。"
                    "★ 2026-10-14 更正：初版把 GIL `edge.a_id/b_id` 一律当 obs id 比对，"
                    "而 relay/relay_gate 实为 **entity 级**（a_id=entity.id）→ 与 obs id 整数撞号，"
                    "得出「relay 6 条且全 audit」的假数。现按 a_kind/b_kind 正确解析；"
                    "正确的 relay 污染为 31 条（relay_gate 3 条），其中端点=纯倒影实体 23 条；"
                    "权威口径见 reflection_ab_v1.json（entity 级 + 重定向对账）。",
               edge_contamination=edge_hit,
               visit_contamination=dict(n=len(vrows), rows=vrows),
               gperson_contamination=dict(
                   n_gperson_affected=len(gph),
                   n_ghost=sum(1 for x in gph if x["verdict"] == "ghost"),
                   rows=gph))
    p = os.path.join(OUT, "reflection_impact_v1.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {p}")
    print(f"  墙镜倒影（在 GIL）{len(wall_ids)} 条；地板倒影（S5 残留）{len(floor_ids)} 条")
    for k, v in edge_hit.items():
        print(f"  edge {k:12s} 被污染 {v['n']:3d}（其中端点=纯倒影实体，即该端点不是独立的人 {v['n_mirror_only_endpoint']}）")
    print(f"  visit 被污染 {len(vrows)} 条")
    print(f"  gperson 受影响 {len(gph)}（其中幽灵人 {out['gperson_contamination']['n_ghost']}）")
    for x in gph:
        print(f"    g{x['gperson_id']}: {x['n_wall_reflection']}/{x['n_members']} "
              f"({x['ratio']:.0%}) {x['verdict']} 机位{dict(x['cams'])}")


if __name__ == "__main__":
    main()
