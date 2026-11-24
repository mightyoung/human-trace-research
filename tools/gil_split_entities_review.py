#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复核名单：多 obs 实体被 gperson 拆分清单（裁决版口径）+ 边界对专项。

背景：S3 传导报告 §遗留——"剩余 27 个被拆实体（318-E003 跨 5 簇等）：建议排查"；
      以及"343-E009↔E025 边界对：进下一批人工复核"。
口径：从**当前主库**（裁决版 gperson_v2 membership）重算：
  entity(n_obs≥2) 的成员 obs → 所属 gperson 集合；|set|≥2 即"被拆"。
输出：analysis/gil/split_entity_review_v1.json + 控制台清单
用法：DET_VIEW=clean2 python tools/gil_split_entities_review.py [--out ...]
"""
import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIL = os.path.join(ROOT, "output_crosscam", "analysis", "gil", "store_v1", "gil.sqlite")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "gil", "split_entity_review_v1.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import sqlite3
    con = sqlite3.connect(GIL)
    con.row_factory = sqlite3.Row

    # entity 成员 obs → gperson（裁决版 latest membership）
    obs2gp = {}
    for r in con.execute("SELECT node_id, gperson_id FROM membership "
                         "WHERE node_kind='obs' AND src LIKE 'gperson_v2%'"):
        obs2gp[int(r["node_id"])] = int(r["gperson_id"])
    # 备选：老版本 src（若库被更新过）
    if not obs2gp:
        for r in con.execute("SELECT node_id, gperson_id FROM membership WHERE node_kind='obs'"):
            obs2gp[int(r["node_id"])] = int(r["gperson_id"])

    gplab = {r["id"]: r["label"] for r in con.execute("SELECT id, label FROM gperson")}

    obs_meta = {}
    for r in con.execute("SELECT id, cam, tid, t_first, t_last, entity_id FROM obs WHERE tombstone=0"):
        obs_meta[r["id"]] = dict(cam=r["cam"], tid=r["tid"], t0=r["t_first"], t1=r["t_last"],
                                 ent=r["entity_id"])
    ent2obs = collections.defaultdict(list)
    for oid, m in obs_meta.items():
        if m["ent"] is not None:
            ent2obs[m["ent"]].append(oid)

    ents = {}
    for r in con.execute("SELECT id, cam, eid, n_obs, t0, t1 FROM entity"):
        ents[r["id"]] = dict(r)

    multi = [e for e in ents.values() if e["n_obs"] >= 2]
    splits = []
    no_mem = []
    for e in multi:
        obs = ent2obs.get(e["id"], [])
        gs = collections.Counter()
        for o in obs:
            g = obs2gp.get(o)
            if g is None:
                no_mem.append((e["eid"], o))
            else:
                gs[g] += 1
        if len(gs) >= 2:
            splits.append(dict(
                eid=e["eid"], cam=e["cam"], n_obs=e["n_obs"], n_clusters=len(gs),
                t0=e["t0"], t1=e["t1"],
                clusters=[dict(gperson=g, label=gplab.get(g), n=c)
                          for g, c in sorted(gs.items(), key=lambda x: -x[1])],
            ))
    splits.sort(key=lambda x: (-x["n_clusters"], -x["n_obs"]))

    print(f"=== 裁决版口径：多 obs 实体 {len(multi)}，被拆 {len(splits)}，"
          f"未入 membership 的 obs {len(no_mem)} ===")
    for s in splits:
        print(f"  {s['eid']:<12} n_obs={s['n_obs']:<2} 跨 {s['n_clusters']} 簇: "
              + ", ".join(f"{c['label']}({c['n']})" for c in s["clusters"]))

    # 边界对专项
    border = []
    for eid in ("343-E009", "316-E025"):
        for e in ents.values():
            if e["eid"] == eid:
                obs = ent2obs.get(e["id"], [])
                gs = collections.Counter(obs2gp.get(o) for o in obs)
                border.append(dict(eid=eid, n_obs=e["n_obs"], t0=e["t0"], t1=e["t1"],
                                   clusters=[dict(gperson=g, label=gplab.get(g), n=c)
                                             for g, c in sorted(gs.items(), key=lambda x: -x[1])]))
    print("\n=== 边界对 343-E009 / 316-E025（S3 传导中'破坏对'，进展复核）===")
    for b in border:
        print(f"  {b['eid']}: n_obs={b['n_obs']} clusters="
              + ", ".join(f"{c['label']}({c['n']})" for c in b["clusters"]))

    # relay 边涉入（被拆实体出现在哪些 relay 边）
    gated = os.path.join(ROOT, "output_crosscam", "analysis", "relay_v2",
                         "edges_v2_gated_human.json")
    rel = []
    if os.path.exists(gated):
        gd = json.load(open(gated, encoding="utf-8"))["edges"]
        split_eids = {s["eid"] for s in splits}
        for e in gd:
            ea, eb = e["a"].split(":")[1], e["b"].split(":")[1]
            if ea in split_eids or eb in split_eids:
                rel.append(dict(edge=f"{e['a']}->{e['b']}", status=e["status"],
                                sim=e.get("sim"), dt=e.get("dt"),
                                note=("a 被拆" if ea in split_eids else "") +
                                     (" b 被拆" if eb in split_eids else "")))
    print(f"\n=== 被拆实体涉入 relay 边 {len(rel)} 条 ===")
    for r in rel[:15]:
        print(f"  {r['edge']:<26} {r['status']:<10} sim={r['sim']} dt={r['dt']} {r['note']}")

    out = dict(
        version="gil/split_entity_review/1",
        generated_from="store_v1/gil.sqlite（裁决版 gperson_v2 membership）",
        n_multi_obs_entities=len(multi),
        n_split=len(splits),
        n_obs_no_membership=len(no_mem),
        splits=splits,
        border_pairs=border,
        relay_edges_with_split_entities=rel,
        note="复核建议：按 n_clusters 降序人工审；318 为小目标机位（cos 不稳）优先；"
             "跨 5 簇的 318-E003 首查。343-E009↔E025 为 S3 传导中的贪心重排边界对，"
             "本轮清单显示其当前簇构成，供下一批判读确认。",
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
