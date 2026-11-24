# -*- coding: utf-8 -*-
"""S6-b 人工裁决回灌：把控制台 30 项判定落成 GIL 可消费的「裁决层」。

判定来源：控制台真值页「S6-b 待裁决」→ truth/labels.json 的 pair 命名空间（前缀 S6B_）。
判定语义（前端按钮）：
  ① relay_conflict   : 1=同人  0=不是  2=看不清
  ② bridge_merge     : 1=同人(该留)  0=不同人(该拆)  2=看不清
  ③ reflection_entity: 1=是倒影  0=是真人  2=看不清

动作映射（三层，逐层可独立执行）：
  ① relay 边 —— 1 → confirm（边为真；身份层漏接，建议合并 g_neighbor×g_source）
                0 → reject （假链；可选把 GIL edge.status 置 rejected）
                2 → defer
  ② 搭桥合并 —— 1 → keep   （B 臂拆分被否决，维持 A 臂合并）
                0 → split  （确认倒影搭桥=错误合并，按 B 臂 b_clusters 拆）
                2 → defer
  ③ 倒影实体 —— 1 → confirm_reflection（确认倒影，维持降权）
                0 → revoke_reflection （是真人 → 撤销倒影判定/降权）
                2 → defer

默认 **dry-run**：只打印 + 写台账 `analysis/gil/human_adjudication_v1.json`，**不写 GIL**。
`--apply`    ：写 GIL 的 `annotation` 表（target_kind='human_verdict' 独立命名空间，**不覆盖既有
               obs/pair 标注**）；`--apply-edges` 才把 rejected 边落到 edge.status（默认不动，避免
               改动 relay 消费口径与 C0 continuity）。`--apply-entities` 才清 4 条误判倒影的 obs.flags。

用法：
  python tools/gil_adjudication_ingest.py                  # dry-run
  python tools/gil_adjudication_ingest.py --apply          # 只写 annotation 裁决层
  python tools/gil_adjudication_ingest.py --apply --apply-edges
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LABELS = os.path.join(ROOT, "output_crosscam/analysis/truth/remote/labels.json")
AB = os.path.join(ROOT, "output_crosscam/analysis/gil/reflection_ab_v1.json")
GIL = os.path.join(ROOT, "output_crosscam/analysis/gil/store_v1/gil.sqlite")
OUT = os.path.join(ROOT, "output_crosscam/analysis/gil/human_adjudication_v1.json")

KIND_OF_PREFIX = {"S6B_P_": "relay_conflict", "S6B_B_": "bridge_merge", "S6B_R_": "reflection_entity"}


def load_labels():
    d = json.load(open(LABELS, encoding="utf-8"))
    out = {}
    for k, v in (d.get("pair") or {}).items():
        if k.startswith("S6B_"):
            out[k] = v
    return out


def gi(labels):
    """group → {id: {v,c,n}}"""
    g = {k: {} for k in KIND_OF_PREFIX.values()}
    for k, v in labels.items():
        for pre, grp in KIND_OF_PREFIX.items():
            if k.startswith(pre):
                g[grp][k] = v
                break
    return g


def parse_relay_id(pid):
    """S6B_P_343_E_343-E010__316_E_316-E015 → (ca, eida, cb, eidb)"""
    body = pid[len("S6B_P_"):]
    a, b = body.split("__")
    ca, eida = a.split("_E_")
    cb, eidb = b.split("_E_")
    return ca, eida, cb, eidb


def norm_eid(eid):
    """A/B 产物里实体写成 '316-316-E015'（cam-cam-EXXX 重复前缀）；GIL 里是 '316-E015'。"""
    return eid.split("-", 1)[1] if eid.count("-") == 2 else eid


def main():
    global LABELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="写 annotation 裁决层")
    ap.add_argument("--apply-edges", action="store_true", help="把 rejected relay 边落 edge.status（默认不动）")
    ap.add_argument("--apply-entities", action="store_true", help="清误判倒影的 obs.flags（默认不动）")
    ap.add_argument("--labels", default=LABELS)
    args = ap.parse_args()

    LABELS = args.labels

    labels = load_labels()
    grp = gi(labels)
    ab = json.load(open(AB, encoding="utf-8"))

    con = sqlite3.connect(GIL)
    con.row_factory = sqlite3.Row
    eid2id = {r["eid"]: r["id"] for r in con.execute("SELECT id,eid FROM entity")}
    id2eid = {v: k for k, v in eid2id.items()}
    gp_by_label = {r["label"]: dict(r) for r in con.execute("SELECT * FROM gperson")}

    # 期望清单（来自 A/B 产物）
    exp_relay = {}
    for row in ab["relay_contamination"]["rows"]:
        if row["verdict"] != "contradicted":
            continue
        ea, eb = row["edge"].split("->")
        ca, eida = ea.split(":"); cb, eidb = eb.split(":")
        exp_relay[f"S6B_P_{ca}_E_{eida}__{cb}_E_{eidb}"] = row
    exp_bridge = {f"S6B_B_{s['a_cluster']}": s for s in ab["spurious_merges"]}
    exp_entity = {f"S6B_R_{v['entity']}": v for v in ab["reflection_entities"]}

    print("=" * 78)
    print("S6-b 人工裁决回灌 · %s" % ("APPLY" if args.apply else "DRY-RUN"))
    print("=" * 78)
    print("判定文件: %s" % LABELS)
    print("期望: relay %d / bridge %d / entity %d = %d 项"
          % (len(exp_relay), len(exp_bridge), len(exp_entity),
             len(exp_relay) + len(exp_bridge) + len(exp_entity)))
    print("实收: relay %d / bridge %d / entity %d = %d 项"
          % (len(grp["relay_conflict"]), len(grp["bridge_merge"]),
             len(grp["reflection_entity"]), len(labels)))

    items = []
    n_missing = []

    # ---------------- ① relay 冲突边 ----------------
    print("\n" + "-" * 78)
    print("① relay 冲突边（16）—— 1=同人(边真) / 0=不是(假链) / 2=看不清")
    print("-" * 78)
    relay_stat = {"confirm": 0, "reject": 0, "defer": 0}
    for pid, row in sorted(exp_relay.items()):
        lab = grp["relay_conflict"].get(pid)
        if lab is None:
            n_missing.append(pid); continue
        v = lab.get("v")
        ca, eida, cb, eidb = parse_relay_id(pid)
        ia, ib = eid2id.get(eida), eid2id.get(eidb)
        st = con.execute(
            "SELECT id,status,score FROM edge WHERE kind='relay' AND a_kind='entity' AND b_kind='entity' "
            "AND a_id=? AND b_id=?", (ia, ib)).fetchone()
        act = "confirm" if v == 1 else "reject" if v == 0 else "defer"
        relay_stat[act] += 1
        items.append(dict(
            id=pid, group="relay_conflict", verdict=v, confidence=lab.get("c"), note=lab.get("n") or "",
            edge=row["edge"], edge_id=(st["id"] if st else None),
            edge_status=(st["status"] if st else None), cos=row["score"],
            g_neighbor=row["g_neighbor"], g_source=row["g_source"],
            action=act,
            rationale=("判为同一人 → relay 边成立；身份层漏接（邻居 %s vs 源 %s 未合并）"
                       % (row["g_neighbor"], row["g_source"])) if act == "confirm" else
                      ("判为不同人 → relay 边为假链（候选剔除）") if act == "reject" else
                      "看不清 → 保持待定",
        ))
        tag = {1: "同人", 0: "不是", 2: "看不清"}.get(v, "?")
        print("  [%s] %-45s cos=%.4f  %s→%s  edge#%s(%s)"
              % (tag, row["edge"], row["score"], row["g_neighbor"], row["g_source"],
                 (st["id"] if st else "—"), (st["status"] if st else "—")))

    # ---------------- ② 搭桥合并 ----------------
    print("\n" + "-" * 78)
    print("② 倒影搭桥合并（3）—— 1=同人(该留) / 0=不同人(该拆) / 2=看不清")
    print("-" * 78)
    bridge_stat = {"keep": 0, "split": 0, "defer": 0}
    for pid, s in sorted(exp_bridge.items()):
        lab = grp["bridge_merge"].get(pid)
        if lab is None:
            n_missing.append(pid); continue
        v = lab.get("v")
        act = "keep" if v == 1 else "split" if v == 0 else "defer"
        bridge_stat[act] += 1
        g = gp_by_label.get(s["a_cluster"], {})
        bcl = s["b_clusters"]
        items.append(dict(
            id=pid, group="bridge_merge", verdict=v, confidence=lab.get("c"), note=lab.get("n") or "",
            a_cluster=s["a_cluster"], n_reflection=s["n_reflection"],
            real_members=s["real_members"], b_clusters=bcl,
            gperson_id=g.get("id"), action=act,
            split_into=([{"label": k, "members": v2} for k, v2 in bcl.items()] if act == "split" else None),
            rationale=("同人 → 该合并保留（B 臂拆分为误拆）" if act == "keep" else
                       "不同人 → 确认倒影搭桥=错误合并，应拆回 %s" % list(bcl.keys()) if act == "split"
                       else "看不清 → 维持现状"),
        ))
        tag = {1: "同人/该留", 0: "不同人/该拆", 2: "看不清"}.get(v, "?")
        print("  [%s] %-12s 含倒影 %d 条  真实成员=%s  拆法=%s"
              % (tag, s["a_cluster"], s["n_reflection"], s["real_members"], list(bcl.keys())))

    # ---------------- ③ 倒影实体 ----------------
    print("\n" + "-" * 78)
    print("③ 倒影实体（11）—— 1=是倒影 / 0=是真人(撤销) / 2=看不清")
    print("-" * 78)
    ent_stat = {"confirm_reflection": 0, "revoke_reflection": 0, "defer": 0}
    for pid, v in sorted(exp_entity.items()):
        lab = grp["reflection_entity"].get(pid)
        if lab is None:
            n_missing.append(pid); continue
        vv = lab.get("v")
        act = ("confirm_reflection" if vv == 1 else
               "revoke_reflection" if vv == 0 else "defer")
        ent_stat[act] += 1
        eid = norm_eid(v["entity"])
        e_id = eid2id.get(eid)
        ids = [x["id"] for x in con.execute(
            "SELECT id FROM obs WHERE entity_id=?", (e_id,))]
        flags = []
        for oid in ids:
            f = con.execute("SELECT flags FROM obs WHERE id=?", (oid,)).fetchone()
            fl = json.loads(f["flags"]) if f and f["flags"] else {}
            flags.append({k: fl[k] for k in ("reflection", "explained_by", "wall_mirror",
                                             "wall_mirror_src", "reflection_score", "src")
                          if k in fl})
        items.append(dict(
            id=pid, group="reflection_entity", verdict=vv, confidence=lab.get("c"),
            note=lab.get("n") or "", entity=eid, entity_id=e_id,
            members=v["members"], n_reflection=v["n_reflection"],
            pure=len(v["members"]) == v["n_reflection"],
            obs_ids=ids, obs_flags=flags, action=act,
            rationale=("确认倒影 → 维持降权（非独立身份节点）" if act == "confirm_reflection" else
                       "判为真人 → 撤销倒影判定/降权（S5/S6 误报）" if act == "revoke_reflection"
                       else "看不清 → 保持现状"),
        ))
        tag = {1: "是倒影", 0: "是真人", 2: "看不清"}.get(vv, "?")
        print("  [%s] %-22s 成员=%s 纯倒影=%s  flags=%s"
              % (tag, eid, v["members"], len(v["members"]) == v["n_reflection"], flags))

    # ---------------- 影响面 ----------------
    n_relay_reject_active = sum(1 for i in items if i["group"] == "relay_conflict"
                                and i["action"] == "reject" and i["edge_status"] == "active")
    n_relay_reject_audit = sum(1 for i in items if i["group"] == "relay_conflict"
                               and i["action"] == "reject" and i["edge_status"] == "audit")
    revokes = [i["entity"] for i in items if i["group"] == "reflection_entity"
               and i["action"] == "revoke_reflection"]
    confirms = [i["entity"] for i in items if i["group"] == "reflection_entity"
                and i["action"] == "confirm_reflection"]

    impact = dict(
        relay=dict(confirm=relay_stat["confirm"], reject=relay_stat["reject"], defer=relay_stat["defer"],
                   reject_active=n_relay_reject_active, reject_audit=n_relay_reject_audit),
        bridge=dict(bridge_stat),
        entity=dict(confirm=ent_stat["confirm_reflection"], revoke=ent_stat["revoke_reflection"],
                    defer=ent_stat["defer"], revoked_entities=revokes, confirmed_entities=confirms),
        missing=n_missing,
        gil_read_only=(not args.apply),
    )

    print("\n" + "=" * 78)
    print("影响面")
    print("=" * 78)
    print("  relay 边：confirm %d / reject %d（active %d + audit %d）/ defer %d"
          % (relay_stat["confirm"], relay_stat["reject"], n_relay_reject_active,
             n_relay_reject_audit, relay_stat["defer"]))
    print("  搭桥合并：keep %d / split %d / defer %d  → 若执行 split，gperson +%d"
          % (bridge_stat["keep"], bridge_stat["split"], bridge_stat["defer"], bridge_stat["split"]))
    print("  倒影实体：确认倒影 %d / 撤销(误报) %d / 待定 %d"
          % (ent_stat["confirm_reflection"], ent_stat["revoke_reflection"], ent_stat["defer"]))
    if revokes:
        print("    ★ 撤销清单（原判倒影实为真人）：%s" % ", ".join(revokes))
    if n_missing:
        print("  ⚠ 清单与判定不匹配（缺判）：%s" % n_missing)

    ledger = dict(
        version="human_adjudication_v1",
        created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source=os.path.relpath(LABELS, ROOT).replace("\\", "/"),
        applied=bool(args.apply),
        applied_edges=bool(args.apply_edges),
        applied_entities=bool(args.apply_entities),
        n=len(items), by_group={k: len(v) for k, v in
                                (("relay_conflict", grp["relay_conflict"]),
                                 ("bridge_merge", grp["bridge_merge"]),
                                 ("reflection_entity", grp["reflection_entity"]))},
        impact=impact, items=items,
    )
    json.dump(ledger, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[台账] %s（%d 项）" % (OUT, len(items)))

    # ---------------- 写入 GIL ----------------
    if args.apply:
        ts = datetime.now().isoformat(timespec="seconds")
        n = 0
        for it in items:
            val = {1: "1", 0: "0", 2: "?"}.get(it["verdict"], "?")
            note = "[%s] %s%s" % (it["group"], it["rationale"],
                                  (" 备注:" + it["note"]) if it["note"] else "")
            con.execute(
                "INSERT OR REPLACE INTO annotation(target_kind,target_id,value,confidence,note,author,ts) "
                "VALUES('human_verdict',?,?,?,?,'human',?)",
                (it["id"], it["action"], it.get("confidence"), note, ts))
            n += 1
        con.commit()
        print("[apply] annotation 写入 %d 条（target_kind='human_verdict'）" % n)
        if args.apply_edges:
            m = 0
            for it in items:
                if it["group"] == "relay_conflict" and it["action"] == "reject" and it["edge_id"]:
                    con.execute("UPDATE edge SET status='rejected' WHERE id=?", (it["edge_id"],))
                    m += 1
            con.commit()
            print("[apply] edge.status → rejected %d 条" % m)
        if args.apply_entities:
            k = 0
            for it in items:
                if it["group"] == "reflection_entity" and it["action"] == "revoke_reflection":
                    for m in it["members"]:
                        c, t = m.split("|")
                        o = con.execute("SELECT id,flags FROM obs WHERE cam=? AND tid=?",
                                        (c, int(t))).fetchone()
                        if not o:
                            continue
                        fl = json.loads(o["flags"]) if o["flags"] else {}
                        for key in ("reflection", "explained_by", "wall_mirror",
                                    "wall_mirror_src", "wall_mirror_u_axis"):
                            fl.pop(key, None)
                        fl["human_revoked_reflection"] = True
                        con.execute("UPDATE obs SET flags=? WHERE id=?",
                                    (json.dumps(fl, ensure_ascii=False), o["id"]))
                        # obs 级标注：reflection → person（撤销倒影判定）
                        con.execute(
                            "INSERT OR REPLACE INTO annotation"
                            "(target_kind,target_id,value,confidence,note,author,ts) "
                            "VALUES('obs',?,'person',?,?, 'human',?)",
                            (m, it.get("confidence"),
                             "人工裁决：原判倒影实为真人（S6-b 待裁决）", ts))
                        k += 1
            con.commit()
            print("[apply] 撤销倒影：obs 标注/flag 修正 %d 条观测" % k)


if __name__ == "__main__":
    main()
