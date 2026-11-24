# -*- coding: utf-8 -*-
"""把人工裁决版 relay 边集镜像同步到 GIL 的 edge 表。

背景：磁盘 `relay_v2/edges_v2_gated_human.json` 已含人工裁决（confirmed / rejected），
但 GIL 的 `edge` 表（kind='relay' / 'relay_gate'）仍是原 gated 口径 → **镜像不一致**。
本工具把两者的 status 对齐，并在 `evidence` 里留痕（human_verdict / prev_status），可回退。

★ 只改 GIL 侧的 `status` 与 `evidence`；不动任何实体/成员/visit/owner/annotation。
★ 默认 dry-run；`--apply` 才写。写前请先备份 gil.sqlite。

用法：
  python tools/gil_relay_mirror_sync.py            # dry-run
  python tools/gil_relay_mirror_sync.py --apply
"""
import argparse
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GIL = os.path.join(ROOT, "output_crosscam/analysis/gil/store_v1/gil.sqlite")
SRC = os.path.join(ROOT, "output_crosscam/analysis/relay_v2/edges_v2_gated_human.json")
KINDS = ("relay", "relay_gate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--store", default=GIL)
    a = ap.parse_args()

    d = json.load(open(a.src, encoding="utf-8"))
    edges = d["edges"]

    con = sqlite3.connect(a.store)
    con.row_factory = sqlite3.Row
    ent = {(r["cam"], r["eid"]): r["id"] for r in con.execute("SELECT id,cam,eid FROM entity")}

    # GIL 索引：(kind, a_id, b_id) → row（含双向）
    idx = {}
    for r in con.execute("SELECT id,kind,a_id,b_id,status,evidence FROM edge "
                         "WHERE kind IN ('relay','relay_gate')"):
        idx[(r["kind"], r["a_id"], r["b_id"])] = dict(r)
        idx.setdefault((r["kind"], r["b_id"], r["a_id"]), dict(r))

    print("磁盘边集 %s" % os.path.relpath(a.src, ROOT).replace("\\", "/"))
    print("  状态分布：%s" % dict(Counter(e["status"] for e in edges)))
    print("GIL 镜像：%s" % dict(Counter(
        (r["kind"], r["status"]) for r in con.execute(
            "SELECT kind,status FROM edge WHERE kind IN ('relay','relay_gate')"))))
    print("-" * 74)

    changes, miss = [], []
    for e in edges:
        ca, ea = e["a"].split(":"); cb, eb = e["b"].split(":")
        ia, ib = ent.get((ca, ea)), ent.get((cb, eb))
        if ia is None or ib is None:
            miss.append((e["a"], e["b"], "entity 缺失"))
            continue
        kind = "relay_gate" if e["status"] == "gate" else "relay"
        row = idx.get((kind, ia, ib))
        if row is None:
            # 退化：不区分 kind 再找一次
            row = idx.get(("relay", ia, ib)) or idx.get(("relay_gate", ia, ib))
        if row is None:
            miss.append((e["a"], e["b"], "GIL 无此边"))
            continue
        if row["status"] != e["status"]:
            changes.append(dict(edge_id=row["id"], kind=row["kind"], a=e["a"], b=e["b"],
                                old=row["status"], new=e["status"],
                                human=e.get("human_verdict"), note=e.get("human_note")))

    print("需变更 %d 条 / 未匹配 %d 条" % (len(changes), len(miss)))
    for c in changes:
        print("  #%-4s [%s] %-38s %-8s → %-10s (human=%s)"
              % (c["edge_id"], c["kind"], "%s->%s" % (c["a"], c["b"]),
                 c["old"], c["new"], c["human"]))
    if miss:
        print("  ⚠ 未匹配：")
        for m in miss[:10]:
            print("     ", m)

    if a.apply and changes:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for c in changes:
            r = con.execute("SELECT evidence FROM edge WHERE id=?", (c["edge_id"],)).fetchone()
            ev = json.loads(r["evidence"]) if r["evidence"] else {}
            ev["prev_status"] = c["old"]
            ev["human_verdict"] = c["human"]
            ev["human_sync_ts"] = ts
            if c["note"]:
                ev["human_note"] = c["note"]
            con.execute("UPDATE edge SET status=?, evidence=? WHERE id=?",
                        (c["new"], json.dumps(ev, ensure_ascii=False), c["edge_id"]))
        con.commit()
        print("\n[apply] 已同步 %d 条" % len(changes))
        print("GIL 镜像（后）：%s" % dict(Counter(
            (r["kind"], r["status"]) for r in con.execute(
                "SELECT kind,status FROM edge WHERE kind IN ('relay','relay_gate')"))))
    elif a.apply:
        print("\n[apply] 无变更")
    else:
        print("\n(dry-run；加 --apply 落库)")
    con.close()


if __name__ == "__main__":
    main()
