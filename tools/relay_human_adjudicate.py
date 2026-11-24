# -*- coding: utf-8 -*-
"""人工裁决版 relay 边集：把 S6-b 裁决并入 relay v2 门控边集。

输入：analysis/relay_v2/edges_v2_gated.json（196 边）+ 裁决台账（--ver，默认 v1）
规则：
  - 裁决 reject（不同人）→ 边 status='rejected'（保留记录，带 note）
  - 裁决 confirm（同人） → 边 status='confirmed'（人工背书，active 仍为 active 语义）
  - 其余（defer / 无裁决）→ 原样
- 默认 dry-run（只打印）；--write 才落文件（--out，默认 edges_v2_gated_human.json）
- 幂等：台账应用可重复执行（v1 ⊂ v2，重复应用同一裁决不产生额外变化）
- 不覆盖原 gated 文件；不动 GIL。

用法：
  python tools/relay_human_adjudicate.py [--write]
  python tools/relay_human_adjudicate.py --ver output_crosscam/analysis/gil/human_adjudication_v2.json --write
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "output_crosscam/analysis/relay_v2/edges_v2_gated.json")
VER = os.path.join(ROOT, "output_crosscam/analysis/gil/human_adjudication_v1.json")
OUT = os.path.join(ROOT, "output_crosscam/analysis/relay_v2/edges_v2_gated_human.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--ver", default=VER, help="裁决台账路径（默认 human_adjudication_v1.json）")
    ap.add_argument("--out", default=OUT, help="输出边集路径（默认 edges_v2_gated_human.json）")
    a = ap.parse_args()
    ver_p = a.ver if os.path.isabs(a.ver) else os.path.join(ROOT, a.ver)
    out_p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)

    d = json.load(open(SRC, encoding="utf-8"))
    vj = json.load(open(ver_p, encoding="utf-8"))
    edges = d["edges"]

    by_key = {}
    for e in edges:
        by_key[(e["a"], e["b"])] = e

    verdicts = {}
    for it in vj["items"]:
        if it["group"] != "relay_conflict":
            continue
        ea, eb = it["edge"].split("->")
        verdicts[(ea, eb)] = it

    print("relay 边集 %d 条（%s）" % (len(edges), Counter(e["status"] for e in edges)))
    print("裁决覆盖 relay 冲突边 %d 条" % len(verdicts))
    print("-" * 72)

    n_hit = n_miss = 0
    changes = []
    for k, it in sorted(verdicts.items()):
        e = by_key.get(k)
        if e is None:
            n_miss += 1
            print("  ⚠ 未在 v2 边集中找到：%s->%s（裁决 v=%s）" % (k[0], k[1], it["verdict"]))
            continue
        n_hit += 1
        old = e["status"]
        if it["action"] == "reject":
            e["status"] = "rejected"
            e["human_verdict"] = "reject"
        elif it["action"] == "confirm":
            e["status"] = "confirmed"
            e["human_verdict"] = "confirm"
        else:
            e["human_verdict"] = "defer"
        e["human_note"] = it.get("rationale", "")
        changes.append((k, old, e["status"], it["verdict"]))
        print("  [%s] %-38s %s → %s" % ({1: "同人", 0: "不是", 2: "看不清"}[it["verdict"]],
                                        "%s->%s" % k, old, e["status"]))

    after = Counter(e["status"] for e in edges)
    print("-" * 72)
    print("命中 %d / 缺失 %d" % (n_hit, n_miss))
    print("状态分布：%s → %s" % (dict(Counter(e["status"] for e in d["edges"]
                                              if not e.get("human_verdict"))), dict(after)))

    if a.write:
        out = dict(d)
        out["version"] = "edges_v2_gated_human"
        out["based_on"] = d.get("version")
        out["human_adjudication"] = dict(
            source=os.path.relpath(ver_p, ROOT).replace("\\", "/"),
            created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_adjudicated=len(verdicts), n_applied=n_hit,
            by_action=dict(Counter(it["action"] for it in verdicts.values())),
        )
        out["summary"] = dict(d.get("summary") or {},
                              by_status=dict(after))
        json.dump(out, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[out] %s" % out_p)
    else:
        print("(dry-run；加 --write 落文件)")


if __name__ == "__main__":
    main()
