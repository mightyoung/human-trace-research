# -*- coding: utf-8 -*-
"""ops_run 验收用命令：生成一份台账总览快照（可直接用于汇报/交接）。

产物：ops/reports/overview_snapshot.json + metrics.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ops_db as db  # noqa: E402

OUT = os.path.join(ROOT, "ops", "reports")
os.makedirs(OUT, exist_ok=True)

conn = db.connect()
db.init_db(conn)
try:
    ov = db.overview(conn)
finally:
    conn.close()

with open(os.path.join(OUT, "overview_snapshot.json"), "w", encoding="utf-8") as f:
    json.dump(ov, f, ensure_ascii=False, indent=1)

ann = ov["annotation"]
with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"schema": "metrics/v1", "split": "all", "items": [
        {"name": "pending_items", "value": ann["pending"], "n": ann["items"],
         "note": "待判条目数"},
        {"name": "done_pct", "value": ann["pct"], "n": ann["items"],
         "note": "标注完成率（按候选全集口径）"},
        {"name": "runs_total", "value": ov["runs"]["total"], "n": ov["runs"]["total"]},
    ]}, f, ensure_ascii=False, indent=1)

print(f"快照完成：{ann['done']}/{ann['items']} 已判，待判 {ann['pending']}")
