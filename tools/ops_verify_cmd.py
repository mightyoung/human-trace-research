# -*- coding: utf-8 -*-
"""ops_run 验收用命令：跑一次双写一致性校验并把报告落盘。

产物：ops/reports/verify_projection.json（台账 vs 原始 json 的差异报告）
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ops_db as db          # noqa: E402
import ops_ingest as ingest  # noqa: E402

OUT = os.path.join(ROOT, "ops", "reports")
os.makedirs(OUT, exist_ok=True)

conn = db.connect()
db.init_db(conn)
try:
    rep = ingest.verify(conn)
finally:
    conn.close()

total_bad = sum(v["n_missing_in_db"] + v["n_missing_in_json"] + v["n_conflict"]
                for v in rep.values())
path = os.path.join(OUT, "verify_projection.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump({"ok": total_bad == 0, "total_diff": total_bad, "detail": rep},
              f, ensure_ascii=False, indent=1)

# 同时写 metrics.json（ops_run 会把它登记为本次 run 的指标）
n_items = sum(v["items"] for v in rep.values())
with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"schema": "metrics/v1", "split": "all", "items": [
        {"name": "projection_diff", "value": total_bad, "n": n_items,
         "note": "台账与原始 json 的差异条数，0 表示双写一致"},
        {"name": "tasks_checked", "value": len(rep), "n": len(rep)},
    ]}, f, ensure_ascii=False, indent=1)

print(f"投影校验完成：差异 {total_bad} 项")
print(f"报告 -> {os.path.relpath(path, ROOT)}")
sys.exit(0 if total_bad == 0 else 3)
