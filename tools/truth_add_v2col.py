#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给现有 truth 配对候选 CSV 补 `in_relay_v2` 列（Relay v2 消费端适配，S1 后处理）。

- 就地升级 `pair_candidates.csv`（机器版）与 `pair_candidates_labeled.csv`（人工标注版）：
  在 `in_relay_v5` 后插入 `in_relay_v2`（active / audit / blocked / ""），
  **旧列与已有数据一字不动**。
- 首跑把原文件备份到 `truth/backup_pre_v2col/`；幂等（重复运行只刷新值，不重复插列）。
- 未来重跑 `portal_truth_candidates.py` 会自动带该列（本脚本仅用于升级既有产物）。

用法:
  python tools/truth_add_v2col.py [--dry-run]
"""
import argparse
import csv
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
TRUTH = os.path.join(ANA, "truth")
EDGES_V2 = os.path.join(ANA, "relay_v2", "edges_v2.json")
BACKUP = os.path.join(TRUTH, "backup_pre_v2col")
FILES = [os.path.join(TRUTH, "pair_candidates.csv"),
         os.path.join(TRUTH, "pair_candidates_labeled.csv")]


def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def load_state():
    d = json.load(open(EDGES_V2, encoding="utf-8"))
    state = {(e["a"], e["b"]): ("active" if e.get("status") == "active" else "audit")
             for e in d["edges"]}
    for g in d.get("conflict_groups", []):
        for x in g["blocked"]:
            state[(g["a"], x["b"])] = "blocked"
    return state


def upgrade(path, state, dry=False):
    name = os.path.basename(path)
    if not os.path.exists(path):
        print(f"[skip] {name} 不存在")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        print(f"[skip] {name} 空文件")
        return
    fields = list(rows[0].keys())
    added = False
    if "in_relay_v2" not in fields:
        added = True
        i = fields.index("in_relay_v5") + 1
        fields = fields[:i] + ["in_relay_v2"] + fields[i:]
    if not dry and added:
        os.makedirs(BACKUP, exist_ok=True)
        bkp = os.path.join(BACKUP, name)
        if not os.path.exists(bkp):
            shutil.copy2(path, bkp)
            print(f"  [backup] {bkp}")
    n_stat = {"active": 0, "audit": 0, "blocked": 0, "": 0}
    n_err = 0
    for r in rows:
        try:
            a, b = parse_pid(r["pair_id"])
            r["in_relay_v2"] = state.get((a, b), "")
        except Exception:  # noqa: BLE001
            r["in_relay_v2"] = ""
            n_err += 1
        n_stat[r["in_relay_v2"]] += 1
    if not dry:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"[{'dry' if dry else 'ok'}] {name}: {len(rows)} 行 | "
          f"{'插列' if added else '刷新值'} | "
          f"active={n_stat['active']} audit={n_stat['audit']} "
          f"blocked={n_stat['blocked']} 空={n_stat['']}"
          + (f" | parse_err={n_err}" if n_err else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    state = load_state()
    print(f"[src] edges_v2.json 状态表 {len(state)} 条")
    for p in FILES:
        upgrade(p, state, dry=args.dry_run)


if __name__ == "__main__":
    main()
