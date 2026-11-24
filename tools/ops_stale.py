#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stale 判定：输入变了但结果没重跑，就该标出来

为什么需要
----
SCRFD 三文件修复后，历史人脸坐标产物不能再与新的混用——但台账里它们仍是
`succeeded`。这类「输入变了、结果没重跑」的状态现在只能靠人记，而这正是
最容易出事的地方：**一个过期的成功，比一个明确的失败更危险**。

判据
----
只看会让结果失效的三样：数据快照、模型权重、执行脚本。
超参不纳入（由 run 自己记录，不会事后变化）。

* 默认只比 **数据 + 权重**，代码变化不算 stale（代码改动不等于结果失效，且太频繁）
* `--include-code` 才把执行脚本 sha1 纳入
* 任一分量缺失就不判定——拿不准的事不假装拿得准

用法
----
    python tools/ops_stale.py check --dry-run      # 先看会标哪些
    python tools/ops_stale.py check                # 落库
    python tools/ops_stale.py check --include-code
    python tools/ops_stale.py show <run_id>        # 看某条 run 的指纹与差异

约定
----
* 只在显式调用时判定，不在启动时自动跑（一次开服务就把 41 条全标黄没有意义）
* 历史 run 若没有 `input_breakdown`，跳过并计数；不追溯猜测
* 标记写 `status='stale'`，差异写进 `note`（保留原 note 前缀）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402


def sha1_file(path: str, chunk: int = 1 << 20) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def current_parts(conn, run: dict, include_code: bool) -> dict:
    """按 run 记录的分量重新取值（不是按当前配置）。"""
    rec = db.jload(run.get("input_breakdown"), {}) or {}
    cur = {}

    ds_id = rec.get("dataset_id") or run.get("dataset_id")
    if ds_id:
        r = conn.execute("SELECT snapshot_hash FROM dataset WHERE id=?", (ds_id,)).fetchone()
        cur["dataset_id"] = ds_id
        cur["dataset"] = r["snapshot_hash"] if r else None

    mid = rec.get("model_id") or run.get("model_id")
    if mid:
        r = conn.execute("SELECT sha1 FROM model WHERE id=?", (mid,)).fetchone()
        cur["model_id"] = mid
        cur["model"] = r["sha1"] if r else None

    if include_code:
        code = (db.jload(run.get("env_json"), {}) or {}).get("code") or {}
        path = code.get("path")
        cur["code_path"] = path
        cur["code"] = sha1_file(os.path.join(db.ROOT, path)) if path else None
    return cur


def diff_parts(recorded: dict, cur: dict, include_code: bool) -> list[str]:
    """返回差异描述，空列表 = 没变。"""
    out = []
    for key, label in (("dataset", "数据快照"), ("model", "模型权重"), ("code", "执行脚本")):
        if key == "code" and not include_code:
            continue
        if recorded.get(key) is None and cur.get(key) is None:
            continue          # 两边都没有，不算变化
        if recorded.get(key) is None:
            continue          # 当时没记，无法判定
        if cur.get(key) is None:
            out.append(f"{label}已缺失（原 {str(recorded.get(key))[:10]}）")
        elif recorded.get(key) != cur.get(key):
            out.append(f"{label}变了（{str(recorded.get(key))[:10]} → {str(cur.get(key))[:10]}）")
    return out


def cmd_check(a) -> int:
    conn = db.connect(a.db)
    marked, skipped, clean = [], 0, 0
    try:
        db.init_db(conn)
        kinds = [k.strip() for k in (a.kind or "").split(",") if k.strip()]
        sql = "SELECT * FROM run WHERE status IN ('succeeded')"
        args: list = []
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        sql += " ORDER BY started_at DESC"
        rows = [dict(r) for r in conn.execute(sql, args)]

        for run in rows:
            if not run.get("input_breakdown"):
                skipped += 1
                continue
            rec = db.jload(run["input_breakdown"], {}) or {}
            cur = current_parts(conn, run, a.include_code)
            diffs = diff_parts(rec, cur, a.include_code)
            if not diffs:
                clean += 1
                continue
            marked.append({"id": run["id"], "name": run["name"], "diffs": diffs})
            if a.dry_run:
                continue
            note = run.get("note") or ""
            tag = "[stale] " + "；".join(diffs)
            new_note = tag if tag in note else (f"{note} | {tag}" if note else tag)
            conn.execute("UPDATE run SET status='stale', note=? WHERE id=?",
                         (new_note, run["id"]))
            db.audit(conn, "ops_stale", "run.stale", "run", run["id"],
                     {"status": run["status"]}, {"diffs": diffs})
        if not a.dry_run:
            conn.commit()
    finally:
        conn.close()

    print(("[演练] " if a.dry_run else "") +
          f"检查 {len(rows)} 条 · 未变 {clean} · 无指纹跳过 {skipped} · 判 stale {len(marked)}")
    for m in marked:
        print(f"  {m['id']}  {m['name']}")
        for d in m["diffs"]:
            print(f"      {d}")
    return 0


def cmd_show(a) -> int:
    conn = db.connect(a.db)
    try:
        run = conn.execute("SELECT * FROM run WHERE id=?", (a.run_id,)).fetchone()
    finally:
        conn.close()
    if not run:
        raise SystemExit(f"run 不存在: {a.run_id}")
    run = dict(run)
    print(f"run        {run['id']}  [{run['status']}]  {run['name']}")
    print(f"input_hash {run.get('input_hash') or '—（历史 run 未记录）'}")
    rec = db.jload(run.get("input_breakdown"), {}) or {}
    print("记录的分量:")
    for k in ("dataset_id", "dataset", "dataset_alg", "model_id", "model", "code"):
        if k in rec:
            print(f"  {k:<14}{rec[k]}")
    if rec:
        conn = db.connect(a.db)
        try:
            cur = current_parts(conn, run, include_code=True)
        finally:
            conn.close()
        print("当前:")
        for k in ("dataset_id", "dataset", "model_id", "model", "code_path", "code"):
            if k in cur:
                print(f"  {k:<14}{cur[k]}")
        diffs = diff_parts(rec, cur, include_code=True)
        print("差异: " + ("；".join(diffs) if diffs else "无"))
    return 0


def cmd_revert(a) -> int:
    """把 stale 退回 succeeded，并以当前输入刷新指纹。

    用于两种情况：判定误报；或已确认结果在新输入下依然成立。
    不刷新指纹就直接改回 succeeded，下次 check 会立刻再标一次。
    """
    conn = db.connect(a.db)
    try:
        db.init_db(conn)
        run = conn.execute("SELECT * FROM run WHERE id=?", (a.run_id,)).fetchone()
        if not run:
            raise SystemExit(f"run 不存在: {a.run_id}")
        run = dict(run)
        if run["status"] != "stale":
            raise SystemExit(f"{a.run_id} 当前是 {run['status']}，只有 stale 可 revert")
        cur = current_parts(conn, run, include_code=True)
        h, parts = db.input_fingerprint(conn, cur.get("dataset_id") or run.get("dataset_id"),
                                        cur.get("model_id") or run.get("model_id"),
                                        cur.get("code"))
        note = run.get("note") or ""
        note = note.replace("[stale]", "[stale→已确认]")
        if a.note:
            note = f"{note} | {a.note}" if note else a.note
        conn.execute("UPDATE run SET status='succeeded', input_hash=?, input_breakdown=?, "
                     "note=? WHERE id=?", (h, db.jdump(parts), note, a.run_id))
        db.audit(conn, "ops_stale", "run.revert", "run", a.run_id,
                 {"status": "stale"}, {"status": "succeeded", "input_hash": h})
        conn.commit()
    finally:
        conn.close()
    print(f"{a.run_id} → succeeded，指纹已按当前输入刷新（{str(h)[:12]}）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="stale 判定（输入指纹比对）")
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check")
    c.add_argument("--dry-run", action="store_true")
    c.add_argument("--include-code", action="store_true",
                   help="把执行脚本 sha1 纳入判定（默认只比数据与权重）")
    c.add_argument("--kind", default="", help="逗号分隔，如 infer,train；空=全部")

    s = sub.add_parser("show")
    s.add_argument("run_id")

    r = sub.add_parser("revert", help="stale → succeeded，并按当前输入刷新指纹")
    r.add_argument("run_id")
    r.add_argument("--note", default=None)

    a = ap.parse_args()
    return {"check": cmd_check, "show": cmd_show, "revert": cmd_revert}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
