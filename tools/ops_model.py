#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型注册：给权重一个台账身份

为什么需要
----
`model` 表长期 0 条，而 `models/` 下有 8 个权重文件、训练也产出过 best.pt。
结果是任何指标都无法回答"这是哪个权重的结果"——`run → model → artifact`
这条链缺中间一环，设计文档 P2 的验收标准（任取一个指标能点到模型 sha）走不通。

做什么
----
* `register`：登记磁盘上的权重（算 sha1 + bytes）
* `from-run`：把某条训练 run 的产物提升为模型版本（`trained_by_run` 指回 run）
* `ls` / `show`

用法
----
    python tools/ops_model.py register \
        --path models/face/scrfd_2.5g_bnkps.onnx \
        --name scrfd-2.5g --backbone SCRFD --license InsightFace

    python tools/ops_model.py from-run run_20261109_115354_36cfa3 --name intent-cc-e2e

    python tools/ops_model.py ls

约定
----
* `id = model_<slug>_<sha1[:6]>`，`version = sha1[:8]`
* 同一文件重复登记幂等（同 id）
* `deployable` 默认 0：注册 ≠ 可上线，上线要单独标
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402

WEIGHT_EXT = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".engine"}


def sha1_file(path: str, chunk: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha1()
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            size += len(b)
    return h.hexdigest(), size


def slug(s: str) -> str:
    keep = []
    for ch in s.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in "-_.":
            keep.append("_")
    return "".join(keep).strip("_") or "model"


def register(conn, path: str, name: str, backbone: str | None = None,
             license_: str | None = None, deployable: int = 0,
             note: str | None = None) -> dict:
    ap = os.path.abspath(path)
    if not os.path.isfile(ap):
        raise SystemExit(f"权重文件不存在: {path}")
    if os.path.splitext(ap)[1].lower() not in WEIGHT_EXT:
        raise SystemExit(f"不认识的权重后缀: {os.path.splitext(ap)[1]}（支持 {sorted(WEIGHT_EXT)}）")

    sha, size = sha1_file(ap)
    mid = f"model_{slug(name)}_{sha[:6]}"
    row = conn.execute("SELECT * FROM model WHERE id=?", (mid,)).fetchone()
    conn.execute(
        "INSERT INTO model(id, name, version, backbone, weights_uri, sha1, license, "
        "trained_by_run, eval_json, deployable, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET name=excluded.name, version=excluded.version, "
        "backbone=excluded.backbone, weights_uri=excluded.weights_uri, sha1=excluded.sha1, "
        "license=excluded.license, deployable=excluded.deployable",
        (mid, name, sha[:8], backbone, db.rel(ap), sha, license_, None, None,
         deployable, db.now()))
    db.audit(conn, "ops_model", "model.register" if not row else "model.update",
             "model", mid, None, {"path": db.rel(ap), "sha1": sha, "bytes": size})
    return {"id": mid, "sha1": sha, "bytes": size, "existed": bool(row), "uri": db.rel(ap)}


def from_run(conn, run_id: str, name: str | None = None,
             backbone: str | None = None, deployable: int = 0) -> dict:
    run = conn.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
    if not run:
        raise SystemExit(f"run 不存在: {run_id}")
    arts = [dict(r) for r in conn.execute(
        "SELECT uri, bytes, sha1, kind FROM artifact WHERE run_id=? ORDER BY id", (run_id,))]
    cand = [a for a in arts if os.path.splitext(a["uri"])[1].lower() in WEIGHT_EXT]
    if not cand:
        raise SystemExit(f"{run_id} 没有权重类产物（{sorted(WEIGHT_EXT)}）；"
                         f"现有产物: {[a['uri'] for a in arts] or '无'}")
    a = cand[0]
    mname = name or run["name"] or "model"
    sha = a["sha1"] or ""
    mid = f"model_{slug(mname)}_{sha[:6] or db.new_id('x')[-6:]}"
    conn.execute(
        "INSERT INTO model(id, name, version, backbone, weights_uri, sha1, license, "
        "trained_by_run, eval_json, deployable, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET weights_uri=excluded.weights_uri, "
        "trained_by_run=excluded.trained_by_run, eval_json=excluded.eval_json, "
        "deployable=excluded.deployable",
        (mid, mname, (sha[:8] or None), backbone, a["uri"], sha, None, run_id,
         run["metrics_json"], deployable, db.now()))
    # 反向挂到 run 上，形成 run.model_id ←→ model.trained_by_run
    conn.execute("UPDATE run SET model_id=? WHERE id=? AND model_id IS NULL", (mid, run_id))
    db.audit(conn, "ops_model", "model.from_run", "model", mid, None,
             {"run_id": run_id, "uri": a["uri"], "sha1": sha})
    return {"id": mid, "run_id": run_id, "uri": a["uri"], "sha1": sha, "bytes": a["bytes"]}


def cmd_register(a) -> int:
    conn = db.connect(a.db)
    try:
        db.init_db(conn)
        r = register(conn, a.path, a.name, a.backbone, a.license, a.deployable)
        conn.commit()
    finally:
        conn.close()
    print(f"{'更新' if r['existed'] else '登记'} {r['id']}")
    print(f"  {r['uri']}  {r['bytes']/1e6:.1f} MB  sha1 {r['sha1'][:16]}")
    return 0


def cmd_from_run(a) -> int:
    conn = db.connect(a.db)
    try:
        db.init_db(conn)
        r = from_run(conn, a.run_id, a.name, a.backbone, a.deployable)
        conn.commit()
    finally:
        conn.close()
    print(f"登记 {r['id']}  （来自 run {r['run_id']}）")
    print(f"  {r['uri']}  sha1 {str(r['sha1'])[:16]}")
    return 0


def cmd_ls(a) -> int:
    conn = db.connect(a.db)
    try:
        rows = conn.execute(
            "SELECT id, name, version, backbone, sha1, deployable, trained_by_run, weights_uri "
            "FROM model ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()
    if not rows:
        print("（空）")
        return 0
    print(f"{'id':<38}{'ver':<10}{'deploy':>7}  {'backbone':<12}name")
    for r in rows:
        print(f"{r['id']:<38}{r['version'] or '—':<10}{r['deployable'] or 0:>7}  "
              f"{r['backbone'] or '—':<12}{r['name']}")
        print(f"    {r['weights_uri']}   trained_by={r['trained_by_run'] or '—'}")
    return 0


def cmd_show(a) -> int:
    conn = db.connect(a.db)
    try:
        r = conn.execute("SELECT * FROM model WHERE id=?", (a.id,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise SystemExit(f"model 不存在: {a.id}")
    for k in r.keys():
        v = r[k]
        if k == "eval_json" and v:
            v = f"{len(v)} chars"
        print(f"  {k:<14}{v}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="模型注册（磁盘权重 / 训练产物 → model 表）")
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--path", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--backbone", default=None)
    r.add_argument("--license", default=None)
    r.add_argument("--deployable", type=int, default=0)

    f = sub.add_parser("from-run")
    f.add_argument("run_id")
    f.add_argument("--name", default=None)
    f.add_argument("--backbone", default=None)
    f.add_argument("--deployable", type=int, default=0)

    sub.add_parser("ls")
    s = sub.add_parser("show")
    s.add_argument("id")

    a = ap.parse_args()
    return {"register": cmd_register, "from-run": cmd_from_run,
            "ls": cmd_ls, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
