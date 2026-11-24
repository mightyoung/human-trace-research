#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集登记：把「一份数据」变成台账里的一个可引用对象

为什么需要
----
`run` 记录里已经有 `dataset_id`，但 `dataset` 表长期只有 3 条、其中 2 条没有
`snapshot_hash`。没有 hash，`run` 就无法回答"我用的到底是哪一版数据"——
这正是 SCRFD 坐标变更后历史产物不可混用的同一类问题，只是发生在数据侧。

做什么
----
扫目录 → 算 `snapshot_hash` → 写 `dataset` + 一份 manifest 清单。

`snapshot_hash` = sha1(按路径排序的 `rel|size|mtime_ns` 行)。取 mtime 是故意的：
文件被重新编码、覆盖、替换都会变。只想比内容规模时用 `--no-mtime`。

用法
----
    # 登记 / 刷新一份数据
    python tools/ops_dataset.py scan --id ds_6cam_synced10 \
        --name "6 机位同步 10min（08:55–09:05）" \
        --root media-resource/multi-cemra/synced-10min --dialect clean2

    # 只看会变成什么，不落库
    python tools/ops_dataset.py scan --id x --name X --root <dir> --dry-run

    # 列出 / 查看
    python tools/ops_dataset.py ls
    python tools/ops_dataset.py show ds_6cam_synced10

约定
----
* 只登记指针，不搬文件；manifest 存 `ops/manifests/<id>.jsonl`（rel/size/mtime）
* 同 id 重复 scan 幂等：hash 不变则只刷新统计，不改语义
* 目录不存在直接报错，不静默建空记录（空 dataset 比没有更危险）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".ipynb_checkpoints"}
SKIP_EXT = {".tmp", ".part", ".crdownload", ".swp"}


def walk_files(root: str, max_files: int = 200000) -> list[tuple[str, int, int]]:
    """返回 [(rel_path, size, mtime_ns)]，按路径排序。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() in SKIP_EXT:
                continue
            p = os.path.join(dirpath, fn)
            try:
                st = os.stat(p)
            except OSError:
                continue
            rel = os.path.relpath(p, root).replace("\\", "/")
            out.append((rel, st.st_size, st.st_mtime_ns))
            if len(out) > max_files:
                raise SystemExit(f"文件数超过 {max_files}，疑似扫错目录：{root}")
    out.sort(key=lambda x: x[0])
    return out


def snapshot_hash(files: list[tuple[str, int, int]], use_mtime: bool = True) -> str:
    h = hashlib.sha1()
    for rel, size, mtime in files:
        line = f"{rel}|{size}|{mtime}" if use_mtime else f"{rel}|{size}"
        h.update(line.encode("utf-8") + b"\n")
    return h.hexdigest()


def write_manifest(ds_id: str, files: list[tuple[str, int, int]]) -> str:
    d = os.path.join(db.OPS_DIR, "manifests")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{ds_id}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rel, size, mtime in files:
            f.write(json.dumps({"rel": rel, "bytes": size, "mtime_ns": mtime},
                               ensure_ascii=False) + "\n")
    return path


def cmd_scan(a) -> int:
    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        raise SystemExit(f"目录不存在: {root}")
    files = walk_files(root, a.max_files)
    if not files:
        raise SystemExit(f"目录为空，拒绝登记空数据集: {root}")

    snap = snapshot_hash(files, use_mtime=not a.no_mtime)
    alg = "rel-size-v1" if a.no_mtime else "rel-size-mtime-v1"
    total = sum(s for _, s, _ in files)

    conn = db.connect(a.db)
    try:
        db.init_db(conn)
        prev = conn.execute("SELECT snapshot_hash, hash_alg, file_count, total_bytes "
                            "FROM dataset WHERE id=?", (a.id,)).fetchone()
        # 已有 hash 但算法不同（含手填的 legacy）→ 覆盖会让引用它的 run 失去可比性
        if prev and prev["snapshot_hash"] and prev["hash_alg"] != alg and not a.force:
            raise SystemExit(
                f"{a.id} 已有 snapshot_hash（算法 {prev['hash_alg'] or 'legacy/手填'}），"
                f"本次要写入 {alg}。\n换算法后与历史 run 不可比，确认换代际请加 --force。")
        changed = (not prev) or prev["snapshot_hash"] != snap
        if a.dry_run:
            print(f"[演练] {a.id}: {len(files)} 文件 / {total/1e9:.2f} GB / {alg} {snap[:12]}")
            if prev:
                print(f"       原 {prev['hash_alg'] or 'legacy'} {str(prev['snapshot_hash'])[:12]}"
                      f" → {'变化' if changed else '未变'}")
            return 0

        manifest = write_manifest(a.id, files)
        db.upsert_dataset(conn, a.id, a.name, kind=a.kind,
                          root_uri=db.rel(root), file_count=len(files), total_bytes=total,
                          dialect=a.dialect, manifest_uri=db.rel(manifest),
                          snapshot_hash=snap, note=a.note)
        conn.execute("UPDATE dataset SET hash_alg=? WHERE id=?", (alg, a.id))
        db.audit(conn, "ops_dataset", "dataset.scan", "dataset", a.id,
                 {"hash": prev["snapshot_hash"] if prev else None,
                  "alg": prev["hash_alg"] if prev else None},
                 {"hash": snap, "alg": alg, "files": len(files), "bytes": total})
        conn.commit()
    finally:
        conn.close()

    print(f"{'更新' if prev else '登记'} {a.id}")
    print(f"  文件 {len(files)} · {total/1e9:.2f} GB · dialect {a.dialect or '—'}")
    print(f"  {alg} {snap[:16]}  {'（内容有变）' if changed and prev else ''}")
    print(f"  manifest {db.rel(manifest)}")
    return 0


def cmd_ls(a) -> int:
    conn = db.connect(a.db)
    try:
        rows = conn.execute(
            "SELECT id, name, dialect, file_count, total_bytes, snapshot_hash, root_uri "
            "FROM dataset ORDER BY created_at DESC").fetchall()
    finally:
        conn.close()
    if not rows:
        print("（空）")
        return 0
    print(f"{'id':<24}{'文件':>7}{'大小':>10}  {'hash':<18}{'dialect':<10}name")
    for r in rows:
        h = (r["snapshot_hash"] or "")[:16] or "—"
        print(f"{r['id']:<24}{r['file_count'] or 0:>7}{(r['total_bytes'] or 0)/1e9:>9.2f}G  "
              f"{h:<18}{r['dialect'] or '—':<10}{r['name']}")
    return 0


def cmd_show(a) -> int:
    conn = db.connect(a.db)
    try:
        r = conn.execute("SELECT * FROM dataset WHERE id=?", (a.id,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise SystemExit(f"dataset 不存在: {a.id}")
    for k in r.keys():
        print(f"  {k:<14}{r[k]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集登记（扫描 + snapshot_hash）")
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--id", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--kind", default="video_set", choices=["video_set", "crop_set", "mixed"])
    s.add_argument("--dialect", default=None, help="clean2 / raw / e21-joined ...")
    s.add_argument("--note", default=None)
    s.add_argument("--no-mtime", action="store_true", help="hash 只算路径+大小（忽略 mtime）")
    s.add_argument("--max-files", type=int, default=200000)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true",
                   help="已有 hash 且算法不同时仍覆盖（会让历史 run 不可比）")

    sub.add_parser("ls")
    sh = sub.add_parser("show")
    sh.add_argument("id")

    a = ap.parse_args()
    return {"scan": cmd_scan, "ls": cmd_ls, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
