#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run 包（bundle）· 远程台账 → 本机主库的单条搬运

为什么需要
----
「本机为主库，远程只做算力」意味着**远程也会写自己的 ops.db**：训练在那边跑，
`ops_run` 在那边记账，曲线也在那边。跑完要把这一条 run 搬回本机主库，
而不是把整个 ops.db 拷回来（会撞 id、也会把远程的临时库变成第二权威）。

做法
----
导出的是**一条 run 的完整闭包**：run + run_metric + artifact + 所属 experiment
+ 引用的 dataset / model。导入按主键 upsert，重复导入同一条不会膨胀。

用法
----
    # 远程：导出一条 run（跑完再导，running 状态会被拒绝，除非 --force）
    python tools/ops_bundle.py export --run run_20261109_143000_a1b2c3 --out /tmp/run.json

    # 本机：导入（路径改写：远程 out_dir → 本机对应目录）
    python tools/ops_bundle.py import /tmp/run.json \
        --rewrite-out "intent_v3/output=output_crosscam/analysis/intent/v3/output"

    # 只看包里有什么，不落库
    python tools/ops_bundle.py inspect /tmp/run.json

约定
----
* 只搬指针，不搬大文件（artifact 仍只有 uri + bytes + sha1）
* 路径改写只影响文本字段，不改动任何文件
* 导入不删本机已有 run；同 id 覆盖，幂等
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402

BUNDLE_VERSION = 1

# 按外键依赖排序：先被引用的在前
UPSERT_ORDER = ["dataset", "model", "experiment", "run", "artifact", "run_metric",
                "experiment_run"]
# 主键（None = 自增，需特殊处理）
PK = {
    "dataset": ["id"],
    "model": ["id"],
    "run": ["id"],
    "experiment": ["id"],
    "experiment_run": ["experiment_id", "run_id"],
    "run_metric": ["run_id", "step", "name"],
    "artifact": None,
}


# ------------------------------------------------------------------ 工具
def table_cols(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def table_exists(conn, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def rewrite(value, pairs: list[tuple[str, str]]):
    """对字符串做前缀/子串替换；非字符串原样返回。"""
    if not isinstance(value, str) or not pairs:
        return value
    for old, new in pairs:
        if old and old in value:
            value = value.replace(old, new)
    return value


def parse_rewrites(items: list[str] | None, flag: str) -> list[tuple[str, str]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{flag} 需要 OLD=NEW 形式，收到: {it}")
        old, new = it.split("=", 1)
        out.append((old, new))
    return out


# ------------------------------------------------------------------ 导出
def export_bundle(conn, run_id: str, force: bool = False) -> dict:
    row = conn.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise SystemExit(f"run 不存在: {run_id}")
    run = dict(row)
    if run["status"] == "running" and not force:
        raise SystemExit(f"run 仍在 running（{run_id}）；跑完再导出，或加 --force")

    b = {
        "bundle_version": BUNDLE_VERSION,
        "exported_at": db.now(),
        "source": {"db": db.rel(db.DB_PATH), "host": run.get("host"),
                   "root": db.rel(db.ROOT)},
        "run": run,
        "metrics": [], "artifacts": [], "experiments": [],
        "dataset": None, "model": None,
    }

    if table_exists(conn, "run_metric"):
        b["metrics"] = [dict(r) for r in conn.execute(
            "SELECT * FROM run_metric WHERE run_id=? ORDER BY step, name", (run_id,))]

    if table_exists(conn, "artifact"):
        b["artifacts"] = [dict(r) for r in conn.execute(
            "SELECT * FROM artifact WHERE run_id=? ORDER BY id", (run_id,))]

    if run.get("dataset_id"):
        r = conn.execute("SELECT * FROM dataset WHERE id=?", (run["dataset_id"],)).fetchone()
        if r:
            b["dataset"] = dict(r)

    if run.get("model_id"):
        r = conn.execute("SELECT * FROM model WHERE id=?", (run["model_id"],)).fetchone()
        if r:
            b["model"] = dict(r)

    if table_exists(conn, "experiment_run"):
        for link in conn.execute("SELECT * FROM experiment_run WHERE run_id=?", (run_id,)):
            e = conn.execute("SELECT * FROM experiment WHERE id=?",
                             (link["experiment_id"],)).fetchone()
            if e:
                b["experiments"].append({"experiment": dict(e), "role": link["role"]})
    return b


# ------------------------------------------------------------------ 导入
def upsert(conn, table: str, row: dict, dry: bool = False) -> str:
    """按主键 upsert 一行；返回 'insert' / 'update' / 'skip'。"""
    cols = [c for c in table_cols(conn, table) if c in row]
    if not cols:
        return "skip"
    pk = PK.get(table)
    if pk is None:                                  # 自增表：调用方已处理
        return "skip"

    where = " AND ".join(f"{k}=?" for k in pk)
    exist = conn.execute(f"SELECT 1 FROM {table} WHERE {where}",
                         tuple(row[k] for k in pk)).fetchone()
    if dry:
        return "update" if exist else "insert"

    sets = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in pk)
    sql = (f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
           f"ON CONFLICT({','.join(pk)}) DO " +
           (f"UPDATE SET {sets}" if sets else "NOTHING"))
    conn.execute(sql, tuple(row[c] for c in cols))
    return "update" if exist else "insert"


def import_bundle(conn, b: dict, dry: bool = False,
                  out_pairs: list[tuple[str, str]] | None = None,
                  uri_pairs: list[tuple[str, str]] | None = None) -> dict:
    if b.get("bundle_version") != BUNDLE_VERSION:
        raise SystemExit(f"bundle 版本不符: {b.get('bundle_version')} != {BUNDLE_VERSION}")
    stats: dict[str, dict[str, int]] = {}

    def bump(t: str, verb: str) -> None:
        stats.setdefault(t, {"insert": 0, "update": 0, "skip": 0})[verb] += 1

    run = dict(b["run"])
    run["out_dir"] = rewrite(run.get("out_dir"), out_pairs or [])
    run["log_uri"] = rewrite(run.get("log_uri"), uri_pairs or [])

    if b.get("dataset"):
        bump("dataset", upsert(conn, "dataset", b["dataset"], dry))
    if b.get("model"):
        m = dict(b["model"])
        m["weights_uri"] = rewrite(m.get("weights_uri"), uri_pairs or [])
        bump("model", upsert(conn, "model", m, dry))

    for item in b.get("experiments") or []:
        bump("experiment", upsert(conn, "experiment", item["experiment"], dry))

    bump("run", upsert(conn, "run", run, dry))

    # artifact 是自增主键：整条 run 的产物按快照替换，避免重复导入翻倍
    arts = []
    for a in b.get("artifacts") or []:
        a = dict(a)
        a.pop("id", None)
        a["uri"] = rewrite(a.get("uri"), uri_pairs or [])
        arts.append(a)
    if not dry:
        conn.execute("DELETE FROM artifact WHERE run_id=?", (run["id"],))
    for a in arts:
        if not dry:
            cols = [c for c in table_cols(conn, "artifact") if c in a and c != "id"]
            conn.execute(f"INSERT INTO artifact({','.join(cols)}) "
                         f"VALUES({','.join('?' * len(cols))})",
                         tuple(a[c] for c in cols))
    bump("artifact", "insert" if not dry else "skip")
    stats["artifact"] = {"insert": len(arts), "update": 0, "skip": 0}

    for m in b.get("metrics") or []:
        bump("run_metric", upsert(conn, "run_metric", m, dry))

    for item in b.get("experiments") or []:
        link = {"experiment_id": item["experiment"]["id"], "run_id": run["id"],
                "role": item.get("role") or "variant"}
        bump("experiment_run", upsert(conn, "experiment_run", link, dry))

    if not dry:
        db.audit(conn, "ops_bundle", "run.import", "run", run["id"],
                 None, {"from": b.get("source", {}), "n_metric": len(b.get("metrics") or []),
                        "n_artifact": len(arts)})
        conn.commit()
    return stats


# ------------------------------------------------------------------ CLI
def main() -> int:
    ap = argparse.ArgumentParser(description="run 包导出/导入（远程 → 本机主库）")
    ap.add_argument("--db", default=None, help="台账路径（默认 ops/ops.db）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="导出一条 run 的完整闭包")
    e.add_argument("--run", required=True)
    e.add_argument("--out", default=None, help="输出 json（默认打印到 stdout）")
    e.add_argument("--force", action="store_true", help="允许导出 running 状态的 run")

    i = sub.add_parser("import", help="导入 bundle 到本机台账")
    i.add_argument("file")
    i.add_argument("--rewrite-out", action="append", metavar="OLD=NEW",
                   help="改写 out_dir 前缀（可多次）")
    i.add_argument("--rewrite-uri", action="append", metavar="OLD=NEW",
                   help="改写 artifact / log 的 uri 前缀（可多次）")
    i.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("inspect", help="查看 bundle 内容，不落库")
    s.add_argument("file")

    a = ap.parse_args()
    db_path = a.db or db.DB_PATH

    if a.cmd == "export":
        conn = db.connect(db_path)
        try:
            b = export_bundle(conn, a.run, force=a.force)
        finally:
            conn.close()
        text = json.dumps(b, ensure_ascii=False, indent=1)
        if a.out:
            out_path = a.out
            # 允许直接给目录：自动命名为 <run_id>.json（批量导出时省事）
            if os.path.isdir(out_path) or out_path.endswith(("/", os.sep)):
                os.makedirs(out_path, exist_ok=True)
                out_path = os.path.join(out_path, f"{a.run}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"导出 {a.run} → {out_path}")
            print(f"  run_metric {len(b['metrics'])} · artifact {len(b['artifacts'])} "
                  f"· experiment {len(b['experiments'])}")
        else:
            print(text)
        return 0

    if a.cmd == "inspect":
        with open(a.file, encoding="utf-8") as f:
            b = json.load(f)
        r = b.get("run", {})
        print(f"run        {r.get('id')}  [{r.get('status')}]  {r.get('name')}")
        print(f"kind       {r.get('kind')}   entry {r.get('entrypoint')}")
        print(f"时间段     {r.get('started_at')} → {r.get('ended_at')}")
        print(f"out_dir    {r.get('out_dir')}")
        print(f"指标       {len(b.get('metrics') or [])} 点"
              f"  名称 {sorted({m['name'] for m in (b.get('metrics') or [])})}")
        print(f"产物       {len(b.get('artifacts') or [])} 个")
        print(f"实验       {[x['experiment']['name'] for x in (b.get('experiments') or [])]}")
        print(f"dataset    {(b.get('dataset') or {}).get('id')}   "
              f"model {(b.get('model') or {}).get('id')}")
        print(f"来源       {b.get('source')}")
        return 0

    with open(a.file, encoding="utf-8") as f:
        b = json.load(f)
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        st = import_bundle(conn, b, dry=a.dry_run,
                           out_pairs=parse_rewrites(a.rewrite_out, "--rewrite-out"),
                           uri_pairs=parse_rewrites(a.rewrite_uri, "--rewrite-uri"))
    finally:
        conn.close()

    print(("[演练] " if a.dry_run else "") + f"导入 {b['run']['id']}")
    for t in UPSERT_ORDER:
        if t in st:
            print(f"  {t:<15} insert {st[t]['insert']:>3}  update {st[t]['update']:>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
