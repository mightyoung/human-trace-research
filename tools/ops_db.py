#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""算法中台台账 · 访问层（SQLite，零依赖）

定位
----
本机为主库：`ops/ops.db` 是唯一写入点。远程只做算力与预览，不持有权威数据。
大文件不搬：artifact 只登记 uri + bytes + sha1。

用法（CLI，便于排查）
----
    python tools/ops_db.py init                 建库/迁移（幂等）
    python tools/ops_db.py overview             打印总览
    python tools/ops_db.py sql "select ..."     只读查询（拒绝写语句）

被 ops_run.py / ops_ingest.py / webapp/backend/ops_api.py 复用。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from typing import Any, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPS_DIR = os.path.join(ROOT, "ops")
DB_PATH = os.path.join(OPS_DIR, "ops.db")
SCHEMA_PATH = os.path.join(OPS_DIR, "schema.sql")

SCHEMA_VERSION = "1"


# ------------------------------------------------------------------ 基础工具
def now() -> str:
    """统一时间口径：本地时区，与项目既有 json 一致。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def sha1_file(path: str, chunk: int = 1 << 20) -> str | None:
    """文件 sha1；不存在返回 None（不抛，产物可能已被清理）。"""
    if not os.path.isfile(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def jload(s: str | None, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def rel(path: str) -> str:
    """登记用相对路径，避免绝对路径进库（换机器/换盘符即失效）。"""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except Exception:
        return path.replace("\\", "/")


# ------------------------------------------------------------------ 连接
def connect(db_path: str | None = None, create: bool = True) -> sqlite3.Connection:
    path = db_path or DB_PATH
    if create:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def init_db(conn: sqlite3.Connection) -> dict:
    """建表/迁移，幂等。返回各表行数。"""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    # 迁移：v1 早期的唯一索引不含 value，导致同一秒内改判被拒
    conn.execute("DROP INDEX IF EXISTS ux_label_once")
    _migrate_add_columns(conn)
    conn.execute("INSERT INTO meta(k, v) VALUES('schema_version', ?) "
                 "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (SCHEMA_VERSION,))
    conn.execute("INSERT INTO meta(k, v) VALUES('authority', 'local') "
                 "ON CONFLICT(k) DO UPDATE SET v=excluded.v")
    return table_counts(conn)


# 后加的列：老库用 ALTER 补，避免重建表
ADDED_COLUMNS = [
    ("dataset", "hash_alg", "TEXT"),      # snapshot_hash 用哪种算法算的
    ("run", "input_hash", "TEXT"),        # 输入指纹（dataset+model+code），stale 判定用
    ("run", "input_breakdown", "TEXT"),   # 指纹各分量，出问题时能说清哪个变了
    ("label", "data_json", "TEXT"),       # 原始判定对象（证据），value 只保主值
    # paper_claim.kind 已于 2026-11-24 迁入通用 claim 表（paper_claim 现为视图），不再需要 ALTER
    ("deliverable_ref", "anchor_id", "TEXT"),  # K0/Q12：能挂锚点就挂，C4 分「锚定/文本」两列
    ("claim", "card_state", "TEXT"),           # K0/Q2：NULL=待归属，'n/a'=豁免（说明/日期类）
    ("run", "card_id", "TEXT"),                # 执行准入：该 run 挂在哪张 exp_card 下
    ("run", "card_role", "TEXT"),              # baseline|variant|ablation|explore
]


def _migrate_add_columns(conn: sqlite3.Connection) -> None:
    for table, col, typ in ADDED_COLUMNS:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (table,)).fetchone():
            continue
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def table_counts(conn: sqlite3.Connection) -> dict:
    out = {}
    for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "AND name NOT IN ('meta') ORDER BY name"):
        out[name] = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    return out


# ------------------------------------------------------------------ 审计
def audit(conn: sqlite3.Connection, actor: str, action: str, obj_type: str,
          obj_id: str, before: Any = None, after: Any = None) -> None:
    conn.execute(
        "INSERT INTO audit(actor, action, object_type, object_id, before_json, after_json, at) "
        "VALUES(?,?,?,?,?,?,?)",
        (actor, action, obj_type, str(obj_id),
         jdump(before) if before is not None else None,
         jdump(after) if after is not None else None, now()))


# ------------------------------------------------------------------ 数据集
def upsert_dataset(conn: sqlite3.Connection, ds_id: str, name: str, **kw) -> str:
    """登记/更新数据集。

    只更新**显式传入且非 None** 的字段：`ops_run --dataset` 的自动登记只带
    root/dialect，不能把 `ops_dataset scan` 算出的 snapshot_hash 和人工起的名字抹掉。
    """
    cur = conn.execute("SELECT id FROM dataset WHERE id=?", (ds_id,)).fetchone()
    cand = dict(kind=kw.get("kind"), root_uri=kw.get("root_uri"),
                file_count=kw.get("file_count"), total_bytes=kw.get("total_bytes"),
                dialect=kw.get("dialect"), manifest_uri=kw.get("manifest_uri"),
                snapshot_hash=kw.get("snapshot_hash"), hash_alg=kw.get("hash_alg"),
                note=kw.get("note"))
    fields = {k: v for k, v in cand.items() if v is not None}
    if cur:
        if not fields:
            return ds_id
        sets = ", ".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE dataset SET {sets} WHERE id=?", (*fields.values(), ds_id))
    else:
        conn.execute(
            "INSERT INTO dataset(id, name, kind, root_uri, file_count, total_bytes, dialect, "
            "manifest_uri, snapshot_hash, hash_alg, created_at, note) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (ds_id, name, fields.get("kind"), fields.get("root_uri"),
             fields.get("file_count", 0), fields.get("total_bytes", 0),
             fields.get("dialect"), fields.get("manifest_uri"),
             fields.get("snapshot_hash"), fields.get("hash_alg"), now(),
             fields.get("note")))
    return ds_id


# ------------------------------------------------------------------ 标注
def upsert_task(conn: sqlite3.Connection, task_id: str, name: str, kind: str, **kw) -> str:
    cur = conn.execute("SELECT id FROM annotation_task WHERE id=?", (task_id,)).fetchone()
    if cur:
        conn.execute("UPDATE annotation_task SET name=?, kind=?, dataset_id=COALESCE(?,dataset_id), "
                     "dialect=COALESCE(?,dialect), mode=COALESCE(?,mode), note=COALESCE(?,note) "
                     "WHERE id=?",
                     (name, kind, kw.get("dataset_id"), kw.get("dialect"),
                      kw.get("mode"), kw.get("note"), task_id))
    else:
        conn.execute(
            "INSERT INTO annotation_task(id, name, kind, dataset_id, spec_json, dialect, mode, "
            "assignee, budget, due_at, status, created_by, created_at, note) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, name, kind, kw.get("dataset_id"),
             jdump(kw.get("spec")) if kw.get("spec") else None,
             kw.get("dialect"), kw.get("mode", "single"), kw.get("assignee"),
             kw.get("budget"), kw.get("due_at"), kw.get("status", "open"),
             kw.get("created_by", "system"), now(), kw.get("note")))
    return task_id


def upsert_item(conn: sqlite3.Connection, item_id: str, task_id: str, **kw) -> str:
    """插入或更新条目。已有 label 时**不改 status**（避免把已判的打回 pending）。"""
    cur = conn.execute("SELECT id, status FROM annotation_item WHERE id=?", (item_id,)).fetchone()
    payload = jdump(kw.get("payload")) if kw.get("payload") is not None else None
    refs = jdump(kw.get("asset_refs")) if kw.get("asset_refs") is not None else None
    if cur:
        conn.execute("UPDATE annotation_item SET payload_json=COALESCE(?,payload_json), "
                     "asset_refs_json=COALESCE(?,asset_refs_json), "
                     "stratum=COALESCE(?,stratum), priority=COALESCE(?,priority) WHERE id=?",
                     (payload, refs, kw.get("stratum"), kw.get("priority"), item_id))
    else:
        conn.execute(
            "INSERT INTO annotation_item(id, task_id, asset_refs_json, payload_json, stratum, "
            "priority, status, gold_value, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (item_id, task_id, refs, payload, kw.get("stratum", "topk"),
             kw.get("priority", 0), kw.get("status", "pending"),
             kw.get("gold_value"), now()))
    return item_id


def latest_label(conn: sqlite3.Connection, item_id: str):
    return conn.execute(
        "SELECT * FROM label WHERE item_id=? ORDER BY id DESC LIMIT 1", (item_id,)).fetchone()


def set_label(conn: sqlite3.Connection, item_id: str, value: str, annotator: str,
              confidence: float | None = None, seconds: float | None = None,
              ui_version: str | None = None, note: str | None = None,
              at: str | None = None, data: Any = None) -> int:
    """追加一条判定（append-only）。同 annotator 同值重复提交不新增行。

    `data` 是**原始判定对象**（如 `{v,c,n}` / `{pid,fp}` / `{verdict,name}` / 多边形），
    落到 `label.data_json`。`value` 只承担"可跨任务比较的主值"，展示所需的置信度、
    自由备注、多字段并存都必须靠 `data` 才能无损往返（读切换的前置）。

    返回 label.id。改判 = 新行 + supersedes 指向上一行。
    同值但 data 有变化 → 不新增行，只补该行的 data_json（判定未变，仅证据补全）。
    """
    prev = latest_label(conn, item_id)
    if prev and prev["annotator"] == annotator and str(prev["value"]) == str(value):
        if data is not None:
            new = jdump(data)
            old = prev["data_json"] if "data_json" in prev.keys() else None
            if old != new:
                # 同值补证据：不产生新判定行，避免把「补证据」误记成「改判」
                conn.execute("UPDATE label SET data_json=? WHERE id=?", (new, int(prev["id"])))
                audit(conn, annotator, "label.data", "annotation_item", item_id,
                      {"data": jload(old)} if old else None, {"data": data})
        return int(prev["id"])
    lid = conn.execute(
        "INSERT INTO label(item_id, annotator, value, confidence, seconds, ui_version, "
        "created_at, supersedes, note, data_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (item_id, annotator, str(value), confidence, seconds, ui_version, at or now(),
         int(prev["id"]) if prev else None, note,
         jdump(data) if data is not None else None)).lastrowid
    conn.execute("UPDATE annotation_item SET status=? WHERE id=? AND status='pending'",
                 ("labeled", item_id))
    audit(conn, annotator, "label.set", "annotation_item", item_id,
          {"value": prev["value"]} if prev else None, {"value": value})
    return int(lid)


def label_data(row) -> Any:
    """从 label 行取回原始判定对象；老行无 data_json 时退回 value 的 JSON 解析。"""
    if row is None:
        return None
    d = jload(row["data_json"], "__missing__") if "data_json" in row.keys() else "__missing__"
    if d != "__missing__":
        return d
    v = row["value"]
    if isinstance(v, str) and v[:1] in ("[", "{"):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def task_progress(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT task_id, name, kind, items, done, pending FROM v_task_progress "
        "ORDER BY (CAST(done AS REAL)/NULLIF(items,0)) ASC, items DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["pct"] = round(100.0 * (d["done"] or 0) / d["items"], 1) if d["items"] else 0.0
        out.append(d)
    return out


# ------------------------------------------------------------------ 运行
def create_run(conn: sqlite3.Connection, name: str, kind: str, entrypoint: str | None = None,
               params: Any = None, env: Any = None, dataset_id: str | None = None,
               model_id: str | None = None, host: str | None = None, gpu: str | None = None,
               git_rev: str | None = None, out_dir: str | None = None,
               status: str = "running", note: str | None = None,
               run_id: str | None = None, card_id: str | None = None,
               card_role: str | None = None) -> str:
    rid = run_id or new_id("run")
    conn.execute(
        "INSERT INTO run(id, name, kind, entrypoint, params_json, env_json, dataset_id, model_id, "
        "host, gpu, git_rev, status, started_at, log_uri, out_dir, note, created_at, "
        "card_id, card_role) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rid, name, kind, entrypoint, jdump(params) if params is not None else None,
         jdump(env) if env is not None else None, dataset_id, model_id, host, gpu, git_rev,
         status, now(), None, rel(out_dir) if out_dir else None, note, now(),
         card_id, card_role))
    audit(conn, "ops_run", "run.create", "run", rid, None,
          {"name": name, "kind": kind, "card_id": card_id, "card_role": card_role})
    return rid


def finish_run(conn: sqlite3.Connection, run_id: str, status: str, exit_code: int | None,
               metrics: Any = None, log_uri: str | None = None) -> None:
    conn.execute("UPDATE run SET status=?, exit_code=?, ended_at=?, "
                 "metrics_json=COALESCE(?, metrics_json), log_uri=COALESCE(?, log_uri) WHERE id=?",
                 (status, exit_code, now(),
                  jdump(metrics) if metrics is not None else None, log_uri, run_id))
    audit(conn, "ops_run", f"run.{status}", "run", run_id, None, {"exit_code": exit_code})


def add_artifact(conn: sqlite3.Connection, run_id: str, path: str, kind: str | None = None,
                 summary: Any = None) -> int | None:
    """登记产物。同 run 同 uri 不重复登记（幂等）。"""
    uri = rel(path)
    got = conn.execute("SELECT id FROM artifact WHERE run_id=? AND uri=?", (run_id, uri)).fetchone()
    if got:
        return None
    size = os.path.getsize(path) if os.path.isfile(path) else None
    kind = kind or (os.path.splitext(path)[1].lstrip(".").lower() or "file")
    aid = conn.execute(
        "INSERT INTO artifact(run_id, kind, uri, bytes, sha1, summary_json, at) "
        "VALUES(?,?,?,?,?,?,?)",
        (run_id, kind, uri, size, sha1_file(path),
         jdump(summary) if summary is not None else None, now())).lastrowid
    return int(aid)


def list_runs(conn: sqlite3.Connection, limit: int = 200, kind: str | None = None,
              status: str | None = None) -> list[dict]:
    sql = ("SELECT r.*, (SELECT COUNT(*) FROM artifact a WHERE a.run_id=r.id) AS n_artifacts "
           "FROM run r WHERE 1=1")
    args: list = []
    if kind:
        sql += " AND r.kind=?"
        args.append(kind)
    if status:
        sql += " AND r.status=?"
        args.append(status)
    sql += " ORDER BY COALESCE(r.started_at, r.created_at) DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in conn.execute(sql, args):
        d = dict(r)
        d["params"] = jload(d.pop("params_json"), {})
        d["metrics"] = jload(d.pop("metrics_json"), None)
        d["env"] = jload(d.pop("env_json"), {})
        out.append(d)
    return out


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["params"] = jload(d.pop("params_json"), {})
    d["env"] = jload(d.pop("env_json"), {})
    d["metrics"] = normalize_metrics(jload(d.pop("metrics_json"), None))
    d["artifacts"] = [dict(x) for x in conn.execute(
        "SELECT * FROM artifact WHERE run_id=? ORDER BY id", (run_id,))]
    for a in d["artifacts"]:
        a["summary"] = jload(a.pop("summary_json"), None)
    d["series"] = get_series(conn, run_id)
    exp = conn.execute(
        "SELECT e.id, e.name FROM experiment_run er JOIN experiment e ON e.id=er.experiment_id "
        "WHERE er.run_id=?", (run_id,)).fetchone()
    d["experiment"] = dict(exp) if exp else None

    # 溯源要能直接看到「哪份数据、哪个权重」，而不只是一个 id
    if d.get("dataset_id"):
        ds = conn.execute("SELECT name, snapshot_hash, hash_alg, file_count FROM dataset "
                          "WHERE id=?", (d["dataset_id"],)).fetchone()
        d["dataset"] = dict(ds) if ds else None
    if d.get("model_id"):
        m = conn.execute("SELECT name, version, sha1, backbone FROM model WHERE id=?",
                         (d["model_id"],)).fetchone()
        d["model"] = dict(m) if m else None
    d["input_breakdown"] = jload(d.get("input_breakdown"), None)
    return d


def normalize_metrics(raw: Any) -> Any:
    """把各训练/评测脚本自己的 metrics.json 归一到 {acc, macro_f1, n, best_ep, ...}。

    已知形态：
      intent_model*.py      -> {"test": {...}, "val_best": {"ep","macro_f1"}, "hist": [...]}
      metrics_full.json     -> {"overall": {...}, "config": {...}, "per_cam": {...}}
      自定义                -> {"acc": ..., "macro_f1": ...}
    原始内容保留在 `_raw`，避免丢信息；已归一过的直接返回。
    """
    if not isinstance(raw, dict) or "_raw" in raw:
        return raw
    orig = raw
    # intent_model_cc / intent_model_social 的 metrics_full.json 会再包一层 rep：
    #   {"rep": {"config","overall","per_cam","per_known"}, "hist": [...], "args": {...}}
    rep = raw.get("rep")
    if isinstance(rep, dict) and any(k in rep for k in ("overall", "per_cam", "config", "per_known")):
        raw = rep

    src = None
    for k in ("test", "overall", "metrics", "eval"):
        v = raw.get(k)
        if isinstance(v, dict) and any(x in v for x in ("acc", "macro_f1")):
            src = v
            break
    if src is None and any(x in raw for x in ("acc", "macro_f1")):
        src = raw
    if src is None or src is raw:
        # 找不到已知结构，或本来就是标准形态（顶层就有 acc/macro_f1）→ 原样返回，不套 _raw
        return raw

    out: dict = {}
    for k in ("acc", "macro_f1", "n", "precision", "recall"):
        v = src.get(k)
        if isinstance(v, (int, float)):
            out[k] = v
    if "n" not in out and isinstance(src.get("support"), int):
        out["n"] = src["support"]
    if "n" not in out:
        cm = src.get("confusion")
        if isinstance(cm, list) and cm and isinstance(cm[0], list):
            try:
                out["n"] = int(sum(sum(row) for row in cm))
            except (TypeError, ValueError):
                pass

    vb = raw.get("val_best")
    if isinstance(vb, dict):
        if isinstance(vb.get("ep"), int):
            out["best_ep"] = vb["ep"]
        if isinstance(vb.get("macro_f1"), (int, float)):
            out["best_val_f1"] = vb["macro_f1"]

    per_cam = raw.get("per_cam")
    if isinstance(per_cam, dict):
        out["per_cam"] = {k: (v or {}).get("acc") if isinstance(v, dict) else v
                          for k, v in per_cam.items()}

    if not out:
        return raw
    out["_raw"] = orig
    return out


def set_metric(conn: sqlite3.Connection, run_id: str, step: int, name: str,
               value: float | None) -> None:
    """写一个序列点（幂等：同 run/step/name 覆盖）。"""
    conn.execute(
        "INSERT INTO run_metric(run_id, step, name, value) VALUES(?,?,?,?) "
        "ON CONFLICT(run_id, step, name) DO UPDATE SET value=excluded.value",
        (run_id, int(step), name, value))


def get_series(conn: sqlite3.Connection, run_id: str) -> dict:
    """返回 {名字: [{step, value}, ...]}；无数据返回空 dict。"""
    out: dict[str, list] = {}
    for r in conn.execute(
            "SELECT name, step, value FROM run_metric WHERE run_id=? ORDER BY name, step",
            (run_id,)):
        out.setdefault(r["name"], []).append({"step": r["step"], "value": r["value"]})
    return out


def train_overview(conn: sqlite3.Connection, limit: int = 60) -> list[dict]:
    """训练批次总览：按 experiment 分组，带变体数、最佳指标、最新时间。"""
    rows = conn.execute(
        "SELECT e.id, e.name, e.status, e.conclusion, "
        "       COUNT(er.run_id) n_runs, MAX(r.started_at) last_at "
        "FROM experiment e "
        "LEFT JOIN experiment_run er ON er.experiment_id=e.id "
        "LEFT JOIN run r ON r.id=er.run_id "
        "GROUP BY e.id ORDER BY last_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def input_fingerprint(conn: sqlite3.Connection, dataset_id: str | None = None,
                      model_id: str | None = None,
                      code_sha1: str | None = None) -> tuple[str | None, dict]:
    """算输入指纹 + 各分量。

    只把「会让结果失效」的东西纳入：数据快照、模型权重、执行脚本。
    超参不纳入——它由 run 自己记录，不会事后变化。

    任一分量缺失就不返回总 hash（返回 None）：拿不准就不要假装算得准。
    """
    parts: dict[str, Any] = {}
    if dataset_id:
        r = conn.execute("SELECT snapshot_hash, hash_alg FROM dataset WHERE id=?",
                         (dataset_id,)).fetchone()
        parts["dataset_id"] = dataset_id
        parts["dataset"] = r["snapshot_hash"] if r else None
        parts["dataset_alg"] = r["hash_alg"] if r else None
    if model_id:
        r = conn.execute("SELECT sha1 FROM model WHERE id=?", (model_id,)).fetchone()
        parts["model_id"] = model_id
        parts["model"] = r["sha1"] if r else None
    if code_sha1:
        parts["code"] = code_sha1

    core = [f"{k}={parts.get(k) or ''}" for k in ("dataset", "model", "code")]
    if not any(parts.get(k) for k in ("dataset", "model", "code")):
        return None, parts
    return hashlib.sha1("|".join(core).encode("utf-8")).hexdigest(), parts


def stale_runs(conn: sqlite3.Connection) -> list[dict]:
    """占位：P2 由输入指纹变化驱动。P0 返回空，接口先可用。"""
    return [dict(r) for r in conn.execute(
        "SELECT id, name, status, started_at FROM run WHERE status='stale' ORDER BY started_at DESC")]


# ------------------------------------------------------------------ 任务派发
JOB_FLOW = {
    "backlog": {"assigned", "blocked"},
    "assigned": {"doing", "backlog", "blocked"},
    "doing": {"review", "blocked", "backlog"},
    "review": {"done", "doing", "blocked"},
    "done": set(),
    "blocked": {"backlog", "assigned", "doing"},
}


def create_job(conn: sqlite3.Connection, *, title: str, kind: str = "label",
               goal: str | None = None, inputs: Any = None, acceptance: Any = None,
               assignee_kind: str = "human", assignee: str | None = None,
               priority: int = 0, budget: str | None = None,
               authorization: str | None = None, status: str = "backlog",
               job_id: str | None = None) -> str:
    jid = job_id or new_id("job")
    conn.execute(
        "INSERT INTO job(id, kind, title, goal, inputs_json, acceptance_json, assignee_kind, "
        "assignee, status, priority, budget, authorization, created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (jid, kind, title, goal,
         jdump(inputs) if inputs is not None else None,
         jdump(acceptance) if acceptance is not None else None,
         assignee_kind, assignee, status, priority, budget, authorization, now()))
    audit(conn, "console", "job.create", "job", jid, None, {"title": title, "kind": kind})
    return jid


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["inputs"] = jload(d.pop("inputs_json"), {})
    d["acceptance"] = jload(d.pop("acceptance_json"), [])
    d["evidence"] = jload(d.pop("evidence_json"), None)
    return d


def update_job_status(conn: sqlite3.Connection, job_id: str, status: str,
                      evidence: Any = None, note: str | None = None) -> dict:
    """状态流转。review→done 必须有证据（本项目「声明必须有证据」的硬约束）。"""
    cur = get_job(conn, job_id)
    if not cur:
        raise KeyError(f"job 不存在: {job_id}")
    allowed = JOB_FLOW.get(cur["status"], set())
    if status not in allowed:
        raise ValueError(f"非法流转 {cur['status']} → {status}；允许: {sorted(allowed) or '无'}")
    if status == "done":
        ev = evidence if evidence is not None else cur.get("evidence")
        # label 类卡片：关闭的前提是队列真的清零，而不是「我说做完了」
        if cur.get("kind") == "label":
            tid = (cur.get("inputs") or {}).get("task_id")
            t = label_progress(conn).get(tid) if tid else None
            if t and (t["pending"] or 0) > 0:
                raise ValueError(f"队列还有 {t['pending']} 条未判完，不能关闭"
                                 f"（{t['name']} · {tid}）")
            if t and not ev:
                # 证据由事实生成：队列归零 + 完成数，不让人手填
                ev = {"auto": "queue_drained", "task_id": tid, "name": t["name"],
                      "items": t["items"], "done": t["done"], "at": now()}
        if not ev:
            raise ValueError("关闭任务必须挂证据（文件路径 / run_id / 指标，至少一项）")
        conn.execute("UPDATE job SET status=?, evidence_json=?, closed_at=? WHERE id=?",
                     (status, jdump(ev), now(), job_id))
    else:
        if evidence is not None:
            conn.execute("UPDATE job SET status=?, evidence_json=? WHERE id=?",
                         (status, jdump(evidence), job_id))
        else:
            conn.execute("UPDATE job SET status=? WHERE id=?", (status, job_id))
    audit(conn, "console", f"job.{status}", "job", job_id,
          {"status": cur["status"]}, {"status": status, "note": note})
    return get_job(conn, job_id)


def derive_label_jobs(conn: sqlite3.Connection, min_pending: int = 1) -> list[str]:
    """按标注任务的待判量派生 label job（已存在未关闭的同类 job 则跳过）。

    卡片**不写死待判数**：数字由 list_jobs 的 `live` 实时给出，否则一周后就变成假话。
    """
    made = []
    for t in task_progress(conn):
        if (t["pending"] or 0) < min_pending:
            continue
        exists = conn.execute(
            "SELECT 1 FROM job WHERE kind='label' AND status!='done' AND inputs_json LIKE ?",
            (f'%"{t["task_id"]}"%',)).fetchone()
        if exists:
            continue
        jid = create_job(
            conn, title=f"判读 {t['name']}", kind="label",
            goal="清空该队列的待判条目（实时进度见卡片）",
            inputs={"task_id": t["task_id"], "pending_at_create": t["pending"]},
            acceptance=["判定写入台账且 verify 零差异", "待判数降为 0"],
            priority=int(min(t["pending"], 99)),
            budget=f"人工 ≤{t['pending']} 条（派生时）",
            authorization="只写标注，不改算法与模型")
        made.append(jid)
    return made


def sync_label_jobs(conn: sqlite3.Connection) -> list[dict]:
    """把已清零的 label 卡推进到「待验收」。

    只处理 `doing`：卡是人在做、队列也空了，就该来验收。
    `backlog` / `assigned` 不动——还没开始做的卡直接进待验收没有意义。
    **不自动 done**：关闭仍要人点一次，保留「必须有人对结果负责」。
    """
    moved = []
    for j in list_jobs(conn, status="doing"):
        if j.get("kind") != "label" or not j.get("drained"):
            continue
        live = j.get("live") or {}
        update_job_status(conn, j["id"], "review",
                          note=f"队列已清零（{live.get('done')}/{live.get('items')}）")
        moved.append({"id": j["id"], "title": j["title"],
                      "from": j["status"], "to": "review",
                      "pending": live.get("pending"), "done": live.get("done")})
    conn.commit()
    return moved


def label_progress(conn: sqlite3.Connection) -> dict[str, dict]:
    """标注队列的实时进度，按 task_id 索引。

    job 卡上的 pending 是派生那一刻的快照，**不能当作现状**；展示与关闭校验
    一律读这里。
    """
    return {t["task_id"]: t for t in task_progress(conn)}


def list_jobs(conn: sqlite3.Connection, status: str | None = None,
              live: bool = True) -> list[dict]:
    sql = "SELECT * FROM job WHERE 1=1"
    args = []
    if status:
        sql += " AND status=?"
        args.append(status)
    sql += " ORDER BY priority DESC, created_at DESC"
    out = []
    prog = label_progress(conn) if live else {}
    for r in conn.execute(sql, args):
        d = dict(r)
        d["inputs"] = jload(d.pop("inputs_json"), {})
        d["acceptance"] = jload(d.pop("acceptance_json"), [])
        d["evidence"] = jload(d.pop("evidence_json"), None)
        if live and d.get("kind") == "label":
            tid = (d["inputs"] or {}).get("task_id")
            t = prog.get(tid)
            if t:
                d["live"] = {"task_id": tid, "name": t["name"], "items": t["items"],
                             "done": t["done"], "pending": t["pending"], "pct": t["pct"]}
                d["drained"] = (t["pending"] or 0) == 0
            else:
                d["live"] = None
                d["drained"] = False
        out.append(d)
    return out


# ------------------------------------------------------------------ 总览
def overview(conn: sqlite3.Connection) -> dict:
    def one(sql, *a):
        r = conn.execute(sql, a).fetchone()
        return r[0] if r else 0

    tasks = task_progress(conn)
    pending = sum(t["pending"] or 0 for t in tasks)
    done = sum(t["done"] or 0 for t in tasks)
    items = sum(t["items"] or 0 for t in tasks)
    runs_total = one("SELECT COUNT(*) FROM run")
    runs_running = one("SELECT COUNT(*) FROM run WHERE status='running'")
    runs_failed = one("SELECT COUNT(*) FROM run WHERE status='failed'")
    runs_stale = one("SELECT COUNT(*) FROM run WHERE status='stale'")
    last = conn.execute("SELECT id, name, kind, status, started_at, ended_at FROM run "
                        "ORDER BY COALESCE(started_at, created_at) DESC LIMIT 8").fetchall()
    return {
        "authority": "local",
        "db": rel(DB_PATH),
        "generated_at": now(),
        "annotation": {"tasks": len(tasks), "items": items, "done": done,
                       "pending": pending,
                       "pct": round(100.0 * done / items, 1) if items else 0.0,
                       "by_task": tasks},
        "runs": {"total": runs_total, "running": runs_running, "failed": runs_failed,
                 "stale": runs_stale, "recent": [dict(r) for r in last]},
        "train": {
            "runs": one("SELECT COUNT(*) FROM run WHERE kind='train'"),
            "experiments": one("SELECT COUNT(*) FROM experiment"),
            "points": one("SELECT COUNT(*) FROM run_metric"),
            "last_at": one("SELECT MAX(started_at) FROM run WHERE kind='train'"),
        },
        "jobs": {"total": one("SELECT COUNT(*) FROM job"),
                 "open": one("SELECT COUNT(*) FROM job WHERE status IN "
                             "('backlog','assigned','doing','review')")},
        "counts": table_counts(conn),
    }


# ------------------------------------------------------------------ CLI
def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    conn = connect()
    try:
        if cmd == "init":
            print(json.dumps(init_db(conn), ensure_ascii=False, indent=1))
        elif cmd == "overview":
            print(json.dumps(overview(conn), ensure_ascii=False, indent=1))
        elif cmd == "sql":
            q = " ".join(argv[1:]).strip()
            if not q.lower().startswith("select"):
                print("只允许 SELECT", file=sys.stderr)
                return 2
            rows = [dict(r) for r in conn.execute(q)]
            print(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
        else:
            print(f"未知命令: {cmd}", file=sys.stderr)
            return 2
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
