# -*- coding: utf-8 -*-
"""算法中台 · 后端接口（/api/ops/*）

权威库在本机 `ops/ops.db`。本模块**可选**：
* 本机跑服务 → 正常读写台账；
* 远程只做预览 → 找不到 ops 库时返回 503 + 明确原因，不影响既有页面。

挂载方式见 main.py 末尾（include_router，失败不阻塞启动）。
"""
from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

router = APIRouter(prefix="/api/ops", tags=["ops"])

_OPS_ERR: str | None = None
try:
    import ops_db as opdb          # noqa: E402
    import ops_ingest as opin      # noqa: E402
except Exception as e:            # pragma: no cover - 远程无台账时走这里
    opdb = None
    opin = None
    _OPS_ERR = f"{type(e).__name__}: {e}"

ACTOR = "console"


def _conn():
    if opdb is None:
        raise HTTPException(503, f"台账未部署（本机为主库，远程仅预览）：{_OPS_ERR}")
    if not os.path.isfile(opdb.DB_PATH):
        raise HTTPException(503, f"台账库不存在：{opdb.rel(opdb.DB_PATH)}，先跑 "
                                 f"`python tools/ops_db.py init`")
    c = opdb.connect()
    opdb.init_db(c)
    return c


# ------------------------------------------------------------------ 总览
@router.get("/status")
def status():
    """轻量自检：前端据此决定是否显示台账入口。"""
    ok = opdb is not None and os.path.isfile(opdb.DB_PATH)
    return {"ok": ok, "db": opdb.rel(opdb.DB_PATH) if opdb else None,
            "authority": "local", "error": None if ok else _OPS_ERR}


@router.get("/overview")
def overview():
    c = _conn()
    try:
        return opdb.overview(c)
    finally:
        c.close()


# ------------------------------------------------------------------ 标注
@router.get("/tasks")
def tasks():
    c = _conn()
    try:
        return {"tasks": opdb.task_progress(c)}
    finally:
        c.close()


@router.get("/queue")
def queue(task_id: str, status: str | None = None,
          limit: int = Query(2000, le=5000), offset: int = 0):
    """★ 标注页读切换入口（唯一来源 = 台账）。

    `task_id` 支持逗号分隔（一页一次取全自己的几个任务）。返回：

    * `tasks[]`：任务元信息 + 实时进度（v_task_progress），页面用它渲染标题/进度；
    * `items{}`：按 task 分组的条目，`payload` 是原候选的全部展示字段（几何/分数/缩略图定位），
      `label` 是**最新一条**判定的 `{value, data, by_whom, at, confidence}`。

    `label.data` 是原始判定对象（`{v,c,n}` / `{pid,fp}` / `{verdict,name}` / 多边形）——
    只读 `value` 会丢置信度与备注，所以两个都给。
    """
    tids = [t.strip() for t in str(task_id).split(",") if t.strip()]
    if not tids:
        raise HTTPException(400, "task_id 必填")
    c = _conn()
    try:
        rows = {r["task_id"]: dict(r) for r in opdb.task_progress(c)}
        ph = ",".join("?" * len(tids))
        tasks = []
        for t in tids:
            meta = c.execute("SELECT * FROM annotation_task WHERE id=?", (t,)).fetchone()
            if not meta:
                continue
            m = dict(meta)
            p = rows.get(t) or {}
            by_src = {r["annotator"]: r["n"] for r in c.execute(
                "SELECT l.annotator, COUNT(*) n FROM annotation_item i "
                "JOIN label l ON l.id=(SELECT MAX(x.id) FROM label x WHERE x.item_id=i.id) "
                "WHERE i.task_id=? GROUP BY l.annotator", (t,))}
            tasks.append({"id": t, "name": m.get("name"), "kind": m.get("kind"),
                          "dialect": m.get("dialect"), "note": m.get("note"),
                          "spec": opdb.jload(m.get("spec_json"), None),
                          "items": p.get("items") or 0, "done": p.get("done") or 0,
                          "pending": p.get("pending") or 0,
                          "by_source": by_src,
                          # seed = 页面历史预填（非真人判定），进度里算已判但要能区分
                          "seeded": by_src.get("seed", 0)})

        sql = ("SELECT i.id, i.task_id, i.stratum, i.status, i.priority, i.payload_json, "
               "       l.value, l.data_json, l.annotator, l.created_at, l.confidence "
               "FROM annotation_item i "
               "LEFT JOIN label l ON l.id=(SELECT MAX(x.id) FROM label x WHERE x.item_id=i.id) "
               f"WHERE i.task_id IN ({ph})")
        args: list[Any] = list(tids)
        if status:
            sql += " AND i.status=?"
            args.append(status)
        sql += " ORDER BY i.task_id, i.priority DESC, i.id LIMIT ? OFFSET ?"
        args += [limit, offset]

        out: dict[str, list] = {t: [] for t in tids}
        for r in c.execute(sql, args):
            d = dict(r)
            tid = d["task_id"]
            out.setdefault(tid, []).append({
                "id": d["id"], "key": d["id"].split(":", 1)[1],
                "stratum": d["stratum"], "status": d["status"], "priority": d["priority"],
                "payload": opdb.jload(d["payload_json"], {}),
                "label": None if d["value"] is None and d["data_json"] is None else {
                    "value": d["value"],
                    "data": opdb.jload(d["data_json"], "__none__") if d["data_json"]
                            else opdb.jload(d["value"], d["value"]),
                    "by_whom": d["annotator"], "at": d["created_at"],
                    "confidence": d["confidence"]},
            })
        return {"tasks": tasks, "items": out}
    finally:
        c.close()


@router.get("/tasks/{task_id}/items")
def items(task_id: str, status: str | None = None,
          limit: int = Query(50, le=500), offset: int = 0):
    c = _conn()
    try:
        sql = ("SELECT i.id, i.task_id, i.stratum, i.status, i.payload_json, "
               "(SELECT value FROM label l WHERE l.item_id=i.id ORDER BY l.id DESC LIMIT 1) AS value, "
               "(SELECT annotator FROM label l WHERE l.item_id=i.id ORDER BY l.id DESC LIMIT 1) AS by_whom, "
               "(SELECT created_at FROM label l WHERE l.item_id=i.id ORDER BY l.id DESC LIMIT 1) AS at "
               "FROM annotation_item i WHERE i.task_id=?")
        args: list[Any] = [task_id]
        if status:
            sql += " AND i.status=?"
            args.append(status)
        total = c.execute(f"SELECT COUNT(*) FROM ({sql})", args).fetchone()[0]
        sql += " ORDER BY i.priority DESC, i.id LIMIT ? OFFSET ?"
        args += [limit, offset]
        rows = []
        for r in c.execute(sql, args):
            d = dict(r)
            d["key"] = d["id"].split(":", 1)[1]
            d["payload"] = opdb.jload(d.pop("payload_json"), {})
            rows.append(d)
        return {"task_id": task_id, "total": total, "offset": offset, "items": rows}
    finally:
        c.close()


@router.post("/label")
def post_label(payload: dict = Body(...)):
    """写一条判定：台账为准，随后投影回原始 json（双写期）。"""
    item_id = str(payload.get("item_id") or "").strip()
    if not item_id:
        raise HTTPException(400, "item_id 必填")
    if "value" not in payload:
        raise HTTPException(400, "value 必填（传 null 表示清除）")
    value = payload.get("value")
    annotator = str(payload.get("annotator") or ACTOR)
    c = _conn()
    try:
        row = c.execute("SELECT id, task_id FROM annotation_item WHERE id=?", (item_id,)).fetchone()
        if not row:
            raise HTTPException(404, f"条目不存在: {item_id}")
        prev = opdb.latest_label(c, item_id)
        if value is None:
            # 清除 = 追加一条 value=NULL 的判定（保持 append-only，latest 即为空）
            c.execute("INSERT INTO label(item_id, annotator, value, created_at, note) "
                      "VALUES(?,?,NULL,?, 'cleared')", (item_id, annotator, opdb.now()))
            c.execute("UPDATE annotation_item SET status='pending' WHERE id=?", (item_id,))
            opdb.audit(c, annotator, "label.clear", "annotation_item", item_id,
                       {"value": prev["value"]} if prev else None, None)
        else:
            # data = 原始判定对象（{v,c,n} 等）；不传则用 value 自身兜底
            data = payload.get("data")
            if data is None and isinstance(value, str) and value[:1] in ("[", "{"):
                try:
                    import json as _json
                    data = _json.loads(value)
                except Exception:
                    data = None
            opdb.set_label(c, item_id, str(value), annotator,
                           confidence=payload.get("confidence"),
                           seconds=payload.get("seconds"),
                           ui_version=payload.get("ui_version") or "webapp-v2",
                           data=data)
        proj = opin.project_item(item_id, value) if opin else {"ok": False}
        return {"ok": True, "item_id": item_id, "value": value, "projection": proj}
    finally:
        c.close()


@router.post("/verify")
def verify():
    c = _conn()
    try:
        return opin.verify(c) if opin else {"error": "ops_ingest 不可用"}
    finally:
        c.close()


# ------------------------------------------------------------------ 运行
@router.get("/runs")
def runs(limit: int = Query(100, le=500), kind: str | None = None, status: str | None = None):
    c = _conn()
    try:
        return {"runs": opdb.list_runs(c, limit=limit, kind=kind, status=status)}
    finally:
        c.close()


@router.get("/runs/{run_id}")
def run_detail(run_id: str):
    c = _conn()
    try:
        d = opdb.get_run(c, run_id)
        if not d:
            raise HTTPException(404, f"run 不存在: {run_id}")
        return d
    finally:
        c.close()


@router.get("/train")
def train(limit: int = Query(60, le=200)):
    """训练批次：实验组 → 变体（含 epoch 曲线），供训练页一次取全。"""
    c = _conn()
    try:
        exps = opdb.train_overview(c, limit)
        for e in exps:
            variants = []
            for r in c.execute(
                    "SELECT r.id, r.name, r.status, r.started_at, r.ended_at, r.metrics_json, "
                    "       r.params_json, r.note "
                    "FROM experiment_run er JOIN run r ON r.id=er.run_id "
                    "WHERE er.experiment_id=? ORDER BY r.name", (e["id"],)):
                v = dict(r)
                v["metrics"] = opdb.normalize_metrics(opdb.jload(v.pop("metrics_json"), None))
                v["params"] = opdb.jload(v.pop("params_json"), {})
                v["series"] = opdb.get_series(c, v["id"])
                v.pop("note", None)
                variants.append(v)
            variants.sort(key=lambda x: (x["metrics"] or {}).get("macro_f1") or -1, reverse=True)
            e["variants"] = variants

        # 没有 experiment 归属的训练 run（如临时冒烟）单独兜底，避免"跑了但训练页看不到"
        loose = []
        for r in c.execute(
                "SELECT r.id, r.name, r.status, r.started_at, r.ended_at, r.metrics_json, "
                "       r.params_json FROM run r WHERE r.kind='train' AND NOT EXISTS "
                "(SELECT 1 FROM experiment_run er WHERE er.run_id=r.id) "
                "ORDER BY COALESCE(r.started_at, r.created_at) DESC LIMIT 50"):
            v = dict(r)
            v["metrics"] = opdb.normalize_metrics(opdb.jload(v.pop("metrics_json"), None))
            v["params"] = opdb.jload(v.pop("params_json"), {})
            v["series"] = opdb.get_series(c, v["id"])
            loose.append(v)
        if loose:
            loose.sort(key=lambda x: (x["metrics"] or {}).get("macro_f1") or -1, reverse=True)
            exps.append({"id": "exp_ungrouped", "name": "未归入批次", "status": "done",
                         "conclusion": f"{len(loose)} 个训练 run 未挂 experiment",
                         "n_runs": len(loose), "last_at": loose[0].get("started_at"),
                         "variants": loose})
        return {"experiments": exps}
    finally:
        c.close()


@router.get("/runs/{run_id}/log")
def run_log(run_id: str, tail: int = Query(200, le=2000)):
    c = _conn()
    try:
        d = opdb.get_run(c, run_id)
        if not d:
            raise HTTPException(404, f"run 不存在: {run_id}")
        uri = d.get("log_uri")
        if not uri:
            return {"lines": [], "note": "该 run 未记录日志"}
        p = os.path.join(ROOT, uri)
        if not os.path.isfile(p):
            return {"lines": [], "note": f"日志文件不存在: {uri}"}
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-tail:]
        return {"lines": [x.rstrip("\n") for x in lines], "uri": uri}
    finally:
        c.close()


# ------------------------------------------------------------------ 派发
@router.get("/jobs")
def jobs(status: str | None = None):
    c = _conn()
    try:
        return {"jobs": opdb.list_jobs(c, status=status)}
    finally:
        c.close()


@router.post("/jobs/sync")
def sync_jobs():
    """把队列已清零的 label 卡推进到「待验收」（不自动关闭）。"""
    c = _conn()
    try:
        moved = opdb.sync_label_jobs(c)
        return {"ok": True, "moved": moved, "n": len(moved)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        c.close()


@router.post("/jobs")
def create_job(payload: dict = Body(...)):
    title = str(payload.get("title") or "").strip()
    if not title:
        raise HTTPException(400, "title 必填")
    c = _conn()
    try:
        jid = opdb.create_job(
            c, title=title, kind=str(payload.get("kind") or "label"),
            goal=payload.get("goal"), inputs=payload.get("inputs"),
            acceptance=payload.get("acceptance"),
            assignee_kind=str(payload.get("assignee_kind") or "human"),
            assignee=payload.get("assignee"),
            priority=int(payload.get("priority") or 0),
            budget=payload.get("budget"), authorization=payload.get("authorization"),
            status=str(payload.get("status") or "backlog"))
        return {"ok": True, "job": opdb.get_job(c, jid)}
    finally:
        c.close()


@router.post("/jobs/derive")
def derive_jobs(payload: dict = Body(default={})):
    """按标注待判量派生活动任务（幂等：已有未关闭的同类 job 会跳过）。"""
    c = _conn()
    try:
        made = opdb.derive_label_jobs(c, int((payload or {}).get("min_pending") or 1))
        return {"ok": True, "created": made, "n": len(made)}
    finally:
        c.close()


@router.post("/jobs/{job_id}/status")
def job_status(job_id: str, payload: dict = Body(...)):
    status = str(payload.get("status") or "").strip()
    if not status:
        raise HTTPException(400, "status 必填")
    c = _conn()
    try:
        try:
            j = opdb.update_job_status(c, job_id, status,
                                        evidence=payload.get("evidence"),
                                        note=payload.get("note"))
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "job": j}
    finally:
        c.close()
