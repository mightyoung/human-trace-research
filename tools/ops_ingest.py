#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ops ingest · 标注适配层（现有 json ↔ 台账）

把散在 4 个文件、3 套键名空间里的标注，映射成统一的 task / item / label，
并把「候选全集」补进来（否则只能看到已判的，看不到还差多少）。

幂等：重复跑不产生重复行（item 按 id upsert；label 同 annotator 同值去重）。

用法
----
    python tools/ops_ingest.py                 # 摄入（幂等）
    python tools/ops_ingest.py --verify        # 比对台账与 json，输出差异（只读）
    python tools/ops_ingest.py --export        # 从台账反向投影回 4 个 json（写前打印差异）
    python tools/ops_ingest.py --export --yes  # 确认写入

映射表
----
    t_pair_318x320   pair          pairs.json (75)            labels.json:pair_decisions（原始键）
    t_mirror_6cam    mirror        mirror.json (28+29)        labels.json:mirror_decisions
    t_mask_glass     mask          masks.json (6 机位)        labels.json:glass_masks
    t_face_person    face_person   face_data.json (45)        faces_labels.json:persons
    t_face_pair      face_pair     faces_labels.json (28)     faces_labels.json:pairs
    t_truth_pair     truth_pair    truth/state.json (241)     truth/labels.json:pair
    t_truth_event    truth_event   truth/state.json (107)     truth/labels.json:event
    t_s6b_pending    s6b           s6b_pending.json (70)      labels.json:pair_decisions（S6B_/PRT_ 前缀）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db  # noqa: E402

ROOT = db.ROOT
DATA = os.path.join(ROOT, "webapp", "backend", "data")
PUB = os.path.join(ROOT, "webapp", "frontend", "public")

LABELS = os.path.join(DATA, "labels.json")
PAIRS = os.path.join(DATA, "pairs.json")
MIRROR = os.path.join(DATA, "mirror.json")
MASKS = os.path.join(DATA, "masks.json")
FACE_LABELS = os.path.join(DATA, "faces_labels.json")
FACE_DATA = os.path.join(PUB, "faces", "face_data.json")
TRUTH_STATE = os.path.join(DATA, "truth", "state.json")
TRUTH_LABELS = os.path.join(DATA, "truth", "labels.json")
TRUTH_SEED = os.path.join(DATA, "truth", "seed_labels.json")   # 页面预填（非真人判定）
S6B = os.path.join(PUB, "s6b_pending.json")

DATASET_ID = "ds_6cam_synced10"
DATASET_ROOT = os.path.join(ROOT, "media-resource", "multi-cemra", "synced-10min")
DIALECT = "clean2"

LEGACY = "legacy"      # 迁移期来源标注的匿名作者
UI_VERSION = "webapp-v2"
# ★ S6-b 批次的键带前缀，历史上被写进了 truth/labels.json.pair（而非 labels.json）
PREFIXES = ("PRT_", "S6B_")


def is_s6b(key: str) -> bool:
    return str(key).startswith(PREFIXES)


def rj(p: str, default: Any = None):
    if not os.path.isfile(p):
        return default
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ingest] 读取失败 {db.rel(p)}: {e}", file=sys.stderr)
        return default


def ensure_dataset(conn) -> str:
    """轻量指纹：文件名 + 大小 + mtime 的 sha1，不读文件内容（几秒内完成）。"""
    fc, total, parts = 0, 0, []
    if os.path.isdir(DATASET_ROOT):
        for name in sorted(os.listdir(DATASET_ROOT)):
            p = os.path.join(DATASET_ROOT, name)
            if not os.path.isfile(p):
                continue
            st = os.stat(p)
            fc += 1
            total += st.st_size
            parts.append(f"{name}|{st.st_size}|{int(st.st_mtime)}")
    h = db.sha1_text("\n".join(parts)) if parts else None
    db.upsert_dataset(conn, DATASET_ID, "6 机位同步 10min（08:55–09:05）",
                      kind="video_set", root_uri=db.rel(DATASET_ROOT),
                      file_count=fc, total_bytes=total, dialect=DIALECT,
                      snapshot_hash=h, note="ops_ingest 自动登记；快照=文件名+大小+mtime")
    return DATASET_ID


def _mk_task(conn, task_id: str, name: str, kind: str, note: str | None = None):
    db.upsert_task(conn, task_id, name, kind, dataset_id=DATASET_ID, dialect=DIALECT,
                   mode="single", created_by="ops_ingest", note=note,
                   spec={"source": "legacy-json", "dialect": DIALECT,
                         "ui_version": UI_VERSION, "seed": 42})


# ★ 值为结构化对象（点列表等）的任务：入库前 JSON 编码，投影时解码。
#   否则 set_label 的 str() 会把 [[x,y],...] 变成不可解析的 Python repr。
JSON_VALUE_TASKS = {"t_mask_glass"}


def _put(conn, task_id: str, key: str, payload: Any, value: Any = None,
         confidence: float | None = None, at: str | None = None,
         stratum: str = "topk", annotator: str = LEGACY, data: Any = None) -> None:
    """写一条 item；若有判定则追加 label。value=None 表示未判。

    幂等：label 走 db.set_label（同 annotator 同值不新增行）。
    `data` = 原始判定对象（value 只保主值，证据靠它无损往返）。
    """
    item_id = f"{task_id}:{key}"
    db.upsert_item(conn, item_id, task_id, payload=payload, stratum=stratum,
                   status="labeled" if value is not None else "pending")
    if value is not None:
        if task_id in JSON_VALUE_TASKS and not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        db.set_label(conn, item_id, str(value), annotator, confidence=confidence,
                     ui_version=UI_VERSION, note="imported", at=at,
                     data=data if data is not None else value)


# ------------------------------------------------------------------ 各任务
def ingest_pair(conn) -> int:
    tid = "t_pair_318x320"
    _mk_task(conn, tid, "素材标注 · 跨机候选对（318×320）", "pair",
             "候选全集来自 pairs.json（cam_a/tid_a 形式）；判定来自 labels.json.pair_decisions")
    cand = rj(PAIRS, []) or []
    dec = (rj(LABELS, {}) or {}).get("pair_decisions", {}) or {}
    n = 0
    for p in cand:
        a = f"{p.get('cam_a')}:{p.get('tid_a')}"
        b = f"{p.get('cam_b')}:{p.get('tid_b')}"
        if not p.get("cam_a") or not p.get("cam_b"):
            continue
        key = f"{a}|{b}"
        _put(conn, tid, key, {k: v for k, v in p.items() if k not in ("cam_a", "cam_b")},
             dec.get(key))
        n += 1
    # 有判定但不在候选集里的（历史遗留），也补进来；带前缀的归 S6-b，不在此列
    for key, v in dec.items():
        if "|" in key and not is_s6b(key):
            _put(conn, tid, key, {"legacy": True}, v)
    return n


def _mirror_key(x) -> str:
    """mirror.json 元素形如 ["316", 10] → "316:10"。"""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return f"{x[0]}:{x[1]}"
    if isinstance(x, dict):
        return x.get("key") or x.get("obs") or ""
    return ""


def ingest_mirror(conn) -> int:
    tid = "t_mirror_6cam"
    _mk_task(conn, tid, "素材标注 · 倒影 / 源", "mirror",
             "候选来自 mirror.json；判定来自 labels.json.mirror_decisions")
    m = rj(MIRROR, {}) or {}
    dec = (rj(LABELS, {}) or {}).get("mirror_decisions", {}) or {}
    keys = []
    for kind in ("reflection", "source"):
        for x in m.get(kind, []) or []:
            k = _mirror_key(x)
            if k:
                keys.append((k, kind))
    n = 0
    seen = set()
    for k, kind in keys:
        if k in seen:
            continue
        seen.add(k)
        _put(conn, tid, k, {"kind": kind}, dec.get(k))
        n += 1
    for k, v in dec.items():
        if k not in seen:
            _put(conn, tid, k, {"legacy": True}, v)
            n += 1
    return n


def ingest_mask(conn) -> int:
    tid = "t_mask_glass"
    _mk_task(conn, tid, "素材标注 · 玻璃区域掩膜", "mask",
             "6 机位多边形；labels.json.glass_masks 为空 = 未圈改")
    m = rj(MASKS, {}) or {}
    dec = (rj(LABELS, {}) or {}).get("glass_masks", {}) or {}
    cams = list((m.get("cameras") or {}).keys()) or sorted(dec.keys())
    for cam in cams:
        _put(conn, tid, cam, {"cam": cam, "draft": (m.get("cameras") or {}).get(cam)},
             dec.get(cam), data=dec.get(cam))
    return len(cams)


def ingest_faces(conn) -> tuple[int, int]:
    fd = rj(FACE_DATA, {}) or {}
    fl = rj(FACE_LABELS, {}) or {}
    persons = fd.get("persons") or []
    plabels = fl.get("persons") or {}
    pair_labels = fl.get("pairs") or {}

    t1 = "t_face_person"
    _mk_task(conn, t1, "人脸库 · 人卡判定", "face_person",
             "45 张人卡；判定来自 faces_labels.json.persons（verdict + name）")
    for p in persons:
        lab = p.get("label")
        if not lab:
            continue
        got = plabels.get(lab) or {}
        _put(conn, t1, lab, {**p, "name": got.get("name")}, got.get("verdict"),
             at=got.get("at"), data=got)
    for lab, got in plabels.items():
        if lab not in {p.get("label") for p in persons}:
            _put(conn, t1, lab, {"legacy": True, **got}, got.get("verdict"), at=got.get("at"),
                 data=got)

    t2 = "t_face_pair"
    _mk_task(conn, t2, "人脸库 · 候选合并对", "face_pair",
             "候选即已判 28 对（faces_labels.json.pairs）；上游未保留未判候选")
    for key, got in pair_labels.items():
        _put(conn, t2, key, got, got.get("decision"), at=got.get("at"), data=got)
    return len(persons), len(pair_labels)


def ingest_truth(conn) -> tuple[int, int]:
    st = rj(TRUTH_STATE, {}) or {}
    lb = rj(TRUTH_LABELS, {}) or {}
    pairs = st.get("pairs") or []
    events = st.get("events") or []
    pl = lb.get("pair") or {}
    el = lb.get("event") or {}

    t1 = "t_truth_pair"
    _mk_task(conn, t1, "真值标注 · 跨机实体配对（343×316）", "truth_pair",
             "241 对；判定来自 truth/labels.json.pair（v=1 同人 / 0 不是 / 2 看不清）")
    strat = {}
    sel = (st.get("meta") or {}).get("select") or {}
    for p in pairs:
        pid = p.get("pair_id")
        if not pid:
            continue
        got = pl.get(pid) or {}
        # 分层：直接取 state.json 的 role（topk/audit/rest）——它是候选生成的权威标注。
        # ★ 不能用 rank<k_topk 反推：rank 从 1 起，会把第 k 名错划进下一层（曾 30→29）。
        role = p.get("role")
        s = role if role in ("topk", "audit", "rest") else "audit"
        strat[s] = strat.get(s, 0) + 1
        _put(conn, t1, pid, p, got.get("v"), at=got.get("at"), stratum=s,
             confidence={"high": 1.0, "mid": 0.6, "low": 0.3}.get(str(got.get("c") or "").lower()),
             data=got)

    t2 = "t_truth_event"
    _mk_task(conn, t2, "真值标注 · 出入口事件", "truth_event",
             "107 事件；判定来自 truth/labels.json.event（pid 人物编号 / fp 误报）")
    for e in events:
        eid = e.get("eid")
        if not eid:
            continue
        got = el.get(eid) or {}
        val = None
        if got.get("fp") is True:
            val = "fp"
        elif got.get("pid") not in (None, ""):
            val = f"pid:{got['pid']}"
        _put(conn, t2, eid, e, val, at=got.get("at"), data=got or None)
    return len(pairs), len(events)


def ingest_s6b(conn) -> int:
    tid = "t_s6b_pending"
    _mk_task(conn, tid, "待裁决批 · S6-b 倒影 + 门户交接补口", "s6b",
             "75 项 / 5 组；历史判定分散在 labels.json.pair_decisions 与 truth/labels.json.pair"
             "（带 PRT_/S6B_ 前缀），此处合并读取")
    d = rj(S6B, {}) or {}
    # ★ 两处来源合并：历史原因，同一批的判定写在两个文件里
    dec = {}
    dec.update((rj(TRUTH_LABELS, {}) or {}).get("pair") or {})
    dec.update((rj(LABELS, {}) or {}).get("pair_decisions") or {})
    n = 0
    for g in d.get("groups") or []:
        gk = g.get("key") or "g"
        for it in g.get("items") or []:
            iid = it.get("id")
            if not iid:
                continue
            raw = dec.get(iid)
            val = raw.get("v") if isinstance(raw, dict) else raw
            _put(conn, tid, iid, {"group": gk, "title": g.get("title"), **it}, val,
                 stratum="conflict" if "conflict" in gk else "boundary",
                 data=raw if isinstance(raw, dict) else None)
            n += 1
    return n


# ------------------------------------------------------------------ 主流程
def ingest(conn) -> dict:
    ensure_dataset(conn)
    # 人脸/真值两个函数各写两个 task，避免重复调用
    ingest_pair(conn)
    ingest_mirror(conn)
    ingest_mask(conn)
    n_person, n_fpair = ingest_faces(conn)
    n_tpair, n_tevent = ingest_truth(conn)
    n_s6b = ingest_s6b(conn)
    return {"face_person": n_person, "face_pair": n_fpair,
            "truth_pair": n_tpair, "truth_event": n_tevent, "s6b": n_s6b,
            "counts": db.table_counts(conn)}


def verify(conn) -> dict:
    """比对台账最新 label 与源 json，输出三类差异。"""
    truth_pair_all = ((rj(TRUTH_LABELS, {}) or {}).get("pair") or {})
    truth_event_all = ((rj(TRUTH_LABELS, {}) or {}).get("event") or {})
    s6b_labels = {k: (v or {}).get("v") if isinstance(v, dict) else v
                  for k, v in truth_pair_all.items() if is_s6b(k)}
    s6b_labels.update({k: v for k, v in
                       ((rj(LABELS, {}) or {}).get("pair_decisions") or {}).items()
                       if is_s6b(k)})
    # 真值配对的 json 侧 = labels.json.pair ∪ seed_labels（seed 是更早的历史预填，
    # 页面一直把它当已判；台账已回填为 annotator='seed'。不并入会恒报 15 条"账有 json 无"）
    truth_pair_json = {k: (v or {}).get("v") if isinstance(v, dict) else v
                       for k, v in truth_pair_all.items() if not is_s6b(k)}
    for k, v in (rj(TRUTH_SEED, {}) or {}).items():
        if not is_s6b(k):
            truth_pair_json.setdefault(k, (v or {}).get("v") if isinstance(v, dict) else v)
    checks = [
        ("t_pair_318x320", "labels.json:pair_decisions",
         {k: v for k, v in ((rj(LABELS, {}) or {}).get("pair_decisions") or {}).items()
          if not is_s6b(k)}),
        ("t_mirror_6cam", "labels.json:mirror_decisions",
         (rj(LABELS, {}) or {}).get("mirror_decisions", {})),
        ("t_mask_glass", "labels.json:glass_masks",
         {k: (json.dumps(v, ensure_ascii=False, sort_keys=True)
              if isinstance(v, (list, dict)) else v)
          for k, v in ((rj(LABELS, {}) or {}).get("glass_masks") or {}).items()}),
        ("t_s6b_pending", "labels.json 与 truth/labels.json（前缀键合并）", s6b_labels),
        ("t_truth_pair", "truth/labels.json:pair ∪ seed_labels（排除 S6-b 前缀）",
         truth_pair_json),
        ("t_truth_event", "truth/labels.json:event（pid/fp 归一）",
         {k: _truth_event_value(v) for k, v in truth_event_all.items()}),
        ("t_face_pair", "faces_labels.json:pairs",
         {k: (v or {}).get("decision")
          for k, v in ((rj(FACE_LABELS, {}) or {}).get("pairs") or {}).items()}),
        ("t_face_person", "faces_labels.json:persons",
         {k: (v or {}).get("verdict")
          for k, v in ((rj(FACE_LABELS, {}) or {}).get("persons") or {}).items()}),
    ]
    out = {}
    for task_id, src, jmap in checks:
        rows = conn.execute(
            "SELECT i.id, (SELECT value FROM label l WHERE l.item_id=i.id ORDER BY l.id DESC "
            "LIMIT 1) AS v FROM annotation_item i WHERE i.task_id=?", (task_id,)).fetchall()
        dbmap = {r["id"].split(":", 1)[1]: r["v"] for r in rows}
        jkeys = {str(k): (str(v) if v is not None else None) for k, v in jmap.items()}
        miss_db = sorted(set(jkeys) - set(dbmap))
        miss_json = sorted(k for k, v in dbmap.items() if v is not None and k not in jkeys)
        conflict = sorted(k for k in set(jkeys) & set(dbmap)
                          if dbmap[k] is not None and jkeys[k] != str(dbmap[k]))
        out[task_id] = {"source": src,
                        "items": len(dbmap),
                        "db_labeled": sum(1 for v in dbmap.values() if v is not None),
                        "json_labeled": len(jkeys),
                        "missing_in_db": miss_db[:10], "n_missing_in_db": len(miss_db),
                        "missing_in_json": miss_json[:10], "n_missing_in_json": len(miss_json),
                        "conflict": conflict[:10], "n_conflict": len(conflict)}
    return out


# ------------------------------------------------------------------ 反向镜像（老页面 → 台账）
# 老页面（AnnotateView / FacesView / TruthView）直接写 json，不经台账。
# 以下函数由 main.py 在写接口末尾调用，把改动补进台账；失败只告警，不影响页面保存。
_TASK_META = {
    "t_pair_318x320": ("素材标注 · 跨机候选对（318×320）", "pair"),
    "t_mirror_6cam": ("素材标注 · 倒影 / 源", "mirror"),
    "t_mask_glass": ("素材标注 · 玻璃区域掩膜", "mask"),
    "t_face_person": ("人脸库 · 人卡判定", "face_person"),
    "t_face_pair": ("人脸库 · 候选合并对", "face_pair"),
    "t_truth_pair": ("真值标注 · 跨机实体配对（343×316）", "truth_pair"),
    "t_truth_event": ("真值标注 · 出入口事件", "truth_event"),
    "t_s6b_pending": ("待裁决批 · S6-b 倒影 + 门户交接补口", "s6b"),
}


def _route_pair(key: str) -> str:
    return "t_s6b_pending" if is_s6b(key) else "t_pair_318x320"


def _ensure_task(conn, task_id: str) -> None:
    if conn.execute("SELECT 1 FROM annotation_task WHERE id=?", (task_id,)).fetchone():
        return
    name, kind = _TASK_META[task_id]
    ds = DATASET_ID if conn.execute(
        "SELECT 1 FROM dataset WHERE id=?", (DATASET_ID,)).fetchone() else None
    db.upsert_task(conn, task_id, name, kind, dataset_id=ds, dialect=DIALECT,
                   mode="single", created_by="mirror", note="mirror 自动建任务")


def _ensure_item(conn, item_id: str, task_id: str, payload: Any = None) -> None:
    if not conn.execute("SELECT 1 FROM annotation_item WHERE id=?", (item_id,)).fetchone():
        db.upsert_item(conn, item_id, task_id,
                       payload=payload if payload is not None else {"source": "console"})


def _apply(conn, item_id: str, value: Any, annotator: str, data: Any = None) -> None:
    """设置或清除一条判定（与 ops_api 的清除语义一致：追加 NULL 行）。

    `data` 存原始判定对象；不传则退回 value（字符串/多边形，足以往返）。
    """
    if value is None:
        conn.execute("INSERT INTO label(item_id, annotator, value, created_at, note) "
                     "VALUES(?,?,NULL,?, 'cleared-mirror')", (item_id, annotator, db.now()))
        conn.execute("UPDATE annotation_item SET status='pending' WHERE id=?", (item_id,))
    else:
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        db.set_label(conn, item_id, str(value), annotator, ui_version=UI_VERSION, note="mirror",
                     data=data if data is not None else value)


def mirror_labels(payload: dict, annotator: str = "console") -> dict:
    """与 POST /api/labels 同形：{pair_decisions, mirror_decisions, glass_masks, notes}。"""
    c = db.connect()
    db.init_db(c)
    n = 0
    try:
        for field, fixed_task in (("pair_decisions", None),
                                  ("mirror_decisions", "t_mirror_6cam"),
                                  ("glass_masks", "t_mask_glass")):
            entries = payload.get(field)
            if not isinstance(entries, dict):
                continue
            for k, v in entries.items():
                task = _route_pair(k) if fixed_task is None else fixed_task
                _ensure_task(c, task)
                item_id = f"{task}:{k}"
                _ensure_item(c, item_id, task, {"key": k, "source": "console"})
                _apply(c, item_id, v, annotator, data=v)
                n += 1
        return {"ok": True, "mirrored": n}
    finally:
        c.close()


def mirror_face_person(label: str, item: dict | None, annotator: str = "console") -> dict:
    c = db.connect()
    db.init_db(c)
    try:
        _ensure_task(c, "t_face_person")
        item_id = f"t_face_person:{label}"
        _ensure_item(c, item_id, "t_face_person", {"label": label, "source": "console"})
        db.upsert_item(c, item_id, "t_face_person",
                       payload={"label": label, "name": (item or {}).get("name"),
                                "source": "console"})
        _apply(c, item_id, (item or {}).get("verdict"), annotator, data=item or None)
        return {"ok": True, "item_id": item_id}
    finally:
        c.close()


def mirror_face_pair(key: str, decision: Any, annotator: str = "console") -> dict:
    c = db.connect()
    db.init_db(c)
    try:
        _ensure_task(c, "t_face_pair")
        item_id = f"t_face_pair:{key}"
        _ensure_item(c, item_id, "t_face_pair", {"key": key, "source": "console"})
        _apply(c, item_id, decision, annotator)
        return {"ok": True, "item_id": item_id}
    finally:
        c.close()


def _truth_event_value(v: Any) -> Any:
    if isinstance(v, dict):
        if v.get("fp") is True:
            return "fp"
        pid = v.get("pid")
        return f"pid:{pid}" if pid not in (None, "") else None
    return v


def mirror_truth(payload: dict, annotator: str = "console") -> dict:
    """与 POST /api/truth/labels 同形：{pair: {pid: {...}}, event: {eid: {...}}}。"""
    c = db.connect()
    db.init_db(c)
    n = 0
    try:
        for pid, v in (payload.get("pair") or {}).items():
            # ★ 真值 pair 段里混着 S6-b 批次的键（PRT_/S6B_ 前缀），必须按前缀路由，
            #   否则同一判定会在 t_truth_pair 与 t_s6b_pending 各存一份，口径对不上。
            task = "t_s6b_pending" if is_s6b(pid) else "t_truth_pair"
            _ensure_task(c, task)
            item_id = f"{task}:{pid}"
            _ensure_item(c, item_id, task, {"pair_id": pid, "source": "console"})
            _apply(c, item_id, (v or {}).get("v") if isinstance(v, dict) else v, annotator,
                   data=v if isinstance(v, dict) else None)
            n += 1
        for eid, v in (payload.get("event") or {}).items():
            _ensure_task(c, "t_truth_event")
            item_id = f"t_truth_event:{eid}"
            _ensure_item(c, item_id, "t_truth_event", {"eid": eid, "source": "console"})
            _apply(c, item_id, _truth_event_value(v), annotator,
                   data=v if isinstance(v, dict) else None)
            n += 1
        return {"ok": True, "mirrored": n}
    finally:
        c.close()


# ------------------------------------------------------------------ 单条投影（双写用）
# task_id -> (路径常量名, 段, 写法)。★ 存常量名而非路径：测试可 monkeypatch 模块常量，
#   若在此处直接求值会把测试写入落到真实文件上（曾发生过）。
PROJECTION = {
    "t_pair_318x320": ("LABELS", "pair_decisions", "raw"),
    "t_mirror_6cam": ("LABELS", "mirror_decisions", "raw"),
    "t_mask_glass": ("LABELS", "glass_masks", "raw"),
    "t_face_person": ("FACE_LABELS", "persons", "face_person"),
    "t_face_pair": ("FACE_LABELS", "pairs", "face_pair"),
    "t_truth_pair": ("TRUTH_LABELS", "pair", "dict_v"),
    "t_truth_event": ("TRUTH_LABELS", "event", "truth_event"),
    # S6-b 历史判定在 truth/labels.json.pair，沿用同一位置以保持下游兼容
    "t_s6b_pending": ("TRUTH_LABELS", "pair", "dict_v"),
}


def _write_json_atomic(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def project_item(item_id: str, value: Any) -> dict:
    """把一条判定写回原始 json（双写期的投影，非权威）。"""
    if ":" not in item_id:
        return {"ok": False, "reason": "bad_item_id"}
    task_id, key = item_id.split(":", 1)
    spec = PROJECTION.get(task_id)
    if not spec:
        return {"ok": False, "reason": "no_projection"}
    path_name, section, mode = spec
    path = globals()[path_name]          # 运行时解析，支持测试注入
    data = rj(path, {}) or {}
    sec = data.setdefault(section, {})
    if mode == "raw":
        if value is None:
            sec.pop(key, None)
        else:
            if task_id in JSON_VALUE_TASKS and isinstance(value, str) \
                    and value[:1] in ("[", "{"):
                try:
                    value = json.loads(value)
                except Exception:
                    pass
            sec[key] = value
    elif mode == "dict_v":
        if value is None:
            sec.pop(key, None)
        else:
            it = sec.setdefault(key, {})
            it["v"] = int(value) if str(value).lstrip("-").isdigit() else value
    elif mode == "face_person":
        it = sec.setdefault(key, {})
        it["verdict"] = value
        it["at"] = db.now()
    elif mode == "face_pair":
        it = sec.setdefault(key, {})
        if "@" in key:
            it.setdefault("a", key.split("@")[0])
            it.setdefault("b", key.split("@")[1])
        it["decision"] = value
        it["at"] = db.now()
    elif mode == "truth_event":
        it = sec.setdefault(key, {})
        if isinstance(value, str) and value.startswith("pid:"):
            it["pid"] = value[4:]
        elif value == "fp":
            it["fp"] = True
        else:
            it["pid"] = "" if value is None else value
    _write_json_atomic(path, data)
    return {"ok": True, "file": db.rel(path), "section": section, "key": key, "value": value}


def export(conn, apply: bool) -> dict:
    """从台账反向投影（P0 只覆盖 labels.json 的三类，其余只报告）。"""
    plan: dict[str, dict] = {}
    for task_id, field in [("t_pair_318x320", "pair_decisions"),
                           ("t_mirror_6cam", "mirror_decisions"),
                           ("t_mask_glass", "glass_masks")]:
        rows = conn.execute(
            "SELECT i.id, (SELECT value FROM label l WHERE l.item_id=i.id ORDER BY l.id DESC "
            "LIMIT 1) AS v FROM annotation_item i WHERE i.task_id=? AND EXISTS("
            "SELECT 1 FROM label l2 WHERE l2.item_id=i.id)", (task_id,)).fetchall()
        plan[field] = {r["id"].split(":", 1)[1]: r["v"] for r in rows if r["v"] is not None}

    before = rj(LABELS, {}) or {}
    after = {**before}
    for field, mapping in plan.items():
        cur = dict(before.get(field) or {})
        cur.update(mapping)
        after[field] = cur
    diff = {f: {"before": len(before.get(f) or {}), "after": len(after.get(f) or {})}
            for f in plan}
    if apply:
        tmp = LABELS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(after, f, ensure_ascii=False, indent=1)
        os.replace(tmp, LABELS)
        db.audit(conn, "ops_ingest", "projection.export", "file", db.rel(LABELS),
                 {"counts": {f: len(before.get(f) or {}) for f in plan}}, diff)
    return {"applied": apply, "diff": diff}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)

    conn = db.connect()
    try:
        db.init_db(conn)
        if args.verify:
            print(json.dumps(verify(conn), ensure_ascii=False, indent=1))
            return 0
        if args.export:
            print(json.dumps(export(conn, args.yes), ensure_ascii=False, indent=1))
            return 0
        print(json.dumps(ingest(conn), ensure_ascii=False, indent=1))
        print(json.dumps(db.task_progress(conn), ensure_ascii=False, indent=1))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
