# -*- coding: utf-8 -*-
"""E21 全天单机位：人工标簇（控制台页 #/e21-label 的后端）。

为什么单独一个模块
------------------
E21 三天（10-22/23/26）原始卷共 88 机位小时，现有实验只用了一个 36–68 min 的
「6 路交集窗」（15 机位小时）。全天重跑后得到每机位 1.9–6.1 小时的连续轨迹，
但**三天零身份标注**——召回不可测。本模块把聚类结果做成可标注的卡片：

  簇（average linkage, cosine 距离 .22 = cos .78，与生产阈值同口径）
  → 每簇取最多 4 张代表裁剪（最长轨、轨内 25%/75% 处）
  → 人工给一个身份名 / 判非人 / 跳过；**单张裁剪也可以单独判非人**
  → 导出 CSV 就是簇级真值表 + 假框表。

口径（必须随结果一起引用）
  · 卡片只代表**该簇**，不是该人的全部出现；漏掉的簇要靠"同名"合并补上。
  · 单机位里时间重叠的两条轨 = 两个人（排除 IoU<0.2 的重复框后），
    所以「同一个人被标进两个簇」与「一个簇里两人」都能从标签表直接查出来。
  · **单张判非人 ≠ 整簇判非人**：前者是"这一帧检测错了"（用来量检测假框率），
    后者是"这条轨/这个簇不是人"。两者分开记，不能互相替代。
  · 标签存 data/e21_label/labels_{day}.json，与 clusters_{day}.json 分离，
    重跑聚类不会覆盖人工结果。

接口（挂在 /api/e21）
  GET  /days                     有卡片的天与进度
  GET  /clusters?day=            簇清单（含已标标签与单图非人标记）
  GET  /crop/{day}/{file}        裁剪图
  POST /label                    {day, cid, action: person|reject|skip, person?}
  POST /label/clear              {day, cid}
  POST /crop_label               {day, cid, k, action: reject|clear}
  GET  /summary?day=             身份计数 / 整簇非人 / 单图非人
  GET  /export?day=              簇级真值 CSV
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

router = APIRouter(prefix="/api/e21", tags=["e21"])

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "e21_label"

_CACHE: dict[str, dict] = {}


def _clusters_path(day: str) -> Path:
    return DATA / f"clusters_{day}.json"


def _labels_path(day: str) -> Path:
    return DATA / f"labels_{day}.json"


def _load_clusters(day: str) -> dict:
    p = _clusters_path(day)
    if not p.exists():
        raise HTTPException(404, f"没有 {day} 的卡片，先生成 clusters_{day}.json")
    mtime = p.stat().st_mtime
    c = _CACHE.get(day)
    if c and c["mtime"] == mtime:
        return c["data"]
    data = json.loads(p.read_text(encoding="utf-8"))
    _CACHE[day] = {"mtime": mtime, "data": data}
    return data


def _s(v) -> str:
    """安全转字符串。

    ★★★ 不能写 `str(v or "")`：**cid=0 是合法簇号**，而 Python 里 `0 or ""` 得 `""`
      → 后端判成「cid 必填」直接 400。11-23 踩过一次：用户点「可疑优先」后第一张卡
      就是 #0，一标就报保存失败。
    """
    return "" if v is None else str(v).strip()


def _load_priority(day: str) -> dict:
    """可疑度（由 tmp/r5-e1/e21_priority_rank.py 产出，只保留有信号的那部分）。

    ★ 只收录两类簇：`fm > 0`（时空互斥对 = 同一时刻出现在两处，混人的硬证据）
      以及 risk ≥ 0.5 的边界簇。其余簇不写进文件，前端按 0 处理。
    ★★ 为什么不用「准静止 + 小框 + 短轨」那套（junk 分）：11-23 抽检 8/8 全是真人
      （背影、后脑勺、坐姿）—— --drop-static 已把真正的静止静物剔干净，再做就是
      在抓「坐着办公的人」。
    """
    p = DATA / f"priority_{day}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_labels(day: str) -> dict:
    p = _labels_path(day)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_labels(day: str, labels: dict) -> None:
    _labels_path(day).write_text(
        json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")


def _hms(sec: float) -> str:
    s = int(round(sec)) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def _crop_rejects(rec: dict | None) -> list[int]:
    """逐图判「不是人」的下标（只认 'reject'，不含 'person:*'）。"""
    if not rec:
        return []
    out = []
    for k, v in (rec.get("crops") or {}).items():
        if v != "reject":
            continue
        try:
            out.append(int(k))
        except Exception:
            pass
    return sorted(out)


def _crop_persons(rec: dict | None) -> dict[str, str]:
    """逐图单独指定的身份（'person:张三' → {'0': '张三'}）。

    一个簇里混了两个人时用它拆开：簇标签仍是默认归属，
    单图标签是覆盖，两者分开存，导出时可以按图重建真值。
    """
    if not rec:
        return {}
    out = {}
    for k, v in (rec.get("crops") or {}).items():
        if isinstance(v, str) and v.startswith("person:"):
            out[k] = v.split(":", 1)[1]
    return out


def _split_day(day: str) -> tuple[str, str, bool]:
    """`2026-10-22_办公区2` -> (日期, 机位, 是否旧版)。

    旧版是 2026-11-19 那批「按天」的产物（实为单机位、未做静止假框过滤），
    文件名不带机位下划线；新版按 `日期_机位` 落盘，两者并存互不覆盖。
    """
    if "_" in day:
        date, cam = day.split("_", 1)
        return date, cam, False
    return day, "旧版·未过滤", True


@router.get("/days")
def days():
    out = []
    for p in sorted(DATA.glob("clusters_*.json")):
        day = p.stem.replace("clusters_", "")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        labels = _load_labels(day)
        done = sum(1 for c in d["clusters"] if (labels.get(str(c["cid"])) or {}).get("action"))
        prio = _load_priority(day)
        n_susp = sum(1 for v in prio.values() if (v or {}).get("fm"))
        n_susp_todo = sum(1 for cid, v in prio.items()
                          if (v or {}).get("fm")
                          and not (labels.get(str(cid)) or {}).get("action"))
        date, cam, legacy = _split_day(day)
        out.append({
            "day": day,
            "date": date,
            "cam": cam,
            "legacy": legacy,
            "n_clusters": d.get("n_clusters", len(d["clusters"])),
            "n_tracks": d.get("n_tracks", 0),
            "n_tracks_all": d.get("n_tracks_all"),
            "n_dropped_static": d.get("n_dropped_static"),
            "thr_cos": d.get("thr_cos"),
            "n_labeled": done,
            "n_suspect": n_susp,
            "n_suspect_todo": n_susp_todo,
        })
    # 日期升序；同一天里新段按机位名排，旧版垫底
    out.sort(key=lambda x: (x["date"], x["legacy"], x["cam"]))
    return {"days": out}


@router.get("/clusters")
def clusters(day: str):
    d = _load_clusters(day)
    labels = _load_labels(day)
    prio = _load_priority(day)
    cs = []
    for c in d["clusters"]:
        lab = labels.get(str(c["cid"])) or {}
        pr = prio.get(str(c["cid"])) or {}
        cs.append({
            "cid": c["cid"],
            "n_tracks": c["n_tracks"],
            "frames": c["frames"],
            "h_med": c["h_med"],
            "t0": _hms(c["t0"]),
            "t1": _hms(c["t1"]),
            "span_s": c["span_s"],
            "crops": [x["file"] for x in c.get("crops", [])],
            "crops_meta": [{"k": x["k"], "t": x["t"], "tid": x["tid"]}
                           for x in c.get("crops", [])],
            "action": lab.get("action"),
            "person": lab.get("person"),
            "crop_reject": _crop_rejects(lab),
            "crop_persons": _crop_persons(lab),
            "fm_pairs": pr.get("fm", 0),
            "risk": pr.get("risk", 0.0),
            "suspect": pr.get("why") or [],
        })
    return {"day": day, "thr_cos": d.get("thr_cos"),
            "n_tracks": d.get("n_tracks"), "clusters": cs}


@router.get("/crop/{day}/{file}")
def crop(day: str, file: str):
    if "/" in file or "\\" in file or ".." in file:
        raise HTTPException(400, "bad file")
    p = DATA / "crops" / day / file
    if not p.exists():
        raise HTTPException(404, "crop missing")
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@router.post("/label")
def label(payload: dict = Body(...)):
    day = str(payload.get("day") or "")
    cid = _s(payload.get("cid"))
    action = str(payload.get("action") or "")
    person = (payload.get("person") or "").strip()
    if not day or not cid:
        raise HTTPException(400, "day/cid 必填")
    if action not in ("person", "reject", "skip"):
        raise HTTPException(400, "action 必须是 person/reject/skip")
    if action == "person" and not person:
        raise HTTPException(400, "标为人时必须给名字")
    labels = _load_labels(day)
    prev = labels.get(cid) or {}
    keep_crops = prev.get("crops") or {}
    if action == "skip":
        if keep_crops:
            labels[cid] = {"action": None, "person": "", "crops": keep_crops,
                           "at": round(time.time(), 1)}
        else:
            labels.pop(cid, None)
    else:
        labels[cid] = {"action": action, "person": person if action == "person" else "",
                       "crops": keep_crops, "at": round(time.time(), 1)}
    _save_labels(day, labels)
    return {"ok": True, "n_labeled": sum(1 for v in labels.values() if v.get("action"))}


@router.post("/label/clear")
def label_clear(payload: dict = Body(...)):
    day = str(payload.get("day") or "")
    cid = _s(payload.get("cid"))
    labels = _load_labels(day)
    labels.pop(cid, None)
    _save_labels(day, labels)
    return {"ok": True, "n_labeled": sum(1 for v in labels.values() if v.get("action"))}


@router.post("/crop_label")
def crop_label(payload: dict = Body(...)):
    """单张 / 多张裁剪打标（不改变整簇的判定）。

    action:
      reject —— 这一帧不是人（帧级假框），用来量检测假框率
      person —— 这几张属于 person 指定的那个人（拆混合簇用）
      clear  —— 撤销这些图的单独标注（回到继承簇标签）
    k 与 ks 二选一：k 单张，ks 批量。
    """
    day = str(payload.get("day") or "")
    cid = _s(payload.get("cid"))
    ks = payload.get("ks")
    if ks is None:
        k = payload.get("k")
        ks = [] if k is None else [k]
    ks = [int(x) for x in ks]
    action = str(payload.get("action") or "")
    person = (payload.get("person") or "").strip()
    if not day or not cid or not ks:
        raise HTTPException(400, "day/cid/k(ks) 必填")
    if action not in ("reject", "person", "clear"):
        raise HTTPException(400, "action 必须是 reject/person/clear")
    if action == "person" and not person:
        raise HTTPException(400, "标为人时必须给名字")
    labels = _load_labels(day)
    rec = labels.setdefault(cid, {"action": None, "person": "", "crops": {}})
    rec.setdefault("crops", {})
    for k in ks:
        if action == "reject":
            rec["crops"][str(k)] = "reject"
        elif action == "person":
            rec["crops"][str(k)] = f"person:{person}"
        else:
            rec["crops"].pop(str(k), None)
    if not rec.get("action") and not rec["crops"]:
        labels.pop(cid, None)
    else:
        rec["at"] = round(time.time(), 1)
        labels[cid] = rec
    _save_labels(day, labels)
    return {"ok": True, "crops": _crop_rejects(rec),
            "crop_persons": _crop_persons(rec)}


@router.get("/summary")
def summary(day: str):
    labels = _load_labels(day)
    by: dict[str, int] = {}
    n_rej = n_skip = n_crop_rej = n_crop_per = 0
    for v in labels.values():
        if v.get("action") == "person":
            by[v.get("person", "")] = by.get(v.get("person", ""), 0) + 1
        elif v.get("action") == "reject":
            n_rej += 1
        elif v.get("action") == "skip":
            n_skip += 1
        n_crop_rej += len(_crop_rejects(v))
        n_crop_per += len(_crop_persons(v))
    return {"day": day, "persons": by, "n_reject": n_rej, "n_skip": n_skip,
            "n_crop_reject": n_crop_rej, "n_crop_person": n_crop_per,
            "n_labeled": sum(1 for v in labels.values() if v.get("action"))}


@router.get("/export")
def export(day: str):
    d = _load_clusters(day)
    labels = _load_labels(day)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["day", "cid", "action", "person", "n_tracks", "frames",
                "t0", "t1", "span_s", "h_med", "n_crop",
                "n_crop_reject", "n_crop_person", "crop_reject", "crop_persons",
                "tracks"])
    for c in d["clusters"]:
        lab = labels.get(str(c["cid"])) or {}
        cp = _crop_persons(lab)
        w.writerow([day, c["cid"], lab.get("action") or "", lab.get("person") or "",
                    c["n_tracks"], c["frames"], _hms(c["t0"]), _hms(c["t1"]),
                    round(c["span_s"], 1), c["h_med"],
                    len(c.get("crops") or []),
                    len(_crop_rejects(lab)), len(cp),
                    " ".join(str(x) for x in _crop_rejects(lab)),
                    " ".join(f"{k}:{v}" for k, v in sorted(cp.items())),
                    " ".join(str(t) for t in c["tracks"])])
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="e21_{day}.csv"'})
