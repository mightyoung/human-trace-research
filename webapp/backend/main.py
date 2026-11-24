# -*- coding: utf-8 -*-
"""human-count 可视化/标注/配置 Web 服务（FastAPI）。

零外部依赖（远程 rfdetr 环境自带 fastapi/uvicorn/supervision/cv2）。
- 托管前端静态站（webapp/frontend/dist）
- REST API：相机/轨迹/实体/接力链/指标/掩膜/标签/配置
- 视频 Range 流 + supervision 单帧渲染 / 标注视频导出
"""
from __future__ import annotations
import csv
import io
import json
import mimetypes
import os
import re
import sys
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import evaluate as ev
import render
import algo
import predict as pdc

# 运行时身份选择器（形态条件化模板 + 实时池）；数据缺库时静默降级，不影响控制台其它页
try:
    import selector as selmod
    import selthumb
except Exception as _e:  # pragma: no cover
    selmod = None
    selthumb = None
    print(f"[init] selector 导入失败：{_e}")

# 自动人脸（视频侧人脸模板 → 身份建议 → 复核 → 写池）
try:
    import faceauto as famod
except Exception as _e:  # pragma: no cover
    famod = None
    print(f"[init] faceauto 导入失败：{_e}")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
STATIC_DIR = Path(os.environ.get("WEBAPP_STATIC", str(BASE.parent / "frontend" / "dist")))
MEDIA_ROOT = render.MEDIA_ROOT
CONFIG_PATH = DATA / "config.json"
LABELS_PATH = DATA / "labels.json"

# ---- 台账镜像（可选模块）----
# 老页面的写接口只写 json；这里把改动补进本机台账，避免"双写"只做单向。
# 远程无台账 / 台账出错时一律静默跳过，绝不影响页面保存。
try:
    import ops_api as _opsapi
except Exception:  # pragma: no cover
    _opsapi = None


def _ops_mirror(fn: str, *args, **kwargs) -> None:
    if _opsapi is None:
        return
    mod = getattr(_opsapi, "opin", None)
    if mod is None:
        return
    try:
        getattr(mod, fn)(*args, **kwargs)
    except Exception as e:  # 台账失败不得影响页面
        print(f"[ops] mirror {fn} 失败: {type(e).__name__}: {e}", file=sys.stderr)

app = FastAPI(title="human-count 控制台", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 训练曲线/混淆矩阵等静态图片（注意：不能用 /assets，会遮蔽前端构建产物）
_ASSETS = DATA / "assets"
if _ASSETS.exists():
    app.mount("/figs", StaticFiles(directory=str(_ASSETS)), name="figs")

# 算法工作台数据（ReID 库 / 聚类工作台状态）——启动即加载
try:
    algo.init()
except Exception as _e:  # pragma: no cover
    print(f"[init] algo.init 失败：{_e}")

# 身份选择器复用同一份 store_emb / entity_index；池状态在 data/reid/selector_pool.json
try:
    if selmod:
        selmod.init()
except Exception as _e:  # pragma: no cover
    print(f"[init] selector.init 失败：{_e}")


def _read(name: str, default=None):
    p = DATA / name
    if not p.exists():
        if default is None:
            raise HTTPException(404, f"{name} 不存在")
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _write(name: str, obj):
    (DATA / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "supervision": render.HAS_SV,
        "cv2": render.cv2 is not None,
        "media_root": str(MEDIA_ROOT),
        "media_ok": MEDIA_ROOT.exists(),
        "static": STATIC_DIR.exists(),
        "reid": bool(algo.STORE and algo.STORE.ready),
        "reid_n": len(algo.STORE.meta) if (algo.STORE and algo.STORE.ready) else 0,
        "intent_model": pdc.available(),
    }


@app.on_event("startup")
def _startup():
    try:
        algo.init()
    except Exception as e:  # pragma: no cover
        print(f"[startup] algo.init 失败：{e}")


@app.get("/api/cams")
def cams():
    return _read("cams.json")


@app.get("/api/tracks/{cam}")
def tracks(cam: str):
    return _read(f"tracks/{cam}.json")


@app.get("/api/entities")
def entities():
    return _read("entities.json")


@app.get("/api/suspects")
def suspects():
    """疑似同人（换号）候选：外观像 + 时间紧邻但拼接未连上的相邻实体段。

    只用于页面标注「这段可能是同一个人换号」，**不是判定**，不参与实体图与指标。
    """
    return _read("suspects.json", {"groups": [], "n_groups": 0, "n_pairs": 0})


@app.get("/api/chains")
def chains():
    return _read("chains.json")


@app.get("/api/metrics")
def metrics():
    return _read("metrics.json")


@app.get("/api/gil")
def gil():
    """GIL 统一存储概况（数据代际 / 边构成 / 源哈希），不存在则返回 available=False"""
    return _read("gil.json", {"available": False})


@app.get("/api/masks")
def masks():
    return _read("masks.json", {})


@app.get("/api/mirror")
def mirror():
    return _read("mirror.json", {})


@app.get("/api/pairs")
def pairs():
    return _read("pairs.json", [])


@app.get("/api/labels")
def get_labels():
    return _read("labels.json", {"version": 1, "pair_decisions": {}, "mirror_decisions": {},
                                 "glass_masks": {}, "notes": {}})


@app.post("/api/labels")
def post_labels(payload: dict = Body(...)):
    cur = _read("labels.json", {"version": 1, "pair_decisions": {}, "mirror_decisions": {},
                                "glass_masks": {}, "notes": {}})
    for k in ("pair_decisions", "mirror_decisions", "glass_masks", "notes"):
        if k in payload and isinstance(payload[k], dict):
            cur.setdefault(k, {}).update(payload[k])
    _write("labels.json", cur)
    _ops_mirror("mirror_labels", payload)
    return {"ok": True, "counts": {k: len(cur.get(k, {})) for k in
                                   ("pair_decisions", "mirror_decisions", "glass_masks", "notes")}}


# ---- 人脸库人工标注（独立文件，不动 labels.json / GIL；外挂审计线）----
_FACE_VERDICTS = ("person", "nonperson", "unclear")
_FACE_DECISIONS = ("CONF", "REJ", "MAYBE")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _face_labels() -> dict:
    return _read("faces_labels.json", {"version": 1, "persons": {}, "pairs": {}, "updated": None})


@app.get("/api/faces/labels")
def faces_labels_():
    d = _face_labels()
    return {"ok": True, "updated": d.get("updated"),
            "persons": d.get("persons", {}), "pairs": d.get("pairs", {})}


@app.post("/api/faces/label")
def faces_label(payload: dict = Body(...)):
    """人卡标注：verdict ∈ person/nonperson/unclear，name = 命名（均传 null/'' 清除）。"""
    label = str(payload.get("label") or "").strip()
    if not label:
        raise HTTPException(400, "label 必填")
    cur = _face_labels()
    item = dict(cur.setdefault("persons", {}).get(label) or {})
    if "verdict" in payload:
        v = payload.get("verdict")
        if v is None or v == "":
            item.pop("verdict", None)
        elif v in _FACE_VERDICTS:
            item["verdict"] = v
        else:
            raise HTTPException(400, f"verdict 非法: {v}")
    if "name" in payload:
        n = str(payload.get("name") or "").strip()
        if n:
            item["name"] = n[:60]
        else:
            item.pop("name", None)
    if not (set(item) - {"at"}):     # 除时间戳外无有效字段 → 整体删除
        cur["persons"].pop(label, None)
    else:
        item["at"] = _now()
        cur["persons"][label] = item
    cur["updated"] = _now()
    _write("faces_labels.json", cur)
    _ops_mirror("mirror_face_person", label, cur["persons"].get(label))
    return {"ok": True, "label": label, "item": cur["persons"].get(label)}


@app.post("/api/faces/pair")
def faces_pair(payload: dict = Body(...)):
    """候选合并对裁决：decision ∈ CONF/REJ/MAYBE（null 清除）。"""
    a = str(payload.get("a") or "").strip()
    b = str(payload.get("b") or "").strip()
    if not a or not b:
        raise HTTPException(400, "a/b 必填")
    decision = payload.get("decision")
    key = f"{a}@{b}"
    cur = _face_labels()
    if decision is None or decision == "":
        cur.setdefault("pairs", {}).pop(key, None)
    elif decision in _FACE_DECISIONS:
        cur["pairs"][key] = {"a": a, "b": b, "decision": decision, "at": _now()}
    else:
        raise HTTPException(400, f"decision 非法: {decision}")
    cur["updated"] = _now()
    _write("faces_labels.json", cur)
    _ops_mirror("mirror_face_pair", key, decision)
    return {"ok": True, "key": key, "decision": decision}


# ---- 人员属性标注（独立文件 data/attr_labels.json；不动 labels.json / GIL）----
# 字段集与竞品界面口径对齐，但去掉三项不可由单帧图像得出的：
#   年龄精确值（回归误差 >4 岁，不可信）、运动方向（属轨迹层）、通道/抓拍时间（元数据）。
# 空值 = 未判定（UI 显示为「?」），不写进 labels，避免把「没标」当成「判否」。
ATTR_FIELDS = [
    {"key": "top_color", "label": "上衣颜色", "cols": 6,
     "opts": ["黑", "白", "灰", "红", "橙", "黄", "绿", "蓝", "紫", "棕", "花/多色"]},
    {"key": "bottom_color", "label": "下装颜色", "cols": 6,
     "opts": ["黑", "白", "灰", "红", "橙", "黄", "绿", "蓝", "紫", "棕", "花/多色"]},
    {"key": "top_type", "label": "上衣类型", "cols": 5,
     "hint": "外套 = 穿在其他衣服外面的开襟衣物（夹克/西装外套/风衣）；有外套时按外套记",
     "opts": ["长袖", "短袖", "外套", "裙装", "其他"]},
    {"key": "bottom_type", "label": "下装类型", "cols": 4,
     "opts": ["长裤", "短裤", "裙", "其他"]},
    {"key": "gender", "label": "性别", "cols": 2, "opts": ["男", "女"]},
    {"key": "age_group", "label": "年龄段", "cols": 4,
     "hint": "少年<18 / 青年18–35 / 壮年36–55 / 老年56+（看轮廓与体态，看不清就留空）",
     "opts": ["少年", "青年", "壮年", "老年"]},
    {"key": "glasses", "label": "戴眼镜", "cols": 2, "hint": "需看到正面或侧面脸部", "opts": ["是", "否"]},
    {"key": "mask", "label": "戴口罩", "cols": 2, "opts": ["是", "否"]},
    {"key": "hat", "label": "戴帽子", "cols": 2, "opts": ["是", "否"]},
    {"key": "backpack", "label": "背包", "cols": 2,
     "hint": "背在身上或挎在肩上的包（双肩包/单肩挎包）", "opts": ["是", "否"]},
    {"key": "carrying", "label": "拎东西", "cols": 2,
     "hint": "提在手上或抱着的包/袋/箱；手机、水杯、文件不算", "opts": ["是", "否"]},
]
ATTR_SKIP = ("tiny", "blur", "occluded", "notperson")
ATTR_SKIP_ZH = {"tiny": "框太小", "blur": "画面模糊", "occluded": "严重遮挡", "notperson": "不是人"}


def _attr_labels() -> dict:
    return _read("attr_labels.json", {"version": 1, "updated": None, "labels": {}})


def _attr_items_doc() -> dict:
    return _read("attr_items.json", {"items": [], "count": 0, "band_zh": {}, "canvas": [300, 500]})


@app.get("/api/attr/schema")
def attr_schema():
    d = _attr_items_doc()
    return {
        "fields": ATTR_FIELDS,
        "skip": [{"key": k, "label": v} for k, v in ATTR_SKIP_ZH.items()],
        "band_zh": d.get("band_zh", {}),
        "canvas": d.get("canvas", [300, 500]),
        "count": d.get("count", 0),
        "min_bh": d.get("min_bh"), "min_bw": d.get("min_bw"),
        "skipped": d.get("skipped", {}),
    }


@app.get("/api/attr/predict")
def attr_predict():
    """模型预填建议。

    来源不绑定具体模型：当前是 EG2 图向量 + 逐字段逻辑回归
    （见 tmp/attr/make_prefill.py）。fields 里带每个字段的实测 CV 准确率，
    前端用它决定要不要显示"可信"标记 —— 不要让用户以为置信度就是准确率。
    """
    d = _read("attr_predict.json", {"items": {}, "fields": {}, "conf_threshold": 0.6})
    return {
        "source": d.get("source", ""),
        "conf_threshold": d.get("conf_threshold", 0.6),
        "fields": d.get("fields", {}),
        "count": len(d.get("items", {}) or {}),
        "items": d.get("items", {}) or {},
    }


@app.get("/api/attr/items")
def attr_items():
    d = _attr_items_doc()
    labs = _attr_labels().get("labels", {})
    out = []
    for it in d.get("items", []):
        lb = labs.get(it["key"]) or {}
        out.append({**it,
                    "attrs": lb.get("attrs") or {},
                    "skip": lb.get("skip") or "",
                    "at": lb.get("at")})
    return {"count": len(out), "items": out, "updated": _attr_labels().get("updated")}


@app.post("/api/attr/label")
def attr_label(payload: dict = Body(...)):
    """单条标注：{key, attrs:{field:value}, skip}。attrs 全空且无 skip → 删除该条记录。"""
    key = str(payload.get("key") or "").strip()
    if not key:
        raise HTTPException(400, "key 必填")
    cur = _attr_labels()
    labs = cur.setdefault("labels", {})
    item = dict(labs.get(key) or {})

    if "attrs" in payload:
        valid = {f["key"]: set(f["opts"]) for f in ATTR_FIELDS}
        attrs: dict[str, str] = {}
        for k, v in (payload.get("attrs") or {}).items():
            if k not in valid:
                raise HTTPException(400, f"字段非法: {k}")
            if v in (None, ""):
                continue
            if v not in valid[k]:
                raise HTTPException(400, f"{k} 取值非法: {v}")
            attrs[k] = v
        item["attrs"] = attrs

    if "skip" in payload:
        s = str(payload.get("skip") or "").strip()
        if s and s not in ATTR_SKIP:
            raise HTTPException(400, f"skip 非法: {s}")
        if s:
            item["skip"] = s
        else:
            item.pop("skip", None)

    if not item.get("attrs") and not item.get("skip"):
        labs.pop(key, None)
    else:
        item["at"] = _now()
        labs[key] = item
    cur["updated"] = _now()
    _write("attr_labels.json", cur)
    return {"ok": True, "key": key, "item": labs.get(key)}


@app.get("/api/attr/stats")
def attr_stats():
    """进度 + 每个字段的标注覆盖与取值分布，按框高档 / 机位分层。"""
    d = _attr_items_doc()
    labs = _attr_labels().get("labels", {})
    items = d.get("items", [])
    fields = [f["key"] for f in ATTR_FIELDS]

    done = [it for it in items if labs.get(it["key"])]
    skipped = [it for it in done if (labs[it["key"]] or {}).get("skip")]
    # 标签为真但单帧无证据（如长外套盖住下摆）→ 不进字段分布，否则「裙」会被当成可训练样本
    dropped = [it for it in done if (labs[it["key"]] or {}).get("no_train")]

    def _bucket(pred):
        sel = [it for it in items if pred(it)]
        d_ = [it for it in sel if labs.get(it["key"])]
        return {"n": len(sel), "labeled": len(d_)}

    by_band = {b: _bucket(lambda it, b=b: it.get("band") == b)
               for b in (d.get("band_zh") or {}).keys()}
    by_cam = {}
    for it in items:
        c = it["cam"]
        e = by_cam.setdefault(c, {"n": 0, "labeled": 0})
        e["n"] += 1
        if labs.get(it["key"]):
            e["labeled"] += 1

    per_field = {}
    for f in fields:
        cnt: dict[str, int] = {}
        n = 0
        for it in items:
            lv = labs.get(it["key"]) or {}
            # 跳过（整条不可判）与 no_train（单帧无证据）都不进取值分布，
            # 否则「长裤」会混进 4 条被标为遮挡的条目、页面数字与评估口径对不上
            if lv.get("skip") or lv.get("no_train"):
                continue
            v = (lv.get("attrs") or {}).get(f)
            if v:
                cnt[v] = cnt.get(v, 0) + 1
                n += 1
        per_field[f] = {"n": n, "dist": dict(sorted(cnt.items(), key=lambda kv: -kv[1]))}

    # 按框高档看可判性（每档「该字段被判定」的比例）——决定哪些机位值得开属性
    judge_rate = {}
    for b in by_band:
        sel = [it for it in items if it.get("band") == b and labs.get(it["key"])
               and not ((labs[it["key"]] or {}).get("no_train")
                        or (labs[it["key"]] or {}).get("skip"))]
        if not sel:
            judge_rate[b] = {}
            continue
        judge_rate[b] = {f: round(sum(1 for it in sel
                                      if (((labs[it["key"]] or {}).get("attrs") or {}).get(f))) / len(sel), 3)
                         for f in fields}

    return {
        "count": len(items), "labeled": len(done), "skipped": len(skipped),
        "dropped": len(dropped),
        "progress": round(len(done) / len(items), 4) if items else 0,
        "by_band": by_band, "by_cam": by_cam,
        "per_field": per_field, "judge_rate": judge_rate,
        "skip_zh": ATTR_SKIP_ZH,
        "updated": _attr_labels().get("updated"),
    }


@app.get("/api/attr/export")
def attr_export(fmt: str = "csv"):
    """导出标注：csv（每条一行、字段一列）或 json。"""
    d = _attr_items_doc()
    labs = _attr_labels().get("labels", {})
    fields = [f["key"] for f in ATTR_FIELDS]
    rows = []
    for it in d.get("items", []):
        lb = labs.get(it["key"]) or {}
        if not lb:
            continue
        row = {"key": it["key"], "cam": it["cam"], "tid": it["tid"],
               "frame": it["frame"], "h": it["h"], "w": it["w"], "band": it["band"],
               "entity": it.get("entity", ""),
               "skip": lb.get("skip", ""), "at": lb.get("at", "")}
        for f in fields:
            row[f] = ((lb.get("attrs") or {}).get(f) or "")
        rows.append(row)
    if fmt == "json":
        return {"count": len(rows), "fields": fields, "rows": rows}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["key", "cam", "tid", "frame", "h", "w", "band", "entity",
                                        "skip"] + fields + ["at"])
    w.writeheader()
    w.writerows(rows)
    return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="attr_labels.csv"'})


@app.get("/attr_crops/{name}")
def attr_crop_file(name: str):
    root = (DATA / "attr_crops").resolve()
    p = (root / name).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


_CROP_LOCKS: dict[str, threading.Lock] = {}
_CROP_LOCK_GUARD = threading.Lock()
# ★ 原生 1440p 解码全局串行：按机位分锁时，6 个机位仍可能同时解码。
#   单帧只要 0.2–0.4 s，串行代价远小于并发风险（OpenCV 并发解码会读帧失败，
#   已实测出 404；是否有 native 崩溃未确认，见 docs 的排查记录）。
_NATIVE_DECODE_LOCK = threading.Lock()


def _crop_lock(key: str) -> threading.Lock:
    if key.endswith(":native"):
        return _NATIVE_DECODE_LOCK
    with _CROP_LOCK_GUARD:
        lk = _CROP_LOCKS.get(key)
        if lk is None:
            lk = _CROP_LOCKS[key] = threading.Lock()
        return lk


@app.get("/api/attr/crop")
def attr_crop_live(cam: str, tid: int, f: int | None = None, w: int = 300, src: str = "proxy",
                   pad: float = 0.28, mark: int = 0, scene: int = 0):
    """兜底：预生成图缺失时实时裁剪。

    src=native 从 1440p 原生裁（点开卡片核对用），原生缺失/解码失败时回退代理片，
    并在 `X-Crop-Src` 里说明用了哪一路 —— 别让人以为看到的是原生清晰度。
    pad 是外扩比例（默认 0.28 = 原框的 1.56 倍）；看「人在哪」用 scene=1（固定 16:9 场景窗口，
    比 pad 稳：按倍数外扩在细长人形框上会退化成竖条）。宽画面**必须带 mark=1 画框**：
    画面里往往站着别人，不标框就只能靠猜谁是被检索的那个人。

    ★ 必须按 (机位, 源) 串行解码：页面一次要 5 帧，5 个并发请求同时打开同一个
      1440p h264，OpenCV 会偶发 `decode_slice_header error` → 读帧失败 → 404
      （实测 5 张里挂 2 张，重试同一 URL 又好了）。串行 + 退一帧重试才稳；
      原生解码用**全局锁**（不只是按机位）——并发 1440p 解码曾让进程整体静默退出。
    """
    w2 = max(120, min(960, w))
    p2 = max(0.0, min(2.5, pad))
    root, used = None, "proxy"
    if src == "native" and render.NATIVE_ROOT.exists():
        root, used = render.NATIVE_ROOT, "native"

    def _render(rt, fq):
        with _crop_lock(f"{cam}:{'native' if rt else 'proxy'}"):
            return render.crop_thumb(cam, tid, frame=fq, pad=p2, width=w2,
                                     jpeg_quality=90, root=rt, mark=bool(mark),
                                     scene=bool(scene))

    last: Exception | None = None
    for i in range(3):                       # 失败就挪 ±5 帧再试，肉眼看不出差别
        try:
            img = _render(root, None if f is None else f + i * 5)
            return Response(content=img, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=600",
                                     "X-Crop-Src": used})
        except Exception as e:
            last = e
    if used == "native":                     # 原生三连失败 → 退回代理片（至少给张图）
        try:
            img = _render(None, f)
            return Response(content=img, media_type="image/jpeg",
                            headers={"Cache-Control": "public, max-age=600",
                                     "X-Crop-Src": "proxy"})
        except Exception as e:
            last = e
    raise HTTPException(404, f"裁剪失败: {last}")


# ---- 属性索引：全实体预测结果（独立文件 data/attr_index.json）----
# 与标注池的区别：标注池是「抽样出来的 733 条待标样本」，索引是「全部实体各一条结论」。
# 每个实体带 level（该字段在该框高档位下能不能承诺），前端必须按 level 显示，
# 不能把所有字段一视同仁地当成准确率。
_AI_CACHE = {"mtime": None, "doc": None}


def _attr_index_doc() -> dict:
    p = DATA / "attr_index.json"
    if not p.exists():
        return {}
    mt = p.stat().st_mtime
    if _AI_CACHE["mtime"] != mt:
        _AI_CACHE["doc"] = json.loads(p.read_text(encoding="utf-8"))
        _AI_CACHE["mtime"] = mt
    return _AI_CACHE["doc"] or {}


@app.get("/attr_ent_crops/{name}")
def attr_ent_crop_file(name: str):
    root = (DATA / "attr_ent_crops").resolve()
    p = (root / name).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/attr/index")
def attr_index(conds: str = "", field: str = "", value: str = "", min_conf: float = 0.60,
               min_person: float = 0.50, cam: str = "", band: str = "",
               level: str = "", q: str = "", cap: str = "judge",
               facet_mode: str = "cross",
               limit: int = 60, offset: int = 0):
    """按属性筛实体。

    conds: JSON 数组 `[{"f":"top_color","v":"黑","conf":0.7}, ...]`，**AND** 语义；
           与旧参数 `field`/`value`/`min_conf` 二选一（conds 优先）。取值留空 = 只看该字段有结论。
    cap:   judge（默认，排除该档位不可判的字段）/ ok（只留可承诺档）/ off（不过闸门）
    min_conf: 单条件的置信下限，默认 **0.60**。「黑 + 女」在 .70 下只剩 5 人、.60 下 9 人、
           .50 下 14 人（未过门共 21）——softmax 未校准，每道门都在吃召回，多条件叠加时
           .70 会把结果打到个位数。历史上属性预测的接受口径就是 ≥.6，这里沿用。
    facet_mode: cross（默认，各字段的可选值计数按「其他字段条件已生效」统计，
           经典分面检索语义，选完第一个条件后第二个下拉里的数字才有意义）/
           base（只按机位·档位·搜索过滤后统计，旧行为）

    ★ 能力闸门的必要性：索引里 673 个实体 × 9 字段**全部有值**，但 S 档的下装/眼镜/
      背包/拎东西 level='n/a' —— 模型给了数，却没有可靠证据。不设闸门就会把
      「S 档判出来的背包」当成结论返回，与同一页面上的能力矩阵自相矛盾。
    """
    d = _attr_index_doc()
    ents = d.get("entities") or {}
    if not ents:
        raise HTTPException(404, "尚未生成属性索引（tmp/attr/build_attr_index.py）")
    # 选了字段但没选取值 = 「列出所有该字段有结论的实体」，不是错误
    valid = {f["key"] for f in ATTR_FIELDS}
    if field and field not in valid:
        raise HTTPException(400, f"字段非法: {field}")
    if cap not in ("judge", "ok", "off"):
        raise HTTPException(400, f"cap 非法: {cap}")

    cl: list[dict] = []
    if conds:
        try:
            arr = json.loads(conds)
        except Exception:
            raise HTTPException(400, "conds 不是合法 JSON")
        if not isinstance(arr, list):
            raise HTTPException(400, "conds 必须是数组")
        for c in arr:
            f = str((c or {}).get("f") or "")
            if f not in valid:
                raise HTTPException(400, f"字段非法: {f}")
            cf = (c or {}).get("conf")
            cl.append({"f": f, "v": str((c or {}).get("v") or ""),
                       "conf": float(cf) if cf not in (None, "") else min_conf})
    elif field:
        cl = [{"f": field, "v": value, "conf": min_conf}]

    def cap_ok(lvv) -> bool:
        if cap == "off":
            return True
        if cap == "ok":
            return lvv == "ok"
        return lvv not in (None, "n/a")

    if facet_mode not in ("cross", "base"):
        raise HTTPException(400, f"facet_mode 非法: {facet_mode}")
    cross = facet_mode == "cross"

    def cond_hit(c: dict, attrs: dict, probs: dict) -> bool:
        """单条件命中：取值相符（留空=只要有结论）且置信度达标。不涉及档位闸门。"""
        f, want, cf = c["f"], c["v"], c["conf"]
        av = attrs.get(f)
        if not av or (want and av != want):
            return False
        return float(probs.get(f) or 0) >= cf

    def base_ok(ek: str, v: dict) -> bool:
        """与属性无关的基础过滤：像人门、机位、档位、关键词搜索。"""
        ip = float(v.get("is_person") if v.get("is_person") is not None else 1.0)
        if ip < min_person:
            return False
        if cam and v.get("cam") != cam:
            return False
        if band and v.get("band") != band:
            return False
        if not q:
            return True
        qq = q.strip().lower()
        if qq in ek.lower():
            return True
        # ★ 也搜 VLM 生成的整体描述（「按电梯」「拖把」「低头看电脑」这些
        #   9 个字段里根本没有的词，是这一路信息的唯一价值所在）
        return qq in str(v.get("summary") or "").lower()

    facet = {f["key"]: {} for f in ATTR_FIELDS}
    cap_ex: dict[str, int] = {}
    cap_ex_ents: set[str] = set()
    conf_ex_ents: set[str] = set()
    n_person = 0
    matched = []
    # 每个字段若被选为条件会用的置信门（已在条件里的用它的，否则用默认 min_conf）
    conf_of = {c["f"]: c["conf"] for c in cl}
    for ek, v in ents.items():
        if not base_ok(ek, v):
            continue
        n_person += 1
        attrs = v.get("attrs") or {}
        probs = v.get("prob") or {}
        lv = v.get("level") or {}
        # facets：cross = 基础过滤 + **本字段以外的**条件（分面检索）；base = 只基础过滤
        # ★ 本字段自己的置信门也要算进去：下拉里写「黑（9）」就该是选中后的 9 条，
        #   而不是「判为黑的有 18 个、其中一半置信不够」。两套数会让用户以为筛丢了人。
        for k, val in attrs.items():
            if k not in facet or not cap_ok(lv.get(k)):
                continue
            if float(probs.get(k) or 0) < conf_of.get(k, min_conf):
                continue
            if cross and cl and not all(cond_hit(c, attrs, probs)
                                        for c in cl if c["f"] != k):
                continue
            facet[k][val] = facet[k].get(val, 0) + 1
        if level:
            if cl:
                if lv.get(cl[0]["f"]) != level:
                    continue
            elif level not in lv.values():
                continue
        hit = True
        blocked = []
        for c in cl:
            if not cond_hit(c, attrs, probs):
                hit = False
                break
            if not cap_ok(lv.get(c["f"])):
                blocked.append(c["f"])
        if not hit:
            # 取值都对、档位也能判，只是置信度没过门 —— 单独记一笔，别让「查不到」看起来像「没有」
            if cl and all(
                (attrs.get(c["f"]) == c["v"] if c["v"] else bool(attrs.get(c["f"])))
                and cap_ok(lv.get(c["f"]))
                for c in cl
            ):
                conf_ex_ents.add(ek)
            continue
        if blocked:
            for f in blocked:
                cap_ex[f] = cap_ex.get(f, 0) + 1
            cap_ex_ents.add(ek)
            continue
        matched.append((ek, v, float(v.get("is_person")
                                     if v.get("is_person") is not None else 1.0)))

    # 空结果诊断：逐个去掉一条条件还剩多少人 —— 直接指出是哪一条把结果清空了
    drop_one = []
    if cl and not matched:
        for i, c in enumerate(cl):
            rest = [x for j, x in enumerate(cl) if j != i]
            n = 0
            for ek, v in ents.items():
                if not base_ok(ek, v):
                    continue
                attrs = v.get("attrs") or {}
                probs = v.get("prob") or {}
                lv = v.get("level") or {}
                if not all(cond_hit(x, attrs, probs) for x in rest):
                    continue
                if not all(cap_ok(lv.get(x["f"])) for x in rest):
                    continue
                n += 1
            drop_one.append({"f": c["f"], "v": c["v"], "n": n})
        drop_one.sort(key=lambda x: -x["n"])

    def _score(t) -> float:
        pr = t[1].get("prob") or {}
        if not cl:
            return 0.0
        return min(float(pr.get(c["f"]) or 0) for c in cl)

    if cl:
        matched.sort(key=lambda t: (-_score(t), t[0]))
    else:
        matched.sort(key=lambda t: (-t[2], t[0]))
    page = matched[offset:offset + max(1, min(300, limit))]
    items = []
    for ek, v, ip in page:
        items.append({"entity": ek, "cam": v.get("cam"), "band": v.get("band"),
                      "h": v.get("h"), "n_crop": v.get("n_crop"),
                      "is_person": round(ip, 3),
                      "attrs": v.get("attrs") or {}, "prob": v.get("prob") or {},
                      "agree": v.get("agree") or {}, "level": v.get("level") or {},
                      # ★ VLM 生成的整体外观描述（不在 9 个字段里，是额外的一路信息：
                      #   含动作/职业/交互对象，如「戴眼镜男子按电梯」「女保洁员手持拖把」）。
                      #   没跑过 VLM 的实体这里是 None，前端按空处理。
                      "summary": v.get("summary"),
                      "crops": (v.get("crops") or [])[:3]})
    return {
        "total": len(ents), "n_person": n_person, "matched": len(matched),
        "offset": offset, "limit": limit, "items": items,
        "conds": cl, "cap": cap, "facet_mode": facet_mode,
        "cap_excluded": len(cap_ex_ents), "cap_excluded_by_field": cap_ex,
        "conf_excluded": len(conf_ex_ents),
        "drop_one": drop_one,
        "facets": facet,
        "fields": [{"key": f["key"], "label": f["label"], "opts": f["opts"]}
                   for f in ATTR_FIELDS if f["key"] in (d.get("fields") or {})],
        "capability": d.get("capability") or {},
        "meta": {k: d.get(k) for k in ("built", "source", "n_train", "n_pa", "dom_reps",
                                       "heads_built", "is_person_auc", "match_check",
                                       "match_check_note", "fields")},
    }


# ---- 属性索引：检索结果导出 ----
# ★ 导出必须走**同一套筛选**，不能前端把当前页 60 条存下来：用户看到的是分页后的一页，
#   而「把一批人交出去」的价值在全部命中。这里复用 attr_index 翻页取全量（每页 300），
#   筛选语义零复制 —— 否则两个入口迟早给出不同的人。
_ENT_CACHE = {"mtime": None, "doc": None}


def _entities_doc() -> dict:
    p = DATA / "entities.json"
    if not p.exists():
        return {}
    mt = p.stat().st_mtime
    if _ENT_CACHE["mtime"] != mt:
        _ENT_CACHE["doc"] = json.loads(p.read_text(encoding="utf-8"))
        _ENT_CACHE["mtime"] = mt
    return _ENT_CACHE["doc"] or {}


def _attr_all_matched(**kw) -> dict:
    """按给定条件取全部命中实体（跳过 300 条分页上限）。"""
    rows: list[dict] = []
    first = None
    offset = 0
    for _ in range(20):                      # 20×300 = 6000，远大于全库 1028
        r = attr_index(**{**kw, "limit": 300, "offset": offset})
        first = first or r
        rows.extend(r.get("items") or [])
        offset += 300
        if offset >= int(r.get("matched") or 0):
            break
    out = dict(first or {})
    out["items"] = rows
    return out


# 导出列顺序与页面一致（年龄段已从索引移除；下装用「下装长度」口径，见 README）
_EXPORT_FIELDS = ["gender", "top_color", "top_type", "bottom_color", "bottom_type",
                  "glasses", "hat", "backpack", "carrying"]
_EXPORT_LABEL = {"bottom_type": "下装长度"}
_EXPORT_VAL = {"bottom_type": {"长裤": "过膝", "短裤": "未过膝"}}

_EXPORT_README = """属性索引检索结果导出
====================================================================

文件
  entities.csv    每条实体一行；9 个字段各拆成「取值 + 置信度」两列
  crops/          代表帧（which=all 时为该实体的全部候选帧），文件名 = 机位_轨迹_帧号.jpg
  query.json      导出时的检索条件（conds / 档位 / 机位 / 像人门）

怎么定位原片
  「机位」「视频文件」「起始帧 / 结束帧」三者一起用：视频是
  media-resource/multi-cemra/long-35min/{机位}_long35.mp4，按帧号打开即可。
  「起始秒」= 起始帧 / 帧率（25 fps），是**段内相对时间**，不是 OSD 绝对时钟；
  要跨机位对时，请回原片看画面上的 OSD，别用这一列做加减。

怎么读置信度
  「置信度」是模型置信度，**不是准确率**。某个字段整体准多少，看控制台「能承诺到什么
  程度」表里该字段的留出指标：性别 .960 / 上衣色 .884 / 上衣型 .930 / 下装色 .905 /
  眼镜 .991 / 帽 .982 可交付；背包 AP .458、拎东西 AP .656 只当提示（P≥.9 时召回 11% / 18%）。
  softmax 未校准，多条件叠加时召回是乘性的 —— 导出条数少于预期，先把置信门往下调。

怎么读「不可判（X 档）」
  框高档位（S <80px / M 80–240 / L ≥240）下该字段没有可靠证据：模型仍然会给出一个值，
  但索引按能力矩阵把它标成不可判，**不参与筛选、不进结论**。CSV 里写成
  「不可判（S 档）」，置信度列留空而不是写 0 —— 空表示「没有证据」，0 表示「证据指向反面」。

下装长度
  「过膝 / 未过膝」对应模型原始类「长裤 / 短裤」。单帧分不出裙与短裤（长外套盖住胯部时
  两者在画面上无差别，实测 20/23 是同一个人），索引不承诺这个区分。
"""


@app.get("/api/attr/index/export")
def attr_index_export(conds: str = "", field: str = "", value: str = "", min_conf: float = 0.60,
                      min_person: float = 0.50, cam: str = "", band: str = "",
                      level: str = "", q: str = "", cap: str = "judge",
                      facet_mode: str = "cross", fmt: str = "csv", which: str = "rep"):
    """把当前检索条件命中的**全部**实体导出（不受页面 60 条分页限制）。

    fmt=csv   每条一行，UTF-8 BOM（Excel 直接打开中文不乱码）
    fmt=zip   截图包：entities.csv + query.json + README.txt + crops/
              which=rep 每实体一张代表帧（默认）/ which=all 该实体全部候选帧
    """
    if fmt not in ("csv", "zip"):
        raise HTTPException(400, f"fmt 非法: {fmt}")
    if which not in ("rep", "all"):
        raise HTTPException(400, f"which 非法: {which}")
    r = _attr_all_matched(conds=conds, field=field, value=value, min_conf=min_conf,
                          min_person=min_person, cam=cam, band=band, level=level,
                          q=q, cap=cap, facet_mode=facet_mode)
    items = r.get("items") or []
    if fmt == "zip" and not items:
        raise HTTPException(400, "没有命中的实体，无法打包")

    idx = _attr_index_doc().get("entities") or {}
    ent = _entities_doc()
    cams = {c.get("id"): c for c in (_read("cams.json", {}) or {}).get("cameras", [])}
    labels = {f["key"]: f["label"] for f in ATTR_FIELDS}
    conds_used = "；".join(
        f"{_EXPORT_LABEL.get(c['f']) or labels.get(c['f'], c['f'])}="
        f"{_EXPORT_VAL.get(c['f'], {}).get(c['v'], c['v']) or '任意'}≥{c['conf']:.2f}"
        for c in (r.get("conds") or [])
    ) or "（无条件：全部有效实体）"

    head = ["实体编号", "机位", "机位名", "视频文件", "轨迹号", "起始帧", "结束帧",
            "起始秒", "起始时刻", "时长秒", "框高档位", "框高px", "帧数", "像人置信度"]
    for k in _EXPORT_FIELDS:
        lb = _EXPORT_LABEL.get(k) or labels.get(k, k)
        head += [lb, lb + "置信度"]
    head += ["不可判字段", "命中条件"]

    rows_out = []
    for it in items:
        ek, camid = it["entity"], str(it.get("cam") or "")
        m = ((ent.get(camid) or {}).get("entities") or {}).get(ek) or {}
        fps = float((cams.get(camid) or {}).get("fps") or 25.0)
        f0, f1 = m.get("f0"), m.get("f1")
        bandv = it.get("band") or ""
        attrs, prob, lv = it.get("attrs") or {}, it.get("prob") or {}, it.get("level") or {}
        na = []
        t0 = f0 / fps if isinstance(f0, (int, float)) else None
        row = [ek, camid, (cams.get(camid) or {}).get("name", ""),
               f"{camid}_long35.mp4",
               "|".join(str(t) for t in (m.get("tracks") or [])),
               f0 if f0 is not None else "", f1 if f1 is not None else "",
               round(t0, 1) if t0 is not None else "",
               f"{int(t0 // 60):02d}:{t0 % 60:04.1f}" if t0 is not None else "",
               round((f1 - f0) / fps, 1) if isinstance(f0, (int, float))
               and isinstance(f1, (int, float)) else "",
               bandv, round(float(it.get("h") or 0), 1), it.get("n_crop"), it.get("is_person")]
        for k in _EXPORT_FIELDS:
            if lv.get(k) == "n/a":
                na.append(_EXPORT_LABEL.get(k) or labels.get(k, k))
                row += [f"不可判（{bandv} 档）", ""]     # ★ 空 ≠ 0：没证据 vs 证据指向反面
            else:
                v = _EXPORT_VAL.get(k, {}).get(attrs.get(k), attrs.get(k)) or ""
                row += [v, round(float(prob.get(k) or 0), 3)]
        row += ["·".join(na), conds_used]
        rows_out.append(row)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(head)
    for row in rows_out:
        w.writerow([_csv_safe(v) for v in row])
    csv_bytes = buf.getvalue().encode("utf-8-sig")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    if fmt == "csv":
        return Response(content=csv_bytes, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="attr_index_{stamp}.csv"'})

    qdoc = {"exported_at": datetime.now().isoformat(timespec="seconds"),
            "matched": len(items), "conds": r.get("conds") or [],
            "conds_text": conds_used, "cap": r.get("cap"), "facet_mode": r.get("facet_mode"),
            "min_person": min_person, "cam": cam, "band": band, "q": q,
            "which": which,
            "note": "conds 为后端实际生效的条件（含每条自带的置信门）；空数组 = 无条件"}
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("entities.csv", csv_bytes)
        z.writestr("README.txt", _EXPORT_README)
        z.writestr("query.json", json.dumps(qdoc, ensure_ascii=False, indent=1))
        root = DATA.resolve() / "attr_ent_crops"
        for it in items:
            crops = ((idx.get(it["entity"]) or {}).get("crops") or [])[: None if which == "all" else 1]
            for name in crops:
                p = root / name
                if p.is_file():
                    z.write(str(p), arcname=f"crops/{name}")
    return Response(content=zbuf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="attr_index_{stamp}.zip"'})


def _csv_safe(v) -> str:
    """防 CSV 公式注入：以 = + - @ 开头的值加前导单引号。"""
    s = str(v if v is not None else "")
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


# ---- 属性索引：单个实体的画像（点开卡片看原片）----
# ★ 一张代表帧不够判断：坐姿/走动/远近会让同一个人看起来完全不像（实测框高跨度中位 3.76×）。
#   这里按实体**时间段均分**取帧，每帧都实时从原片裁，附带帧号与 mm:ss —— 用户能自己回原片核对。
_tracks_cache: dict[str, tuple[float, dict]] = {}


# ---- 存命名查询：属性检索的条件组合（独立文件 data/attr_queries.json）----
# 为什么要有这一层：一次检索是「条件 × 机位 × 档位 × 闸门」的组合，重建一遍要四五步；
# 而现场真正反复问的就那么几句（找穿黑衣戴帽的人 / 找拎包进走廊的人）。
# 只存「怎么问」，不存结果 —— 名单每次重算，索引更新后命名查询自动跟着变，
# 不会出现「存下来的名单和当前数据打架」。
_ATTR_QFILE = "attr_queries.json"
_QCAP = ("judge", "ok", "off")


def _attr_queries() -> dict:
    return _read(_ATTR_QFILE, {"version": 1, "updated": None, "items": []})


def _clean_conds(raw, valid: set[str]) -> list[dict]:
    """校验并归一化条件数组；valid 传**索引实际有的字段**（不含 age_group 这类已删字段）。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(400, "conds 必须是数组")
    out, seen = [], set()
    for c in raw:
        f = str((c or {}).get("f") or "")
        if f not in valid:
            raise HTTPException(400, f"字段非法: {f}")
        if f in seen:                    # 同字段两条条件恒为空，前端也去重，这里兜底
            continue
        seen.add(f)
        conf = (c or {}).get("conf")
        conf = 0.6 if conf in (None, "") else float(conf)
        out.append({"f": f, "v": str((c or {}).get("v") or ""),
                    "conf": max(0.5, min(0.99, conf))})
    return out


@app.get("/api/attr/queries")
def attr_queries_list():
    d = _attr_queries()
    items = sorted(d.get("items") or [], key=lambda x: (-int(x.get("n_used") or 0), x.get("name") or ""))
    return {"items": items, "updated": d.get("updated")}


@app.post("/api/attr/queries")
def attr_queries_save(payload: dict = Body(...)):
    """按 id 更新，按名字去重新建；返回落库后的条目。"""
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "名字不能为空")
    if len(name) > 24:
        raise HTTPException(400, "名字最多 24 个字")
    cap = str(payload.get("cap") or "judge")
    if cap not in _QCAP:
        raise HTTPException(400, f"cap 非法: {cap}")
    cam = str(payload.get("cam") or "")
    band = str(payload.get("band") or "")
    idx_fields = set((_attr_index_doc().get("fields") or {}).keys())
    conds = _clean_conds(payload.get("conds"), idx_fields or {f["key"] for f in ATTR_FIELDS})
    if not conds and not cam and not band:
        raise HTTPException(400, "至少一个条件（或限定了机位/档位）才值得存")

    d = _attr_queries()
    items = d.get("items") or []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    qid = str(payload.get("id") or "")
    hit = None
    if qid:
        hit = next((x for x in items if x.get("id") == qid), None)
    if hit is None:
        hit = next((x for x in items if x.get("name") == name), None)
    if hit is None:
        # ★ id 必须真的唯一：秒级时间戳在同一秒内连存两条会撞，撞了之后下拉里
        #   两个选项 value 相同，选谁都套用第一条（实测踩过）。加毫秒 + 兜底自增。
        base = "q" + datetime.now().strftime("%m%d%H%M%S%f")[:-3]
        qid_new, k = base, 1
        while any(x.get("id") == qid_new for x in items):
            k += 1
            qid_new = f"{base}-{k}"
        hit = {"id": qid_new, "n_used": 0}
        items.append(hit)
    hit.update({"name": name, "conds": conds, "cam": cam, "band": band,
                "cap": cap, "updated": now})
    d["items"] = items
    d["updated"] = now
    _write(_ATTR_QFILE, d)
    return {"item": hit, "count": len(items)}


@app.delete("/api/attr/queries/{qid}")
def attr_queries_del(qid: str):
    d = _attr_queries()
    items = [x for x in (d.get("items") or []) if x.get("id") != qid]
    d["items"] = items
    d["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _write(_ATTR_QFILE, d)
    return {"ok": True, "count": len(items)}


@app.post("/api/attr/queries/{qid}/used")
def attr_queries_used(qid: str):
    """记一次使用（用于把常用的排前面）。失败不影响检索，故容错。"""
    d = _attr_queries()
    for x in d.get("items") or []:
        if x.get("id") == qid:
            x["n_used"] = int(x.get("n_used") or 0) + 1
            x["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _write(_ATTR_QFILE, d)
    return {"ok": True}


def _tracks_doc(cam: str) -> dict:
    """带 mtime 缓存的轨迹表：每开一张卡片要吸附 5 帧，不能每次重解析几 MB JSON。"""
    p = DATA / "tracks" / f"{cam}.json"
    if not p.exists():
        return {}
    mt = p.stat().st_mtime
    hit = _tracks_cache.get(cam)
    if not hit or hit[0] != mt:
        hit = (mt, json.loads(p.read_text(encoding="utf-8")))
        _tracks_cache[cam] = hit
    return hit[1]


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


@app.get("/api/attr/entity")
def attr_entity(entity: str, n: int = 5, w: int = 420):
    """单个实体的时间段、轨迹与 n 张时间线帧（实时裁，不依赖预生成图）。

    ★ 帧的落点必须吸附到**该实体某一轨的观测点**：实体跨多个碎片轨，
      按 [f0,f1] 硬取中点很可能落在两段之间的空档里，裁出来是别人的半张脸。
    """
    v = (_attr_index_doc().get("entities") or {}).get(entity)
    if not v:
        raise HTTPException(404, f"实体不存在: {entity}")
    cam = str(v.get("cam") or "")
    m = ((_entities_doc().get(cam) or {}).get("entities") or {}).get(entity) or {}
    tr = _tracks_doc(cam)
    pts_by_tid = tr.get("tracks") or {}
    tids = [int(t) for t in (m.get("tracks") or [])]
    cams = {c.get("id"): c for c in (_read("cams.json", {}) or {}).get("cameras", [])}
    fps = float((cams.get(cam) or {}).get("fps") or 25.0)
    f0, f1 = m.get("f0"), m.get("f1")

    def snap(fx: float):
        """把目标帧吸附到该实体任一轨最近的观测点；返回 (帧号, 轨号)。"""
        best = None
        for tid in tids:
            for p in pts_by_tid.get(str(tid)) or []:
                df = abs(p[0] - fx)
                if best is None or df < best[0]:
                    best = (df, int(p[0]), tid)
        return (best[1], best[2]) if best else (None, None)

    frames, seen = [], set()
    if isinstance(f0, (int, float)) and isinstance(f1, (int, float)) and f1 >= f0:
        n = max(1, min(9, int(n)))
        targets = [f0] if f1 == f0 else [f0 + (f1 - f0) * i / (n - 1) for i in range(n)]
        for tf in targets:
            f, tid = snap(tf)
            if f is None or (f, tid) in seen:
                continue
            seen.add((f, tid))
            frames.append({"f": f, "tid": tid, "t": round(f / fps, 1), "mmss": _mmss(f / fps),
                           "url": f"/api/attr/crop?cam={cam}&tid={tid}&f={f}"
                                  f"&w={int(w)}&src=native"})
    t0 = f0 / fps if isinstance(f0, (int, float)) else None
    t1 = f1 / fps if isinstance(f1, (int, float)) else None
    na = [f["label"] for f in ATTR_FIELDS
          if (v.get("level") or {}).get(f["key"]) == "n/a"]
    # 带环境的宽画面：中段那一帧，pad 放到 1.1（≈原框 3.2 倍）——只给特写判不了「他在哪」
    ctx = None
    if frames:
        mf = frames[len(frames) // 2]
        ctx = {"f": mf["f"], "tid": mf["tid"], "mmss": mf["mmss"],
               "url": f"/api/attr/crop?cam={cam}&tid={mf['tid']}&f={mf['f']}"
                      f"&w=960&scene=1&mark=1&src=native"}
    return {
        "entity": entity, "cam": cam, "cam_name": (cams.get(cam) or {}).get("name", ""),
        "band": v.get("band"), "h": v.get("h"), "n_crop": v.get("n_crop"),
        "is_person": v.get("is_person"), "tracks": tids, "fps": fps,
        "f0": f0, "f1": f1,
        "t0": round(t0, 1) if t0 is not None else None,
        "t1": round(t1, 1) if t1 is not None else None,
        "dur": round(t1 - t0, 1) if t0 is not None else None,
        "mmss": {"t0": _mmss(t0) if t0 is not None else "",
                 "t1": _mmss(t1) if t1 is not None else ""},
        # ★ 代理片（960×540）；原生 1440p 在素材盘，这里只做「快速核对」
        "video": f"/media/{cam}.mp4",
        "video_note": "网页版为 960×540 代理，原生 1440p 在 media-resource/multi-cemra/long-35min/",
        "frames": frames,
        "context": ctx,
        "crops": v.get("crops") or [],
        "attrs": v.get("attrs") or {}, "prob": v.get("prob") or {},
        "level": v.get("level") or {}, "agree": v.get("agree") or {},
        "na": na,
    }


# ═══ 单图属性试跑：本地后端代理到远程算法容器 ═══
# 为什么不在浏览器里直连 192.168.11.16:18080：
#   ① 算法服务没有 CORS，跨源读 JSON 会被浏览器拦；
#   ② 浏览器所在网段不一定通算力机，后端一定通；
#   ③ 上传大小、超时、错误文案只在一处把关。
ALGO_ATTR_URL = os.environ.get("HC_ATTR_URL", "http://192.168.11.16:18080").rstrip("/")
MAX_INFER_BYTES = int(os.environ.get("HC_ATTR_MAX_MB", "12")) * 1024 * 1024


@app.get("/api/attr/infer/status")
def attr_infer_status():
    """探活 + 版本：页面要如实显示算法服务能不能用，不能假装能跑。"""
    import requests
    t0 = time.time()
    try:
        # ★ trust_env=False：局域网地址不该走系统 HTTP 代理（实测开着代理时会被拦成 502，
        #   报错文案还变成代理的 "upstream connect failed"，排查时会误导）
        with requests.Session() as sess:
            sess.trust_env = False
            r = sess.get(f"{ALGO_ATTR_URL}/healthz", timeout=4)
        r.raise_for_status()
        return {"ok": True, "url": ALGO_ATTR_URL, "health": r.json(),
                "ms": round((time.time() - t0) * 1000, 1)}
    except Exception as e:
        return {"ok": False, "url": ALGO_ATTR_URL, "error": f"{type(e).__name__}: {e}"[:200],
                "hint": "远程 docker ps | grep hc-attr；没跑就 docker start hc-attr"}


def _post_image(raw: bytes, params: dict) -> dict:
    """把图转发给 /v1/analyze。requests 是同步的，调用方必须放进线程池。"""
    import requests
    try:
        with requests.Session() as sess:      # 同上：不走系统代理
            sess.trust_env = False
            r = sess.post(f"{ALGO_ATTR_URL}/v1/analyze", data=raw, params=params,
                          headers={"Content-Type": "application/octet-stream"}, timeout=120)
    except requests.RequestException as e:
        raise HTTPException(503, f"算法服务不可达（{ALGO_ATTR_URL}）：{type(e).__name__}。"
                                 f"远程确认 docker ps | grep hc-attr")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text[:300]
        raise HTTPException(502 if r.status_code >= 500 else r.status_code,
                            f"算法服务返回 {r.status_code}：{detail}")
    return r.json()


@app.post("/api/attr/infer")
async def attr_infer(request: Request, crops: int = 1, boxes: str = "", drop_non_person: int = 0):
    """上传一张图 → 每个人 + 9 个属性。

    请求体 = 图片字节（与算法服务同一套约定，不用 multipart）。
    ★ 必须 async 读 body + 线程池转发：一次推理最长 120 s，直接在事件循环里跑
      会把整页其它接口（列表、裁图、导出）一起卡住。
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "请求体为空：直接把图片字节放进 body（不是 multipart）")
    if len(raw) > MAX_INFER_BYTES:
        raise HTTPException(413, f"图片 {len(raw) / 2**20:.1f} MB，超过上限 "
                                 f"{MAX_INFER_BYTES // 2**20} MB")
    params = {"crops": int(bool(crops)), "drop_non_person": int(bool(drop_non_person))}
    if boxes:
        params["boxes"] = boxes
    t0 = time.time()
    d = await run_in_threadpool(_post_image, raw, params)
    d["_proxy"] = {"url": ALGO_ATTR_URL, "ms": round((time.time() - t0) * 1000, 1)}
    return d


@app.get("/api/faces/export")
def faces_export():
    d = _face_labels()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kind", "key", "verdict_or_decision", "name", "cam_tid_a", "cam_tid_b", "at"])
    for label, it in sorted((d.get("persons") or {}).items()):
        w.writerow(["person", _csv_safe(label), _csv_safe(it.get("verdict", "")),
                    _csv_safe(it.get("name", "")), "", "", _csv_safe(it.get("at", ""))])
    for key, it in sorted((d.get("pairs") or {}).items()):
        w.writerow(["pair", _csv_safe(key), _csv_safe(it.get("decision", "")),
                    "", _csv_safe(it.get("a", "")), _csv_safe(it.get("b", "")), _csv_safe(it.get("at", ""))])
    return Response(content="\ufeff" + buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=faces_labels.csv"})


@app.get("/api/config")
def get_config():
    return _read("config.json", ev.DEFAULT_PARAMS) if CONFIG_PATH.exists() else ev.DEFAULT_PARAMS


@app.post("/api/config")
def post_config(payload: dict = Body(...)):
    _write("config.json", payload)
    return {"ok": True, "config": payload}


@app.post("/api/evaluate")
def evaluate(params: dict = Body(default={})):
    merged = dict(ev.DEFAULT_PARAMS)
    merged.update(params or {})
    return ev.evaluate(merged)


@app.post("/api/sweep")
def sweep(payload: dict = Body(...)):
    return ev.sweep(payload.get("base", {}), payload.get("key", "sp_threshold"),
                    payload.get("values", []))


@app.get("/api/frame")
def frame(cam: str, f: int = 0, view: str = "all", trail: int = 0, masks: int = 1, arm: str = ""):
    try:
        img = render.render_frame(cam, f, view=view, trail=trail, show_masks=bool(masks),
                                  arm=(arm or None))
    except Exception as e:
        raise HTTPException(500, str(e))
    return Response(content=img, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=60"})


@app.post("/api/export")
def export(payload: dict = Body(...)):
    cam = payload["cam"]
    f0 = int(payload.get("f0", 0))
    f1 = int(payload.get("f1", 1500))
    view = payload.get("view", "all")
    trail = int(payload.get("trail", 12))
    arm = payload.get("arm") or None
    job_id, path = render.export_video(cam, f0, f1, view=view, trail=trail, arm=arm)
    return {"job_id": job_id, "path": path}


@app.get("/api/export/status/{job_id}")
def export_status(job_id: str):
    return render.job_status(job_id)


# ================= 算法工作台：ReID / 语义 / 聚类 / 预测 =================

def _tracks(cam: str) -> dict:
    return render._load_tracks(cam)


@app.get("/api/track/thumb")
def track_thumb(cam: str, tid: int, f: int | None = None, w: int = 192):
    try:
        img = render.crop_thumb(cam, tid, frame=f, width=max(64, min(384, w)))
    except Exception as e:
        raise HTTPException(404, str(e))
    return Response(content=img, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=600"})


@app.get("/api/reid/store")
def reid_store():
    if not (algo.STORE and algo.STORE.ready):
        raise HTTPException(503, "ReID 库未加载")
    s = algo.STORE
    return {"n": len(s.meta), "version": getattr(s, "version", {}),
            "phrases": s.vocab_keys,
            "tracks": s.meta}


@app.get("/api/reid/similar")
def reid_similar(cam: str, tid: int, k: int = 12):
    if not (algo.STORE and algo.STORE.ready):
        raise HTTPException(503, "ReID 库未加载")
    r = algo.STORE.similar(cam, tid, k)
    if r is None:
        raise HTTPException(404, f"{cam}#{tid} 不在 ReID 库")
    return r


@app.get("/api/reid/text")
def reid_text(q: str, k: int = 24):
    if not (algo.STORE and algo.STORE.ready):
        raise HTTPException(503, "ReID 库未加载")
    return algo.STORE.search_text(q, k)


@app.get("/api/reid/entity")
def reid_entity(cam: str, tid: int):
    if not (algo.STORE and algo.STORE.ready):
        raise HTTPException(503, "ReID 库未加载")
    return algo.STORE.entity_of(cam, tid)


@app.get("/api/reid/report")
def reid_report():
    return _read("reid/report.json", {})


@app.get("/api/reid/text-eval")
def reid_text_eval():
    return _read("reid/text_eval.json", {})


# ---- 运行时身份选择器（形态条件化模板 + 人脸触发实时池）----
# 口径全写在 selector.py 顶部，界面文案与这里保持同一套说法：
#   · 池的触发源是**人工确认过的人脸卡**，不是自动人脸识别；识别全对时的收益是 8 段那组数
#   · 查询一律留一（查询轨从模板与池里剔除），否则 top-1 恒等于自己
#   · 库与池默认按机位分层：外观跨机位在实测里是随机水平，scope=all 只作对照
def _sel():
    if not (selmod and selmod.SEL and selmod.SEL.ready):
        raise HTTPException(503, "身份选择器未就绪（ReID 库未加载）")
    return selmod.SEL


@app.get("/api/selector/state")
def selector_state():
    return _sel().state()


@app.get("/api/selector/tracks")
def selector_tracks(cam: str = "", only_multi: int = 1):
    s = _sel()
    rows = s.tracks()
    if cam:
        rows = [r for r in rows if r["cam"] == cam]
    if only_multi:
        rows = [r for r in rows if r["n_group"] >= 2]
    return {"n": len(rows), "tracks": rows}


@app.get("/api/selector/query")
def selector_query(cam: str, tid: int, scope: str = "cam", tau: float = 0.80,
                   mode: str = "abs", margin: float = 0.10, floor: float = 0.60):
    try:
        return _sel().query(cam, tid, scope=scope, tau=tau,
                            mode=mode, margin=margin, floor=floor)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/selector/thumb")
def selector_thumb(cam: str, tid: int, w: int = 160):
    """轨迹缩略图（候选 / 查询轨 / 池共用）。

    ★ 必须按库里的 (u, v, h_med, f_rep) 裁 —— 库的 tid 与 tracks 不同源，
      按 tid 去 tracks 查框会裁到别人（实测同 tid 框高差 20–68%）。
    """
    s = _sel()
    try:
        p = selthumb.ensure(s, cam, tid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return Response(content=selthumb.resize_bytes(p, max(48, min(512, w))),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=604800"})


@app.post("/api/selector/witness")
def selector_witness(payload: dict = Body(...)):
    """把某条轨的全身特征写进某身份的实时池（人脸卡触发 / 人工确认）。"""
    entity = str(payload.get("entity") or "").strip()
    cam = str(payload.get("cam") or "").strip()
    if not entity or not cam or payload.get("tid") is None:
        raise HTTPException(400, "entity / cam / tid 必填")
    try:
        return _sel().witness(entity, cam, int(payload["tid"]),
                              src=str(payload.get("src") or "manual"),
                              q=payload.get("q"), note=str(payload.get("note") or ""),
                              scope=str(payload.get("scope") or "cam"),
                              force=bool(payload.get("force")))
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/selector/revoke")
def selector_revoke(payload: dict = Body(...)):
    entity = str(payload.get("entity") or "").strip()
    cam = str(payload.get("cam") or "").strip()
    if not entity or not cam or payload.get("tid") is None:
        raise HTTPException(400, "entity / cam / tid 必填")
    return _sel().revoke(entity, cam, int(payload["tid"]),
                         scope=str(payload.get("scope") or "cam"))


@app.post("/api/selector/reset")
def selector_reset():
    return _sel().reset_pool()


@app.post("/api/selector/params")
def selector_params(payload: dict = Body(...)):
    return _sel().set_params(sigma=payload.get("sigma"), pool_k=payload.get("pool_k"),
                             pool_max=payload.get("pool_max"), min_h=payload.get("min_h"))


@app.get("/api/selector/eval")
def selector_eval(scope: str = "cam", tau: float = 0.80, mode: str = "real",
                  split: str = "loo", cam: str = ""):
    try:
        return _sel().evaluate(scope=scope, tau=tau, mode=mode, split=split, cam=cam)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/selector/bind_cost")
def selector_bind_cost(tau: float = 0.80, online_tau: float = 0.85,
                       online_margin: float = 0.05, face_gate: float = 0.50,
                       face_margin: float = 0.05):
    """实时写池的四方对账：不写 / 自学习 / 自学习上限 / 真实人脸绑定。

    ★ 自动写池默认**不开**：实测自学习误写 32/55、人脸绑定精度 0.46，
      两者都让选择器略降（-0.6 / -0.2 pt）。上限（online_or）也只有 +0.2 pt，
      因为本包的池内轨本就在历史库里 —— 没有跨时段结构，测不出池的增益。
    """
    try:
        return _sel().bind_cost(tau=tau, online_tau=online_tau,
                                online_margin=online_margin,
                                face_gate=face_gate, face_margin=face_margin)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/selector/auto_face")
def selector_auto_face():
    """自动人脸的身份建议（缓存）。口径与数字全在 faceauto.py 顶部与 meta 里。"""
    if famod is None:
        raise HTTPException(503, "faceauto 未加载")
    d = famod.load_suggest()
    if not d.get("ok"):
        raise HTTPException(404, d.get("reason") or "无建议")
    return d


@app.post("/api/selector/auto_face/refresh")
def selector_auto_face_refresh(payload: dict = Body(default={})):
    """用已采集的人脸证据重算建议（约 2–3 s）。视频侧采集是离线脚本，不在这里跑。"""
    if famod is None:
        raise HTTPException(503, "faceauto 未加载")
    d = famod.suggest()
    if not d.get("ok"):
        raise HTTPException(400, d.get("reason") or "重算失败")
    return {k: v for k, v in d.items() if k != "suggestions"} | {
        "n_suggestions": len(d.get("suggestions") or [])}


@app.post("/api/selector/auto_face/apply")
def selector_auto_face_apply(payload: dict = Body(...)):
    """把勾选的建议写成选择器的池条目。keys 为空时按 level/min_score 全选。"""
    if famod is None:
        raise HTTPException(503, "faceauto 未加载")
    keys = payload.get("keys") or None
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.split(",") if k.strip()]
    try:
        return famod.apply_to_pool(keys=keys,
                                   min_score=payload.get("min_score"),
                                   level=payload.get("level") or None)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@app.get("/api/selector/auto_face/crop")
def selector_auto_face_crop(f: str):
    p = (DATA / "reid" / "face_q" / Path(f).name)
    if not p.exists():
        raise HTTPException(404, "crop 不存在")
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@app.get("/api/selector/auto_face/cards")
def selector_auto_face_cards(force: int = 0):
    """45 张人脸卡的对齐自检。**不用于匹配**，用来说明为什么不用卡图建库。"""
    if famod is None:
        raise HTTPException(503, "faceauto 未加载")
    try:
        return famod.card_audit(force=bool(force))
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@app.get("/api/selector/faces")
def selector_faces():
    """人脸卡（人工判定过的 g####）+ 它落在 ReID 库的哪条轨 → 一键写池的入口。"""
    s = _sel()
    cand = [BASE.parent / "frontend" / "dist" / "faces" / "face_data.json",
            BASE.parent / "frontend" / "public" / "faces" / "face_data.json",
            STATIC_DIR / "faces" / "face_data.json"]
    path = next((p for p in cand if p.exists()), None)
    if path is None:
        return {"ok": False, "persons": [], "note": "face_data.json 不存在（前端未构建？）"}
    d = json.loads(path.read_text(encoding="utf-8"))
    labels = _face_labels().get("persons", {})
    pool_tids = {}
    for (scope_k, ent), items in s.pool.items():
        for it in items:
            pool_tids[f"{it['cam']}:{it['tid']}"] = ent
    out = []
    for p in d.get("persons", []):
        key = f"{p.get('cam')}:{int(p.get('tid') or -1)}"
        i = s.idx_of(p.get("cam"), p.get("tid") or -1)
        mark = labels.get(p.get("label")) or {}
        out.append({
            "label": p.get("label"), "cam": p.get("cam"), "tid": p.get("tid"),
            "q": p.get("q"), "pd": p.get("pd"), "w": p.get("w"), "usable": p.get("usable"),
            "img": p.get("img_best") or p.get("img_single"),
            "in_store": i is not None,
            "entity": s.ent_of[i] if i is not None else None,
            "h_med": s.meta[i]["h_med"] if i is not None else None,
            "pool_of": pool_tids.get(key),
            "verdict": mark.get("verdict"), "name": mark.get("name"),
        })
    out.sort(key=lambda r: (not r["in_store"], -(r["q"] or 0)))
    return {"ok": True, "n": len(out),
            "n_in_store": sum(1 for r in out if r["in_store"]),
            "persons": out}


@app.get("/api/cluster/state")
def cluster_state():
    if not (algo.DESK and algo.STORE and algo.STORE.ready):
        raise HTTPException(503, "聚类工作台未就绪")
    return algo.DESK.snapshot(algo.STORE)


@app.post("/api/cluster/rename")
def cluster_rename(payload: dict = Body(...)):
    ok = algo.DESK.rename(payload.get("cid"), payload.get("label", ""))
    return {"ok": ok}


@app.post("/api/cluster/delete")
def cluster_delete(payload: dict = Body(...)):
    ok = algo.DESK.delete(payload.get("cid"))
    return {"ok": ok}


@app.post("/api/cluster/move")
def cluster_move(payload: dict = Body(...)):
    ok = algo.DESK.move(payload.get("items", []), payload.get("to_cid"),
                        create_new=bool(payload.get("create_new")),
                        label=payload.get("label", ""))
    return {"ok": ok}


@app.post("/api/cluster/remove")
def cluster_remove(payload: dict = Body(...)):
    ok = algo.DESK.remove_items(payload.get("cid"), payload.get("items", []))
    return {"ok": ok}


@app.post("/api/cluster/reset")
def cluster_reset():
    algo.DESK.reset()
    return {"ok": True}


@app.get("/api/predict/future")
def predict_future(cam: str, tid: int, f: int, horizon: float = 2.0, trail: int = 8):
    r = pdc.future(_tracks(cam), tid, f, horizon_s=horizon, trail=trail)
    if r is None:
        raise HTTPException(404, "无法预测（轨迹过短或未到该时刻）")
    return r


@app.get("/api/predict/intent")
def predict_intent(cam: str, tid: int, f: int):
    return pdc.intent(_tracks(cam), tid, f)


@app.get("/api/predict/all")
def predict_all(cam: str, tid: int, f: int, horizon: float = 2.0):
    fut = pdc.future(_tracks(cam), tid, f, horizon_s=horizon)
    itn = pdc.intent(_tracks(cam), tid, f)
    return {"future": fut, "intent": itn}


# ================= 真值标注：343 × 316 短窗出入口事件判定 =================

TRUTH_DIR = DATA / "truth"
TRUTH_THUMBS = TRUTH_DIR / "thumbs"
TRUTH_LABELS = TRUTH_DIR / "labels.json"

if TRUTH_THUMBS.exists():
    app.mount("/truth/thumbs", StaticFiles(directory=str(TRUTH_THUMBS)), name="truth_thumbs")


def _truth_labels() -> dict:
    if TRUTH_LABELS.exists():
        try:
            d = json.loads(TRUTH_LABELS.read_text(encoding="utf-8"))
            return {"pair": d.get("pair", {}) or {}, "event": d.get("event", {}) or {}}
        except Exception:
            pass
    return {"pair": {}, "event": {}}


def _truth_merged():
    st = _read("truth/state.json")
    seed = _read("truth/seed_labels.json", {}) or {}
    lab = _truth_labels()
    merged = dict(seed)
    merged.update(lab["pair"])
    a_cnt, b_cnt = {}, {}
    for p in st.get("pairs", []):
        if (merged.get(p["pair_id"]) or {}).get("v") == 1:
            a_cnt[p["a"]] = a_cnt.get(p["a"], 0) + 1
            b_cnt[p["b"]] = b_cnt.get(p["b"], 0) + 1
    st["labels"] = merged
    st["event_labels"] = lab["event"]
    st["conflicts"] = {"a": {k: v for k, v in a_cnt.items() if v > 1},
                       "b": {k: v for k, v in b_cnt.items() if v > 1}}
    vv = [x.get("v") for x in merged.values()]
    st["stats"] = {"labeled": sum(1 for v in vv if v in (0, 1, 2)),
                   "same": vv.count(1), "diff": vv.count(0), "unclear": vv.count(2),
                   "total": len(st.get("pairs", [])),
                   "events_labeled": sum(1 for x in lab["event"].values() if x.get("pid") or x.get("fp"))}
    st["label_file"] = str(TRUTH_LABELS)
    return st


@app.get("/api/truth/state")
def truth_state():
    return _truth_merged()


def _truth_metrics() -> dict:
    """实时指标：排序能力 / Brier / 分层无偏估计 / 同预算效率 / relay v5 精度。

    p_pred 采用**不依赖标签**的排名先验 p = 1 - (rank-0.5)/N（避免自证）；
    分层的"其余"同人率由随机审计样本无偏估计 → 可外推全池同人数。
    """
    st = _truth_merged()
    pairs = st.get("pairs", [])
    labels = st.get("labels", {})
    N = len(pairs)
    sel = (st.get("meta", {}).get("select") or {})
    K_t = sel.get("k_topk", 0)
    K_a = sel.get("k_audit", 0)
    budget = K_t + K_a

    def L(p):
        return (labels.get(p["pair_id"]) or {}).get("v")

    pos = [p for p in pairs if L(p) == 1]
    neg = [p for p in pairs if L(p) == 0]
    unk = [p for p in pairs if L(p) == 2]

    # 排序准确率（两两比较，等价 AUC；并列记 0.5）
    hit = tot = 0.0
    for a in pos:
        for b in neg:
            tot += 1
            hit += 1.0 if a["pred"] > b["pred"] else (0.5 if a["pred"] == b["pred"] else 0.0)
    rank_acc = (hit / tot) if tot else None

    # Brier（排名先验 p_pred）
    se, n = 0.0, 0
    for p in pairs:
        v = L(p)
        if v in (0, 1):
            pp = 1.0 - (p["rank"] - 0.5) / N
            se += (pp - v) ** 2
            n += 1
    brier = (se / n) if n else None

    # 分层：已标数(含看不清) / 已决数(0/1) / 同人数 / 同人率（审计层=其余层无偏估计）
    strata = {}
    for name in ("topk", "audit", "rest"):
        rows = [p for p in pairs if p["role"] == name]
        lb = [p for p in rows if L(p) in (0, 1, 2)]     # 已标（预算消耗）
        dc = [p for p in rows if L(p) in (0, 1)]        # 已决（可用于指标）
        s = sum(1 for p in dc if L(p) == 1)
        strata[name] = {"size": len(rows), "labeled": len(lb), "decided": len(dc), "same": s,
                        "rate": (s / len(dc)) if dc else None}

    # 全池同人数无偏估计：
    #   topk 层按自身已决率外扩；**非 topk 的 (audit+rest) 框架统一用 audit 的无偏率**。
    #   ⚠ rest 是"预先判定的高信号子集"，不是随机样本 → **不得单独外推**（否则严重高估）。
    est_total = None
    if strata["topk"]["rate"] is not None and strata["audit"]["rate"] is not None:
        est_total = (strata["topk"]["rate"] * strata["topk"]["size"]
                     + strata["audit"]["rate"] * (strata["audit"]["size"] + strata["rest"]["size"]))
    prevalence = (est_total / N) if est_total is not None else None

    # 同预算(50)对照：本方案实际命中 vs 纯随机 50 的期望命中
    arm_found = strata["topk"]["same"] + strata["audit"]["same"]
    random_expected = (budget * prevalence) if prevalence is not None else None
    efficiency = (arm_found / random_expected) if random_expected else None

    # relay v5 精度（仅已判样本内）
    r_ann = [p for p in pairs if p["relay"] == 1 and L(p) in (0, 1)]
    r_tp = sum(1 for p in r_ann if L(p) == 1)
    r_fp = sum(1 for p in r_ann if L(p) == 0)
    r_tot_same = len([p for p in pairs if p["relay"] == 1 and L(p) == 1])
    relay = {"annotated": len(r_ann), "tp": r_tp, "fp": r_fp,
             "precision": (r_tp / len(r_ann)) if r_ann else None,
             "recall_on_judged": (r_tot_same / len(pos)) if pos else None}

    return {
        "n_pairs": N, "annotated": len(pos) + len(neg) + len(unk),
        "pos": len(pos), "neg": len(neg), "unclear": len(unk),
        "budget": budget,
        "budget_done": strata["topk"]["labeled"] + strata["audit"]["labeled"],
        "decided": strata["topk"]["decided"] + strata["audit"]["decided"],
        "rank_acc": rank_acc, "brier": brier,
        "strata": strata, "prevalence": prevalence,
        "est_total_same": (est_total if est_total is not None else None),
        "arm_found": arm_found, "random_expected": random_expected, "efficiency": efficiency,
        "relay": relay,
        "note": "排序准确率=两两比较(AUC)；Brier 的 p_pred=1-(rank-0.5)/N（不依赖标签）；"
                "prevalence/est_total = rate(topk)×|topk| + rate(audit)×|audit+rest|，"
                "需 topk 与 audit 均有已决（rest 是预判子集，不单独外推）。",
    }


@app.get("/api/truth/metrics")
def truth_metrics_endpoint():
    return _truth_metrics()


def _truth_csv(cur: dict, st: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["pair_id", "cam_a", "event_a", "t_a", "cam_b", "event_b", "t_b", "dt",
                "in_relay_v5", "cos_appear", "machine",
                "human_same_person", "human_confidence", "human_note"])
    for p in st.get("pairs", []):
        lab = cur["pair"].get(p["pair_id"]) or {}
        v = lab.get("v")
        vs = "1" if v == 1 else "0" if v == 0 else "?" if v == 2 else ""
        w.writerow([p["pair_id"], "343", p["a"], p["ta"], "316", p["b"], p["tb"], p["dt"],
                    p["relay"], p["cos"], p["machine"], vs, lab.get("c", ""),
                    (lab.get("n", "") or "").replace(",", "，")])
    return buf.getvalue()


@app.post("/api/truth/labels")
def truth_save(payload: dict = Body(...)):
    cur = _truth_labels()
    for k in ("pair", "event"):
        if isinstance(payload.get(k), dict):
            for kk, vv in payload[k].items():
                if vv is None:
                    cur[k].pop(kk, None)
                else:
                    cur[k][kk] = vv
    TRUTH_LABELS.parent.mkdir(parents=True, exist_ok=True)
    TRUTH_LABELS.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    _ops_mirror("mirror_truth", payload)
    try:
        st = _read("truth/state.json")
        ev = io.StringIO()
        we = csv.writer(ev)
        we.writerow(["event_id", "cam", "kind", "edge", "t", "human_person_id"])
        for e in st.get("events", []):
            r = cur["event"].get(e["eid"]) or {}
            we.writerow([e["eid"], e["cam"], e["kind"], e["edge"], e["t"],
                         "FP" if r.get("fp") else (r.get("pid") or "")])
        (TRUTH_DIR / "labels.csv").write_text(_truth_csv(cur, st), encoding="utf-8")
        (TRUTH_DIR / "event_labels.csv").write_text(ev.getvalue(), encoding="utf-8")
    except Exception as e:  # pragma: no cover
        print(f"[truth] CSV 快照失败：{e}")
    return {"ok": True, "n_pair": len(cur["pair"]), "n_event": len(cur["event"])}


@app.get("/api/truth/export")
def truth_export(which: str = "pair"):
    cur = _truth_labels()
    st = _read("truth/state.json")
    if which == "event":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["event_id", "cam", "kind", "edge", "t", "human_person_id"])
        for e in st.get("events", []):
            r = cur["event"].get(e["eid"]) or {}
            w.writerow([e["eid"], e["cam"], e["kind"], e["edge"], e["t"],
                        "FP" if r.get("fp") else (r.get("pid") or "")])
        body = buf.getvalue()
    else:
        body = _truth_csv(cur, st)
    return Response(content="\ufeff" + body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="truth_labels_{which}.csv"'})


@app.get("/api/media-list")
def media_list():
    out = []
    if MEDIA_ROOT.exists():
        for p in sorted(MEDIA_ROOT.glob("*.mp4")):
            out.append({"name": p.name, "cam": p.stem, "size": p.stat().st_size})
    return out


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


@app.get("/media/{name}")
def media(name: str, request: Request):
    p = (MEDIA_ROOT / name).resolve()
    if not str(p).startswith(str(MEDIA_ROOT.resolve())) or not p.exists():
        raise HTTPException(404, "not found")
    size = p.stat().st_size
    ctype = mimetypes.guess_type(str(p))[0] or "video/mp4"
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(str(p), media_type=ctype)
    m = _RANGE_RE.match(rng)
    if not m:
        raise HTTPException(416, "bad range")
    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else size - 1
    end = min(end, size - 1)
    length = end - start + 1

    def iter_file():
        with p.open("rb") as fh:
            fh.seek(start)
            remain = length
            while remain > 0:
                chunk = fh.read(min(1024 * 512, remain))
                if not chunk:
                    break
                remain -= len(chunk)
                yield chunk

    return StreamingResponse(iter_file(), status_code=206, media_type=ctype, headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    })


# ---- 轨迹同人复核（全量卷 v2559 / v2526；独立 trackpair_labels.json，不动 labels.json）----
# 为什么单独一条线：这两卷**没有任何身份真值**，是全量聚类唯一无法评估的空白。
# 四类候选对：A=簇内(测精度) B=跨簇高 cos(测召回) C=外观中等+位置近 D=外观中等+位置远(对照)。
# ★ C 与 D 必须配对看：只标 C 无法排除「外观中等的对本来就大多不是同一个人」。
TP_DIR = DATA / "trackpair"
TP_PAIRS = TP_DIR / "pairs.json"
TP_STRIPS = TP_DIR / "strips"
_TP_KIND = {"A": "簇内·测精度", "B": "跨簇·测召回", "C": "中等外观·位置近", "D": "中等外观·位置远"}
_TP_VALID = ("same", "diff", "unsure")


def _tp_pairs() -> list:
    if not TP_PAIRS.exists():
        return []
    return json.loads(TP_PAIRS.read_text(encoding="utf-8"))


def _tp_labels() -> dict:
    return _read("trackpair_labels.json", {"version": 1, "updated": None, "labels": {}})


def _tp_counts(pairs: list, labels: dict) -> dict:
    by: dict = {}
    for p in pairs:
        d = by.setdefault(p.get("kind", "?"),
                          {"n": 0, "labeled": 0, "same": 0, "diff": 0, "unsure": 0})
        d["n"] += 1
        v = (labels.get(p["id"]) or {}).get("v")
        if v:
            d["labeled"] += 1
            if v in d:
                d[v] += 1
    for k, d in by.items():
        d["desc"] = _TP_KIND.get(k, k)
        d["same_rate"] = round(d["same"] / d["labeled"], 4) if d["labeled"] else None
    return dict(sorted(by.items()))


@app.get("/api/trackpair/pairs")
def tp_pairs_api():
    pairs, labels = _tp_pairs(), _tp_labels().get("labels", {})
    out = []
    for p in pairs:
        rec = labels.get(p["id"]) or {}
        out.append({**p, "label": rec.get("v"), "label_at": rec.get("at")})
    return {"ok": True, "n": len(out), "n_labeled": sum(1 for q in out if q["label"]),
            "updated": _tp_labels().get("updated"),
            "vols": sorted({p.get("vol") for p in pairs if p.get("vol")}),
            "counts": _tp_counts(pairs, labels), "pairs": out}


@app.post("/api/trackpair/label")
def tp_label_api(payload: dict = Body(...)):
    """一条一对。val=None 表示清除判定。

    ★ 必须校验 id 存在于 pairs.json：拼错的 id 会让台账悄悄长出一堆幽灵条目，
      之后统计里「已标数」永远追不上总数。
    """
    pid = str(payload.get("id") or "")
    val = payload.get("label")
    if val in ("",):
        val = None
    if val is not None and val not in _TP_VALID:
        raise HTTPException(400, "label 只能是 same / diff / unsure / null")
    pmap = {p["id"]: p for p in _tp_pairs()}
    if pid not in pmap:
        raise HTTPException(404, f"没有这一对：{pid}")
    cur = _tp_labels()
    labels = cur.setdefault("labels", {})
    if val is None:
        labels.pop(pid, None)
    else:
        labels[pid] = {"v": val, "at": _now(), "kind": pmap[pid].get("kind"),
                       "vol": pmap[pid].get("vol")}
    cur["updated"] = _now()
    _write("trackpair_labels.json", cur)
    return {"ok": True, "id": pid, "label": val, "n_labeled": len(labels),
            "counts": _tp_counts(list(pmap.values()), labels)}


@app.get("/api/trackpair/stats")
def tp_stats_api():
    pairs, labels = _tp_pairs(), _tp_labels().get("labels", {})
    return {"ok": True, "n": len(pairs), "n_labeled": len(labels),
            "counts": _tp_counts(pairs, labels), "updated": _tp_labels().get("updated")}


@app.get("/api/trackpair/export")
def tp_export():
    labels = _tp_labels().get("labels", {})
    rows = [{"id": p["id"], "vol": p.get("vol"), "kind": p.get("kind"), "tier": p.get("tier"),
             "a": p.get("a"), "b": p.get("b"), "cos": p.get("cos"), "dist": p.get("dist"),
             "gap": p.get("gap"), "same_cluster": p.get("same_cluster"),
             "label": (labels.get(p["id"]) or {}).get("v")} for p in _tp_pairs()]
    body = json.dumps({"n": len(rows), "n_labeled": sum(1 for r in rows if r["label"]),
                       "exported": _now(), "rows": rows}, ensure_ascii=False, indent=1)
    return Response(content=body, media_type="application/json; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="trackpair_labels.json"'})


@app.get("/trackpair/strips/{name}")
def tp_strip_file(name: str):
    root = TP_STRIPS.resolve()
    p = (root / name).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


# ---- 簇纯度核验（v2545 走廊机位；独立 clusterver_labels.json，不动 labels.json）----
# 为什么是「簇级」而不是「对级」：v2545 在 0.78 配置下只有 15 个单卷簇覆盖 129 条轨，
# 15 个判断就能覆盖全部轨；而对级要几百对，且只能给 P/R 约束、给不出「这个簇里混了几个人」。
# 判定语义：每簇三态（整簇同一人 / 有外人 / 不确定），簇内每条轨可额外标 外人 或 看不清。
#   ★ 看不清的轨要从纯度分母里剔除——走廊 19% 的轨框高 < 80 px，只有剪影，
#     把「看不清」算成「不同人」会系统性地把纯度压低。
CV_DIR = DATA / "clusterver"
CV_ITEMS = CV_DIR / "items.json"
CV_STRIPS = CV_DIR / "strips"
_CV_VERDICT = ("pure", "mixed", "unsure")
_CV_MARK = ("out", "unread")


def _cv_items() -> list:
    if not CV_ITEMS.exists():
        return []
    return json.loads(CV_ITEMS.read_text(encoding="utf-8"))


def _cv_labels() -> dict:
    return _read("clusterver_labels.json", {"version": 1, "updated": None, "labels": {}})


def _cv_estimate(items: list, labels: dict) -> dict:
    """把人工判定折算成「簇纯度」与「簇内对精度」。

    - pure：整簇同一人 → 纯度 1
    - mixed：标了 k 个外人 → 保守按「每个外人各是一个独立身份」计，纯度 (n-k)/n
    - multi：一眼多人但没逐条标 → **不计入估计**（只计数量）。
      ★ 试过按 k=1 保守计入，结果是「46 轨的簇标了明显多人，纯度还是 0.978」——
        这不是保守，是自欺。要么标清楚，要么别算。
    - unsure：同样不计入，单独计数
    看不清的轨从分母剔除。
    """
    rows, n_lab = [], 0
    n_unsure = n_multi = 0
    w_num = w_den = 0.0            # 纯度：按轨加权
    p_num = p_den = 0.0            # 对精度：按簇内对数加权
    for it in items:
        rec = labels.get(it["id"]) or {}
        v = rec.get("v")
        if not v:
            continue
        n_lab += 1
        if v in ("unsure", "multi"):
            if v == "unsure":
                n_unsure += 1
            else:
                n_multi += 1
            rows.append({"id": it["id"], "cid": it["cid"], "n": it["n"], "verdict": v,
                         "k_out": None, "n_judge": None, "purity": None, "pair_p": None})
            continue
        marks = rec.get("marks") or {}
        n_unread = sum(1 for m in it["members"] if marks.get(m["k"]) == "unread")
        k_out = sum(1 for m in it["members"] if marks.get(m["k"]) == "out")
        n_judge = max(1, it["n"] - n_unread)
        k_out = min(k_out, n_judge)
        purity = (n_judge - k_out) / n_judge
        cs = n_judge * (n_judge - 1) / 2.0
        ps = (n_judge - k_out) * (n_judge - k_out - 1) / 2.0
        pair_p = (ps / cs) if cs > 0 else None
        w_num += n_judge * purity
        w_den += n_judge
        if pair_p is not None:
            p_num += cs * pair_p
            p_den += cs
        rows.append({"id": it["id"], "cid": it["cid"], "n": it["n"], "verdict": v,
                     "k_out": k_out, "n_judge": n_judge,
                     "purity": round(purity, 4),
                     "pair_p": None if pair_p is None else round(pair_p, 4)})
    return {"n_clusters": len(items), "n_labeled": n_lab, "n_unsure": n_unsure,
            "n_multi_unquantified": n_multi,
            "purity": round(w_num / w_den, 4) if w_den else None,
            "pair_precision": round(p_num / p_den, 4) if p_den else None,
            "n_tracks_judged": int(w_den), "rows": rows}


def _cv_counts(items: list, labels: dict) -> dict:
    by = {"pure": 0, "mixed": 0, "multi": 0, "unsure": 0, "undone": 0}
    for it in items:
        v = (labels.get(it["id"]) or {}).get("v")
        by[v if v in by else "undone"] += 1
    by["n"] = len(items)
    by["n_tracks"] = sum(it["n"] for it in items)
    return by


@app.get("/api/clusterver/items")
def cv_items_api():
    items, labels = _cv_items(), _cv_labels().get("labels", {})
    out = []
    for it in items:
        rec = labels.get(it["id"]) or {}
        out.append({**it, "verdict": rec.get("v"), "marks": rec.get("marks") or {},
                    "at": rec.get("at")})
    return {"ok": True, "n": len(out),
            "n_labeled": sum(1 for x in out if x["verdict"]),
            "updated": _cv_labels().get("updated"),
            "vols": sorted({x.get("vol") for x in items if x.get("vol")}),
            "counts": _cv_counts(items, labels), "items": out}


@app.post("/api/clusterver/label")
def cv_label_api(payload: dict = Body(...)):
    """一簇一条。verdict=None 表示清除。

    marks: {track_key: "out"|"unread"}，只允许本簇成员，且 verdict 必须是 mixed/multi 才有意义。
    """
    iid = str(payload.get("id") or "")
    verdict = payload.get("verdict")
    if verdict in ("", None):
        verdict = None
    if verdict is not None and verdict not in _CV_VERDICT + ("multi",):
        raise HTTPException(400, "verdict 只能是 pure / mixed / multi / unsure / null")
    imap = {it["id"]: it for it in _cv_items()}
    if iid not in imap:
        raise HTTPException(404, f"没有这一簇：{iid}")
    mem = {m["k"] for m in imap[iid]["members"]}
    raw = payload.get("marks") or {}
    marks = {}
    for k, v in raw.items():
        if k not in mem:
            raise HTTPException(400, f"{k} 不属于簇 {iid}")
        if v not in _CV_MARK:
            raise HTTPException(400, "mark 只能是 out / unread")
        marks[k] = v
    cur = _cv_labels()
    labels = cur.setdefault("labels", {})
    if verdict is None:
        labels.pop(iid, None)
    else:
        labels[iid] = {"v": verdict, "marks": marks, "at": _now(),
                       "cid": imap[iid]["cid"], "n": imap[iid]["n"]}
    cur["updated"] = _now()
    _write("clusterver_labels.json", cur)
    return {"ok": True, "id": iid, "verdict": verdict,
            "n_marks": len(marks), "n_labeled": len(labels),
            "counts": _cv_counts(_cv_items(), labels),
            "estimate": _cv_estimate(_cv_items(), labels)}


@app.get("/api/clusterver/stats")
def cv_stats_api():
    items, labels = _cv_items(), _cv_labels().get("labels", {})
    est = _cv_estimate(items, labels)
    return {"ok": True, "n": len(items), "updated": _cv_labels().get("updated"),
            "counts": _cv_counts(items, labels),
            **{k: v for k, v in est.items() if k != "rows"}, "rows": est["rows"]}


@app.get("/api/clusterver/export")
def cv_export():
    labels = _cv_labels().get("labels", {})
    rows = []
    for it in _cv_items():
        rec = labels.get(it["id"]) or {}
        rows.append({"id": it["id"], "vol": it.get("vol"), "cid": it["cid"],
                     "n": it["n"], "verdict": rec.get("v"),
                     "marks": rec.get("marks") or {},
                     "members": [m["k"] for m in it["members"]]})
    body = json.dumps({"n": len(rows), "n_labeled": sum(1 for r in rows if r["verdict"]),
                       "exported": _now(), "rows": rows}, ensure_ascii=False, indent=1)
    return Response(content=body, media_type="application/json; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="clusterver_labels.json"'})


@app.get("/clusterver/strips/{name}")
def cv_strip_file(name: str):
    root = CV_STRIPS.resolve()
    p = (root / name).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


# ---- 位置合并判读（步骤 2 的 26 组无真值合并 + 7 组跨机位时间重叠）----
# 为什么必须判：步骤 2 在 8 段上 10 次合并里只有 6 次能用真值定性；7 卷 24 次里只有 1 次。
# 也就是说 7 卷那个「P 0.9912」没有真值支撑，只说明最大簇没变大、同帧冲突为 0。
# 判读集 = 8 段 3 组无真值 + 7 卷 23 组无真值（不抽样，全判）+ 7 组时间重叠对。
#
# 两种口径都要报：
#   组口径   —— 一次判断算一次（回答问题「这类合并靠不靠谱」）
#   对口径   —— 按 nA×nB 加权（回答问题「精度会掉多少」：一个大簇错并，代价是该簇所有对）
# 停止条件：可判部分同人率 <95% → 位置合并只限定在已验证机位，不做默认配置。
MV_DIR = DATA / "mergever"
MV_ITEMS = MV_DIR / "items.json"
MV_STRIPS = MV_DIR / "strips"
_MV_VERDICT = ("same", "diff", "unsure")


def _mv_items() -> list:
    if not MV_ITEMS.exists():
        return []
    return json.loads(MV_ITEMS.read_text(encoding="utf-8"))


def _mv_labels() -> dict:
    return _read("mergever_labels.json", {"version": 1, "updated": None, "labels": {}})


def _mv_counts(items: list, labels: dict) -> dict:
    by = {"same": 0, "diff": 0, "unsure": 0, "undone": 0}
    for it in items:
        v = (labels.get(it["id"]) or {}).get("v")
        by[v if v in by else "undone"] += 1
    by["n"] = len(items)
    by["n_merge"] = sum(1 for it in items if it.get("kind") == "merge")
    by["n_overlap"] = sum(1 for it in items if it.get("kind") == "overlap")
    return by


def _mv_estimate(items: list, labels: dict) -> dict:
    """按 kind 分开统计；每种都出「组口径」与「对口径」。

    对口径的权重 = nA*nB（这次合并新增的跨簇轨对数），重叠对权重恒为 1。
    unsure 一律剔出分母——看不清不算判错。
    """
    def block(kind):
        rows, w_num = [], 0.0
        g_same = g_diff = 0
        w_same = w_diff = 0.0
        for it in items:
            if it.get("kind", "merge") != kind:
                continue
            rec = labels.get(it["id"]) or {}
            v = rec.get("v")
            wgt = float(it.get("nA", 1)) * float(it.get("nB", 1)) if kind == "merge" else 1.0
            if v == "same":
                g_same += 1
                w_same += wgt
            elif v == "diff":
                g_diff += 1
                w_diff += wgt
            rows.append({"id": it["id"], "verdict": v, "w": wgt,
                         "pos_cos": it.get("pos_cos"), "pos_one": it.get("pos_one"),
                         "nA": it.get("nA"), "nB": it.get("nB")})
        g_judg = g_same + g_diff
        w_judg = w_same + w_diff
        return {"n_judged": g_judg,
                "same_rate": round(g_same / g_judg, 4) if g_judg else None,
                "same_rate_pairs": round(w_same / w_judg, 4) if w_judg else None,
                "n_same": g_same, "n_diff": g_diff,
                "pairs_judged": round(w_judg, 1),
                "rows": rows}

    merges = block("merge")
    ovs = block("overlap")
    # pos==1.0 单独看：两簇落进同一个 12×8 格子，远景下是弱证据
    p1 = [r for r in merges["rows"] if r.get("pos_one") and r["verdict"] in ("same", "diff")]
    p1_same = sum(1 for r in p1 if r["verdict"] == "same")
    return {"merges": merges, "overlaps": ovs,
            "pos_one": {"n_judged": len(p1), "n_same": p1_same,
                        "same_rate": round(p1_same / len(p1), 4) if p1 else None}}


@app.get("/api/mergever/items")
def mv_items_api():
    items, labels = _mv_items(), _mv_labels().get("labels", {})
    out = []
    for it in items:
        rec = labels.get(it["id"]) or {}
        out.append({**it, "verdict": rec.get("v"), "at": rec.get("at")})
    return {"ok": True, "n": len(out),
            "n_labeled": sum(1 for x in out if x["verdict"]),
            "updated": _mv_labels().get("updated"),
            "counts": _mv_counts(items, labels), "items": out}


@app.post("/api/mergever/label")
def mv_label_api(payload: dict = Body(...)):
    iid = str(payload.get("id") or "")
    v = payload.get("verdict")
    if v in ("", None):
        v = None
    if v is not None and v not in _MV_VERDICT:
        raise HTTPException(400, "verdict 只能是 same / diff / unsure / null")
    imap = {it["id"]: it for it in _mv_items()}
    if iid not in imap:
        raise HTTPException(404, f"没有这一条：{iid}")
    cur = _mv_labels()
    labels = cur.setdefault("labels", {})
    if v is None:
        labels.pop(iid, None)
    else:
        labels[iid] = {"v": v, "at": _now(), "kind": imap[iid].get("kind", "merge")}
    cur["updated"] = _now()
    _write("mergever_labels.json", cur)
    return {"ok": True, "id": iid, "verdict": v, "n_labeled": len(labels),
            "counts": _mv_counts(_mv_items(), labels),
            "estimate": _mv_estimate(_mv_items(), labels)}


@app.get("/api/mergever/stats")
def mv_stats_api():
    items, labels = _mv_items(), _mv_labels().get("labels", {})
    return {"ok": True, "n": len(items), "updated": _mv_labels().get("updated"),
            "counts": _mv_counts(items, labels), **_mv_estimate(items, labels)}


@app.get("/api/mergever/export")
def mv_export():
    labels = _mv_labels().get("labels", {})
    rows = [{"id": it["id"], "kind": it.get("kind"), "verdict": (labels.get(it["id"]) or {}).get("v"),
             "pos_cos": it.get("pos_cos"), "pos_one": it.get("pos_one"),
             "nA": it.get("nA"), "nB": it.get("nB"), "segsA": it.get("segsA"),
             "segsB": it.get("segsB"), "samples": it.get("samples")}
            for it in _mv_items()]
    body = json.dumps({"n": len(rows), "n_labeled": sum(1 for r in rows if r["verdict"]),
                       "exported": _now(), "rows": rows}, ensure_ascii=False, indent=1)
    return Response(content=body, media_type="application/json; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="mergever_labels.json"'})


@app.get("/mergever/strips/{name}")
def mv_strip_file(name: str):
    root = MV_STRIPS.resolve()
    p = (root / name).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(p), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


# ---- E21 全天单机位人工标簇（可选模块：卡片未生成时返回 404，不影响其它页）----
try:
    import e21_label

    app.include_router(e21_label.router)
except Exception as _e21_err:  # pragma: no cover
    import logging

    logging.getLogger("uvicorn").warning("e21_label 未挂载：%s", _e21_err)


# ---- 算法中台台账（可选模块：本机为主库；远程找不到库时返回 503，不影响既有页面）----
try:
    import ops_api

    app.include_router(ops_api.router)
except Exception as _ops_err:  # pragma: no cover
    import logging

    logging.getLogger("uvicorn").warning("ops_api 未挂载：%s", _ops_err)


# ---- 科研工作台（实验卡 + 四关审批；与 ops 台账同库，缺库时 503）----
try:
    import workbench_api

    app.include_router(workbench_api.router)
except Exception as _wb_err:  # pragma: no cover
    import logging

    logging.getLogger("uvicorn").warning("workbench_api 未挂载：%s", _wb_err)


# ---- 静态前端（放在最后，避免吞掉 /api）----
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    @app.get("/")
    def _no_static():
        return JSONResponse({"ok": True, "msg": "前端未构建：请在 webapp/frontend 执行 npm run build",
                             "api": "/api/health"})
