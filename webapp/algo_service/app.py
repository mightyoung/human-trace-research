# -*- coding: utf-8 -*-
"""单图人物属性服务。

输入一张图片 → 输出图中每个人的框、框高档位、9 个属性字段（值 + 置信度 + 可承诺档位）。
检测用 YOLO11 person 类，属性用 EG2 嵌入 + 线上训练好的线性头（attr_heads.joblib）。

三条口径（与线下评估保持一致，不要在服务里"优化"掉）：
1. **裁图必须 bbox + 10%**：训练集的 crop 就是这么裁的（build_ent_crops.py），
   换口径等于换分布。
2. **按框高档位闸门**：S<80 / M 80–240 / L ≥240。某字段在该档位上不可判时返回
   `null` + `reason`，不硬猜；只到「参考」档的字段照常返回但标 `cap="ref"`。
3. **is_person 单独给分**：模型对假框（倒影、海报、货架）会照样输出属性，
   调用方需要这个分数自己决定要不要用。
"""
from __future__ import annotations

import base64
import binascii
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

EG2_DIR = os.environ.get("EG2_DIR", "/opt/models/eg2")
HEADS = os.environ.get("HEADS", "/opt/models/attr_heads.joblib")
DETECTOR = os.environ.get("DETECTOR", "/opt/models/yolo11m.pt")
DEVICE = os.environ.get("DEVICE", "cuda:0")
DET_CONF = float(os.environ.get("DET_CONF", "0.35"))
DET_IMGSZ = int(os.environ.get("DET_IMGSZ", "1280"))
IS_PERSON_MIN = float(os.environ.get("IS_PERSON_MIN", "0.50"))

app = FastAPI(title="human attribute service", version="1.0")
S: dict = {}


@app.on_event("startup")
def _load():
    import joblib
    import torch
    from sentence_transformers import SentenceTransformer
    from ultralytics import YOLO

    t0 = time.time()
    S["det"] = YOLO(DETECTOR)
    S["eg2"] = SentenceTransformer(EG2_DIR, device=DEVICE,
                                   model_kwargs={"dtype": torch.bfloat16},
                                   config_kwargs={"audio_config": None},
                                   local_files_only=True)
    S["eg2"].eval()
    S["b"] = joblib.load(HEADS)
    S["ready_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    S["load_s"] = round(time.time() - t0, 1)
    print(f"[startup] 加载完成 {S['load_s']}s device={DEVICE}", flush=True)


@app.get("/healthz")
def healthz():
    return {"ok": bool(S.get("b")), "loaded_in_s": S.get("load_s"), "ready_at": S.get("ready_at"),
            "device": DEVICE, "fields": sorted((S.get("b") or {}).get("fields", {}).keys())}


@app.get("/v1/spec")
def spec():
    """把「能承诺什么」也做成接口：调用方不该靠猜哪些字段可信。"""
    b = S.get("b") or {}
    sp = b.get("spec") or {}
    return {"version": b.get("version"), "built": b.get("built"),
            "spec": sp, "capability": b.get("capability"), "fields": b.get("meta"),
            "notes": ["cap=ok 可承诺 / ref 仅供参考 / null 不可判（该框高档位没有证据）",
                      "conf 是模型置信度，不是准确率；建议生产再卡 conf ≥ .8",
                      "背包 / 拎东西召回只有 .16–.50，只能当提示",
                      # ★ 这里不写死指标：换权重时数字会过期，来源以 spec 为准
                      f"训练数据 {sp.get('dataset', 'long35')}"
                      f"（域内 {sp.get('n_dom')} 条 ×{sp.get('dom_reps')} + 域外 {sp.get('n_pa')} 张）；"
                      "指标见服务目录 README 与 docs/algo-service-acceptance-2026-11-12.md",
                      "换场景精度会明显下降：新场景上线前必须用该场景 100–200 张图重测",
                      "自带框时 det_conf 固定 1.0，不是质量分"]}


def _band(h: float, band=(80.0, 240.0)) -> str:
    return "S" if h < band[0] else ("M" if h < band[1] else "L")


def _crop_pad(img, box, pad: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    ax1, ay1 = max(0, int(np.floor(x1 - px))), max(0, int(np.floor(y1 - py)))
    ax2, ay2 = min(w, int(np.ceil(x2 + px))), min(h, int(np.ceil(y2 + py)))
    if ax2 - ax1 < 8 or ay2 - ay1 < 8:
        return None
    return img[ay1:ay2, ax1:ax2]


@app.post("/v1/analyze")
async def analyze(request: Request,
                  crops: int = Query(0, description="1=返回每个人的裁图 base64"),
                  drop_non_person: int = Query(0, description="1=过滤 is_person 低于阈值的框"),
                  boxes: str = Query("", description='自备框：full=整图当一人；或 [[x1,y1,x2,y2],...]')):
    """请求体就是图片字节（image/jpeg、image/png、octet-stream 均可）。

    ★ 不用 multipart：镜像离线构建装不了 python-multipart，而"裸 body 传图"对
      推理服务本来就更直接（curl --data-binary @x.jpg 即可）。
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "请求体为空（直接 POST 图片字节，不是 multipart）")
    return _analyze_bytes(raw, request.headers.get("x-filename", ""), crops,
                          drop_non_person, boxes)


@app.post("/v1/analyze_json")
async def analyze_json(payload: dict, crops: int = Query(0),
                       drop_non_person: int = Query(0), boxes: str = Query("")):
    """JSON 版：{"image_b64": "..."}，给不方便发二进制的调用方。"""
    s = (payload or {}).get("image_b64") or ""
    if "," in s[:64]:                     # 容忍 data:image/jpeg;base64, 前缀
        s = s.split(",", 1)[1]
    try:
        raw = base64.b64decode(s, validate=False)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "image_b64 不是合法 base64")
    return _analyze_bytes(raw, (payload or {}).get("name", ""), crops,
                          drop_non_person, boxes)


def _parse_boxes(spec: str, W: int, H: int):
    """自备框解析。返回 (boxes Nx4, conf N) 或 None 表示走检测。"""
    s = (spec or "").strip()
    if not s:
        return None
    if s.lower() == "full":
        return np.array([[0.0, 0.0, float(W), float(H)]]), np.ones(1)
    import json
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        raise HTTPException(400, "boxes 不是合法 JSON；应为 full 或 [[x1,y1,x2,y2],...]")
    if not isinstance(arr, list) or not arr:
        raise HTTPException(400, "boxes 必须是非空数组")
    out = []
    for b in arr:
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise HTTPException(400, "每个框必须是 [x1,y1,x2,y2]")
        x1, y1, x2, y2 = [float(v) for v in b]
        x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        out.append([x1, y1, x2, y2])
    if not out:
        raise HTTPException(400, "boxes 里没有有效框（贴合图像范围后都小于 4px）")
    return np.array(out), np.ones(len(out))


def _analyze_bytes(raw: bytes, name: str, crops: int, drop_non_person: int,
                   box_spec: str = ""):
    if not S.get("b"):
        raise HTTPException(503, "模型尚未加载完成")
    import cv2
    import torch
    from PIL import Image

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "不是可解码的图片")
    H, W = img.shape[:2]

    t0 = time.time()
    given = _parse_boxes(box_spec, W, H)
    if given is not None:                 # 调用方自备框：跳过检测，det_conf 记 1.0
        boxes, det_conf = given
        det_ms = 0.0
    else:
        res = S["det"].predict(source=img, classes=[0], conf=DET_CONF, imgsz=DET_IMGSZ,
                               verbose=False, device=DEVICE)[0]
        det_ms = (time.time() - t0) * 1000
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        det_conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))

    b = S["b"]
    spec = b.get("spec") or {}
    pad = float(spec.get("pad", 0.10))
    lower_from = float(spec.get("lower_from", 0.38))
    band_th = tuple(spec.get("band", [80.0, 240.0]))
    cap_tab = b.get("capability") or {}

    # ---- 裁图（与训练同口径）----
    crops_rgb, lower_rgb, keep = [], [], []
    for i, box in enumerate(boxes):
        patch = _crop_pad(img, box, pad)
        if patch is None:
            continue
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        crops_rgb.append(pil)
        w0, h0 = pil.size
        lower_rgb.append(pil.crop((0, int(h0 * lower_from), w0, h0)))
        keep.append(i)
    t_crop = time.time() - t0

    out_persons = []
    if crops_rgb:
        t1 = time.time()
        with torch.inference_mode():
            E = np.asarray(S["eg2"].encode(crops_rgb, batch_size=8, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False),
                           dtype=np.float32)
            # ★ 下半身嵌入只对 L 档算：carrying 在 S/M 档不可判，那些行的结果会被丢弃，
            #   白算一遍要多花一半嵌入时间（10 人图实测 1099 ms → 约 600 ms）。
            lo_idx = [k for k, p in enumerate(keep)
                      if (float(boxes[p][3]) - float(boxes[p][1])) >= band_th[1]]
            Elo = np.zeros((len(crops_rgb), E.shape[1]), dtype=np.float32)
            if lo_idx:
                Elo[lo_idx] = np.asarray(
                    S["eg2"].encode([lower_rgb[k] for k in lo_idx], batch_size=8,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False), dtype=np.float32)
        emb_ms = (time.time() - t1) * 1000

        t2 = time.time()
        p_person = b["is_person"].predict_proba(E)[:, 1]
        preds = {}
        for f, pipe in b["fields"].items():
            X = np.concatenate([E, Elo], axis=1) if (b["meta"].get(f) or {}).get("use_lower") else E
            pp = pipe.predict_proba(X)
            cls = np.asarray(pipe.classes_)
            top = pp.argmax(1)
            preds[f] = (cls[top], pp[np.arange(len(X)), top])
        head_ms = (time.time() - t2) * 1000

        for k, i in enumerate(keep):
            x1, y1, x2, y2 = [float(v) for v in boxes[i]]
            h = y2 - y1
            band = _band(h, band_th)
            attrs, na = {}, []
            for f in ("gender", "top_color", "top_type", "bottom_color", "bottom_type",
                      "glasses", "hat", "backpack", "carrying"):
                if f not in preds:
                    continue
                cap = (cap_tab.get(f) or {}).get(band)
                v, c = preds[f]
                if cap is None:
                    attrs[f] = {"v": None, "conf": None, "cap": None,
                                "reason": f"{band} 档不可判（框高 {h:.0f}px，该字段没有可靠证据）"}
                    na.append(f)
                else:
                    attrs[f] = {"v": str(v[k]), "conf": round(float(c[k]), 4), "cap": cap}
            item = {"idx": int(i), "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "h": round(h, 1), "band": band,
                    "det_conf": round(float(det_conf[i]), 4),
                    "is_person": round(float(p_person[k]), 4),
                    "attrs": attrs, "not_judgeable": na}
            if crops:
                ok, buf = cv2.imencode(".jpg", _crop_pad(img, boxes[i], pad),
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    item["crop_b64"] = base64.b64encode(buf.tobytes()).decode()
            out_persons.append(item)

    if drop_non_person:
        out_persons = [p for p in out_persons if p["is_person"] >= IS_PERSON_MIN]
    out_persons.sort(key=lambda p: -p["h"])

    return JSONResponse({
        "image": {"w": W, "h": H, "name": name},
        "n_persons": len(out_persons),
        "persons": out_persons,
        "model": {"heads_built": b.get("built"), "is_person_min": IS_PERSON_MIN,
                  "det_conf": DET_CONF, "device": DEVICE},
        "timing_ms": {"det": round(det_ms, 1), "crop": round(t_crop * 1000, 1),
                      "emb": round(emb_ms, 1) if crops_rgb else 0,
                      "head": round(head_ms, 1) if crops_rgb else 0,
                      "total": round((time.time() - t0) * 1000, 1)},
    })
