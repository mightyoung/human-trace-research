# -*- coding: utf-8 -*-
"""轨迹预测后端：几何外推 + 运动意图模型（BERT 范式 Transformer 编码器）。

- future(): 由最近若干采样点估计速度，外推未来 horizon 秒的落脚点
- intent(): 取最近 L_in 个采样点构成历史窗，预测未来 1s 的运动类别
            （stop/forward/backward/turn_left/turn_right）
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
INTENT_DIR = DATA / "intent"

_params = None
_model = None
_torch = None
_lock = threading.Lock()


def params():
    global _params
    if _params is None:
        _params = json.loads((INTENT_DIR / "params.json").read_text(encoding="utf-8"))
    return _params


def _load_model():
    """惰性加载意图模型；torch/权重缺失时返回 None（不影响几何预测）。"""
    global _model, _torch
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        try:
            import torch
            import torch.nn as nn
            _torch = torch

            class IntentBERT(nn.Module):
                def __init__(self, n_feat=7, d_model=256, n_layers=4, n_heads=4,
                             dim_ff=1024, dropout=0.1, n_classes=5, max_len=64):
                    super().__init__()
                    self.proj = nn.Linear(n_feat, d_model)
                    self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
                    self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
                    layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
                        dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
                    self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
                    self.norm = nn.LayerNorm(d_model)
                    self.head = nn.Linear(d_model, n_classes)

                def forward(self, x):
                    h = self.proj(x)
                    b, l, _ = h.shape
                    h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
                    h = h + self.pos[:, :l + 1]
                    h = self.encoder(h)
                    return self.head(self.norm(h[:, 0]))

            sd = torch.load(str(INTENT_DIR / "model.pt"), map_location="cpu", weights_only=False)
            state = sd.get("model", sd) if isinstance(sd, dict) else sd
            p = params()
            m = IntentBERT(n_feat=p["n_feat"], d_model=p["d_model"], n_layers=p["layers"],
                           n_heads=p["heads"], dim_ff=p["ff"])
            m.load_state_dict(state, strict=False)
            m.eval()
            _model = m
        except Exception as e:  # pragma: no cover
            _model = False
            print(f"[predict] 意图模型不可用：{e}")
    return _model


def _series(tracks_json, tid):
    """返回该轨迹的 (frames, feats[u,v,du,dv,w,h,DT])，按帧升序。"""
    pts = (tracks_json.get("tracks") or {}).get(str(tid)) or []
    if len(pts) < 3:
        return None, None
    W, H = tracks_json.get("w", 1), tracks_json.get("h", 1)
    fps = tracks_json.get("fps", 25.0)
    stride = tracks_json.get("stride", 5) or 5
    DT = stride / fps
    feats, frames = [], []
    prev = None
    for p in pts:
        u, v = (p[1] + p[3]) / 2.0 / W, p[4] / H
        w, h = (p[3] - p[1]) / W, (p[4] - p[2]) / H
        du, dv = (0.0, 0.0) if prev is None else (u - prev[0], v - prev[1])
        feats.append([u, v, du, dv, w, h, DT])
        frames.append(int(p[0]))
        prev = (u, v)
    return np.asarray(frames), np.asarray(feats, dtype=np.float32)


def _contiguous_end(frames, f):
    """以 ≤f 的最末采样点为终点，向前取连续段（容忍 1.5*stride 间隔）。"""
    idx = [i for i, fr in enumerate(frames) if fr <= f]
    if not idx:
        return None
    end = idx[-1]
    start = end
    step = 1 if len(frames) < 2 else (frames[1] - frames[0])
    while start > 0 and (frames[start] - frames[start - 1]) <= step * 1.6:
        start -= 1
    return start, end


def future(tracks_json, tid, f, horizon_s=2.0, trail=8, smooth=4):
    frames, feats = _series(tracks_json, tid)
    if frames is None:
        return None
    seg = _contiguous_end(frames, f)
    if seg is None:
        return None
    s, e = seg
    pts = feats[s:e + 1, :2]
    n = len(pts)
    if n < 2:
        return None
    dt = float(feats[e, 6])
    k = max(1, min(smooth, n - 1))
    d = pts[n - 1] - pts[n - 1 - k]
    vel = d / (k * dt)              # 归一化坐标 / 秒
    steps = max(1, int(round(horizon_s / dt)))
    pred = [pts[n - 1] + vel * (i * dt) for i in range(1, steps + 1)]
    trail_pts = pts[max(0, n - trail):]
    speed = float(np.hypot(vel[0] * tracks_json.get("w", 1),
                           vel[1] * tracks_json.get("h", 1)))  # 原始像素/秒
    ang = float(np.degrees(np.arctan2(vel[1], vel[0])))
    return {
        "tid": int(tid), "frame": int(frames[e]), "dt": round(dt, 4),
        "history": [[round(float(u), 5), round(float(v), 5)] for u, v in trail_pts],
        "predicted": [[round(float(u), 5), round(float(v), 5)] for u, v in pred],
        "horizon_s": horizon_s, "steps": steps,
        "speed_px_s": round(speed, 1), "heading_deg": round(ang, 1),
        "anchor": [round(float(pts[n - 1][0]), 5), round(float(pts[n - 1][1]), 5)]
    }


def intent(tracks_json, tid, f):
    model = _load_model()
    if not model:
        return {"available": False, "reason": "意图模型未加载"}
    frames, feats = _series(tracks_json, tid)
    if frames is None:
        return {"available": False, "reason": "轨迹过短"}
    p = params()
    L = p["L_in"]
    seg = _contiguous_end(frames, f)
    if seg is None:
        return {"available": False, "reason": "无历史窗"}
    s, e = seg
    if e - s + 1 < L:
        return {"available": False, "reason": f"历史不足 {L} 个采样点"}
    x = feats[e - L + 1:e + 1].copy()
    mu = np.asarray(p["mu"], dtype=np.float32)
    sd = np.asarray(p["sd"], dtype=np.float32)
    x = (x - mu) / sd
    torch = _torch
    with torch.no_grad():
        logits = model(torch.from_numpy(x[None, :, :]))
        prob = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    order = np.argsort(-prob)
    classes = p["classes"]
    return {
        "available": True,
        "frame": int(frames[e]),
        "top": classes[int(order[0])],
        "confidence": round(float(prob[order[0]]), 3),
        "probs": {classes[i]: round(float(prob[i]), 3) for i in range(len(classes))},
        "L_in": L, "horizon_s": round(p["L_fut"] * p["stride"] / p["fps"], 2)
    }


def available():
    return bool(_load_model())
