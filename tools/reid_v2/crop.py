# -*- coding: utf-8 -*-
"""人体裁剪 v2：按 ReID 惯例的 2:1（高:宽）比例，从**原分辨率帧**取真实上下文，不做边缘复制补边。

旧实现（embed_server._prep_crop / crosscam_reid.prep_crop）的问题（评审 F1/F2）：
  - 先缩到高 224，再把宽**复制边缘像素**补成正方形 → 316/323/343 机位方图中 56–66% 是"涂抹"像素；
  - 宽框（坐姿）居中裁掉两侧；再被 RADIO 处理器放大到 512（远景总放大 5.6×）。

新实现：
  1. 把检测框按目标宽高比扩展（宽不足→左右扩、宽过大→上下扩），扩展区域取**真实画面**；
  2. 扩展框若越界，先在画面内平移；仍放不下（贴边/超大框）才用**常量色**补齐（不涂抹）；
  3. 一次 resize 到目标尺寸（缩小用 INTER_AREA，放大用 INTER_LINEAR），不二次放大；
  4. 返回 CropMeta（越界比例、缩放系数），供质量评分使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# ImageNet 均值（RGB, 0-255）作为常量填充色：归一化后≈0，对网络最"中性"
FILL_RGB = (124, 116, 104)


@dataclass
class CropMeta:
    box_in: tuple            # 原检测框 xyxy
    box_ctx: tuple           # 扩展后（未裁剪前）的目标框 xyxy
    fill_frac: float         # 输出图中常量填充所占面积比例（越界造成）
    scale: float             # 输出高 / 扩展框高（<1 缩小，>1 放大）
    touches_border: tuple    # (left, top, right, bottom) 原框是否贴画面边缘


def expand_to_aspect(box, aspect_wh: float, pad: float = 0.06):
    """把 xyxy 框外扩 pad 后，调整到宽/高 = aspect_wh（只扩不缩，中心不变）。"""
    x1, y1, x2, y2 = map(float, box)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = w * (1 + 2 * pad), h * (1 + 2 * pad)
    if w / h < aspect_wh:
        w = h * aspect_wh
    else:
        h = w / aspect_wh
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _shift_inside(b, W, H):
    """在画面内平移扩展框（尽量用真实像素替代填充）；放不下的维度保持居中。"""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    if w <= W:
        if x1 < 0:
            x1, x2 = 0.0, w
        elif x2 > W:
            x1, x2 = W - w, float(W)
    if h <= H:
        if y1 < 0:
            y1, y2 = 0.0, h
        elif y2 > H:
            y1, y2 = H - h, float(H)
    return (x1, y1, x2, y2)


def border_touch(box, W, H, margin_px: float = 2.0):
    x1, y1, x2, y2 = box
    return (x1 <= margin_px, y1 <= margin_px, x2 >= W - margin_px, y2 >= H - margin_px)


def crop_person(frame: np.ndarray, box, out_hw=(256, 128), pad: float = 0.06,
                fill=FILL_RGB, bgr_input: bool = True):
    """返回 (rgb uint8 [out_h,out_w,3], CropMeta) ；框无效返回 (None, None)。"""
    if cv2 is None:
        raise RuntimeError("需要 opencv-python")
    H, W = frame.shape[:2]
    out_h, out_w = out_hw
    x1, y1, x2, y2 = map(float, box)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None, None
    ctx = _shift_inside(expand_to_aspect(box, out_w / out_h, pad), W, H)
    cx1, cy1, cx2, cy2 = ctx
    cw, ch = cx2 - cx1, cy2 - cy1
    # 画布坐标：扩展框映射到 out_hw
    sx, sy = out_w / cw, out_h / ch
    ix1, iy1 = max(0, int(np.floor(cx1))), max(0, int(np.floor(cy1)))
    ix2, iy2 = min(W, int(np.ceil(cx2))), min(H, int(np.ceil(cy2)))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None, None
    patch = frame[iy1:iy2, ix1:ix2]
    if bgr_input:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    # 先把真实像素块整体缩放，再按"与画布的交集"放置（亚像素取整可能使偏移为 −1，必须裁剪）
    ox, oy = int(round((ix1 - cx1) * sx)), int(round((iy1 - cy1) * sy))
    pw = max(1, int(round((ix2 - ix1) * sx)))
    ph = max(1, int(round((iy2 - iy1) * sy)))
    interp = cv2.INTER_AREA if (ph < patch.shape[0]) else cv2.INTER_LINEAR
    patch = cv2.resize(patch, (pw, ph), interpolation=interp)
    canvas = np.empty((out_h, out_w, 3), np.uint8)
    canvas[:] = np.asarray(fill, np.uint8)
    x0, y0 = max(0, ox), max(0, oy)
    x1_, y1_ = min(out_w, ox + pw), min(out_h, oy + ph)
    if x1_ <= x0 or y1_ <= y0:
        return None, None
    canvas[y0:y1_, x0:x1_] = patch[y0 - oy:y1_ - oy, x0 - ox:x1_ - ox]
    fill_frac = 1.0 - ((x1_ - x0) * (y1_ - y0)) / float(out_w * out_h)
    meta = CropMeta(box_in=(x1, y1, x2, y2), box_ctx=ctx, fill_frac=round(max(0.0, fill_frac), 4),
                    scale=out_h / ch, touches_border=border_touch(box, W, H))
    return canvas, meta


def square_letterbox(rgb: np.ndarray, size: int = 224, fill=FILL_RGB):
    """给只吃方图的旧编码器（FastEncoder 模板 224²）用：等比缩放 + 常量色左右/上下补齐（不复制边缘）。"""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    im = cv2.resize(rgb, (nw, nh), interpolation=interp)
    out = np.empty((size, size, 3), np.uint8)
    out[:] = np.asarray(fill, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = im
    return out


def legacy_prep_crop(frame, box, out_h=224, pad=0.08):
    """旧口径（逐行复刻 embed_server._prep_crop），仅供 A/B 对照。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad)); Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad)); Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
        return None
    s = out_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, out_h))
    if nw >= out_h:
        off = (nw - out_h) // 2
        c = c[:, off:off + out_h]
    else:
        padw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
