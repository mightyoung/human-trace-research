# -*- coding: utf-8 -*-
"""逐帧观测质量评分（评审 F4/F5）：决定哪些帧进入身份特征、权重多大。

旧实现：每轨均匀取 8 帧、等权平均；接力只用首/末碎片——恰好是贴边、截断、背身最多的时刻。

质量因子（均为 [0,1]，越大越好；可由配置关闭任一项）：
  trunc   画面边缘截断：框贴左/右/下边视为截断（贴上边常见于近景，权重较低）
  occl    被其他检测框覆盖的比例（IoA = 交集 / 本框面积）——重叠帧会混入他人外观
  size    人高像素（太小的远景帧信息不足）
  sharp   拉普拉斯方差（运动模糊 / 低光）
  refl    倒影/墙镜标记（来自 clean2 的 S5/S6 结果）→ 直接置 0
  det     检测分（低分帧更可能是部分可见或误检）
合成：w = size·sharp·det^γ·(1−trunc)²·(1−occl)²，refl → 0。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass
class QualityConfig:
    h_ref: float = 160.0            # 人高达到该像素即 size=1（按 2560×1440 / 1920×1080 原分辨率）
    h_min: float = 40.0             # 低于此高度 size=0
    sharp_ref: float = 120.0        # 拉普拉斯方差达到该值 sharp=1（256×128 裁剪上测）
    det_gamma: float = 0.5
    trunc_weights: tuple = (1.0, 0.3, 1.0, 1.0)   # 左、上、右、下 贴边的截断权重
    occl_cap: float = 0.9           # IoA 超过即 occl 因子归零
    min_weight: float = 0.02        # 低于此权重视为不可用
    use: dict = field(default_factory=lambda: dict(trunc=True, occl=True, size=True,
                                                   sharp=True, det=True, refl=True))


def box_ioa(box, others):
    """本框被 others 覆盖的最大比例（交集 / 本框面积）。others: (M,4)"""
    if others is None or len(others) == 0:
        return 0.0
    b = np.asarray(box, float)
    o = np.asarray(others, float).reshape(-1, 4)
    lt = np.maximum(b[:2], o[:, :2])
    rb = np.minimum(b[2:], o[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return float(np.max(inter) / area)


def truncation(touches_border, weights=(1.0, 0.3, 1.0, 1.0)):
    t = sum(w for flag, w in zip(touches_border, weights) if flag)
    return float(min(1.0, t))


def sharpness(rgb_crop):
    if rgb_crop is None or cv2 is None:
        return 0.0
    g = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def frame_quality(box, other_boxes, touches_border, score=None, rgb_crop=None,
                  is_reflection=False, cfg: QualityConfig | None = None):
    """返回 (weight, factors dict)。rgb_crop 可为 None（只用几何因子，适合离线预筛选）。"""
    cfg = cfg or QualityConfig()
    u = cfg.use
    h = float(box[3] - box[1])
    f = {}
    f["size"] = float(np.clip((h - cfg.h_min) / max(cfg.h_ref - cfg.h_min, 1e-6), 0, 1)) if u["size"] else 1.0
    f["trunc"] = truncation(touches_border, cfg.trunc_weights) if u["trunc"] else 0.0
    ioa = box_ioa(box, other_boxes) if u["occl"] else 0.0
    f["occl"] = float(min(1.0, ioa / cfg.occl_cap))
    f["sharp"] = float(np.clip(sharpness(rgb_crop) / cfg.sharp_ref, 0, 1)) if (u["sharp"] and rgb_crop is not None) else 1.0
    f["det"] = float(np.clip(score, 0, 1) ** cfg.det_gamma) if (u["det"] and score is not None) else 1.0
    f["refl"] = bool(is_reflection) if u["refl"] else False
    w = f["size"] * f["sharp"] * f["det"] * (1 - f["trunc"]) ** 2 * (1 - f["occl"]) ** 2
    if f["refl"]:
        w = 0.0
    w = float(w) if w >= cfg.min_weight else 0.0
    return w, f
