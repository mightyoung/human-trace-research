# -*- coding: utf-8 -*-
"""轨迹 / 实体级身份特征聚合（评审 F4/F5）。

- select_frames：按质量选 K 帧，同时保证时间分散（避免 K 帧挤在同一秒、同一姿态）。
- aggregate：质量加权均值（L2 后）；可选 trimmed（去掉与中心最不一致的 20%，抗混入他人）。
- EntityFeature：实体 = 若干碎片拼接；接力打分改用**整条实体**的高质量特征，
  头/尾特征仅作补充（取 max），不再"只用首尾碎片"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l2n(x, axis=-1):
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def select_frames(frames, weights, k=8, min_gap=5):
    """frames: 帧号序列；weights: 对应质量权重。返回被选下标（按帧号升序）。
    贪心：每次取剩余最高权重帧，与已选帧的帧号间隔 < min_gap 则跳过；不足 K 时放宽间隔再补。"""
    frames = np.asarray(frames)
    w = np.asarray(weights, float)
    order = np.argsort(-w, kind="stable")
    chosen = []
    for gap in (min_gap, max(1, min_gap // 2), 0):
        for i in order:
            if len(chosen) >= k:
                break
            if w[i] <= 0 or i in chosen:
                continue
            if all(abs(int(frames[i]) - int(frames[j])) >= gap for j in chosen):
                chosen.append(int(i))
        if len(chosen) >= k:
            break
    return sorted(chosen, key=lambda i: frames[i])


def aggregate(embs, weights=None, trim=0.0):
    """embs: (N,D)；weights: (N,)。返回 L2 归一的 (D,)；N=0 返回 None。"""
    E = l2n(np.asarray(embs, np.float32).reshape(-1, np.asarray(embs).shape[-1]))
    if len(E) == 0:
        return None
    w = np.ones(len(E)) if weights is None else np.asarray(weights, float)
    if w.sum() <= 0:
        return None
    c = l2n((E * w[:, None]).sum(0))
    if trim > 0 and len(E) >= 5:
        sim = E @ c
        keep = sim >= np.quantile(sim, trim)
        E, w = E[keep], w[keep]
        c = l2n((E * w[:, None]).sum(0))
    return c


@dataclass
class EntityFeature:
    full: np.ndarray            # 全实体质量加权特征
    head: np.ndarray | None     # 入画后 head_s 秒内的高质量特征
    tail: np.ndarray | None     # 出画前 tail_s 秒内的高质量特征
    n_used: int
    q_mean: float


def entity_feature(frame_ids, embs, weights, fps=25.0, head_s=4.0, tail_s=4.0, trim=0.2):
    """frame_ids/embs/weights: 该实体（碎片已拼接）所有**已编码帧**。"""
    f = np.asarray(frame_ids, float)
    w = np.asarray(weights, float)
    if len(f) == 0 or w.sum() <= 0:
        return None
    full = aggregate(embs, w, trim=trim)
    t0, t1 = f.min(), f.max()
    hm = f <= t0 + head_s * fps
    tm = f >= t1 - tail_s * fps
    head = aggregate(np.asarray(embs)[hm], w[hm]) if (hm.any() and w[hm].sum() > 0) else None
    tail = aggregate(np.asarray(embs)[tm], w[tm]) if (tm.any() and w[tm].sum() > 0) else None
    return EntityFeature(full=full, head=head, tail=tail, n_used=int((w > 0).sum()),
                         q_mean=float(w[w > 0].mean()) if (w > 0).any() else 0.0)


def relay_similarity(a: EntityFeature, b: EntityFeature, mode="full_max"):
    """A（离开）→ B（到达）的外观相似度。
    full      : 仅全实体特征
    full_max  : max(full·full, tail_A·head_B)——全实体为主，交接两端近景特征作补充（默认）
    legacy    : tail_A·head_B（近似旧"末碎片⇄首碎片"口径，仅对照用）"""
    s_full = float(a.full @ b.full)
    if mode == "full":
        return s_full
    s_th = float(a.tail @ b.head) if (a.tail is not None and b.head is not None) else None
    if mode == "legacy":
        return s_th if s_th is not None else s_full
    return max(s_full, s_th) if s_th is not None else s_full
