#!/usr/bin/env python3
"""Containment-NMS：针对"同一目标被多尺度重复框"的去重。

背景
----
开放词表检测器（如 GroundingDINO）在 prompt="person" 时，会在多个语义尺度上
同时响应——"人头""上半身""全身"都符合语义，于是同一个人的位置会输出
多个大小不一的框。

普通 NMS 只看 IoU，而这类嵌套框的 IoU 往往只有 0.2~0.65（因为面积差 1.5~3 倍），
低于常规 NMS 阈值（0.5/0.7），因此**完全漏网**。

本模块改用「包含度」作为判据：

    containment = 交集面积 / 小框面积

只要小框有 ≥ `containment_threshold` 的面积落在大框内，就判定为同一目标的
重复响应，直接压制其中一个，保留面积更大的（即更完整的"全身"框）。

为什么保留大框而不是高分框
------------------------
实测同一人嵌套对的分数差仅 ±0.02（中位数 0.002），**分数几乎无区分度**；
而面积比中位数稳定在 1.9~2.0，是可靠特征。且业务是"数人"，全身框更符合
"一个人"的语义，也便于后续跟踪。

算法
----
1. 按面积从大到小排序，面积最大的框优先「先占」。
2. 依次考察后续框：若它与任意一个已保留框的包含度 ≥ 阈值，则丢弃。
3. 否则保留。

贪心 + 优先大框，保证「大框吞并小框」且结果稳定（不受输入顺序影响）。

注意
----
判据只看「包含」方向，因此**不会误伤真实相邻的人**——两个人并排时，
各占各的区域，互相包含度很低，会被完整保留。
"""
from __future__ import annotations

import numpy as np


def _areas(boxes: np.ndarray) -> np.ndarray:
    """xyxy 格式的框面积。"""
    w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return w * h


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐对计算交集面积。

    a: (N,4)  b: (M,4)  ->  (N,M)
    """
    lt = np.maximum(a[:, None, :2], b[None, :, :2])   # (N,M,2)
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])   # (N,M,2)
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def containment_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    containment_threshold: float = 0.85,
    keep: str = "larger",
    hy_score_k: float = 0.0,
    hy_min_h: float = 0.0,
) -> np.ndarray:
    """返回应该保留的索引（按面积降序）。

    参数
    ----
    boxes : (N,4) float，xyxy，像素坐标
    scores : (N,) float
    containment_threshold : float
        小框落在某大框内的最小面积占比，超过即视为重复。默认 0.85。
        调低更激进（去重更狠，但可能误杀）；调高更保守。
    keep : {"larger", "higher"}
        冲突时保留谁：
          - "larger"：保留面积更大的框（默认，推荐——分数无区分度，面积稳定）
          - "higher"：保留分数更高的框
    hy_score_k : float
        >0 时启用 **hybrid 选择**（C 方案，2026-11-10）：大框先占位，但若
        `scores[小框] >= scores[大框] * hy_score_k`，改由小框占位。
        起因（305 f4435 实测）：`keep=larger` 会保留 h=862 的超尺框（含桌面/椅子、
        分数 0.330），删掉 h=696 的高分框（0.414）；跟踪器尺寸跳变 +36% → 换号。
        6 路验证：305 换号 83→72（−13.3%），316/318/320/323/343 零负收益。
        **不要用 1.0**（等价 keep=larger）；推荐 0.8。
    hy_min_h : float
        hybrid 只对「小框高度 ≥ 此值」的嵌套组生效（0=不限制）。
        **必须 >0**：远景机位（318 框高 P50=105px）让「头/上半身」的小框顶掉
        「全身」大框会让跟踪目标更抖，实测 tracks 33→39、breaks 16→21（全面变差）。
        设 200 后 318/323 完全回到 base 行为。推荐 200。

    返回
    ----
    np.ndarray[int]：保留下来的框下标（已按面积从大到小排序）
    """
    n = len(boxes)
    if n == 0:
        return np.zeros(0, dtype=int)
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if n == 1:
        return np.array([0], dtype=int)

    areas = _areas(boxes)
    hybrid = hy_score_k > 0

    if keep == "larger" or hybrid:
        # ★ kind="stable"：面积完全相同的框（同一目标被重复检测）若用默认快排会
        # 得到不确定的顺序，进而改变 ByteTrack 的输入顺序 → 结果不可复现
        # （305 f23880 实测：不稳定排序导致与离线实验实现差 1 个框）。
        order = np.argsort(-areas, kind="stable")   # 面积大 -> 小
    elif keep == "higher":
        order = np.argsort(-scores, kind="stable")
    else:
        raise ValueError(f"keep must be 'larger' or 'higher', got {keep!r}")

    # 预计算所有框两两之间的包含度（对称矩阵的对应方向在下面取用）
    inter_all = _intersection(boxes, boxes)                 # (N,N)
    small_area = np.minimum(areas[:, None], areas[None, :])  # 每对的较小面积
    with np.errstate(divide="ignore", invalid="ignore"):
        contain_all = np.where(small_area > 0, inter_all / small_area, 0.0)

    if hybrid:
        # ---- hybrid：冲突时可能让位给小框（C 方案） ----
        heights = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
        kept_h: list[int] = []
        for idx in order:
            if not kept_h:
                kept_h.append(int(idx))
                continue
            cs = contain_all[idx, kept_h]
            pos = int(np.argmax(cs))
            if cs[pos] < containment_threshold:
                kept_h.append(int(idx))
                continue
            taker = kept_h[pos]
            if heights[idx] >= hy_min_h and scores[idx] >= scores[taker] * hy_score_k:
                kept_h.pop(pos)          # 大框让位
                kept_h.append(int(idx))
            # 否则丢弃小框（= keep=larger 的原语义）
        # 排序口径与 base 一致（面积降序），否则会改变 ByteTrack 的输入顺序
        return np.array(sorted(kept_h, key=lambda i: -areas[i]), dtype=int)

    kept: list[int] = []
    for idx in order:
        if not kept:
            kept.append(int(idx))
            continue
        # 与所有已保留框比较：只要有一个包含度超阈值，就丢弃
        if np.any(contain_all[idx, kept] >= containment_threshold):
            continue
        kept.append(int(idx))

    return np.array(kept, dtype=int)


def dedup_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    containment_threshold: float = 0.85,
    keep: str = "larger",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """便捷封装：直接返回去重后的 (boxes, scores, keep_indices)。"""
    keep_idx = containment_nms(boxes, scores, containment_threshold, keep)
    boxes = np.asarray(boxes, dtype=np.float64)
    return boxes[keep_idx], np.asarray(scores)[keep_idx], keep_idx
