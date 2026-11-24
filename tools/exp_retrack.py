#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验：不重跑检测器，只把已有的「每帧框」重新喂给 ByteTrack，看换号/断点能不能减少。

为什么可行：
  `{cam}_detections.jsonl` 每行都存了该帧去重后的 `boxes`/`scores`，只有 `track_ids`
  是旧 tracker 分配的。所以可以在纯 CPU 上重放跟踪（几千帧几秒），不需要重跑 DINO。
  局限：Containment-NMS 已经删过的框救不回来（305 占原始框 13.7%），所以这是"上限之内的改善"。

对照：现状 `minimum_matching_threshold=0.8`（ByteTrack 默认，面向 30fps 逐帧），
      但我们是 stride=5（等效 5fps），相邻观测位移更大，0.8 偏严。

用法：
  python tools/exp_retrack.py --cam 305 --root output_crosscam/long35 --suffix _long35_detections.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def stats(tracks: dict[int, list[tuple[int, list[float]]]], stride: int = 5) -> dict:
    f1 = max(f for seq in tracks.values() for f, _ in seq) if tracks else 0
    n_break = 0
    for t, seq in tracks.items():
        seq.sort(key=lambda x: x[0])
        for i in range(1, len(seq)):
            if seq[i][0] - seq[i - 1][0] > stride and seq[i - 1][0] < f1 - 2 * stride:
                n_break += 1
                break
    # ID switch：A 结束后 8s 内同位置出现新 B
    per = defaultdict(list)
    for t, seq in tracks.items():
        for f, b in seq:
            per[f].append((t, b))
    starts = {t: min(f for f, _ in seq) for t, seq in tracks.items()}
    n_sw = 0
    for t, seq in tracks.items():
        last_f, last_b = seq[-1]
        if last_f >= f1 - 2 * stride:
            continue
        for g in range(stride, stride * 41, stride):
            fr = last_f + g
            if fr not in per:
                continue
            for t2, b2 in per[fr]:
                if t2 == t or starts.get(t2, 0) <= last_f:
                    continue
                if iou(last_b, b2) >= 0.5:
                    n_sw += 1
                    break
            else:
                continue
            break
    lens = sorted(len(s) for s in tracks.values())
    return {"tracks": len(tracks), "breaks": n_break, "switches": n_sw,
            "len_p50": lens[len(lens) // 2] if lens else 0,
            "short_lt5": sum(1 for x in lens if x < 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="305")
    ap.add_argument("--root", default="output_crosscam/long35")
    ap.add_argument("--suffix", default="_long35_detections.jsonl")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--thrs", default="0.8,0.6,0.5,0.4")
    a = ap.parse_args()

    import supervision as sv

    p = Path(a.root) / f"{a.cam}{a.suffix}"
    frames = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            frames.append((int(d["frame"]), d.get("boxes") or [], d.get("scores") or []))
    print(f"{a.cam}: {len(frames)} 帧")

    # 现状 track（从文件里的 track_ids）
    base = defaultdict(list)
    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for t, b in zip(d.get("track_ids") or [], d.get("boxes") or []):
                base[int(t)].append((int(d["frame"]), b))
    s0 = stats(base)
    print(f"  现状 tracker  : {s0}")

    for thr in [float(x) for x in a.thrs.split(",")]:
        tracker = sv.ByteTrack(track_activation_threshold=0.10,
                               lost_track_buffer=90,
                               minimum_matching_threshold=thr,
                               frame_rate=a.fps)
        out = defaultdict(list)
        for fr, boxes, scores in frames:
            if not boxes:
                tracker.update_with_detections(sv.Detections.empty())
                continue
            det = sv.Detections(xyxy=np.asarray(boxes, dtype=np.float32),
                               confidence=np.asarray(scores, dtype=np.float32),
                               class_id=np.zeros(len(boxes), dtype=int))
            det = tracker.update_with_detections(det)
            tids = det.tracker_id
            if tids is None or len(tids) == 0:
                continue
            for t, b in zip(tids, det.xyxy):
                out[int(t)].append((fr, [float(v) for v in b]))
        s = stats(out)
        print(f"  重放 thr={thr:.1f}  : {s}   Δ轨 {s['tracks']-s0['tracks']:+d} "
              f"Δ断点 {s['breaks']-s0['breaks']:+d} Δ切换 {s['switches']-s0['switches']:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
