#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""排查「两人重合就丢身份 id」：量化 track 断点 / ID switch / 遮挡期丢失。

现象（用户报）：两个人走近重叠时，其中一个人的框/id 消失，之后要么以新 id 出现，
要么彻底没有。可能发生在三层，本脚本按层分别量化：

  1. raw 检测层：`{cam}_detections.jsonl`（模型 + Containment-NMS + ByteTrack）
     —— Containment-NMS 会把「落在别人框里的小框」删掉（默认 0.85 包含度），
        两人前后重叠时，后面那个人可能整框落进前面那个人框内 → 被判重复删掉。
  2. 清洗层：`cleaning/{cam}_cleaned.jsonl`
     —— 清洗会不会在断点处进一步删框。
  3. 实体层：`stitch_long35/net_tracks_{cam}.json`
     —— 拼接把断掉的 track 接回去没有。

判定口径（不依赖真值）：
  * **断点**：track 的帧序列出现 gap（相邻观测差 > stride），且不是视频边界。
  * **ID switch**：track A 在帧 f 之后消失，同机位在 f~f+gap 内出现 track B，
    且 B 的首框与 A 的末框 IoU ≥ 0.5（同一位置、同一尺度）。
  * **重合帧**：同帧存在两框 IoU ≥ 0.5 或包含度 ≥ 0.85。

用法：
  python tools/diag_id_drop.py --root output_crosscam/long35 --cams 305,318 --det-suffix "_long35_detections.jsonl"
  python tools/diag_id_drop.py --root output_crosscam --cams 305 --det-suffix "_detections.jsonl"
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


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(inner, outer) -> float:
    """inner 有多少面积落在 outer 里。"""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = area(inner)
    return (iw * ih) / a if a > 0 else 0.0


def load_rows(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def analyze(rows: list[dict], stride: int, gap_tol: int = 1) -> dict:
    """返回 track 断点与 ID switch 统计。"""
    # track -> [(frame, box)]
    tr: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    frame_boxes: dict[int, list[tuple[int, list[float]]]] = {}
    for r in rows:
        fr = int(r["frame"])
        tids = r.get("track_ids") or []
        boxes = r.get("boxes") or []
        frame_boxes[fr] = [(int(t), b) for t, b in zip(tids, boxes)]
        for t, b in zip(tids, boxes):
            tr[int(t)].append((fr, b))
    for t in tr:
        tr[t].sort(key=lambda x: x[0])

    frames = sorted(frame_boxes)
    f0, f1 = frames[0], frames[-1]

    # ---- 断点 ----
    breaks = []
    for t, seq in tr.items():
        step = stride * (1 + gap_tol)
        for i in range(1, len(seq)):
            d = seq[i][0] - seq[i - 1][0]
            if d > step:
                # 最后一条观测不是视频结尾才算"中途断"
                if seq[i - 1][0] < f1 - step:
                    breaks.append({"tid": t, "f_last": seq[i - 1][0], "f_next": seq[i][0],
                                   "gap_frames": d, "gap_s": d / 25.0,
                                   "box_last": seq[i - 1][1], "box_next": seq[i][1]})
                break  # 只记第一次断

    # ---- ID switch：A 消失后，同位置出现新 B ----
    switches = []
    for t, seq in tr.items():
        last_f, last_b = seq[-1]
        if last_f >= f1 - stride * 2:      # 自然结束（走到视频末尾）不算
            continue
        for g in range(stride, stride * 41, stride):   # 往后看 8s
            fr = last_f + g
            if fr not in frame_boxes:
                continue
            for (t2, b2) in frame_boxes[fr]:
                if t2 == t:
                    continue
                if t2 in tr and tr[t2][0][0] <= last_f:  # B 必须是在 A 之后新出现的
                    continue
                ov = iou(last_b, b2)
                if ov >= 0.5:
                    switches.append({"tid_old": t, "tid_new": t2, "f_last": last_f,
                                     "f_new": fr, "dt_s": g / 25.0, "iou": round(ov, 3),
                                     "box": [round(float(v), 1) for v in last_b]})
                    break
            else:
                continue
            break

    # ---- 重合帧统计 ----
    n_overlap_frames = 0
    n_pair = 0
    contain_hits = 0
    for fr, items in frame_boxes.items():
        hit = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                bi, bj = items[i][1], items[j][1]
                if iou(bi, bj) >= 0.5:
                    hit = True
                    n_pair += 1
                ai, aj = area(bi), area(bj)
                if ai > 0 and aj > 0:
                    small, big = (bi, bj) if ai < aj else (bj, bi)
                    if containment(small, big) >= 0.85:
                        contain_hits += 1
                        hit = True
        if hit:
            n_overlap_frames += 1

    n_tracks = len(tr)
    return {
        "frames": len(frames), "frame_range": [f0, f1],
        "tracks": n_tracks,
        "breaks": len(breaks), "break_detail": breaks[:40],
        "switches": len(switches), "switch_detail": switches[:40],
        "overlap_frames": n_overlap_frames,
        "overlap_pairs": n_pair,
        "containment_pairs": contain_hits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output_crosscam/long35")
    ap.add_argument("--cams", default="305,316,318,320,323,343")
    ap.add_argument("--det-suffix", default="_long35_detections.jsonl")
    ap.add_argument("--cleaned-suffix", default="")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    root = Path(a.root)
    cams = [c.strip() for c in a.cams.split(",") if c.strip()]
    allout = {}
    print(f"{'cam':4s} {'轨':>5s} {'断点':>5s} {'ID切换':>6s} {'重合帧':>6s} {'重合对':>6s} {'包含对':>6s}")
    for cam in cams:
        p = root / f"{cam}{a.det_suffix}"
        if not p.exists():
            print(f"{cam} 缺 {p}")
            continue
        rows = load_rows(p)
        r = analyze(rows, a.stride)
        allout[cam] = r
        print(f"{cam:4s} {r['tracks']:5d} {r['breaks']:5d} {r['switches']:6d} "
              f"{r['overlap_frames']:6d} {r['overlap_pairs']:6d} {r['containment_pairs']:6d}")

    if a.cleaned_suffix:
        print(f"\n--- 清洗层（{a.cleaned_suffix}）---")
        for cam in cams:
            p = root / "cleaning" / f"{cam}{a.cleaned_suffix}"
            if not p.exists():
                continue
            r = analyze(load_rows(p), a.stride)
            allout[f"{cam}_cleaned"] = r
            print(f"{cam:4s} {r['tracks']:5d} {r['breaks']:5d} {r['switches']:6d} "
                  f"{r['overlap_frames']:6d} {r['overlap_pairs']:6d} {r['containment_pairs']:6d}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(allout, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        print(f"\n-> {a.json_out}")

    # 抽样打印最可疑的 ID switch
    for cam, r in allout.items():
        if not r["switch_detail"]:
            continue
        print(f"\n== {cam} ID switch 样本（IoU 高=同一人被换号）")
        for s in r["switch_detail"][:5]:
            print(f"   f={s['f_last']:6d} t{s['tid_old']:>3d} → t{s['tid_new']:<3d} "
                  f"(+{s['dt_s']:.1f}s, IoU {s['iou']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
