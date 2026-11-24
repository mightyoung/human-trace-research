#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为一段新视频生成 query 候选标注包（拼图 + 模板 CSV）。

输出与 `gt/crops/{seg}/{tid}.jpg` 同格式：1200x634，2 行 4 列，
每格为目标中心裁剪（框扩展 2.2×）、绿框标注、底部帧号，顶部标题含 seg/tid/帧数。

用法：
  python tools/make_query_sheets.py --seg 318_test_534b \
      --det output_crosscam_local/1012_yolo8_dd_diag/318_test_534b/318_test_534b_detections.jsonl \
      --video media-resource/1012/0100000253400000000.mp4 \
      --out-dir deliverable/data-request-v2/gt_new --min-frames 6
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

import numpy as np
from PIL import Image, ImageDraw

import cv2

CELL_W, CELL_H = 300, 290
HEAD_H, FOOT_H = 26, 18
COLS, ROWS = 4, 2
GRID_W = CELL_W * COLS
GRID_H = HEAD_H + (CELL_H + FOOT_H) * ROWS
GREEN = (0, 200, 0)


def load_tracks(det_path, min_frames):
    t = {}
    for line in open(det_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tids = r.get("track_ids") or []
        boxes = r.get("boxes") or []
        for i, tid in enumerate(tids):
            if tid is None or tid < 0 or i >= len(boxes):
                continue
            t.setdefault(int(tid), []).append((int(r["frame"]), [float(v) for v in boxes[i]]))
    for k in t:
        t[k].sort()
    return {k: v for k, v in t.items() if len(v) >= min_frames}


def crop_around(frame, box, scale=2.2):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    # 按目标纵横比裁剪，避免拉伸
    target_ar = CELL_W / CELL_H
    if w / max(h, 1) > target_ar:
        h = w / target_ar
    else:
        w = h * target_ar
    ax1 = int(max(0, min(W - 2, cx - w / 2)))
    ay1 = int(max(0, min(H - 2, cy - h / 2)))
    ax2 = int(max(ax1 + 2, min(W, ax1 + w)))
    ay2 = int(max(ay1 + 2, min(H, ay1 + h)))
    patch = frame[ay1:ay2, ax1:ax2].copy()
    # 在 patch 上画目标框
    bx1, by1 = int(x1 - ax1), int(y1 - ay1)
    bx2, by2 = int(x2 - ax1), int(y2 - ay1)
    cv2.rectangle(patch, (bx1, by1), (bx2, by2), GREEN, 2)
    return cv2.resize(patch, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default="deliverable/data-request-v2/gt_new")
    ap.add_argument("--min-frames", type=int, default=6)
    ap.add_argument("--samples", type=int, default=8)
    a = ap.parse_args()

    tracks = load_tracks(a.det, a.min_frames)
    print(f"[{a.seg}] 轨迹 {len(tracks)} 条（>= {a.min_frames} 帧）")

    # 每条轨迹取 samples 个均匀帧
    req = []          # (frame_no, tid, box_of_that_frame)
    pick = {}
    for tid, tf in tracks.items():
        idxs = np.linspace(0, len(tf) - 1, min(a.samples, len(tf))).astype(int)
        pick[tid] = [tf[i] for i in idxs]
        for fno, box in pick[tid]:
            req.append((fno, tid, box))
    req.sort(key=lambda x: x[0])

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    got = {}          # tid -> list[(frame_no, patch)]
    cur = -1
    frame = None
    for fno, tid, box in req:
        while cur < fno:
            ok, frame = cap.read()
            if not ok:
                break
            cur += 1
        if frame is None or cur != fno:
            continue
        got.setdefault(tid, []).append((fno, crop_around(frame, box)))
    cap.release()

    out_seg = os.path.join(a.out_dir, "crops", a.seg)
    os.makedirs(out_seg, exist_ok=True)
    rows = []
    for tid in sorted(tracks):
        items = got.get(tid, [])
        if not items:
            continue
        canvas = Image.new("RGB", (GRID_W, GRID_H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, GRID_W, HEAD_H - 1], fill=(24, 26, 30))
        hs = [b[3] - b[1] for _, b in tracks[tid]]
        d.text((6, 7), f"{a.seg} #{tid}  n_frames={len(tracks[tid])}", fill=(255, 255, 255))
        for j, (fno, patch) in enumerate(items[:COLS * ROWS]):
            r, c = divmod(j, COLS)
            x, y = c * CELL_W, HEAD_H + r * (CELL_H + FOOT_H)
            canvas.paste(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)), (x, y))
            d.text((x + 4, y + CELL_H + 3), f"f{fno}", fill=(60, 60, 60))
        canvas.save(os.path.join(out_seg, f"{tid}.jpg"), quality=92)
        rows.append(dict(
            query_id=f"{a.seg}#{tid}", seg=a.seg, cam=a.seg.split("_")[0],
            n_frames=len(tracks[tid]),
            median_box_h=f"{st.median(hs):.1f}",
            evaluable="", exclude_reason="",
            prompt_jpg=f"crops/{a.seg}/{tid}.jpg",
            person_id="", box_type="", duplicate_of="", unk_reason="",
            evidence="", annotator="", time_spent_s=""))

    csv_path = os.path.join(a.out_dir, f"{a.seg}_query_TEMPLATE.csv")
    cols = ["query_id", "seg", "cam", "n_frames", "median_box_h", "evaluable",
            "exclude_reason", "prompt_jpg", "person_id", "box_type", "duplicate_of",
            "unk_reason", "evidence", "annotator", "time_spent_s"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} 条 -> {csv_path}")
    print(f"       拼图 -> {out_seg}/")


if __name__ == "__main__":
    sys.exit(main())
