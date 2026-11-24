#!/usr/bin/env python
"""把指定轨迹群的成员各取代表帧做 crop 拼图（核验"是否同一人"）。

用法:
  python tools/group_strip.py --video 318.mp4 --jsonl cleaned.jsonl \
      --tids "3,42,49,50" --out strip.png --crop-h 170 --cols 7
"""
import argparse
import json

import cv2
import numpy as np


def load_tracks(jsonl_path):
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((r["frame"], b))
    for k in t:
        t[k].sort()
    return t


def crop_person(frame, box, crop_h, pad=0.45, max_w=240):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad))
    Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad))
    Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0:
        return None
    s = crop_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, crop_h))
    if nw > max_w:
        off = (nw - max_w) // 2
        c = c[:, off:off + max_w]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--tids", required=True, help="逗号分隔的 tid 列表")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-h", type=int, default=170)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--samples", type=int, default=2,
                    help="每条轨迹取几个采样帧")
    args = ap.parse_args()

    tids = [int(x) for x in args.tids.split(",")]
    tracks = load_tracks(args.jsonl)
    cap = cv2.VideoCapture(args.video)

    cell_w = 250
    label_h = 40
    cell_h = args.crop_h + label_h
    rows = (len(tids) + args.cols - 1) // args.cols
    canvas = np.full((rows * cell_h + 60, args.cols * cell_w + 20, 3),
                     24, dtype=np.uint8)

    header = f"group strip - {len(tids)} tracks (each: {args.samples} frames)"
    cv2.putText(canvas, header, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (120, 220, 255), 2, cv2.LINE_AA)

    for i, tid in enumerate(tids):
        obs = tracks.get(tid, [])
        if not obs:
            continue
        n = len(obs)
        idxs = [n // 4, 3 * n // 4] if args.samples == 2 else [n // 2]
        crops = []
        for ix in idxs[:args.samples]:
            fr_i, box = obs[ix]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr_i)
            ok, fr = cap.read()
            if not ok:
                continue
            c = crop_person(fr, box, args.crop_h)
            if c is not None:
                crops.append(c)
        if not crops:
            continue
        r, c = divmod(i, args.cols)
        x0 = 10 + c * cell_w
        y0 = 60 + r * cell_h + label_h
        lab = f"#{tid} n={n}"
        cv2.putText(canvas, lab, (x0 + 4, y0 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (80, 255, 120), 2,
                    cv2.LINE_AA)
        xx = x0
        for cr in crops:
            h, w = cr.shape[:2]
            if xx + w > x0 + cell_w - 4:
                w = x0 + cell_w - 4 - xx
            if w <= 2:
                break
            canvas[y0:y0 + h, xx:xx + w] = cr[:, :w]
            xx += w + 3
        cv2.rectangle(canvas, (x0, y0 - 4), (x0 + cell_w - 2, y0 + args.crop_h),
                      (70, 70, 70), 1)
    cv2.imwrite(args.out, canvas)
    print(f"saved {args.out} {canvas.shape}")


if __name__ == "__main__":
    main()
