#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""强收紧目检：同一帧、同一框，按不同缩放系数裁切，检查是否切到人体。

目的：决定"A 强收紧（0.65）"能不能替代人体掩膜。
  若 0.65 仍能完整保留人体 -> 可低成本试用；
  若明显切头/切身 -> 直接进入 B（SAM2/人体解析）。

用法：
  python tools/tighten_scale_check.py --seg 305_test_544 --tids 50,4,84 \
      --scales 1.0,0.85,0.75,0.65 --out output_crosscam_local/demo_tighten/check_305.jpg
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(ROOT, "media-resource/1012/segments")
AB = os.path.join(ROOT, "output_crosscam_local/1012_ab")
TILE_H = 260


def tracks_of(seg):
    t = {}
    for ab in ("a", "b"):
        d = {}
        for line in open(os.path.join(AB, f"{seg}_{ab}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            tids = r.get("track_ids") or []
            for i, b in enumerate(r.get("boxes") or []):
                if i < len(tids) and tids[i] is not None and int(tids[i]) >= 0:
                    d.setdefault(int(tids[i]), []).append((int(r["frame"]), b))
        for k in d:
            d[k].sort()
        for k, v in d.items():
            t.setdefault(k, []).extend(v)
    for k in t:
        t[k].sort()
    return t


def crop_scale(frame, box, s):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * s, (y2 - y1) * s
    X1, Y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    X2, Y2 = min(frame.shape[1], int(cx + w / 2)), min(frame.shape[0], int(cy + h / 2))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0:
        return None
    r = TILE_H / float(c.shape[0])
    return cv2.resize(c, (max(1, int(c.shape[1] * r)), TILE_H), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--tids", required=True, help="逗号分隔的 track id")
    ap.add_argument("--scales", default="1.0,0.85,0.75,0.65")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(",")]
    tids = [int(x) for x in a.tids.split(",")]

    tr = tracks_of(a.seg)
    cap = cv2.VideoCapture(os.path.join(VID, f"{a.seg}.mp4"))
    rows = []
    labels = []
    for tid in tids:
        lst = tr.get(tid) or []
        if not lst:
            print(f"[skip] #{tid} 无轨迹")
            continue
        # 代表性帧：轨迹中点（不是最大框，避免只挑最有利的样本）
        fnum, box = lst[len(lst) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fnum))
        ok, fr = cap.read()
        if not ok:
            print(f"[skip] #{tid} 帧 {fnum} 读取失败")
            continue
        tiles = []
        for s in scales:
            c = crop_scale(fr, box, s)
            if c is None:
                continue
            cv2.putText(c, f"{s:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3)
            cv2.putText(c, f"{s:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
            tiles.append(c)
        rows.append(tiles)
        labels.append(f"#{tid} f{fnum} box={int(box[2]-box[0])}x{int(box[3]-box[1])}")
    cap.release()
    if not rows:
        raise SystemExit("没有可用样本")

    w = max(sum(t.shape[1] for t in r) + 8 * (len(r) + 1) for r in rows)
    row_h = TILE_H + 26
    canvas = np.full((26 + row_h * len(rows), w, 3), 255, np.uint8)
    cv2.putText(canvas, f"{a.seg}  同帧同框不同缩放：1.00=原框，0.65=强收紧",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for r, (tiles, lab) in enumerate(zip(rows, labels)):
        y = 26 + r * row_h
        cv2.putText(canvas, lab, (8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        x = 8
        for t in tiles:
            h, ww = t.shape[:2]
            canvas[y + 20:y + 20 + h, x:x + ww] = t
            x += ww + 8
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, canvas)
    print(f"[out] {a.out}  rows={len(rows)}")


if __name__ == "__main__":
    main()
