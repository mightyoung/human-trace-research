#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把误合并清单渲染成人工复核用的拼图（每对一张裁切，附段/帧/id）。

输入：export_false_merges.py 产出的 CSV（seg,id_u,id_v,frame,cluster,score,dir）
输出：fm_audit/audit_<k>.jpg，每张 9 对（3x3），每格画 U/V 两个框并标注 id。

用法：
  python tools/render_fm_pairs.py --csv .../fm_pairs_yolo_same95.csv \
      --det-dir output_crosscam_local/1012_yolo8 --out-dir .../fm_audit
"""
import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np

CELL = 360
PAD = 40


def load_dets(p):
    frames = {}
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        frames[d["frame"]] = (d.get("track_ids") or [], d.get("boxes") or [])
    return frames


def crop_with_boxes(img, boxes, tags):
    h, w = img.shape[:2]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    cx1, cy1 = int(max(0, x1 - PAD)), int(max(0, y1 - PAD))
    cx2, cy2 = int(min(w, x2 + PAD)), int(min(h, y2 + PAD))
    if cx2 - cx1 < 24 or cy2 - cy1 < 24:
        cx1, cy1, cx2, cy2 = 0, 0, w, h
    crop = img[cy1:cy2, cx1:cx2].copy()
    for b, (lab, col) in zip(boxes, tags):
        cv2.rectangle(crop, (int(b[0]) - cx1, int(b[1]) - cy1),
                      (int(b[2]) - cx1, int(b[3]) - cy1), col, 2)
        cv2.putText(crop, lab, (int(b[0]) - cx1, max(12, int(b[1]) - cy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    crop = cv2.resize(crop, (CELL, CELL), interpolation=cv2.INTER_LINEAR)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--det-pattern", default="{det}/{seg}/{seg}_detections.jsonl",
                    help="DINO 基线用 \"{det}/{seg}.jsonl\"")
    ap.add_argument("--vid-dir", default="media-resource/1012/segments")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-sheet", type=int, default=9)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_det, cache_cap = {}, {}
    tiles, labels = [], []

    for r in rows:
        seg, f = r["seg"], int(r["frame"])
        iu, iv = int(r["id_u"]), int(r["id_v"])
        if seg not in cache_det:
            cache_det[seg] = load_dets(a.det_pattern.format(det=a.det_dir, seg=seg))
        ids, boxes = cache_det[seg].get(f, ([], []))
        if iu not in ids or iv not in ids:
            print(f"[miss] {seg} f={f} ids={iu},{iv}")
            continue
        bu, bv = boxes[ids.index(iu)], boxes[ids.index(iv)]
        img = None
        if seg not in cache_cap:
            cap = cv2.VideoCapture(os.path.join(a.vid_dir, f"{seg}.mp4"))
            cache_cap[seg] = cap
        cap = cache_cap[seg]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            print(f"[noread] {seg} f={f}")
            continue
        tile = crop_with_boxes(img, [bu, bv], [(f"U{iu}", (0, 255, 255)), (f"V{iv}", (0, 0, 255))])
        cv2.putText(tile, f"{seg} f={f} s={r['score']}", (6, CELL - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        tiles.append(tile)
        labels.append(f"{seg} f={f} {iu}/{iv}")

    n = a.per_sheet
    cols = 3
    for k in range(0, len(tiles), n):
        chunk = tiles[k:k + n]
        rows_n = (len(chunk) + cols - 1) // cols
        sheet = np.zeros((rows_n * CELL, cols * CELL, 3), np.uint8)
        for i, t in enumerate(chunk):
            rr, cc = divmod(i, cols)
            sheet[rr * CELL:(rr + 1) * CELL, cc * CELL:(cc + 1) * CELL] = t
        p = out / f"audit_{k // n + 1}.jpg"
        cv2.imwrite(str(p), sheet)
        print(f"[out] {p}  ({len(chunk)} 对)")
    for cap in cache_cap.values():
        cap.release()
    print(f"[total] {len(tiles)} 对 -> {out}")


if __name__ == "__main__":
    main()
