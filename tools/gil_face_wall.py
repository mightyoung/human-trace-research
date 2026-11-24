#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 交付可视化：per-gperson「全楼最佳人脸墙」+ 覆盖统计

用法：python tools/gil_face_wall.py --tag p0
输出：analysis/face/face_wall_gperson.jpg + stdout 统计
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "face")
CROPS_DIR = os.path.join(FACE_DIR, "crops")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--cell", type=int, default=180)
    ap.add_argument("--out", default="face_wall_gperson.jpg")
    args = ap.parse_args()

    with open(os.path.join(FACE_DIR, f"face_best_{args.tag}.json"), encoding="utf-8") as f:
        best = json.load(f)
    crop_map = best["crop_map"]
    per_gp = best["per_gperson"]
    gp_label = best.get("gp_label", {})

    # ---- 统计 ----
    n_gp_any = len(per_gp)
    n_gp_crop = 0
    qs = []
    per_cam = defaultdict(lambda: dict(gp=0, crops=0))
    tiles = []
    for gp_id, lst in sorted(per_gp.items(), key=lambda x: int(x[0])):
        s0 = lst[0]
        key = f'{s0["cam"]}_{s0["tid"]}' if s0.get("tid") is not None else None
        fn = crop_map.get(key) if key else None
        qs.append(s0["q"])
        per_cam[s0["cam"]]["gp"] += 1
        if fn and os.path.exists(os.path.join(CROPS_DIR, fn)):
            n_gp_crop += 1
            per_cam[s0["cam"]]["crops"] += 1
            tiles.append((gp_label.get(str(gp_id), str(gp_id)), fn, s0["q"], s0.get("usable", False)))

    print(f"gperson 有脸(任意质量): {n_gp_any}")
    print(f"gperson 有可用人脸图(crop): {n_gp_crop}")
    print(f"per-gperson 最佳脸 q: 中位={np.median(qs):.3f} 均值={np.mean(qs):.3f}")
    n_usable = sum(1 for _, _, _, u in tiles if u)
    print(f"其中 usable(过门) 的 gperson: {n_usable}")
    for cam, d in sorted(per_cam.items()):
        print(f"  {cam}: gperson={d['gp']} crops={d['crops']}")

    # ---- 人脸墙 ----
    cols = args.cols
    cell = args.cell
    pad = 6
    header = 30
    rows = (len(tiles) + cols - 1) // cols
    W = cols * (cell + pad) + pad
    H = header + rows * (cell + pad + 20) + pad
    canvas = np.full((H, W, 3), 245, np.uint8)
    cv2.putText(canvas, f"per-gperson best face (n={len(tiles)})  green=usable  red=below-gate",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    for i, (label, fn, q, usable) in enumerate(tiles):
        img = cv2.imread(os.path.join(CROPS_DIR, fn))
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = cell / max(h, w)
        img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))))
        r, c = divmod(i, cols)
        y0 = header + r * (cell + pad + 20) + pad
        x0 = c * (cell + pad) + pad
        ih, iw = img.shape[:2]
        canvas[y0:y0 + ih, x0:x0 + iw] = img
        color = (0, 150, 0) if usable else (0, 0, 220)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), color, 1)
        txt = f"{label} q={q:.2f}"
        cv2.putText(canvas, txt, (x0, y0 + cell + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    out = os.path.join(FACE_DIR, args.out)
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wall -> {out} ({W}x{H})")


if __name__ == "__main__":
    main()
