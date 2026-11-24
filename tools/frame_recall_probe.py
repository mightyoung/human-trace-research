#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按帧抽检：把两个检测器的框画到同一帧上，供人工点数「真实在场人数」。

用途：覆盖率的统计分母是"检测到的轨迹"，无法奖励召回。要判断换检测器是否
      少漏人，必须先有按帧的人数真值。本脚本只负责出图，人数由人眼看。

约定：
  - A 检测器的框：绿色细线（--label-a，默认 s-seg）
  - B 检测器独有（与 A 无 IoU>=0.5 匹配）：红色粗线（--label-b，默认 l-seg）
  - 两者都有的框：绿色（因为 A 有）——判断"这人被 A 检到了吗"看有没有绿框

用法：
  python tools/frame_recall_probe.py --seg 323_test_527 \
      --a <s 的 detections.jsonl> --b <l 的 detections.jsonl> \
      --n 4 --crop 0.15,0.05,0.75,0.55 --out-dir .../recall_probe/323
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_DIR = os.path.join(ROOT, "media-resource/1012/segments")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load(p):
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[int(d["frame"])] = d.get("boxes") or []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--a", required=True, help="检测器 A 的 detections.jsonl（生产默认 s-seg）")
    ap.add_argument("--b", required=True, help="检测器 B 的 detections.jsonl（候选 l-seg）")
    ap.add_argument("--label-a", default="s-seg")
    ap.add_argument("--label-b", default="l-seg")
    ap.add_argument("--n", type=int, default=4, help="抽几帧")
    ap.add_argument("--crop", default="", help="x0,y0,x1,y1 归一化裁切（用于远场放大），留空=全帧")
    ap.add_argument("--out-width", type=int, default=1560)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    A = load(a.a)
    B = load(a.b)
    frames = sorted(set(A) & set(B))
    if not frames:
        sys.exit("A/B 没有共同帧")
    idx = [round(k * (len(frames) - 1) / max(1, a.n - 1)) for k in range(a.n)]
    picked = [frames[i] for i in sorted(set(idx))]

    os.makedirs(a.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(SEG_DIR, f"{a.seg}.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop = None
    if a.crop:
        v = [float(x) for x in a.crop.split(",")]
        crop = (int(v[0] * W), int(v[1] * H), int(v[2] * W), int(v[3] * H))

    rows = []
    print(f"{'frame':>7} {'A':>3} {'B':>3} {'B_only':>6}  file")
    for fr in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            print(f"[warn] 读不到帧 {fr}")
            continue
        ba, bb = A[fr], B[fr]
        b_only = [y for y in bb if max([iou(x, y) for x in ba] + [0]) < 0.5]

        if crop:
            x0, y0, x1, y1 = crop
            canvas = img[y0:y1, x0:x1].copy()
            ox, oy = x0, y0
        else:
            canvas = img.copy()
            ox, oy = 0, 0
        for x in ba:
            cv2.rectangle(canvas, (int(x[0] - ox), int(x[1] - oy)),
                          (int(x[2] - ox), int(x[3] - oy)), (0, 255, 0), 2)
        for y in b_only:
            cv2.rectangle(canvas, (int(y[0] - ox), int(y[1] - oy)),
                          (int(y[2] - ox), int(y[3] - oy)), (0, 0, 255), 3)

        s = a.out_width / canvas.shape[1]
        canvas = cv2.resize(canvas, (a.out_width, int(canvas.shape[0] * s)),
                            interpolation=cv2.INTER_CUBIC)
        hdr = np.zeros((44, canvas.shape[1], 3), np.uint8)
        txt = (f"{a.seg} f={fr}  green={a.label_a}:{len(ba)}  "
               f"red={a.label_b}-only:{len(b_only)}")
        cv2.putText(hdr, txt, (10, 31), 0, 0.8, (255, 255, 255), 2)
        canvas = np.vstack([hdr, canvas])

        name = f"frame_f{fr:06d}.jpg"
        cv2.imwrite(os.path.join(a.out_dir, name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({"seg": a.seg, "frame": fr, "n_a": len(ba), "n_b": len(bb),
                     "n_b_only": len(b_only), "file": name,
                     "true_count": "", "missed_by_a": "", "missed_by_b": "", "note": ""})
        print(f"{fr:>7} {len(ba):>3} {len(bb):>3} {len(b_only):>6}  {name}")
    cap.release()

    csv_path = os.path.join(a.out_dir, "frames.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {len(rows)} 帧 -> {a.out_dir}")
    print(f"[out] 待填 CSV: {csv_path}")


if __name__ == "__main__":
    main()
