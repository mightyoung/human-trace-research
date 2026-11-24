#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测级去重消融：同帧内按 conf 从高到低保留，IoU>=thr 的低分框丢弃。

背景：YOLO26s-seg 在 iou=0.7 的 NMS 下会保留同一个人的两个框（头/全身各一），
      形成"一个人两条轨迹"，在共现误合并指标里被算成两个人。本脚本只做消融验证，
      不改变 track id（框被删，id 仍在后续帧出现）。

用法：
  python tools/dedup_detections.py --in-dir output_crosscam_local/1012_yolo8 \
      --out-dir output_crosscam_local/1012_yolo8_dd --iou 0.5
"""
import argparse
import json
import os
from pathlib import Path

SEGS = ["305_train_529", "318_train_525", "320_train_514", "323_train_510",
        "305_test_544", "318_test_534", "320_test_518", "323_test_527"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="output_crosscam_local/1012_yolo8")
    ap.add_argument("--out-dir", default="output_crosscam_local/1012_yolo8_dd")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--segs", default=",".join(SEGS))
    a = ap.parse_args()

    tot_in = tot_out = 0
    for seg in a.segs.split(","):
        src = Path(a.in_dir) / seg / f"{seg}_detections.jsonl"
        if not src.exists():
            print(f"[miss] {src}")
            continue
        dst_dir = Path(a.out_dir) / seg
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{seg}_detections.jsonl"
        n_in = n_out = 0
        with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
            for line in fi:
                if not line.strip():
                    continue
                d = json.loads(line)
                boxes = d.get("boxes") or []
                scores = d.get("scores") or [1.0] * len(boxes)
                ids = d.get("track_ids") or []
                order = sorted(range(len(boxes)), key=lambda i: -scores[i])
                keep = []
                for i in order:
                    if all(iou(boxes[i], boxes[j]) < a.iou for j in keep):
                        keep.append(i)
                keep.sort()
                n_in += len(boxes)
                n_out += len(keep)
                rec = dict(d)
                rec["boxes"] = [boxes[i] for i in keep]
                rec["track_ids"] = [ids[i] for i in keep]
                if "scores" in d:
                    rec["scores"] = [scores[i] for i in keep]
                fo.write(json.dumps(rec) + "\n")
        tot_in += n_in
        tot_out += n_out
        print(f"[{seg}] {n_in} -> {n_out} ({100*(n_in-n_out)/max(1,n_in):.1f}% 去掉)")
    print(f"[total] {tot_in} -> {tot_out}  去掉 {tot_in-tot_out} "
          f"({100*(tot_in-tot_out)/max(1,tot_in):.1f}%)")


if __name__ == "__main__":
    main()
