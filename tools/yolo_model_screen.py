#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO26 变体筛选：同一段、同样帧数，比检测密度/重复框/速度。

只跑检测，不编码不评估——先用便宜的指标决定哪个模型值得上完整评估。
用法：
  python tools/yolo_model_screen.py --tag m-seg --model .../yolo26m-seg.pt \
      --seg 305_test_544 --frames 1000 --iou 0.7
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seg", default="305_test_544")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "output_crosscam_local/_screen"),
                    help="会再拼一层 tag，避免并行筛选互相覆盖")
    a = ap.parse_args()

    a.out_root = os.path.join(a.out_root, a.tag)
    t0 = time.time()
    cmd = [PY, os.path.join(ROOT, "tools/yolo_track_det.py"), "--seg", a.seg,
           "--model", a.model, "--stride", str(a.stride), "--max-frames", str(a.frames),
           "--iou", str(a.iou), "--imgsz", str(a.imgsz), "--out-dir", a.out_root]
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"[{a.tag}] 失败 rc={r.returncode}")

    p = os.path.join(a.out_root, a.seg, f"{a.seg}_detections.jsonl")
    frames = boxes = dup_frames = dup_pairs = 0
    obs = {}
    areas = []
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        bs = d.get("boxes") or []
        frames += 1
        boxes += len(bs)
        for b in bs:
            w, h = b[2] - b[0], b[3] - b[1]
            areas.append(w * h)
        flag = 0
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                if iou(bs[i], bs[j]) >= 0.5:
                    dup_pairs += 1
                    flag = 1
        dup_frames += flag
        for t in d.get("track_ids") or []:
            obs[t] = obs.get(t, 0) + 1

    print(f"[{a.tag}] imgsz={a.imgsz} iou={a.iou} frames={frames} wall={wall:.0f}s "
          f"({frames/wall:.2f} f/s) boxes/frame={boxes/max(1,frames):.2f} "
          f"tracks={len(obs)} medObs={st.median(sorted(obs.values())) if obs else 0:.0f} "
          f"dupFrames={dup_frames/max(1,frames)*100:.1f}% dupPairs/frame={dup_pairs/max(1,frames):.3f} "
          f"medArea={st.median(areas) if areas else 0:.0f}px2")


if __name__ == "__main__":
    main()
