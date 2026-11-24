#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""随机抽样轨迹做目检：判断是否存在幽灵轨迹（框里没人）。

替代 track_gate 对 YOLO 的验收：YOLO 轨迹本身来自 person 类实例分割，
掩膜命中率恒近 1，原门控失去信息量；改为直接抽轨迹看图。

用法：
  python tools/sample_tracks.py --det-dir output_crosscam_local/1012_yolo8_dd \
      --segs 305_test_544,318_test_534,320_test_518,323_test_527 --per-seg 5 --min-obs 10 \
      --out output_crosscam_local/selector_osnet_1012/track_sample.jpg
"""
import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np

CELL = 260
PAD = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--vid-dir", default="media-resource/1012/segments")
    ap.add_argument("--segs", required=True)
    ap.add_argument("--per-seg", type=int, default=5)
    ap.add_argument("--min-obs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    random.seed(a.seed)

    tiles, caps = [], {}
    for seg in a.segs.split(","):
        p = os.path.join(a.det_dir, seg, f"{seg}_detections.jsonl")
        obs = defaultdict(list)          # tid -> [(frame, box)]
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            for t, b in zip(d.get("track_ids") or [], d.get("boxes") or []):
                obs[t].append((d["frame"], b))
        cand = [t for t, v in obs.items() if len(v) >= a.min_obs]
        pick = random.sample(cand, min(a.per_seg, len(cand)))
        cap = cv2.VideoCapture(os.path.join(a.vid_dir, f"{seg}.mp4"))
        caps[seg] = cap
        for t in pick:
            f, b = obs[t][len(obs[t]) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            x1, y1 = int(max(0, b[0] - PAD)), int(max(0, b[1] - PAD))
            x2, y2 = int(min(w, b[2] + PAD)), int(min(h, b[3] + PAD))
            crop = img[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            cv2.rectangle(crop, (int(b[0]) - x1, int(b[1]) - y1),
                          (int(b[2]) - x1, int(b[3]) - y1), (0, 255, 0), 2)
            crop = cv2.resize(crop, (CELL, CELL))
            cv2.putText(crop, f"{seg[:3]}#{t} n={len(obs[t])}", (5, CELL - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)
            tiles.append(crop)
    for c in caps.values():
        c.release()

    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * CELL, cols * CELL, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = t
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f"[out] {a.out}  {len(tiles)} 条")


if __name__ == "__main__":
    main()
