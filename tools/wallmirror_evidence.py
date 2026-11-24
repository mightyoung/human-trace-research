#!/usr/bin/env python
"""S6 墙镜判定证据图：对每个候选对抽 3 帧，红框=候选倒影(R)、蓝框=源(S)，并画出自监督镜面竖线。

用法：python tools/wallmirror_evidence.py
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
from mirror_kinematics import load_tracks  # noqa: E402

MEDIA = os.path.join(ROOT, "webapp/backend/data/media")
OUT = os.path.join(ROOT, "output_crosscam/analysis/gil")
CASES = {
    "316": [21, 55, 105],      # 3 个代表性：短/中/长
    "323": [50, 172, 216],     # 全是 keep<-keep
}


def draw(cam, tids, out_path):
    d = json.load(open(os.path.join(OUT, "wallmirror_assign_v1.json"), encoding="utf-8"))
    rows = {x["tid"]: x for x in d[cam]["mirror_rows"]}
    axis = d[cam]["axis_main"]
    tr = load_tracks(os.path.join(ROOT, "output_crosscam", f"{cam}_detections.jsonl"))
    cap = cv2.VideoCapture(os.path.join(MEDIA, f"{cam}.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cells = []
    for tid in tids:
        row = rows[tid]
        src = row["source"]
        dr = dict(tr[tid]); ds = dict(tr[src])
        common = sorted(set(dr) & set(ds))
        picks = [common[len(common) // 4], common[len(common) // 2], common[3 * len(common) // 4]]
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            x1, y1, x2, y2 = [int(v) for v in dr[f]]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 3)          # 红 = 候选倒影
            sx1, sy1, sx2, sy2 = [int(v) for v in ds[f]]
            cv2.rectangle(im, (sx1, sy1), (sx2, sy2), (255, 128, 0), 3)    # 蓝 = 源
            cv2.putText(im, f"R#{tid}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(im, f"S#{src}", (sx1, max(24, sy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 128, 0), 3, cv2.LINE_AA)
            ax = int(axis * W)
            cv2.line(im, (ax, 0), (ax, H), (0, 255, 255), 3)               # 黄 = 自监督镜面
            cv2.putText(im, f"f={f} plane u={axis:.3f} ducorr={row['du_corr']:+.2f} dvcorr={row['dv_corr']:+.2f}",
                        (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
            im = cv2.resize(im, (im.shape[1] * 620 // im.shape[0], 620))
            cells.append(im)
    if not cells:
        print("no frames", cam); return
    canvas = np.hstack(cells)
    cv2.imwrite(out_path, canvas)
    print("out", out_path, canvas.shape)


if __name__ == "__main__":
    for cam, tids in CASES.items():
        draw(cam, tids, os.path.join(OUT, f"wallmirror_evidence_{cam}.png"))
