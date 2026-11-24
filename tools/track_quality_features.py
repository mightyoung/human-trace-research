#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹质量特征：只在「框几何 + 轨迹长度 + 检测分」上算，不重跑检测。

背景：人工复核发现存在「框里只有小臂/手」的残框轨迹，任何特征空间都无法识别，却会被计为一次接受。
本脚本把这类轨迹的可用几何信号量化出来，供质量门使用。

输出：selector_osnet_1012/track_quality.csv
字段：seg,tid,cam,n_obs,h_px_p50,h_frac_p50,aspect_p50,aspect_p10,bottom_frac,
      score_p50,area_cv,frame_w,frame_h
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402

ROOT = S.ROOT
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
VID = os.path.join(ROOT, "media-resource/1012/segments")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/track_quality.csv")


def seg_size(seg, cache):
    if seg not in cache:
        cap = cv2.VideoCapture(os.path.join(VID, f"{seg}.mp4"))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        cache[seg] = (w, h)
    return cache[seg]


def main():
    cache = {}
    rows = []
    for seg in S.TEST_SEGS:
        W, H = seg_size(seg, cache)
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        acc = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                for tid, b, sc in zip(o["track_ids"], o["boxes"], o.get("scores", [1.0] * len(o["boxes"]))):
                    x1, y1, x2, y2 = b
                    h, w = y2 - y1, x2 - x1
                    if h <= 0 or w <= 0:
                        continue
                    acc.setdefault(tid, []).append((h, w / h, y2 / H, float(sc), w * h))
        for tid, v in acc.items():
            h = np.array([t[0] for t in v])
            ar = np.array([t[1] for t in v])
            bo = np.array([t[2] for t in v])
            sc = np.array([t[3] for t in v])
            area = np.array([t[4] for t in v])
            rows.append([seg, tid, seg.split("_")[0], len(v),
                         round(float(np.median(h)), 1),
                         round(float(np.median(h) / H), 4),
                         round(float(np.median(ar)), 3),
                         round(float(np.percentile(ar, 10)), 3),
                         round(float((bo >= 0.98).mean()), 3),
                         round(float(np.median(sc)), 3),
                         round(float(area.std() / max(area.mean(), 1e-6)), 3),
                         W, H])
    rows.sort(key=lambda r: (r[0], r[1]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["seg", "tid", "cam", "n_obs", "h_px_p50", "h_frac_p50", "aspect_p50",
                    "aspect_p10", "bottom_frac", "score_p50", "area_cv", "frame_w", "frame_h"])
        w.writerows(rows)
    print(f"[out] {OUT}  {len(rows)} 条轨迹")

    # 摘要
    for cam in ("305", "318", "320", "323"):
        sub = [r for r in rows if r[2] == cam]
        if not sub:
            continue
        print(f"{cam}: n={len(sub):3d} h_px_p50={np.median([r[4] for r in sub]):6.1f} "
              f"h_frac_p50={np.median([r[5] for r in sub]):.3f} "
              f"aspect_p50={np.median([r[6] for r in sub]):.2f}")

    # 已复核的 7 条
    checked = {("323_test_527", 6010), ("323_test_527", 1112), ("318_test_534", 882),
               ("305_test_544", 892), ("320_test_518", 828), ("320_test_518", 1225),
               ("320_test_518", 1050)}
    print("\n已复核 7 条：")
    for r in rows:
        if (r[0], r[1]) in checked:
            print(f"  {r[0]}#{r[1]:<5d} n={r[3]:3d} h_px={r[4]:6.1f} h_frac={r[5]:.3f} "
                  f"ar={r[6]:.2f} ar_p10={r[7]:.2f} bot={r[8]:.2f} sc={r[9]:.2f}")


if __name__ == "__main__":
    main()
