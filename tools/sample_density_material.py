#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""剩余素材人口密度抽样：对未使用卷各抽 3 个时间点，跑 YOLO26s-seg 数 person 框。

目的：验证「素材存在 ≠ 有人」。只抽帧检测，不建轨迹、不编码，分钟级完成。

用法：
  python tools/sample_density_material.py [--per-vol 3] [--json out.json]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "media-resource", "1012")
MODEL = os.path.join(ROOT, "models", "Ultralytics--YOLO26", "snapshots", "master", "yolo26s-seg.pt")

# 只看未使用的卷（已用卷 len 由 segments_meta 决定，这里显式列出）
TARGETS = [
    ("320", "0100000251400000000.mp4", 64.9),   # 已用前60min，抽尾部
    ("320", "0100000248000000000.mp4", 73.1),
    ("320", "0100000254000000000.mp4", 64.3),
    ("320", "0100000254300000000.mp4", 73.4),
    ("320", "0100000251900000000.mp4", 39.9),   # 已用前15min
    ("320", "0100000254200000000.mp4", 11.2),
    ("320", "0100000251800000000.mp4", 64.6),   # 已用前15min
    ("320", "0100000254900000000.mp4", 12.0),
    ("320", "0100000256000000000.mp4", 13.9),
    ("318", "0100000252500000000.mp4", 299.9),  # 已用前60min + 240-255min
    ("318", "0100000253400000000.mp4", 233.8),  # 已用前15min
    ("323", "0100000253600000000.mp4", 65.6),
    ("323", "0100000251300000000.mp4", 65.7),
    ("323", "0100000253800000000.mp4", 65.7),   # 已用 24-39min
    ("323", "0100000252700000000.mp4", 65.7),   # 已用 48-63min
    ("305", "0100000252900000000.mp4", 116.2),  # 已用前60min
    ("305", "0100000252100000000(1).mp4", 121.7),
    ("305", "0100000251700000000.mp4", 123.8),  # 已用 66-81min
    ("305", "0100000254400000000.mp4", 117.2),  # 已用 62-77min
    ("305", "0100000255900000000.mp4", 118.0),
]


def grab(path, t_sec):
    """seek 后连续读 30 帧（跳过 GOP 花屏），返回最后一帧。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    fr = None
    for _ in range(30):
        ok, f = cap.read()
        if not ok:
            break
        fr = f
    cap.release()
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-vol", type=int, default=3)
    ap.add_argument("--json", default=os.path.join(ROOT, "docs", "material-density-2026-10-20.json"))
    args = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(MODEL)

    rows = []
    print(f"{'点位':<5}{'卷':<26}{'抽点(min)':<24}{'人数':<16}{'最大框高px':<10}")
    print("-" * 92)
    for cam, vol, dur_min in TARGETS:
        p = os.path.join(SRC, vol)
        if not os.path.exists(p):
            print(f"{cam:<5}{vol:<26} **缺失**")
            continue
        fracs = [0.1, 0.5, 0.9] if args.per_vol == 3 else [i / (args.per_vol + 1) for i in range(1, args.per_vol + 1)]
        cnts, hs, ts = [], [], []
        for fr_ in fracs:
            t = dur_min * 60 * fr_
            img = grab(p, t)
            if img is None:
                cnts.append(-1); hs.append(0); ts.append(round(t / 60, 1)); continue
            r = m.predict(img, imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]
            n = 0 if r.boxes is None else len(r.boxes)
            hmax = 0
            if n:
                xy = r.boxes.xyxy.cpu().numpy()
                hmax = int(np.max(xy[:, 3] - xy[:, 1]))
            cnts.append(n); hs.append(hmax); ts.append(round(t / 60, 1))
        rows.append({"cam": cam, "vol": vol, "dur_min": dur_min, "t_min": ts,
                     "n_person": cnts, "max_box_h": hs})
        cstr = "/".join(str(c) for c in cnts)
        hstr = "/".join(str(h) for h in hs)
        print(f"{cam:<5}{vol:<26}{str(ts):<24}{cstr:<16}{hstr:<10}")

    valid = [r for r in rows if all(c >= 0 for c in r["n_person"])]
    tot = sum(sum(r["n_person"]) for r in valid)
    nfr = sum(len(r["n_person"]) for r in valid)
    print("-" * 92)
    print(f"有效样本 {len(valid)} 卷 / {nfr} 帧；平均 {tot/max(nfr,1):.2f} 人/帧；"
          f"零人帧 {sum(1 for r in valid for c in r['n_person'] if c==0)}/{nfr}")
    json.dump({"rows": rows, "mean_person": round(tot / max(nfr, 1), 3)}, 
              open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[DONE] -> {args.json}")


if __name__ == "__main__":
    sys.exit(main())
