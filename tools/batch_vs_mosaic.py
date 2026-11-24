# -*- coding: utf-8 -*-
"""batch_vs_mosaic.py — "多路一次输入"的三种形式对比：逐路串行 / batch 并行 / 网格拼图。

目的：验证"把 6 路合成一次输入"到底该用哪种形式。
  - serial : 6 次单图 forward（生产现状）
  - batch  : 1 次 6 图 forward（像素零损失，真正省调用开销）
  - mosaic : 1 次大图 forward（像素被重采样）

用法：python tools/batch_vs_mosaic.py --imgsz 1280 --reps 3
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

VID = os.path.join(ROOT, "media-resource/multi-cemra/synced-10min")
CAMS = [
    ("316", "01_行政走廊通道_316", "316_085500-090500.mp4"),
    ("305", "02_办公区1_305", "305_085500-090500.mp4"),
    ("318", "03_办公区2_318", "318_085500-090500.mp4"),
    ("320", "04_办公区球机_320", "320_085500-090500.mp4"),
    ("323", "05_茶水间_323", "323_085500-090500.mp4"),
    ("343", "06_电梯间_343", "343_085500-090500.mp4"),
]
MODEL = os.path.join(ROOT, "models/Ultralytics--YOLO26/snapshots/master/yolo26s-seg.pt")
OUT = os.path.join(ROOT, "output_crosscam_local/mosaic_feasibility")
CELL_W, CELL_H = 2560, 1440


def frames_at(t):
    out = {}
    for c, rel, fn in CAMS:
        cap = cv2.VideoCapture(os.path.join(VID, rel, fn))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        cap.release()
        if ok:
            out[c] = f
    return out


def grid_of(frames):
    g = np.zeros((CELL_H * 2, CELL_W * 3, 3), np.uint8)
    for i, (c, _r, _f) in enumerate(CAMS):
        r, col = divmod(i, 3)
        img = frames[c]
        h, w = img.shape[:2]
        ox = col * CELL_W + (CELL_W - w) // 2
        oy = r * CELL_H + (CELL_H - h) // 2
        g[oy:oy + h, ox:ox + w] = img
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--t", type=float, default=300.0)
    a = ap.parse_args()

    from ultralytics import YOLO
    m = YOLO(MODEL)
    fr = frames_at(a.t)
    imgs = [fr[c] for c, _r, _f in CAMS]
    g = grid_of(fr)

    # warmup
    m.predict(imgs[0], imgsz=a.imgsz, conf=0.10, classes=[0], device="cpu", verbose=False)

    res = {}
    # serial
    ts = []
    for _ in range(a.reps):
        t0 = time.time()
        n = 0
        for im in imgs:
            r = m.predict(im, imgsz=a.imgsz, conf=0.10, classes=[0], device="cpu", verbose=False)[0]
            n += 0 if r.boxes is None else len(r.boxes)
        ts.append(time.time() - t0)
    res["serial"] = dict(sec=round(min(ts), 3), n=n, note="6 次单图 forward")

    # batch
    tb = []
    for _ in range(a.reps):
        t0 = time.time()
        rs = m.predict(imgs, imgsz=a.imgsz, conf=0.10, classes=[0], device="cpu",
                       verbose=False, batch=len(imgs))
        n = sum(0 if r.boxes is None else len(r.boxes) for r in rs)
        tb.append(time.time() - t0)
    res["batch"] = dict(sec=round(min(tb), 3), n=n, note=f"1 次 {len(imgs)} 图 forward")

    # mosaic（同样本预算下，imgsz = 6 路等面积）
    tm = []
    for _ in range(max(1, a.reps - 1)):
        t0 = time.time()
        r = m.predict(g, imgsz=a.imgsz * 3, conf=0.10, classes=[0], device="cpu", verbose=False)[0]
        n = 0 if r.boxes is None else len(r.boxes)
        tm.append(time.time() - t0)
    res["mosaic"] = dict(sec=round(min(tm), 3), n=n, note="1 次大图 forward，每路等效分辨率不变")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "batch_vs_mosaic.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
