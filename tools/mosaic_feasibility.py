# -*- coding: utf-8 -*-
"""mosaic_feasibility.py — 六路画面拼接送模型的可行性实测。

三个问题依次回答：
  Q1 算力：YOLO26s-seg 在 CPU 上，输入尺寸 -> 单帧耗时/内存的曲线（640..N）
  Q2 像素：拼成一张大图后，为了塞进模型输入尺寸，每路被压缩多少 -> 小目标还检不检得到
  Q3 等效性：拼图送一次 vs 逐路各送一次，同一帧的 person 检出是否等价

用法：
  python tools/mosaic_feasibility.py --mode latency --imgsz 640 1280 1920 2560
  python tools/mosaic_feasibility.py --mode compare --t 300
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
    ("316", "01_行政走廊通道_316", "316_085500-090500.mp4", 3945),
    ("305", "02_办公区1_305", "305_085500-090500.mp4", 3800),
    ("318", "03_办公区2_318", "318_085500-090500.mp4", 3330),
    ("320", "04_办公区球机_320", "320_085500-090500.mp4", 620),
    ("323", "05_茶水间_323", "323_085500-090500.mp4", 1685),
    ("343", "06_电梯间_343", "343_085500-090500.mp4", 2975),
]
MODEL = os.path.join(ROOT, "models/Ultralytics--YOLO26/snapshots/master/yolo26s-seg.pt")
OUT_DIR = os.path.join(ROOT, "output_crosscam_local/mosaic_feasibility")

CELL_W, CELL_H = 2560, 1440   # 网格单元 = 最高分辨率机位，1920x1080 源不缩放居中放置
GRID_COLS, GRID_ROWS = 3, 2


def frame_at(cam, t_sec):
    """取某路相对切片起点 t 秒处的帧。"""
    for c, rel, fn, _ in CAMS:
        if c == cam:
            path = os.path.join(VID, rel, fn)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
            ok, f = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"read fail {path}")
            return f
    raise KeyError(cam)


def build_grid(frames):
    """3x2 网格，单元 2560x1440；1920x1080 源居中不缩放，其余填 0。返回大图与每路在画布上的偏移。"""
    grid = np.zeros((CELL_H * GRID_ROWS, CELL_W * GRID_COLS, 3), np.uint8)
    meta = {}
    for i, (cam, _r, _f, _s) in enumerate(CAMS):
        r, col = divmod(i, GRID_COLS)
        img = frames[cam]
        h, w = img.shape[:2]
        ox = col * CELL_W + (CELL_W - w) // 2
        oy = r * CELL_H + (CELL_H - h) // 2
        grid[oy:oy + h, ox:ox + w] = img
        meta[cam] = dict(ox=ox, oy=oy, w=w, h=h)
    return grid, meta


def get_model():
    from ultralytics import YOLO
    return YOLO(MODEL)


def predict(model, img, imgsz, conf=0.10):
    t0 = time.time()
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=[0], device="cpu",
                      verbose=False, max_det=300)[0]
    dt = time.time() - t0
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0, 4))
    scores = r.boxes.conf.cpu().numpy() if r.boxes is not None and len(r.boxes) else np.zeros((0,))
    return boxes, scores, dt


def mode_latency(args):
    model = get_model()
    img = frame_at("323", 300)
    print(f"[src] 323 {img.shape[1]}x{img.shape[0]}", flush=True)
    rows = []
    for s in args.imgsz:
        b, sc, dt = predict(model, img, s)
        n10 = int((sc >= 0.10).sum())
        n25 = int((sc >= 0.25).sum())
        row = dict(imgsz=s, sec=round(dt, 3), n10=n10, n25=n25,
                   megapix=round(img.shape[0] * img.shape[1] / 1e6, 2))
        rows.append(row)
        print(json.dumps(row), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT_DIR, "latency.json"), "w"), indent=1)


def mode_compare(args):
    model = get_model()
    frames = {c: frame_at(c, args.t) for c, _r, _f, _s in CAMS}
    for c, _r, _f, _s in CAMS:
        h, w = frames[c].shape[:2]
        print(f"[frame] {c} {w}x{h}", flush=True)

    res = {"t": args.t, "per_cam": {}, "mosaic": {}}

    # A: 逐路 imgsz=1280（生产现状）
    tA = time.time()
    for c, _r, _f, _s in CAMS:
        b, sc, dt = predict(model, frames[c], args.per_cam_sz)
        res["per_cam"][c] = dict(n10=int((sc >= .10).sum()), n25=int((sc >= .25).sum()),
                                 sec=round(dt, 3), src=f"{frames[c].shape[1]}x{frames[c].shape[0]}")
    res["per_cam"]["_total_sec"] = round(time.time() - tA, 3)
    print("[A per-cam]", json.dumps(res["per_cam"], ensure_ascii=False), flush=True)

    grid, meta = build_grid(frames)
    gh, gw = grid.shape[:2]
    print(f"[grid] {gw}x{gh}  ({gw*gh/1e6:.2f} MP, 原生合计 {sum(f.shape[0]*f.shape[1] for f in frames.values())/1e6:.2f} MP)", flush=True)

    # C: 逐路但压到"拼图 imgsz=1280 时每路的有效分辨率"（426x240），量化纯像素损失
    small = {}
    tC = time.time()
    for c, _r, _f, _s in CAMS:
        h, w = frames[c].shape[:2]
        k = 1280.0 / gw
        tw, th = int(round(w * k)), int(round(h * k))
        im = cv2.resize(frames[c], (tw, th), interpolation=cv2.INTER_AREA)
        b, sc, dt = predict(model, im, 640)
        small[c] = dict(n10=int((sc >= .10).sum()), n25=int((sc >= .25).sum()), size=f"{tw}x{th}")
    res["per_cam_downscaled"] = small
    res["per_cam_downscaled"]["_total_sec"] = round(time.time() - tC, 3)
    print("[C per-cam downscaled]", json.dumps(small, ensure_ascii=False), flush=True)

    # B: 拼图整张送模型。ultralytics 返回的框已在 grid 坐标系
    for sz in args.mosaic_sz:
        b, sc, dt = predict(model, grid, sz)
        per = {c: 0 for c in meta}
        per25 = {c: 0 for c in meta}
        for (x1, y1, x2, y2), s in zip(b, sc):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for c, m in meta.items():
                if m["ox"] <= cx < m["ox"] + m["w"] and m["oy"] <= cy < m["oy"] + m["h"]:
                    per[c] += 1
                    if s >= 0.25:
                        per25[c] += 1
                    break
        res["mosaic"][f"imgsz{sz}"] = dict(
            sec=round(dt, 3), n10=int((sc >= .10).sum()), n25=int((sc >= .25).sum()),
            per_cam=per, per_cam25=per25,
            eff_w=int(round(gw * (sz / float(max(gw, gh))) / GRID_COLS)),
            eff_h=int(round(gh * (sz / float(max(gw, gh))) / GRID_ROWS)),
        )
        print(f"[B mosaic imgsz={sz}]", json.dumps(res["mosaic"][f"imgsz{sz}"], ensure_ascii=False), flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, f"compare_t{args.t}.json"), "w"),
              ensure_ascii=False, indent=1)
    cv2.imwrite(os.path.join(OUT_DIR, f"grid_t{args.t}.jpg"),
                cv2.resize(grid, (gw // 4, gh // 4), interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="compare", choices=["latency", "compare"])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280, 1920, 2560])
    ap.add_argument("--mosaic-sz", type=int, nargs="+", default=[1280, 2560])
    ap.add_argument("--per-cam-sz", type=int, default=1280)
    ap.add_argument("--t", type=float, default=300.0)
    a = ap.parse_args()
    if a.mode == "latency":
        mode_latency(a)
    else:
        mode_compare(a)


if __name__ == "__main__":
    main()
