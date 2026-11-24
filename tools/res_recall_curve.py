# -*- coding: utf-8 -*-
"""res_recall_curve.py — 单路分辨率降到多少会开始丢人（回答"拼图后每路剩 427px 还能不能用"）。

做法：以「原分辨率 + 同尺寸推理」为参照，把同一帧缩到不同宽度再推理，
按 IoU 0.5 匹配，统计分层的召回率（按参照框高度分桶）。

用法：
  python tools/res_recall_curve.py --ts 60 180 300 420 540 --widths 2560 1920 1280 854 640 427
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

CFG = {"conf": 0.10, "iou": 0.5, "match_iou": 0.5, "buckets": [(0, 32), (32, 64), (64, 128), (128, 10 ** 6)]}


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(1e-9, aa[:, None] + bb[None, :] - inter)


def frame_at(rel, fn, t):
    cap = cv2.VideoCapture(os.path.join(VID, rel, fn))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=float, nargs="+", default=[60, 180, 300, 420, 540])
    ap.add_argument("--widths", type=int, nargs="+", default=[2560, 1920, 1280, 854, 640, 427])
    ap.add_argument("--cams", nargs="+", default=None)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(MODEL)

    rows = []
    for cam, rel, fn in CAMS:
        if a.cams and cam not in a.cams:
            continue
        for t in a.ts:
            f = frame_at(rel, fn, t)
            if f is None:
                continue
            H, W = f.shape[:2]
            ref_b, ref_s, _ = infer(model, f, max(H, W))
            ref_h = ref_b[:, 3] - ref_b[:, 1] if len(ref_b) else np.zeros(0)
            if len(ref_b) == 0:
                continue
            rec = {"cam": cam, "t": t, "src": f"{W}x{H}", "ref_n": int(len(ref_b)),
                   "ref_h_med": float(np.median(ref_h)), "levels": []}
            for w in a.widths:
                k = w / float(W)
                im = f if abs(k - 1) < 1e-6 else cv2.resize(f, (w, int(round(H * k))), interpolation=cv2.INTER_AREA)
                b, s, dt = infer(model, im, 1280)
                # 小图框 -> 原图坐标
                bb = b / max(k, 1e-9)
                m = iou_mat(ref_b, bb)
                hit = (m >= CFG["match_iou"]).any(axis=1) if m.size else np.zeros(len(ref_b), bool)
                per_bucket = {}
                for lo, hi in CFG["buckets"]:
                    sel = (ref_h >= lo) & (ref_h < hi)
                    if sel.sum() == 0:
                        continue
                    per_bucket[f"{lo}-{hi if hi < 10**6 else '+'}"] = dict(
                        n=int(sel.sum()), recall=round(float(hit[sel].mean()), 3))
                rec["levels"].append(dict(w=w, sec=round(dt, 3), n10=int(len(b)),
                                          recall=round(float(hit.mean()), 3) if len(ref_b) else None,
                                          buckets=per_bucket))
                print(f"[{cam} t={t}] w={w:5d} n={len(b):3d} recall={hit.mean():.3f}", flush=True)
            rows.append(rec)

    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "res_recall.json"), "w"), ensure_ascii=False, indent=1)
    # 汇总：按宽度聚合分层召回
    print("\n=== 聚合（按框高分层，跨机位/时刻）===")
    agg = {}
    for w in a.widths:
        bok = {}
        for r in rows:
            for lv in r["levels"]:
                if lv["w"] != w:
                    continue
                for k, v in lv["buckets"].items():
                    d = bok.setdefault(k, [0, 0])
                    d[0] += int(round(v["recall"] * v["n"]))
                    d[1] += v["n"]
        agg[w] = {k: (round(v[0] / v[1], 3) if v[1] else None, v[1]) for k, v in sorted(bok.items())}
        print(w, json.dumps(agg[w]))
    json.dump(agg, open(os.path.join(OUT, "res_recall_agg.json"), "w"), indent=1)


def infer(model, img, imgsz):
    t0 = time.time()
    r = model.predict(img, imgsz=imgsz, conf=CFG["conf"], iou=CFG["iou"], classes=[0],
                      device="cpu", verbose=False, max_det=300)[0]
    dt = time.time() - t0
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), dt
    return r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy(), dt


if __name__ == "__main__":
    main()
