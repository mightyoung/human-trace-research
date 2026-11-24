#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""掩膜效果演示：DINO 框 / YOLO 框 / YOLO 掩膜抠图 三栏对照。

用途：C1（换 YOLO-seg）决策的直观证据——几何框无法去掉的椅子/桌面像素，
实例掩膜可以。选 305（俯视工位，椅子混入最严重）与 323（走廊）各一帧。

用法: python tools/yolo_mask_demo.py --seg 305_test_544 --frame 9865
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEG_DIR = ROOT / "media-resource" / "1012" / "segments"
DINO_DIR = ROOT / "output_crosscam_local" / "1012_ab"
OUT_DIR = ROOT / "output_crosscam_local" / "yolo_vs_dino"
MODEL = ROOT / "models" / "Ultralytics--YOLO26" / "snapshots" / "master" / "yolo26s-seg.pt"

GREEN = (60, 200, 60)
RED = (60, 60, 230)
WHITE = (240, 240, 240)


def read_frame(video, target):
    cap = cv2.VideoCapture(str(video))
    fi = 0
    img = None
    while cap.grab():
        if fi == target:
            ok, img = cap.retrieve()
            break
        fi += 1
    cap.release()
    return img


def load_dino_boxes(seg, frame):
    p = DINO_DIR / f"{seg}_a.jsonl"
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if int(r["frame"]) == frame:
            return np.asarray(r["boxes"], np.float32).reshape(-1, 4)
    return np.zeros((0, 4), np.float32)


def panel(img, boxes, color, title, thick=3):
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
    head = np.full((40, out.shape[1], 3), 28, np.uint8)
    cv2.putText(head, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    return np.vstack([head, out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="305_test_544")
    ap.add_argument("--frame", type=int, default=9865)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--width", type=int, default=760, help="每栏像素宽")
    ap.add_argument("--layout", choices=["two", "three"], default="three")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    img = read_frame(SEG_DIR / f"{args.seg}.mp4", args.frame)
    if img is None:
        raise SystemExit("frame not found")
    H, W = img.shape[:2]

    res = model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=[0],
                        verbose=False, device="cpu", retina_masks=True)[0]
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
    masks = res.masks.data.cpu().numpy().astype(np.float32) if res.masks is not None else None
    dbox = load_dino_boxes(args.seg, args.frame)

    p1 = panel(img, dbox, GREEN, f"GroundingDINO  {len(dbox)} boxes")
    p2 = panel(img, boxes, RED, f"YOLO26s-seg  {len(boxes)} boxes (conf>={args.conf})")

    # 第三栏：掩膜抠图（只保留人体像素，其余压暗）
    cut = (img.astype(np.float32) * 0.18).astype(np.uint8)
    if masks is not None:
        mmax = masks.max(axis=0)
        mmax = cv2.resize(mmax, (W, H), interpolation=cv2.INTER_LINEAR)
        m3 = np.repeat((mmax > 0.5)[:, :, None], 3, axis=2)
        cut = np.where(m3, img, cut)
        frac = float((mmax > 0.5).mean())
    else:
        frac = 0.0
    p3 = panel(cut, boxes, (40, 220, 250), f"mask cutout  person px={frac*100:.1f}%", thick=2)

    if args.layout == "two":
        row = np.hstack([p1, p3])
        ncol = 2
    else:
        row = np.hstack([p1, p2, p3])
        ncol = 3
    h, w = row.shape[:2]
    row = cv2.resize(row, (args.width * ncol, int(h * args.width * ncol / w)),
                     interpolation=cv2.INTER_AREA)
    tag = "" if args.layout == "three" else "_2up"
    out = OUT_DIR / f"mask_demo_{args.seg}_f{args.frame}{tag}.jpg"
    cv2.imwrite(str(out), row, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    print(f"[out] {out}")
    print(f"[stat] dino={len(dbox)} yolo={len(boxes)} person_px={frac*100:.2f}%")


if __name__ == "__main__":
    main()
