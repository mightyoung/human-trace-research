#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""掩膜诊断：未匹配到 YOLO 实例的 DINO 框到底是谁的错。

输出两张图：
  {seg}_diag_miss.jpg  —— 整帧，蓝=DINO 框，绿=YOLO(conf 0.05) 框，用于判断
                          未匹配框是"DINO 误检（椅子上没人）"还是"YOLO 漏检"。
  {seg}_diag_mask.jpg  —— 已匹配裁剪的掩膜边界叠加（2x），用于判断掩膜是否溢出。

用法：python tools/mask_diag.py --seg 320_train_514 --n 24
"""
import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encode_1012_osnet_seq as E  # noqa: E402
from yolo_vs_dino_frames import DEFAULT_MODEL, greedy_match, iou_matrix  # noqa: E402

ROOT = E.ROOT
OUT = ROOT / "output_crosscam_local/1012_osnet_crops_mask"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(DEFAULT_MODEL))

    tracks = E.load_ab(E.AB_DIR / f"{a.seg}_a.jsonl")
    by_frame = defaultdict(list)
    for tid, tf in tracks.items():
        if len(tf) < 2:
            continue
        for k in np.linspace(0, len(tf) - 1, min(8, len(tf))).astype(int):
            by_frame[int(tf[k][0])].append(tf[k][1])
    frames = sorted(by_frame)
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, a.n).astype(int)]

    cap = cv2.VideoCapture(str(E.VID_DIR / f"{a.seg}.mp4"))
    cur, miss_tiles, mask_tiles = -1, [], []
    n_box = n_miss = 0
    for f in pick:
        while cur < f and cap.grab():
            cur += 1
        ok, frame = cap.retrieve()
        if not ok:
            break
        H, W = frame.shape[:2]
        res = model.predict(frame, imgsz=a.imgsz, conf=0.05, classes=[0], verbose=False,
                            device="cpu", retina_masks=True, max_det=100)[0]
        yb = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        ys = res.boxes.conf.cpu().numpy()
        db = np.asarray(by_frame[f], np.float32).reshape(-1, 4)
        pairs = greedy_match(iou_matrix(db, yb), 0.3) if len(yb) else []
        pmap = {i: j for i, j, _v in pairs}

        vis = frame.copy()
        for i, b in enumerate(db):
            hit = i in pmap
            n_box += 1
            n_miss += 0 if hit else 1
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                          (255, 120, 0) if hit else (0, 0, 255), 2)
            if not hit:
                cv2.putText(vis, "MISS", (int(b[0]) + 2, max(11, int(b[1]) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        for j, b in enumerate(yb):
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
        cv2.putText(vis, f"f{f} dino={len(db)} yolo={len(yb)} miss={len(db)-len(pairs)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        miss_tiles.append(cv2.resize(vis, (W // 2, H // 2), interpolation=cv2.INTER_AREA))

        # 掩膜边界叠加（已匹配的前 3 个）
        if res.masks is not None:
            mdata = res.masks.data.cpu().numpy()
            for i, j in list(pmap.items())[:3]:
                m = mdata[j]
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                x1, y1, x2, y2 = [int(v) for v in db[i]]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                c = frame[y1:y2, x1:x2].copy()
                mm = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8)
                cont, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(c, cont, -1, (0, 255, 0), 2)
                if c.size:
                    sc = 128.0 / max(1, c.shape[0])
                    c = cv2.resize(c, (int(c.shape[1] * sc), 128))
                    mask_tiles.append(c)

    def grid(tiles, cols, w, h, path):
        if not tiles:
            return
        tiles = [cv2.resize(t, (w, h)) for t in tiles[:cols * 6]]
        while len(tiles) % cols:
            tiles.append(np.zeros((h, w, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(path), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    grid(miss_tiles, 2, miss_tiles[0].shape[1] if miss_tiles else 640,
         miss_tiles[0].shape[0] if miss_tiles else 360, OUT / f"{a.seg}_diag_miss.jpg")
    grid(mask_tiles, 8, 64, 128, OUT / f"{a.seg}_diag_mask.jpg")
    print(f"[{a.seg}] boxes={n_box} miss={n_miss} ({100.0*n_miss/max(1,n_box):.1f}%) -> "
          f"{OUT}/{a.seg}_diag_{{miss,mask}}.jpg", flush=True)


if __name__ == "__main__":
    main()
