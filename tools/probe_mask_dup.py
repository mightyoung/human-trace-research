#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证「框重叠」与「掩膜重叠」是否能区分同一人的重复框。

背景：YOLO26s-seg 在 iou=0.5 的 NMS + 检测级去重后，仍有部分帧存在 IoU 0.25–0.5 的
      重叠框对。目检发现其中既有「同一人的头/身两框」，也有「前景人头压在后景人身框内」
      （两个真人）。仅凭框几何无法区分，而实例掩膜可以：
        - 同一人两框：掩膜**包含/重叠**（头 ⊂ 身）
        - 两个真人投影重叠：掩膜**互斥**（前景人挡住了后景人，像素不共享）

用法：
  python tools/probe_mask_dup.py --seg 305_test_544 --frames 300 --stride 5
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models/Ultralytics--YOLO26/snapshots/master/yolo26s-seg.pt")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="305_test_544")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-box-iou", type=float, default=0.20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    cap = cv2.VideoCapture(os.path.join(ROOT, "media-resource/1012/segments", f"{a.seg}.mp4"))
    recs = []
    done = 0
    f = 0
    while done < a.frames:
        ok = cap.grab()
        if not ok:
            break
        if f % a.stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            r = model.predict(img, classes=[0], conf=a.conf, iou=a.iou, imgsz=a.imgsz, verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            masks = r.masks.data.cpu().numpy() if (r.masks is not None) else None
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    bi = iou(boxes[i], boxes[j])
                    if bi < a.min_box_iou:
                        continue
                    mi = None
                    if masks is not None and i < len(masks) and j < len(masks):
                        m1, m2 = masks[i] > 0.5, masks[j] > 0.5
                        inter = np.logical_and(m1, m2).sum()
                        union = np.logical_or(m1, m2).sum()
                        mi = float(inter / union) if union else 0.0
                        # 小掩膜被大掩膜包含的比例
                        s1, s2 = m1.sum(), m2.sum()
                        small, big = (m1, m2) if s1 <= s2 else (m2, m1)
                        sb = small.sum()
                        ctn = float(np.logical_and(small, big).sum() / sb) if sb else 0.0
                    else:
                        ctn = None
                    recs.append({"frame": f, "box_iou": round(float(bi), 3),
                                 "mask_iou": None if mi is None else round(mi, 3),
                                 "mask_contain": None if ctn is None else round(ctn, 3),
                                 "box_a": [round(float(x), 1) for x in boxes[i]],
                                 "box_b": [round(float(x), 1) for x in boxes[j]]})
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{a.frames} frames, pairs={len(recs)}", flush=True)
        f += 1
    cap.release()

    if not recs:
        print("没有满足条件（box IoU >= %.2f）的框对" % a.min_box_iou)
        return
    bi = np.array([r["box_iou"] for r in recs])
    mi = np.array([r["mask_iou"] for r in recs if r["mask_iou"] is not None])
    print(f"\n[{a.seg}] 框对 {len(recs)} 个（{done} 帧）")
    print(f"  box IoU  中位 {np.median(bi):.2f}  p90 {np.percentile(bi,90):.2f}")
    if len(mi):
        print(f"  mask IoU 中位 {np.median(mi):.2f}  p10 {np.percentile(mi,10):.2f}  p90 {np.percentile(mi,90):.2f}")
        for t in (0.1, 0.3, 0.5, 0.7):
            print(f"    mask IoU >= {t}: {np.mean(mi>=t)*100:.0f}%")
        print(f"  双峰检验：mask IoU<0.1 占 {np.mean(mi<0.1)*100:.0f}%，>0.5 占 {np.mean(mi>0.5)*100:.0f}%")

    out = a.out or os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012", f"mask_dup_{a.seg}.json")
    json.dump({"seg": a.seg, "frames": done, "pairs": recs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
