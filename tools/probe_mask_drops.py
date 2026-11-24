#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""复现 yolo_track_det.py 的 --dedup-mask 判定，并把「谁删了谁」记录下来供目检。

与 probe_mask_dup.py 的区别：那个只统计重叠框对的掩膜分布；这个走完全相同的贪心删除，
并输出 帧号 / 被删框 / 触发删除的保留框 / 掩膜包含率 / 掩膜IoU / 框IoU，用于判断规则是否误删。

用法：
  python tools/probe_mask_drops.py --seg 305_test_544 --frames 30 --contain 0.9
"""
import argparse
import json
import os

import cv2
import numpy as np

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
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--contain", type=float, default=0.9)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/mask_drops"))
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    cap = cv2.VideoCapture(os.path.join(ROOT, "media-resource/1012/segments", f"{a.seg}.mp4"))
    os.makedirs(a.out_dir, exist_ok=True)
    recs, samples = [], []
    done, f = 0, 0
    while done < a.frames:
        if not cap.grab():
            break
        if f % a.stride == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            r = model.predict(img, classes=[0], conf=a.conf, iou=a.iou, imgsz=a.imgsz, verbose=False)[0]
            bs = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            cf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            ms = r.masks.data.cpu().numpy() if r.masks is not None else None
            if ms is not None and len(bs) > 1:
                keep = []
                for j in sorted(range(len(bs)), key=lambda k: -float(cf[k])):
                    hit = None
                    for k in keep:
                        m1, m2 = ms[j] > 0.5, ms[k] > 0.5
                        s1, s2 = int(m1.sum()), int(m2.sum())
                        if not s1 or not s2:
                            continue
                        small, big = (m1, m2) if s1 <= s2 else (m2, m1)
                        ctn = float(np.logical_and(small, big).sum()) / min(s1, s2)
                        if ctn >= a.contain:
                            inter = np.logical_and(m1, m2).sum()
                            union = np.logical_or(m1, m2).sum()
                            hit = (k, ctn, float(inter / union) if union else 0.0)
                            break
                    if hit:
                        rec = {"frame": f, "drop_box": [round(float(v), 1) for v in bs[j]],
                               "keep_box": [round(float(v), 1) for v in bs[hit[0]]],
                               "mask_contain": round(float(hit[1]), 3), "mask_iou": round(float(hit[2]), 3),
                               "box_iou": round(float(iou(bs[j], bs[hit[0]])), 3),
                               "drop_area": round(float((bs[j][2]-bs[j][0])*(bs[j][3]-bs[j][1])), 0),
                               "keep_area": round(float((bs[hit[0]][2]-bs[hit[0]][0])*(bs[hit[0]][3]-bs[hit[0]][1])), 0)}
                        recs.append(rec)
                        if len(samples) < 8:
                            samples.append((img.copy(), rec))
                    else:
                        keep.append(j)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{a.frames} drops={len(recs)}", flush=True)
        f += 1
    cap.release()

    json.dump(recs, open(os.path.join(a.out_dir, f"drops_{a.seg}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    bi = np.array([r["box_iou"] for r in recs]) if recs else np.zeros(1)
    ar = np.array([r["drop_area"] / max(1, r["keep_area"]) for r in recs]) if recs else np.zeros(1)
    print(f"\n[{a.seg}] 删除 {len(recs)} 个框（{done} 帧）")
    print(f"  框IoU 中位 {np.median(bi):.2f}；<0.05 的 {np.mean(bi < 0.05) * 100:.0f}%（越低越可疑）")
    print(f"  被删框面积/保留框面积 中位 {np.median(ar):.2f}")

    tiles = []
    for img, r in samples:
        b = r["drop_box"]; k = r["keep_box"]
        x0 = max(0, int(min(b[0], k[0])) - 260); y0 = max(0, int(min(b[1], k[1])) - 180)
        x1 = min(img.shape[1], int(max(b[2], k[2])) + 260); y1 = min(img.shape[0], int(max(b[3], k[3])) + 180)
        c = img[y0:y1, x0:x1].copy()
        cv2.rectangle(c, (int(k[0]) - x0, int(k[1]) - y0), (int(k[2]) - x0, int(k[3]) - y0), (0, 165, 255), 3)
        cv2.rectangle(c, (int(b[0]) - x0, int(b[1]) - y0), (int(b[2]) - x0, int(b[3]) - y0), (0, 0, 255), 3)
        s = 620 / max(1, c.shape[1])
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))
        cv2.putText(c, f"f={r['frame']} red=drop orange=keep IoU={r['box_iou']:.2f}", (6, 20), 0, 0.55, (255, 255, 255), 2)
        tiles.append(c)
    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, min(8, len(tiles)), 2)]
        cv2.imwrite(os.path.join(a.out_dir, "drops_preview.jpg"), np.vstack(rows))
        print(f"[out] {os.path.join(a.out_dir, 'drops_preview.jpg')}")


if __name__ == "__main__":
    main()
