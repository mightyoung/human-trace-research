#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把「人脸检出」和「整体轨迹」画在同一张图上，并给出对齐后的人脸裁图。

目的：让「pd 17px 到底能不能认人」变成看得见的事，而不是只看 AUC。
两栏对照：
  · 305 的可用轨迹（pd 22–30）—— 该用的那批
  · 320 的错判轨迹（pd 17–18）—— 想用但不够的那批

用法：python tools/vis_face_tags.py
输出：selector_osnet_1012/face_routing/vis/face_quality_compare.jpg
      selector_osnet_1012/face_routing/vis/face_tag_demo.jpg
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from scrfd_dnn import ScrfdDNN  # noqa: E402
from probe_face_routing import (load_tracks, detect_faces_grid, assign_faces_to_boxes,  # noqa: E402
                                face_quality, align112)

FR = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/face_routing")
OUT = os.path.join(FR, "vis")
VID = os.path.join(ROOT, "media-resource/1012/segments")
FONT = "C:/Windows/Fonts/msyh.ttc"
if not os.path.exists(FONT):
    FONT = "C:/Windows/Fonts/simhei.ttf"


def font(sz):
    return ImageFont.truetype(FONT, sz)


def obs_by_track(segs):
    d = defaultdict(list)
    for seg in segs:
        p = os.path.join(FR, f"face_obs_{seg}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            o = json.loads(line)
            d[(seg, str(o["tid"]))].append(o)
    return d


def tier(pd):
    if pd >= 20:
        return "usable", (34, 158, 117)
    if pd >= 12:
        return "grey", (239, 159, 39)
    return "too_small", (140, 84, 196)


def render_rows(items, det, obs, frame_cache, TB, W_CELL=300, H_CELL=226):
    """返回 (PIL 图, 行数)"""
    TITLE_H, LABEL_W, PAD = 54, 250, 12
    img_h = TITLE_H + len(items) * (H_CELL + PAD) + PAD
    img_w = PAD + LABEL_W + 3 * (W_CELL + PAD) + 3 * (96 + PAD) + PAD
    canvas = Image.new("RGB", (img_w, img_h), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    dr.text((PAD, 14), "人脸检出 ←→ 整体轨迹：同一帧、同一人，对齐后的脸能不能认人", font=font(19), fill=(30, 30, 30))
    dr.text((PAD + 470, 20), "人脸框：绿=瞳距≥20px 可识别　黄=12–20px 灰区　紫=<12px 不可用（红框=人体框）",
            font=font(13), fill=(110, 110, 110))

    for ri, (seg, tid, label, note) in enumerate(items):
        y0 = TITLE_H + ri * (H_CELL + PAD)
        dr.text((PAD, y0 + 2), f"{seg.split('_')[0]} #{tid}", font=font(17), fill=(30, 30, 30))
        dr.text((PAD, y0 + 26), label, font=font(15), fill=(120, 60, 60) if "错" in label else (30, 110, 80))
        dr.text((PAD, y0 + 48), note, font=font(12), fill=(120, 120, 120))

        cand = sorted(obs.get((seg, str(tid)), []), key=lambda o: -o["pd"])[:18]
        rendered = []
        for o in cand:
            if len(rendered) >= 3:
                break
            fr = o["frame"]
            key = (seg, fr)
            if key not in frame_cache:
                cap = cv2.VideoCapture(os.path.join(VID, f"{seg}.mp4"))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ok, im = cap.read()
                cap.release()
                if not ok:
                    continue
                frame_cache[key] = im
            img = frame_cache[key]
            boxes = TB[seg].get(fr, [])
            asg, _ = assign_faces_to_boxes(detect_faces_grid(img, det), boxes)
            if tid not in asg:
                continue
            f, d, cov = asg[tid]
            box = dict((t, b) for t, b in boxes)[tid]
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx1 = int(max(0, x1 - 0.18 * w))
            cx2 = int(min(img.shape[1], x2 + 0.18 * w))
            cy1 = int(max(0, y1 - 0.22 * h))
            cy2 = int(min(img.shape[0], y2 + 0.12 * h))
            crop = img[cy1:cy2, cx1:cx2].copy()
            sc = H_CELL / max(crop.shape[0], 1)
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), H_CELL))

            def m(px, py, cx1=cx1, cy1=cy1, sc=sc):
                return int((px - cx1) * sc), int((py - cy1) * sc)

            cv2.rectangle(crop, m(x1, y1), m(x2, y2), (40, 40, 220), 2)
            tl, col = tier(o["pd"])
            bgr = (col[2], col[1], col[0])
            cv2.rectangle(crop, m(f[0], f[1]), m(f[2], f[3]), bgr, 2)
            cv2.putText(crop, "pd %.0f" % o["pd"], (m(f[0], f[1])[0], max(14, m(f[0], f[1])[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
            ali = align112(img, f[4:14].reshape(5, 2).astype(np.float32))
            if ali is not None:
                ali = cv2.resize(ali, (96, 96), interpolation=cv2.INTER_CUBIC)
            rendered.append((o, crop, ali, col))

        for ci, (o, crop, ali, col) in enumerate(rendered):
            px = PAD + LABEL_W + ci * (W_CELL + PAD)
            canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (px, y0))
            if ali is not None:
                ax = PAD + LABEL_W + 3 * (W_CELL + PAD) + ci * (96 + PAD)
                canvas.paste(Image.fromarray(cv2.cvtColor(ali, cv2.COLOR_BGR2RGB)), (ax, y0 + 60))
                dr.rectangle([ax - 1, y0 + 59, ax + 96, y0 + 156], outline=col, width=2)
                dr.text((ax, y0 + 158), "对齐 112px", font=font(11), fill=(140, 140, 140))
    return canvas


def main():
    os.makedirs(OUT, exist_ok=True)
    det = ScrfdDNN(os.path.join(ROOT, "models/face/scrfd_2.5g_bnkps.onnx"), thresh=0.5, nms=0.4)
    segs = ["305_test_544", "320_test_518"]
    obs = obs_by_track(segs)
    cache = {}

    items = [
        ("305_test_544", 4, "305 可用（pd 30）", "办公区近景，正脸"),
        ("305_test_544", 3597, "305 可用（pd 26）", "办公区近景"),
        ("305_test_544", 9, "305 可用（pd 24）", "可识别下限附近"),
        ("320_test_518", 886, "320 判错（pd 18）", "球机俯视，侧脸/低头"),
        ("320_test_518", 534, "320 判错（pd 18）", "球机俯视"),
        ("320_test_518", 542, "320 判错（pd 18）", "球机俯视"),
        ("320_test_518", 99, "320 判错（pd 18）", "球机俯视，侧脸"),
    ]
    TB = {s: load_tracks(s) for s in segs}
    img = render_rows(items, det, obs, cache, TB)
    p1 = os.path.join(OUT, "face_quality_compare.jpg")
    img.save(p1, quality=90)
    print("[out]", p1, img.size)

    # 第二张：整帧打标签示意（320#900，3 张脸全部分配）
    seg, fr = "320_test_518", 900
    cap = cv2.VideoCapture(os.path.join(VID, f"{seg}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, img0 = cap.read()
    cap.release()
    boxes = load_tracks(seg)[fr]
    faces = detect_faces_grid(img0, det)
    asg, _ = assign_faces_to_boxes(faces, boxes)
    vis = img0.copy()
    for tid, b in boxes:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (40, 40, 220), 3)
        tag, col = "整体 only", (120, 120, 120)
        if tid in asg:
            f, d, cov = asg[tid]
            q = face_quality(img0, f)
            pd = q["pd"] if q else 0
            tl, c = tier(pd)
            tag = {"usable": "脸可用 pd=%.0f" % pd, "grey": "灰区 pd=%.0f" % pd,
                   "too_small": "脸太小 pd=%.0f" % pd}[tl]
            col = (c[2], c[1], c[0])
            cv2.rectangle(vis, (int(f[0]), int(f[1])), (int(f[2]), int(f[3])), col, 3)
        cv2.rectangle(vis, (int(b[0]), int(b[1]) - 34), (int(b[0]) + 250, int(b[1])), col, -1)
        cv2.putText(vis, "#%d %s" % (tid, tag), (int(b[0]) + 6, int(b[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    p2 = os.path.join(OUT, "face_tag_demo.jpg")
    cv2.imwrite(p2, cv2.resize(vis, (1400, 788)), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("[out]", p2)


if __name__ == "__main__":
    main()
