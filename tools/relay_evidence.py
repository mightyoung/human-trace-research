#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
接力链证据图生成：把链的每个切换点渲染为「A 末帧 | B 首帧」对照。

用法：
  python tools/relay_evidence.py --chains relay_chains.json --out evid_chains.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_view import local_det  # noqa: E402
OUT_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "relay")
VID_DIR = os.path.join(ROOT, "media-resource", "multi-cemra", "synced-10min")
VID = {
    "316": "01_行政走廊通道_316", "305": "02_办公区1_305", "318": "03_办公区2_318",
    "320": "04_办公区球机_320", "323": "05_茶水间_323", "343": "06_电梯间_343",
}
T0_OFFSET = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0


def vid_path(cam):
    d = os.path.join(VID_DIR, VID[cam])
    for f in os.listdir(d):
        if f.endswith(".mp4"):
            return os.path.join(d, f)
    raise FileNotFoundError(d)


def load_dets(cam):
    """{frame: {tid: box}}"""
    path = local_det(cam)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            m = {}
            for tid, box in zip(d["track_ids"], d["boxes"]):
                m[tid] = box
            out[fr] = m
    return out


def load_events(cam):
    with open(os.path.join(OUT_DIR, f"events_{cam}.json"), encoding="utf-8") as f:
        data = json.load(f)
    return {r["eid"].replace(f"{cam}-", ""): r for r in data["events"]}


def crop_frame(cap, frame, dets, tid, pad=0.28, out_h=300):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    if not ok:
        return None
    H, W = img.shape[:2]
    box = dets.get(frame, {}).get(tid)
    if box is None:
        # 邻近帧找
        for df in [1, -1, 2, -2, 3, -3, 5, -5]:
            box = dets.get(frame + df, {}).get(tid)
            if box is not None:
                break
    if box is None:
        return None
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # 以人为中心扩 pad（相对尺寸），并保持 3:4 画幅
    half_h = bh * (1 + pad) / 2
    half_w = max(bw * (1 + pad) / 2, half_h * 0.75)
    X1 = int(max(0, cx - half_w)); X2 = int(min(W, cx + half_w))
    Y1 = int(max(0, cy - half_h)); Y2 = int(min(H, cy + half_h))
    if X2 - X1 < 8 or Y2 - Y1 < 8:
        return None
    crop = img[Y1:Y2, X1:X2]
    h = out_h
    w = int(crop.shape[1] * h / crop.shape[0])
    return cv2.resize(crop, (max(w, 40), h), interpolation=cv2.INTER_CUBIC)

def ts(cam, frame):
    t = T0_OFFSET[cam] + frame / FPS
    base = 8 * 3600 + 55 * 60  # 08:55:00
    tt = base + t
    hh = int(tt // 3600) % 24
    mm = int(tt % 3600 // 60)
    ss = tt % 60
    return f"{hh:02d}:{mm:02d}:{ss:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default=os.path.join(OUT_DIR, "relay_chains.json"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "evid_chains.png"))
    ap.add_argument("--max-chains", type=int, default=12)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--cell-h", type=int, default=420, help="单元格高度")
    ap.add_argument("--font", type=float, default=1.0, help="字号倍率")
    ap.add_argument("--per-chain", action="store_true",
                    help="每链单独输出 evid_chain_{idx:02d}.jpg")
    ap.add_argument("--prefix", default="evid_chain",
                    help="每链输出的文件名前缀（默认 evid_chain）")
    args = ap.parse_args()

    with open(args.chains, encoding="utf-8") as f:
        data = json.load(f)
    chains = data["chains"]
    meta = {(e["a"], e["b"]): e for e in data["edges"]}
    chains = [c for c in chains if len(c) >= args.min_len][: args.max_chains]

    caps, dets, events = {}, {}, {}
    rows = []
    for chain in chains:
        cells = []  # 每格：(img, 标签)
        ok_chain = True
        for i in range(len(chain) - 1):
            na, nb = chain[i], chain[i + 1]
            ca, ea = na.split(":")
            cb, eb = nb.split(":")
            ea = ea.replace(f"{ca}-", ""); eb = eb.replace(f"{cb}-", "")
            for c in (ca, cb):
                if c not in caps:
                    caps[c] = cv2.VideoCapture(vid_path(c))
                    dets[c] = load_dets(c)
                    events[c] = load_events(c)
            ma = meta.get((na, nb))
            ra = events[ca].get(ea); rb = events[cb].get(eb)
            if ra is None or rb is None or ma is None:
                ok_chain = False
                break
            img_a = crop_frame(caps[ca], ra["f_last"], dets[ca],
                               ma.get("a_frag_last", None) or ra.get("tid_last"),
                               out_h=args.cell_h)
            img_b = crop_frame(caps[cb], rb["f_first"], dets[cb],
                               ma.get("b_frag_first", None) or rb.get("tid_first"),
                               out_h=args.cell_h)
            if img_a is None or img_b is None:
                ok_chain = False
                break
            cells.append((img_a, f"{ca}:{ea} LAST {ts(ca, ra['f_last'])} h{ra['h_med']:.0f}px", (0, 165, 255)))
            cells.append((img_b, f"{cb}:{eb} FIRST {ts(cb, rb['f_first'])} h{rb['h_med']:.0f}px dt={ma['dt']}s sim={ma['sim']:.2f}", (0, 220, 0)))
        for c in caps:
            pass
        if ok_chain and cells:
            # 保障统一高度
            hmax = max(c[0].shape[0] for c in cells)
            cells = [(cv2.resize(im, (int(im.shape[1] * hmax / im.shape[0]), hmax)), t, col) for im, t, col in cells]
            rows.append((chain, cells))
    for c in caps:
        caps[c].release()

    if not rows:
        print("no rows rendered")
        return

    # 画布：每链一行
    gap = 6
    pad_top = int(42 * max(args.font, 1.0))
    row_imgs = []
    W = max(sum(im.shape[1] for im, _, _ in cells) + gap * (len(cells) - 1) + 20
            for _, cells in rows)
    for ci, (chain, cells) in enumerate(rows):
        h = cells[0][0].shape[0]
        canvas = np.full((h + pad_top + 8, W, 3), 24, np.uint8)
        title = " -> ".join(chain)
        cv2.putText(canvas, title, (12, int(32 * max(args.font, 1.0))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * args.font, (255, 255, 255), 2, cv2.LINE_AA)
        x = 10
        for im, label, col in cells:
            canvas[pad_top:pad_top + im.shape[0], x:x + im.shape[1]] = im
            fs = 0.62 * args.font
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.rectangle(canvas, (x + 2, pad_top + im.shape[0] - th - 12),
                          (x + tw + 10, pad_top + im.shape[0] - 2), (0, 0, 0), -1)
            cv2.putText(canvas, label, (x + 6, pad_top + im.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2, cv2.LINE_AA)
            x += im.shape[1] + gap
        row_imgs.append(canvas)
        if args.per_chain:
            p = os.path.join(OUT_DIR, f"{args.prefix}_{ci:02d}.jpg")
            cv2.imwrite(p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"[evid] {p}")
    sep = np.full((10, W, 3), 90, np.uint8)
    full = row_imgs[0]
    for r in row_imgs[1:]:
        full = np.vstack([full, sep, r])
    cv2.imwrite(args.out, full, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[evid] {len(rows)} 链 -> {args.out} ({full.shape})")


if __name__ == "__main__":
    main()
