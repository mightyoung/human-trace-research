#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO vs DINO 诊断图：三色对照 + 争议框裁剪网格。

回答的问题：
  1. YOLO 多出来的框（yolo_only）是真的人，还是椅子/显示器的误检？
  2. DINO 有而 YOLO 没有的框（dino_only），是真小目标还是 OSD/背景误检？
  3. 匹配上的框，两者贴合度如何？

输出（output_crosscam_local/yolo_vs_dino/）：
  {seg}_triage.jpg      整帧三色图（绿=匹配 红=仅YOLO 黄=仅DINO）
  {seg}_crops_yolo_only.jpg  YOLO 独有框裁剪（按 conf 分层抽样）
  {seg}_crops_dino_only.jpg  DINO 独有框裁剪（按尺寸分层抽样）

用法:
  python tools/yolo_dino_diagnose.py --seg 305_test_544 --conf 0.25
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

GREEN = (60, 200, 60)
RED = (60, 60, 230)
YELLOW = (40, 200, 240)
WHITE = (245, 245, 245)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    aa = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    ba = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = aa + ba[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(iou, thr):
    if iou.size == 0:
        return []
    idx = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    ua, ub, pairs = set(), set(), []
    for i, j in idx:
        v = float(iou[i, j])
        if v < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(int(i))
        ub.add(int(j))
        pairs.append((int(i), int(j), v))
    return pairs


def load_dino(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[int(r["frame"])] = np.asarray(r.get("boxes") or [], np.float32).reshape(-1, 4)
    return out


def load_yolo(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        out[int(r["frame"])] = (np.asarray(r["boxes"], np.float32).reshape(-1, 4),
                                np.asarray(r["scores"], np.float32).reshape(-1),
                                np.asarray(r.get("mask_frac") or [], np.float32).reshape(-1))
    return out


def read_frames(video, wanted):
    """顺序读指定帧号 -> {frame: img}"""
    wanted = sorted(wanted)
    out, ti, fi = {}, 0, 0
    cap = cv2.VideoCapture(str(video))
    while ti < len(wanted) and cap.grab():
        if fi == wanted[ti]:
            ok, img = cap.retrieve()
            if not ok:
                break
            out[fi] = img
            ti += 1
        fi += 1
    cap.release()
    return out


def triage_tile(img, db, yb, ys, thr=0.5, scale=1.0):
    iou = iou_matrix(db, yb)
    pairs = greedy_match(iou, thr)
    md = {i for i, _, _ in pairs}
    my = {j for _, j, _ in pairs}
    out = img.copy()
    for i, b in enumerate(db):
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        if i in md:
            cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 3)
            cv2.putText(out, "D", (x1 + 2, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, YELLOW, 2, cv2.LINE_AA)
    for j, b in enumerate(yb):
        if j in my:
            continue
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, 3)
        cv2.putText(out, f"Y{ys[j]:.2f}", (x1 + 2, max(y2 + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, RED, 2, cv2.LINE_AA)
    return out, len(pairs), len(db) - len(md), len(yb) - len(my)


def legend(img):
    h, w = img.shape[:2]
    bar = np.full((36, w, 3), 30, np.uint8)
    items = [(GREEN, "matched"), (RED, "YOLO only"), (YELLOW, "DINO only")]
    x = 10
    for color, txt in items:
        cv2.line(bar, (x, 18), (x + 26, 18), color, 5)
        cv2.putText(bar, txt, (x + 32, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        x += 170
    return np.vstack([bar, img])


def crop_grid(img, boxes, labels, cell=(180, 260), cols=8, ctx=1.8):
    """boxes: list of (x1,y1,x2,y2) 原图坐标。按上下文裁剪并缩放拼网格。"""
    cw, ch = cell
    tiles = []
    h, w = img.shape[:2]
    for b, lab in zip(boxes, labels):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        bw, bh = max(b[2] - b[0], 8), max(b[3] - b[1], 8)
        bw, bh = bw * ctx, bh * ctx
        x1 = int(max(cx - bw / 2, 0)); x2 = int(min(cx + bw / 2, w))
        y1 = int(max(cy - bh / 2, 0)); y2 = int(min(cy + bh / 2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 22, 4, 2, 2, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(crop, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(crop)
    if not tiles:
        return None
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="305_test_544")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--n-triage", type=int, default=6)
    ap.add_argument("--n-crops", type=int, default=40)
    args = ap.parse_args()

    seg = args.seg
    video = SEG_DIR / f"{seg}.mp4"
    dino = load_dino(DINO_DIR / f"{seg}_a.jsonl")
    yolo = load_yolo(OUT_DIR / f"{seg}_yolo.jsonl")

    frames = [f for f in yolo if f in dino]
    # 每帧统计：DINO 漏检数、YOLO 独有数
    per = []
    for f in frames:
        db = dino[f]
        yb, ys, _ = yolo[f]
        keep = ys >= args.conf
        yb, ys = yb[keep], ys[keep]
        iou = iou_matrix(db, yb)
        pairs = greedy_match(iou, args.thr)
        per.append((len(db) - len(pairs), len(yb) - len(pairs), len(db), f))
    per.sort(reverse=True)
    # 交替取"DINO 漏检最多"和"YOLO 独有最多"的代表帧
    picks = []
    for k in range(min(args.n_triage, len(per))):
        picks.append(per[k][3])
    picks = sorted(set(picks))[:args.n_triage]

    imgs = read_frames(video, picks)
    tiles = []
    for f in picks:
        if f not in imgs:
            continue
        db = dino[f]
        yb, ys, _ = yolo[f]
        keep = ys >= args.conf
        yb, ys = yb[keep], ys[keep]
        tile, m, dn, yonly = triage_tile(imgs[f], db, yb, ys, args.thr)
        h, w = tile.shape[:2]
        sc = 760 / w
        tile = cv2.resize(tile, (760, int(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"f{f}  matched={m} dino_only={dn} yolo_only={yonly}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        ncol = 2
        while len(tiles) % ncol:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)]
        grid = legend(np.vstack(rows))
        p = OUT_DIR / f"{seg}_triage.jpg"
        cv2.imwrite(str(p), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        print(f"[out] {p}  ({len(picks)} frames)", flush=True)

    # ---- 裁剪：YOLO 独有（按 conf 分层）
    cand_y = []
    for f in frames:
        db = dino[f]
        yb, ys, mf = yolo[f]
        keep = ys >= args.conf
        idx = np.where(keep)[0]
        yb2, ys2 = yb[keep], ys[keep]
        iou = iou_matrix(db, yb2)
        pairs = greedy_match(iou, args.thr)
        my = {j for _, j, _ in pairs}
        for j in range(len(yb2)):
            if j in my:
                continue
            cand_y.append((f, yb2[j].tolist(), float(ys2[j])))
    cand_y.sort(key=lambda x: -x[2])
    if len(cand_y) > args.n_crops:
        k = args.n_crops
        step = len(cand_y) / k
        sel_y = [cand_y[int(i * step)] for i in range(k)]
    else:
        sel_y = cand_y
    if sel_y:
        need = {c[0] for c in sel_y}
        im = read_frames(video, need)
        boxes, labs = [], []
        for f, b, s in sel_y:
            if f not in im:
                continue
            boxes.append(b)
            labs.append(f"f{f} c{s:.2f}")
        # 按帧分组绘制（不同帧各自裁剪）
        tiles = []
        for f, b, s in sel_y:
            if f not in im:
                continue
            cw, ch = 180, 260
            img = im[f]
            h, w = img.shape[:2]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            bw, bh = max(b[2] - b[0], 8) * 1.8, max(b[3] - b[1], 8) * 1.8
            x1, x2 = int(max(cx - bw / 2, 0)), int(min(cx + bw / 2, w))
            y1, y2 = int(max(cy - bh / 2, 0)), int(min(cy + bh / 2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(img[y1:y2, x1:x2], (cw, ch), interpolation=cv2.INTER_CUBIC)
            crop = cv2.copyMakeBorder(crop, 22, 4, 2, 2, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            cv2.putText(crop, f"f{f} c{s:.2f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(crop)
        if tiles:
            cols = 8
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
            p = OUT_DIR / f"{seg}_crops_yolo_only.jpg"
            cv2.imwrite(str(p), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            print(f"[out] {p}  ({len(tiles)} crops)", flush=True)

    # ---- 裁剪：DINO 独有（按面积从小到大 = 最可能是漏检小目标）
    cand_d = []
    for f in frames:
        db = dino[f]
        yb, ys, _ = yolo[f]
        keep = ys >= args.conf
        yb = yb[keep]
        iou = iou_matrix(db, yb)
        pairs = greedy_match(iou, args.thr) if len(yb) else []
        md = {i for i, _, _ in pairs}
        for i in range(len(db)):
            if i in md:
                continue
            b = db[i]
            cand_d.append((f, b.tolist(), (b[2] - b[0]) * (b[3] - b[1])))
    cand_d.sort(key=lambda x: x[2])  # 最小优先
    if len(cand_d) > args.n_crops:
        step = len(cand_d) / args.n_crops
        sel_d = [cand_d[int(i * step)] for i in range(args.n_crops)]
    else:
        sel_d = cand_d
    if sel_d:
        im = read_frames(video, {c[0] for c in sel_d})
        tiles = []
        for f, b, a in sel_d:
            if f not in im:
                continue
            img = im[f]
            h, w = img.shape[:2]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            bw, bh = max(b[2] - b[0], 6) * 2.6, max(b[3] - b[1], 6) * 2.6
            x1, x2 = int(max(cx - bw / 2, 0)), int(min(cx + bw / 2, w))
            y1, y2 = int(max(cy - bh / 2, 0)), int(min(cy + bh / 2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(img[y1:y2, x1:x2], (180, 260), interpolation=cv2.INTER_CUBIC)
            crop = cv2.copyMakeBorder(crop, 22, 4, 2, 2, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            cv2.putText(crop, f"f{f} a={a:.0f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (240, 240, 240), 1, cv2.LINE_AA)
            tiles.append(crop)
        if tiles:
            cols = 8
            while len(tiles) % cols:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
            p = OUT_DIR / f"{seg}_crops_dino_only.jpg"
            cv2.imwrite(str(p), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            print(f"[out] {p}  ({len(tiles)} crops)", flush=True)

    # 汇总
    print(f"[stat] yolo_only candidates={len(cand_y)}  dino_only candidates={len(cand_d)} "
          f"(conf>={args.conf}, iou>={args.thr})", flush=True)


if __name__ == "__main__":
    main()
