#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为人工标注生成每轨迹 8 帧拼图（D3 图库层 / D4 查询层用）。

对 `ledger/tracks.csv` 里每条轨迹，从原视频取 8 帧（按轨迹自身帧序列均匀采样，
跨 a/b 两半），拼成 2x4 图，写 `gt/crops/<seg>/<tid>.jpg`。

纪律：
  - 图上只写轨迹 id / 帧号 / 帧数，**不写簇号、分数、接受状态**（避免标注被系统输出带偏）。
  - 图库与查询分别放各自目录，标注时不同时看。

用法：
  python tools/render_track_sheets.py --ledger deliverable/data-request-v2/ledger/tracks.csv
  python tools/render_track_sheets.py --split query     # 只渲染查询
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VID = os.path.join(ROOT, "media-resource/1012/segments")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
CELL, PAD, COLS = 300, 1, 4
NFRAME = 8


def pick_even(seq, k):
    if not seq:
        return []
    if len(seq) <= k:
        return list(seq)
    return [seq[int(round(i * (len(seq) - 1) / (k - 1)))] for i in range(k)]


def tile(img, box, label):
    out = np.full((CELL, CELL, 3), 24, np.uint8)
    if img is None:
        cv2.putText(out, "read fail", (8, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 200), 2)
        return out
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    pad = max(12, int(0.25 * max(x2 - x1, y2 - y1)))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        return out
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (70, 220, 90), 2)
    ch, cw = crop.shape[:2]
    s = CELL / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (CELL - crop.shape[0]) // 2, (CELL - crop.shape[1]) // 2
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    cv2.putText(out, label, (8, CELL - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(ROOT, "deliverable/data-request-v2/ledger/tracks.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "deliverable/data-request-v2/gt/crops"))
    ap.add_argument("--split", default="", help="query / gallery；留空=两者")
    ap.add_argument("--max-per-seg", type=int, default=0, help="调试用：每段最多渲染几条")
    ap.add_argument("--quality", type=int, default=86)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.ledger, encoding="utf-8-sig")))
    if a.split:
        rows = [r for r in rows if r["split"] == a.split]
    by_seg = collections.defaultdict(list)
    for r in rows:
        by_seg[r["seg"]].append(r)
    print(f"[in] {len(rows)} 条轨迹，{len(by_seg)} 段")

    for seg, items in sorted(by_seg.items()):
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        dets = {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                dets[o["frame"]] = (o["track_ids"], o["boxes"])
        vp = os.path.join(VID, f"{seg}.mp4")
        if not os.path.exists(vp):
            print(f"[skip] 缺视频 {vp}")
            continue
        outd = os.path.join(a.out, seg)
        os.makedirs(outd, exist_ok=True)
        if a.max_per_seg:
            items = items[:a.max_per_seg]

        # 每条的取帧集合
        plan = {}
        for r in items:
            tid = int(r["tid"])
            fr = sorted(f for f, (ts, _b) in dets.items() if tid in ts)
            plan[tid] = pick_even(fr, NFRAME)

        want = sorted({f for v in plan.values() for f in v})
        wset = set(want)
        cap = cv2.VideoCapture(vp)
        cache, cur = {}, -1   # cur=-1 让首个目标帧（可能是 0）前先 grab 一次
        for f in want:
            if f < cur:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                cur = f - 1
            ok = True
            while cur < f:
                ok = cap.grab()
                if not ok:
                    break
                cur += 1
            if not ok:
                break
            ok, img = cap.retrieve()
            if ok:
                cache[f] = img
        cap.release()

        n_done = 0
        real_n = {int(r["tid"]): int(r["n_frames"]) for r in items}
        for tid, frames in plan.items():
            cells = []
            for f in frames:
                img = cache.get(f)
                ids, boxes = dets.get(f, ([], []))
                if img is not None and tid in ids:
                    cells.append(tile(img, boxes[ids.index(tid)], f"f{f}"))
                else:
                    cells.append(tile(None, [0, 0, 1, 1], f"f{f}"))
            while len(cells) < NFRAME:
                cells.append(np.full((CELL, CELL, 3), 24, np.uint8))
            grid = np.vstack([np.hstack(cells[i * COLS:(i + 1) * COLS]) for i in range(NFRAME // COLS)])
            hdr = np.full((34, grid.shape[1], 3), 245, np.uint8)
            cv2.putText(hdr, f"{seg} #{tid}   n_frames={real_n.get(tid, len(frames))}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
            out = np.vstack([hdr, grid])
            cv2.imwrite(os.path.join(outd, f"{tid}.jpg"), out,
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            n_done += 1
            if n_done % 25 == 0:
                print(f"  [{seg}] {n_done}/{len(plan)}", flush=True)
        print(f"[done] {seg}: {n_done} 张 -> {outd}", flush=True)


if __name__ == "__main__":
    main()
