#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器匹配对目视渲染：把 top 跨集匹配（或任意对）渲染为并排对比图。

输入：matches csv（selector_eval 产出：rank, query_seg, query_tid, match_seg, match_tid, sim）
输出：output_crosscam/analysis/selector_review/pair_rXX.jpg（每对一图）+ montage 大图

用法：python tools/render_match_pairs.py --matches <csv> --top 20 --out-dir <dir>
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "output_crosscam", "1012_final")
SEG_DIR = os.path.join(ROOT, "media-resource", "1012", "segments")
RES = {"305": (2560, 1440), "320": (2560, 1440), "318": (1920, 1080), "323": (1920, 1080)}


def load_track(jsonl_path, tid):
    """返回该 tid 的 [(frame, box)]"""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for t, b in zip(r.get("track_ids") or [], r.get("boxes") or []):
                if t is not None and int(t) == tid:
                    rows.append((int(r["frame"]), [float(x) for x in b]))
    rows.sort(key=lambda x: x[0])
    return rows


_cache = {}


def get_frame(seg, frame_no):
    cap = _cache.get(seg)
    if cap is None:
        vp = os.path.join(SEG_DIR, f"{seg}.mp4")
        cap = cv2.VideoCapture(vp)
        _cache[seg] = cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, fr = cap.read()
    return fr if ok else None


def render_person_crop(seg, tid, out_h=460):
    """抽 tid 中点帧，裁剪 2.2 倍框范围，缩放至 out_h 高。"""
    cam = seg.split("_")[0]
    jp = os.path.join(FINAL, f"{seg}.jsonl")
    rows = load_track(jp, tid)
    if not rows:
        return None, f"{seg}#{tid}: no rows"
    mid = rows[len(rows) // 2]
    fr = get_frame(seg, mid[0])
    if fr is None:
        return None, f"{seg}#{tid}: frame read fail"
    x1, y1, x2, y2 = [int(v) for v in mid[1]]
    w, h = x2 - x1, y2 - y1
    mx = int(w * 0.9) + 20
    my = int(h * 0.35) + 20
    X0, Y0 = max(0, x1 - mx), max(0, y1 - my)
    X1, Y1 = min(fr.shape[1], x2 + mx), min(fr.shape[0], y2 + my)
    crop = fr[Y0:Y1, X0:X1].copy()
    cv2.rectangle(crop, (x1 - X0, y1 - Y0), (x2 - X0, y2 - Y0), (0, 0, 255), 3)
    sc = out_h / crop.shape[0]
    z = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4)
    return z, None


def label_bar(img, text, h=40):
    bar = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (255, 255, 255), 2)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--montage-per", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    with open(a.matches, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows = rows[: a.top]
    made = []
    for r in rows:
        qs, qt = r["query_seg"], int(r["query_tid"])
        ms, mt = r["match_seg"], int(r["match_tid"])
        sim = float(r["sim"])
        za, err1 = render_person_crop(qs, qt)
        zb, err2 = render_person_crop(ms, mt)
        if za is None or zb is None:
            print(f"[skip] {qs}#{qt} -> {ms}#{mt}: {err1 or err2}")
            continue
        H = max(za.shape[0], zb.shape[0])
        if za.shape[0] != H:
            za = cv2.copyMakeBorder(za, 0, H - za.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        if zb.shape[0] != H:
            zb = cv2.copyMakeBorder(zb, 0, H - zb.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        gap = np.full((H, 14, 3), 60, dtype=np.uint8)
        pair = np.hstack([za, gap, zb])
        pair = label_bar(pair, f"#{r['rank']} sim={sim:.3f}   Q:{qs}#{qt}  ->  M:{ms}#{mt}")
        p = os.path.join(a.out_dir, f"pair_r{int(r['rank']):02d}.jpg")
        cv2.imwrite(p, pair)
        made.append(pair)
        print(f"[pair] {p}  sim={sim:.4f}  {qs}#{qt} -> {ms}#{mt}")

    # montage
    per = a.montage_per
    for bi in range(0, len(made), per):
        group = made[bi:bi + per]
        W = max(g.shape[1] for g in group)
        H = sum(g.shape[0] for g in group) + 6 * (len(group) - 1)
        canvas = np.full((H, W, 3), 15, dtype=np.uint8)
        y = 0
        for g in group:
            canvas[y:y + g.shape[0], :g.shape[1]] = g
            y += g.shape[0] + 6
        mp = os.path.join(a.out_dir, f"montage_{bi // per + 1}.jpg")
        cv2.imwrite(mp, canvas)
        print(f"[montage] {mp} ({len(group)} pairs)")


if __name__ == "__main__":
    main()
