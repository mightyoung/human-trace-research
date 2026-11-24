#!/usr/bin/env python
"""跨画面候选对的视觉核验：裁出两路对应目标，拼图供人工确认。

对 candidates 中同步分最高的若干对，各自取 3 个时间点（前/中/后），
从两路视频裁出对应目标框并排拼图，一眼可判"是不是同一个人"。

用法:
    python tools/crosscam_verify.py \
        --a-video ... --b-video ... \
        --a-jsonl ... --b-jsonl ... \
        --pairs .../correspondence_pairs.csv \
        --candidates .../crosscam_candidates.json \
        --out-dir ... --top 4
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_boxes(jsonl_path):
    """(frame, tid) -> box"""
    m = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    m[(r["frame"], int(tids[i]))] = b
    return m


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.cur = -1

    def frame(self, idx):
        if idx != self.cur + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        self.cur = idx
        if not ok:
            raise RuntimeError(f"cannot read frame {idx}")
        return fr


def crop(frame, box, pad=0.25, min_h=220):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    px = w * pad
    py = h * pad
    H, W = frame.shape[:2]
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(W, int(x2 + px))
    y2 = min(H, int(y2 + py))
    c = frame[y1:y2, x1:x2].copy()
    # 统一到相同高度便于并排
    scale = min_h / c.shape[0]
    c = cv2.resize(c, (max(1, int(c.shape[1] * scale)), min_h),
                   interpolation=cv2.INTER_LINEAR)
    return c


def put_caption(img, text, color=(20, 20, 20)):
    """白底黑字 caption 条压在图片底部。"""
    bar = 30
    out = cv2.copyMakeBorder(img, 0, bar, 0, 0, cv2.BORDER_CONSTANT,
                             value=(245, 245, 245))
    cv2.putText(out, text, (6, img.shape[0] + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    shift = cand["align"]["shift_frames"]
    fps = cand["fps"]

    boxes_a = load_boxes(args.a_jsonl)
    boxes_b = load_boxes(args.b_jsonl)

    rows = list(csv.DictReader(open(args.pairs, encoding="utf-8")))
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)

    # 按 sync_r 排序，取前 top 对
    ranked = sorted(by_pair.items(),
                    key=lambda kv: -float(kv[1][0]["sync_r"]))[: args.top]

    va = VideoReader(args.a_video)
    vb = VideoReader(args.b_video)

    for pair, prows in ranked:
        tid_a, tid_b = (int(x) for x in pair.split("-"))
        prows = sorted(prows, key=lambda r: float(r["t_sec"]))
        n = len(prows)
        idxs = ([0, n // 2, n - 1] if args.samples == 3 else
                [int(i * (n - 1) / (args.samples - 1)) for i in range(args.samples)])
        idxs = sorted(set(idxs))

        tiles = []
        for i in idxs:
            r = prows[i]
            fa = int(float(r["t_sec"]) * fps)
            fb = fa + shift
            # 找最近的实际框
            ba = None
            for d in range(0, 15, 5):
                for f2 in ([fa] if d == 0 else [fa - d, fa + d]):
                    if (f2, tid_a) in boxes_a:
                        ba = boxes_a[(f2, tid_a)]
                        fa_use = f2
                        break
                if ba:
                    break
            bb = None
            for d in range(0, 15, 5):
                for f2 in ([fb] if d == 0 else [fb - d, fb + d]):
                    if (f2, tid_b) in boxes_b:
                        bb = boxes_b[(f2, tid_b)]
                        fb_use = f2
                        break
                if bb:
                    break
            if ba is None or bb is None:
                print(f"[warn] pair {pair}: box missing near f={fa}")
                continue
            ca = crop(va.frame(fa_use), ba)
            cb = crop(vb.frame(fb_use), bb)
            wall = 8 * 3600 + 55 * 60 + 20 + float(r["t_sec"])  # A 路起点 08:55:20
            hh, rem = divmod(int(wall), 3600)
            mm, ss = divmod(rem, 60)
            cap_a = f"A(318) f={fa_use} sync={r['sync_r']} t={hh:02d}:{mm:02d}:{ss:02d}"
            cap_b = f"B(320) f={fb_use}"
            ca = put_caption(ca, cap_a)
            cb = put_caption(cb, cap_b)
            gap = np.full((ca.shape[0], 12, 3), 255, np.uint8)
            tiles.append(np.hstack([ca, gap, cb]))

        if not tiles:
            continue
        maxw = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, maxw - t.shape[1]), (0, 0)),
                        constant_values=255) for t in tiles]
        vgap = np.full((10, maxw, 3), 255, np.uint8)
        body = tiles[0]
        for t in tiles[1:]:
            body = np.vstack([body, vgap, t])
        head = np.full((44, maxw, 3), 255, np.uint8)
        cv2.putText(head, f"pair A(318)tid={tid_a}  B(320)tid={tid_b}  "
                          f"sync_r={prows[0]['sync_r']}  "
                          f"overlap={len(prows)*0.4:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (30, 30, 30), 2, cv2.LINE_AA)
        out = np.vstack([head, body])
        p = out_dir / f"verify_pair_{tid_a}_{tid_b}.png"
        cv2.imwrite(str(p), out)
        print(f"[out] {p}  ({len(tiles)} samples)")


if __name__ == "__main__":
    main()
