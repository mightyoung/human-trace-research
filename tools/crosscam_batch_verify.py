#!/usr/bin/env python
"""批量核验指定候选对：每对生成一张裁剪对照图（供人工确认）。

与 crosscam_verify.py 的区别：不依赖自动 CSV，直接给轨迹对列表，
公共窗口与采样点由 jsonl 轨迹跨度计算。

用法:
    python tools/crosscam_batch_verify.py \
        --candidates .../crosscam_candidates.json \
        --a-video ... --b-video ... --a-jsonl ... --b-jsonl ... \
        --pairs "146:28,127:19,61:14" \
        --out-dir .../verify_batch --samples 3
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_boxes(jsonl_path):
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


class VR:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {path}")

    def frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if not ok:
            raise RuntimeError(f"read fail @{idx}")
        return fr


def nearest_box(boxes, fa, tid, stride=5, max_d=45):
    base = int(round(fa / stride)) * stride
    best = None
    for d in range(0, max_d + 1, stride):
        for f2 in ({base} if d == 0 else {base - d, base + d}):
            if (f2, tid) in boxes:
                if best is None or abs(f2 - fa) < abs(best[0] - fa):
                    best = (f2, boxes[(f2, tid)])
    return best


def crop(frame, box, H=260, pad=0.3):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - w * pad)), max(0, int(y1 - h * pad))
    X2, Y2 = min(frame.shape[1], int(x2 + w * pad)), min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2].copy()
    if c.size == 0:
        return np.full((H, H, 3), 200, np.uint8)
    s = H / c.shape[0]
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), H))


def tile_caption(text, img):
    bar = 26
    out = cv2.copyMakeBorder(img, 0, bar, 0, 0, cv2.BORDER_CONSTANT,
                             value=(248, 248, 248))
    cv2.putText(out, text, (5, img.shape[0] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--pairs", required=True, help="如 '146:28,127:19'")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    shift = cand["align"]["shift_frames"]
    sync_map = {(p["tid_a"], p["tid_b"]): p["sync_r"] for p in cand["pairs"]}

    boxes_a = load_boxes(args.a_jsonl)
    boxes_b = load_boxes(args.b_jsonl)
    va, vb = VR(args.a_video), VR(args.b_video)

    def spans(track_map, tid):
        fr = [k[0] for k in track_map if k[1] == tid]
        return (min(fr), max(fr)) if fr else (None, None)

    for spec in args.pairs.split(","):
        spec = spec.strip()
        if not spec:
            continue
        ta, tb = (int(x) for x in spec.split(":"))
        a0, a1 = spans(boxes_a, ta)
        b0, b1 = spans(boxes_b, tb)
        if a0 is None or b0 is None:
            print(f"[skip] {spec}: track missing")
            continue
        lo, hi = max(a0, b0 - shift), min(a1, b1 - shift)
        if hi - lo < args.fps * 1.0:
            print(f"[skip] {spec}: window {hi-lo:.0f}f too small")
            continue
        ts = np.linspace(lo, hi, args.samples + 2)[1:-1]

        rows = []
        for t in ts:
            fa = int(round(t))
            ha = nearest_box(boxes_a, fa, ta)
            hb = nearest_box(boxes_b, fa + shift, tb)
            if ha is None or hb is None:
                continue
            fa_use, ba = ha
            fb_use, bb = hb
            ca = crop(va.frame(fa_use), ba)
            cb = crop(vb.frame(fb_use), bb)
            wall = 8 * 3600 + 55 * 60 + 20 + fa_use / args.fps
            hh, rem = divmod(int(wall), 3600)
            mm, ss = divmod(rem, 60)
            ca = tile_caption(f"A(318) f={fa_use} t={hh:02d}:{mm:02d}:{ss:02d}", ca)
            cb = tile_caption(f"B(320) f={fb_use}", cb)
            gap = np.full((ca.shape[0], 12, 3), 255, np.uint8)
            rows.append(np.hstack([ca, gap, cb]))
        if not rows:
            print(f"[skip] {spec}: no samples")
            continue
        maxw = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0)),
                       constant_values=255) for r in rows]
        vg = np.full((8, maxw, 3), 255, np.uint8)
        body = rows[0]
        for r in rows[1:]:
            body = np.vstack([body, vg, r])
        head = np.full((38, maxw, 3), 255, np.uint8)
        sr = sync_map.get((ta, tb))
        cv2.putText(head, f"pair A#{ta} x B#{tb}   sync_r={sr}   "
                          f"window={(hi-lo)/args.fps:.1f}s   shift={shift}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 1,
                    cv2.LINE_AA)
        out = np.vstack([head, body])
        p = out_dir / f"batch_{ta}_{tb}.png"
        cv2.imwrite(str(p), out)
        print(f"[out] {p}  ({len(rows)} samples)")


if __name__ == "__main__":
    main()
