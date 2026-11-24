#!/usr/bin/env python
"""候选对的外观校验：用裁剪图像的 HSV 颜色直方图相似度独立打第二分。

动机（实证）：仅用"运动同步"分无法区分不同外观的人——黑卫衣男与白衣女士
的走停节奏也可能相关（实测 r=0.76 的假阳性）。颜色直方图对尺度/视角
不敏感，可作为独立判别信号。

架构对应：运动同步（候选生成）+ 外观相似（排序过滤）+ 人工确认（终审）。

用法:
    python tools/crosscam_appearance.py \
        --candidates .../crosscam_candidates.json \
        --a-video ... --b-video ... --a-jsonl ... --b-jsonl ... \
        --a-wh 1920 1080 --b-wh 2560 1440 \
        --top 25 --out output_crosscam/analysis/appearance_scores.json
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


def track_span(jsonl_path, tid):
    first, last = None, None
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            if tid in tids:
                if first is None:
                    first = r["frame"]
                last = r["frame"]
    return first, last


class VR:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.cur = -1

    def frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if not ok:
            raise RuntimeError(f"read fail @{idx}")
        return fr


def hist_of(frame, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = frame.shape[:2]
    w, h = x2 - x1, y2 - y1
    # 取中心区域（x 20%-80%，y 5%-65%）：尽量只含躯干衣物，排除背景与地面
    x1c = max(0, x1 + int(w * 0.20))
    x2c = min(W, x1 + int(w * 0.80))
    y1c = max(0, y1 + int(h * 0.05))
    y2c = min(H, y1 + int(h * 0.65))
    if x2c - x1c < 6 or y2c - y1c < 12:
        return None
    crop = frame[y1c:y2c, x1c:x2c]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def nearest_box(boxes, fa, tid, stride=5, max_d=40):
    """在 jsonl 帧网格（stride 倍数）上找距 fa 最近且有该 tid 的框。"""
    base = int(round(fa / stride)) * stride
    best = None
    for d in range(0, max_d + 1, stride):
        for f2 in ({base} if d == 0 else {base - d, base + d}):
            if (f2, tid) in boxes:
                if best is None or abs(f2 - fa) < abs(best[0] - fa):
                    best = (f2, boxes[(f2, tid)])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-wh", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--b-wh", nargs=2, type=int, default=[2560, 1440])
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--sync-min", type=float, default=0.2)
    ap.add_argument("--extra-pairs", default="",
                    help="附加评估的人工确认对，如 '139:24,1:1'（自动列表外的）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cand = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    shift = cand["align"]["shift_frames"]
    pairs = cand["pairs"][: args.top]

    boxes_a = load_boxes(args.a_jsonl)
    boxes_b = load_boxes(args.b_jsonl)
    va = VR(args.a_video)
    vb = VR(args.b_video)

    out = []
    work = [dict(p, _src="auto") for p in pairs
            if p["sync_r"] is not None and p["sync_r"] >= args.sync_min]
    if args.extra_pairs:
        for spec in args.extra_pairs.split(","):
            ta, tb = (int(x) for x in spec.strip().split(":"))
            a0, a1 = track_span(args.a_jsonl, ta)
            b0, b1 = track_span(args.b_jsonl, tb)
            if a0 is None or b0 is None:
                continue
            work.append({"tid_a": ta, "tid_b": tb, "sync_r": None,
                         "inter_sec": round((min(a1, b1 - shift) - max(a0, b0 - shift))
                                            / args.fps, 1),
                         "a_first": a0, "a_last": a1,
                         "b_first": b0, "b_last": b1, "_src": "manual"})

    for p in work:
        ta, tb = p["tid_a"], p["tid_b"]
        # 公共窗口（A 帧坐标：B 帧 - shift）
        a0, a1 = p["a_first"], p["a_last"]
        b0, b1 = p["b_first"] - shift, p["b_last"] - shift
        lo, hi = max(a0, b0), min(a1, b1)
        if hi - lo < args.fps * 1.0:  # 至少 1 秒
            continue
        ts = np.linspace(lo, hi, args.samples + 2)[1:-1]

        sims = []
        for t in ts:
            fa = int(round(t))
            hit_a = nearest_box(boxes_a, fa, ta)
            hit_b = nearest_box(boxes_b, fa + shift, tb)
            if hit_a is None or hit_b is None:
                continue
            fa_use, ba = hit_a
            fb_use, bb = hit_b
            ha = hist_of(va.frame(fa_use), ba)
            hb = hist_of(vb.frame(fb_use), bb)
            if ha is None or hb is None:
                continue
            r = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
            sims.append(float(r))
        out.append({
            "tid_a": ta, "tid_b": tb,
            "sync_r": p["sync_r"],
            "app_r": round(float(np.mean(sims)), 3) if sims else None,
            "app_n": len(sims),
            "inter_sec": p["inter_sec"],
            "src": p.get("_src", "auto"),
        })

    out.sort(key=lambda r: -(r["app_r"] if r["app_r"] is not None else -2))
    print(f"{'pair':>10} {'sync_r':>7} {'app_r':>7} {'n':>3} {'inter_s':>7}")
    for r in out:
        sr = r["sync_r"] if r["sync_r"] is not None else float("nan")
        ar = r["app_r"] if r["app_r"] is not None else float("nan")
        print(f"{r['tid_a']:>4}:{r['tid_b']:<5} {sr:>7} {ar:>7} "
              f"{r['app_n']:>3} {r['inter_sec']:>7}  {r['src']}")

    Path(args.out).write_text(
        json.dumps({"shift": shift, "rows": out}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
