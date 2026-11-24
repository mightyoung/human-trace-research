#!/usr/bin/env python
"""P2 拼接候选对证据图生成（远程运行，输入为视频+cleaned jsonl）。

对每个候选对 (A->B)：取 A 尾段 3 帧人框 + B 头段 3 帧人框，
裁剪放大后拼接成一行证据块；每页若干对，输出多张 PNG。

用法:
  python tools/stitch_evidence.py --video video-6way/320.mp4 \
      --jsonl stitch_in/320_cleaned.jsonl --wh 2560 1440 \
      --csv stitch_out/stitch_320.csv \
      --links stitch_out/stitch_320_groups.json \
      --out-prefix stitch_out/evid_320
  # 手动指定对:
  ... --pairs "18:23,3:50"
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_tracks(jsonl_path):
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((r["frame"], b))
    for k in t:
        t[k].sort()
    return t


def crop_person(frame, box, crop_h, pad=0.35, max_w=210):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad))
    Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad))
    Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0:
        return None
    s = crop_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, crop_h))
    if nw > max_w:
        off = (nw - max_w) // 2
        c = c[:, off:off + max_w]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--wh", nargs=2, type=int, required=True)
    ap.add_argument("--csv", required=True, help="stitch 全量候选 csv")
    ap.add_argument("--links", default="", help="groups.json（画 links 里的对）")
    ap.add_argument("--pairs", default="", help="手动指定 如 '18:23,3:50'")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--per-page", type=int, default=12)
    ap.add_argument("--crop-h", type=int, default=220)
    args = ap.parse_args()

    # 读全量候选
    rows = {}
    with open(args.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(int(r["tid_a"]), int(r["tid_b"]))] = r

    # 选择要画的 pairs
    pairs = []
    if args.pairs:
        for tok in args.pairs.split(","):
            a, b = tok.strip().split(":")
            pairs.append((int(a), int(b)))
    elif args.links:
        lj = json.loads(Path(args.links).read_text(encoding="utf-8"))
        for lk in lj.get("links", []):
            pairs.append((lk["tid_a"], lk["tid_b"]))
    if not pairs:
        print("no pairs to draw")
        return
    print(f"[evid] drawing {len(pairs)} pairs")

    tracks = load_tracks(args.jsonl)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")

    frame_cache = {}

    def get_frame(idx):
        idx = max(0, idx)
        if idx not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                return None
            # 控制缓存规模
            if len(frame_cache) > 60:
                frame_cache.clear()
            frame_cache[idx] = fr
        return frame_cache[idx]

    PAGE_W = 1760
    TEXT_W = 430
    ROW_H = args.crop_h + 46
    PAGE_H = ROW_H * args.per_page + 90

    def draw_row(canvas, y, row, ta, tb):
        obs_a = tracks.get(ta, [])
        obs_b = tracks.get(tb, [])
        if not obs_a or not obs_b:
            return
        # A 尾段 3 帧、B 头段 3 帧（按观测序号取点）
        def pick(obs, tail):
            n = len(obs)
            step = max(1, n // 12)
            if tail:
                idxs = [max(0, n - 1 - 2 * step), max(0, n - 1 - step), n - 1]
            else:
                idxs = [0, min(n - 1, step), min(n - 1, 2 * step)]
            return [obs[i] for i in sorted(set(idxs))]

        a_obs = pick(obs_a, True)
        b_obs = pick(obs_b, False)
        crops = []
        for fr_i, fr_box in a_obs:
            fr = get_frame(fr_i)
            if fr is None:
                continue
            c = crop_person(fr, fr_box, args.crop_h)
            if c is not None:
                crops.append((c, (90, 160, 255)))  # A: 橙色边 (BGR)
        for fr_i, fr_box in b_obs:
            fr = get_frame(fr_i)
            if fr is None:
                continue
            c = crop_person(fr, fr_box, args.crop_h)
            if c is not None:
                crops.append((c, (80, 220, 80)))  # B: 绿色边
        if not crops:
            return
        # 左侧文本
        ty = y + 26
        lines = [
            f"[{row['tid_a']}] -> [{row['tid_b']}]",
            f"gap {row['gap_s']}s  cos {row['cos']}  pe {row['pe_adj']}",
            f"dist {row['dist_raw']}  scale {row['scale']}",
            f"n {row['n_a']}/{row['n_b']}  still {row['still']} "
            f"edge {row['edge_a']}{row['edge_b']}",
            f"f {row['f1a']}(e) -> {row['f0b']}(s)  25fps",
            "A=orange(3 tail)  B=green(3 head)",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(canvas, ln, (14, ty + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (235, 235, 235), 2, cv2.LINE_AA)
        # 右侧 crop
        x = TEXT_W
        for c, color in crops:
            h, w = c.shape[:2]
            if x + w > PAGE_W:
                break
            canvas[y:y + h, x:x + w] = c
            cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), color, 3)
            x += w + 8
        # 分隔线
        cv2.line(canvas, (0, y + ROW_H - 8), (PAGE_W, y + ROW_H - 8),
                 (60, 60, 60), 1)

    # 分页绘制
    page = 0
    out_files = []
    for pi in range(0, len(pairs), args.per_page):
        chunk = pairs[pi:pi + args.per_page]
        canvas = np.full((PAGE_H, PAGE_W, 3), 24, dtype=np.uint8)
        cv2.putText(canvas,
                    f"P2 stitch candidates (page {page + 1}) "
                    f"- pairs {pi + 1}..{pi + len(chunk)} of {len(pairs)}",
                    (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (120, 220, 255), 2, cv2.LINE_AA)
        y = 70
        for a, b in chunk:
            row = rows.get((a, b))
            if row is None:
                # 尝试反向
                row = rows.get((b, a))
                if row is not None:
                    a, b = b, a
            if row is None:
                continue
            draw_row(canvas, y, row, a, b)
            y += ROW_H
        fn = f"{args.out_prefix}_p{page + 1:02d}.png"
        cv2.imwrite(fn, canvas)
        out_files.append(fn)
        print(f"[evid] saved {fn}")
        page += 1
    print("[evid] DONE", len(out_files), "pages")


if __name__ == "__main__":
    main()
