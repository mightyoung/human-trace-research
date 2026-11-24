#!/usr/bin/env python
"""为人工判定生成"待定对"证据图。

与 batch_verify 的区别：不按时间均匀采样，而是**按清晰度自动挑选最清晰帧**
（Laplacian 方差），并做轻度锐化增强——用于 A 侧模糊、需人眼细看的疑难对。

用法:
    python tools/crosscam_user_review.py --out-dir .../for_user
"""
import argparse
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


class VR:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))

    def frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if not ok:
            raise RuntimeError(f"read fail @{idx}")
        return fr


def crop_v(frame, box, pad=0.25):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - w * pad)), max(0, int(y1 - h * pad))
    X2, Y2 = min(frame.shape[1], int(x2 + w * pad)), min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2].copy()
    if c.size == 0:
        return None
    return c


def sharpen_mild(c):
    g = cv2.GaussianBlur(c, (0, 0), 2.0)
    return cv2.addWeighted(c, 1.6, g, -0.6, 0)


def fit_h(c, H):
    s = H / c.shape[0]
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), H),
                      interpolation=cv2.INTER_CUBIC)


def sharpest_crops(vr, track, k=4, min_gap=75, enhance=True, H=330):
    """按 Laplacian 方差挑最清晰帧，互相间隔≥min_gap 帧。"""
    scored = []
    for fnum, box in track:
        try:
            fr = vr.frame(fnum)
        except RuntimeError:
            continue
        c = crop_v(fr, box)
        if c is None or c.shape[0] < 24 or c.shape[1] < 14:
            continue
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_64F).var()
        # 归一化清晰度：除以尺寸影响（大目标天然分高）
        score = lap / (c.shape[0] * c.shape[1]) * 1e4
        scored.append((score, fnum, c))
    scored.sort(key=lambda x: -x[0])
    picked = []
    for score, fnum, c in scored:
        if all(abs(fnum - p[0]) >= min_gap for p in picked):
            picked.append((fnum, c, score))
        if len(picked) >= k:
            break
    picked.sort()
    out = []
    for fnum, c, score in picked:
        if enhance:
            c = sharpen_mild(c)
        out.append((fnum, fit_h(c, H)))
    return out


def caption(txt, w, h=30, fs=0.6, color=(25, 25, 25)):
    bar = np.full((h, w, 3), 252, np.uint8)
    cv2.putText(bar, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    return bar


def wall_str(fnum, base="08:55:20"):
    hh, mm, ss = 8, 55, 20
    t = hh * 3600 + mm * 60 + ss + fnum / 25.0
    hh, rem = divmod(int(t), 3600)
    mm, ss = divmod(rem, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def sheet(path, title, blocks):
    """blocks: list of (row_label, [(caption, img), ...])"""
    rows = []
    for rlabel, items in blocks:
        imgs = []
        for cap, im in items:
            imgs.append(np.vstack([caption(cap, im.shape[1]), im]))
        hmax = max(im.shape[0] for im in imgs)
        imgs = [np.pad(im, ((0, hmax - im.shape[0]), (0, 0), (0, 0)),
                       constant_values=255) for im in imgs]
        gap = np.full((hmax, 14, 3), 235, np.uint8)
        row = imgs[0]
        for im in imgs[1:]:
            row = np.hstack([row, gap, im])
        rows.append((rlabel, row))
    wmax = max(r.shape[1] for _, r in rows)
    out = []
    out.append(caption(title, wmax, h=40, fs=0.72))
    for rlabel, row in rows:
        rl = caption(rlabel, wmax, h=30, fs=0.62, color=(120, 30, 30))
        out.append(rl)
        out.append(np.pad(row, ((0, 0), (0, wmax - row.shape[1]), (0, 0)),
                          constant_values=255))
        out.append(np.full((10, wmax, 3), 255, np.uint8))
    cv2.imwrite(str(path), np.vstack(out))
    print(f"[out] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ta = load_tracks(args.a_jsonl)
    tb = load_tracks(args.b_jsonl)
    va, vb = VR(args.a_video), VR(args.b_video)

    def a_blocks(tid, k=3, mlab="A(318)"):
        crops = sharpest_crops(va, ta[tid], k=k)
        return [(f"{mlab} #{tid} f={f} t={wall_str(f)}", im) for f, im in crops]

    def b_blocks(tid, k=3, mlab="B(320)"):
        crops = sharpest_crops(vb, tb[tid], k=k)
        return [(f"{mlab} #{tid} f={f}", im) for f, im in crops]

    # ---- Q1: 53:11 ----
    sheet(out / "Q1_pair_53_11.png",
          "Q1  请判断：A#53 与 B#11 是否同一人？(A 侧此前过小/模糊无法判读)",
          [("A#53（318）最清晰帧：", a_blocks(53)),
           ("B#11（320）最清晰帧（若与 A#37 的确认冲突请指出）：", b_blocks(11))])

    # ---- Q2: glasses man B26/B27 vs A142/A98 ----
    sheet(out / "Q2_glasses_man.png",
          "Q2  请判断：B 侧眼镜男(#26/#27，球机近处俯身) 对应 A 侧哪位？(A 侧模糊)",
          [("B#26 / B#27（320）：", b_blocks(26, 3) + b_blocks(27, 2)),
           ("A#142（318）最清晰帧：", a_blocks(142)),
           ("A#98（318）最清晰帧：", a_blocks(98))])

    # ---- Q3: B4 归属（3:4 vs 14:4）----
    sheet(out / "Q3_B4_owner.png",
          "Q3  请判断：B#4 的俯视后脑属于谁？我的推断：A#14(黑衣)坐下后的延续；备选 A#3(蓝衣)",
          [("B#4（320）f=905-955：", b_blocks(4, 3)),
           ("A#14（318）同时段 f=885-935：", a_blocks(14, 3)),
           ("A#3（318）同时段 f=885-935：", a_blocks(3, 3))])

    # ---- Q4: 37:11 复核 ----
    sheet(out / "Q4_pair_37_11.png",
          "Q4  请复核：A#37 与 B#11 我已判为同一人(★★)，varsity↔blazer 细节请细看",
          [("A#37（318）：", a_blocks(37)),
           ("B#11（320）：", b_blocks(11))])

    # ---- Q5: B19 三对（信息不足，供知情参考）----
    sheet(out / "Q5_B19_topdown.png",
          "Q5  B#19 仅拍到头顶（球机正下方），无法从画面对应到 A 侧——供知情判断",
          [("B#19（320）：", b_blocks(19, 3)),
           ("A#127（318，同步分最高的候选）：", a_blocks(127, 2)),
           ("A#126（318）：", a_blocks(126, 2)),
           ("A#123（318）：", a_blocks(123, 2))])


if __name__ == "__main__":
    main()
