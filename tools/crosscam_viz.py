#!/usr/bin/env python
"""确认对应关系可视化：生成"视野交叠图"（交付物）。

上排：318 / 320 参考帧，叠加各确认轨迹对的落脚点路径（按对配色）
下排：落脚点映射散点（A 视图 -> B 视图，同一配色）
用法:
    python tools/crosscam_viz.py \
        --pairs output_crosscam/analysis/confirmed_pairs.csv \
        --a-video ... --b-video ... --out-dir ... [--frame-t 1.0]
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PAIR_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00b4d8",
               "#f032e6", "#008080"]
PAIR_LABELS = {
    "1:1": "cleaner  A#1~B#1  (confirmed)",
    "14:3": "man black tee+glasses  A#14~B#3  (confirmed)",
    "69:14": "man olive jacket+backpack  A#69~B#14  (confirmed)",
    "120:18": "woman navy+leather collar  A#120~B#18  (confirmed)",
    "120:23": "  same woman (2nd frag)  A#120~B#23  (confirmed)",
    "139:24": "man black hoodie  A#139~B#24  (confirmed)",
    "146:28": "woman cap+floral jacket  A#146~B#28  (confirmed)",
    "37:11": "man navy jacket  A#37~B#11  (confirmed)",
}


def read_frame(video, t_sec):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {t_sec}s from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-t", type=float, default=1.0)
    ap.add_argument("--out-name", default="fig_view_overlap.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.pairs, encoding="utf-8")))
    if not rows:
        raise SystemExit("empty pairs csv")

    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)
    for v in by_pair.values():
        v.sort(key=lambda r: float(r["t_sec"]))

    fa = read_frame(args.a_video, args.frame_t)
    fb = read_frame(args.b_video, args.frame_t)
    ha, wa = fa.shape[:2]
    hb, wb = fb.shape[:2]

    fig = plt.figure(figsize=(17, 10.5), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 2.0], hspace=0.15, wspace=0.04)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_a.imshow(fa)
    ax_b.imshow(fb)

    for k, (pair, prows) in enumerate(by_pair.items()):
        color = PAIR_COLORS[k % len(PAIR_COLORS)]
        for ax, un, vn, w, h in [(ax_a, "u_a", "v_a", wa, ha),
                                 (ax_b, "u_b", "v_b", wb, hb)]:
            us = np.array([float(r[un]) for r in prows]) * w
            vs = np.array([float(r[vn]) for r in prows]) * h
            ax.plot(us, vs, "-", color=color, lw=2.4, alpha=0.9, zorder=4)
            ax.scatter(us, vs, color=color, s=34, zorder=5,
                       edgecolors="white", linewidths=0.9)

    ax_a.set_title("A: 318 办公区2  (markers = confirmed foot points)", fontsize=12)
    ax_b.set_title("B: 320 办公区球机", fontsize=12)
    for ax, w, h in [(ax_a, wa, ha), (ax_b, wb, hb)]:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

    handles = [mlines.Line2D([], [], color=PAIR_COLORS[k % len(PAIR_COLORS)],
                             lw=2.6, marker="o", markersize=5,
                             label=PAIR_LABELS.get(p, p))
               for k, p in enumerate(by_pair.keys())]
    ax_a.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.85)

    # 下排：映射散点
    ax_m1 = fig.add_subplot(gs[1, 0])
    ax_m2 = fig.add_subplot(gs[1, 1])
    for k, (pair, prows) in enumerate(by_pair.items()):
        color = PAIR_COLORS[k % len(PAIR_COLORS)]
        ua = np.array([float(r["u_a"]) for r in prows])
        va = np.array([float(r["v_a"]) for r in prows])
        ub = np.array([float(r["u_b"]) for r in prows])
        vb = np.array([float(r["v_b"]) for r in prows])
        ax_m1.scatter(ua, va, color=color, s=28, edgecolors="k", linewidths=0.3)
        ax_m2.scatter(ub, vb, color=color, s=28, edgecolors="k", linewidths=0.3)
    ax_m1.set_title("A-view foot positions (normalized)")
    ax_m2.set_title("B-view foot positions (normalized)")
    for ax in (ax_m1, ax_m2):
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.grid(alpha=0.3)

    title = args.title or ("Cross-camera confirmed correspondences "
                           "(318 x 320, shift=+20f/+0.8s)")
    fig.suptitle(title, fontsize=13.5, y=0.965)
    out = out_dir / args.out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
