# -*- coding: utf-8 -*-
"""多目标轨迹可视化：单机位多目标追踪渲染（框 + ID + 尾迹 + HUD）。

用法示例：
  python tools/long35_viz.py --cam 318 --tag v10 --t0 20 --t1 45 \
      --out output_crosscam/analysis/long35/viz_318_t20-45.mp4
  python tools/long35_viz.py --cam 316 --tag long35 --t0 900 --t1 930 ...
"""
import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VID = {
    # 10 分钟同步切片
    ("v10", "316"): "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    ("v10", "318"): "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4",
    ("v10", "320"): "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4",
    # 35 分钟长窗
    ("long35", "316"): "media-resource/multi-cemra/long-35min/316_long35.mp4",
    ("long35", "318"): "media-resource/multi-cemra/long-35min/318_long35.mp4",
    ("long35", "320"): "media-resource/multi-cemra/long-35min/320_long35.mp4",
}
DET = {
    "v10": lambda cam: os.path.join(ROOT, "output_crosscam", f"{cam}_detections.jsonl"),
    "long35": lambda cam: os.path.join(ROOT, "output_crosscam", "long35", f"{cam}_long35_detections.jsonl"),
}
T0 = {"316": 4.0, "318": 20.0, "320": 19.0}
FPS = {"316": 24.999, "318": 24.991, "320": 25.003}

# 20 色调色板（BGR），亮色为主，避让常见背景色
PALETTE = [
    (0, 216, 255), (80, 220, 80), (60, 120, 255), (255, 160, 60), (200, 80, 220),
    (255, 255, 100), (100, 255, 200), (255, 100, 140), (160, 200, 255), (140, 255, 60),
    (220, 200, 60), (60, 255, 255), (180, 120, 255), (255, 200, 180), (120, 80, 200),
    (90, 160, 90), (100, 100, 255), (200, 255, 160), (255, 120, 80), (160, 160, 160),
]


def color_of(tid):
    return PALETTE[int(tid) % len(PALETTE)]


def load_det_rows(cam, tag):
    rows = {}
    with open(DET[tag](cam), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows[d["frame"]] = d
    return rows


def render(cam, tag, t0, t1, out, font_path="C:/Windows/Fonts/msyhbd.ttc"):
    fps = FPS[cam]
    video = os.path.join(ROOT, VID[(tag, cam)])
    rows = load_det_rows(cam, tag)
    f0 = int((t0 - T0[cam]) * fps)
    f1 = int((t1 - T0[cam]) * fps)

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OW = 1280
    OH = int(H0 * OW / W0)
    sx, sy = OW / W0, OH / H0

    w4 = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out, w4, np.float32(fps), (OW, OH))

    def sn(t):  # t（相对切片起点秒）→ OSD 时钟串（08:55:00 + t）
        tt = 8 * 3600 + 55 * 60 + t
        return f"{int(tt // 3600):02d}:{int(tt % 3600 // 60):02d}:{int(tt % 60):02d}"

    font_h = ImageFont.truetype(font_path, 30)
    font_s = ImageFont.truetype(font_path, 22)
    tails = {}          # tid -> deque of (cx, cy) in output coords
    last_rows = None
    n_total = f1 - f0
    print(f"[viz] {cam} t{t0}-{t1}s -> frames {f0}..{f1} ({n_total} 帧), out {OW}x{OH}", flush=True)

    for i in range(n_total):
        ok, frame = cap.read()
        if not ok:
            print(f"[viz] video ended at frame {f0+i}", flush=True)
            break
        small = cv2.resize(frame, (OW, OH), interpolation=cv2.INTER_AREA)

        # 找最近的检测行（≤当前帧）
        cur_f = f0 + i
        row = rows.get(cur_f)
        if row is None:
            # 向上找最近的 stride 帧
            for back in range(1, 6):
                r2 = rows.get(cur_f - back)
                if r2 is not None:
                    row = r2
                    break
        if row is not None:
            last_rows = row

        # 画框 + 尾迹
        if last_rows is not None:
            tids = last_rows.get("track_ids", [])
            boxes = last_rows.get("boxes", [])
            for tid, b in zip(tids, boxes):
                if tid is None or int(tid) < 0:
                    continue
                tid = int(tid)
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1 = int(x1 * sx), int(y1 * sy)
                x2, y2 = int(x2 * sx), int(y2 * sy)
                col = color_of(tid)
                cv2.rectangle(small, (x1, y1), (x2, y2), col, 2)
                # 标签
                lab = f"#{tid}"
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                yy = max(th + 6, y1 - 4)
                cv2.rectangle(small, (x1, yy - th - 6), (x1 + tw + 10, yy + 2), col, -1)
                cv2.putText(small, lab, (x1 + 5, yy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 15, 15), 2)
                # 尾迹（脚点）
                dq = tails.setdefault(tid, deque(maxlen=30))
                cx, cy = (x1 + x2) // 2, y2
                dq.append((cx, cy))
                pts = list(dq)
                for j in range(1, len(pts)):
                    a = int(255 * j / len(pts))
                    cv2.line(small, pts[j - 1], pts[j], col, 2)

        # HUD（PIL 中文）
        img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pim = Image.fromarray(img)
        d = ImageDraw.Draw(pim, "RGBA")
        t_cur = T0[cam] + (f0 + i) / fps
        n_act = last_rows["count"] if last_rows else 0
        d.rectangle([0, 0, OW, 44], fill=(12, 14, 18, 190))
        d.text((14, 7), f"{cam}  ·  2026-10-10 {sn(t_cur)}", font=font_h, fill=(255, 255, 255, 255))
        d.text((OW - 260, 7), f"同时目标: {n_act} 人", font=font_h,
               fill=(0, 230, 255, 255) if n_act >= 3 else (200, 205, 215, 255))
        d.text((14, OH - 34), f"多目标轨迹追踪 · {tag} · frame {f0+i}", font=font_s,
               fill=(190, 195, 205, 230))
        small = cv2.cvtColor(np.asarray(pim), cv2.COLOR_RGB2BGR)
        vw.write(small)

        if (i + 1) % 100 == 0:
            print(f"  [viz] {i+1}/{n_total}", flush=True)

    cap.release()
    vw.release()
    print(f"[viz] done -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True, choices=["316", "318", "320"])
    ap.add_argument("--tag", default="long35", choices=["v10", "long35"])
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    render(args.cam, args.tag, args.t0, args.t1, args.out)


if __name__ == "__main__":
    main()
