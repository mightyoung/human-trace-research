#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cradio_v3 跨机识别结果 → 可视化拼接视频（supervision + PIL HUD）。

场景 A「白帽人 4 机位全链路」：
  343 电梯间 → 316 行政走廊 → 318 办公区2 → 320 球机
  cradio_v3（C-RADIOv4 × EmbeddingGemma2 桥接，干净三集协议）判定的匹配：
    343-E046 → 316-E105   sim=0.9629   (Δt 1.2s)
    316-E105 → 318-E099   sim=0.9755   (Δt 1.2s)
    318#146  ↔ 320#28     sim=0.9482   (同屏)
  碎片 tid 对照：343-E046→#57, 316-E105→#144, 318-E099→#146；320#28 为同屏对 B 侧。
  来源：analysis/relay/relay_chains_cradio_v3.json + cradio_scale_v3 npz。

用法：
  python tools/cradio_v3_viz.py --scene a --smoke     # 抽关键帧 PNG 检查
  python tools/cradio_v3_viz.py --scene a             # 全量渲染 mp4
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import supervision as sv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "viz")
FPS = 25.0
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
BASE_SEC = 8 * 3600 + 55 * 60  # 墙钟 08:55:00 = t 0

VID = {
    "343": os.path.join(ROOT, "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4"),
    "316": os.path.join(ROOT, "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4"),
    "318": os.path.join(ROOT, "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4"),
    "320": os.path.join(ROOT, "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4"),
}
SRC_RES = {"343": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080), "320": (2560, 1440)}

CANVAS = (1920, 1080)
CELL = (960, 540)
GRID = {"343": (0, 0), "316": (1, 0), "318": (0, 1), "320": (1, 1)}  # (col,row)
CELL_NAME = {"343": "电梯间", "316": "行政走廊", "318": "办公区2", "320": "球机"}

GOLD = (0, 220, 255)      # BGR 亮青金（框）
GOLD_HEX = "#00DCFF"
GRAY_HEX = "#B8B8B8"

FONT_BD = "C:/Windows/Fonts/msyhbd.ttc"
FONT_RG = "C:/Windows/Fonts/msyh.ttc"

# ---------------- 场景 A 定义 ----------------
SCENE_A = dict(
    t0=580.6, t1=608.4,
    order=["343", "316", "318", "320"],
    star={
        "343": dict(tid=57,  t_first=582.4, t_last=589.0),
        "316": dict(tid=144, t_first=590.2, t_last=596.8),
        "318": dict(tid=146, t_first=598.0, t_last=607.2),
        "320": dict(tid=28,  t_first=604.0, t_last=615.0),
    },
    events=[
        dict(t0=589.4, t1=592.8, a="343", b="316", text="343 → 316  ·  sim=0.963  ·  Δt=1.2s"),
        dict(t0=597.2, t1=600.6, a="316", b="318", text="316 → 318  ·  sim=0.976  ·  Δt=1.2s"),
        dict(t0=604.2, t1=607.8, a="318", b="320", text="318 ↔ 320 同屏  ·  sim=0.948"),
    ],
    title="C-RADIOv4 × EmbeddingGemma 2 桥接（v3）· 跨机 ReID 接力识别",
    subject="G1 · 白帽人（白帽 + 白色上衣 + 黑裤）",
    subj2="身份判定：cradio_v3 · sim 0.948–0.976 · 人工复核 evid_chain 系列",
    output=os.path.join(OUT_DIR, "cradio_v3_sceneA_whitehat_4cam.mp4"),
)


def wall_str(t):
    s = BASE_SEC + t
    return f"{int(s)//3600:02d}:{int(s)//60%60:02d}:{int(s)%60:02d}"


def load_det(cam):
    """检测索引 {frame: (tids, boxes, scores)}"""
    idx = {}
    with open(os.path.join(ROOT, "output_crosscam", f"{cam}_detections.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            idx[d["frame"]] = (d["track_ids"],
                               [tuple(float(v) for v in b) for b in d["boxes"]],
                               d["scores"])
    return idx


def star_series(cam, tid, det):
    """该 tid 的 {frame: box}"""
    ser = {}
    for fr, (tids, boxes, _) in det.items():
        if tid in tids:
            ser[fr] = boxes[tids.index(tid)]
    return dict(sorted(ser.items()))


def box_at(ser, frames_sorted, f, max_gap=14):
    """f 帧处该轨道的框（hold 到下一个检测帧，间隔>max_gap 视为消失）。"""
    import bisect
    i = bisect.bisect_right(frames_sorted, f) - 1
    if i < 0:
        fr = frames_sorted[0]
        return ser[fr] if fr - f <= max_gap else None
    fr = frames_sorted[i]
    if f - fr > max_gap:
        return None
    return ser[fr]


# ---------------- PIL 绘制工具 ----------------
def capsule(d, xy, text, font, fg=(255, 255, 255, 255), bg=(0, 0, 0, 165),
            radius=9, pad=(12, 7)):
    x, y = xy
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    w, h = (r - l) + pad[0] * 2, (b - t) + pad[1] * 2
    d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=bg)
    d.text((x + pad[0] - l, y + pad[1] - t), text, font=font, fill=fg)
    return x + w, y + h


def draw_arrow_line(overlay, p0, p1, t_phase, color=(0, 220, 255, 235), width=5):
    """两点连线 + 流动光点 + 箭头头部"""
    d = ImageDraw.Draw(overlay)
    d.line([p0, p1], fill=color, width=width)
    # 箭头头部
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    L = max((vx * vx + vy * vy) ** 0.5, 1e-3)
    ux, uy = vx / L, vy / L
    px, py = -uy, ux
    ah, aw = 26, 14
    tip = p1
    b1 = (p1[0] - ux * ah + px * aw, p1[1] - uy * ah + py * aw)
    b2 = (p1[0] - ux * ah - px * aw, p1[1] - uy * ah - py * aw)
    d.polygon([tip, b1, b2], fill=color)
    # 流动光点
    for k in range(3):
        frac = (t_phase + k / 3.0) % 1.0
        cx = p0[0] + vx * frac
        cy = p0[1] + vy * frac
        r = 10
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 245))
        d.ellipse([cx - r - 4, cy - r - 4, cx + r + 4, cy + r + 4], outline=(0, 220, 255, 150), width=3)


# ---------------- 场景 A 渲染 ----------------
def render_scene_a(smoke=False):
    sc = SCENE_A
    os.makedirs(OUT_DIR, exist_ok=True)
    cams = sc["order"]
    dets = {c: load_det(c) for c in cams}
    stars = {}
    for c in cams:
        tid = sc["star"][c]["tid"]
        ser = star_series(c, tid, dets[c])
        stars[c] = dict(ser=ser, frames=sorted(ser.keys()), tid=tid)

    f0_all = {c: int(round((sc["t0"] - T0[c]) * FPS)) for c in cams}
    n_frames = int(round((sc["t1"] - sc["t0"]) * FPS))

    fonts = {sz: ImageFont.truetype(FONT_BD, sz) for sz in (30, 24, 21, 19, 17, 16)}
    font_rg = {sz: ImageFont.truetype(FONT_RG, sz) for sz in (21, 19, 17, 16, 15)}

    box_plain = sv.BoxAnnotator(thickness=2, color=sv.Color.from_hex(GRAY_HEX),
                                color_lookup=sv.ColorLookup.INDEX)
    box_star = sv.BoxAnnotator(thickness=5, color=sv.Color.from_hex(GOLD_HEX),
                               color_lookup=sv.ColorLookup.INDEX)

    if smoke:
        writer = None
        smoke_ts = [583.0, 590.8, 595.0, 599.0, 605.8, 610.5]
        smoke_set = {int(round((t - sc["t0"]) * FPS)) for t in smoke_ts}
    else:
        writer = cv2.VideoWriter(sc["output"], cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, CANVAS)
        assert writer.isOpened(), "VideoWriter 打开失败"
        smoke_set = set()

    caps = {}
    for c in cams:
        cap = cv2.VideoCapture(VID[c])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0_all[c])
        caps[c] = cap

    frozen_cache = {}   # cam -> 最后可用 small 帧（画框后）
    frozen_now = set()  # 本帧处于冻结的 cam

    wall0 = time.time()
    for i in range(n_frames):
        t = sc["t0"] + i / FPS
        f_cam = {c: f0_all[c] + i for c in cams}

        # --- 读帧 + 画框 ---
        cells = {}
        cell_labels = {}
        star_info = {}
        frozen_now = set()
        for c in cams:
            ok, frame = caps[c].read()
            if not ok:
                # 录像片段比同步窗短：冻结最后可用帧（316 仅 597s）
                if c in frozen_cache:
                    cells[c] = frozen_cache[c].copy()
                    cell_labels[c] = []
                    frozen_now.add(c)
                    sb = box_at(stars[c]["ser"], stars[c]["frames"], f_cam[c])
                    star_info[c] = sb
                    continue
                raise RuntimeError(f"{c} 帧读取失败 @ {f_cam[c]}（无可用冻结帧）")
            W, H = SRC_RES[c]
            sx, sy = CELL[0] / W, CELL[1] / H
            small = cv2.resize(frame, CELL, interpolation=cv2.INTER_AREA)

            # 本帧检测（hold 到 stride 检测帧）
            det_fr = f_cam[c] - (f_cam[c] % 5)
            tids, boxes, scores = dets[c].get(det_fr, ([], [], []))
            # 主角（独立序列，带 gap 判断）
            sb = box_at(stars[c]["ser"], stars[c]["frames"], f_cam[c])
            star_info[c] = sb

            # 普通框（排除主角） + 标签收集（PIL 统一绘制，避免被顶部条遮挡）
            p_xy, labels_cell = [], []
            for tid, b in zip(tids, boxes):
                if tid == sc["star"][c]["tid"]:
                    continue
                x1, y1, x2, y2 = b
                p_xy.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                labels_cell.append((x1 * sx + 4, y1 * sy + 4, f"#{tid}", "plain"))
            if p_xy:
                det_p = sv.Detections(xyxy=np.array(p_xy, dtype=np.float32))
                small = box_plain.annotate(small, det_p)

            # 主角框
            if sb is not None:
                x1, y1, x2, y2 = sb
                st = sv.Detections(xyxy=np.array([[x1 * sx, y1 * sy, x2 * sx, y2 * sy]], dtype=np.float32))
                small = box_star.annotate(small, st)
                labels_cell.append((x1 * sx + 5, y1 * sy + 5, "G1 √ cradio_v3", "star"))

            cells[c] = small
            cell_labels[c] = labels_cell
            frozen_cache[c] = small.copy()

        # --- 拼网格 ---
        canvas = np.zeros((CANVAS[1], CANVAS[0], 3), np.uint8)
        for c in cams:
            col, row = GRID[c]
            canvas[row * CELL[1]:(row + 1) * CELL[1], col * CELL[0]:(col + 1) * CELL[0]] = cells[c]
        # 缝隙线
        cv2.line(canvas, (CELL[0] - 1, 0), (CELL[0] - 1, CANVAS[1]), (18, 18, 18), 3)
        cv2.line(canvas, (0, CELL[1] - 1), (CANVAS[0], CELL[1] - 1), (18, 18, 18), 3)

        # --- PIL HUD ---
        img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
        ov = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        # 顶部条
        d.rectangle([0, 0, CANVAS[0], 56], fill=(8, 10, 14, 205))
        d.text((22, 11), sc["title"], font=fonts[24], fill=(255, 255, 255, 255))
        ws = wall_str(t)
        l, _, r, _ = d.textbbox((0, 0), ws, font=fonts[30])
        d.text((CANVAS[0] - 24 - (r - l), 8), ws, font=fonts[30], fill=(0, 220, 255, 255))

        # 每格徽章 + 主角状态（上排避开顶部条，下压 56px）
        for c in cams:
            col, row = GRID[c]
            ox, oy = col * CELL[0], row * CELL[1]
            by = oy + (64 if row == 0 else 12)
            capsule(d, (ox + 12, by), f"{c} · {CELL_NAME[c]}", fonts[19],
                    fg=(255, 255, 255, 245), bg=(0, 0, 0, 150))
            sb = star_info[c]
            if sb is not None:
                txt = "● G1 · cradio_v3"
                tw = d.textlength(txt, font=fonts[17]) + 24
                capsule(d, (ox + CELL[0] - 12 - tw, by), txt, fonts[17],
                        fg=(10, 10, 10, 255), bg=(0, 220, 255, 220))

        # 检测框标签（框内顶部，PIL 统一绘制）
        for c in cams:
            col, row = GRID[c]
            ox, oy = col * CELL[0], row * CELL[1]
            for (lx, ly, txt, kind) in cell_labels.get(c, []):
                if kind == "star":
                    capsule(d, (ox + lx, oy + ly), txt, fonts[16],
                            fg=(10, 10, 10, 255), bg=(0, 220, 255, 232), pad=(8, 2), radius=6)
                else:
                    capsule(d, (ox + lx, oy + ly), txt, font_rg[15],
                            fg=(255, 255, 255, 235), bg=(0, 0, 0, 135), pad=(7, 1), radius=5)

        # 冻结指示（录像片段结束）：压暗 + 中央说明
        for c in frozen_now:
            col, row = GRID[c]
            ox, oy = col * CELL[0], row * CELL[1]
            d.rectangle([ox, oy, ox + CELL[0], oy + CELL[1]], fill=(0, 0, 0, 78))
            txt = f"■ {c} · 录像片段结束（该路 597s）"
            l, tp, r, b = d.textbbox((0, 0), txt, font=fonts[17])
            w, h = r - l + 32, b - tp + 20
            x0, y0 = ox + (CELL[0] - w) / 2, oy + CELL[1] / 2 - 20
            d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=10, fill=(10, 10, 12, 200))
            d.text((x0 + 16 - l, y0 + 10 - tp), txt, font=fonts[17], fill=(235, 235, 235, 245))

        # 事件层：连线 + 文字条
        def center_of(c, fallback):
            """连线端点：当前帧有框用当前；否则 fallback='last'/'first' 取轨迹末/首框"""
            sb = star_info[c]
            if sb is None:
                ser = stars[c]["ser"]
                frames = stars[c]["frames"]
                if not frames:
                    return None
                fr = frames[-1] if fallback == "last" else frames[0]
                sb = ser[fr]
            col, row = GRID[c]
            ox, oy = col * CELL[0], row * CELL[1]
            x1, y1, x2, y2 = sb
            W, H = SRC_RES[c]
            cx = ox + (x1 + x2) / 2 * CELL[0] / W
            cy = oy + (y1 + y2) / 2 * CELL[1] / H
            return (cx, cy)

        for e in sc["events"]:
            if e["t0"] <= t <= e["t1"]:
                p0 = center_of(e["a"], "last")
                p1 = center_of(e["b"], "first")
                if p0 and p1:
                    phase = (t - e["t0"]) / 1.3
                    draw_arrow_line(ov, p0, p1, phase)
                    # 文字条（连线中点）
                    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                    txt = "√ " + e["text"]
                    f = fonts[21]
                    l, tp, r, b = d.textbbox((0, 0), txt, font=f)
                    w, h = r - l + 36, b - tp + 22
                    x0, y0 = mx - w / 2, my - h - 34
                    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=12,
                                        fill=(8, 10, 14, 225), outline=(0, 220, 255, 235), width=3)
                    d.text((x0 + 18 - l, y0 + 11 - tp), txt, font=f, fill=(0, 220, 255, 255))

        # 底部条
        d.rectangle([0, CANVAS[1] - 118, CANVAS[0], CANVAS[1]], fill=(8, 10, 14, 210))
        # 时钟
        d.text((26, CANVAS[1] - 100), wall_str(t), font=fonts[30], fill=(255, 255, 255, 255))
        d.text((26, CANVAS[1] - 58), f"相对切片起点 +{t:.1f}s",
               font=font_rg[16], fill=(170, 175, 185, 255))
        # Gantt
        gx0, gx1 = 300, 1120
        gy0 = CANVAS[1] - 104
        row_h = 25
        tmap = lambda tt: gx0 + (tt - sc["t0"]) / (sc["t1"] - sc["t0"]) * (gx1 - gx0)
        for k, c in enumerate(cams):
            yy = gy0 + k * row_h
            d.text((232, yy - 2), c, font=fonts[17], fill=(200, 205, 215, 255))
            d.line([gx0, yy + 8, gx1, yy + 8], fill=(70, 74, 84, 200), width=2)
            s = sc["star"][c]
            xa, xb = tmap(s["t_first"]), tmap(s["t_last"])
            d.rounded_rectangle([xa, yy + 2, xb, yy + 14], radius=5, fill=(0, 220, 255, 235))
        cx = tmap(min(max(t, sc["t0"]), sc["t1"]))
        d.line([cx, gy0 - 6, cx, gy0 + len(cams) * row_h + 4], fill=(255, 255, 255, 240), width=3)

        # 右侧身份信息
        rx = 1210
        capsule(d, (rx, CANVAS[1] - 102), sc["subject"], fonts[19],
                fg=(10, 10, 10, 255), bg=(0, 220, 255, 225))
        d.text((rx, CANVAS[1] - 66), sc["subj2"], font=font_rg[16], fill=(200, 205, 215, 255))
        d.text((rx, CANVAS[1] - 38), "数据：2026-10-10 08:55–09:05 六路同步切片 · 25fps", font=font_rg[15],
               fill=(150, 155, 165, 255))

        img = Image.alpha_composite(img, ov).convert("RGB")
        out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        if smoke:
            if i in smoke_set:
                fp = os.path.join(OUT_DIR, f"smokeA_{sc['t0'] + i / FPS:.1f}s.jpg")
                cv2.imwrite(fp, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print("[smoke]", fp, flush=True)
        else:
            writer.write(out)
        if not smoke and (i + 1) % 100 == 0:
            el = time.time() - wall0
            print(f"[prog] {i+1}/{n_frames}  {(i+1)/el:.1f} fps  eta {(n_frames-i-1)/((i+1)/el):.0f}s", flush=True)

    for cap in caps.values():
        cap.release()
    if writer is not None:
        writer.release()
        print("[done]", sc["output"], flush=True)
    else:
        print("[smoke done]", flush=True)


# ---------------- 场景 B：318×320 同屏对精选 ----------------
CANVAS_B = (1920, 640)
PAIRS_B = [
    dict(a=14,  b=3,  name="黑T恤眼镜男（白耳机）", sim=0.9638, t0=48.4,  t1=56.4),
    dict(a=37,  b=11, name="藏青外套男",           sim=0.9555, t0=215.0, t1=225.4),
    dict(a=69,  b=14, name="背包男",               sim=0.9586, t0=326.6, t1=333.4),
    dict(a=120, b=18, name="皮领女士",             sim=0.9700, t0=530.6, t1=541.2),
    dict(a=139, b=24, name="黑卫衣男",             sim=0.9543, t0=571.8, t1=578.6),
    dict(a=146, b=28, name="白帽人",               sim=0.9482, t0=601.8, t1=609.4),
]


def draw_link_line(ov, p0, p1, color=(0, 220, 255, 215), width=4):
    """同屏同人连线：直线 + 两端圆点"""
    d = ImageDraw.Draw(ov)
    d.line([p0, p1], fill=color, width=width)
    for p in (p0, p1):
        d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7],
                  fill=(255, 255, 255, 240), outline=(0, 220, 255, 255), width=3)


def card_frame_b(k, pr, n_pairs, fonts, font_rg):
    """段间信息卡（黑底静态帧）"""
    base = Image.new("RGB", CANVAS_B, (10, 11, 14))
    ov = Image.new("RGBA", CANVAS_B, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    lines = [
        (f"第 {k + 1} / {n_pairs} 对", fonts[30], (150, 155, 165, 255), 150),
        (f"318 # {pr['a']}   =   320 # {pr['b']}", fonts[44], (0, 220, 255, 255), 225),
        (pr["name"], fonts[30], (255, 255, 255, 255), 330),
        (f"cradio_v3 sim = {pr['sim']:.3f} · 同屏同时出现 · 人工复核通过", font_rg[21], (185, 190, 200, 255), 400),
    ]
    for txt, f, col, y in lines:
        l, tp, r, b = d.textbbox((0, 0), txt, font=f)
        d.text(((CANVAS_B[0] - (r - l)) / 2 - l, y - tp), txt, font=f, fill=col)
    img = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def render_scene_b(smoke=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "cradio_v3_sceneB_pairs_318x320.mp4")
    cams = ["318", "320"]
    dets = {c: load_det(c) for c in cams}
    need = {"318": {pr["a"] for pr in PAIRS_B}, "320": {pr["b"] for pr in PAIRS_B}}
    ser = {c: {tid: {} for tid in need[c]} for c in cams}
    for c in cams:
        for fr, (tids, boxes, _) in dets[c].items():
            for tid in need[c]:
                if tid in tids:
                    ser[c][tid][fr] = boxes[tids.index(tid)]
    ser = {c: {tid: dict(sorted(v.items())) for tid, v in ser[c].items()} for c in cams}
    frames_sorted = {c: {tid: sorted(v.keys()) for tid, v in ser[c].items()} for c in cams}

    fonts = {sz: ImageFont.truetype(FONT_BD, sz) for sz in (44, 36, 30, 24, 21, 19, 17, 16)}
    font_rg = {sz: ImageFont.truetype(FONT_RG, sz) for sz in (21, 19, 17, 16, 15)}

    box_plain = sv.BoxAnnotator(thickness=2, color=sv.Color.from_hex(GRAY_HEX),
                                color_lookup=sv.ColorLookup.INDEX)
    box_star = sv.BoxAnnotator(thickness=5, color=sv.Color.from_hex(GOLD_HEX),
                               color_lookup=sv.ColorLookup.INDEX)

    writer = None if smoke else cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, CANVAS_B)
    if writer is not None:
        assert writer.isOpened()
    smoke_marks = {3: 12.0, 4: 6.0}  # (段索引: 段内秒) 抽帧检查
    smoke_target_i = {k: int(round(s * FPS)) for k, s in smoke_marks.items()}

    TOPH, VIDY, BOTH = 58, 58, 598  # 顶条高 / 视频区起点 / 视频区终点

    for k, pr in enumerate(PAIRS_B):
        # 段间信息卡
        cf = card_frame_b(k, pr, len(PAIRS_B), fonts, font_rg)
        n_card = int(FPS * 1.2)
        if not smoke:
            for _ in range(n_card):
                writer.write(cf)
        elif k in smoke_marks:
            pass

        cap = {}
        for c in cams:
            cp = cv2.VideoCapture(VID[c])
            cp.set(cv2.CAP_PROP_POS_FRAMES, int(round((pr["t0"] - T0[c]) * FPS)))
            cap[c] = cp

        n = int(round((pr["t1"] - pr["t0"]) * FPS))
        for i in range(n):
            t = pr["t0"] + i / FPS
            cells, labels_all, centers = {}, {}, {}
            for c in cams:
                ok, frame = cap[c].read()
                if not ok:
                    raise RuntimeError(f"[B] {c} 帧读取失败 @ t={t:.1f}")
                W, H = SRC_RES[c]
                sx, sy = CELL[0] / W, CELL[1] / H
                small = cv2.resize(frame, CELL, interpolation=cv2.INTER_AREA)
                f_cam = int(round((t - T0[c]) * FPS))
                tid_star = pr["a"] if c == "318" else pr["b"]
                frs = frames_sorted[c][tid_star]
                sb = box_at(ser[c][tid_star], frs, f_cam)
                centers[c] = sb
                det_fr = f_cam - (f_cam % 5)
                tids, boxes, _ = dets[c].get(det_fr, ([], [], []))
                p_xy, labs = [], []
                for tid, b in zip(tids, boxes):
                    if tid == tid_star:
                        continue
                    x1, y1, x2, y2 = b
                    p_xy.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                    labs.append((x1 * sx + 4, y1 * sy + 4, f"#{tid}", "plain"))
                if p_xy:
                    small = box_plain.annotate(small, sv.Detections(
                        xyxy=np.array(p_xy, dtype=np.float32)))
                if sb is not None:
                    x1, y1, x2, y2 = sb
                    small = box_star.annotate(small, sv.Detections(xyxy=np.array(
                        [[x1 * sx, y1 * sy, x2 * sx, y2 * sy]], dtype=np.float32)))
                    labs.append((x1 * sx + 5, y1 * sy + 5, f"√ sim={pr['sim']:.3f}", "star"))
                cells[c] = small
                labels_all[c] = labs

            # 拼接
            canvas = np.zeros((CANVAS_B[1], CANVAS_B[0], 3), np.uint8)
            canvas[VIDY:VIDY + CELL[1], 0:CELL[0]] = cells["318"]
            canvas[VIDY:VIDY + CELL[1], CELL[0]:2 * CELL[0]] = cells["320"]
            cv2.line(canvas, (CELL[0] - 1, VIDY), (CELL[0] - 1, VIDY + CELL[1]), (18, 18, 18), 3)

            img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
            ov = Image.new("RGBA", CANVAS_B, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)

            # 同人连线（当前/回退端点）
            def ctr(c, fallback):
                sb = centers[c]
                if sb is None:
                    t2 = pr["a"] if c == "318" else pr["b"]
                    fs = frames_sorted[c][t2]
                    if not fs:
                        return None
                    sb = ser[c][t2][fs[-1] if fallback == "last" else fs[0]]
                W, H = SRC_RES[c]
                cx = (sb[0] + sb[2]) / 2 * CELL[0] / W + (0 if c == "318" else CELL[0])
                cy = (sb[1] + sb[3]) / 2 * CELL[1] / H + VIDY
                return (cx, cy)

            p0 = ctr("318", "last")
            p1 = ctr("320", "first")
            if p0 and p1:
                draw_link_line(ov, p0, p1)

            # 顶部条
            d.rectangle([0, 0, CANVAS_B[0], TOPH], fill=(8, 10, 14, 210))
            d.text((22, 12), "318 × 320 重叠视野 · 同屏身份识别（cradio_v3）", font=fonts[24],
                   fill=(255, 255, 255, 255))
            ws = wall_str(t)
            l, _, r, _ = d.textbbox((0, 0), ws, font=fonts[24])
            d.text((CANVAS_B[0] - 24 - (r - l), 13), ws, font=fonts[24], fill=(0, 220, 255, 255))

            # 格徽章 + 状态胶囊 + 标签
            for ci, c in enumerate(cams):
                ox = ci * CELL[0]
                capsule(d, (ox + 12, VIDY + 12), f"{c} · {CELL_NAME[c]}", fonts[19],
                        fg=(255, 255, 255, 245), bg=(0, 0, 0, 150))
                if centers[c] is not None:
                    txt = "● cradio_v3"
                    tw = d.textlength(txt, font=fonts[17]) + 24
                    capsule(d, (ox + CELL[0] - 12 - tw, VIDY + 12), txt, fonts[17],
                            fg=(10, 10, 10, 255), bg=(0, 220, 255, 220))
                for (lx, ly, txt, kind) in labels_all[c]:
                    if kind == "star":
                        capsule(d, (ox + lx, VIDY + ly), txt, fonts[16],
                                fg=(10, 10, 10, 255), bg=(0, 220, 255, 232), pad=(8, 2), radius=6)
                    else:
                        capsule(d, (ox + lx, VIDY + ly), txt, font_rg[15],
                                fg=(255, 255, 255, 235), bg=(0, 0, 0, 135), pad=(7, 1), radius=5)

            # 底部条
            d.rectangle([0, BOTH, CANVAS_B[0], CANVAS_B[1]], fill=(8, 10, 14, 215))
            info = (f"第 {k + 1}/{len(PAIRS_B)} 对 ｜ 318 # {pr['a']} = 320 # {pr['b']} ｜ "
                    f"{pr['name']} ｜ sim = {pr['sim']:.3f} ｜ 人工复核通过")
            d.text((24, BOTH + 11), info, font=fonts[19], fill=(235, 238, 245, 255))
            d.text((CANVAS_B[0] - 360, BOTH + 12), "数据 2026-10-10 08:55–09:05", font=font_rg[16],
                   fill=(150, 155, 165, 255))

            out = cv2.cvtColor(np.asarray(Image.alpha_composite(img, ov).convert("RGB")), cv2.COLOR_RGB2BGR)
            if smoke:
                if k in smoke_target_i and i == smoke_target_i[k]:
                    fp = os.path.join(OUT_DIR, f"smokeB_pair{k}_t{t:.1f}.jpg")
                    cv2.imwrite(fp, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    print("[smoke]", fp, flush=True)
            else:
                writer.write(out)

        for cp in cap.values():
            cp.release()
        if not smoke:
            print(f"[seg] pair {k + 1}/{len(PAIRS_B)} done  ({pr['name']})", flush=True)

    if writer is not None:
        writer.release()
        print("[done]", out_path, flush=True)
    else:
        print("[smoke B done]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="a", choices=["a", "b"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.scene == "a":
        render_scene_a(smoke=args.smoke)
    else:
        render_scene_b(smoke=args.smoke)


if __name__ == "__main__":
    main()
