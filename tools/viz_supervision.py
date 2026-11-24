#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 supervision 把 1012 的检测 + 跨机位身份判定画成可看的视频。

输入全部来自既有产物，不重跑模型：
  det   : output_crosscam_local/1012_yolo8_dd/<seg>/<seg>_detections.jsonl
  verdict: deliverable/data-request-v2/results/verdicts/<cfg>.csv（每条查询的 c_pid / verdict）
  queries: deliverable/data-request-v2/results/queries/<cfg>.csv（final_status，用于区分"未识别"）

画法：
  · 框与拖尾颜色 = 身份（class_id 绑定身份，颜色跨帧稳定）
  · 标签 = "<身份> #<轨迹号>"；未通过阈值的轨迹标 "? #<轨迹号>"（灰色，表示"看到了但不猜"）
  · 左上 HUD：机位 / 时间段 / 本帧统计；底部图例：本帧出现的身份

用法：
  python tools/viz_supervision.py --seg 305_test_544 --start-sec 676 --dur 40 --out dist/viz/305_test_544_676s.mp4
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import supervision as sv

FONT_CJK = r"C:\Windows\Fonts\msyh.ttc"
FONT_CJK_BD = r"C:\Windows\Fonts\msyhbd.ttc"

# 身份配色：避开纯红/纯绿（留给核对标记），按 class_id 取色
PALETTE_RGB = [
    (64, 158, 255), (255, 138, 61), (52, 199, 168), (196, 120, 255), (255, 196, 61),
    (86, 209, 255), (255, 99, 146), (126, 211, 33), (255, 160, 122), (120, 144, 255),
    (0, 184, 148), (255, 133, 192), (140, 200, 90), (100, 181, 246), (255, 171, 64),
    (171, 71, 188), (38, 166, 154), (255, 112, 67), (124, 179, 66), (79, 195, 247),
    (240, 98, 146), (149, 117, 205), (102, 187, 106), (255, 167, 38), (66, 165, 245),
    (236, 64, 122), (141, 110, 99),
]
GRAY_RGB = (150, 152, 158)


def load_det(path):
    frames = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        f = int(r["frame"])
        ids = r.get("track_ids") or []
        bxs = r.get("boxes") or []
        frames[f] = [(int(t), [float(v) for v in b]) for t, b in zip(ids, bxs) if b and len(b) >= 4]
    return frames


def load_verdicts(path, seg):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if not r["query_id"].startswith(seg + "#"):
            continue
        out[int(r["query_id"].split("#")[1])] = dict(pid=r["c_pid"], verdict=r["verdict"],
                                                     cluster=r["cluster_local"], n_lab=r["cluster_n_lab"])
    return out


def load_queries(path, seg):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        if not r["query_id"].startswith(seg + "#"):
            continue
        out[int(r["query_id"].split("#")[1])] = r["final_status"]
    return out


def lerp(a, b, w):
    return [a[i] + (b[i] - a[i]) * w for i in range(4)]


def boxes_at(frames_sorted, frames, t):
    """在采样时刻 t（原始帧号，可为小数）插值出所有在场轨迹的框。"""
    lo = None
    hi = None
    # 二分找相邻采样帧
    import bisect
    i = bisect.bisect_right(frames_sorted, t)
    if i > 0:
        lo = frames_sorted[i - 1]
    if i < len(frames_sorted):
        hi = frames_sorted[i]
    out = {}
    if lo is None and hi is None:
        return out
    if hi is None or lo is None:
        src = lo if lo is not None else hi
        for tid, b in frames[src]:
            out[tid] = b
        return out
    a = dict(frames[lo])
    b = dict(frames[hi])
    w = (t - lo) / max(1e-6, (hi - lo))
    for tid in set(a) | set(b):
        if tid in a and tid in b:
            out[tid] = lerp(a[tid], b[tid], w)
        elif tid in a:
            # 只在这一侧出现：靠近边界时淡出（保持在边界处，不抖）
            out[tid] = a[tid]
        else:
            out[tid] = b[tid]
    return out



def make_intro(out_w, out_h, title, subtitle, bullets, footer, secs, fps):
    """生成片头卡（返回 list[np.ndarray]，RGB）。"""
    im = Image.new("RGB", (out_w, out_h), (12, 16, 24))
    d = ImageDraw.Draw(im)
    # 背景渐变
    for y in range(out_h):
        k = y / out_h
        d.line([(0, y), (out_w, y)], fill=(int(12 + 10 * k), int(16 + 12 * k), int(24 + 18 * k)))
    f_t = ImageFont.truetype(FONT_CJK_BD, int(out_h * 0.062))
    f_s = ImageFont.truetype(FONT_CJK, int(out_h * 0.032))
    f_b = ImageFont.truetype(FONT_CJK, int(out_h * 0.028))
    f_f = ImageFont.truetype(FONT_CJK, int(out_h * 0.022))
    x = int(out_w * 0.08)
    y = int(out_h * 0.26)
    d.rectangle([x, y - int(out_h * 0.055), x + int(out_w * 0.006), y + int(out_h * 0.075)],
                fill=(64, 158, 255))
    d.text((x + int(out_w * 0.022), y - int(out_h * 0.060)), title, font=f_t, fill=(255, 255, 255))
    d.text((x, y + int(out_h * 0.010)), subtitle, font=f_s, fill=(150, 190, 235))
    yy = y + int(out_h * 0.105)
    for b in bullets:
        d.text((x, yy), "·  " + b, font=f_b, fill=(205, 216, 230))
        yy += int(out_h * 0.056)
    d.text((x, int(out_h * 0.88)), footer, font=f_f, fill=(120, 132, 150))
    frame = np.asarray(im)
    return [frame.copy() for _ in range(max(0, int(secs * fps)))]


def paste_zoom(img, roi, frac, top_pad, bottom_pad, smooth=None):
    """把 roi（输出坐标系）区域放大贴到右上角，返回 (img, roi_used)。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in roi]
    # 16:9 + 最小尺寸，避免退化
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = max(x2 - x1, w * 0.10)
    bh = max(y2 - y1, h * 0.12)
    if bw / bh < 16 / 9:
        bw = bh * 16 / 9
    else:
        bh = bw * 9 / 16
    bw *= 1.22
    bh *= 1.22
    x1, x2 = cx - bw / 2, cx + bw / 2
    y1, y2 = cy - bh / 2, cy + bh / 2
    x1 = max(0, min(x1, w - bw))
    y1 = max(top_pad, min(y1, h - bottom_pad - bh))
    x2, y2 = x1 + bw, y1 + bh
    roi_now = np.array([x1, y1, x2, y2], dtype=np.float32)
    if smooth is not None:
        roi_now = 0.72 * smooth + 0.28 * roi_now
    x1, y1, x2, y2 = roi_now
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return img, roi_now
    iw = int(w * frac)
    ih = int(iw * 9 / 16)
    panel = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_CUBIC)
    # 面板放到 ROI 的对角，避免盖住被放大的目标
    cx_roi = (x1 + x2) / 2
    cy_roi = (y1 + y2) / 2
    margin = int(w * 0.018)
    px = (w - iw - margin) if cx_roi < w / 2 else margin
    py = (top_pad + int(h * 0.014)) if cy_roi > h / 2 else (h - bottom_pad - ih - int(h * 0.014))
    px = max(margin, min(px, w - iw - margin))
    py = max(top_pad, min(py, h - bottom_pad - ih))
    cv2.rectangle(panel, (0, 0), (iw - 1, ih - 1), (64, 158, 255), 2)
    img[py:py + ih, px:px + iw] = panel
    tag = f"x{iw / max(1.0, bw):.1f} zoom"
    cv2.rectangle(img, (px, py + ih), (px + 138, py + ih + 26), (9, 13, 20), -1)
    cv2.putText(img, tag, (px + 8, py + ih + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (64, 158, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (64, 158, 255), 1)
    return img, roi_now


def draw_hud(img_rgb, seg, cam, clock, n_id, n_unk, legend, progress, meta):
    h, w = img_rgb.shape[:2]
    base = Image.fromarray(img_rgb).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_title = ImageFont.truetype(FONT_CJK_BD, max(20, int(h * 0.032)))
    f_body = ImageFont.truetype(FONT_CJK_BD, max(15, int(h * 0.023)))
    f_small = ImageFont.truetype(FONT_CJK, max(13, int(h * 0.019)))
    f_chip = ImageFont.truetype(FONT_CJK, max(12, int(h * 0.018)))

    # ---- 顶部实心条：盖住原片 OSD，避免文字打架
    bar = int(h * 0.098)
    d.rectangle([0, 0, w, bar], fill=(9, 13, 20, 252))
    d.line([(0, bar), (w, bar)], fill=(64, 158, 255, 110), width=2)

    d.text((int(w * 0.018), int(h * 0.012)), f"1012 现场 · {cam} 号机位 · TEST",
           font=f_title, fill=(255, 255, 255, 248))
    d.text((int(w * 0.018), int(h * 0.055)), meta, font=f_small, fill=(168, 188, 214, 235))

    tw = d.textlength(clock, font=f_body)
    d.text((w - tw - int(w * 0.02), int(h * 0.014)), clock, font=f_body, fill=(255, 255, 255, 245))

    tag = f"已识别 {n_id} · 未识别 {n_unk}"
    tw2 = d.textlength(tag, font=f_small)
    d.text((w - tw2 - int(w * 0.02), int(h * 0.056)), tag, font=f_small,
           fill=(120, 210, 170, 240) if n_unk == 0 else (250, 210, 120, 240))

    # ---- 底部实心条：图例 + 进度
    bh = int(h * 0.100)
    d.rectangle([0, h - bh, w, h], fill=(9, 13, 20, 255))
    chip = int(h * 0.020)
    x = int(w * 0.018)
    cy = h - bh + (bh - chip) // 2
    for name, color in legend:
        tw3 = d.textlength(name, font=f_chip)
        need = chip + 7 + tw3 + 18
        if x + need > w - int(w * 0.02):
            d.text((x, cy - 2), "…", font=f_chip, fill=(160, 170, 185, 220))
            break
        d.rounded_rectangle([x, cy, x + chip, cy + chip], radius=max(2, chip // 4),
                            fill=color + (255,))
        d.text((x + chip + 7, cy - 2), name, font=f_chip, fill=(238, 243, 250, 245))
        x += need

    d.rectangle([0, h - 4, w, h], fill=(255, 255, 255, 30))
    d.rectangle([0, h - 4, int(w * progress), h], fill=(64, 158, 255, 235))
    return np.asarray(Image.alpha_composite(base, layer).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="305_test_544")
    ap.add_argument("--start-sec", type=float, default=676.0)
    ap.add_argument("--dur", type=float, default=40.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--trail", type=int, default=45, help="拖尾帧数（0=关）")
    ap.add_argument("--cfg", default="A_baseline")
    ap.add_argument("--det-root", default="output_crosscam_local/1012_yolo8_dd")
    ap.add_argument("--pkg", default="deliverable/data-request-v2")
    ap.add_argument("--snap", default="", help="额外导出若干预览 PNG 的目录")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="放大镜面板宽度占画面比例（0=关）。远场机位建议 0.36–0.45")
    ap.add_argument("--intro", type=float, default=0.0, help="片头卡秒数（0=不要）")
    ap.add_argument("--intro-note", default="", help="片头卡第三行自定义说明")
    a = ap.parse_args()

    seg = a.seg
    cam = seg.split("_")[0]
    video = f"media-resource/1012/segments/{seg}.mp4"
    det = os.path.join(a.det_root, seg, f"{seg}_detections.jsonl")
    vpath = os.path.join(a.pkg, "results/verdicts", f"{a.cfg}.csv")
    qpath = os.path.join(a.pkg, "results/queries", f"{a.cfg}.csv")
    for p in (video, det, vpath):
        if not os.path.exists(p):
            raise SystemExit(f"缺文件: {p}")

    frames = load_det(det)
    frames_sorted = sorted(frames)
    verdicts = load_verdicts(vpath, seg)
    queries = load_queries(qpath, seg)

    # 身份 → class_id（按名字排序，保证颜色稳定）
    ids = sorted({v["pid"] for v in verdicts.values() if v["pid"]})
    cls_of = {p: i for i, p in enumerate(ids)}
    unk_cls = len(ids)
    colors = [sv.Color(*PALETTE_RGB[i % len(PALETTE_RGB)]) for i in range(len(ids))]
    colors.append(sv.Color(*GRAY_RGB))
    palette = sv.ColorPalette(colors=colors)

    box_an = sv.BoxAnnotator(color=palette, thickness=3, color_lookup=sv.ColorLookup.CLASS)
    lab_an = sv.LabelAnnotator(color=palette, text_color=sv.Color.WHITE,
                               text_scale=0.62, text_thickness=1, text_padding=5,
                               color_lookup=sv.ColorLookup.CLASS, border_radius=4)
    tr_an = (sv.TraceAnnotator(color=palette, trace_length=a.trail, thickness=3,
                               color_lookup=sv.ColorLookup.CLASS) if a.trail > 0 else None)

    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = a.width
    out_h = int(round(out_w * H0 / W0 / 2) * 2)
    start_f = int(round(a.start_sec * src_fps))
    end_f = min(n_total, start_f + int(round(a.dur * src_fps)))
    out = a.out or f"dist/viz/{seg}_{int(a.start_sec)}s.mp4"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{out_w}x{out_h}", "-r", f"{a.fps}", "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_written = 0
    if a.intro > 0:
        bullets = [f"框色 = 身份，标签 = 身份 + 轨迹号；「? #123」= 已检出但未通过阈值，系统不猜",
                   a.intro_note or "每条轨迹两侧各取一半帧编码，跨机位聚类后再按阈值决定是否给出身份",
                   "评估口径：229 条查询 / 接受 124 条 / 逐条精度 0.952 / 错判 6 条"]
        card_frames = make_intro(out_w, out_h, "跨机位身份匹配 · 可视化",
                                 f"1012 现场 · {cam} 号机位 · {seg}（TEST，评估集）",
                                 [b for b in bullets if b], "supervision 0.24 · YOLO26s-seg + ByteTrack + OSNet + 机位感知聚类",
                                 a.intro, a.fps)
        for cf in card_frames:
            proc.stdin.write(cf.tobytes())
            n_written += 1

    cam = seg.split("_")[0]
    meta = f"{a.cfg} · 图库 25 身份 / K=111 簇 · θ=1.85 · 覆盖 0.664 · 逐条精度 0.952"
    step = src_fps / a.fps
    snaps = [0.15, 0.4, 0.65, 0.9] if a.snap else []
    snap_done = set()
    if a.snap:
        os.makedirs(a.snap, exist_ok=True)

    stat_people = []
    roi_smooth = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    src_i = start_f
    acc = 0.0
    while src_i < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        acc += 1.0
        if acc + 1e-9 < step:
            src_i += 1
            continue
        acc -= step
        t = float(src_i)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        boxes = boxes_at(frames_sorted, frames, t)
        xyxy, cids, tids, labels = [], [], [], []
        sx = out_w / W0
        sy = out_h / H0
        present = []
        for tid, b in sorted(boxes.items()):
            x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
            if x2 - x1 < 6 or y2 - y1 < 10:
                continue
            v = verdicts.get(tid)
            if v and v["pid"]:
                cid = cls_of.get(v["pid"], unk_cls)
                labels.append(f"{v['pid']} #{tid}")
                present.append(v["pid"])
            else:
                cid = unk_cls
                labels.append(f"? #{tid}")
            xyxy.append([x1, y1, x2, y2])
            cids.append(cid)
            tids.append(tid)

        if xyxy:
            dets = sv.Detections(xyxy=np.array(xyxy, dtype=np.float32),
                                 class_id=np.array(cids, dtype=int),
                                 tracker_id=np.array(tids, dtype=int))
            if tr_an is not None:
                img = tr_an.annotate(scene=img, detections=dets)
            img = box_an.annotate(scene=img, detections=dets)
            img = lab_an.annotate(scene=img, detections=dets, labels=labels)

        if a.zoom > 0 and xyxy:
            arr = np.array(xyxy)
            roi = [arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()]
            bar_h = int(out_h * 0.098)
            bot_h = int(out_h * 0.100)
            img, roi_smooth = paste_zoom(img, roi, a.zoom, bar_h, bot_h, roi_smooth)
            img = np.ascontiguousarray(img)

        n_id = len(present)
        n_unk = len(xyxy) - n_id
        stat_people.append(len(xyxy))
        seen = collections.OrderedDict()
        for p in present:
            seen.setdefault(p, PALETTE_RGB[cls_of[p] % len(PALETTE_RGB)])
        pc = collections.Counter(present)
        legend = [(f"{p} ×{pc[p]}" if pc[p] > 1 else p, c) for p, c in seen.items()]
        sec = int(t / src_fps - a.start_sec)   # 相对本段起点的秒数
        clock = f"{int(sec)//60:02d}:{int(sec)%60:02d} / {int(a.dur)//60:02d}:{int(a.dur)%60:02d}"
        prog = (t - start_f) / max(1.0, (end_f - start_f))
        img = draw_hud(img, seg, cam, clock, n_id, n_unk, legend, prog, meta)

        proc.stdin.write(img.tobytes())
        n_written += 1

        for k in snaps:
            if k not in snap_done and prog >= k:
                cv2.imwrite(os.path.join(a.snap, f"snap_{int(prog*100):03d}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                snap_done.add(k)
        src_i += 1

    cap.release()
    proc.stdin.close()
    rc = proc.wait()
    size = os.path.getsize(out) / 1048576 if os.path.exists(out) else 0
    print(f"[out] {out}")
    print(f"[stat] 帧 {n_written}｜{out_w}x{out_h}@{a.fps:g}｜时长 {n_written/a.fps:.1f}s｜{size:.1f} MiB｜ffmpeg rc={rc}")
    if stat_people:
        arr = np.array(stat_people)
        print(f"[stat] 同框人数 均值 {arr.mean():.1f}｜中位 {np.median(arr):.0f}｜峰值 {arr.max()}｜空帧 {int((arr==0).sum())}")
    print(f"[stat] 该段身份 {len(ids)} 个：{', '.join(ids)}")
    print(f"[stat] 段内查询 {len(queries)} 条，其中已判定 {len(verdicts)} 条")


if __name__ == "__main__":
    main()
