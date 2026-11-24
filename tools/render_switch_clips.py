#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-A2：把 46 条串人轨迹渲染成「带框」的复核片段。

为什么：`switch_review.html` 只放原视频时看不到跟踪框，无法判断框在哪一刻换人。
本脚本用 supervision 画框（目标轨迹高亮 + 同场其他人灰框 + 右上角放大镜），
再用 ffmpeg(libx264) 编成浏览器可播的 H.264。

每条约两个片段：
  focus  原记录点 ±8s（原记录为末帧/不可解析时取轨迹末 20s，覆盖过低时滑到最密处），原生 25fps
  scan   整条轨迹，5fps 抽样（时长与真实一致，可快速扫完全程）

窗口定义由 `tools/switch_windows.py` 统一计算，页面读同一份 JSON，避免渲染与播放对不上。
帧号口径：一律用 ledger 的 25fps 帧号；容器时长有约 0.17% 漂移，因此**按帧定位、不按秒定位**。

用法：
  PY tools/switch_windows.py --show        # 先看窗口
  PY tools/render_switch_clips.py          # 全量渲染（约 9 分钟）
  PY tools/render_switch_clips.py --only 323_train_510#3169,318_train_525#1041
输出：deliverable/data-request-v2/gt/clips/{track_id}_{focus,scan}.mp4 + manifest.json
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import supervision as sv

from switch_windows import DetStore, target_box, load_items, FPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
SEG_DIR = os.path.join(ROOT, 'media-resource/1012/segments')
OUT_DIR = os.path.join(DELIV, 'gt/clips')
SCAN_STEP = 5


def find_ffmpeg():
    p = os.environ.get('IMAGEIO_FFMPEG_EXE')
    if p and os.path.exists(p):
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import glob
    for pat in ('imageio_ffmpeg/binaries/ffmpeg-*.exe',
                '../imageio_ffmpeg/binaries/ffmpeg-*.exe'):
        for f in glob.glob(os.path.join(os.path.dirname(sv.__file__), pat)):
            return os.path.abspath(f)
    from shutil import which
    p = which('ffmpeg')
    if p:
        return p
    raise SystemExit('找不到 ffmpeg（pip install imageio-ffmpeg 即可）')


def make_annotators():
    # supervision 0.24 只内置 7 种颜色，灰/亮绿自己给（sv.Color 用 RGB 序）
    return dict(
        target=sv.BoxAnnotator(color=sv.Color(r=40, g=230, b=90), thickness=3),
        other=sv.BoxAnnotator(color=sv.Color(r=155, g=155, b=155), thickness=1),
        label=sv.LabelAnnotator(color=sv.Color(r=40, g=230, b=90), text_color=sv.Color.BLACK,
                                text_scale=0.55, text_thickness=1, text_padding=4),
        other_label=sv.LabelAnnotator(color=sv.Color(r=140, g=140, b=140), text_color=sv.Color.WHITE,
                                      text_scale=0.4, text_thickness=1, text_padding=2),
    )


def _band(img, x0, y0, x1, y1, alpha=0.55):
    """半透明黑底，保证叠加文字在任意画面上可读。"""
    roi = img[y0:y1, x0:x1]
    if roi.size:
        roi[:] = (roi * (1 - alpha)).astype(roi.dtype)


def draw_hud(img, item, frame, tid, has_target, box_src, W, H, win):
    cyan, yellow, red, white = (255, 214, 0), (0, 200, 255), (60, 60, 235), (240, 240, 240)
    _band(img, 0, 0, 470, 88)
    lines = [
        ('%s  cam %s' % (item['track_id'], item['cam']), white),
        ('frame %d   t=%.2fs' % (frame, frame / FPS), white),
        ('target tid %d  %s' % (tid, 'BOX ' + box_src if has_target else 'LOST'), cyan if has_target else red),
    ]
    y = 24
    for s, col in lines:
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        y += 24

    _band(img, 0, H - 40, W, H)
    a, b = win['t0'] - 25, win['t1'] + 25
    bar_y, pad = H - 14, 12
    x0, x1 = pad, W - 160
    cv2.line(img, (x0, bar_y), (x1, bar_y), (90, 90, 90), 4)

    def px(f):
        r = (f - a) / float(max(1, b - a))
        return int(x0 + max(0.0, min(1.0, r)) * (x1 - x0))

    xa, xb = px(win['start']), px(win['end'])
    cv2.line(img, (xa, bar_y), (xb, bar_y), (170, 170, 170), 4)
    if win['old'] is not None:
        xo = px(win['old'])
        cv2.line(img, (xo, bar_y - 11), (xo, bar_y + 5), yellow, 2)
        cv2.putText(img, 'rec', (xo + 3, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, yellow, 1, cv2.LINE_AA)
    xc = px(frame)
    cv2.circle(img, (xc, bar_y), 5, (0, 255, 0), -1)
    cv2.putText(img, '%.1fs / %.1fs' % (frame / FPS, win['t1'] / FPS),
                (W - 150, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def draw_zoom(canvas, clean, box, W, H, size=250):
    """右上角放大镜：远处目标只有几十像素，不放大看不出是谁。

    clean 是未标注的干净帧——放大镜里只画目标框，避免叠加文字糊成一团。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * 0.95 + 12
    sx1, sx2 = int(max(0, cx - half)), int(min(W, cx + half))
    sy1, sy2 = int(max(0, cy - half)), int(min(H, cy + half))
    if sx2 - sx1 < 24 or sy2 - sy1 < 24:
        return
    crop = clean[sy1:sy2, sx1:sx2]
    inset = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    sc = size / float(max(1, sx2 - sx1))
    cv2.rectangle(inset,
                  (int((x1 - sx1) * sc), int((y1 - sy1) * sc)),
                  (int((x2 - sx1) * sc), int((y2 - sy1) * sc)), (40, 230, 90), 2)
    x0, y0 = W - size - 14, 14
    canvas[y0:y0 + size, x0:x0 + size] = inset
    cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + size + 1, y0 + size + 1), (40, 230, 90), 2)
    txt = 'zoom x%.1f' % (size / float(max(1, sx2 - sx1)))
    cv2.putText(canvas, txt, (x0, y0 + size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, txt, (x0, y0 + size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 230, 90), 1, cv2.LINE_AA)


def open_writer(path, w, h, fps, crf):
    cmd = [find_ffmpeg(), '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (w, h),
           '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf),
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render(item, det, ann, outdir, kind, width, crf):
    cap = cv2.VideoCapture(os.path.join(SEG_DIR, item['seg'] + '.mp4'))
    if not cap.isOpened():
        raise SystemExit('打不开视频: ' + item['seg'])
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = width / float(src_w)
    w, h = width, int(round(src_h * s))

    start, end = item[kind + '_start'], item[kind + '_end']
    step = 1 if kind == 'focus' else SCAN_STEP
    out_fps = FPS if kind == 'focus' else FPS / SCAN_STEP

    series = det.series(item['tid'])
    win = dict(t0=item['t0_f'], t1=item['t1_f'], old=item['old_frame'],
               start=start, end=end)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    path = os.path.join(outdir, '%s_%s.mp4' % (item['file_base'], kind))
    proc = open_writer(path, w, h, out_fps, crf)

    n = 0
    lost = 0
    frame = start
    while frame < end:
        # 顺序读：step=1 用 read()；step>1 先 grab 到目标帧再 retrieve（避免反复 seek）
        if step == 1:
            ok, img = cap.read()
        else:
            ok = cap.grab()
            if not ok:
                break
            ok, img = cap.retrieve()
            for _ in range(step - 1):
                if not cap.grab():
                    ok = False
                    break
        if not ok:
            break
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        clean = img.copy()

        nf, others = det.nearest(frame)
        tids = [t for t in others.keys() if t != item['tid']]
        if tids:
            xy = np.array([others[t] for t in tids], dtype=np.float32) * s
            ann['other'].annotate(img, sv.Detections(xyxy=xy, class_id=np.arange(len(tids))))
            # 只给近场（框高 ≥55px）标 tid，避免远处糊成一片
            big = [(t, others[t]) for t in tids if (others[t][3] - others[t][1]) * s >= 55]
            if big:
                ann['other_label'].annotate(
                    img, sv.Detections(xyxy=np.array([b for _, b in big], dtype=np.float32) * s,
                                       class_id=np.arange(len(big))),
                    labels=['#%d' % t for t, _ in big])

        tb = target_box(series, frame)
        if tb is not None:
            xy = np.array([tb], dtype=np.float32) * s
            d = sv.Detections(xyxy=xy, class_id=np.array([0]))
            ann['target'].annotate(img, d)
            ann['label'].annotate(img, d, labels=['TARGET %d' % item['tid']])
            src_tag = 'det' if nf == frame else 'interp'
        else:
            lost += 1
            src_tag = 'none'

        img = draw_hud(img, item, frame, item['tid'], tb is not None, src_tag, w, h, win)
        if tb is not None:
            draw_zoom(img, clean, np.array(tb, dtype=np.float32) * s, w, h)
        proc.stdin.write(img.tobytes())
        n += 1
        frame += step

    cap.release()
    proc.stdin.close()
    proc.wait()
    size = os.path.getsize(path) / 1048576.0 if os.path.exists(path) else 0
    print('    %-6s %5d 帧 %6.1fs  %5.1f MB  lost=%d  %s'
          % (kind, n, n / out_fps, size, lost, os.path.basename(path)))
    return dict(kind=kind, file=os.path.basename(path), start=start, end=end,
                fps=out_fps, n=n, size_mb=round(size, 2), lost=lost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=OUT_DIR)
    ap.add_argument('--only', default='', help='逗号分隔的 track_id')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--kinds', default='focus,scan')
    ap.add_argument('--width-focus', type=int, default=1280)
    ap.add_argument('--width-scan', type=int, default=768)
    ap.add_argument('--crf', type=int, default=27)
    ap.add_argument('--no-merge', action='store_true', help='不合并旧 manifest')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    kinds = [k.strip() for k in a.kinds.split(',') if k.strip()]

    items = load_items()
    if a.only:
        want = set(x.strip() for x in a.only.split(','))
        items = [it for it in items if it['track_id'] in want]
    if a.limit:
        items = items[:a.limit]

    mp = os.path.join(a.outdir, 'manifest.json')
    manifest = {}
    if os.path.exists(mp) and not a.no_merge:
        try:
            manifest = json.load(open(mp, encoding='utf-8'))
        except Exception:
            manifest = {}

    ann = make_annotators()
    stores = {}
    print('渲染 %d 条 × %s → %s' % (len(items), kinds, a.outdir))
    for i, it in enumerate(items, 1):
        det = stores.get(it['seg'])
        if det is None:
            det = stores[it['seg']] = DetStore(it['seg'])
        print('[%2d/%d] %s  tid=%d  %d–%d（%.1f–%.1fs）  原记录 %s  %s cov=%.2f'
              % (i, len(items), it['track_id'], it['tid'], it['focus_start'], it['focus_end'],
                 it['focus_start'] / FPS, it['focus_end'] / FPS,
                 it['old_frame'] if it['old_usable'] else '不可用',
                 it['win_mode'], it['coverage']))
        clips = [render(it, det, ann, a.outdir, k,
                        a.width_focus if k == 'focus' else a.width_scan, a.crf)
                 for k in kinds]
        manifest[it['track_id']] = dict(item=it, clips=clips)

    json.dump(manifest, open(mp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    tot = sum(c['size_mb'] for v in manifest.values() for c in v['clips'])
    print('\nmanifest 共 %d 条 / %d 个片段 / %.0f MB → %s'
          % (len(manifest), sum(len(v['clips']) for v in manifest.values()), tot, mp))


if __name__ == '__main__':
    main()
