#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B4：外观能不能**定位**换人点？——在 ±4s 内每 0.4s 取一帧编码，看余弦最低点落在哪。

这是「ReID 通道值不值得做」的决定性小实验，成本约 4 分钟 CPU：
对每个 usable 换人点，在 [f−4s, f+4s] 每 0.4s 取该 tid 的框 → OSNet 编码 → 相邻余弦，
最低的那一跳就是外观给出的切点。统计它离人工标点的距离。

判读：
  命中率高（≥60% 落在 ±0.5s）→ 外观通道可用，值得写进 M1
  命中率低                    → 「遮挡让 ReID 失效」成立，M1 只能报几何天花板

用法：PY tools/diag_appearance_localize.py [--span 4] [--step 10]
输出：results/appearance_localize.csv + 汇总
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
DET = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd')
SEG_DIR = os.path.join(ROOT, 'media-resource/1012/segments')
OUT = os.path.join(DELIV, 'results/appearance_localize.csv')
W = ('models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_'
     'stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
FPS = 25.0


def load_det(seg):
    d = {}
    for line in open(os.path.join(DET, seg, '%s_detections.jsonl' % seg), encoding='utf-8'):
        r = json.loads(line)
        d[r['frame']] = {int(t): b for t, b in zip(r['track_ids'], r['boxes'])}
    return d


def nearest(det, frame):
    import bisect
    fs = sorted(det)
    i = bisect.bisect_left(fs, frame)
    c = [j for j in (i - 1, i) if 0 <= j < len(fs)]
    j = min(c, key=lambda k: abs(fs[k] - frame))
    return fs[j], det[fs[j]]


def crop(cap, frame, box, margin=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok:
        return None
    H, Wd = img.shape[:2]
    x1, x2 = max(0, int(x1)), min(Wd, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 24:
        return None
    return img[y1:y2, x1:x2].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--span', type=float, default=4.0)
    ap.add_argument('--step', type=int, default=10)
    a = ap.parse_args()

    from torchreid.reid.utils import FeatureExtractor
    import torch
    torch.set_flush_denormal(True)
    fe = FeatureExtractor(model_name='osnet_x1_0', model_path=os.path.join(ROOT, W),
                          device='cpu', verbose=False)

    def enc(crops):
        with torch.no_grad():
            v = fe(crops).detach().cpu().numpy().astype(np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        return v

    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                          encoding='utf-8-sig'))
             if r['new_frame'] and r['cls'] == 'usable']
    by_seg = {}
    for r in marks:
        by_seg.setdefault(r['seg'], []).append(r)

    rows = []
    for seg, rs in by_seg.items():
        det = load_det(seg)
        cap = cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4'))
        for r in rs:
            tid, f = int(r['track_id'].split('#')[1]), int(float(r['new_frame']))
            lo, hi = int(f - a.span * FPS), int(f + a.span * FPS)
            frames, imgs = [], []
            for fr in range(lo, hi + 1, a.step):
                nf, m = nearest(det, fr)
                if tid not in m or abs(nf - fr) > a.step:
                    frames.append(fr)
                    imgs.append(None)
                    continue
                c = crop(cap, fr, m[tid])
                frames.append(fr)
                imgs.append(c)
            keep = [(fr, im) for fr, im in zip(frames, imgs) if im is not None]
            if len(keep) < 5:
                rows.append(dict(track_id=r['track_id'], status='too_few'))
                continue
            V = enc([im for _, im in keep])
            cs = [float(V[i] @ V[i + 1]) for i in range(len(V) - 1)]
            mid = [(keep[i][0] + keep[i + 1][0]) / 2.0 for i in range(len(cs))]
            j = int(np.argmin(cs))
            rows.append(dict(
                track_id=r['track_id'], cam=r['cam'], cls=r['cls'], status='ok',
                switch_frame=f, argmin_frame=int(mid[j]),
                err_s=round((mid[j] - f) / FPS, 2),
                cos_min=round(cs[j], 3),
                cos_median=round(float(np.median(cs)), 3),
                n=int(len(cs)), pos=r['pos_ratio']))
        cap.release()

    ok = [x for x in rows if x['status'] == 'ok']
    with open(OUT, 'w', encoding='utf-8-sig', newline='') as fh:
        keys = ['track_id', 'cam', 'cls', 'status', 'switch_frame', 'argmin_frame',
                'err_s', 'cos_min', 'cos_median', 'n', 'pos']
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in rows:
            w.writerow({k: x.get(k, '') for k in keys})

    if ok:
        errs = [abs(x['err_s']) for x in ok]
        print('可定位 %d 条；|误差| 中位 %.2fs' % (len(ok), float(np.median(errs))))
        for th in (0.5, 1.0, 2.0):
            n = sum(1 for e in errs if e <= th)
            print('  ±%.1fs 内命中 %d/%d（%.0f%%）' % (th, n, len(ok), 100 * n / len(ok)))
        print('  余弦最低值中位 %.3f｜同窗口其它跳的中位 %.3f'
              % (float(np.median([x['cos_min'] for x in ok])),
                 float(np.median([x['cos_median'] for x in ok]))))
        byc = {}
        for x in ok:
            byc.setdefault(x['cam'], []).append(abs(x['err_s']))
        print('  按机位 |误差| 中位：%s' % {k: round(float(np.median(v)), 2) for k, v in sorted(byc.items())})
        for x in sorted(ok, key=lambda y: abs(y['err_s'])):
            print('    %-22s 误差 %+5.2fs  最低 %.2f（窗口其它 %.2f）'
                  % (x['track_id'], x['err_s'], x['cos_min'], x['cos_median']))
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
