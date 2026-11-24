#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B 合并版切分器：几何（断开/跳框/尺寸）+ 外观（两段式），输出子轨迹。

两段式外观（依据 docs/m1-b-appearance-2026-10-23.md 的实测）：
  粗筛   每轨 8 个样本（`encode_dd_coarse.py` 已编好）→ min/median < 0.60 → 候选
  精定位 候选轨在粗定位点 ±8s 内每 0.4s 采样 → 下探聚簇 → 最低点 < 0.65 才切

几何（独立出切点，与外观 2s 内合并）：
  G1 断开  相邻观测间隔 > 2s（原 1s 太密，碎片爆炸）
  G2 跳框  相邻样本 IoU < 0.40
  G3 尺寸  框宽 > 前 5 个样本中位的 1.5 倍

输出：
  `output_crosscam_local/split_v2/<seg>_subtracks.jsonl`
  `results/split_v2_eval.json`（25 条命中、碎片倍率、过短率）

用法：
  PY tools/tracklet_split_v2.py --segs 323_train_510,305_train_529
  PY tools/tracklet_split_v2.py --all
  PY tools/tracklet_split_v2.py --all --eval
"""
import argparse
import collections
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
DET = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd')
COARSE = os.path.join(ROOT, 'output_crosscam_local/emb_cache/dd_coarse_8.npz')
OUTDIR = os.path.join(ROOT, 'output_crosscam_local/split_v2')
MIN_SAMPLES = 6

# —— 参数（全部来自实测，见 docs/m1-b-appearance-2026-10-23.md）——
TAU_COARSE_RATIO = 0.60     # 粗筛 min/median
TAU_REFINE = 0.65           # 精定位绝对余弦
GAP_S = 2.0
JUMP_IOU = 0.40
GROW = 1.5
MERGE_S = 2.0


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_det(seg):
    p = os.path.join(DET, seg, '%s_detections.jsonl' % seg)
    if not os.path.exists(p):
        return None
    d = {}
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        d[r['frame']] = {int(t): b for t, b in zip(r['track_ids'], r['boxes'])}
    return d


def load_coarse():
    if not os.path.exists(COARSE):
        raise SystemExit('缺少粗筛向量：先跑 tools/encode_dd_coarse.py')
    z = np.load(COARSE, allow_pickle=True)
    out = collections.defaultdict(list)
    for k, v in zip(z['keys'], z['vecs']):
        seg, tid, frame = str(k).split('|')
        out[(seg, int(tid))].append((int(frame), v))
    for k in out:
        out[k].sort()
    return out


def geo_cuts(det_seg, tid, fps=25.0):
    fs = sorted(f for f, m in det_seg.items() if tid in m)
    cuts = []
    if len(fs) < 3:
        return cuts, fs
    gap_thr = int(GAP_S * fps)
    for i, (a, b) in enumerate(zip(fs, fs[1:])):
        da, db = det_seg[a][tid], det_seg[b][tid]
        if b - a > gap_thr:
            cuts.append((a + b) // 2)
            continue
        if b - a <= 30 and iou(da, db) < JUMP_IOU:
            cuts.append(b)
            continue
        if i >= 3:
            ref = statistics.median([det_seg[x][tid][2] - det_seg[x][tid][0] for x in fs[max(0, i - 5):i]])
            if (db[2] - db[0]) > GROW * ref:
                cuts.append(b)
    return cuts, fs


def merge_cuts(cuts, gap_frames):
    out = []
    for c in sorted(cuts):
        if not out or c - out[-1] > gap_frames:
            out.append(c)
    return out


def coarse_candidates(series):
    """返回 [(候选帧, 比值)]，只保留比值最低的那一跳。"""
    if len(series) < 4:
        return []
    cs = [float(series[i][1] @ series[i + 1][1]) for i in range(len(series) - 1)]
    med = float(np.median(cs))
    if med <= 0:
        return []
    j = int(np.argmin(cs))
    ratio = cs[j] / med
    if ratio < TAU_COARSE_RATIO:
        mid = (series[j][0] + series[j + 1][0]) // 2
        return [(mid, ratio)]
    return []


def refine(seg, tid, coarse_frame, enc, cap, det_seg, span_s=8.0, step=10, fps=25.0):
    """在粗定位点 ±span 内每 step 帧采样，聚簇下探，返回切点或 None。"""
    lo, hi = int(coarse_frame - span_s * fps), int(coarse_frame + span_s * fps)
    fs = sorted(det_seg)
    import bisect
    frames, vecs = [], []
    for f in range(lo, hi + 1, step):
        i = bisect.bisect_left(fs, f)
        cand = [j for j in (i - 1, i) if 0 <= j < len(fs)]
        if not cand:
            continue
        j = min(cand, key=lambda k: abs(fs[k] - f))
        nf = fs[j]
        if abs(nf - f) > step or tid not in det_seg[nf]:
            continue
        im = enc_crop(cap, f, det_seg[nf][tid])
        if im is None:
            continue
        frames.append(f)
        vecs.append(enc(seg, tid, f, im))
    if len(vecs) < 4:
        return None
    cs = [float(vecs[i] @ vecs[i + 1]) for i in range(len(vecs) - 1)]
    mids = [(frames[i] + frames[i + 1]) / 2.0 for i in range(len(cs))]
    # 聚簇
    ev = []
    for m, c in zip(mids, cs):
        if c >= TAU_REFINE:
            continue
        if ev and m - ev[-1][0] <= 50:
            if c < ev[-1][1]:
                ev[-1] = [m, c]
        else:
            ev.append([m, c])
    if not ev:
        return None
    best = min(ev, key=lambda e: e[1])
    return int(best[0])


def enc_crop(cap, frame, box, margin=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok:
        return None
    H, Wd = img.shape[:2]
    x1, x2 = max(0, int(x1)), min(Wd, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 10 or y2 - y1 < 24:
        return None
    return img[y1:y2, x1:x2].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--eval', action='store_true', help='对 25 条可用切点算命中')
    ap.add_argument('--no-appear', action='store_true', help='只跑几何（对照用）')
    ap.add_argument('--only-marked', action='store_true',
                    help='只处理 m1a 里 usable 的那 25 条（快速端到端验证）')
    a = ap.parse_args()
    segs = ([s.strip() for s in a.segs.split(',') if s.strip()] if a.segs
            else (sorted(os.listdir(DET)) if a.all else ['323_train_510']))
    os.makedirs(OUTDIR, exist_ok=True)
    coarse = {} if a.no_appear else load_coarse()

    import cv2
    from eval_appearance_rule import Enc
    enc_obj = None if a.no_appear else Enc()
    enc = (lambda s, t, f, im: enc_obj.get(s, t, f, im)) if enc_obj else None

    marks = [r for r in csv_read(os.path.join(DELIV, 'results/m1a_switch_review.csv'))
             if r['cls'] == 'usable'] if (a.eval or a.only_marked) else []
    marked = collections.defaultdict(set)
    for r in marks:
        marked[r['seg']].add(int(r['track_id'].split('#')[1]))

    stats = {}
    for seg in segs:
        det = load_det(seg)
        if det is None:
            print('[skip] %s' % seg)
            continue
        tids = sorted({t for m in det.values() for t in m})
        if a.only_marked:
            tids = [t for t in tids if t in marked.get(seg, ())]
        cap = cv2.VideoCapture(os.path.join(ROOT, 'media-resource/1012/segments', seg + '.mp4'))
        subtracks = []
        n_cut = 0
        for tid in tids:
            gcuts, fs = geo_cuts(det, tid)
            if not fs:
                continue
            acuts = []
            if not a.no_appear:
                series = coarse.get((seg, tid), [])
                for cand, _r in coarse_candidates(series):
                    c = refine(seg, tid, cand, enc, cap, det, )
                    if c is not None:
                        acuts.append(c)
            cuts = merge_cuts(gcuts + acuts, int(MERGE_S * 25))
            n_cut += len(cuts)
            bounds = [fs[0]] + cuts + [fs[-1] + 10]
            for k in range(len(bounds) - 1):
                f0, f1 = bounds[k], bounds[k + 1] - 10
                if f1 < f0:
                    continue
                ns = len([f for f in fs if f0 <= f <= f1])
                if ns == 0:
                    continue
                subtracks.append(dict(tid=tid, part=k, f0=f0, f1=f1, n_samples=ns,
                                      cut_src='geo' if k == 0 else 'mix'))
        cap.release()
        short = sum(1 for s in subtracks if s['n_samples'] < MIN_SAMPLES)
        stats[seg] = dict(tracks=len(tids), subtracks=len(subtracks),
                          cuts=n_cut, ratio=round(len(subtracks) / max(1, len(tids)), 2),
                          short=short, short_rate=round(short / max(1, len(subtracks)), 3))
        print('[%s] 轨迹 %d → 子轨迹 %d（%.2f×）｜切点 %d｜<6 样本 %d（%.0f%%）'
              % (seg, len(tids), len(subtracks), stats[seg]['ratio'], n_cut,
                 short, 100 * stats[seg]['short_rate']))
        with open(os.path.join(OUTDIR, '%s_subtracks.jsonl' % seg), 'w', encoding='utf-8') as fh:
            for s in subtracks:
                fh.write(json.dumps(s, ensure_ascii=False) + '\n')

    if a.eval:
        hit = 0
        for r in marks:
            seg = r['seg']
            p = os.path.join(OUTDIR, '%s_subtracks.jsonl' % seg)
            if not os.path.exists(p):
                continue
            tid = int(r['track_id'].split('#')[1])
            f = int(float(r['new_frame']))
            cuts = set()
            for line in open(p, encoding='utf-8'):
                s = json.loads(line)
                if s['tid'] == tid and s['part'] > 0:
                    cuts.add(s['f0'])
            if any(abs(c - f) <= 25 for c in cuts):
                hit += 1
        stats['eval'] = dict(hit=hit, n=len(marks), recall=round(hit / max(1, len(marks)), 3))
        print('\n25 条可用切点：命中 %d/%d（%.0f%%）'
              % (hit, len(marks), 100 * hit / max(1, len(marks))))

    os.makedirs(os.path.join(DELIV, 'results'), exist_ok=True)
    json.dump(stats, open(os.path.join(DELIV, 'results/split_v2_eval.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('→ %s' % os.path.join(DELIV, 'results/split_v2_eval.json'))


def csv_read(p):
    import csv
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))


if __name__ == '__main__':
    main()
