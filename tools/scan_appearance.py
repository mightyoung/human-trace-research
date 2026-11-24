#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""沿轨迹按固定步长做外观序列扫描，定位「同一个框内换了人」的时刻。

设计要点（实测决定）：
* 视频解码是本机瓶颈：随机 seek 9 帧/秒，顺序 grab 约 500 帧/秒、retrieve 约 33 帧/秒。
  所以**每段只做一次顺序前进**，只在需要的帧上 retrieve，其余 grab 跳过。
* 编码不是瓶颈（batch 8：45 张/秒），按帧聚合后批量编码。
* 缓存增量落盘，可断点续跑（`--resume`）。

用法：
  PY tools/scan_appearance.py --marks            # 25 条已标切点 + 12 条对照
  PY tools/scan_appearance.py --marks --control 0
  PY tools/scan_appearance.py --all --segs 323_train_510   # 全段扫描（供切分器用）
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracklet_split_v2 as S  # noqa: E402

FPS = 25.0
SEG_DIR = os.path.join(S.ROOT, 'media-resource/1012/segments')
CACHE = os.path.join(S.ROOT, 'output_crosscam_local/emb_cache/scan_appearance.npz')
DELIV = S.DELIV
DIP_MAX = 0.75          # 只记录低于此值的下探（再按 τ 过滤）


def plan_tracks(det, tids, step):
    """{frame: [(tid, box)]}；每条轨迹按 step 帧步长取点（吸附到最近的已有检测帧）。"""
    plan = collections.defaultdict(list)
    per_track = {}
    for tid in tids:
        fs = sorted(f for f, m in det.items() if tid in m)
        if len(fs) < 8:
            continue
        picked, t = [], fs[0]
        while t <= fs[-1]:
            i = int(np.searchsorted(fs, t))
            cand = [j for j in (i - 1, i) if 0 <= j < len(fs)]
            j = min(cand, key=lambda k: abs(fs[k] - t))
            f = fs[j]
            if f not in picked:
                picked.append(f)
            t = max(f, t) + step
        if len(picked) < 8:
            continue
        per_track[tid] = picked
        for f in picked:
            plan[f].append((tid, det[f][tid]))
    return plan, per_track


class Enc:
    def __init__(self, path=CACHE, batch=16):
        from torchreid.reid.utils import FeatureExtractor
        import torch
        self.torch = torch
        self.fe = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=os.path.join(S.ROOT, 'models/reid_osnet/'
                                    'osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_'
                                    'stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'),
            device='cpu', verbose=False)
        self.batch = batch
        self.path = path
        self.cache = {}
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            self.cache = {str(k): v for k, v in zip(z['keys'], z['vecs'])}
            print('[cache] %d 条' % len(self.cache), flush=True)

    def encode(self, keys, imgs):
        todo = [(k, im) for k, im in zip(keys, imgs) if k not in self.cache]
        for i in range(0, len(todo), self.batch):
            part = todo[i:i + self.batch]
            with self.torch.no_grad():
                v = self.fe([im for _, im in part]).detach().cpu().numpy().astype(np.float32)
            v /= np.maximum(1e-9, np.linalg.norm(v, axis=1, keepdims=True))
            for (k, _), row in zip(part, v):
                self.cache[k] = row

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        keys = list(self.cache)
        np.savez_compressed(self.path, keys=np.array(keys),
                            vecs=np.stack([self.cache[k] for k in keys]))
        print('[cache] 保存 %d 条 → %s' % (len(keys), self.path), flush=True)


def crop(img, box, margin=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    H, W = img.shape[:2]
    x1, x2 = max(0, int(x1)), min(W, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 10 or y2 - y1 < 24:
        return None
    return img[y1:y2, x1:x2].copy()


def scan_segment(seg, det, tids, enc, step, flush_every=400):
    import cv2
    plan, per_track = plan_tracks(det, tids, step)
    if not plan:
        return {}
    frames = sorted(plan)
    cap = cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4'))
    if not cap.isOpened():
        print('[skip] 打不开 %s' % seg, flush=True)
        return {}
    cur = frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
    t0 = time.time()
    pend_k, pend_im, n_new = [], [], 0
    done = 0
    for f in frames:
        while cur < f:
            cap.grab()
            cur += 1
        ok, img = cap.read()
        cur += 1
        done += 1
        if not ok:
            continue
        for tid, box in plan[f]:
            im = crop(img, box)
            if im is not None:
                pend_k.append('%s|%d|%d' % (seg, tid, f))
                pend_im.append(im)
        if len(pend_k) >= 64:
            enc.encode(pend_k, pend_im)
            n_new += len(pend_k)
            pend_k, pend_im = [], []
            if n_new >= min(flush_every, 200):
                enc.save()
                n_new = 0
        if done % 200 == 0:
            rate = done / max(1e-6, time.time() - t0)
            print('  %s %d/%d 帧 %.0f 帧/秒' % (seg, done, len(frames), rate), flush=True)
    if pend_k:
        enc.encode(pend_k, pend_im)
    cap.release()
    print('[%s] 采样帧 %d → 向量 %d，用时 %.1f 分钟'
          % (seg, len(frames), sum(len(v) for v in per_track.values()),
             (time.time() - t0) / 60), flush=True)
    return per_track


def series_stats(seg, tid, frames, enc, tau_hi=DIP_MAX):
    vs = []
    fs = []
    for f in frames:
        k = '%s|%d|%d' % (seg, tid, f)
        if k in enc.cache:
            fs.append(f)
            vs.append(enc.cache[k])
    if len(vs) < 6:
        return None
    cs = [float(vs[i] @ vs[i + 1]) for i in range(len(vs) - 1)]
    mids = [(fs[i] + fs[i + 1]) / 2.0 for i in range(len(cs))]
    ev = []
    for m, c in zip(mids, cs):
        if c >= tau_hi:
            continue
        if ev and m - ev[-1][0] <= 50:
            if c < ev[-1][1]:
                ev[-1] = [m, c]
        else:
            ev.append([m, c])
    best = min(ev, key=lambda e: e[1]) if ev else None
    return dict(n=len(vs), med=round(float(np.median(cs)), 3),
                best=[round(best[0]), round(best[1], 3)] if best else None,
                dips=[[round(m), round(c, 3)] for m, c in ev],
                dur_s=round((fs[-1] - fs[0]) / FPS, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--marks', action='store_true')
    ap.add_argument('--cls', default='usable',
                    help='要扫描的标注类别（逗号分隔）：usable,tail,head')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--segs', default='')
    ap.add_argument('--control', type=int, default=12)
    ap.add_argument('--step', type=int, default=20, help='采样步长（帧）；20 = 0.8s')
    ap.add_argument('--out', default=os.path.join(DELIV, 'results/scan_appearance.json'))
    a = ap.parse_args()

    want_cls = {c.strip() for c in a.cls.split(',') if c.strip()}
    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                          encoding='utf-8-sig')) if r['cls'] in want_cls]
    segs = ([s.strip() for s in a.segs.split(',') if s.strip()] if a.segs
            else sorted({r['seg'] for r in marks}) if a.marks else sorted(os.listdir(S.DET)))
    enc = Enc()
    out, ctrl = [], []
    rng = np.random.RandomState(3)
    for seg in segs:
        det = S.load_det(seg)
        if det is None:
            continue
        all_tids = {t for m in det.values() for t in m}
        if a.all:
            tids = sorted(all_tids)
        else:
            want = {int(r['track_id'].split('#')[1]) for r in marks if r['seg'] == seg}
            extra = []
            if a.control:
                pool = sorted(all_tids - want)
                extra = [pool[i] for i in rng.permutation(len(pool))[:a.control]]
            tids = sorted(want | set(extra))
        per_track = scan_segment(seg, det, tids, enc, a.step)
        enc.save()
        for tid, frames in per_track.items():
            st = series_stats(seg, tid, frames, enc)
            if st is None:
                continue
            row = dict(track='%s#%d' % (seg, tid), seg=seg, tid=tid, **st)
            m = [r for r in marks if r['track_id'] == row['track']]
            if m:
                row.update(true=int(float(m[0]['new_frame'])), cam=m[0]['cam'], kind='mark')
                out.append(row)
            else:
                row['kind'] = 'control'
                ctrl.append(row)
    enc.save()

    good = out
    print('\n可评估 %d/%d 条已标轨迹' % (len(good), len(marks)))
    print('%-6s %-8s %-10s %-12s' % ('tau', '召回', '事件/轨', '对照误切'))
    for tau in (0.55, 0.60, 0.65, 0.70, 0.75):
        h = sum(1 for x in good if x['best'] and abs(x['best'][0] - x['true']) <= 25
                and x['best'][1] < tau)
        ev = sum(sum(1 for _, c in x['dips'] if c < tau) for x in good) / max(1, len(good))
        fp = sum(1 for x in ctrl if any(c < tau for _, c in x['dips']))
        print('%-6.2f %2d/%-2d %4.0f%%   %6.2f      %d/%d (%.0f%%)'
              % (tau, h, len(good), 100 * h / max(1, len(good)), ev, fp, len(ctrl),
                 100 * fp / max(1, len(ctrl))))
    print('\n逐条：')
    for x in sorted(good, key=lambda y: y['track']):
        d = (x['best'][0] - x['true']) / FPS if x['best'] else None
        print('  %-22s dur=%6.1fs 点 %3d 事件 %d best=%s 误差 %s'
              % (x['track'], x['dur_s'], x['n'], len(x['dips']), x['best'],
                 ('%+.1fs' % d) if d is not None else 'NA'))
    json.dump(dict(marks=out, control=ctrl), open(a.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n→ %s' % a.out)


if __name__ == '__main__':
    main()
