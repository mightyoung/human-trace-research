#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全轨密采样（1s 步长、无窗口）能否定位换人点——两段式失败的补救方案验证。

与 `diag_appearance_localize.py` 的区别：那个在 **真值 ±4s 窗口**里找（等于作弊），
本脚本沿整条轨迹每 1s 采一帧，取**全轨最低一跳**，是真正可上线的定位方式。

输出 `results/dense_appearance.json`：
  每条：轨迹、真值帧、全轨跳数、最低跳位置与误差、各 τ 下的事件数与是否命中
对照：未标注轨迹同样处理，给误切率。

用法：
  PY tools/diag_dense_appearance.py            # 25 条 + 12 条对照
  PY tools/diag_dense_appearance.py --control 30
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracklet_split_v2 as S  # noqa: E402

FPS = 25.0
STEP = 25              # 1 秒
DELIV = S.DELIV
SEG_DIR = os.path.join(S.ROOT, 'media-resource/1012/segments')


def crop_from(img, box, margin=0.08):
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


def samples_for(det, tid, step=STEP):
    fs = sorted(f for f, m in det.items() if tid in m)
    if len(fs) < 12:
        return [], {}
    out, box = [], {}
    t = fs[0]
    while t <= fs[-1]:
        i = int(np.searchsorted(fs, t))
        cand = [j for j in (i - 1, i) if 0 <= j < len(fs)]
        j = min(cand, key=lambda k: abs(fs[k] - t))
        f = fs[j]
        if f not in box:
            box[f] = det[f][tid]
            out.append(f)
        t = max(f, t) + step
    return sorted(out), box


class BatchedEnc:
    def __init__(self, cache_path, batch=32):
        from torchreid.reid.utils import FeatureExtractor
        import torch
        self.torch = torch
        self.fe = FeatureExtractor(model_name='osnet_x1_0',
                                  model_path=os.path.join(S.ROOT, 'models/reid_osnet/'
                                          'osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_'
                                          'stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'),
                                  device='cpu', verbose=False)
        self.batch = batch
        self.cache = {}
        self.path = cache_path
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            self.cache = {str(k): v for k, v in zip(z['keys'], z['vecs'])}
            print('[cache] %d 条' % len(self.cache))

    def get(self, keys, imgs):
        todo = [(k, im) for k, im in zip(keys, imgs) if k not in self.cache]
        if todo:
            with self.torch.no_grad():
                for i in range(0, len(todo), self.batch):
                    part = todo[i:i + self.batch]
                    v = self.fe([im for _, im in part]).detach().cpu().numpy().astype(np.float32)
                    v /= np.maximum(1e-9, np.linalg.norm(v, axis=1, keepdims=True))
                    for (k, _), row in zip(part, v):
                        self.cache[k] = row
        return [self.cache[k] for k in keys]

    def save(self):
        keys = list(self.cache)
        np.savez_compressed(self.path, keys=np.array(keys),
                            vecs=np.stack([self.cache[k] for k in keys]))
        print('[cache] 保存 %d 条 → %s' % (len(keys), self.path))


def series_one(cap, seg, tid, det, enc):
    fr, box = samples_for(det, tid)
    if len(fr) < 6:
        return None
    import cv2
    keys = ['%s|%d|%d' % (seg, tid, f) for f in fr]
    need = [k for k in keys if k not in enc.cache]
    if need:
        imgs = []
        pos = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr[0])
        cur = fr[0]
        for f in fr:
            while cur < f:
                cap.grab()
                cur += 1
            ok, img = cap.read()
            cur += 1
            imgs.append(crop_from(img, box[f]) if ok else None)
        keep = [(f, k, im) for f, k, im in zip(fr, keys, imgs) if im is not None]
        if len(keep) < 6:
            return None
        vs = enc.get([k for _, k, _ in keep], [im for _, _, im in keep])
        fr2 = [f for f, _, _ in keep]
        cs = [float(vs[i] @ vs[i + 1]) for i in range(len(vs) - 1)]
        mids = [(fr2[i] + fr2[i + 1]) / 2.0 for i in range(len(cs))]
    else:
        vs = [enc.cache[k] for k in keys]
        cs = [float(vs[i] @ vs[i + 1]) for i in range(len(vs) - 1)]
        mids = [(fr[i] + fr[i + 1]) / 2.0 for i in range(len(cs))]
    ev = []
    for m, c in zip(mids, cs):
        if c >= 0.75:
            continue
        if ev and m - ev[-1][0] <= 50:
            if c < ev[-1][1]:
                ev[-1] = [m, c]
        else:
            ev.append([m, c])
    best = min(ev, key=lambda e: e[1]) if ev else None
    return dict(n_jump=len(cs), med=round(float(np.median(cs)), 3),
                dips=[[round(m), round(c, 3)] for m, c in ev],
                best=[round(best[0]), round(best[1], 3)] if best else None,
                dur_s=round((fr[-1] - fr[0]) / FPS, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', type=int, default=12)
    ap.add_argument('--out', default=os.path.join(DELIV, 'results/dense_appearance.json'))
    a = ap.parse_args()

    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                          encoding='utf-8-sig')) if r['cls'] == 'usable']
    enc = BatchedEnc(os.path.join(S.ROOT, 'output_crosscam_local/emb_cache/dd_dense_1s.npz'))
    import cv2
    dets, caps = {}, {}
    out = []
    for r in marks:
        seg, tid = r['seg'], int(r['track_id'].split('#')[1])
        dets.setdefault(seg, S.load_det(seg))
        caps.setdefault(seg, cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4')))
        s = series_one(caps[seg], seg, tid, dets[seg], enc)
        if s is None:
            out.append(dict(track=r['track_id'], cam=r['cam'], ok=False))
            continue
        out.append(dict(track=r['track_id'], cam=r['cam'], ok=True,
                        true=int(float(r['new_frame'])), **s))

    # 对照：未标注轨迹
    rng = np.random.RandomState(3)
    ctrl = []
    for seg in sorted({r['seg'] for r in marks}):
        dets.setdefault(seg, S.load_det(seg))
        caps.setdefault(seg, cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4')))
        labeled = {int(r['track_id'].split('#')[1]) for r in marks if r['seg'] == seg}
        cand = [t for t in {t for m in dets[seg].values() for t in m} if t not in labeled]
        for tid in rng.permutation(len(cand))[:a.control]:
            s = series_one(caps[seg], cand[tid], seg, dets[seg], enc)
            if s:
                ctrl.append(dict(track='%s#%d' % (seg, cand[tid]), **s))
    for c in caps.values():
        c.release()
    enc.save()

    def hit_at(x, tau):
        if not x.get('best'):
            return 0
        return int(abs(x['best'][0] - x['true']) <= 25 and x['best'][1] < tau)

    good = [x for x in out if x.get('ok')]
    print('可评估 %d/%d 条（有足够检测点）' % (len(good), len(out)))
    print('\n%-6s %-8s %-10s %-12s' % ('τ', '召回', '事件/轨', '对照误切率'))
    for tau in (0.55, 0.60, 0.65, 0.70, 0.75):
        h = sum(hit_at(x, tau) for x in good)
        ev = sum(sum(1 for m, c in x['dips'] if c < tau) for x in good) / max(1, len(good))
        fp = sum(1 for x in ctrl if x['dips'] and min(c for _, c in x['dips']) < tau)
        print('%-6.2f %2d/%-2d %4.0f%%   %6.2f      %d/%d (%.0f%%)'
              % (tau, h, len(good), 100 * h / max(1, len(good)), ev, fp, len(ctrl),
                 100 * fp / max(1, len(ctrl))))
    print('\n逐条（best 与真值的差）：')
    for x in good:
        d = (x['best'][0] - x['true']) / FPS if x['best'] else None
        print('  %-22s dur=%6.1fs 跳数 %2d 事件 %d  best=%s 误差 %s'
              % (x['track'], x['dur_s'], x['n_jump'], len(x['dips']),
                 x['best'], ('%+.1fs' % d) if d is not None else 'NA'))
    json.dump(dict(marks=out, control=ctrl), open(a.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n→ %s' % a.out)


if __name__ == '__main__':
    main()
