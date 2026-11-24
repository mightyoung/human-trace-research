#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""比较「选哪一跳」的三种策略（只用已编码向量，不解码视频）。

动机：全轨密扫时最低相邻余弦常常选错事件（长轨上转身/遮挡的跳比起换人还低）。
策略：
  A 相邻最低      argmin_i  c[i]                       （现用）
  B 跨窗最低      argmin_i  v[i-w] · v[i+w]            （隔 ±w 个样本直接比，能过滤"短暂遮挡"）
  C 相邻+跨窗     argmin_i  min(c[i], v[i-w]·v[i+w])

输出 `results/changepoint_compare.json`。
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DELIV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'deliverable/data-request-v2')
CACHE = os.path.join(os.path.dirname(os.path.dirname(DELIV)),
                     'output_crosscam_local/emb_cache/scan_appearance.npz')
FPS = 25.0


def load_series():
    z = np.load(CACHE, allow_pickle=True)
    d = {}
    for k, v in zip(z['keys'], z['vecs']):
        seg, tid, f = str(k).split('|')
        d.setdefault((seg, int(tid)), []).append((int(f), v))
    for k in d:
        d[k].sort()
    return d


def scores(series, w):
    fs = [f for f, _ in series]
    V = np.stack([v for _, v in series])
    c = np.array([float(V[i] @ V[i + 1]) for i in range(len(V) - 1)])
    mids = np.array([(fs[i] + fs[i + 1]) / 2.0 for i in range(len(c))])
    far = np.ones(len(c))
    for i in range(len(c)):
        a, b = i - w, i + w
        if 0 <= a and b < len(V):
            far[i] = float(V[a] @ V[b])
        else:
            far[i] = np.nan
    return c, far, mids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=int, default=2, help='跨窗样本数（2 ≈ ±1.6s）')
    ap.add_argument('--out', default=os.path.join(DELIV, 'results/changepoint_compare.json'))
    a = ap.parse_args()

    gt = {r['track_id']: r for r in csv.DictReader(
        open(os.path.join(DELIV, 'results/m1a_switch_review.csv'), encoding='utf-8-sig'))}
    ser = load_series()
    print('缓存轨迹 %d 条' % len(ser))

    res = []
    for k, r in gt.items():
        if r['cls'] not in ('usable', 'tail', 'head') or not (r['new_frame'] or '').strip():
            continue
        seg = r['seg']
        tid = int(r['track_id'].split('#')[1])
        s = ser.get((seg, tid))
        if not s or len(s) < 8:
            continue
        c, far, mids = scores(s, a.w)
        true = int(float(r['new_frame']))
        picks = {}
        picks['A'] = int(mids[int(np.argmin(c))])
        if np.isfinite(far).sum() >= 3:
            picks['B'] = int(mids[int(np.nanargmin(far))])
            m = np.minimum(c, np.where(np.isfinite(far), far, 1.0))
            picks['C'] = int(mids[int(np.argmin(m))])
        res.append(dict(track=k, cls=r['cls'], true=true,
                        pos=round((true - s[0][0]) / max(1, s[-1][0] - s[0][0]), 3),
                        picks=picks, min_c=round(float(np.min(c)), 3),
                        min_far=round(float(np.nanmin(far)), 3) if np.isfinite(far).any() else None))

    def rate(method, cls_set, pos_pred=None):
        sub = [x for x in res if x['cls'] in cls_set and method in x['picks']
               and (pos_pred is None or pos_pred(x))]
        h = [x for x in sub if abs(x['picks'][method] - x['true']) <= 25]
        return len(h), len(sub), [x['track'] for x in h]

    print('\n%-8s %-22s %-22s %-22s' % ('策略', 'usable(n)', 'tail(n)', 'usable 且 pos<0.9'))
    for m in ('A', 'B', 'C'):
        u = rate(m, ('usable',))
        t = rate(m, ('tail',))
        mid = rate(m, ('usable',), lambda x: x['pos'] < 0.9)
        print('%-8s %-22s %-22s %-22s'
              % (m, '%d/%d (%.0f%%)' % (u[0], u[1], 100 * u[0] / max(1, u[1])),
                 '%d/%d (%.0f%%)' % (t[0], t[1], 100 * t[0] / max(1, t[1])),
                 '%d/%d (%.0f%%)' % (mid[0], mid[1], 100 * mid[0] / max(1, mid[1]))))

    print('\n逐条（误差秒；A=相邻最低 B=跨窗最低 C=两者取小）')
    for x in sorted(res, key=lambda y: (y['cls'], y['pos'])):
        cells = []
        for m in ('A', 'B', 'C'):
            if m in x['picks']:
                e = (x['picks'][m] - x['true']) / FPS
                cells.append('%s%+7.1f%s' % (m, e, '*' if abs(e) <= 1 else ' '))
            else:
                cells.append('%s   -   ' % m)
        print('  %-22s %-7s pos=%.2f  %s' % (x['track'], x['cls'], x['pos'], '  '.join(cells)))

    json.dump(dict(w=a.w, rows=res), open(a.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n→ %s' % a.out)


if __name__ == '__main__':
    main()
