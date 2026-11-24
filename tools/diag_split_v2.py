#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐条诊断两段式切分：25 条可用切点上，损失发生在粗筛还是精定位。

输出 `results/split_v2_diag.csv`，每行一条：
  ratio        粗筛 min/median（<0.60 才进精定位）
  coarse_off   粗筛候选点 - 真值（秒）
  refined      精定位给出的切点（帧）
  ref_off      精定位切点 - 真值（秒）
  geo_off      最近几何切点 - 真值（秒，正=在真值之后）
  hit_*        各自是否落在 ±1s
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
from eval_appearance_rule import Enc  # noqa: E402

FPS = 25.0
DELIV = S.DELIV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(DELIV, 'results/split_v2_diag.csv'))
    a = ap.parse_args()

    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                          encoding='utf-8-sig')) if r['cls'] == 'usable']
    coarse = S.load_coarse()
    enc_obj = Enc()
    import cv2
    caps = {}
    dets = {}
    rows = []
    for r in marks:
        seg, tid = r['seg'], int(r['track_id'].split('#')[1])
        f_true = int(float(r['new_frame']))
        det = dets.setdefault(seg, S.load_det(seg))
        gcuts, fs = S.geo_cuts(det, tid)
        ser = coarse.get((seg, tid), [])
        # 粗筛：全部低于阈值的跳（不只 argmin），便于看真值那一跳排第几
        info = dict(track=r['track_id'], cam=r['cam'], seg=seg, true=f_true,
                    n_coarse=len(ser), ratio='', coarse_f='', coarse_off='',
                    rank=f'{len(ser)}', refined='', ref_off='',
                    geo_off='', n_geo=len(gcuts))
        if len(ser) >= 2:
            cs = np.array([float(ser[i][1] @ ser[i + 1][1]) for i in range(len(ser) - 1)])
            med = float(np.median(cs))
            mids = [(ser[i][0] + ser[i + 1][0]) // 2 for i in range(len(cs))]
            j = int(np.argmin(cs))
            info['ratio'] = round(cs[j] / med, 3) if med > 0 else ''
            info['coarse_f'] = mids[j]
            info['coarse_off'] = round((mids[j] - f_true) / FPS, 1)
            # 真值那一跳在序列里的名次（按余弦升序）
            k = min(range(len(mids)), key=lambda i: abs(mids[i] - f_true))
            info['rank'] = int((cs < cs[k]).sum()) + 1
        if info['ratio'] != '' and info['ratio'] < S.TAU_COARSE_RATIO:
            cap = caps.setdefault(seg, cv2.VideoCapture(
                os.path.join(S.ROOT, 'media-resource/1012/segments', seg + '.mp4')))
            c = S.refine(seg, tid, info['coarse_f'], lambda s, t, f, im: enc_obj.get(s, t, f, im),
                         cap, det)
            info['refined'] = c if c is not None else ''
            info['ref_off'] = round((c - f_true) / FPS, 1) if c is not None else ''
        if gcuts:
            g = min(gcuts, key=lambda x: abs(x - f_true))
            info['geo_off'] = round((g - f_true) / FPS, 1)
        for key, off in (('hit_coarse', info['coarse_off']), ('hit_refined', info['ref_off']),
                         ('hit_geo', info['geo_off'])):
            v = off
            info[key] = int(v != '' and abs(float(v)) <= 1.0) if isinstance(v, (int, float)) else 0
        rows.append(info)
    for c in caps.values():
        c.release()
    enc_obj.save()

    with open(a.out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    trig = [x for x in rows if x['ratio'] != '' and x['ratio'] < S.TAU_COARSE_RATIO]
    print('25 条：粗筛触发 %d｜精定位给出切点 %d' % (len(trig), sum(1 for x in rows if x['refined'] != '')))
    print('命中：粗筛 %d｜精定位 %d｜几何 %d｜几何∪精定位 %d｜三者∪ %d'
          % (sum(x['hit_coarse'] for x in rows), sum(x['hit_refined'] for x in rows),
             sum(x['hit_geo'] for x in rows),
             sum(1 for x in rows if x['hit_refined'] or x['hit_geo']),
             sum(1 for x in rows if x['hit_refined'] or x['hit_geo'] or x['hit_coarse'])))
    print('\n逐条（off = 与真值的秒差；rank = 真值那一跳在粗筛里的名次）')
    for x in rows:
        print('  %-22s ratio=%-6s rank=%-2s coarse=%+6s refined=%+6s geo=%+6s'
              % (x['track'], x['ratio'], x['rank'], x['coarse_off'] or 'NA',
                 x['ref_off'] or 'NA', x['geo_off'] or 'NA'))
    miss = [x['track'] for x in rows if not (x['hit_refined'] or x['hit_geo'])]
    print('\n仍漏 %d 条：%s' % (len(miss), miss))
    json.dump(dict(rows=rows, n=n), open(os.path.join(DELIV, 'results/split_v2_diag.json'),
                                         'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
