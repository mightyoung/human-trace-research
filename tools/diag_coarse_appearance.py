#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B6：粗筛阶段是否可行——整轨只取 8 个样本，换人点那一跳还压得下阈值吗？

为什么问这个：全区 883 轨不可能按 0.4s 采样（要十几万张裁剪）。计划是两段式：
  粗筛  每轨 8 个样本（整轨均分）→ 只看"有没有一跳特别低"，负责**发现**
  精定位 只对被粗筛标出的轨迹，在粗定位点 ±8s 内每 0.4s 采样 → 负责**切准**
本脚本验证第一段：8 样本时，最低那一跳是否仍然 ≤τ、位置是否在切点附近。

用法：PY tools/diag_coarse_appearance.py
输出：results/coarse_appearance.json
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_appearance_rule as E   # 复用检测读取 / 裁剪 / 编码缓存

ROOT = E.ROOT
DELIV = E.DELIV
OUT = os.path.join(DELIV, 'results/coarse_appearance.json')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', type=int, default=0, help='额外取 N 条未标注轨迹做误触发对照')
    a = ap.parse_args()
    enc = E.Enc()
    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                          encoding='utf-8-sig')) if r['cls'] == 'usable']
    by = {}
    for r in marks:
        by.setdefault(r['seg'], []).append(r)

    res = []
    for seg, rs in by.items():
        det = E.load_det(seg)
        fs = sorted(det)
        cap = cv2.VideoCapture(os.path.join(E.SEG_DIR, seg + '.mp4'))
        for r in rs:
            tid, f = int(r['track_id'].split('#')[1]), int(float(r['new_frame']))
            obs = [x for x in fs if tid in det[x]]
            if len(obs) < 8:
                continue
            idx = np.linspace(0, len(obs) - 1, 8).astype(int)
            fr = [obs[i] for i in idx]
            vs = []
            for x in fr:
                im = E.crop(cap, x, det[x][tid])
                vs.append(None if im is None else enc.get(seg, tid, x, im))
            keep = [(a, b) for a, b in zip(fr, vs) if b is not None]
            if len(keep) < 4:
                continue
            cs = [float(keep[i][1] @ keep[i + 1][1]) for i in range(len(keep) - 1)]
            mid = [(keep[i][0] + keep[i + 1][0]) / 2.0 for i in range(len(cs))]
            j = int(np.argmin(cs))
            res.append(dict(track=r['track_id'], f=f, min_cos=round(cs[j], 3),
                            at=int(mid[j]), err_s=round((mid[j] - f) / 25, 1),
                            second=round(sorted(cs)[1], 3) if len(cs) > 1 else None,
                            med=round(float(np.median(cs)), 3)))
        cap.release()
    # 对照：未标注轨迹同样取 8 样本，看有多少条会低于 τ（误触发率）
    ctrl = []
    if a.control:
        rng = np.random.RandomState(2)
        for seg in by:
            det = E.load_det(seg)
            fs = sorted(det)
            labeled = {int(r['track_id'].split('#')[1]) for r in by[seg]}
            tids = [t for t in {t for m in det.values() for t in m} if t not in labeled]
            cap = cv2.VideoCapture(os.path.join(E.SEG_DIR, seg + '.mp4'))
            for tid in rng.permutation(len(tids))[:a.control]:
                tid = tids[tid]
                obs = [x for x in fs if tid in det[x]]
                if len(obs) < 30:
                    continue
                idx = np.linspace(0, len(obs) - 1, 8).astype(int)
                fr = [obs[i] for i in idx]
                vs = []
                for x in fr:
                    im = E.crop(cap, x, det[x][tid])
                    vs.append(None if im is None else enc.get(seg, tid, x, im))
                keep = [(u, v) for u, v in zip(fr, vs) if v is not None]
                if len(keep) < 5:
                    continue
                cs = [float(keep[i][1] @ keep[i + 1][1]) for i in range(len(keep) - 1)]
                ctrl.append(dict(track='%s#%d' % (seg, tid), min_cos=round(min(cs), 3),
                                 med=round(float(np.median(cs)), 3)))
            cap.release()
        json.dump(ctrl, open(os.path.join(DELIV, 'results/coarse_control.json'), 'w',
                             encoding='utf-8'), ensure_ascii=False, indent=1)
        print('\n对照：未标注轨迹 %d 条' % len(ctrl))
        for tau in (0.5, 0.55, 0.6, 0.65, 0.7):
            n = sum(1 for x in ctrl if x['min_cos'] < tau)
            print('  τ=%.2f 误触发 %d/%d（%.0f%%）'
                  % (tau, n, len(ctrl), 100 * n / max(1, len(ctrl))))

    enc.save()
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print('可测 %d 条' % len(res))
    for tau in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        sub = [x for x in res if x['min_cos'] < tau]
        hit = [x for x in sub if abs(x['err_s']) <= 1.0]
        print('  τ=%.2f 触发 %d/%d，其中落在 ±1s %d（占全部 %.0f%%）'
              % (tau, len(sub), len(res), len(hit), 100 * len(hit) / max(1, len(res))))
    print('\n逐条（按最低余弦排序）：')
    for x in sorted(res, key=lambda y: y['min_cos']):
        print('  %-22s min %.2f  次低 %-4s  误差 %+5.1fs  中位 %.2f'
              % (x['track'], x['min_cos'], x['second'], x['err_s'], x['med']))
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
