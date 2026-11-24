#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0-2 判据敏感性的标签侧：C(θ) 曲线，用来回答「换选点规则会不会多答对」。

现有 self-cal 的判据是「same≥0.90 且 evaluable≥100 下取覆盖最大」。
6 机位实验显示这道门槛会惩罚「候选变多」的配置。问题是：
**在 1012 上放开这道门槛（改成 fm≤2% 内取覆盖最大），C/229 会升还是降？**

本脚本固定配置（生产图库、无 veto、K=111，即 A_baseline），只扫 θ，
对每个 θ 算接受集合与逐条对错；覆盖/same/fm 直接取 results/curve.csv（官方口径），
避免自己重实现 false-merge。

用法：PY tools/eval_theta_curve_precision.py
输出：deliverable/data-request-v2/results/theta_precision_curve.{csv,json}
"""
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S                 # noqa: E402
import diag_320_gallery as D                    # noqa: E402
from case_control_modal import load_gt           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
FM_BUDGET = 0.02


def main():
    ap_emb = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd_emb')
    gal, qry = load_gt(os.path.join(DELIV, 'gt'))
    gallery_ids = set(gal.values())
    res = D.build_variant(ap_emb, list(S.TRAIN_SEGS), veto=None)

    cid_pid = {}
    for pi, c in enumerate(res['kept']):
        votes = collections.Counter(gal[k] for k in
                                    [res['ktr'][i] for i in np.where(res['lab'] == c)[0]] if k in gal)
        if votes:
            cid_pid[pi] = votes.most_common(1)[0][0]

    # 官方曲线的覆盖/same/fm（同 config 名 A_baseline）
    curve = {}
    with open(os.path.join(DELIV, 'results/curve.csv'), encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['config'] == 'A_baseline':
                curve[round(float(r['theta']), 4)] = r

    rows = []
    for thr in np.round(np.arange(1.30, 3.01, 0.05), 4):
        acc = 0
        C = W = U = 0
        for i, k in enumerate(res['kte']):
            pid, ev = qry.get(k, ('', False))
            ia, ib = int(np.argmax(res['Sa'][i])), int(np.argmax(res['Sb'][i]))
            sa, sb = float(res['Sa'][i][ia]), float(res['Sb'][i][ib])
            if not (sa >= thr and sb >= thr and ia == ib):
                continue
            acc += 1
            if not ev or not pid:
                U += 1
                continue
            pred = cid_pid.get(ia, '')
            if pred and pred == pid:
                C += 1
            elif pid not in gallery_ids:
                W += 1          # 图库外新人被接受 = 结构性错误
            else:
                W += 1
        c = curve.get(float(thr))
        rows.append(dict(theta=float(thr), accepted=acc, C=C, W=W, U=U,
                         precision=(C / (C + W)) if C + W else None,
                         C_over_229=C / 229,
                         coverage=float(c['coverage']) if c else None,
                         same=float(c['same']) if c else None,
                         fm=float(c['fm']) if c else None))

    def show(tag, r):
        print('%-28s θ=%.2f | 接受 %3d | C %3d W %d U %2d | 精度 %s | C/229 %.3f | '
              'cover %.3f | same %s | fm %s'
              % (tag, r['theta'], r['accepted'], r['C'], r['W'], r['U'],
                 ('%.4f' % r['precision']) if r['precision'] else '  -  ',
                 r['C_over_229'], r['coverage'] if r['coverage'] else 0,
                 ('%.3f' % r['same']) if r['same'] else ' - ',
                 ('%.2f%%' % (100 * r['fm'])) if r['fm'] is not None else ' - '))

    print('扫 %d 个 θ（%.2f–%.2f）' % (len(rows), rows[0]['theta'], rows[-1]['theta']))
    cur = min(rows, key=lambda r: abs(r['theta'] - 1.85))
    show('现用选点（C1, self-cal）', cur)

    # 判据 C3：fm 预算内 C 最大
    feas = [r for r in rows if r['fm'] is not None and r['fm'] <= FM_BUDGET]
    best_c = max(feas, key=lambda r: (r['C'], -r['theta'])) if feas else None
    if best_c:
        show('C3：fm≤2% 内 C 最大', best_c)
    # 判据 C4：fm 预算内 C/229 最大且精度≥0.95
    feas2 = [r for r in feas if r['precision'] and r['precision'] >= 0.95]
    best_p = max(feas2, key=lambda r: (r['C'], -r['theta'])) if feas2 else None
    if best_p:
        show('C4：fm≤2% 且精度≥0.95 内 C 最大', best_p)

    out = dict(config='A_baseline', fm_budget=FM_BUDGET,
               current=cur, best_C=best_c, best_C_prec95=best_p, curve=rows)
    p = os.path.join(DELIV, 'results/theta_precision_curve.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    with open(os.path.join(DELIV, 'results/theta_precision_curve.csv'), 'w',
              encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('\nwritten:', p)


if __name__ == '__main__':
    main()
