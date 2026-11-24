#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0-2 判据敏感性：换一个「选点规则」，42 个配置的结论会不会翻。

背景：6 机位实验发现，self-cal（same≥0.90 下取覆盖最大）会把「补库带来的覆盖增益」
判成负收益——因为 K 变大让 same 掉到 0.90 以下，选点被迫抬 θ。于是问题变成：
**这是 6 机位的特例，还是判据本身在惩罚所有「候选变多」的配置？**

本脚本在 1012 的 42 个配置上、用同一批曲线数据（results/curve.csv），比较三种选点规则：
  C1（现用） feasible = same≥0.90 且 evaluable≥100；目标 = 覆盖最大
  C2（换目标） 同一可行域；目标 = 最终产出率（覆盖×same）最大
  C3（放宽）   feasible = fm≤2% 且 evaluable≥100；目标 = 最终产出率最大
报告：每个配置选中的 θ、以及三个规则下的覆盖/稳定率/fm。

注意：C1/C2 的可行域一致，差别只在目标函数；C3 放开了 same 门槛，
所以它一定会选更低的 θ、更高的覆盖和更差的 same——这条用来量化「放开判据要付多少 fm」。

用法：PY tools/criterion_sensitivity.py
输出：deliverable/data-request-v2/results/criterion_sensitivity.{json,md}
"""
import collections
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
CURVE = os.path.join(DELIV, 'results/curve.csv')
SAME_MIN, MIN_EVAL, FM_BUDGET = 0.90, 100, 0.02


def load():
    by = collections.defaultdict(list)
    with open(CURVE, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if not r['same'] or r['same'] == 'nan':
                continue
            by[r['config']].append(dict(
                config=r['config'], group=r['group'], K=int(r['K']), theta=float(r['theta']),
                coverage=float(r['coverage']), same=float(r['same']),
                fm=float(r['fm']), evaluable=int(r['evaluable']),
                out=float(r['final_output_rate'])))
    for k in by:
        by[k].sort(key=lambda x: x['theta'])
    return by


def pick(rows, criterion):
    if criterion == 'C1':
        feas = [r for r in rows if r['same'] >= SAME_MIN and r['evaluable'] >= MIN_EVAL]
        return max(feas, key=lambda r: r['coverage']) if feas else None
    if criterion == 'C2':
        feas = [r for r in rows if r['same'] >= SAME_MIN and r['evaluable'] >= MIN_EVAL]
        return max(feas, key=lambda r: r['out']) if feas else None
    if criterion == 'C3':
        feas = [r for r in rows if r['fm'] <= FM_BUDGET and r['evaluable'] >= MIN_EVAL]
        return max(feas, key=lambda r: r['out']) if feas else None
    raise SystemExit(criterion)


def main():
    by = load()
    picks = {}
    rows = []
    for cfg, rs in by.items():
        p1, p2, p3 = pick(rs, 'C1'), pick(rs, 'C2'), pick(rs, 'C3')
        picks[cfg] = {c: (p['theta'] if p else None) for c, p in
                      (('C1', p1), ('C2', p2), ('C3', p3))}
        rows.append(dict(config=cfg, group=rs[0]['group'], K=rs[0]['K'],
                         C1=p1, C2=p2, C3=p3))

    def brief(p):
        return None if not p else dict(theta=p['theta'], cov=round(p['coverage'], 4),
                                       same=round(p['same'], 4), fm=round(p['fm'], 5),
                                       out=round(p['out'], 4))

    changed_12 = [r['config'] for r in rows if r['C1'] and r['C2']
                  and abs(r['C1']['theta'] - r['C2']['theta']) > 1e-9]
    print('配置数 %d' % len(rows))
    print('C1 vs C2（只换目标函数，可行域不变）：选点不同的配置 %d 个' % len(changed_12))
    for c in changed_12[:12]:
        r = next(x for x in rows if x['config'] == c)
        print('  %-22s θ %.2f→%.2f | 覆盖 %.3f→%.3f | same %.3f→%.3f | 产出率 %.3f→%.3f'
              % (c, r['C1']['theta'], r['C2']['theta'], r['C1']['coverage'], r['C2']['coverage'],
                 r['C1']['same'], r['C2']['same'], r['C1']['out'], r['C2']['out']))

    print('\nC1 vs C3（放开 same 门槛）：')
    for r in rows[:6]:
        if not (r['C1'] and r['C3']):
            continue
        print('  %-22s θ %.2f→%.2f | 覆盖 %.3f→%.3f | same %.3f→%.3f | fm %.2f%%→%.2f%%'
              % (r['config'], r['C1']['theta'], r['C3']['theta'],
                 r['C1']['coverage'], r['C3']['coverage'],
                 r['C1']['same'], r['C3']['same'],
                 100 * r['C1']['fm'], 100 * r['C3']['fm']))
    worse_fm = [r['config'] for r in rows if r['C1'] and r['C3'] and r['C3']['fm'] > r['C1']['fm']]
    print('\nC3 相对 C1 的 fm 变化：变差 %d / %d 个配置' % (len(worse_fm), len(rows)))
    d = [r['C3']['fm'] - r['C1']['fm'] for r in rows if r['C1'] and r['C3']]
    if d:
        d.sort()
        print('  Δfm 中位 %+.4f（%.2f%%→中位 %.2f%%）' % (
            d[len(d) // 2], 100 * sum(r['C1']['fm'] for r in rows if r['C1']) / len(d),
            100 * sorted(r['C3']['fm'] for r in rows if r['C3'])[len(d) // 2]))

    out = dict(criteria={
        'C1': 'feasible same>=0.90 & evaluable>=100; max coverage（现用）',
        'C2': 'feasible 同上; max coverage*same（产出率）',
        'C3': 'feasible fm<=2% & evaluable>=100; max coverage*same（放开 same）'},
        n_configs=len(rows),
        c1_c2_changed=len(changed_12),
        rows=[dict(config=r['config'], group=r['group'], K=r['K'],
                   C1=brief(r['C1']), C2=brief(r['C2']), C3=brief(r['C3']))
              for r in rows])
    p = os.path.join(DELIV, 'results/criterion_sensitivity.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\nwritten:', p)


if __name__ == '__main__':
    main()
