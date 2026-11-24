#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 验收闸门 v1（A+C 口径，2026-10-23 锁定）。

## 判定规则（每条轨迹最多一个动作 → 碎片有硬上限）
  外观最低跳 < τ 且位置在尾部（pos ≥ 0.90） → **截断**：丢掉换人点之后的样本（不产生碎片）
  外观最低跳 < τ 且位置在中段              → **拆分**：在该点切成两条子轨迹
  其余                                      → **保留**（不动）

## 四项指标
  M1 中段召回   pos<0.9 的真混人轨里，动作切点落在真值 ±1s 的比例
  M2 尾部污染   pos≥0.9 的真混人轨里，「换人之后的采样点数 / 总采样点数」的中位数
                （这个数很小说明尾部换人本来就不污染表征，不需要额外动作）
  M3 群体碎片   全部轨迹的平均子轨迹数（按「干净轨触发率 + 混人比例」折算）
  M4 处理率     所有已标注混人轨里，被正确动作覆盖的比例（截断/拆分都算）

## 闸门类型：**不退化闸门**，不是增益目标
首次运行把实测值写成基线（`results/m1_gate_baseline.json`）；
之后每次运行与基线比，任何一项明显变差就 FAIL。
`≥80% 拆分` 属于 M1.5 的增益目标，前置条件是**逐帧检测表**（见 docs/m1-gate-2026-10-23.md）。

用法：
  PY tools/m1_gate.py                 # 首次=写基线；之后=对比基线
  PY tools/m1_gate.py --emit-actions  # 另出动作清单
  PY tools/m1_gate.py --reset-baseline
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DELIV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'deliverable/data-request-v2')
FPS = 25.0
TAU = 0.60
TAIL_POS = 0.90
TOL_F = 25            # 命中容差 ±1s
SWITCH_PREV = 0.066   # 图库真混人比例 17/257
EPS = 1e-9


def rd(p):
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))


def load_scan(path):
    d = json.load(open(path, encoding='utf-8'))
    return {r['track']: r for r in d.get('marks', [])}, d.get('control', [])


def decide(row, tau, tail_pos):
    best = row.get('best')
    if not best or best[1] >= tau:
        return 'keep', None, None, None
    f, c = best
    fs0, fs1 = row.get('f0'), row.get('f1')
    pos = (f - fs0) / float(fs1 - fs0) if (fs0 is not None and fs1 and fs1 > fs0) else None
    act = 'truncate' if (pos is not None and pos >= tail_pos) else 'split'
    return act, f, pos, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', default=os.path.join(DELIV, 'results/scan_appearance.json'))
    ap.add_argument('--control', default=os.path.join(DELIV, 'results/scan_appearance_v1_usable21ctl.json'))
    ap.add_argument('--marks', default=os.path.join(DELIV, 'results/m1a_switch_review.csv'))
    ap.add_argument('--tau', type=float, default=TAU)
    ap.add_argument('--baseline', default=os.path.join(DELIV, 'results/m1_gate_baseline.json'))
    ap.add_argument('--reset-baseline', action='store_true')
    ap.add_argument('--emit-actions', action='store_true')
    ap.add_argument('--out', default=os.path.join(DELIV, 'results/m1_gate.json'))
    a = ap.parse_args()

    gt = {r['track_id']: r for r in rd(a.marks)}
    marks, _ = load_scan(a.scan)
    _, ctrl = load_scan(a.control)

    import tracklet_split_v2 as S
    dets, spans = {}, {}
    for k, r in marks.items():
        seg, tid = r['seg'], r['tid']
        if seg not in dets:
            dets[seg] = S.load_det(seg)
        fs = sorted(f for f, m in dets[seg].items() if tid in m)
        if fs:
            r['f0'], r['f1'] = fs[0], fs[-1]
            spans[k] = (fs, len(fs))

    rows = []
    for k, r in marks.items():
        g = gt.get(k, {})
        act, f, pos, dip = decide(r, a.tau, TAIL_POS)
        true = r.get('true')
        err = round((f - true) / FPS, 2) if (f is not None and true) else None
        hit = 1 if (f is not None and true and abs(f - true) <= TOL_F) else 0
        # 尾部污染：真值之后还剩多少采样点
        contam = None
        if true and k in spans and r['f1'] > r['f0']:
            contam = round(max(0.0, (r['f1'] - true) / float(r['f1'] - r['f0'])), 4)
        rows.append(dict(track=k, cls=g.get('cls', ''), true=true, action=act, cut=f,
                         pos=None if pos is None else round(pos, 3), dip=dip,
                         err_s=err, hit=hit, contam=contam,
                         n_pts=r.get('n'), dur_s=r.get('dur_s')))

    by = collections.defaultdict(lambda: dict(n=0, hit=0, act=collections.Counter()))
    for r in rows:
        b = by[r['cls']]
        b['n'] += 1
        b['hit'] += r['hit']
        b['act'][r['action']] += 1

    mid = [r for r in rows if r['cls'] == 'usable' and (r['pos'] is None or r['pos'] < TAIL_POS)]
    mid_by_true = [r for r in rows if r['cls'] == 'usable' and r['true']
                   and (r['true'] - marks[r['track']]['f0']) / float(
                       marks[r['track']]['f1'] - marks[r['track']]['f0']) < TAIL_POS]
    tail = [r for r in rows if r['cls'] == 'tail']

    m1 = sum(r['hit'] for r in mid_by_true) / max(1, len(mid_by_true))
    cont = [r['contam'] for r in tail if r['contam'] is not None]
    m2 = sorted(cont)[len(cont) // 2] if cont else None
    trig_ctrl = sum(1 for x in ctrl if x.get('best') and x['best'][1] < a.tau) / max(1, len(ctrl))
    trig_mix = sum(1 for r in rows if r['action'] != 'keep') / max(1, len(rows))
    m3 = 1 + (1 - SWITCH_PREV) * trig_ctrl + SWITCH_PREV * trig_mix
    all_mixed = [r for r in rows if r['true']]
    m4 = sum(r['hit'] for r in all_mixed) / max(1, len(all_mixed))

    cur = dict(m1_mid_recall=round(m1, 3), m2_tail_contam=m2, m3_fragments=round(m3, 3),
               m4_coverage=round(m4, 3),
               detail=dict(n_mid=len(mid_by_true), hit_mid=sum(r['hit'] for r in mid_by_true),
                           n_tail=len(tail), n_all_mixed=len(all_mixed),
                           trig_ctrl=round(trig_ctrl, 3), trig_mix=round(trig_mix, 3),
                           n_ctrl=len(ctrl), tau=a.tau))

    print('M1 闸门 v1（τ=%.2f，尾部 pos≥%.2f，命中 ±1s）' % (a.tau, TAIL_POS))
    print('-' * 78)
    print('%-30s %-10s %-12s %s' % ('类别', '条数', '命中', '动作分布'))
    for cls in ('usable', 'tail', 'head'):
        b = by.get(cls)
        if b:
            dist = ', '.join('%s=%d' % (k, v) for k, v in sorted(b['act'].items()))
            print('%-30s %-10d %-12s %s' % (cls, b['n'],
                                            '%d (%.0f%%)' % (b['hit'], 100 * b['hit'] / b['n']), dist))
    print('-' * 78)
    print('M1 中段召回（pos<0.9，n=%d）      %5.1f%%   %d/%d'
          % (len(mid_by_true), 100 * m1, sum(r['hit'] for r in mid_by_true), len(mid_by_true)))
    print('M2 尾部污染（换人之后采样点占比） %5s   n=%d'
          % ('%.1f%%' % (100 * m2) if m2 is not None else 'NA', len(cont)))
    print('M3 群体碎片                      %5.2f×  （干净轨触发 %.0f%%，混人轨触发 %.0f%%）'
          % (m3, 100 * trig_ctrl, 100 * trig_mix))
    print('M4 总处理率（全部真混人轨）      %5.1f%%   %d/%d'
          % (100 * m4, sum(r['hit'] for r in all_mixed), len(all_mixed)))

    # —— 与基线对比 ——
    if a.reset_baseline and os.path.exists(a.baseline):
        os.remove(a.baseline)
    verdict = 'BASELINE'
    if not os.path.exists(a.baseline):
        json.dump(cur, open(a.baseline, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('\n[基线] 已写入 %s' % a.baseline)
    else:
        base = json.load(open(a.baseline, encoding='utf-8'))
        worse = []
        if cur['m1_mid_recall'] < base['m1_mid_recall'] - EPS:
            worse.append('M1 中段召回 %.3f < %.3f' % (cur['m1_mid_recall'], base['m1_mid_recall']))
        if base['m2_tail_contam'] is not None and cur['m2_tail_contam'] is not None \
                and cur['m2_tail_contam'] > base['m2_tail_contam'] + 1e-6:
            worse.append('M2 尾部污染 %.4f > %.4f' % (cur['m2_tail_contam'], base['m2_tail_contam']))
        if cur['m3_fragments'] > base['m3_fragments'] + 1e-6:
            worse.append('M3 碎片 %.3f > %.3f' % (cur['m3_fragments'], base['m3_fragments']))
        if cur['m4_coverage'] < base['m4_coverage'] - EPS:
            worse.append('M4 处理率 %.3f < %.3f' % (cur['m4_coverage'], base['m4_coverage']))
        verdict = 'PASS' if not worse else 'FAIL'
        print('\n[对比基线] %s' % verdict)
        for w in worse:
            print('  退化：%s' % w)
        if not worse:
            print('  四项指标均不低于基线')
    print('-' * 78)
    print('总判定：%s' % verdict)

    print('\n逐条：')
    order = {'usable': 0, 'tail': 1, 'head': 2}
    for r in sorted(rows, key=lambda x: (order.get(x['cls'], 9), abs(x['err_s'] or 999))):
        print('  %-22s %-7s %-9s cut=%-7s err=%-7s hit=%d pos=%s contam=%s'
              % (r['track'], r['cls'], r['action'],
                 r['cut'] if r['cut'] is not None else '-',
                 ('%+.1fs' % r['err_s']) if r['err_s'] is not None else 'NA',
                 r['hit'], r['pos'],
                 ('%.1f%%' % (100 * r['contam'])) if r['contam'] is not None else 'NA'))

    json.dump(dict(current=cur, tau=a.tau, rows=rows), open(a.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n→ %s' % a.out)

    if a.emit_actions:
        p = os.path.join(DELIV, 'results/m1_actions.csv')
        with open(p, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['track_id', 'action', 'cut_frame', 'true_frame',
                                              'cls', 'err_s', 'hit', 'note'])
            w.writeheader()
            for r in rows:
                w.writerow(dict(track_id=r['track'], action=r['action'],
                                cut_frame=r['cut'] if r['cut'] is not None else '',
                                true_frame=r['true'] or '', cls=r['cls'],
                                err_s=r['err_s'] if r['err_s'] is not None else '',
                                hit=r['hit'],
                                note='' if r['action'] == 'keep' else
                                     ('丢弃 cut 之后' if r['action'] == 'truncate' else '切成两段')))
        print('→ %s（%d 条动作）' % (p, sum(1 for r in rows if r['action'] != 'keep')))


if __name__ == '__main__':
    main()
