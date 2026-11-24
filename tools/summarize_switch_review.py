#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-A 结果汇总：把人工复核的换人时刻按「能不能当切点用」分类。

输入：
  - 复核页导出的 `a3_switch_review_FILLED.csv`（46 条，含 note='无法判定'）
  - `ledger/tracks.csv`（轨迹起止帧）
输出：
  - `results/m1a_switch_review.csv`（逐条：位置、距两端秒数、分类）
  - 终端摘要（供 docs/m1-a-result-2026-10-23.md 引用）

分类口径：
  usable   切点距两端都 ≥2s —— 切开后两段都能独立成轨迹（M1 分母）
  tail     填了，但距轨迹终点 <2s（含切点落在轨迹之后）—— 切开只剩空片段
  head     填了，但距起点 <2s —— 前半段太短
  unknown  标注人按「无法判定」

用法：PY tools/summarize_switch_review.py --csv <导出的 CSV>
"""
import argparse
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
LEDGER = os.path.join(DELIV, 'ledger/tracks.csv')
GAL = os.path.join(DELIV, 'gt/gt_gallery_members_FILLED.csv')
OUT = os.path.join(DELIV, 'results/m1a_switch_review.csv')
FPS = 25.0
EDGE_S = 2.0
TAIL_POS = 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    a = ap.parse_args()
    led = {r['track_id']: r for r in csv.DictReader(open(LEDGER, encoding='utf-8-sig'))}
    gal = [r['track_id'] for r in csv.DictReader(open(GAL, encoding='utf-8-sig'))
           if r['id_switch'].strip() in ('1', '1.0')]
    rows = {r['track_id']: r for r in csv.DictReader(open(a.csv, encoding='utf-8-sig'))}
    edge = int(EDGE_S * FPS)

    out = []
    for tid in gal:
        t = led[tid]
        f0, f1 = int(t['t_start_frame']), int(t['t_end_frame'])
        r = rows.get(tid, {})
        v = (r.get('switch_frame') or '').strip()
        new = int(float(v)) if v else None
        old_s = (r.get('old_switch_frame') or '').strip()
        old = int(float(old_s)) if old_s else None
        note = (r.get('note') or '').strip()
        if new is None:
            cls = 'unknown'
        elif new > f1 - edge:
            cls = 'tail'
        elif new < f0 + edge:
            cls = 'head'
        else:
            cls = 'usable'
        pos = (new - f0) / float(max(1, f1 - f0)) if new is not None else None
        out.append(dict(track_id=tid, seg=t['seg'], cam=t['cam'], n_frames=t['n_frames'],
                        t_start=f0, t_end=f1, old_frame=old if old is not None else '',
                        new_frame=new if new is not None else '',
                        dist_head_s='' if new is None else round((new - f0) / FPS, 2),
                        dist_tail_s='' if new is None else round((f1 - new) / FPS, 2),
                        pos_ratio='' if pos is None else round(pos, 3),
                        cls=cls, note=note))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    c = collections.Counter(x['cls'] for x in out)
    print('46 条串人成员：usable %d ｜ tail %d ｜ head %d ｜ unknown %d'
          % (c['usable'], c['tail'], c['head'], c['unknown']))
    print('  可用切点按机位：%s' % dict(collections.Counter(x['cam'] for x in out if x['cls'] == 'usable')))
    u = [x for x in out if x['cls'] == 'usable']
    n_mid = sum(1 for x in u if x['pos_ratio'] <= TAIL_POS)
    print('  其中「中途换人」（位置 ≤%.1f）：%d 条；靠尾部但 ≥2s：%d 条'
          % (TAIL_POS, n_mid, len(u) - n_mid))
    t = [x for x in out if x['cls'] == 'tail']
    print('  尾部换人 %d 条：距终点中位 %.2fs，其中切点落在轨迹之后 %d 条'
          % (len(t), sorted(float(x['dist_tail_s']) for x in t)[len(t) // 2],
             sum(1 for x in t if float(x['dist_tail_s']) < 0)))
    print('  → %s' % OUT)


if __name__ == '__main__':
    main()
