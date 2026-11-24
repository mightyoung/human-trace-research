#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第一标注人的重测信度（两遍自比）：D5 复标层 vs D3/D4 首轮。

为什么需要：`docs/gt-v3-a1-review-2026-10-26.md` §5 报了一个「剔除 UNK + 作废
female4」口径的 0.944，但复标层里出现非法编号 `male625`（3 条），D3/D4 中都不存在，
D4 对这三条标的是 `male5`。标注人已确认 `male625` 就是 `male5`（复标时没看 D3/D4
的标签，编号凭记忆写错）。本脚本把三种处理都算一遍，主口径取 `--m625 male5`。

口径：
  all      全部 243 条
  drop_unk 剔除任一侧 UNK / 空 / unresolved far-field
  paper    再剔除已作废的 female4（稿件口径）

用法：
  PY tools/eval_retest_kappa.py
  PY tools/eval_retest_kappa.py --m625 asis --out <path>
输出：deliverable/data-request-v2/results/retest_kappa_a1.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_kappa_a3 import GT, RES, is_nonperson, kappa, kappa_ci, rd  # noqa: E402

LAYERS = ('gallery', 'query')
DEPRECATED = {'female4'}          # gt v2 起作废的身份（4 条图库），不进稿件口径
UNK_TOKENS = {'', 'unk', 'unknown', 'none', 'na', '-'}


def empty(v):
    return (v or '').strip().lower() in UNK_TOKENS


def load_first_pass():
    """首轮标注：D3 图库 + D4 查询。返回 {item_id: person_id}。"""
    first = {}
    for name, key in (('gt_gallery_members_FILLED.csv', 'track_id'),
                      ('gt_query_gt_FILLED.csv', 'query_id')):
        for r in rd(os.path.join(GT, name)):
            first[r[key]] = (r.get('person_id') or '').strip()
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m625', default='male5', choices=['asis', 'male5', 'male25'],
                    help='复标层 male625 的处理；asis=当新身份，male5=与 D4 一致（现行），male25=打错字')
    ap.add_argument('--out', default=os.path.join(RES, 'retest_kappa_a1.json'))
    args = ap.parse_args()

    first = load_first_pass()
    raw = rd(os.path.join(GT, 'gt_annotator2_FILLED.csv'))

    rows, missing = [], 0
    for r in raw:
        iid = r['item_id'].strip()
        second = (r.get('person_id') or '').strip()
        if iid not in first:
            missing += 1
            continue
        if second == 'male625' and args.m625 != 'asis':
            second = args.m625
        rows.append(dict(layer=r['layer'].strip(), item_id=iid, cam=(r.get('cam') or '').strip(),
                         a=first[iid], b=second,
                         reason=(r.get('unk_reason') or '').strip()))

    def score(sub, tag):
        pairs = [(x['a'], x['b']) for x in sub]
        k = kappa(pairs)
        lo, hi = kappa_ci(pairs)
        return dict(tag=tag, n=k['n'], po=round(k['po'], 4), k=round(k['k'], 4),
                    ci=[round(lo, 4), round(hi, 4)],
                    disagree=sum(1 for a, b in pairs if a != b))

    def is_unk(x):
        return empty(x['a']) or empty(x['b']) or x['b'].lower().startswith('unk') \
            or x['reason'].lower().startswith('far')

    out = {'m625': args.m625, 'missing_item': missing, 'n_rows': len(rows), 'by_scope': {},
           'by_layer': {}}

    for scope, flt in (('all', lambda x: True),
                       ('drop_unk', lambda x: not is_unk(x)),
                       ('paper', lambda x: (not is_unk(x)) and x['a'] not in DEPRECATED
                        and x['b'] not in DEPRECATED)):
        sub = [x for x in rows if flt(x)]
        out['by_scope'][scope] = score(sub, scope)
        out['by_layer'][scope] = {ly: score([x for x in sub if x['layer'] == ly], ly)
                                  for ly in LAYERS}

    # 分歧清单（论文口径），供人工核对
    paper = [x for x in rows if (not is_unk(x)) and x['a'] not in DEPRECATED
             and x['b'] not in DEPRECATED]
    out['paper_disagreements'] = [dict(layer=x['layer'], item_id=x['item_id'], cam=x['cam'],
                                       first=x['a'], second=x['b'])
                                  for x in paper if x['a'] != x['b']]
    out['paper_identity_counts'] = {
        'first': dict(collections.Counter(x['a'] for x in paper).most_common()),
        'second': dict(collections.Counter(x['b'] for x in paper).most_common()),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('m625 =', args.m625, '| rows', len(rows), '| missing', missing)
    for scope in ('all', 'drop_unk', 'paper'):
        s = out['by_scope'][scope]
        print('  %-9s n=%-4d k=%.4f CI[%.3f, %.3f]  disagree=%d'
              % (scope, s['n'], s['k'], s['ci'][0], s['ci'][1], s['disagree']))
        for ly in LAYERS:
            t = out['by_layer'][scope][ly]
            print('      %-7s n=%-4d k=%.4f  disagree=%d'
                  % (ly, t['n'], t['k'], t['disagree']))
    print('  paper disagreements:', len(out['paper_disagreements']))
    for d in out['paper_disagreements']:
        print('    %-5s %-24s %-8s %s -> %s' % (d['cam'], d['item_id'], d['layer'],
                                                d['first'], d['second']))
    print('->', args.out)


if __name__ == '__main__':
    main()
