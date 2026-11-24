#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0-4 a3 回收：第二标注人交回两份 CSV 后，一条命令出 κ + 95%CI + 分歧清单。

为什么不能直接用 eval_kappa.py：那个脚本假设第二标注人交回的是「长表」
（`annotator2_TEMPLATE.csv`，含 item_id + layer）。新的盲标页导出的是两份宽表
（`a3_gallery_members_FILLED.csv` / `a3_query_gt_FILLED.csv`），且第二标注人
必然自建编号（A01、A02…），**原始一致率没有意义**，必须先做共现对齐。

对齐规则（与 docs 里既有约定一致）：
  两人都标过的同一条目构成一个观测；对每个第一标注人的编号，取共现次数最多的
  第二标注人编号作为映射。一对多/多对一单独报告——那不是「叫法不同」，是真分歧。

用法：
  PY tools/eval_kappa_a3.py                 # 用默认路径的真实数据
  PY tools/eval_kappa_a3.py --selftest      # 合成第二标注人，自检管线（含 5% 噪声）
输出：results/kappa_a3.{json,md}
"""
import argparse
import collections
import csv
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, 'deliverable/data-request-v2/gt')
RES = os.path.join(ROOT, 'deliverable/data-request-v2/results')


def rd(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def norm(p):
    p = (p or '').strip()
    return 'UNK' if (not p or p.upper() in ('UNK', '')) else p


NONPERSON_RE = re.compile(r'^(shadow\d*|no[-_ ]?human|non[-_ ]?person|not[-_ ]?person)$', re.I)


def is_nonperson(p):
    return bool(NONPERSON_RE.match((p or '').strip()))


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return dict(n=0, po=float('nan'), pe=float('nan'), k=float('nan'))
    po = sum(1 for a, b in pairs if a == b) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else float('nan')
    return dict(n=n, po=po, pe=pe, k=k)


def kappa_ci(pairs, n_boot=2000, seed=20261023):
    """bootstrap 95% CI（重采样条目）。"""
    rnd = random.Random(seed)
    n = len(pairs)
    if n < 5:
        return None, None
    ks = []
    for _ in range(n_boot):
        idx = [rnd.randrange(n) for _ in range(n)]
        sub = [pairs[i] for i in idx]
        k = kappa(sub)['k']
        if k == k:
            ks.append(k)
    ks.sort()
    return (ks[int(0.025 * len(ks))], ks[int(0.975 * len(ks))]) if ks else (None, None)


def align(pairs_raw):
    """a1 编号 -> a2 编号 的贪心多对一映射。返回 (map_2to1, 报告)。"""
    co = collections.Counter(pairs_raw)
    mapping, back = {}, collections.defaultdict(list)
    for (a1, a2), c in sorted(co.items(), key=lambda kv: -kv[1]):
        if a1 == 'UNK' or a2 == 'UNK':
            continue
        if a1 not in mapping:
            mapping[a1] = a2
            back[a2].append(a1)
    m2to1 = {}
    for a1, a2 in mapping.items():
        m2to1.setdefault(a2, a1)
    one_to_many = {a2: v for a2, v in back.items() if len(v) > 1}
    many_to_one = {a1: a2 for a1, a2 in mapping.items()
                   if sum(1 for x in mapping.values() if x == a2) > 1}
    return m2to1, dict(one_to_many=one_to_many, many_to_one=many_to_one,
                       n_mapped=len(mapping))


def collect(raw_g, raw_q, gt_g, gt_q):
    """返回 [(key, layer, a1, a2)]，只保留两人都标了的条目。"""
    rows = []
    cammap = {r['track_id']: r.get('cam', '') for r in gt_g}
    for src, gt, key in ((raw_g, gt_g, 'track_id'), (raw_q, gt_q, 'query_id')):
        gtmap = {r[key]: norm(r.get('person_id')) for r in gt}
        for r in src:
            k = r.get(key, '').strip()
            if not k or k not in gtmap:
                continue
            rows.append((k, 'gallery' if key == 'track_id' else 'query',
                         gtmap[k], norm(r.get('person_id')),
                         cammap.get(k, '')))
    return rows


def run(rows, tag):
    pairs_raw = [(r[2], r[3]) for r in rows]
    base = kappa(pairs_raw)
    m2to1, align_rep = align(pairs_raw)
    aligned = [(a, (m2to1.get(b, b) if b != 'UNK' else 'UNK')) for a, b in pairs_raw]
    kal = kappa(aligned)
    lo, hi = kappa_ci(aligned)

    # 口径二：非人标签合并（shadow1/2/3/no-human/shadow → NONPERSON）
    # 两人都认为「不是人」只是分类粒度不同，不该算身份分歧。
    col = [('NONPERSON' if is_nonperson(a) else a,
            'NONPERSON' if is_nonperson(b2) else b2) for (a, b2) in aligned]
    kcol = kappa(col)
    # 口径三：只算人物条目
    per = [(a, b2) for a, b2 in col if a != 'NONPERSON' and b2 != 'NONPERSON' and a != 'UNK' and b2 != 'UNK']
    kper = kappa(per)
    lo2, hi2 = kappa_ci(per)
    diff = [(r[0], r[1], r[2], b2, r[4]) for r, (_, b2) in zip(rows, aligned) if r[2] != b2]
    out = dict(tag=tag, n=len(rows), raw=base, aligned=kal,
               ci95=[lo, hi], align=align_rep,
               items=dict(gallery=sum(1 for x in rows if x[1] == 'gallery'),
                          query=sum(1 for x in rows if x[1] == 'query')),
               disagreements=[dict(item=k, layer=L, a1=a, a2=b, cam=cam) for k, L, a, b, cam in diff])
    print('[%s] n=%d（图库 %d / 查询 %d）'
          % (tag, out['n'], out['items']['gallery'], out['items']['query']))
    print('    原始一致率 %.3f（无对齐，仅作参考）' % base['po'])
    print('    对齐后：一致率 %.3f | κ %.3f | 95%%CI [%.3f, %.3f]'
          % (kal['po'], kal['k'], lo if lo is not None else float('nan'),
             hi if hi is not None else float('nan')))
    print('    非人标签合并后：一致率 %.3f | κ %.3f' % (kcol['po'], kcol['k']))
    print('    仅人物条目（n=%d）：一致率 %.3f | κ %.3f | 95%%CI [%.3f, %.3f]'
          % (kper['n'], kper['po'], kper['k'],
             lo2 if lo2 is not None else float('nan'),
             hi2 if hi2 is not None else float('nan')))
    print('    映射 %d 个编号；一对多 %d 个；真分歧 %d 条'
          % (align_rep['n_mapped'], len(align_rep['one_to_many']), len(diff)))
    cams = sorted({r[4] for r in rows if r[4]})
    by_cam = {}
    for cam in cams:
        idx = [i for i, r in enumerate(rows) if r[4] == cam]
        sub = [col[i] for i in idx]
        sub = [(a, b2) for a, b2 in sub
               if a not in ('NONPERSON', 'UNK') and b2 not in ('NONPERSON', 'UNK')]
        kk = kappa(sub)
        by_cam[cam] = dict(n=kk['n'], po=kk['po'], k=kk['k'])
        print('    机位 %-4s n=%-4d 一致率 %.3f κ %.3f' % (cam, kk['n'], kk['po'], kk['k']))
    out['by_cam'] = by_cam
    out['collapse_nonperson'] = kcol
    out['person_only'] = dict(kper, ci95=[lo2, hi2])
    return out


def selftest(gt_g, gt_q):
    """合成第二标注人：编号整体重命名 + 5% 随机改判 + 3% 置 UNK。"""
    rnd = random.Random(7)
    ids = sorted({norm(r['person_id']) for r in gt_g + gt_q} - {'UNK'})
    ren = {p: 'A%02d' % (i + 1) for i, p in enumerate(ids)}
    rows = []
    for src, gt, key in ((gt_g, gt_g, 'track_id'), (gt_q, gt_q, 'query_id')):
        for r, g in zip(src, gt):
            k = g[key]
            p = norm(g['person_id'])
            q = ren.get(p, 'UNK')
            if p != 'UNK' and rnd.random() < 0.05:
                q = ren[rnd.choice(ids)]
            if rnd.random() < 0.03:
                q = 'UNK'
            rows.append((k, 'gallery' if key == 'track_id' else 'query', p, q,
                         next((x['cam'] for x in gt_g if x['track_id'] == k), '')))
    # 只取 a3 的 100 条查询子集
    keep = set()
    a3q = os.path.join(GT, 'a3_query_TEMPLATE.csv')
    if os.path.exists(a3q):
        keep = {r['query_id'] for r in rd(a3q)}
    rows = [x for x in rows if x[1] == 'gallery' or x[0] in keep]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a2-gallery', default=os.path.join(GT, 'a3_gallery_members_FILLED.csv'))
    ap.add_argument('--a2-query', default=os.path.join(GT, 'a3_query_gt_FILLED.csv'))
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()

    gt_g = rd(os.path.join(GT, 'gt_gallery_members_FILLED.csv'))
    gt_q = rd(os.path.join(GT, 'gt_query_gt_FILLED.csv'))

    if a.selftest:
        rows = selftest(gt_g, gt_q)
        out = run(rows, 'selftest（合成：重命名 + 5% 改判 + 3% UNK）')
        print('    期望：κ 明显 < 1 但 > 0.85；映射应覆盖全部真实编号')
    else:
        if not os.path.exists(a.a2_gallery):
            print('缺少图库层回收文件：', a.a2_gallery)
            print('请让对方在 annotate_a3.html 里点「导出本页 CSV」。')
            return 2
        if not os.path.exists(a.a2_query):
            print('⚠ 只找到图库层文件，查询层（100 条）缺失 → 本次只算图库层 κ。')
            rows = collect(rd(a.a2_gallery), [], gt_g, gt_q)
        else:
            rows = collect(rd(a.a2_gallery), rd(a.a2_query), gt_g, gt_q)
        out = run(rows, 'a3（真人第二标注人）')

    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, 'kappa_a3.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\nwritten:', p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
