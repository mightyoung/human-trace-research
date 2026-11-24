#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0-3 真值改判的影响面测算：改一条图库成员的身份，C/W 会怎么变。

背景：`gt_gallery_members_FILLED.csv` 里有 3 条记录，最终编号与人工备注冲突
（305_train_529#10036 → 备注 female1；323_train_510#287 → 备注 female1；
318_train_525#248 → 备注 male10）。这些成员影响簇 68 / 84 的主体身份，
进而影响落在这些簇里的查询判对与否。

做法（关键：不自己发明并列打破规则）：
  - 基线的主体身份直接取 `clusters_gt.csv` 的 main_pid（当前论文数字的来源）；
  - 改判后只重算受影响簇的票数；若出现票数并列，**枚举每个并列候选**，
    报 C 的区间，而不是替评审选一个；
  - 同时报告基线自身是否存在并列（簇 84 是 4/4/4 三向平票，主体身份靠打破规则决定）。

用法：
  PY tools/eval_relabel_sensitivity.py
输出：
  deliverable/data-request-v2/results/relabel_sensitivity.json
"""
import collections
import csv
import itertools
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable', 'data-request-v2')

CHANGES = [
    ("305_train_529#10036", "female1", "备注「change to female1」，最终 male4"),
    ("323_train_510#287", "female1", "备注「change to female1」，最终 male7"),
    ("318_train_525#248", "male10", "备注「change to male10 [review]」，最终 male4"),
]


def rd(rel):
    with open(os.path.join(DELIV, rel), encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def cluster_counts(labels, member_cluster):
    cnt = collections.defaultdict(collections.Counter)
    for tid, cl in member_cluster.items():
        pid = labels.get(tid, '')
        if pid:
            cnt[cl][pid] += 1
    return cnt


def score(main, queries):
    C = W = U = 0
    flips = []
    for q in queries:
        if q['status'] != 'accepted':
            continue
        pid = q['truth']
        pred = main.get(q['cluster'], '')
        if not pid:
            U += 1
        elif pred == pid:
            C += 1
        else:
            W += 1
            flips.append(dict(query_id=q['query_id'], truth=pid, pred=pred,
                              cluster=q['cluster']))
    return C, W, U, flips


def candidates_for(cnt):
    """每个簇的并列候选（票数最高的那些）。"""
    out = {}
    for cl, c in cnt.items():
        if not c:
            continue
        top = max(c.values())
        out[cl] = sorted([p for p, v in c.items() if v == top])
    return out


def main():
    member_cluster = {r['track_id']: r['cluster_local']
                      for r in rd('results/members/A_baseline.csv')}
    labels = {}
    for r in rd('gt/gt_gallery_members_FILLED.csv'):
        tid = (r['track_id'] or '').strip()
        if tid:
            labels[tid] = (r['person_id'] or '').strip()

    gt_main = {r['cluster_id']: r['main_pid'] for r in rd('gt/clusters_gt.csv') if r['main_pid']}

    gtq = {r['query_id']: r for r in rd('gt/gt_query_gt_FILLED.csv')}
    gal_ids = set(p for p in labels.values() if p)
    queries = []
    for r in rd('results/queries/A_baseline.csv'):
        g = gtq[r['query_id']]
        pid = g['person_id'].strip()
        if g['evaluable'] != '1' or not pid or pid not in gal_ids:
            continue
        queries.append(dict(query_id=r['query_id'], truth=pid,
                            status=r['final_status'],
                            cluster=r['final_cluster_local']))

    C0, W0, U0, flips0 = score(gt_main, queries)
    print('[base ] clusters_gt main_pid → 接受 %d | C %d | W %d | U %d'
          % (sum(1 for q in queries if q['status'] == 'accepted'), C0, W0, U0))

    cnt0 = cluster_counts(labels, member_cluster)
    cand0 = candidates_for(cnt0)
    ties0 = {cl: [c for c in v] for cl, v in cand0.items() if len(v) > 1}
    print('[base ] 票数并列的簇：%s' % (ties0 if ties0 else '无'))
    for cl in ties0:
        print('        簇 %-3s 票数 %s' % (cl, dict(cnt0[cl])))

    variants = []
    for combo in ([[c] for c in CHANGES] + [CHANGES]):
        lab = dict(labels)
        for tid, pid, _ in combo:
            lab[tid] = pid
        cnt = cluster_counts(lab, member_cluster)
        cand = candidates_for(cnt)
        affect = sorted({member_cluster[t] for t, _, _ in combo if t in member_cluster})
        # 只枚举受影响簇的并列候选，其余沿用 clusters_gt
        opts = []
        for cl in affect:
            opts.append([(cl, p) for p in cand[cl]])
        rows = []
        for pick in itertools.product(*opts):
            m = dict(gt_main)
            for cl, p in pick:
                m[cl] = p
            C, W, U, fl = score(m, queries)
            rows.append(dict(assign={str(cl): p for cl, p in pick}, C=C, W=W, U=U,
                             dC=C - C0, dW=W - W0, flips=fl))
        Cs = [r['C'] for r in rows]
        name = ' + '.join(c[0] for c in combo)
        variants.append(dict(changes=[[c[0], c[1]] for c in combo], name=name,
                             affected_clusters=affect,
                             votes={str(cl): dict(cnt[cl]) for cl in affect},
                             C_min=min(Cs), C_max=max(Cs), W_min=min(r['W'] for r in rows),
                             W_max=max(r['W'] for r in rows), rows=rows))
        print('\n[%-46s] 影响簇 %s' % (name, affect))
        for cl in affect:
            print('        簇 %-3s 票数 %s' % (cl, dict(cnt[cl])))
        print('        C 区间 [%d, %d]（基线 %d）| W 区间 [%d, %d]'
              % (min(Cs), max(Cs), C0, min(r['W'] for r in rows), max(r['W'] for r in rows)))
        for r in rows:
            print('        指定 %s → C %d (%+d) W %d (%+d)'
                  % (r['assign'], r['C'], r['dC'], r['W'], r['dW']))
            for f in r['flips']:
                print('            错判 %s（真值 %s，判成 %s，簇 %s）'
                      % (f['query_id'], f['truth'], f['pred'], f['cluster']))

    out = dict(base=dict(C=C0, W=W0, U=U0, base_ties={str(k): v for k, v in ties0.items()},
                         base_tie_votes={str(k): dict(cnt0[k]) for k in ties0}),
               note_3=[[c[0], c[1], c[2]] for c in CHANGES],
               variants=variants)
    p = os.path.join(DELIV, 'results/relabel_sensitivity.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\nwritten:', p)


if __name__ == '__main__':
    main()
