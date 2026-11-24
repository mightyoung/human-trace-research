#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""损失分解（loss decomposition）—— 295 条查询 -> 118 条答对 的逐级账本

口径（与 results/D4-analysis.md v2 一致）：
  correct = (final_cluster 主体身份 == query 真值)，主体身份=簇内已标成员投票多数派
  分母 229 = evaluable=1 的查询；196 = 229 - 22 无真值 - 11 图库外新人
输出：
  results/loss_decomposition.csv   逐级瀑布（总体 + 分机位）
  results/loss_decomposition.json  同内容结构化版（供论文表格/脚本复用）
用法：
  python tools/analyze_loss_decomp.py --pack deliverable/data-request-v2
"""
import argparse
import csv
import json
import os
import collections


def _is_nonperson(pid):
    """no-human / shadow* = 影子·镜像·物件；既不判对错，也不进可答。"""
    p = (pid or '').strip().lower()
    return p == 'no-human' or p.startswith('shadow')


def load(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='deliverable/data-request-v2')
    ap.add_argument('--config', default='A_baseline')
    args = ap.parse_args()
    pack = args.pack

    gt = load(os.path.join(pack, 'gt/gt_query_gt_FILLED.csv'))
    gal = load(os.path.join(pack, 'gt/gt_gallery_members_FILLED.csv'))
    clu = load(os.path.join(pack, 'gt/clusters_gt.csv'))
    qres = load(os.path.join(pack, 'results/queries/%s.csv' % args.config))

    gallery_ids = set(r['person_id'] for r in gal if r['person_id'])
    # 簇主体身份（配置无关）：簇内已标成员投票多数派
    main_pid = {r['cluster_id']: r['main_pid'] for r in clu}

    gtmap = {r['query_id']: r for r in gt}
    qmap = {r['query_id']: r for r in qres}

    cams = ['305', '318', '320', '323']
    rows = []

    def blank():
        return collections.Counter()

    def bucket():
        return {
            'queries': 0,            # 该机位检出并进入标注的查询
            'dropped': 0,            # 轨迹过短/过小，未进 229
            'evaluable': 0,
            'no_truth': 0,           # 可评估但无真值（far）
            'nonperson': 0,          # 可评估且已标非人（影子/镜像/物件）
            'gallery_out': 0,        # 真值人物不在图库
            'answerable': 0,         # 196
            'accepted': 0,
            'correct': 0,
            'wrong': 0,
            'unanswered': 0,
            'ua_both_low': 0,        # 两片都未过阈值
            'ua_one_pass': 0,        # 仅一片过阈值
            'ua_ab_split': 0,        # 两片都被接受但落在不同簇
            'wrong_from_unknown_cluster': 0,
        }

    agg = {'ALL': bucket()}
    for c in cams:
        agg[c] = bucket()

    # 1) 全量 295 的拆分
    for r in gt:
        cam = r['cam']
        for key in ('ALL', cam):
            agg[key]['queries'] += 1
        if r['evaluable'] != '1':
            for key in ('ALL', cam):
                agg[key]['dropped'] += 1
            continue
        for key in ('ALL', cam):
            agg[key]['evaluable'] += 1

    # 2) 229 的可评估查询
    for r in gt:
        if r['evaluable'] != '1':
            continue
        cam = r['cam']
        keys = ('ALL', cam)
        pid = r['person_id']
        q = qmap.get(r['query_id'])
        if _is_nonperson(pid):
            for k in keys:
                agg[k]['nonperson'] += 1
            continue
        if not pid:
            for k in keys:
                agg[k]['no_truth'] += 1
            continue
        if pid not in gallery_ids:
            for k in keys:
                agg[k]['gallery_out'] += 1
            continue
        for k in keys:
            agg[k]['answerable'] += 1
        if q is None:
            continue
        status = q['final_status']
        if status == 'accepted':
            for k in keys:
                agg[k]['accepted'] += 1
            cid = q['final_cluster_local']
            pred = main_pid.get(cid, '')
            ok = (pred == pid)
            for k in keys:
                agg[k]['correct' if ok else 'wrong'] += 1
                if not ok and cid and cid in main_pid and pred == '':
                    agg[k]['wrong_from_unknown_cluster'] += 1
        else:
            for k in keys:
                agg[k]['unanswered'] += 1
            if status == 'rejected_ab_split':
                for k in keys:
                    agg[k]['ua_ab_split'] += 1
            elif q['accept_a'] == '0' and q['accept_b'] == '0':
                for k in keys:
                    agg[k]['ua_both_low'] += 1
            else:
                for k in keys:
                    agg[k]['ua_one_pass'] += 1

    # ---------------- 输出 ----------------
    order = ['ALL'] + cams
    fields = ['group', 'queries', 'dropped', 'evaluable', 'no_truth', 'nonperson',
              'gallery_out', 'answerable', 'accepted', 'correct', 'wrong', 'unanswered',
              'ua_both_low', 'ua_one_pass', 'ua_ab_split']
    os.makedirs(os.path.join(pack, 'results'), exist_ok=True)
    out_csv = os.path.join(pack, 'results/loss_decomposition.csv')
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in order:
            row = {'group': k}
            row.update({f: agg[k].get(f, 0) for f in fields if f != 'group'})
            w.writerow(row)

    payload = {'config': args.config, 'groups': {k: dict(agg[k]) for k in order}}
    out_json = os.path.join(pack, 'results/loss_decomposition.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 一致性断言
    for k in order:
        a = agg[k]
        assert a['evaluable'] + a['dropped'] == a['queries'], (k, 'stage1')
        assert (a['no_truth'] + a['nonperson'] + a['gallery_out'] + a['answerable']
                == a['evaluable']), (k, 'stage2')
        assert a['correct'] + a['wrong'] + a['unanswered'] == a['answerable'], (k, 'stage3')
        assert a['ua_both_low'] + a['ua_one_pass'] + a['ua_ab_split'] == a['unanswered'], (k, 'stage4')
        assert a['accepted'] == a['correct'] + a['wrong'], (k, 'accepted')

    print('%-4s %6s %7s %9s %8s %9s %10s %8s %8s %7s %6s %10s %9s %9s %8s' % (
        'grp', 'query', 'dropped', 'evaluable', 'no_truth', 'nonperson', 'gal_out', 'answerable',
        'accepted', 'C', 'W', 'unanswered', 'both_low', 'one_pass', 'absplit'))
    for k in order:
        a = agg[k]
        print('%-4s %6d %7d %9d %8d %9d %10d %8d %8d %7d %6d %10d %9d %9d %8d' % (
            k, a['queries'], a['dropped'], a['evaluable'], a['no_truth'], a['nonperson'],
            a['gallery_out'],
            a['answerable'], a['accepted'], a['correct'], a['wrong'], a['unanswered'],
            a['ua_both_low'], a['ua_one_pass'], a['ua_ab_split']))
    print('\nwritten:', out_csv, '|', out_json)
    # 派生比率
    for k in order:
        a = agg[k]
        acc_rate = a['accepted'] / a['answerable'] if a['answerable'] else 0
        prec = a['correct'] / a['accepted'] if a['accepted'] else 0
        print('%-4s 应答率 %.3f | 精度 %.3f | C/196 %.3f | C/229 %.3f' % (
            k, acc_rate, prec, a['correct'] / 196, a['correct'] / 229))


if __name__ == '__main__':
    main()
