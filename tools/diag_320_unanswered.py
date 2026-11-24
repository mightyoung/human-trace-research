#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""320 机位「答得少」的逐条拆解（M2 前置，零新数据）。

问题：320 的 31 条可回答查询只答对 4 条（应答率 12.9%，其余三机位 59–82%）。
要分清三种可能：
  A 分数根本到不了 θ（两侧都不过阈值）
  B 分数到了，但 a/b 两片不一致
  C 分数到了也一致，但 top-1 簇本身没身份（图库侧覆盖不足）

做法：复用 diag_320_gallery.build_variant（与 A_baseline 同口径：生产图库、无 veto、
K=111、θ=1.85），对每条测试查询取两片 argmax 与最高分，按机位统计。
注意：本脚本只读，不写生产产物；输出 results/diag_320_unanswered.{json,csv}

用法：
  PY tools/diag_320_unanswered.py
"""
import argparse
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
CAMS = ('305', '318', '320', '323')


def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb-dir', default=os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd_emb_diag'))
    ap.add_argument('--gt', default=os.path.join(ROOT, 'deliverable/data-request-v2/gt'))
    ap.add_argument('--theta', type=float, default=1.85, help='A_baseline 的 θ（no-veto, K=111）')
    ap.add_argument('--out-prefix', default=os.path.join(ROOT, 'deliverable/data-request-v2/results/diag_320_unanswered'))
    a = ap.parse_args()

    gal, qry = load_gt(a.gt)
    gallery_ids = set(gal.values())
    res = D.build_variant(a.emb_dir, list(S.TRAIN_SEGS), veto=None)

    # 簇主体身份（与 D4 一致：已标成员投票多数派）
    cid_pid = {}
    for pi, c in enumerate(res['kept']):
        votes = collections.Counter(gal[k] for k in
                                    [res['ktr'][i] for i in np.where(res['lab'] == c)[0]] if k in gal)
        if votes:
            cid_pid[pi] = votes.most_common(1)[0][0]
    print('[build] 图库轨迹 %d | K=%d | θ=%.2f' % (res['n_tr'], res['K'], a.theta))

    rows = []
    for i, k in enumerate(res['kte']):
        pid, ev = qry.get(k, ('', False))
        sa = res['Sa'][i]
        sb = res['Sb'][i]
        ia, ib = int(np.argmax(sa)), int(np.argmax(sb))
        pa, pb = cid_pid.get(ia, ''), cid_pid.get(ib, '')
        rows.append(dict(
            key='%s#%d' % (k[0], k[1]), seg=k[0], cam=k[0].split('_')[0],
            gt=pid, ev=int(ev),
            score_a=float(sa[ia]), score_b=float(sb[ib]),
            score_max=float(max(sa[ia], sb[ib])),
            score_mean=float((sa[ia] + sb[ib]) / 2.0),
            pid_a=pa, pid_b=pb,
            named_both=int(bool(pa) and bool(pb)),
            top1_ok=int(bool(pa) and pa == pid and pb == pid),
            top1_any=int((pa == pid) or (pb == pid)),
        ))

    # ---- 分机位汇总 ----
    summary = {}
    for cam in ('ALL',) + CAMS:
        sub = [r for r in rows if (cam == 'ALL' or r['cam'] == cam) and r['ev'] and r['gt']]
        ans = [r for r in sub if r['gt'] in gallery_ids]
        named = [r for r in ans if r['named_both']]
        low = [r for r in ans if r['score_max'] < a.theta]
        gap = [r for r in ans if a.theta - 0.25 <= r['score_max'] < a.theta]
        summary[cam] = dict(
            n_eval=len(sub), n_answerable=len(ans), n_named=len(named),
            named_rate=len(named) / max(1, len(ans)),
            top1_hit_named=sum(r['top1_ok'] for r in named),
            top1_rate_named=sum(r['top1_ok'] for r in named) / max(1, len(named)),
            top1_hit_ans=sum(r['top1_ok'] for r in ans),
            top1_rate_ans=sum(r['top1_ok'] for r in ans) / max(1, len(ans)),
            n_below_theta=len(low),
            n_within_025=len(gap),
            score_med=pct([r['score_max'] for r in ans], 50),
            score_p25=pct([r['score_max'] for r in ans], 25),
            score_p75=pct([r['score_max'] for r in ans], 75),
        )

    print('\n%-4s %7s %9s %8s %8s %11s %9s %9s %8s' % (
        'cam', 'eval', 'answerable', 'named', 'named%', 'top1@named', 'top1@ans', '<θ', 'score中位'))
    for cam, v in summary.items():
        print('%-4s %7d %9d %8d %7.1f%% %11s %8.1f%% %9d %8.2f' % (
            cam, v['n_eval'], v['n_answerable'], v['n_named'], v['named_rate'] * 100,
            '%d/%d' % (v['top1_hit_named'], v['n_named']), v['top1_rate_named'] * 100,
            v['n_below_theta'], v['score_med']))

    # ---- 320 的机制归因：把 31 条可回答拆成 互斥 4 类 ----
    s320 = [r for r in rows if r['cam'] == '320' and r['ev'] and r['gt'] and r['gt'] in gallery_ids]
    mech = collections.Counter()
    for r in s320:
        if r['score_max'] < a.theta:
            mech['A 分数到不了 θ（两侧都不过）'] += 1
        elif not r['named_both']:
            mech['C 过阈值但候选簇无身份'] += 1
        elif r['top1_ok']:
            mech['D 答对'] += 1
        else:
            mech['E 过阈值、top-1 有身份但错'] += 1
    print('\n320 的 %d 条可回答查询归因：' % len(s320))
    for k, v in mech.most_common():
        print('  %-28s %2d (%.1f%%)' % (k, v, v / max(1, len(s320)) * 100))

    # 320 若把 θ 降到多少才能答到与 305 同等的比例？给分位用于分机位校准
    q = [r['score_max'] for r in s320]
    print('\n320 的 top-1 分数分位：p50 %.2f | p75 %.2f | p90 %.2f | max %.2f（θ=%.2f）' % (
        pct(q, 50), pct(q, 75), pct(q, 90), max(q) if q else float('nan'), a.theta))

    payload = dict(theta=a.theta, K=res['K'], n_tr=res['n_tr'],
                   summary=summary, mech_320=dict(mech),
                   rows_320=[r for r in rows if r['cam'] == '320'])
    os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)
    with open(a.out_prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(a.out_prefix + '.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('\nwritten:', a.out_prefix + '.json', '|', a.out_prefix + '.csv')


if __name__ == '__main__':
    main()
