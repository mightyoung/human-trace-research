#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-A 回收：把复核页导出的 switch 时刻并回台账，产出「可用切点」清单。

判据要求：切点必须落在轨迹内部（离开起点和终点都 ≥ 2 秒），否则"切开"只会得到空片段。
本脚本做：
  1. 校验每条：帧号在 [t_start+2s, t_end-2s] 内；与 t_end 重合的判无效
  2. 输出 gt/gt_gallery_members_SWITCHFIX.csv（track_id, old_frame, new_frame, status）
  3. 打印：46 条里多少条现在可用；按机位分组
  4. 顺带产出 `switch_valid.json`（M1 判据直接读这个）

用法：
  PY tools/ingest_switch_review.py --csv C:/Users/xxx/Downloads/a3_switch_review_FILLED.csv
"""
import argparse
import collections
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
LEDGER = os.path.join(DELIV, 'ledger/tracks.csv')
GAL = os.path.join(DELIV, 'gt/gt_gallery_members_FILLED.csv')
OUT_CSV = os.path.join(DELIV, 'gt/gt_gallery_members_SWITCHFIX.csv')
OUT_JSON = os.path.join(DELIV, 'results/switch_valid.json')
FPS = 25.0
EDGE_MIN_S = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--fps', type=float, default=FPS)
    a = ap.parse_args()

    led = {r['track_id']: r for r in csv.DictReader(open(LEDGER, encoding='utf-8-sig'))}
    gal = [r for r in csv.DictReader(open(GAL, encoding='utf-8-sig'))
           if r['id_switch'].strip() in ('1', '1.0')]
    want = {r['track_id'] for r in gal}

    rows = list(csv.DictReader(open(a.csv, encoding='utf-8-sig')))
    got = {}
    for r in rows:
        tid = (r.get('track_id') or '').strip()
        v = (r.get('switch_frame') or '').strip()
        if tid and v:
            try:
                got[tid] = int(float(v))
            except ValueError:
                pass

    edge = int(round(EDGE_MIN_S * a.fps))
    out, valid, stat = [], {}, collections.Counter()
    for tid in sorted(want):
        t = led[tid]
        f0, f1 = int(t['t_start_frame']), int(t['t_end_frame'])
        new = got.get(tid)
        if new is None:
            status = 'missing'
        elif not (f0 + edge <= new <= f1 - edge):
            status = 'edge'
        else:
            status = 'ok'
        stat[status] += 1
        if status == 'ok':
            valid[tid] = new
        out.append(dict(track_id=tid, seg=t['seg'], cam=t['cam'],
                        t_start=f0, t_end=f1, old_frame='', new_frame=new or '',
                        status=status))

    with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(dict(fps=a.fps,
                   usable={k: v for k, v in valid.items()},
                   stats=dict(stat)),
              open(OUT_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    print('46 条串人成员：可用切点 %d ｜ 贴边被拒 %d ｜ 未填 %d'
          % (stat['ok'], stat['edge'], stat['missing']))
    by = collections.Counter(led[k]['cam'] for k in valid)
    print('  可用按机位：%s' % dict(by))
    print('  产出：%s' % OUT_CSV)
    print('        %s' % OUT_JSON)
    if stat['ok'] < len(want):
        print('  → 未达 100%% 也可开跑：M1 的"拆开率"分母改用 %d 条，并在论文里写明有效样本量。'
              % stat['ok'])


if __name__ == '__main__':
    main()
