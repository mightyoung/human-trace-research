#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B 前置诊断：25 个已确认的换人点，在检测日志里到底有什么信号？

不先修规则——先问「这些换人点有没有可用的机器信号」。逐点检查四类证据：

  A 断开：切点前后 ±1s 内该 tid 有没有 >1s 的观测空档
  B 交错：切点 ±1s 内该 tid 与**另一个** tid 的框 IoU 是否 ≥0.30
  C 跳框：切点前 0.4s 与后 0.4s 的框 IoU 是否 <0.30（框突然跳到别处）
  D 尺寸跳变：切点后的框宽 / 前 1s 中位宽 是否 >1.5

输出 `results/switch_signals.csv` + 汇总。

用法：PY tools/diag_switch_signals.py
"""
import csv
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
DET = os.path.join(ROOT, 'output_crosscam_local', '1012_yolo8_dd')
IN_CSV = os.path.join(DELIV, 'results/m1a_switch_review.csv')
OUT = os.path.join(DELIV, 'results/switch_signals.csv')
FPS = 25.0
IOU_OVL = 0.30
WIN = 25                     # ±1s
JUMP_D = 10                  # 前后 0.4s


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load(seg):
    p = os.path.join(DET, seg, f'{seg}_detections.jsonl')
    by_frame = {}
    with open(p, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            by_frame[d['frame']] = {int(t): b for t, b in zip(d['track_ids'], d['boxes'])}
    return by_frame


def nearest(by_frame, frame):
    import bisect
    fs = sorted(by_frame)
    i = bisect.bisect_left(fs, frame)
    c = [j for j in (i - 1, i) if 0 <= j < len(fs)]
    j = min(c, key=lambda k: abs(fs[k] - frame))
    return fs[j], by_frame[fs[j]]


def main():
    rows = [r for r in csv.DictReader(open(IN_CSV, encoding='utf-8-sig'))
            if r['cls'] in ('usable', 'tail', 'head')]
    by_seg = {}
    out = []
    for r in rows:
        seg = r['seg']
        if seg not in by_seg:
            by_seg[seg] = load(seg)
        det = by_seg[seg]
        tid, f = int(r['track_id'].split('#')[1]), int(float(r['new_frame']))
        frame = f

        # A 断开：±1s 内最近的两个观测间隔
        obs = sorted(k for k, m in det.items() if tid in m)
        gaps = [(a, b) for a, b in zip(obs, obs[1:]) if a <= frame + WIN and b >= frame - WIN]
        gap = max((b - a for a, b in gaps), default=0)

        # B 交错：±1s 内与其它 tid 的最大 IoU
        best_iou, best_other = 0.0, None
        for fa, m in det.items():
            if abs(fa - frame) > WIN or tid not in m:
                continue
            for t2, b2 in m.items():
                if t2 == tid:
                    continue
                v = iou(m[tid], b2)
                if v > best_iou:
                    best_iou, best_other = v, t2

        # C 跳框：切点前后 0.4s 的框 IoU
        f_pre, m_pre = nearest(det, frame - JUMP_D)
        f_post, m_post = nearest(det, frame + JUMP_D)
        if tid in m_pre and tid in m_post:
            jump_iou = iou(m_pre[tid], m_post[tid])
        else:
            jump_iou = 0.0      # 一侧没有 → 视为跳变

        # D 尺寸跳变
        pre_w = [m[tid][2] - m[tid][0] for fa, m in det.items()
                 if frame - 25 - WIN <= fa <= frame - JUMP_D and tid in m]
        _, m_at = nearest(det, frame)
        w_post = (m_at[tid][2] - m_at[tid][0]) if tid in m_at else None
        grow = (w_post / statistics.median(pre_w)) if (w_post and len(pre_w) >= 3) else None

        out.append(dict(track_id=r['track_id'], cam=r['cam'], cls=r['cls'],
                        frame=frame, pos_ratio=r['pos_ratio'],
                        gap_max=int(gap), overlap_iou=round(best_iou, 3), other_tid=best_other or '',
                        jump_iou=round(jump_iou, 3), grow='' if grow is None else round(grow, 2),
                        sig_gap=int(gap > 25), sig_ovl=int(best_iou >= IOU_OVL),
                        sig_jump=int(jump_iou < IOU_OVL)))

    with open(OUT, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n = len(out)
    g = sum(x['sig_gap'] for x in out)
    o = sum(x['sig_ovl'] for x in out)
    j = sum(x['sig_jump'] for x in out)
    any_ = sum(1 for x in out if x['sig_gap'] or x['sig_ovl'] or x['sig_jump'])
    print('%d 个已确认换人点的机器信号：' % n)
    print('  A 断开(>1s 空档)      %2d (%.0f%%)' % (g, 100 * g / n))
    print('  B 交错(IoU≥0.30)      %2d (%.0f%%)' % (o, 100 * o / n))
    print('  C 跳框(前后 IoU<0.30) %2d (%.0f%%)' % (j, 100 * j / n))
    print('  三类任一命中          %2d (%.0f%%)' % (any_, 100 * any_ / n))
    print('\n只有 A 的：%d｜只有 B 的：%d｜只有 C 的：%d｜三者都无：%d'
          % (sum(1 for x in out if x['sig_gap'] and not x['sig_ovl'] and not x['sig_jump']),
             sum(1 for x in out if x['sig_ovl'] and not x['sig_gap'] and not x['sig_jump']),
             sum(1 for x in out if x['sig_jump'] and not x['sig_gap'] and not x['sig_ovl']),
             n - any_))
    print('\n逐点（cam 位置 断开 交错 跳框）：')
    for x in out:
        print('  %-22s %s %.2f  gap=%3d ovl=%.2f jump=%.2f%s'
              % (x['track_id'], x['cam'], float(x['pos_ratio']), x['gap_max'],
                 x['overlap_iou'], x['jump_iou'],
                 '  <<' if not (x['sig_gap'] or x['sig_ovl'] or x['sig_jump']) else ''))
    print('\n→ %s' % OUT)


if __name__ == '__main__':
    main()
