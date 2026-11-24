#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 第一步：把「交错感知切分」实现为对现有检测日志的后处理（零推理）。

为什么这么做：`tracklet_extract.py` 不在本仓库、`boxmot` 也没装，但
`output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl`
已经逐帧保存了 track_ids / boxes / scores。V0.4 §4.1 里 6 条规则中的 5 条
只需要框，不需要重新推理，也不需要 BoT-SORT。

已实现（参数照 V0.4 默认值）：
  R1 断开：同一 tid 相邻观测间隔 > 1 秒 → 切
  R2 交错：两人框 IoU ≥ 0.30 记为重叠（只标记不切）；重叠结束后比较前后 1 秒的
          相对位置，水平或纵向顺序互换超过平均框宽/高的 15% → 在重叠中点切双方
  R3 消失：重叠结束时一方已不见（前后缺对照）→ 保守切断仍在的一方
  R4 单框多人：框宽 > 近 1 秒中位数的 1.5 倍 → 切

未实现（日志里没有）：掩码重叠 ≥15%、轨迹内外观突变切分。按 V0.4 自己的结论
「重叠切分是纯度的主力、外观切分只是补充」，缺这两条不阻断第一轮验证。

用法：
  PY tools/tracklet_split.py --seg 323_train_510
  PY tools/tracklet_split.py --all --out output_crosscam_local/split_v1
输出：<out>/<seg>_subtracks.jsonl（每条子轨迹：tid、起止帧、规则、片段序号）+ 统计
"""
import argparse
import collections
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = os.path.join(ROOT, 'output_crosscam_local', '1012_yolo8_dd')
SEGS = ['305_train_529', '318_train_525', '320_train_514', '323_train_510',
        '305_test_544', '318_test_534', '320_test_518', '323_test_527']

STRIDE = 5          # 检测采样步长（帧）
FPS = 25.0          # 原始帧率（假设，见 SKIP 说明）
GAP_FRAMES = int(round(1.0 * FPS))       # R1：断开 >1 秒
IOU_OVL = 0.30      # R2：重叠判据
SWAP_RATIO = 0.15   # R2：顺序互换阈值（平均框宽/高的 15%）
GROW_RATIO = 1.5    # R4：尺寸突增


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_seg(seg):
    p = os.path.join(DET, seg, f'{seg}_detections.jsonl')
    obs = collections.defaultdict(dict)      # tid -> frame -> box
    with open(p, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            fr = d['frame']
            for tid, box in zip(d['track_ids'], d['boxes']):
                obs[tid][fr] = box
    return obs


def split_seg(seg, obs):
    cuts = collections.defaultdict(set)      # tid -> set(frame)
    reasons = collections.defaultdict(collections.Counter)

    # R1 断开
    for tid, fr2box in obs.items():
        fs = sorted(fr2box)
        for a, b in zip(fs, fs[1:]):
            if b - a > GAP_FRAMES:
                cuts[tid].add(b)
                reasons[tid]['R1_gap'] += 1

    # R4 尺寸突增
    for tid, fr2box in obs.items():
        fs = sorted(fr2box)
        w = [fr2box[f][2] - fr2box[f][0] for f in fs]
        for i, f in enumerate(fs):
            lo = max(0, i - int(round(FPS / STRIDE)))
            ref = statistics.median(w[lo:i]) if i - lo >= 2 else None
            if ref and w[i] > GROW_RATIO * ref:
                cuts[tid].add(f)
                reasons[tid]['R4_grow'] += 1

    # R2/R3 交错与消失
    alive = collections.defaultdict(set)     # frame -> tids
    for tid, fr2box in obs.items():
        for f in fr2box:
            alive[f].add(tid)
    frames = sorted(alive)

    for i, fa in enumerate(frames):
        tids = sorted(alive[fa])
        for x in range(len(tids)):
            for y in range(x + 1, len(tids)):
                a, b = tids[x], tids[y]
                if iou(obs[a][fa], obs[b][fa]) < IOU_OVL:
                    continue
                # 找到这次重叠的结束帧
                end = None
                for fb in frames[i:]:
                    if a not in alive[fb] or b not in alive[fb]:
                        end = fb
                        break
                    if iou(obs[a][fb], obs[b][fb]) < IOU_OVL:
                        end = fb
                        break
                if end is None:
                    continue
                pre = [f for f in frames[:frames.index(fa)]][-1:] if fa in frames else []
                pre = [f for f in frames if f < fa]
                post = [f for f in frames if f >= end]
                if not pre or not post:
                    continue
                f_pre, f_post = pre[-1], post[0]
                if a not in alive[f_pre] or b not in alive[f_pre]:
                    continue
                if b not in alive[f_post]:
                    cuts[a].add(end)
                    reasons[a]['R3_lost'] += 1
                    continue
                if a not in alive[f_post]:
                    cuts[b].add(end)
                    reasons[b]['R3_lost'] += 1
                    continue
                # 顺序互换
                wx = abs((obs[a][f_post][0] + obs[a][f_post][2]) / 2
                         - (obs[b][f_post][0] + obs[b][f_post][2]) / 2)
                wy = abs((obs[a][f_post][1] + obs[a][f_post][3]) / 2
                         - (obs[b][f_post][1] + obs[b][f_post][3]) / 2)
                dx = (obs[a][f_post][0] + obs[a][f_post][2]) / 2 - (obs[b][f_post][0] + obs[b][f_post][2]) / 2
                dx0 = (obs[a][f_pre][0] + obs[a][f_pre][2]) / 2 - (obs[b][f_pre][0] + obs[b][f_pre][2]) / 2
                dy = (obs[a][f_post][1] + obs[a][f_post][3]) / 2 - (obs[b][f_post][1] + obs[b][f_post][3]) / 2
                dy0 = (obs[a][f_pre][1] + obs[a][f_pre][3]) / 2 - (obs[b][f_pre][1] + obs[b][f_pre][3]) / 2
                mw = statistics.mean([obs[a][f_post][2] - obs[a][f_post][0],
                                      obs[b][f_post][2] - obs[b][f_post][0]])
                mh = statistics.mean([obs[a][f_post][3] - obs[a][f_post][1],
                                      obs[b][f_post][3] - obs[b][f_post][1]])
                if (dx * dx0 < 0 and abs(dx - dx0) > SWAP_RATIO * mw) or \
                   (dy * dy0 < 0 and abs(dy - dy0) > SWAP_RATIO * mh):
                    mid = (fa + end) // 2
                    cuts[a].add(mid)
                    cuts[b].add(mid)
                    reasons[a]['R2_swap'] += 1
                    reasons[b]['R2_swap'] += 1

    subtracks = []
    for tid, fr2box in obs.items():
        fs = sorted(fr2box)
        cs = sorted(c for c in cuts[tid] if fs[0] < c <= fs[-1])
        bounds = [fs[0]] + cs + [fs[-1] + STRIDE]
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1] - STRIDE
            if b < a:
                continue
            subtracks.append(dict(tid=tid, part=k, f0=a, f1=b,
                                  n_samples=len([f for f in fs if a <= f <= b])))
    return subtracks, cuts, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default=os.path.join(ROOT, 'output_crosscam_local', 'split_v1'))
    a = ap.parse_args()
    segs = SEGS if a.all else ([a.seg] if a.seg else ['323_train_510'])
    os.makedirs(a.out, exist_ok=True)
    total_tracks = total_sub = 0
    for seg in segs:
        try:
            obs = load_seg(seg)
        except FileNotFoundError:
            print('[skip] %s 无检测日志' % seg)
            continue
        st, cuts, reasons = split_seg(seg, obs)
        n_tracks = len(obs)
        n_cut = sum(1 for t in cuts if cuts[t])
        frag = collections.Counter(t['tid'] for t in st)
        med = statistics.median(frag.values()) if frag else 0
        short = sum(1 for t in st if t['n_samples'] < 6)   # 与图库 min-frames 6 同口径
        print('[%s] 轨迹 %d → 子轨迹 %d | 有切点的轨迹 %d (%.1f%%) | 每人碎片中位 %.1f | '
              '样本数<6 的子轨迹 %d (%.1f%%) | 规则 %s'
              % (seg, n_tracks, len(st), n_cut, 100 * n_cut / max(1, n_tracks), med,
                 short, 100 * short / max(1, len(st)),
                 dict(sum(reasons.values(), collections.Counter()))))
        with open(os.path.join(a.out, f'{seg}_subtracks.jsonl'), 'w', encoding='utf-8') as f:
            for t in st:
                f.write(json.dumps(t, ensure_ascii=False) + '\n')
        total_tracks += n_tracks
        total_sub += len(st)
    print('\n合计 %d 轨迹 → %d 子轨迹（%.2f 倍）' % (total_tracks, total_sub,
                                              total_sub / max(1, total_tracks)))
    print('written:', a.out)


if __name__ == '__main__':
    main()
