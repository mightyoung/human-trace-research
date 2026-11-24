#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B2：外观通道到底有没有用？——用已有的逐样本嵌入测「换人点前后余弦」。

背景（用户质疑）：「遮挡了 ReID 肯定消失了」。
这一条要分开看：
  · 被遮挡的轨迹（25 个可用点里 7 个属于 missing）→ 确实没有可用外观，几何的空档规则才是对的；
  · 框全程可见的轨迹（14/25 属于 smooth）→ 人一直在画面里，外观没"消失"，
    问题变成：**余弦在换人点前后是否真的会掉**。这一条可以测，不用重新推理。

数据：`output_crosscam_local/1012_osnet_crops/<seg>_crops.npz`
  eids  a#<tid>#<k> / b#<tid>#<k>（每半 8 个样本，linspace 采样）
  emb   (N, 512)
  q     (N, 3) = [置信度, ?, 框面积]
`1012_ab/<seg>_{a,b}.jsonl` 保存了逐帧检测，可按 linspace 复原每个样本的帧号。

输出：`results/appearance_split.json` + 终端对照表。

用法：PY tools/diag_appearance_split.py
"""
import collections
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
AB = os.path.join(ROOT, 'output_crosscam_local/1012_ab')
CROPS = os.path.join(ROOT, 'output_crosscam_local/1012_osnet_crops')
OUT = os.path.join(DELIV, 'results/appearance_split.json')
SAMPLES = 8
FPS = 25.0


def load_ab(seg):
    """返回 {half: {tid: [(frame, box), ...]}}。"""
    out = {}
    for half in ('a', 'b'):
        p = os.path.join(AB, '%s_%s.jsonl' % (seg, half))
        d = collections.defaultdict(list)
        with open(p, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                for t, b in zip(r['track_ids'], r['boxes']):
                    d[int(t)].append((r['frame'], b))
        for t in d:
            d[t].sort()
        out[half] = d
    return out


def sample_frames(tf, n=SAMPLES):
    if len(tf) < 2:
        return []
    idx = np.linspace(0, len(tf) - 1, min(n, len(tf))).astype(int)
    return [tf[i][0] for i in idx], [tf[i][1] for i in idx]


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                           encoding='utf-8-sig')) if r['new_frame']]
    segs = sorted({r['seg'] for r in marks})
    res = {'switch': [], 'control': []}
    for seg in segs:
        z = np.load(os.path.join(CROPS, '%s_crops.npz' % seg), allow_pickle=True)
        eids, emb, q = z['eids'], z['emb'], z['q']
        idx = {str(e): i for i, e in enumerate(eids)}
        ab = load_ab(seg)

        def series(tid):
            """按帧序返回 [(frame, emb, area)]，16 个样本。"""
            out = []
            for half in ('a', 'b'):
                tf = ab[half].get(tid, [])
                got = sample_frames(tf)
                if not got:
                    continue
                frames, boxes = got
                for k, (fr, bx) in enumerate(zip(frames, boxes)):
                    i = idx.get('%s#%d#%d' % (half, tid, k))
                    if i is None:
                        continue
                    out.append((fr, emb[i], (bx[2] - bx[0]) * (bx[3] - bx[1])))
            out.sort(key=lambda x: x[0])
            return out

        # —— 换人点：跨切点那一对的余弦 vs 同轨迹其它相邻对 ——
        for r in marks:
            if r['seg'] != seg:
                continue
            tid = int(r['track_id'].split('#')[1])
            f = int(float(r['new_frame']))
            s = series(tid)
            if len(s) < 3:
                continue
            pairs = [(s[i][0], s[i + 1][0], cos(s[i][1], s[i + 1][1]),
                      abs(s[i + 1][2] / max(1e-6, s[i][2]) - 1)) for i in range(len(s) - 1)]
            straddle = [p for p in pairs if p[0] <= f <= p[1]]
            rest = [p[2] for p in pairs if not (p[0] <= f <= p[1])]
            if not straddle:
                continue
            res['switch'].append(dict(
                track=r['track_id'], cls=r['cls'], cam=r['cam'], pos=float(r['pos_ratio']),
                cos_straddle=round(straddle[0][2], 3),
                area_jump=round(straddle[0][3], 3),
                cos_other_median=round(float(np.median(rest)), 3) if rest else None,
                gap_frames=straddle[0][1] - straddle[0][0]))

        # —— 对照：同段未标注轨迹的相邻对 ——
        labeled = {int(r['track_id'].split('#')[1]) for r in marks if r['seg'] == seg}
        tids = [t for t in ab['a'].keys() if t not in labeled]
        rng = np.random.RandomState(0)
        for tid in rng.choice(tids, size=min(60, len(tids)), replace=False):
            s = series(int(tid))
            if len(s) < 3:
                continue
            for i in range(len(s) - 1):
                res['control'].append(dict(
                    cos=round(cos(s[i][1], s[i + 1][1]), 3),
                    area_jump=round(abs(s[i + 1][2] / max(1e-6, s[i][2]) - 1), 3),
                    gap_frames=s[i + 1][0] - s[i][0]))

    sw = [x for x in res['switch'] if x['cos_other_median'] is not None]
    ctrl = [x['cos'] for x in res['control']]
    print('换人点样本 %d 条（能定位到跨切点样本对的）；对照相邻对 %d' % (len(sw), len(ctrl)))

    def q(v, p):
        return sorted(v)[min(len(v) - 1, int(p / 100 * (len(v) - 1)))]

    print('\n对照（同段未标注轨迹的相邻样本余弦）：p10 %.3f  p50 %.3f  p90 %.3f'
          % (q(ctrl, 10), q(ctrl, 50), q(ctrl, 90)))
    for grp, lab in ((('usable',), 'usable'), (('tail', 'head'), 'tail/head')):
        sub = [x for x in sw if x['cls'] in grp]
        if not sub:
            continue
        cs = [x['cos_straddle'] for x in sub]
        print('  %-10s n=%2d  跨切点余弦 p10 %.3f  p50 %.3f  p90 %.3f'
              % (lab, len(sub), q(cs, 10), q(cs, 50), q(cs, 90)))

    smooth = [x for x in sw if x['cls'] == 'usable']
    drops = [x['cos_other_median'] - x['cos_straddle'] for x in smooth
             if x['cos_other_median'] is not None]
    print('\nusable 里「跨切点余弦」比「同轨迹其它相邻对中位」低多少：'
          '中位 %.3f，>0.10 的 %d/%d' % (float(np.median(drops)), sum(1 for d in drops if d > 0.10), len(drops)))

    # 遮挡混淆：对照集里余弦与面积跳变的关系
    big = [x['cos'] for x in res['control'] if x['area_jump'] > 0.5]
    small = [x['cos'] for x in res['control'] if x['area_jump'] <= 0.2]
    print('\n对照集按面积跳变分组（测「遮挡/进出画是否本身就让余弦掉」）：')
    print('  |Δ面积|≤20%%：n=%d 余弦中位 %.3f｜>50%%：n=%d 余弦中位 %.3f'
          % (len(small), float(np.median(small)) if small else float('nan'),
             len(big), float(np.median(big)) if big else float('nan')))

    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n→ %s' % OUT)
    print('\n逐条（usable）：跨切点余弦 vs 其它中位 vs 面积跳变')
    for x in sorted(smooth, key=lambda y: y['cos_straddle']):
        print('  %-22s pos=%.2f  跨 %.3f  其它 %.3f  Δ面积 %.2f  gap %d 帧'
              % (x['track'], x['pos'], x['cos_straddle'],
                 x['cos_other_median'], x['area_jump'], x['gap_frames']))


if __name__ == '__main__':
    main()
