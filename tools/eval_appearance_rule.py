#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B5：把「外观切分」从离线 argmin 变成可上线规则——扫阈值，同时报命中与误切。

上一轮已经证明：在 ±4s 窗口里取余弦最低点，83% 落在人工切点 ±1s 内。
但规则不能靠 argmin，得靠一个固定阈值 τ：**相邻样本余弦 < τ 就切**。
τ 定得太松会到处切（碎片），太紧则召回不够。本脚本在真实分布上扫这条曲线。

设计：
  召回  —— 25 条 usable 切点，在 ±8s 内每 0.4s 采样，若存在 cos<τ 的一跳落在 ±1s 内则算命中；
  误切  —— 两类证据
         ① 同一条标注轨迹里、离切点 >1s 的其它跳低于 τ 的个数（说明这条规则会在正常片段里乱切）
         ② 30 条未标注轨迹（各取 16s 随机窗）里低于 τ 的跳数 → 折算「每轨每分钟误切数」

裁剪与编码：dd 检测框 → 原视频现裁（含 8% 边距）→ OSNet x1_0/MSMT17 → L2 归一化。
向量缓存到 `output_crosscam_local/emb_cache/switch_emb.npz`，重跑不再编码。

用法：PY tools/eval_appearance_rule.py [--span 8] [--step 10] [--ctrl 30]
输出：results/appearance_rule_sweep.csv + 终端表
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
DET = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd')
SEG_DIR = os.path.join(ROOT, 'media-resource/1012/segments')
CACHE = os.path.join(ROOT, 'output_crosscam_local/emb_cache/switch_emb.npz')
OUT = os.path.join(DELIV, 'results/appearance_rule_sweep.csv')
W = ('models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_'
     'stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
FPS = 25.0


def load_det(seg):
    d = {}
    for line in open(os.path.join(DET, seg, '%s_detections.jsonl' % seg), encoding='utf-8'):
        r = json.loads(line)
        d[r['frame']] = {int(t): b for t, b in zip(r['track_ids'], r['boxes'])}
    return d


def nearest(det_sorted, det, frame):
    import bisect
    i = bisect.bisect_left(det_sorted, frame)
    c = [j for j in (i - 1, i) if 0 <= j < len(det_sorted)]
    j = min(c, key=lambda k: abs(det_sorted[k] - frame))
    return det_sorted[j], det[det_sorted[j]]


def crop(cap, frame, box, margin=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok:
        return None
    H, Wd = img.shape[:2]
    x1, x2 = max(0, int(x1)), min(Wd, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 10 or y2 - y1 < 24:
        return None
    return img[y1:y2, x1:x2].copy()


class Enc:
    def __init__(self):
        from torchreid.reid.utils import FeatureExtractor
        import torch
        torch.set_flush_denormal(True)
        self.fe = FeatureExtractor(model_name='osnet_x1_0',
                                   model_path=os.path.join(ROOT, W),
                                   device='cpu', verbose=False)
        self.torch = torch
        self.cache = {}
        if os.path.exists(CACHE):
            z = np.load(CACHE, allow_pickle=True)
            self.cache = {str(k): v for k, v in zip(z['keys'], z['vecs'])}
            print('[cache] %d 条向量' % len(self.cache))

    def get(self, seg, tid, frame, img):
        key = '%s|%d|%d' % (seg, tid, int(frame))
        if key in self.cache:
            return self.cache[key]
        with self.torch.no_grad():
            v = self.fe([img]).detach().cpu().numpy().astype(np.float32)[0]
        v = v / max(1e-9, float(np.linalg.norm(v)))
        self.cache[key] = v
        return v

    def save(self):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        keys = list(self.cache.keys())
        vecs = np.stack([self.cache[k] for k in keys])
        np.savez_compressed(CACHE, keys=np.array(keys), vecs=vecs)
        print('[cache] 保存 %d 条 → %s' % (len(keys), CACHE))


def series(cap, det, det_sorted, enc, seg, tid, lo, hi, step):
    fr, vs = [], []
    for f in range(lo, hi + 1, step):
        nf, m = nearest(det_sorted, det, f)
        if tid not in m or abs(nf - f) > step:
            continue
        img = crop(cap, f, m[tid])
        if img is None:
            continue
        fr.append(f)
        vs.append(enc.get(seg, tid, f, img))
    if len(vs) < 3:
        return None
    cs = [float(vs[i] @ vs[i + 1]) for i in range(len(vs) - 1)]
    mid = [(fr[i] + fr[i + 1]) / 2.0 for i in range(len(cs))]
    return mid, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--span', type=float, default=8.0)
    ap.add_argument('--step', type=int, default=10)
    ap.add_argument('--ctrl', type=int, default=30)
    a = ap.parse_args()
    enc = Enc()

    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                          encoding='utf-8-sig'))
             if r['new_frame'] and r['cls'] == 'usable']
    by_seg = {}
    for r in marks:
        by_seg.setdefault(r['seg'], []).append(r)

    rec = []          # (track, mid[], cos[])
    ctrl = []         # 控制窗内的 cos 列表
    for seg, rs in by_seg.items():
        det = load_det(seg)
        fs = sorted(det)
        cap = cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4'))
        for r in rs:
            tid, f = int(r['track_id'].split('#')[1]), int(float(r['new_frame']))
            got = series(cap, det, fs, enc, seg, tid,
                         int(f - a.span * FPS), int(f + a.span * FPS), a.step)
            if got:
                rec.append((r['track_id'], r['cam'], f, got[0], got[1]))
        # 控制：未标注轨迹的随机 16s 窗
        labeled = {int(r['track_id'].split('#')[1]) for r in rs}
        tids = [t for t in {t for m in det.values() for t in m} if t not in labeled]
        rng = np.random.RandomState(1)
        pool = rng.permutation(len(tids))[:a.ctrl]
        for idx in pool:
            tid = tids[idx]
            obs = [x for x in fs if tid in det[x]]
            if len(obs) < 60:
                continue
            c = obs[len(obs) // 2]
            got = series(cap, det, fs, enc, seg, tid,
                         int(c - 8 * FPS), int(c + 8 * FPS), a.step)
            if got:
                ctrl.extend(got[1])
        cap.release()
    enc.save()

    json.dump([dict(track=t, cam=c, frame=f, mid=[round(m, 1) for m in mid],
                    cos=[round(x, 3) for x in cs]) for t, c, f, mid, cs in rec],
              open(os.path.join(DELIV, 'results/appearance_series.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    print('\n可评估轨迹 %d 条；控制窗余弦样本 %d 个' % (len(rec), len(ctrl)))
    base = np.array(ctrl)
    rows = []
    print('%-6s %-10s %-14s %-12s %s' % ('τ', '召回', '切点外事件/轨', '事件数/轨', '控制窗误切率'))
    def clusters(mid, cs, tau, merge=50):
        """把低于 τ 的跳聚成事件：相邻两跳相距 ≤merge 帧算同一个事件，取事件内最小值位置。"""
        ev = []
        for m, c in zip(mid, cs):
            if c >= tau:
                continue
            if ev and m - ev[-1][0] <= merge:
                if c < ev[-1][1]:
                    ev[-1] = [m, c]
            else:
                ev.append([m, c])
        return ev

    for tau in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        hit, extra, nt, ncuts = 0, 0, 0, 0
        for tid, cam, f, mid, cs in rec:
            nt += 1
            ev = clusters(mid, cs, tau)
            if not ev:
                continue
            best = min(ev, key=lambda e: abs(e[0] - f))
            if abs(best[0] - f) <= 25 and best[1] < tau:
                hit += 1
            extra += sum(1 for e in ev if abs(e[0] - f) > 50)
            ncuts += len(ev)
        fp_rate = float((base < tau).mean())  # 每个相邻对的误切概率
        rows.append(dict(tau=tau, recall=round(hit / max(1, nt), 3), hit=hit, n=nt,
                         extra_events_per_track=round(extra / max(1, nt), 2),
                         events_per_track=round(ncuts / max(1, nt), 2),
                         ctrl_fp_rate=round(fp_rate, 4)))
        print('%-6.2f %2d/%2d %5.0f%%   %8.2f      %8.2f      %8.3f'
              % (tau, hit, nt, 100 * hit / nt, extra / max(1, nt), ncuts / max(1, nt), fp_rate))

    with open(OUT, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\n→ %s' % OUT)


if __name__ == '__main__':
    main()
