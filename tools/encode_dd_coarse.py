#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B7 第一阶段：给所有轨迹编码 8 个样本（粗筛用）。

为什么要单独一遍：现有 `1012_osnet_emb` 只有整轨均值，`1012_ab` 的逐样本嵌入又是**另一套检测**
（tid ≤ 656，与本系统 dd 的 tid 不通用）。所以必须用 dd 的框从原视频重新裁、重新编。

产物：`output_crosscam_local/emb_cache/dd_coarse_8.npz`
  keys  'seg|tid|frame'
  vecs  (N, 512) L2 归一化
  meta  (N, 3) = [track_id, frame, box_h]
支持断点续跑：文件已存在则只补新增 key。

用法：
  PY tools/encode_dd_coarse.py                # 全部 8 段
  PY tools/encode_dd_coarse.py --segs 323_train_510,305_train_529
"""
import argparse
import collections
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd')
SEG_DIR = os.path.join(ROOT, 'media-resource/1012/segments')
CACHE = os.path.join(ROOT, 'output_crosscam_local/emb_cache/dd_coarse_8.npz')
W = ('models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_'
     'stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
N_SAMPLES = 8
MIN_SAMPLES = 4


def load_det(seg):
    d = {}
    p = os.path.join(DET, seg, '%s_detections.jsonl' % seg)
    if not os.path.exists(p):
        return None
    for line in open(p, encoding='utf-8'):
        r = json.loads(line)
        d[r['frame']] = {int(t): b for t, b in zip(r['track_ids'], r['boxes'])}
    return d


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', default='')
    ap.add_argument('--samples', type=int, default=N_SAMPLES)
    a = ap.parse_args()
    segs = ([s.strip() for s in a.segs.split(',') if s.strip()] if a.segs
            else sorted(os.listdir(DET)))

    cache = {}
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        cache = {str(k): v for k, v in zip(z['keys'], z['vecs'])}
        print('[resume] 已有 %d 条向量' % len(cache))

    from torchreid.reid.utils import FeatureExtractor
    import torch
    torch.set_flush_denormal(True)
    fe = FeatureExtractor(model_name='osnet_x1_0', model_path=os.path.join(ROOT, W),
                          device='cpu', verbose=False)
    print('[osnet] cpu', flush=True)

    meta = []
    for seg in segs:
        det = load_det(seg)
        if det is None:
            print('[skip] %s 无检测' % seg)
            continue
        per_tid = collections.defaultdict(list)
        for f, m in det.items():
            for t, b in m.items():
                per_tid[t].append(f)
        cap = cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4'))
        n_new = 0
        for tid, fs in sorted(per_tid.items()):
            fs.sort()
            if len(fs) < MIN_SAMPLES * 2:
                continue
            idx = np.linspace(0, len(fs) - 1, a.samples).astype(int)
            pick = [fs[i] for i in idx]
            imgs, keep = [], []
            for f in pick:
                key = '%s|%d|%d' % (seg, tid, f)
                if key in cache:
                    continue
                im = crop(cap, f, det[f][tid])
                if im is not None:
                    imgs.append(im)
                    keep.append((key, f, det[f][tid][3] - det[f][tid][1]))
            if imgs:
                with torch.no_grad():
                    v = fe(imgs).detach().cpu().numpy().astype(np.float32)
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
                for (key, f, bh), vec in zip(keep, v):
                    cache[key] = vec
                    meta.append((tid, f, bh))
                    n_new += 1
        cap.release()
        print('[%s] 新增 %d 条（累计 %d 轨）' % (seg, n_new, len(per_tid)), flush=True)
        keys = list(cache.keys())
        vecs = np.stack([cache[k] for k in keys])
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, keys=np.array(keys), vecs=vecs)
        print('    saved %d → %s' % (len(keys), CACHE), flush=True)

    keys = list(cache.keys())
    vecs = np.stack([cache[k] for k in keys])
    np.savez_compressed(CACHE, keys=np.array(keys), vecs=vecs)
    print('完成：%d 条向量 → %s' % (len(keys), CACHE))


if __name__ == '__main__':
    main()
