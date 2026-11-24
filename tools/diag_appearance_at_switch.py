#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-B3：换人点前后「外观余弦」实测——回答「遮挡了 ReID 不就没了？」。

三条对照同框同机位，都用**本系统 dd 检测框**从原视频现裁现编（OSNet x1_0，MSMT17 权重）：

  switch   同一 tid 在换人点前 0.4s 与后 0.4s 的裁剪
  same     同一 tid 在换人点前 10s 与 9.6s 的裁剪（同一人，正常行走）
  other    换人点后 0.4s，该 tid 的框 vs 同帧 IoU 最大的另一个 tid 的框（不同人）

判读：
  cos(switch) ≈ cos(other)  →  外观有完整区分力，切分可行
  cos(switch) ≈ cos(same)   →  外观也看不见换人（远景/同款工装），任何通道都做不到
另外记录面积比与贴边情况，用来检验「遮挡本身会不会让余弦掉」。

用法：PY tools/diag_appearance_at_switch.py
输出：results/appearance_at_switch.csv + 终端汇总
"""
import csv
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
DET = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd')
SEG_DIR = os.path.join(ROOT, 'media-resource/1012/segments')
OUT = os.path.join(DELIV, 'results/appearance_at_switch.csv')
W = ('models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_'
     'stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
D = 10          # ±0.4s
CTRL = 250      # 对照对间隔 10s


def load_det(seg):
    d = {}
    for line in open(os.path.join(DET, seg, '%s_detections.jsonl' % seg), encoding='utf-8'):
        r = json.loads(line)
        d[r['frame']] = {int(t): b for t, b in zip(r['track_ids'], r['boxes'])}
    return d


def nearest(det, frame):
    import bisect
    fs = sorted(det)
    i = bisect.bisect_left(fs, frame)
    c = [j for j in (i - 1, i) if 0 <= j < len(fs)]
    j = min(c, key=lambda k: abs(fs[k] - frame))
    return fs[j], det[fs[j]]


def crop(cap, frame, box, margin=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    if not ok:
        return None, 0
    H, Wd = img.shape[:2]
    clip = (x1 < 0 or y1 < 0 or x2 > Wd or y2 > H)
    x1, x2 = max(0, int(x1)), min(Wd, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None, int(clip)
    return img[y1:y2, x1:x2].copy(), int(clip)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    from torchreid.reid.utils import FeatureExtractor
    import torch
    torch.set_flush_denormal(True)
    fe = FeatureExtractor(model_name='osnet_x1_0', model_path=os.path.join(ROOT, W),
                          device='cpu', verbose=False)

    def enc(crops):
        crops = [c for c in crops if c is not None]
        if not crops:
            return None
        with torch.no_grad():
            v = fe(crops).detach().cpu().numpy().astype(np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        return v

    marks = [r for r in csv.DictReader(open(os.path.join(DELIV, 'results/m1a_switch_review.csv'),
                                          encoding='utf-8-sig')) if r['new_frame']]
    by_seg = {}
    for r in marks:
        by_seg.setdefault(r['seg'], []).append(r)

    rows = []
    for seg, rs in by_seg.items():
        det = load_det(seg)
        cap = cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4'))
        for r in rs:
            tid, f = int(r['track_id'].split('#')[1]), int(float(r['new_frame']))
            fa, ma = nearest(det, f - D)
            fb, mb = nearest(det, f + D)
            if tid not in ma or tid not in mb:
                rows.append(dict(track_id=r['track_id'], cls=r['cls'], status='no_crop'))
                continue
            ca, clipa = crop(cap, fa, ma[tid])
            cb, clipb = crop(cap, fb, mb[tid])
            # 对照：同一人（换人点前 10s 与 9.6s）
            fc, mc = nearest(det, f - CTRL)
            fd, md = nearest(det, f - CTRL + D)
            cc = crop(cap, fc, mc[tid])[0] if tid in mc else None
            cd = crop(cap, fd, md[tid])[0] if tid in md else None
            # 对照：同帧另一个人（IoU 最大者）
            other_box, other_tid, other_iou = None, None, 0.0
            for t2, b2 in mb.items():
                if t2 == tid:
                    continue
                v = iou(mb[tid], b2)
                if v > other_iou:
                    other_box, other_tid, other_iou = b2, t2, v
            co = crop(cap, fb, other_box)[0] if other_box else None
            if any(x is None for x in (ca, cb)):
                rows.append(dict(track_id=r['track_id'], cls=r['cls'], status='no_crop'))
                continue
            V = enc([ca, cb, cc, cd, co])
            k = 0
            ea, eb = V[k], V[k + 1]
            k += 2
            ec = V[k] if cc is not None else None
            k += 1 if cc is not None else 0
            ed = V[k] if cd is not None else None
            k += 1 if cd is not None else 0
            eo = V[k] if co is not None else None
            cos = lambda u, v: float(u @ v) if (u is not None and v is not None) else None
            area_a = (ma[tid][2] - ma[tid][0]) * (ma[tid][3] - ma[tid][1])
            area_b = (mb[tid][2] - mb[tid][0]) * (mb[tid][3] - mb[tid][1])
            rows.append(dict(
                track_id=r['track_id'], cls=r['cls'], cam=r['cam'], pos=r['pos_ratio'],
                status='ok',
                cos_switch=round(cos(ea, eb), 3),
                cos_same=round(cos(ec, ed), 3) if (ec is not None and ed is not None) else '',
                cos_other=round(cos(eb, eo), 3) if eo is not None else '',
                other_tid=other_tid if other_tid else '',
                other_iou=round(other_iou, 2),
                area_ratio=round(area_b / max(1e-6, area_a), 2),
                clip_edge=int(bool(clipa or clipb)),
                box_h=round(mb[tid][3] - mb[tid][1], 1)))
        cap.release()

    ok = [x for x in rows if x['status'] == 'ok']
    with open(OUT, 'w', encoding='utf-8-sig', newline='') as fh:
        keys = ['track_id', 'cls', 'cam', 'pos', 'status', 'cos_switch', 'cos_same', 'cos_other',
                'other_tid', 'other_iou', 'area_ratio', 'clip_edge', 'box_h']
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for x in rows:
            w.writerow({k: x.get(k, '') for k in keys})

    def med(v):
        v = [x for x in v if x != '' and x is not None]
        return float(np.median(v)) if v else float('nan')

    sw = [x['cos_switch'] for x in ok]
    sa = [x['cos_same'] for x in ok]
    ot = [x['cos_other'] for x in ok]
    print('可测 %d / %d 条' % (len(ok), len(rows)))
    print('  换人点前后余弦  中位 %.3f（p10 %.3f p90 %.3f）'
          % (med(sw), np.percentile([v for v in sw if v != ''], 10),
             np.percentile([v for v in sw if v != ''], 90)))
    print('  同人对照(隔10s) 中位 %.3f' % med(sa))
    print('  异人对照(同帧)  中位 %.3f' % med(ot))
    print('\n按类别：')
    for grp in ('usable', 'tail', 'head'):
        sub = [x for x in ok if x['cls'] == grp]
        if sub:
            print('  %-8s n=%2d  换人点 %.3f｜同人 %.3f｜异人 %.3f'
                  % (grp, len(sub), med([x['cos_switch'] for x in sub]),
                     med([x['cos_same'] for x in sub]), med([x['cos_other'] for x in sub])))
    print('\n面积比 >1.5 或贴边的（可能被遮挡/进出画）：%d 条，其中余弦中位 %.3f'
          % (sum(1 for x in ok if x['area_ratio'] > 1.5 or x['clip_edge']),
             med([x['cos_switch'] for x in ok if x['area_ratio'] > 1.5 or x['clip_edge']])))
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
