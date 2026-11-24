#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图库内「同人特征」的异质性测量：同机位 vs 跨机位。

问题：把「稳定轨迹的最新 ReID 特征」持续放进临时库，能不能提高检索？
     这取决于新增样本是否带来新信息。若同人同机位样本之间余弦已接近 1，
     加帧只降低方差、不移动质心，属零信息增量；只有跨机位/跨视角才移动质心。

做法：只用图库已标成员（257 条，train 段），按身份分组算两两余弦，
     分「同机位对」与「跨机位对」，各报中位/四分位与占比。
读：同机位中位 ≫ 跨机位中位 → 「同质增加零信息」成立，临时库应只收异质样本。

用法：PY tools/diag_gallery_homogeneity.py
输出：deliverable/data-request-v2/results/diag_gallery_homogeneity.json
"""
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S                 # noqa: E402
import eval_mask_variants as M                   # noqa: E402
from selector import _norm                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable', 'data-request-v2')


def pctl(xs):
    if not xs:
        return dict(n=0)
    a = np.asarray(xs)
    return dict(n=int(a.size), med=float(np.median(a)),
                p10=float(np.percentile(a, 10)), p25=float(np.percentile(a, 25)),
                p75=float(np.percentile(a, 75)), p90=float(np.percentile(a, 90)),
                mean=float(a.mean()))


def main():
    ap_emb = os.path.join(ROOT, 'output_crosscam_local', '1012_yolo8_dd_emb')
    gal = {}
    NONPERSON = ('shadow1', 'shadow2', 'shadow3', 'no-human')   # 非人标签，勿混入人物特征统计
    for r in csv.DictReader(open(os.path.join(DELIV, 'gt/gt_gallery_members_FILLED.csv'),
                                 encoding='utf-8-sig')):
        tid, pid = (r['track_id'] or '').strip(), (r['person_id'] or '').strip()
        if not tid or not pid or pid in NONPERSON:
            continue
        seg, _, t = tid.rpartition('#')
        gal[(seg, int(t))] = pid

    ktr, Va, Vb = M.load_split(ap_emb, S.TRAIN_SEGS)
    V = _norm((Va + Vb) / 2.0)
    keys = [k for k in ktr]

    # 按身份聚合（只保留已标注成员）
    byid = collections.defaultdict(list)
    for i, k in enumerate(keys):
        pid = gal.get(k)
        if pid:
            byid[pid].append(i)

    n_lab = sum(len(v) for v in byid.values())
    same_cam, cross_cam = [], []
    per_id = []
    for pid, idx in byid.items():
        if len(idx) < 2:
            continue
        X = V[idx]
        cams = [keys[i][0].split('_')[0] for i in idx]
        Cn = X @ X.T
        s, c = [], []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                v = float(Cn[a, b])
                (s if cams[a] == cams[b] else c).append(v)
        same_cam += s
        cross_cam += c
        per_id.append(dict(pid=pid, n=len(idx), n_cams=len(set(cams)),
                           same_med=float(np.median(s)) if s else None,
                           cross_med=float(np.median(c)) if c else None,
                           gap=(float(np.median(s) - np.median(c)) if s and c else None)))

    out = dict(emb_dir=os.path.relpath(ap_emb, ROOT), n_gallery_labelled=n_lab,
               n_ids=len(byid),
               ids_ge2=len(per_id),
               same_camera=pctl(same_cam), cross_camera=pctl(cross_cam),
               per_id=sorted(per_id, key=lambda d: -(d['gap'] if d['gap'] is not None else -9)))
    print('已标图库成员 %d / 身份 %d / 可配对身份 %d' % (n_lab, len(byid), len(per_id)))
    for k in ('same_camera', 'cross_camera'):
        v = out[k]
        print('  %-13s n=%-6d 中位 %.4f | p10 %.4f p25 %.4f p75 %.4f p90 %.4f' % (
            k, v['n'], v['med'], v['p10'], v['p25'], v['p75'], v['p90']))
    if same_cam and cross_cam:
        print('  差距：中位 %.4f（同机位 − 跨机位）' % (np.median(same_cam) - np.median(cross_cam)))
        print('  同机位对中 >0.95 的占比 %.1f%%｜跨机位对中 >0.95 的占比 %.1f%%' % (
            100 * np.mean(np.array(same_cam) > 0.95), 100 * np.mean(np.array(cross_cam) > 0.95)))
        out['share_gt095'] = dict(same_camera=float(np.mean(np.array(same_cam) > 0.95)),
                                  cross_camera=float(np.mean(np.array(cross_cam) > 0.95)))
    p = os.path.join(DELIV, 'results/diag_gallery_homogeneity.json')
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('written:', p)


if __name__ == '__main__':
    main()
