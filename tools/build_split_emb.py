#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 M1 的动作（split / truncate）落到图库表征上，产出可评测的新表征目录。

口径**与生产编码器一致**（`tools/encode_1012_osnet_seq.py`）：
  每半段用 np.linspace 取 8 帧 → OSNet → 逐张 L2 归一 → 均值 → 再归一。

做法：
  * 被改动的轨迹：按 cut 切成 1–2 段，每段重算 a/b（真实视频帧裁权重编码）；
  * 其余轨迹：原样复制，键名不变；
  * 子轨迹新 id = parent + 900000（前段）/ 950000（后段）；
  * 另写一份**评测用**真值（不改原 gt）：样本多的那段继承父轨迹身份，另一段留空。

用法：
  PY tools/build_split_emb.py            # 生成（约 2–4 分钟）
  PY tools/build_split_emb.py --check    # 只做保真度自检：重算未改动轨迹的 a/b 与原 npz 比
"""
import argparse
import collections
import csv
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracklet_split_v2 as S  # noqa: E402
from encode_1012_osnet_seq import OSNetEnc, crop_raw  # noqa: E402

ROOT = S.ROOT
DELIV = S.DELIV
SRC = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd_emb')
DST = os.path.join(ROOT, 'output_crosscam_local/1012_yolo8_dd_emb_split')
SEG_DIR = os.path.join(ROOT, 'media-resource/1012/segments')
GT_SRC = os.path.join(DELIV, 'gt/gt_gallery_members_FILLED.csv')
GT_DST = os.path.join(ROOT, 'tmp/m1_downstream/gt/gt_gallery_members_FILLED.csv')
MAP = os.path.join(DELIV, 'results/split_emb_map.json')
OFF_A, OFF_B = 900000, 950000
SAMPLES = 8


def rd(p):
    return list(csv.DictReader(open(p, encoding='utf-8-sig')))


def frames_of(det, tid):
    return [(f, det[f][tid]) for f in sorted(det) if tid in det[f]]


def sample_half(fl, n=SAMPLES):
    idx = np.linspace(0, len(fl) - 1, min(n, len(fl))).astype(int)
    return [fl[i] for i in idx]


def encode_parts(enc, cap, seg, parts):
    """parts: [(name, [(frame, box), ...])] → [(name, vec_a, vec_b, n_a, n_b)]"""
    import cv2
    jobs = []       # (name, which, [(frame, box)])
    for name, fl in parts:
        h = len(fl) // 2
        if h < 1 or len(fl) - h < 1:
            continue
        jobs.append((name, 'a', sample_half(fl[:h])))
        jobs.append((name, 'b', sample_half(fl[h:])))
    flat = [(name, which, f, box) for name, which, lst in jobs for f, box in lst]
    crops, keep = [], []
    for i, (name, which, f, box) in enumerate(flat):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        c = crop_raw(img, box)
        if c is None:
            continue
        crops.append(c)
        keep.append((name, which))
    if not crops:
        return []
    V = enc.encode(crops)
    acc = collections.defaultdict(list)
    for (name, which), v in zip(keep, V):
        acc[(name, which)].append(v)
    out = []
    for name, _ in parts:
        a, b = acc.get((name, 'a'), []), acc.get((name, 'b'), [])
        if not a or not b:
            continue
        va = np.mean(a, 0)
        vb = np.mean(b, 0)
        va /= max(1e-9, float(np.linalg.norm(va)))
        vb /= max(1e-9, float(np.linalg.norm(vb)))
        out.append((name, va.astype(np.float32), vb.astype(np.float32), len(a), len(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--dst', default=DST)
    ap.add_argument('--n-check', type=int, default=6, help='自检重算的轨迹条数')
    ap.add_argument('--actions', default=os.path.join(DELIV, 'results/m1_actions.csv'))
    ap.add_argument('--gt-out', default=GT_DST)
    ap.add_argument('--map-out', default=MAP)
    a = ap.parse_args()

    import cv2

    acts = rd(a.actions)
    todo = [r for r in acts if r['action'] in ('split', 'truncate') and r['cut_frame']]
    by_seg = collections.defaultdict(list)
    for r in todo:
        by_seg[r['track_id'].split('#')[0]].append(r)
    print('待处理 %d 条（split %d / truncate %d），涉及 %d 段'
          % (len(todo), sum(1 for r in todo if r['action'] == 'split'),
             sum(1 for r in todo if r['action'] == 'truncate'), len(by_seg)))

    enc = OSNetEnc('cpu')
    dets, caps = {}, {}

    if a.check:
        # 抽未改动的轨迹，用同一路径重算 a/b，与原 npz 比余弦
        done = 0
        for seg in ('305_train_529', '323_train_510'):
            det = dets.setdefault(seg, S.load_det(seg))
            cap = caps.setdefault(seg, cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4')))
            za = np.load(os.path.join(SRC, '%s_emb_a.npz' % seg), allow_pickle=True)
            zb = np.load(os.path.join(SRC, '%s_emb_b.npz' % seg), allow_pickle=True)
            skip = {int(r['track_id'].split('#')[1]) for r in by_seg.get(seg, [])}
            for k in list(za.keys()):
                if done >= a.n_check:
                    break
                tid = int(k.split('#')[1])
                if tid in skip:
                    continue
                fl = frames_of(det, tid)
                if len(fl) < 4:
                    continue
                got = encode_parts(enc, cap, seg, [('A', fl)])
                if not got:
                    continue
                _, va, vb, na, nb = got[0]
                print('  %-22s a=%.4f b=%.4f' % (k, float(va @ za[k]), float(vb @ zb[k])))
                done += 1
            cap.release()
        print('自检完成：上面越接近 1.000 说明重建口径越准（生产是 8 帧 linspace 均值）')
        return

    os.makedirs(a.dst, exist_ok=True)
    for f in os.listdir(SRC):
        shutil.copy2(os.path.join(SRC, f), os.path.join(a.dst, f))
    print('已复制 %d 个 npz → %s' % (len(os.listdir(SRC)), a.dst))

    mapping, dropped = [], []
    for seg, rows in sorted(by_seg.items()):
        det = dets.setdefault(seg, S.load_det(seg))
        cap = caps.setdefault(seg, cv2.VideoCapture(os.path.join(SEG_DIR, seg + '.mp4')))
        da = {k: v for k, v in np.load(os.path.join(a.dst, '%s_emb_a.npz' % seg),
                                       allow_pickle=True).items()}
        db = {k: v for k, v in np.load(os.path.join(a.dst, '%s_emb_b.npz' % seg),
                                       allow_pickle=True).items()}
        for r in rows:
            tid = int(r['track_id'].split('#')[1])
            cut = int(float(r['cut_frame']))
            fl = frames_of(det, tid)
            if len(fl) < 2:
                dropped.append(dict(track=r['track_id'], why='检测帧 <2'))
                continue
            parts = [('A', [x for x in fl if x[0] < cut]), ('B', [x for x in fl if x[0] >= cut])]
            if r['action'] == 'truncate':
                parts = parts[:1]
            parts = [(n, x) for n, x in parts if len(x) >= 2]
            got = encode_parts(enc, cap, seg, parts)
            if not got:
                dropped.append(dict(track=r['track_id'], why='两段都过短'))
                continue
            built = []
            for name, va, vb, na, nb in got:
                new_tid = tid + (OFF_A if name == 'A' else OFF_B)
                key = '%s#%d' % (seg, new_tid)
                da[key] = va
                db[key] = vb
                xs = dict(parts)[name]
                built.append(dict(key=key, tid=new_tid, part=name,
                                  f0=xs[0][0], f1=xs[-1][0], n=len(xs)))
            da.pop('%s#%d' % (seg, tid), None)
            db.pop('%s#%d' % (seg, tid), None)
            mapping.append(dict(parent=r['track_id'], seg=seg, tid=tid, action=r['action'],
                                cut=cut, parts=built,
                                keep=built[int(np.argmax([b['n'] for b in built]))]['key']))
        np.savez(os.path.join(a.dst, '%s_emb_a.npz' % seg), **da)
        np.savez(os.path.join(a.dst, '%s_emb_b.npz' % seg), **db)
        cap.release()
        print('[%s] 改组后 %d 条' % (seg, len(da)))

    # 评测用真值
    gt = rd(GT_SRC)
    cols = list(gt[0].keys())
    pid = {r['track_id']: (r['person_id'] or '').strip() for r in gt}
    parents = {m['parent'] for m in mapping}
    rows_out = [dict(r) for r in gt if r['track_id'] not in parents]
    for m in mapping:
        for b in m['parts']:
            row = {c: '' for c in cols}
            row['track_id'] = b['key']
            row['person_id'] = pid.get(m['parent'], '') if b['key'] == m['keep'] else ''
            row['seg'] = m['seg']
            row['cam'] = m['seg'].split('_')[0]
            row['note'] = 'M1 %s from %s' % (m['action'], m['parent'])
            row['id_switch'] = '0'
            rows_out.append(row)
    os.makedirs(os.path.dirname(a.gt_out), exist_ok=True)
    with open(a.gt_out, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_out:
            w.writerow({c: r.get(c, '') for c in cols})

    json.dump(dict(n_parent=len(mapping), n_sub=sum(len(m['parts']) for m in mapping),
                   dropped=dropped, items=mapping),
              open(a.map_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n下架父轨迹 %d，新增子轨迹 %d，丢弃 %d'
          % (len(mapping), sum(len(m['parts']) for m in mapping), len(dropped)))
    for d in dropped:
        print('  丢弃 %s：%s' % (d['track'], d['why']))
    print('→ %s' % a.dst)
    print('→ %s（评测用真值）' % a.gt_out)
    print('→ %s' % a.map_out)


if __name__ == '__main__':
    main()
