#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""图库侧分母核对：可评估率、串人成员数、M1 门槛可测性。"""
import csv
import collections
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable', 'data-request-v2')


def rd(p):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


tr = rd(os.path.join(DELIV, 'ledger/tracks.csv'))
gal_tr = [r for r in tr if r['split'] == 'gallery']
print('tracks.csv rows', len(tr), '| gallery', len(gal_tr))
print('evaluable', collections.Counter(r['evaluable'] for r in gal_tr))
print('exclude_reason', collections.Counter(r['exclude_reason'] for r in gal_tr if r['evaluable'] != '1'))
print('by cam evaluable=1', collections.Counter(r['cam'] for r in gal_tr if r['evaluable'] == '1'))
print('by cam all', collections.Counter(r['cam'] for r in gal_tr))

mem = rd(os.path.join(DELIV, 'gt/gt_gallery_members_FILLED.csv'))
print('\ngallery member rows', len(mem))
lab = [r for r in mem if r['person_id'].strip()]
print('labelled rows', len(lab), '| unlabelled', len(mem) - len(lab))
sw = [r for r in mem if r['id_switch'].strip() in ('1', '1.0')]
print('id_switch=1 rows', len(sw))
print('id_switch values', collections.Counter(r['id_switch'] for r in mem))
print('id_switch by cam', collections.Counter(r['cam'] for r in sw))
print('id_switch labelled?', collections.Counter(bool(r['person_id'].strip()) for r in sw))
print('labelled n_frames<6', sum(1 for r in lab if r['n_frames'].strip() and int(float(r['n_frames'])) < 6))
print('labelled rate of 579:', len(lab) / 579, '| of', len(mem), ':', len(lab) / len(mem))
# 可评估率候选
ev = sum(1 for r in gal_tr if r['evaluable'] == '1')
print('evaluable/579 =', ev / 579, '| evaluable/total =', ev / len(gal_tr))
