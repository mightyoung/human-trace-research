#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器升级 per-camera 曲线诊断：320/323/318 的工作点波动是校准还是真退化？

对每个候选摄像头子集画 {single,multi4,multi8,soft3} 的 coverage-same 曲线（统一 thr 扫描），
并与 study.json 中的 per-cam 工作点（selfcal-percam）数字并排打印——
若曲线整体占优而工作点数字偏低 → 校准位置效应；若曲线整体偏低 → 真退化。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector_upgrade_study import (build_variants, score_matrix,  # noqa: E402
                                    MIN_CLUSTER)
from camgroup_study import load_segs, select_best, cam_of, TRAIN_SEGS, TEST_SEGS, JDL  # noqa: E402
from train_selector_cluster import cooc_pairs  # noqa: E402

ROOT = os.path.dirname(HERE)
TR_keys, TR_Va, TR_Vb, TR_V = load_segs(TRAIN_SEGS)
TE_keys, TE_Va, TE_Vb, TE_V = load_segs(TEST_SEGS)
tr_cam = np.array([cam_of(k[0]) for k in TR_keys])
te_cam = np.array([cam_of(k[0]) for k in TE_keys])
keys_set = set(TR_keys)
cooc_all = []
for s in TRAIN_SEGS:
    for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), keys_set, s, 5):
        cooc_all.append(((s, u), (s, v)))

_, results = select_best(TR_V, TR_Va, TR_Vb, TR_keys, cooc_all)
best = [d for d in results if d["method"] == "mutual-knn"][0]
lab = best["_lab"]
ids = sorted(set(lab))
sizes = np.array([(lab == c).sum() for c in ids])
keep_ids = [c for c, s in zip(ids, sizes) if s >= MIN_CLUSTER]
variants, C = build_variants(TR_V, lab, keep_ids)

study = json.load(open(os.path.join(ROOT, "output_crosscam", "analysis",
                                    "selector_1012", "upgrade", "study.json")))
names = ["single", "multi4", "multi8", "soft3"]

grid = np.arange(0.10, 0.95, 0.10)
for cam in ("318", "320", "323"):
    m_te = te_cam == cam
    print(f"\n===== cam {cam} (test n={int(m_te.sum())}) =====")
    for name in names:
        v = variants[name]
        sa = score_matrix(TE_Va, v, C)[m_te].max(1)
        ca = score_matrix(TE_Va, v, C)[m_te].argmax(1)
        sb = score_matrix(TE_Vb, v, C)[m_te].max(1)
        cb = score_matrix(TE_Vb, v, C)[m_te].argmax(1)
        covs, sames = [], []
        for t in np.arange(0.5, 0.9995, 0.002):
            ia = np.where(sa >= t, ca, -1)
            ib = np.where(sb >= t, cb, -1)
            kn = (ia >= 0) & (ib >= 0)
            covs.append(kn.mean())
            sames.append((ia[kn] == ib[kn]).mean() if kn.sum() else np.nan)
        covs, sames = np.asarray(covs), np.asarray(sames)
        o = np.argsort(covs)
        base_txt = ""
        wp = study["workpoints"][name]["selfcal_percam"]["per_cam"].get(cam)
        if wp:
            base_txt = (f" | selfcal WP: cov={wp['coverage']:.2f} "
                        f"same={None if wp['same_known'] is None else round(wp['same_known'],3)}")
        curve_txt = " ".join(
            f"c{tg:.1f}:{np.interp(tg, covs[o], sames[o]):.2f}" if covs[o][0] <= tg <= covs[o][-1]
            else f"c{tg:.1f}:n/a" for tg in grid)
        print(f"[{name:7s}] {curve_txt}{base_txt}")
