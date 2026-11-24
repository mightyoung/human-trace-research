#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器升级补充实验：N-ULC "一致性权重" 的直接转译 —— margin 拒识。

multi8 判定基础上加"top1-top2 簇分差 ≥ δ"条件（confidence margin），
扫描 δ 观察 coverage / same-known / coll-known 的权衡位移。
若相同 coverage 下 same 提升 → 一致性权重的收益成立。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector_upgrade_study import build_variants, score_matrix, MIN_CLUSTER  # noqa: E402
from camgroup_study import load_segs, select_best, cam_of, CAMS, TRAIN_SEGS, TEST_SEGS, JDL  # noqa: E402
from train_selector_cluster import cooc_pairs  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012", "upgrade")

TR_keys, TR_Va, TR_Vb, TR_V = load_segs(TRAIN_SEGS)
TE_keys, TE_Va, TE_Vb, TE_V = load_segs(TEST_SEGS)
tr_cam = np.array([cam_of(k[0]) for k in TR_keys])
te_cam = np.array([cam_of(k[0]) for k in TE_keys])
keys_set = set(TR_keys)
cooc_all = []
for s in TRAIN_SEGS:
    for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), keys_set, s, 5):
        cooc_all.append(((s, u), (s, v)))
kidx = {k: i for i, k in enumerate(TR_keys)}

_, results = select_best(TR_V, TR_Va, TR_Vb, TR_keys, cooc_all)
best = [d for d in results if d["method"] == "mutual-knn"][0]
lab = best["_lab"]
ids = sorted(set(lab))
sizes = np.array([(lab == c).sum() for c in ids])
keep_ids = [c for c, s in zip(ids, sizes) if s >= MIN_CLUSTER]
variants, C = build_variants(TR_V, lab, keep_ids)

# selfcal-percam thr（从升级实验逻辑复用：直接读 study.json 的数字保证一致）
study = json.load(open(os.path.join(OUT, "study.json")))
thr_pc = {c: study["thr_selfcal"]["multi8"]["per_cam"][c]["thr"] for c in CAMS}
print("[thr] per-cam selfcal(multi8):", {k: round(v, 4) for k, v in thr_pc.items()})


def topk_margin(sc):
    srt = np.sort(sc, axis=1)
    return srt[:, -1], srt[:, -1] - srt[:, -2]


mv = variants["multi8"]
sc_a = score_matrix(TE_Va, mv, C)
sc_b = score_matrix(TE_Vb, mv, C)
s1a, mga = topk_margin(sc_a)
s1b, mgb = topk_margin(sc_b)
ca, cb = sc_a.argmax(1), sc_b.argmax(1)
thr = np.array([thr_pc[c] for c in te_cam])
rng = np.random.default_rng(3)
N = len(TE_keys)
L = rng.integers(0, N, 30000)
R = rng.integers(0, N, 30000)
kp = L != R

report = {}
print(f"\n{'delta':>7s} | {'cov':>6s} {'same':>6s} {'coll':>6s} | {'n_known':>7s}")
for delta in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
    ia = np.where((s1a >= thr) & (mga >= delta), ca, -1)
    ib = np.where((s1b >= thr) & (mgb >= delta), cb, -1)
    kn = (ia >= 0) & (ib >= 0)
    cov = float(kn.mean())
    same = float((ia[kn] == ib[kn]).mean()) if kn.sum() else None
    il, ir = ia[L[kp]], ib[R[kp]]
    kn2 = (il >= 0) & (ir >= 0)
    coll = float((il[kn2] == ir[kn2]).mean()) if kn2.sum() else None
    report[str(delta)] = {"coverage": cov, "same_known": same, "coll_known": coll,
                          "n_known": int(kn.sum())}
    print(f"{delta:7.3f} | {cov:6.3f} {same:6.3f} {coll:6.3f} | {int(kn.sum()):7d}")

json.dump(report, open(os.path.join(OUT, "margin_study.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"[done] -> margin_study.json")
