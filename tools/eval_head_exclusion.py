#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""剔除「头框轨迹」（同一个人被 YOLO 检成头部 + 全身两条轨迹）后重算指标。

不重跑检测：头框轨迹用几何判据直接从现有 detections.jsonl 识别
（某轨迹 ≥70% 的出现帧被另一条更大框包含 ≥80%），剔除只需换 keys。

用法：
    python tools/eval_head_exclusion.py                                      # 8 段全量（生产口径）
    python tools/eval_head_exclusion.py --segs 305_train_529,305_test_544    # 单对段
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/head_exclusion.json")


def head_tracks(seg, ratio=0.70, contain=0.80):
    p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
    if not os.path.exists(p):
        return set(), dict(n=0, head=0, obs_p50=0, pct=0.0)
    fr = [json.loads(l) for l in open(p, encoding="utf-8")]
    total = defaultdict(int)
    for o in fr:
        for t in o["track_ids"]:
            total[t] += 1
    nested = defaultdict(int)
    for o in fr:
        ts = list(zip(o["track_ids"], o["boxes"]))
        for ti, bi in ts:
            hb, wb = bi[3] - bi[1], bi[2] - bi[0]
            if hb <= 0 or wb <= 0:
                continue
            for tj, bj in ts:
                if ti == tj:
                    continue
                if bj[3] - bj[1] <= hb * 1.2:
                    continue
                ix1, iy1 = max(bi[0], bj[0]), max(bi[1], bj[1])
                ix2, iy2 = min(bi[2], bj[2]), min(bi[3], bj[3])
                if max(0, ix2 - ix1) * max(0, iy2 - iy1) / (hb * wb) >= contain:
                    nested[ti] += 1
                    break
    keep = {t for t in total if total[t] >= 20 and nested[t] / total[t] >= ratio}
    return ({(seg, t) for t in keep},
            dict(n=len(total), head=len(keep),
                 obs_p50=float(np.median([total[t] for t in keep])) if keep else 0,
                 pct=round(100 * len(keep) / max(len(total), 1), 1)))


def evaluate(segs_tr, segs_te, exclude=frozenset(), min_eval=100, same_min=0.90):
    ktr, Va_tr, Vb_tr = M.load_split(EMB, segs_tr)
    kte, Va_te, Vb_te = M.load_split(EMB, segs_te)
    if exclude:
        keepi = [i for i, k in enumerate(kte) if k not in exclude]
        kte = [kte[i] for i in keepi]
        Va_te, Vb_te = Va_te[keepi], Vb_te[keepi]
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        if m.sum() == 0:
            continue
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    keys_ok = set(kte)
    cooc = []
    for seg in segs_te:
        for a, b in cooc_pairs(os.path.join(DET, seg, f"{seg}_detections.jsonl"), keys_ok, seg, 5):
            cooc.append(((seg, a), (seg, b)))
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    rows = S.sweep(Sa, Sb, kte, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= same_min and r["evaluable"] >= min_eval]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    return dict(n_train=len(ktr), K=K, n_test=len(kte), n_cooc=len(cooc),
                self_cal=best, rows=rows)


def fmt(d):
    b = d["self_cal"]
    if not b:
        return "no feasible point (n_test=%d, K=%d, n_cooc=%d)" % (d["n_test"], d["K"], d["n_cooc"])
    fm = "%.3f%%" % (100 * b["false_merge"]) if np.isfinite(b["false_merge"]) else "n/a"
    return ("thr=%.2f cov=%.4f same=%.4f fm=%s evaluable=%d | n_test=%d K=%d"
            % (b["thr"], b["coverage"], b["same"], fm, b["evaluable"], d["n_test"], d["K"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", default="", help="逗号分隔；留空=8 段全量")
    ap.add_argument("--min-eval", type=int, default=100)
    a = ap.parse_args()
    want = [s for s in a.segs.split(",") if s] or (S.TRAIN_SEGS + S.TEST_SEGS)
    segs_tr = [s for s in want if s in S.TRAIN_SEGS]
    segs_te = [s for s in want if s in S.TEST_SEGS]

    head, stats = set(), {}
    for seg in segs_te:
        h, st = head_tracks(seg)
        stats[seg] = st
        head |= h
    print("头框轨迹:", json.dumps(stats, ensure_ascii=False))

    base = evaluate(segs_tr, segs_te, min_eval=a.min_eval)
    excl = evaluate(segs_tr, segs_te, exclude=head, min_eval=a.min_eval)
    tag = "" if not a.segs else "_2seg"
    res = {"segs": want, "min_eval": a.min_eval, "head_stats": stats,
           "excluded": sorted([list(x) for x in head]),
           "baseline": {k: v for k, v in base.items() if k != "rows"},
           "excluded_run": {k: v for k, v in excl.items() if k != "rows"}}
    with open(OUT.replace(".json", tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n基线   :", fmt(base))
    print("剔除头框:", fmt(excl))
    print("[out]", OUT.replace(".json", tag + ".json"))


if __name__ == "__main__":
    main()
