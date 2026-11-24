#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E4：轨迹聚合方式对照（质量加权 / 多样性筛选），在 E2 判定管线上评估。

背景：现状是"每轨 8 帧朴素平均"。Plumerai 工程实践指出两个问题——
  ① 低质裁剪污染聚合表征；② 连续帧高度相关，简单平均不降方差、只放大当时的系统性偏差。
对应两种修正：
  Q  质量加权：w = conf × sqrt(面积/中位面积)，按 w 加权平均
  D  多样性筛选：按质量降序贪心选，与已选者余弦 < tau 才加入（至少保留 3 张）

变体：
  A0 uniform      现状（等权平均）
  A1 quality      质量加权
  A2 diversity    多样性筛选 + 等权
  A3 quality+div  多样性筛选 + 质量加权

口径：与 E2 一致（会话对齐 + 每身份 z 校准 + self-cal 定 θ，稳定率≥0.90）。
簇结构固定为 A0 的结果，只改"原型如何由成员估计"与"查询如何聚合"，保证归因清晰。

用法：python tools/e4_aggregation.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
CROP_DIR = os.path.join(ROOT, "output_crosscam_local/1012_osnet_crops")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/e4_aggregation.json")
TAU = 0.95
MIN_KEEP = 3


def load_crops(segs):
    """返回 {(seg,tid): {'a': (emb, q), 'b': (emb, q)}}"""
    data = {}
    miss = []
    for s in segs:
        p = os.path.join(CROP_DIR, f"{s}_crops.npz")
        if not os.path.exists(p):
            miss.append(s)
            continue
        d = np.load(p)
        eids, emb, q = d["eids"], d["emb"], d["q"]
        buckets = defaultdict(lambda: {"emb": [], "q": []})
        for i, eid in enumerate(eids):
            ab, tid, _k = str(eid).split("#")
            b = buckets[(ab, int(tid))]
            b["emb"].append(emb[i])
            b["q"].append(q[i])
        for (ab, tid), b in buckets.items():
            data[(s, tid, ab)] = (np.asarray(b["emb"], np.float32),
                                  np.asarray(b["q"], np.float32).reshape(-1, 3))
    return data, miss


def agg_uniform(emb, q):
    m = emb.mean(0)
    return m / (np.linalg.norm(m) + 1e-9)


def quality_weights(q):
    conf = q[:, 0]
    area = np.maximum(q[:, 1], 1e-8)
    med = float(np.median(area)) if len(area) else 1.0
    w = np.clip(conf, 0.05, 1.0) * np.sqrt(area / max(med, 1e-8))
    return np.clip(w, 1e-3, None)


def agg_quality(emb, q):
    w = quality_weights(q)
    m = (emb * w[:, None]).sum(0) / w.sum()
    return m / (np.linalg.norm(m) + 1e-9)


def diversity_pick(emb, q, tau=TAU, min_keep=MIN_KEEP):
    """按质量降序贪心选，与已选者余弦 < tau 才加入。返回索引数组。"""
    order = np.argsort(-quality_weights(q))
    keep = []
    for i in order:
        v = emb[i]
        if not keep or max(float(v @ emb[j]) for j in keep) < tau:
            keep.append(int(i))
    if len(keep) < min_keep:
        for i in order:
            if int(i) not in keep:
                keep.append(int(i))
            if len(keep) >= min(min_keep, len(order)):
                break
    return np.asarray(keep, int)


def agg_div(emb, q):
    idx = diversity_pick(emb, q)
    return agg_uniform(emb[idx], q[idx])


def agg_div_q(emb, q):
    idx = diversity_pick(emb, q)
    return agg_quality(emb[idx], q[idx])


AGG = {"A0 uniform": agg_uniform, "A1 quality": agg_quality,
       "A2 diversity": agg_div, "A3 quality+div": agg_div_q}


def build_pair(crops, segs, fn):
    """返回 (keys, Va, Vb)，keys 为 a/b 两半都存在的轨迹（与 load_pair 口径一致）。"""
    ma, mb = {}, {}
    for s in segs:
        tids = sorted({t for (ss, t, a) in crops if ss == s})
        for t in tids:
            if (s, t, "a") in crops:
                emb, q = crops[(s, t, "a")]
                if len(emb):
                    ma[(s, t)] = fn(emb, q)
            if (s, t, "b") in crops:
                emb, q = crops[(s, t, "b")]
                if len(emb):
                    mb[(s, t)] = fn(emb, q)
    keys = sorted(set(ma) & set(mb))
    Va = np.asarray([ma[k] for k in keys], np.float32)
    Vb = np.asarray([mb[k] for k in keys], np.float32)
    return keys, Va, Vb


def main():
    crops, miss = load_crops(S.TRAIN_SEGS + S.TEST_SEGS)
    if miss:
        raise SystemExit(f"缺逐裁特征：{miss}\n先跑 tools/run_1012_crops_all.sh")
    print(f"[load] {len(crops)} (seg,tid,half) 组逐裁特征")

    # 固定簇结构：用 A0 的 train 向量
    ktr, Vtr_a0, Vtr_b0 = build_pair(crops, S.TRAIN_SEGS, agg_uniform)
    Vtr0 = _norm((Vtr_a0 + Vtr_b0) / 2.0)
    lab, kept = S.hier_labels(Vtr0)
    K = len(kept)
    print(f"[cluster] clusters kept {K} (固定自 A0)")

    kte, _, _ = build_pair(crops, S.TEST_SEGS, agg_uniform)
    cooc_te = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte), s, 5):
            cooc_te.append(((s, u), (s, v)))

    out = {"n_clusters": K, "tau": TAU, "train_tracks": len(ktr), "test_tracks": len(kte),
           "variants": {}}
    print(f"\n{'variant':16s} {'protos':>6s} {'cov':>6s} {'same':>6s} {'fm':>7s} {'thr':>6s}")
    for name, fn in AGG.items():
        ka, Va_tr, Vb_tr = build_pair(crops, S.TRAIN_SEGS, fn)
        assert ka == ktr
        Vtr = _norm((Va_tr + Vb_tr) / 2.0)
        mu_g = Vtr.mean(0)

        # 原型：固定簇标签，成员均值（用该变体的 V）
        protos, pcl = [], []
        for ci, c in enumerate(kept):
            m = lab == c
            if m.sum() == 0:
                continue
            protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
            pcl.append(ci)
        protos = _norm(np.asarray(protos, np.float32) - mu_g)
        pcl = np.asarray(pcl)

        S_tr = S.score_rows(Vtr, protos, pcl, K, mu=mu_g)
        m_hat, s_hat, _ = S.fit_ident_stats(S_tr, lab, kept)
        calib = (m_hat, s_hat)

        kt2, Va_te, Vb_te = build_pair(crops, S.TEST_SEGS, fn)
        assert kt2 == kte
        mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
        Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=calib)
        Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=calib)
        grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
        rows = S.sweep(Sa, Sb, kte, cooc_te, grid)
        ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90
              and r["evaluable"] >= 100]
        best = max(ok, key=lambda r: r["coverage"]) if ok else None
        out["variants"][name] = {"n_protos": int(len(protos)), "self_cal": best, "sweep": rows}
        b = best or {}
        print(f"{name:16s} {len(protos):6d} {b.get('coverage', float('nan')):6.3f} "
              f"{b.get('same', float('nan')):6.3f} {b.get('false_merge', float('nan'))*100:6.2f}% "
              f"{b.get('thr', float('nan')):6.2f}")

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
