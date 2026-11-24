#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1-NN 的「簇大小门」变体：只在成员数 ≥ K_min 的簇上启用多原型（取 max），
小簇回退到簇均值（等价于生产 scorer）。

动机
----
18 条人工目检（`knn_hit_audit/index.csv`）：8 对 / 9 错 / 1 看不清 = 44.4%，整体否证。
但错误几乎全部集中在 size≤3 的小簇：

    簇 size 2-3 : 对 2 / 错 8
    簇 size 4-6 : 对 1 / 错 1
    簇 size 7-10: 对 2 / 错 0
    簇 size 11+ : 对 3 / 错 0

→ 假设：max 只在「有足够成员支撑的簇」上才可靠；2-3 条轨迹的簇只是凑巧相似的一对。

与「删掉小簇」的区别（后者已证否，fm 3.65%）：**图库不变**，只是小簇的打分方式退回均值，
候选簇集合与之前完全一致，不会把 query 甩到别的簇去。

用法：python tools/eval_knn_size_gate.py
输出：selector_osnet_1012/knn_size_gate.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
import cluster_camera_aware as CA  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/knn_size_gate.json")
GRID = np.round(np.arange(-2.0, 8.01, 0.05), 3)


def build_gated(V, lab, kept, min_size, mu_g):
    """size>=min_size 的簇用成员级原型（max），其余用簇均值。"""
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        idx = np.where(lab == c)[0]
        if idx.size == 0:
            continue
        if idx.size >= min_size:
            for i in idx:
                protos.append(V[i])
                pcl.append(ci)
        else:
            protos.append(_norm(V[idx].mean(0, keepdims=True))[0])
            pcl.append(ci)
    P = _norm(np.asarray(protos, np.float32) - mu_g)
    return P, np.asarray(pcl)


def build_knn(V, lab, kept, mu_g):
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        for i in np.where(lab == c)[0]:
            protos.append(V[i])
            pcl.append(ci)
    return _norm(np.asarray(protos, np.float32) - mu_g), np.asarray(pcl)


def build_mean(V, lab, kept, mu_g):
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        idx = np.where(lab == c)[0]
        if idx.size:
            protos.append(_norm(V[idx].mean(0, keepdims=True))[0])
            pcl.append(ci)
    return _norm(np.asarray(protos, np.float32) - mu_g), np.asarray(pcl)


def evaluate(P, pc, K, Vtr, lab, kept, Qa, Qb, kte, cooc):
    mu_g = Vtr.mean(0)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, P, pc, K, mu=mu_g), lab, kept)
    mu_q = _norm((Qa + Qb) / 2.0).mean(0)
    Sa = S.score_rows(Qa, P, pc, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Qb, P, pc, K, mu=mu_q, calib=(m_hat, s_hat))
    rows = S.sweep(Sa, Sb, kte, cooc, GRID)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    return best, Sa, Sb


def main():
    ktr, Va, Vb = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Qa, Qb = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va + Vb) / 2.0)
    cams = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams, 0.15, 0.25, S.MIN_CLUSTER)
    K = len(kept)
    mu_g = Vtr.mean(0)

    keys_ok = set(kte)
    dirty = []
    for seg in S.TEST_SEGS:
        for a, b in cooc_pairs(os.path.join(DET, seg, f"{seg}_detections.jsonl"), keys_ok, seg, 5):
            dirty.append(((seg, a), (seg, b)))
    # 干净负对（剔除头框↔宿主）
    from render_head_track_audit import find_head_tracks
    te_head = {}
    for seg in S.TEST_SEGS:
        for t, v in find_head_tracks(seg).items():
            te_head[(seg, t)] = v[0]
    clean = [(u, v) for u, v in dirty
             if not ((u in te_head and te_head[u] == v[1]) or (v in te_head and te_head[v] == u[1]))]

    res = {"config": dict(thr_in=0.15, thr_out=0.25, same_min=0.90, min_eval=100),
           "negatives": {"dirty": len(dirty), "clean": len(clean)}, "variants": {}}

    print("%-16s %-5s %-6s | %-28s | %-28s" % ("variant", "K", "nProto", "dirty cov/same/fm", "clean cov/same/fm"))
    variants = [("mean(baseline)", None), ("knn(all)", 0)]
    variants += [(f"knn(size>={t})", t) for t in (3, 4, 5, 6, 8)]
    for name, ms in variants:
        if ms is None:
            P, pc = build_mean(Vtr, lab, kept, mu_g)
        elif ms == 0:
            P, pc = build_knn(Vtr, lab, kept, mu_g)
        else:
            P, pc = build_gated(Vtr, lab, kept, ms, mu_g)
        line = {}
        for tag, neg in (("dirty", dirty), ("clean", clean)):
            best, Sa, Sb = evaluate(P, pc, K, Vtr, lab, kept, Qa, Qb, kte, neg)
            if best is None:
                line[tag] = None
                continue
            line[tag] = dict(thr=best["thr"], coverage=round(best["coverage"], 4),
                             same=round(best["same"], 4),
                             false_merge=round(best["false_merge"], 5) if np.isfinite(best["false_merge"]) else None,
                             evaluable=best["evaluable"], n_accept=round(best["coverage"] * len(kte)))
        res["variants"][name] = dict(n_proto=int(len(P)), **line)
        d, c = line.get("dirty"), line.get("clean")
        f = lambda x: ("--" if not x else
                       "%.4f/%.4f/%.2f%%" % (x["coverage"], x["same"], 100 * (x["false_merge"] or 0)))
        print("%-16s %-5d %-6d | %-28s | %-28s" % (name, K, len(P), f(d), f(c)))

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
