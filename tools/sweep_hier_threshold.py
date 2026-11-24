#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""层次聚类阈值（HIER_THR）扫描：图库纯度 vs 下游三指标。

背景：生产用 average-linkage + cosine 距离阈值 0.2（= 相似度 0.8）。但异人共现对的 p99 余弦是 0.915，
说明 0.8 远低于"把冒充者挡住"的位置。本脚本回答：阈值收紧后，图库误合并率降多少、下游覆盖/一致率付多少代价。
"""
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/hier_thr_sweep.json")
GRID = np.round(np.arange(-2.0, 8.01, 0.05), 3)


def main():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    mu_g, mu_q = Vtr.mean(0), _norm((Va_te + Vb_te) / 2.0).mean(0)

    kidx = {k: i for i, k in enumerate(ktr)}
    tr_pairs = []
    for seg in S.TRAIN_SEGS:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        tr_pairs += [((seg, u), (seg, v)) for u, v in cooc_pairs(p, set(ktr), seg, 5)]

    cooc_te = []
    for seg in S.TEST_SEGS:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        cooc_te += [((seg, u), (seg, v)) for u, v in cooc_pairs(p, set(kte), seg, 5)]

    Z = linkage(Vtr, method="average", metric="cosine")
    res = []
    for thr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
        raw = fcluster(Z, t=thr, criterion="distance")
        uniq = {c: i for i, c in enumerate(sorted(set(raw.tolist())))}
        lab = np.array([uniq[c] for c in raw])
        kept = S.kept_of(lab)
        K = len(kept)
        if K == 0:
            continue

        # 图库纯度
        merges = sum(1 for u, v in tr_pairs if lab[kidx[u]] == lab[kidx[v]])
        # 只保留 >=2 的簇后，再看有多少合并对仍落在 kept 簇里
        kept_set = set(kept)
        merges_live = sum(1 for u, v in tr_pairs
                          if lab[kidx[u]] == lab[kidx[v]] and lab[kidx[u]] in kept_set)

        protos, pcl = [], []
        for ci, c in enumerate(kept):
            m = lab == c
            protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
            pcl.append(ci)
        protos = _norm(np.asarray(protos, np.float32) - mu_g)
        pcl = np.asarray(pcl)
        mh, sh, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
        Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(mh, sh))
        Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(mh, sh))
        rows = S.sweep(Sa, Sb, kte, cooc_te, GRID)
        ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
        best = max(ok, key=lambda r: r["coverage"]) if ok else None
        b = best or {}
        e = {"hier_thr": thr, "sim_thr": round(1 - thr, 3), "K": K,
             "tr_merges_total": merges,
             "tr_merge_rate": round(merges / len(tr_pairs), 4),
             "tr_merges_live": merges_live,
             "coverage": b.get("coverage"), "same": b.get("same"),
             "fm": b.get("false_merge"), "thr": b.get("thr")}
        res.append(e)
        print(f"[thr={thr:.2f} sim>={1-thr:.2f}] K={K:3d} 图库误合并 {merges:3d}/{len(tr_pairs)} "
              f"({100*merges/len(tr_pairs):.2f}%) | cov={e['coverage']:.3f} same={e['same']:.3f} "
              f"fm={e['fm']*100:.2f}%", flush=True)

    json.dump({"n_train": len(ktr), "n_test": len(kte), "train_pairs": len(tr_pairs),
               "rows": res}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
