#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 E1–E3 的判定阈值从"test-oracle"换成"可部署定标"。

问题：session_selector_v1.py 的 θ 是在 test 上按误合并预算挑的（上界，不可部署）。
做法：留一相机交叉定标——对每个相机 c，用其余相机的 train 轨迹建库，拿 c 的轨迹当查询，
      扫描 θ 并记录（覆盖, 稳定, 误合并）；取"四折平均误合并 ≤2% 时平均覆盖最大"的 θ。
      该 θ 随后固定，应用于完整 train 库 + test 查询。

局限（须写进报告）：折内 gallery 只有 3 路（比生产 4 路小），且折内查询与建库同为上午，
      不含真实的光照漂移，因此定出的 θ 偏保守；它只回答"θ 能否不看 test 定出来"。

用法：python tools/session_selector_theta.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/theta_transfer.json")


def build_plan(name, Vtr, lab, kept, ctr, mu_g, percam, use_calib):
    protos, pcl, _ = S.build_gallery(Vtr, lab, ctr, kept, percam=percam)
    protos = _norm(protos - mu_g)
    calib = None
    if use_calib:
        S_tr = S.score_rows(Vtr, protos, pcl, len(kept), mu=mu_g)
        m_hat, s_hat, _ = S.fit_ident_stats(S_tr, lab, kept)
        calib = (m_hat, s_hat)
    return {"name": name, "protos": protos, "pcl": pcl, "K": len(kept),
            "calib": calib, "percam": percam}


def fold_curves(train, plan_cfg, grid):
    """留一相机：返回每折的 sweep 列表。"""
    ktr, Va, Vb, cams = train
    V = _norm((Va + Vb) / 2.0)
    out = []
    for hold in sorted(set(cams.tolist())):
        m_g = cams != hold
        m_q = cams == hold
        if m_g.sum() < 50 or m_q.sum() < 10:
            continue
        Vg = V[m_g]
        lab, kept = S.hier_labels(Vg)
        mu_g = Vg.mean(0)
        plan = build_plan(plan_cfg["name"], Vg, lab, kept, cams[m_g], mu_g,
                          plan_cfg["percam"], plan_cfg["calib"])
        mu_q = V[m_q].mean(0)
        qa = _norm(Va[m_q] - mu_q)
        qb = _norm(Vb[m_q] - mu_q)
        Sa = S.score_rows(qa, plan["protos"], plan["pcl"], plan["K"], mu=None, calib=plan["calib"])
        Sb = S.score_rows(qb, plan["protos"], plan["pcl"], plan["K"], mu=None, calib=plan["calib"])
        keys_q = [k for k, keep in zip(ktr, m_q) if keep]
        cooc_q = []
        # 折内共现负对：只保留双方都在该相机内的
        setq = set(keys_q)
        for s in S.TRAIN_SEGS:
            seg_hold = s.split("_")[0] == hold
            if not seg_hold:
                continue
            for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                                   setq, s, 5):
                cooc_q.append(((s, u), (s, v)))
        rows = S.sweep(Sa, Sb, keys_q, cooc_q, grid)
        out.append({"hold": hold, "n_q": int(m_q.sum()), "n_cooc": len(cooc_q), "rows": rows})
    return out


def pick_theta(folds, budget=0.02):
    """按四折**池化**统计选 θ：池化误合并 ≤ budget 时，按查询数加权平均覆盖最大。"""
    grid = [r["thr"] for r in folds[0]["rows"]]
    w = np.array([f["n_q"] for f in folds], dtype=float)
    best = None
    for i, thr in enumerate(grid):
        mg = ev = 0.0
        covs = []
        for j, f in enumerate(folds):
            r = f["rows"][i]
            if not np.isfinite(r["false_merge"]):
                continue
            e = r["evaluable"]
            mg += r["false_merge"] * e
            ev += e
            covs.append(w[j] * r["coverage"])
        if ev < 150:
            continue
        fm = mg / ev
        if fm <= budget:
            cov = float(np.sum(covs) / np.sum(w))
            cand = {"thr": float(thr), "mean_coverage": cov, "pooled_fm": float(fm),
                    "pooled_evaluable": int(ev)}
            if best is None or cand["mean_coverage"] > best["mean_coverage"]:
                best = cand
    return best


def main():
    train = S.load(S.TRAIN_SEGS)
    ktr, Vtr_a, Vtr_b, ctr = train
    kte, Vte_a, Vte_b, cte = S.load(S.TEST_SEGS)
    Vtr = _norm((Vtr_a + Vtr_b) / 2.0)
    Vte = _norm((Vte_a + Vte_b) / 2.0)
    mu_q_te = Vte.mean(0)

    cooc_te = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte), s, 5):
            cooc_te.append(((s, u), (s, v)))

    grid_raw = np.round(np.arange(0.30, 0.96, 0.01), 3)
    grid_z = np.round(np.arange(-2.0, 6.01, 0.05), 3)

    cfgs = [
        {"name": "E0 raw", "percam": False, "calib": False, "grid": grid_raw},
        {"name": "E1 +align", "percam": False, "calib": False, "grid": grid_raw},
        {"name": "E2 +align+calib", "percam": False, "calib": True, "grid": grid_z},
        {"name": "E3 +align+calib+percam", "percam": True, "calib": True, "grid": grid_z},
    ]

    out = {"folds": {}, "transfer": {}}
    for cfg in cfgs:
        folds = fold_curves(train, cfg, cfg["grid"])
        best = pick_theta(folds)
        out["folds"][cfg["name"]] = {"best_theta": best,
                                     "folds": [{"hold": f["hold"], "n_q": f["n_q"],
                                                "n_cooc": f["n_cooc"]} for f in folds]}
        print(f"[{cfg['name']:24s}] fold-theta: {best}")

        if best is None:
            continue
        # 用完整 train 库 + 该 θ，在 test 上评估
        lab, kept = S.hier_labels(Vtr)
        mu_g = Vtr.mean(0)
        plan = build_plan(cfg["name"], Vtr, lab, kept, ctr, mu_g, cfg["percam"], cfg["calib"])
        Sa = S.score_rows(Vte_a, plan["protos"], plan["pcl"], plan["K"], mu=mu_q_te,
                          calib=plan["calib"])
        Sb = S.score_rows(Vte_b, plan["protos"], plan["pcl"], plan["K"], mu=mu_q_te,
                          calib=plan["calib"])
        r = S.eval_point(Sa, Sb, kte, cooc_te, best["thr"])
        out["transfer"][cfg["name"]] = {"theta": best["thr"], "test": r,
                                        "n_protos": int(len(plan["protos"]))}
        print(f"{'':26s}-> test: cov={r['coverage']:.3f} same={r['same']:.3f} "
              f"fm={r['false_merge']:.4f} (eval={r['evaluable']})")

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
