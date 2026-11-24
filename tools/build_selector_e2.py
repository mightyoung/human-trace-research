#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建 E2 选择器产物（对齐 + 每身份校准），并做自洽性验证。

产物 selector_osnet_1012_e2.npz：
  protos         [P, D]  已按 gallery 均值对齐的原型（单原型/簇）
  proto_cluster  [P]     原型 -> 簇（E2 为一一对应）
  mu_gallery     [D]     建库会话均值
  m_hat, s_hat   [K]     每身份冒充者尾部统计（校准用）
  thr_default    float   train 侧 self-cal 得到的阈值（z 尺度），作为无会话数据时的兜底
  meta           JSON    构建信息

运行期做法（见 predict）：
  mu_session ← 本段轨迹的均值（无标签，缓冲 N 条后估计）
  X ← normalize(v − mu_session)
  S ← X @ protos.T ；按簇取 max ；z ← (S − m_hat) / s_hat
  accept ⇔ max z ≥ θ ；θ 由本段 a/b 一致性自校准（target_same=0.90）

用法：
  python tools/build_selector_e2.py            # 构建 + 验证
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

OUT = os.path.join(S.ROOT, "output_crosscam_local/selector_osnet_1012/selector_osnet_1012_e2.npz")


def build(Vtr, ctr):
    lab, kept = S.hier_labels(Vtr)
    protos, pcl, _ = S.build_gallery(Vtr, lab, ctr, kept, percam=False)
    mu_g = Vtr.mean(0)
    protos_al = _norm(protos - mu_g)
    S_tr = S.score_rows(Vtr, protos_al, pcl, len(kept), mu=mu_g)
    m_hat, s_hat, info = S.fit_ident_stats(S_tr, lab, kept)
    return {"protos": protos_al, "proto_cluster": pcl, "mu_gallery": mu_g,
            "m_hat": m_hat, "s_hat": s_hat, "K": len(kept), "lab": lab, "kept": kept,
            "calib_info": info}


def predict(model, Va, Vb, cams, theta):
    mu_q = _norm((Va + Vb) / 2.0).mean(0)
    Sa = S.score_rows(Va, model["protos"], model["proto_cluster"], model["K"],
                      mu=mu_q, calib=(model["m_hat"], model["s_hat"]))
    Sb = S.score_rows(Vb, model["protos"], model["proto_cluster"], model["K"],
                      mu=mu_q, calib=(model["m_hat"], model["s_hat"]))
    return Sa, Sb


def selfcal_theta(model, Va, Vb, keys, cooc, target=0.90):
    Sa, Sb = predict(model, Va, Vb, None, None)
    grid = np.round(np.arange(-1.0, 8.01, 0.05), 3)
    rows = S.sweep(Sa, Sb, keys, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= target
          and r["evaluable"] >= 100]
    return (max(ok, key=lambda r: r["coverage"]) if ok else None)


def main():
    ktr, Vtr_a, Vtr_b, ctr = S.load(S.TRAIN_SEGS)
    kte, Vte_a, Vte_b, cte = S.load(S.TEST_SEGS)
    Vtr = _norm((Vtr_a + Vtr_b) / 2.0)
    model = build(Vtr, ctr)
    print(f"[build] clusters={model['K']} protos={len(model['protos'])} "
          f"calib m0={model['calib_info']['m0']:.3f} s0={model['calib_info']['s0']:.4f}")

    cooc_tr = []
    for s in S.TRAIN_SEGS:
        for u, v in cooc_pairs(os.path.join(S.ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(ktr), s, 5):
            cooc_tr.append(((s, u), (s, v)))
    tr_pt = selfcal_theta(model, Vtr_a, Vtr_b, ktr, cooc_tr)
    thr_default = float(tr_pt["thr"]) if tr_pt else 2.75
    print(f"[train self-cal] thr={thr_default:.2f} cov={tr_pt['coverage']:.3f} "
          f"same={tr_pt['same']:.3f} fm={tr_pt['false_merge']*100:.2f}%")

    np.savez(OUT, protos=model["protos"], proto_cluster=np.asarray(model["proto_cluster"]),
             mu_gallery=model["mu_gallery"], m_hat=model["m_hat"], s_hat=model["s_hat"],
             thr_default=np.array([thr_default], np.float32),
             meta=np.array(json.dumps({
                 "name": "selector_osnet_1012_e2", "space": "osnet_x1_0_msmt17 512d",
                 "align": "subtract session mean, renormalize",
                 "calib": "per-identity impostor tail p90 + empirical-Bayes shrinkage",
                 "n_clusters": int(model["K"]),
                 "train_segs": S.TRAIN_SEGS,
                 "thr_default_scope": "same-session only",
                 "thr_default_warning": ("该值在训练段样本内标定（cov≈0.99），"
                                         "跨会话会严重偏低；生产必须用本段 a/b 做 self-cal 重定"),
                 "thr_reference_cross_session": 2.75,
                 "built": "2026-10-17"}, ensure_ascii=False)))
    print(f"[save] {OUT}")
    print(f"[warn] thr_default={thr_default:.2f} 仅同会话有效；跨会话须用 self-cal（本次 test 为 2.75）")

    # ---- 自洽性验证：用 test 自校准 θ，复现 E2 那一行 ----
    cooc_te = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(S.ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte), s, 5):
            cooc_te.append(((s, u), (s, v)))
    te_pt = selfcal_theta(model, Vte_a, Vte_b, kte, cooc_te)
    print(f"[test  self-cal] thr={te_pt['thr']:.2f} cov={te_pt['coverage']:.3f} "
          f"same={te_pt['same']:.3f} fm={te_pt['false_merge']*100:.2f}% "
          f"(eval={te_pt['evaluable']})")
    print("[expect] 与 final_e0_e3.json 的 E2 行一致（cov 0.423 / same 0.903 / fm 1.96%）")


if __name__ == "__main__":
    main()
