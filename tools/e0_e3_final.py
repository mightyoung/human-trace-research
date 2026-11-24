#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E0–E3 最终对照（单一脚本、单一口径，结果即报告口径）。

变体（严格区分"原型是否对齐"与"查询是否对齐"，避免空间错配）：
  E0 raw               原型 raw            查询 raw
  E1 +align            原型 减 mu_gallery   查询 减 mu_session
  E2 +align+calib      E1 + 每身份 z 校准（train 冒充者尾部 + 经验贝叶斯收缩）
  E3b +align+percam    原型 按相机分组后对齐，查询对齐（无校准）
  E3  +align+calib+percam  E3b + 校准

两个工作点（必须分开报）：
  A) self-cal（可部署）：θ 取"覆盖最大且 a/b 一致率 ≥0.90"——只用同段一致性，不用标签
  B) oracle（上界）：θ 取"覆盖最大且共现误合并 ≤2%"——用 test 侧的共现代理挑阈值

输出：output_crosscam_local/selector_osnet_1012/final_e0_e3.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

OUT = os.path.join(S.ROOT, "output_crosscam_local/selector_osnet_1012/final_e0_e3.json")


def main():
    ktr, Vtr_a, Vtr_b, ctr = S.load(S.TRAIN_SEGS)
    kte, Vte_a, Vte_b, cte = S.load(S.TEST_SEGS)
    Vtr = _norm((Vtr_a + Vtr_b) / 2.0)
    Vte = _norm((Vte_a + Vte_b) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g, mu_q = Vtr.mean(0), Vte.mean(0)

    cooc = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(S.ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte), s, 5):
            cooc.append(((s, u), (s, v)))

    p_raw, pcl_raw, _ = S.build_gallery(Vtr, lab, ctr, kept, percam=False)
    p_cam, pcl_cam, _ = S.build_gallery(Vtr, lab, ctr, kept, percam=True)
    p_al = _norm(p_raw - mu_g)
    p_cam_al = _norm(p_cam - mu_g)

    def calib_for(protos, pcl):
        S_tr = S.score_rows(Vtr, protos, pcl, K, mu=mu_g)
        m_hat, s_hat, info = S.fit_ident_stats(S_tr, lab, kept)
        return (m_hat, s_hat), info

    c_single, info_s = calib_for(p_al, pcl_raw)
    c_cam, info_c = calib_for(p_cam_al, pcl_cam)

    plans = [
        # name, protos, pcl, gallery_mu, query_mu, calib, grid
        ("E0 raw", p_raw, pcl_raw, None, None, None,
         np.round(np.arange(0.20, 0.99, 0.01), 3)),
        ("E1 +align", p_al, pcl_raw, mu_g, mu_q, None,
         np.round(np.arange(0.20, 0.99, 0.01), 3)),
        ("E2 +align+calib", p_al, pcl_raw, mu_g, mu_q, c_single,
         np.round(np.arange(-2.0, 8.01, 0.05), 3)),
        ("E3b +align+percam", p_cam_al, pcl_cam, mu_g, mu_q, None,
         np.round(np.arange(0.20, 0.99, 0.01), 3)),
        ("E3 +align+calib+percam", p_cam_al, pcl_cam, mu_g, mu_q, c_cam,
         np.round(np.arange(-2.0, 8.01, 0.05), 3)),
    ]

    out = {"train_tracks": len(ktr), "test_tracks": len(kte), "n_clusters": K,
           "n_protos": {"single": int(len(p_raw)), "percam": int(len(p_cam))},
           "cooc_pairs": len(cooc), "calib_info": {"single": info_s, "percam": info_c},
           "variants": {}}

    print(f"train {len(ktr)} / test {len(kte)} / clusters {K} / "
          f"protos single {len(p_raw)} percam {len(p_cam)} / cooc {len(cooc)}\n")

    for name, protos, pcl, mg, mq, calib, grid in plans:
        # 注意：protos 已是对齐后的；score_rows 的 mu 只作用于查询
        Sa = S.score_rows(Vte_a, protos, pcl, K, mu=mq, calib=calib)
        Sb = S.score_rows(Vte_b, protos, pcl, K, mu=mq, calib=calib)
        rows = S.sweep(Sa, Sb, kte, cooc, grid)
        ok_a = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90
                and r["evaluable"] >= 100]
        self_cal = max(ok_a, key=lambda r: r["coverage"]) if ok_a else None
        ok_b = [r for r in rows if np.isfinite(r["false_merge"]) and r["false_merge"] <= 0.02
                and r["evaluable"] >= 100]
        oracle = max(ok_b, key=lambda r: r["coverage"]) if ok_b else None
        out["variants"][name] = {"self_cal_same90": self_cal, "oracle_fm2": oracle, "sweep": rows}
        a = self_cal or {}
        b = oracle or {}
        print(f"[{name:24s}] self-cal: thr={a.get('thr')} cov={a.get('coverage', float('nan')):.3f} "
              f"same={a.get('same', float('nan')):.3f} fm={a.get('false_merge', float('nan'))*100:.2f}%"
              f"  | oracle: cov={b.get('coverage', float('nan')):.3f} "
              f"same={b.get('same', float('nan')):.3f} fm={b.get('false_merge', float('nan'))*100:.2f}%")

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
