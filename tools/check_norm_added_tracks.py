#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R3 复核：归一化带来的新增接受，是「真匹配」还是「一致地错」？

T-norm / AS-norm 用 query 自身的分数分布做尺度，会把「对所有簇都低分但相对有峰」的轨迹抬过门槛。
`same` 抓不到这种情况（两片可能一致地选中同一个错误簇），`fm` 也抓不到（错簇若不在场，没有共现行）。
因此必须看**绝对余弦**：
  base 已接受 / 归一化新增接受 / 深度拒绝 三组的 top-1 原始余弦分布。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/r3_abs_check.json")


def main():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g, mu_q = Vtr.mean(0), _norm((Va_te + Vb_te) / 2.0).mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)

    def parts(X):
        Xa = _norm(_norm(X) - mu_q)
        raw = S.cluster_scores(Xa, protos, pcl, K)      # 原始余弦
        z = (raw - m_hat) / s_hat                        # 生产分数
        return raw, z

    Ra, Za = parts(Va_te)
    Rb, Zb = parts(Vb_te)

    def tnorm(Z):
        return (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-6)

    Ta, Tb = tnorm(Za), tnorm(Zb)

    def accepted(Sa, Sb, thr):
        return (Sa.max(1) >= thr) & (Sb.max(1) >= thr)

    base = accepted(Za, Zb, 2.5)
    tn = accepted(Ta, Tb, 3.1)

    abs_top = np.minimum(Ra.max(1), Rb.max(1))  # 两片中较低的那片，保守
    groups = {
        "base_accepted": base,
        "tnorm_only_added": tn & ~base,
        "tnorm_dropped": base & ~tn,
        "both": base & tn,
        "never_accepted": ~base & ~tn,
    }
    out = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            out[name] = {"n": 0}
            continue
        v = abs_top[mask]
        out[name] = {"n": int(mask.sum()),
                     "abs_top1_p10": round(float(np.percentile(v, 10)), 4),
                     "abs_top1_p50": round(float(np.median(v)), 4),
                     "abs_top1_p90": round(float(np.percentile(v, 90)), 4)}
        print(f"{name:18s} n={mask.sum():3d} | abs top1 cos p10={np.percentile(v,10):.3f} "
              f"p50={np.median(v):.3f} p90={np.percentile(v,90):.3f}")

    # 新增接受里，有多少 top1 绝对余弦低于 base 已接受的 p10（=低于任何正常匹配）
    floor = float(np.percentile(abs_top[base], 10))
    weak = int((abs_top[tn & ~base] < floor).sum())
    n_new = int((tn & ~base).sum())
    print(f"\nbase p10 底线={floor:.3f}；新增 {n_new} 条里有 {weak} 条低于该底线")
    out["floor"] = round(floor, 4)
    out["weak_added"] = weak
    out["n_added"] = n_new

    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
