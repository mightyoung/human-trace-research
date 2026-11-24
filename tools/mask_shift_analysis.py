#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""掩膜对表征的移动量：同裁剪、同权重，只差"掩膜置黑"，cosine 变了多少。

数据来源（无需再推理）：
  基线逐裁特征 output_crosscam_local/1012_osnet_crops/{seg}_crops.npz
  掩膜逐裁特征 output_crosscam_local/1012_osnet_crops_mask/{seg}_crops_{variant}.npz
两者 eid 同一编码规则 "{ab}#{tid}#{k}"，可直接 join。

输出：按段、按 mask_frac 分桶的 cosine；用于判断"掩膜是否真的改变了表征"，
以及变化是否集中在人体占比低的框（即座椅/桌面占比高的框）。

用法：python tools/mask_shift_analysis.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402

ROOT = S.ROOT
BASE = os.path.join(ROOT, "output_crosscam_local/1012_osnet_crops")
MASK = os.path.join(ROOT, "output_crosscam_local/1012_osnet_crops_mask")


def load(p):
    """一次读出数组，避免 NpzFile 每次索引都重新解压（那是 O(n) 次解压）。"""
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    eids = [str(e) for e in z["eids"].tolist()]
    emb = z["emb"]
    q = z["q"] if "q" in z.files else None
    return {e: i for i, e in enumerate(eids)}, emb, q


def main():
    out = {}
    for seg in S.TRAIN_SEGS + S.TEST_SEGS:
        base = load(os.path.join(BASE, f"{seg}_crops.npz"))
        if base is None:
            continue
        bmap, Bemb, _bq = base
        for var in ("maskfill", "maskbbox"):
            cur = load(os.path.join(MASK, f"{seg}_crops_{var}.npz"))
            if cur is None:
                continue
            cmap, Cemb, Cq = cur
            common = [k for k in cmap if k in bmap]
            if not common:
                continue
            E0 = Bemb[[bmap[k] for k in common]]
            E1 = Cemb[[cmap[k] for k in common]]
            q = Cq[[cmap[k] for k in common]]
            cos = np.sum(E0 * E1, axis=1)
            # mask_frac 分桶
            bins = [(None, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]
            bstat = {}
            for lo, hi in bins:
                if lo is None:
                    m = q[:, 1] == 0.0
                else:
                    m = (q[:, 1] > max(lo, 1e-9)) & (q[:, 1] <= hi)
                if m.sum() == 0:
                    continue
                bstat["no-mask" if lo is None else f"{lo:.2f}-{hi:.2f}"] = {"n": int(m.sum()),
                                               "cos_mean": round(float(cos[m].mean()), 4)}
            out[f"{seg}|{var}"] = {
                "crops": len(common),
                "cos_mean": round(float(cos.mean()), 4),
                "cos_p10": round(float(np.percentile(cos, 10)), 4),
                "cos_p50": round(float(np.median(cos)), 4),
                "frac_cos_below_0.99": round(float((cos < 0.99).mean()), 4),
                "by_mask_frac": bstat,
            }
            print(f"{seg:14s} {var:9s} n={len(common):5d} cos={cos.mean():.4f} "
                  f"p10={np.percentile(cos,10):.3f} <0.99:{(cos<0.99).mean()*100:.0f}%", flush=True)

    dst = os.path.join(MASK, "shift_analysis.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {dst}")


if __name__ == "__main__":
    main()
