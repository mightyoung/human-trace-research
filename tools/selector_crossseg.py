#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器跨段身份保持检验（train 原型库 → test 段推理）。

思路（无人工标注下的"伪 ground truth"）：
  1. test 段某轨迹 t 与 train 段某轨迹 t* 的嵌入余弦 sim 高（≥thr）且【互为最近邻】→ 判为"疑似同一人"；
  2. 检查这些跨段配对经选择器分配（assign）后是否落到同一簇（身份保持率）;
  3. 同时报告 test 段 unknown 比例与簇 ID 分布（选择器在测试段的"开放性"行为）。

用法：
  python tools/selector_crossseg.py --selector <selector.npz> \
      --emb-dir output_crosscam/analysis/cradio_1012_emb \
      --train-segs 320_train_514,... --test-segs 320_test_518,... \
      --out output_crosscam/analysis/selector_1012/crossseg.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tail_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_pairs(emb_dir, seg):
    za = np.load(os.path.join(emb_dir, f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(emb_dir, f"{seg}_emb_b.npz"))
    common = sorted(set(za.files) & set(zb.files),
                    key=lambda k: (k.split("#")[0], tail_tid(k)))
    keys, Va, Vb = [], [], []
    for k in common:
        keys.append((seg, k))
        Va.append(za[k])
        Vb.append(zb[k])
    return keys, np.asarray(Va, np.float32), np.asarray(Vb, np.float32)


def load_mean(Va, Vb):
    V = (Va + Vb) / 2.0
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--train-segs", required=True)
    ap.add_argument("--test-segs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thrs", default="0.85,0.90,0.95")
    ap.add_argument("--per-cam", action="store_true",
                    help="按样本摄像头取阈值（需 selector 含 per-cam 校准）")
    ap.add_argument("--loose", action="store_true", help="使用覆盖档阈值（loose）")
    ap.add_argument("--margin", default=None,
                    help="精度档 margin：float 或 'prec'（用库带 per-cam 推荐值）")
    a = ap.parse_args()
    mg = a.margin
    if mg is not None:
        try:
            mg = float(mg)
        except ValueError:
            pass  # 保留 "prec" 字符串

    sel = IdentitySelector.load(a.selector)
    print(f"[selector] protos={sel.protos.shape[0]} n_clusters={sel.n_clusters} "
          f"thr_unknown={sel.thr_unknown:.4f} per_cam={bool(sel.thr_unknown_per_cam)} "
          f"loose={a.loose} per_cam_mode={a.per_cam}")

    tr_segs = [s for s in a.train_segs.split(",") if s]
    te_segs = [s for s in a.test_segs.split(",") if s]

    # ---- 训练段库 ----
    tr_keys, tr_V = [], []
    for s in tr_segs:
        ks, Va, Vb = load_pairs(a.emb_dir, s)
        tr_keys += ks
        tr_V.append(load_mean(Va, Vb))
    tr_V = np.concatenate(tr_V)
    print(f"[train lib] {len(tr_keys)} tracks")

    # ---- 测试段库 ----
    te_keys, te_V = [], []
    for s in te_segs:
        ks, Va, Vb = load_pairs(a.emb_dir, s)
        te_keys += ks
        te_V.append(load_mean(Va, Vb))
    te_V = np.concatenate(te_V)
    print(f"[test lib] {len(te_keys)} tracks")

    # ---- 互最近邻跨段配对（候选） ----
    S = te_V @ tr_V.T                      # (T, R)
    t2r = S.argmax(axis=1)                 # 每个 test 轨 → 最近 train 轨
    S2 = tr_V @ te_V.T
    r2t = S2.argmax(axis=1)                # 每个 train 轨 → 最近 test 轨
    mutual = [(ti, ri) for ti, ri in enumerate(t2r) if r2t[ri] == ti]
    sims = np.array([S[ti, ri] for ti, ri in mutual])
    print(f"[mutual-NN] {len(mutual)} pairs; sim p5={np.percentile(sims,5):.3f} "
          f"p50={np.percentile(sims,50):.3f}")

    # ---- 分配 & 跨段身份保持率（按阈值分档） ----
    cams_te = [s.split("_")[0] for s, _ in te_keys] if a.per_cam else None
    cams_tr = [s.split("_")[0] for s, _ in tr_keys] if a.per_cam else None
    ia, _ = sel.predict(te_V, cams=cams_te, loose=a.loose, margin=mg)
    ib, _ = sel.predict(tr_V, cams=cams_tr, loose=a.loose, margin=mg)  # train 分配（仅供诊断）
    res = {"selector": os.path.basename(a.selector),
           "per_cam": bool(a.per_cam), "loose": bool(a.loose), "margin": a.margin,
           "n_train": len(tr_keys), "n_test": len(te_keys),
           "thr_unknown": float(sel.thr_unknown), "bands": {}}
    for thr in [float(x) for x in a.thrs.split(",")]:
        pairs = [(ti, ri) for ti, ri in mutual if S[ti, ri] >= thr]
        kn = [(ti, ri) for ti, ri in pairs if ia[ti] >= 0 and ib[ri] >= 0]
        if kn:
            keep = sum(1 for ti, ri in kn if ia[ti] == ib[ri])
            rate = keep / len(kn)
        else:
            keep, rate = 0, None
        cov = len(kn) / len(pairs) if pairs else None
        res["bands"][f"{thr:.2f}"] = {"n_pairs": len(pairs), "n_decided": len(kn),
                                      "same_cluster": keep, "keep_rate": rate,
                                      "coverage": cov}
        print(f"[crossseg thr={thr:.2f}] mutual pairs={len(pairs)} decided={len(kn)} "
              f"same_cluster={keep} keep_rate={rate if rate is None else round(rate,4)} "
              f"cov={cov if cov is None else round(cov,3)}")

    # ---- test 段开放性诊断 ----
    unk = float((ia < 0).mean())
    ids, cnts = np.unique(ia, return_counts=True)
    res["test_assign"] = {
        "unknown_ratio": unk,
        "n_clusters_used": int((ids >= 0).sum()),
        "top_clusters": [[int(i), int(c)] for i, c in
                         sorted(zip(ids, cnts), key=lambda x: -x[1])[:10] if i >= 0],
    }
    print(f"[test assign] unknown={unk:.3f}; clusters used={res['test_assign']['n_clusters_used']}")
    print(f"[test assign] top: {res['test_assign']['top_clusters'][:5]}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()
