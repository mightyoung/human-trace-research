# -*- coding: utf-8 -*-
"""T2 实验：双底座分数融合（DINOv3-B + EmbeddingGemma 2）——免训练，30 对。

方法：
  A. 原始 cos 加权: s = w*dino + (1-w)*gemma
  B. z-score 标准化后加权
  C. rank 平均
输出：各方式的 AUC 与 minPOS-maxNEG margin，扫描 w。
"""
import os
import csv
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC = os.path.join(ROOT, "output_crosscam", "analysis", "reid_compare")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "fusion")
os.makedirs(OUT, exist_ok=True)


def load_scores():
    """-> dict: pair_str -> {'dino': v, 'gemma': v, 'label': 'POS'/'NEG'}"""
    rec = {}
    with open(os.path.join(RC, "scores.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pair = r["pair"]
            rec[pair] = {"dino": float(r["dinov3_vitb16"]), "label": r["label"],
                         "gemma": None}
    g = json.load(open(os.path.join(RC, "embgemma2_result_gpu.json"), encoding="utf-8"))
    for pairs, scores, lab in ((g["pos_pairs"], g["pos"], "POS"),
                               (g["neg_pairs"], g["neg"], "NEG")):
        for (a, b), s in zip(pairs, scores):
            key = f"{a}:{b}"
            assert key in rec, f"scores.csv 缺 {key}"
            assert rec[key]["label"] == lab, f"{key} 标签不一致"
            rec[key]["gemma"] = float(s)
    assert all(v["gemma"] is not None for v in rec.values())
    return rec


def auc(pv, nv):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pv for n in nv]))


def main():
    rec = load_scores()
    keys = sorted(rec.keys())
    d = np.array([rec[k]["dino"] for k in keys])
    g = np.array([rec[k]["gemma"] for k in keys])
    lab = np.array([rec[k]["label"] for k in keys])
    pm, nm = lab == "POS", lab == "NEG"
    print(f"样本: {pm.sum()} 正 / {nm.sum()} 负")
    print(f"DINOv3-B : AUC={auc(d[pm], d[nm]):.4f}  minPOS={d[pm].min():.4f} maxNEG={d[nm].max():.4f} margin={d[pm].min()-d[nm].max():+.4f}")
    print(f"Gemma2   : AUC={auc(g[pm], g[nm]):.4f}  minPOS={g[pm].min():.4f} maxNEG={g[nm].max():.4f} margin={g[pm].min()-g[nm].max():+.4f}")
    print()

    results = {}

    # ---- A. 原始 cos 加权 ----
    print("=== A. 原始 cos 加权 w*dino + (1-w)*gemma ===")
    print(f"{'w':>5} {'AUC':>7} {'minPOS':>8} {'maxNEG':>8} {'margin':>8}")
    for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s = w * d + (1 - w) * g
        mp, mn = s[pm].min(), s[nm].max()
        a = auc(s[pm], s[nm])
        print(f"{w:>5.1f} {a:>7.4f} {mp:>8.4f} {mn:>8.4f} {mp-mn:>+8.4f}")
        results[f"A_w{w:.1f}"] = {"auc": a, "margin": float(mp - mn)}
    print()

    # ---- B. z-score 标准化加权 ----
    def zs(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    dz, gz = zs(d), zs(g)
    print("=== B. z-score 标准化后加权 ===")
    print(f"{'w':>5} {'AUC':>7} {'margin':>8}")
    for w in [0, 0.25, 0.5, 0.75, 1.0]:
        s = w * dz + (1 - w) * gz
        mp, mn = s[pm].min(), s[nm].max()
        a = auc(s[pm], s[nm])
        print(f"{w:>5.2f} {a:>7.4f} {mp-mn:>+8.4f}")
        results[f"B_w{w:.2f}"] = {"auc": a, "margin": float(mp - mn)}
    print()

    # ---- C. rank 平均 ----
    def rank(x):
        order = np.argsort(np.argsort(x)).astype(float)
        return order / (len(x) - 1)
    s = 0.5 * rank(d) + 0.5 * rank(g)
    mp, mn = s[pm].min(), s[nm].max()
    a = auc(s[pm], s[nm])
    print("=== C. rank 平均 (0.5/0.5) ===")
    print(f"AUC={a:.4f}  margin={mp-mn:+.4f}")
    results["C_rank"] = {"auc": a, "margin": float(mp - mn)}

    with open(os.path.join(OUT, "score_fusion.json"), "w", encoding="utf-8") as f:
        json.dump({"n_pos": int(pm.sum()), "n_neg": int(nm.sum()),
                   "dino_auc": auc(d[pm], d[nm]), "gemma_auc": auc(g[pm], g[nm]),
                   "results": results}, f, ensure_ascii=False, indent=1)
    print("\n[ok] ->", os.path.join(OUT, "score_fusion.json"))


if __name__ == "__main__":
    main()
