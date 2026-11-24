#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""k-reciprocal 重排实验（R4-5）。

标准做法（Zhong et al. CVPR'17）：
  D -> k1 近邻 -> k-reciprocal 集合 R -> 用 R 的 Jaccard 距离构造 V
  final_d = (1-lambda) * jaccard + lambda * D
本脚本把它接到现有判定管线上：
  · 相似度图建在「train 轨迹 + test 轨迹」的并集上（标准做法，测试样本也参与邻居关系）
  · 簇分数 = 簇内 train 轨迹的重排相似度取 max
  · 其余（每身份 z 校准、θ 扫描、self-cal）与生产完全一致

注意：重排用的是原始 cosine（不做均值对齐），与生产 scorer 不同，
所以**只能按同覆盖点比较**，不能拿 self-cal 单点直接对比。

用法：
  python tools/eval_kreciprocal.py --k1 20 --k2 6 --lambda 0.3
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402
import eval_mask_variants as M  # noqa: E402
import cluster_camera_aware as CA  # noqa: E402

ROOT = S.ROOT
LOCAL = os.path.join(ROOT, "output_crosscam_local")


def load(emb, segs):
    keys, As, Bs = [], [], []
    for s in segs:
        k, a, b = M.load_pair(emb, s)
        if len(k):
            keys += k
            As.append(a)
            Bs.append(b)
    return keys, np.concatenate(As), np.concatenate(Bs)


def k_reciprocal(dist, k1=20, k2=6, lam=0.3):
    """dist: (n,n) 距离矩阵；返回重排后的距离矩阵（越大越不相似）。"""
    n = dist.shape[0]
    k1 = min(k1, n - 1)
    k2 = max(1, min(k2, k1 // 2))
    orig = dist.copy()
    np.fill_diagonal(orig, np.inf)
    order = np.argsort(orig, axis=1)
    # 初始 k 近邻集合
    R = [set(order[i, :k1].tolist()) for i in range(n)]
    # k-reciprocal 扩展：若 j 在 R(i) 且 i 在 R(j) 的 k2 近邻内，则把 j 的 k2 近邻并入 R(i)
    V = []
    for i in range(n):
        ri = set(R[i])
        for j in list(ri):
            if i in set(order[j, :k2].tolist()):
                ri |= set(order[j, :k2].tolist())
        V.append(ri)
    # Jaccard 距离
    J = np.zeros((n, n), np.float64)
    for i in range(n):
        vi = V[i]
        for j in range(i, n):
            vj = V[j]
            inter = len(vi & vj)
            union = len(vi) + len(vj) - inter
            J[i, j] = J[j, i] = 1.0 - inter / max(1, union)
    out = (1.0 - lam) * J + lam * dist
    return out


def build_protos(lab, kept, V):
    mu = V.mean(0)
    protos = _norm(np.asarray([_norm(V[lab == c].mean(0, keepdims=True))[0] for c in kept],
                              np.float32) - mu)
    return mu, protos


def at_cov(rows, target, min_eval=100):
    cand = [r for r in rows if r["coverage"] >= target and r["evaluable"] >= min_eval
            and np.isfinite(r["false_merge"])]
    return min(cand, key=lambda r: r["coverage"]) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(LOCAL, "1012_yolo8_dd_emb"))
    ap.add_argument("--det", default=os.path.join(LOCAL, "1012_yolo8_dd"))
    ap.add_argument("--k1", type=int, default=20)
    ap.add_argument("--k2", type=int, default=6)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--align", action="store_true",
                    help="建图前做会话均值对齐（train 用自身均值、test 用自身均值），与生产 scorer 的 E1 一致")
    ap.add_argument("--out", default=os.path.join(LOCAL, "selector_osnet_1012/kreciprocal.json"))
    a = ap.parse_args()

    ktr, Atr, Btr = load(a.emb, S.TRAIN_SEGS)
    kte, Ate, Bte = load(a.emb, S.TEST_SEGS)
    Vtr = _norm((Atr + Btr) / 2.0)
    Vte = _norm((Ate + Bte) / 2.0)
    if a.align:
        mu_g0 = Vtr.mean(0)
        mu_q0 = Vte.mean(0)
        Vtr = _norm(Vtr - mu_g0)
        Ate = _norm(Ate - mu_q0)
        Bte = _norm(Bte - mu_q0)
    cams = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams, 0.15, 0.25, S.MIN_CLUSTER)
    K = len(kept)

    cooc = []
    for s in S.TEST_SEGS:
        p = os.path.join(a.det, s, f"{s}_detections.jsonl")
        if os.path.exists(p):
            cooc += [((s, u), (s, v)) for u, v in cooc_pairs(p, set(kte), s, 5)]

    ntr, nte = len(Vtr), len(Vte)
    X = np.vstack([Vtr, Vte])
    D = (1.0 - X @ X.T).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    Dr = k_reciprocal(D, a.k1, a.k2, a.lam)
    Sr = 1.0 - Dr

    # 簇分数 = 簇内 train 轨迹的最大重排相似度
    def cluster_scores(rows_slice):
        S0 = np.full((rows_slice.shape[0], K), -np.inf, np.float64)
        for ci, c in enumerate(kept):
            idx = np.where(lab == c)[0]
            if idx.size:
                S0[:, ci] = rows_slice[:, idx].max(1)
        return S0

    S_tr = cluster_scores(1.0 - Dr[:ntr, :ntr])
    m_hat, s_hat, _ = S.fit_ident_stats(S_tr.astype(np.float32), lab, kept)

    # test 的 a/b 两侧特征各自重建图并重排（保持与生产 a/b 双侧判定同构）
    Xa = np.vstack([Vtr, _norm(Ate)])
    Xb = np.vstack([Vtr, _norm(Bte)])
    Da = (1.0 - Xa @ Xa.T).astype(np.float64)
    Db = (1.0 - Xb @ Xb.T).astype(np.float64)
    np.fill_diagonal(Da, 0.0)
    np.fill_diagonal(Db, 0.0)
    Sa = cluster_scores(1.0 - k_reciprocal(Da, a.k1, a.k2, a.lam)[ntr:, :ntr])
    Sb = cluster_scores(1.0 - k_reciprocal(Db, a.k1, a.k2, a.lam)[ntr:, :ntr])
    Sa = ((Sa - m_hat) / s_hat).astype(np.float32)
    Sb = ((Sb - m_hat) / s_hat).astype(np.float32)

    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    rows = S.sweep(Sa, Sb, kte, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    print(f"k-reciprocal k1={a.k1} k2={a.k2} lambda={a.lam}  K={K}")
    if best:
        print(f"[self-cal] cov={best['coverage']:.3f} same={best['same']:.3f} "
              f"fm={best['false_merge']*100:.2f}% eval={best['evaluable']} thr={best['thr']:.2f}")
    for t in (0.55, 0.60, 0.65, 0.70):
        r = at_cov(rows, t)
        print(f"  @cov>={t}: " + ("—" if r is None else
                                  f"same={r['same']:.3f} fm={r['false_merge']*100:.2f}%"))

    json.dump({"k1": a.k1, "k2": a.k2, "lambda": a.lam, "K": K,
               "self_cal": best,
               "at_cov": {str(t): at_cov(rows, t) for t in (0.55, 0.60, 0.65, 0.70)}},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {a.out}")

    # --- 与生产 scorer 的逐轨迹差分（新增接受的是不是碎轨迹）---
    import collections
    mu_g, protos = build_protos(lab, kept, Vtr)
    S_tr2 = S.score_rows(Vtr, protos, np.arange(K), K, mu=mu_g)
    m2, s2, _ = S.fit_ident_stats(S_tr2, lab, kept)
    mu_q = _norm((Ate + Bte) / 2.0).mean(0)
    Pa = S.score_rows(Ate, protos, np.arange(K), K, mu=mu_q, calib=(m2, s2))
    Pb = S.score_rows(Bte, protos, np.arange(K), K, mu=mu_q, calib=(m2, s2))
    rows2 = S.sweep(Pa, Pb, kte, cooc, grid)
    ok2 = [r for r in rows2 if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
    b2 = max(ok2, key=lambda r: r["coverage"])
    acc_p = (Pa.max(1) >= b2["thr"]) & (Pb.max(1) >= b2["thr"])
    acc_k = (Sa.max(1) >= best["thr"]) & (Sb.max(1) >= best["thr"])
    new = [kte[i] for i in range(len(kte)) if acc_k[i] and not acc_p[i]]
    lost = [kte[i] for i in range(len(kte)) if acc_p[i] and not acc_k[i]]
    obs = collections.Counter()
    for s in S.TEST_SEGS:
        p = os.path.join(a.det, s, f"{s}_detections.jsonl")
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                d = json.loads(line)
                for t in d.get("track_ids") or []:
                    obs[(s, int(t))] += 1
    print(f"[vs 生产] 生产 θ={b2['thr']:.2f} cov={b2['coverage']:.3f}；"
          f"k-recip θ={best['thr']:.2f} cov={best['coverage']:.3f}；新增 {len(new)} 丢失 {len(lost)}")
    if new:
        o = sorted(obs.get(k, 0) for k in new)
        print(f"  新增轨迹观测数：中位 {o[len(o)//2]}，<20 观测占 "
              f"{sum(1 for v in o if v < 20)/len(o)*100:.0f}%")


if __name__ == "__main__":
    main()
