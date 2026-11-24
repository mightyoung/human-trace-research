#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务一：训练集无监督聚类 → 身份选择器（原型库）。

输入：训练集切段的 cradio 嵌入（每段两份：轨迹前半 _a / 后半 _b，用于无标签一致性评估）
      {emb_dir}/{seg}_emb_a.npz / {seg}_emb_b.npz，键 = "{seg}#{tid}"

流程：
  ① 合并向量 V = normalize(mean(a, b))
  ② 分片判别力基线：正对(a,b) vs 随机对 的 cos AUC
  ③ 三种无监督聚类对比（无标签选择指标 = 分片一致率 - 随机对碰撞率）：
     - mutual-kNN 图连通分量（k=8, cos 阈值扫描）
     - K-means（scipy kmeans2, K 扫描）
     - average-linkage 层次聚类（阈值扫描）
  ④ 选优 → 原型库（簇内均值向量归一化）→ IdentitySelector 保存

产物：
  {out}/selector.npz          —— 原型库（给 selector.py 加载）
  {out}/cluster_report.json   —— 各方法指标、最优参数、簇统计
  {out}/sim_dist.png          —— 正对/负对相似度分布图
  {out}/cluster_sizes.png     —— 簇规模图
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector, _norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tail_digits(s):
    num = ""
    for ch in reversed(s):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_pair(emb_dir, seg):
    za = np.load(os.path.join(emb_dir, f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(emb_dir, f"{seg}_emb_b.npz"))
    keys, va, vb = [], [], []
    for k in za.files:
        if k not in zb.files:
            continue
        keys.append((seg, tail_digits(k)))
        va.append(za[k])
        vb.append(zb[k])
    return keys, np.asarray(va, np.float32), np.asarray(vb, np.float32)


def pair_auc(pos, neg):
    """cos 相似度的 AUC（pos 应高于 neg）"""
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(len(s))
    # 处理并列（简化：无并列）
    r_pos = ranks[y == 1].sum()
    n1, n0 = len(pos), len(neg)
    return float((r_pos - n1 * (n1 - 1) / 2) / (n1 * n0))


def assign(q, protos):
    q = _norm(q, axis=1)
    S = q @ protos.T
    return S.argmax(1), S.max(1)


def cluster_mutual_knn(V, k, thr):
    S = V @ V.T
    np.fill_diagonal(S, -1.0)
    kk = min(k, len(V) - 1)
    idx = np.argsort(-S, axis=1)[:, :kk]
    A = np.zeros(S.shape, dtype=bool)
    rows = np.arange(len(V))[:, None]
    A[rows, idx] = True
    M = A & A.T & (S >= thr)
    n_comp, lab = connected_components(csr_matrix(M), directed=False)
    return lab, n_comp


def cluster_kmeans(V, K, seed=42):
    try:
        cen, lab = kmeans2(V, K, iter=60, minit="++", seed=seed)
    except Exception:
        return None, None
    return lab, K


def cluster_hier(V, thr, method="average"):
    Z = linkage(V, method=method, metric="cosine")
    lab = fcluster(Z, t=thr, criterion="distance")
    return lab, len(set(lab))


def protos_from_labels(V, lab):
    ids = sorted(set(lab))
    protos = []
    for c in ids:
        m = V[lab == c]
        protos.append(_norm(m.mean(0, keepdims=True))[0])
    return np.asarray(protos, np.float32), ids


def consistency(Va, Vb, protos, seed=0, n_neg=30000):
    ia, _ = assign(Va, protos)
    ib, _ = assign(Vb, protos)
    same = float((ia == ib).mean())
    rng = np.random.default_rng(seed)
    l = rng.integers(0, len(Va), n_neg)
    r = rng.integers(0, len(Vb), n_neg)
    keep = l != r
    coll = float((ia[l[keep]] == ib[r[keep]]).mean())
    return same, coll, same - coll


def pairwise_f1(ia, ib, seed=0, n_neg=30000):
    """pairwise 聚类 F1：正对=同轨迹(a,b)，负对=随机对。
    注意：随机负对中会混入"同人多次出现"的真对（无从确认），在真实场景偏保守；
    真实数据优先用 pairwise_f1_cooc（共现负对，确凿不同人）。"""
    N = len(ia)
    tp = int((ia == ib).sum())
    rng = np.random.default_rng(seed)
    l = rng.integers(0, N, n_neg)
    r = rng.integers(0, N, n_neg)
    keep = l != r
    fp = int((ia[l[keep]] == ib[r[keep]]).sum())
    prec = tp / max(1e-9, tp + fp)
    rec = tp / max(1e-9, N)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "prec": float(prec), "rec": float(rec),
            "f1": float(f1), "neg_kind": "random"}


def cooc_pairs(jsonl_path, keys_set, seg, min_common=5):
    """同帧共现 ≥min_common 的轨迹对（同时在场 = 铁定不同人，确凿负对）。
    keys_set: set of (seg, tid)——只统计双方都有嵌入的对（短轨迹无嵌入则排除）。"""
    if not os.path.exists(jsonl_path):
        return []
    cnt = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ts = sorted(set(int(t) for t in (r.get("track_ids") or [])
                            if t is not None and t >= 0 and (seg, int(t)) in keys_set))
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    cnt[(ts[i], ts[j])] = cnt.get((ts[i], ts[j]), 0) + 1
    return [k for k, v in cnt.items() if v >= min_common]


def pairwise_f1_cooc(ia_map, ib_map, cooc, keys, seed=0, n_cap=8000):
    """共现负对 F1：正对=(a_k,b_k) 同簇；负对=共现对 (a_u, b_v) 同簇（应为不同簇）。
    ia_map/ib_map: key->cluster；keys: key 列表；cooc: [(key_u, key_v), ...]"""
    N = len(keys)
    tp = int(sum(1 for k in keys if ia_map[k] == ib_map[k]))
    if not cooc:
        return None
    rng = np.random.default_rng(seed)
    if len(cooc) > n_cap:
        sel = rng.choice(len(cooc), n_cap, replace=False)
        cooc = [cooc[i] for i in sel]
    # 交叉分片：a_u vs b_v 与 a_v vs b_u 均计（防御：任一键缺失则跳过）
    fp = 0
    tot = 0
    for u, v in cooc:
        if u in ia_map and v in ia_map and u in ib_map and v in ib_map:
            fp += int(ia_map[u] == ib_map[v]) + int(ia_map[v] == ib_map[u])
            tot += 2
    prec = tp / max(1e-9, tp + fp)
    rec = tp / max(1e-9, N)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "prec": float(prec), "rec": float(rec),
            "f1": float(f1), "neg_kind": "cooc", "n_cooc": len(cooc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--segs", required=True, help="训练段（逗号分隔）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cluster", type=int, default=2)
    ap.add_argument("--jsonl-dir", default=os.path.join(ROOT, "output_crosscam", "1012_final"),
                    help="检测 jsonl 目录（计算共现负对）")
    ap.add_argument("--cooc-min", type=int, default=5)
    a = ap.parse_args()
    segs = [s for s in a.segs.split(",") if s]
    os.makedirs(a.out, exist_ok=True)

    keys, Va, Vb = [], [], []
    for s in segs:
        ks, va, vb = load_pair(a.emb_dir, s)
        keys += ks
        Va.append(va)
        Vb.append(vb)
        print(f"[load] {s}: {len(ks)} tracks")
    Va = np.concatenate(Va)
    Vb = np.concatenate(Vb)
    V = _norm((Va + Vb) / 2.0, axis=1)
    N = len(V)
    print(f"[lib] {N} tracks (train)")

    # ---- 分片判别力 ----
    pos = np.sum(Va * Vb, axis=1)
    rng = np.random.default_rng(7)
    l = rng.integers(0, N, 40000)
    r = rng.integers(0, N, 40000)
    keep = l != r
    neg = np.sum(Va[l[keep]] * Vb[r[keep]], axis=1)
    auc = pair_auc(pos, neg)
    print(f"[split-half] pos cos mean={pos.mean():.4f} neg mean={neg.mean():.4f} AUC={auc:.4f}")

    # ---- 共现负对（同帧共现 ≥cooc_min 的两轨迹 = 铁定不同人） ----
    keys_set = set(keys)  # (seg, tid)
    cooc = []
    for s in segs:
        jp = os.path.join(a.jsonl_dir, f"{s}.jsonl")
        for u, v in cooc_pairs(jp, keys_set, s, a.cooc_min):
            cooc.append(((s, u), (s, v)))
    print(f"[cooc] {len(cooc)} pairs (negatives, min_common={a.cooc_min})")

    # ---- thr_unknown 校准：分片正对 p5（同人下界）与共现负对 p99（异人上界）的中点 ----
    kidx = {k: i for i, k in enumerate(keys)}
    pos_p5 = float(np.percentile(pos, 5))
    cooc_p99 = None
    if cooc:
        cs = [float(V[kidx[u]] @ V[kidx[v]]) for u, v in cooc if u in kidx and v in kidx]
        if cs:
            cooc_p99 = float(np.percentile(cs, 99))
    thr_unknown = max(0.2, (pos_p5 + cooc_p99) / 2) if cooc_p99 else 0.3
    print(f"[thr] pos_p5={pos_p5:.3f} cooc_p99={cooc_p99 if cooc_p99 is None else round(cooc_p99,3)} "
          f"-> thr_unknown={thr_unknown:.3f}")

    # ---- 三方法对比（选择标准 = pairwise F1：正对=同轨迹 a/b；负对=共现对） ----
    results = []

    def _eval(method, params, lab, ncl):
        protos, _ = protos_from_labels(V, lab)
        ia, _ = assign(Va, protos)
        ib, _ = assign(Vb, protos)
        pf = None
        if cooc:
            ia_map = {k: int(ia[i]) for i, k in enumerate(keys)}
            ib_map = {k: int(ib[i]) for i, k in enumerate(keys)}
            pf = pairwise_f1_cooc(ia_map, ib_map, cooc, keys)
        if pf is None:
            pf = pairwise_f1(ia, ib)
        sm, cl, sc = consistency(Va, Vb, protos)
        d = {"method": method, **params, "n_clusters": ncl,
             "f1": pf["f1"], "prec": pf["prec"], "rec": pf["rec"],
             "neg_kind": pf.get("neg_kind"), "same": sm, "coll": cl, "score": sc,
             "_lab": lab}
        results.append(d)
        return d

    # 1) mutual-kNN
    for k in (5, 8, 12):
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            lab, ncl = cluster_mutual_knn(V, k, thr)
            if ncl < 2 or ncl > N * 0.75:
                continue
            _eval("mutual-knn", {"k": k, "thr": thr}, lab, ncl)
    # 2) kmeans
    for K in [k for k in (4, 6, 8, 12, 16, 20, 28, 40, 60) if k <= N * 0.5]:
        lab, ncl = cluster_kmeans(V, K)
        if lab is None:
            continue
        _eval("kmeans", {"K": K}, lab, ncl)
    # 3) hierarchy
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        lab, ncl = cluster_hier(V, thr)
        if ncl < 2 or ncl > N * 0.75:
            continue
        _eval("hier-avg", {"thr": thr}, lab, ncl)

    results.sort(key=lambda d: (-round(d["f1"], 4), d["n_clusters"]))
    print("\n[rank] top-8 (pairwise F1):")
    for d in results[:8]:
        print(f"  {d['method']:11s} {str({k:v for k,v in d.items() if k in ('k','K','thr')}):30s} "
              f"ncl={d['n_clusters']:3d} F1={d['f1']:.3f} P={d['prec']:.3f} R={d['rec']:.3f} "
              f"(same={d['same']:.3f} coll={d['coll']:.3f})")

    best = results[0]
    lab_best = best["_lab"]
    # 小簇过滤（<min_cluster 不建原型；分配时自然落到最近大原型）
    protos, ids = protos_from_labels(V, lab_best)
    sizes = np.array([(lab_best == c).sum() for c in ids])
    keep_mask = sizes >= a.min_cluster
    if not keep_mask.any():
        keep_mask = sizes == sizes.max()
        print("[warn] all clusters < min_cluster; fallback to largest cluster")
    protos2 = protos[keep_mask]
    n_removed = int((~keep_mask).sum())
    print(f"\n[best] {best['method']} ncl={best['n_clusters']} F1={best['f1']:.3f}; "
          f"去小簇 {n_removed} -> 原型库 {len(protos2)}")

    sel = IdentitySelector(protos2, thr_unknown=thr_unknown, meta={
        "method": best["method"], "params": {k: v for k, v in best.items()
                                             if k not in ("_lab",)}})
    sel.save(os.path.join(a.out, "selector.npz"))

    rep = {
        "n_train_tracks": N, "split_half": {"pos_mean": float(pos.mean()),
                                            "neg_mean": float(neg.mean()), "auc": auc},
        "thr_calib": {"pos_p5": pos_p5, "cooc_p99": cooc_p99,
                      "thr_unknown": thr_unknown},
        "results": [{k: v for k, v in d.items() if k != "_lab"} for d in results[:12]],
        "best": {k: v for k, v in best.items() if k != "_lab"},
        "cluster_sizes": sorted([int(s) for s in sizes[keep_mask]], reverse=True),
        "n_small_removed": n_removed,
    }
    json.dump(rep, open(os.path.join(a.out, "cluster_report.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(pos, bins=60, alpha=0.6, label=f"pos (same track) AUC={auc:.3f}", density=True)
        axes[0].hist(neg, bins=60, alpha=0.6, label="neg (random)", density=True)
        axes[0].set_title("split-half cos similarity"); axes[0].legend()
        axes[1].bar(range(len(sizes[keep_mask])), sorted(sizes[keep_mask], reverse=True))
        axes[1].set_title(f"cluster sizes (n={len(protos2)})")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "sim_dist.png"), dpi=140)
        print(f"[fig] {os.path.join(a.out, 'sim_dist.png')}")
    except Exception as e:
        print("[fig] skip:", e)

    print(f"[done] selector -> {os.path.join(a.out, 'selector.npz')}")


if __name__ == "__main__":
    main()
