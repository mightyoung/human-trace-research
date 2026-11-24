#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""机位感知的图库聚类（camera-aware clustering）。

背景（docs/neighbor-implementation-survey-2026-10-19.md §3）：
  无监督 ReID 里同一机位的特征天然更近（camera bias），把 4 个机位混在一起做一次
  average-linkage 会让「同机位的不同人」被并进同一簇（本项目实测 2.19% 共现对被并）。
  工业做法是先机位内紧聚类，再做跨机位合并（CALR / O2CAP 的 camera-aware proxy）。

本实现：
  1) 机位内：每段 train 单独做 average-linkage（cosine），阈值 thr_in（紧）。
  2) 跨机位：对子簇质心做 complete-linkage（cosine），阈值 thr_out；
     同机位的两个子簇距离强制为 2.0，因此永远不会被并（complete 取最大距离，
     任何包含同机位子簇对的两簇都不可能低于阈值）。
  3) 丢掉成员数 < min_cluster 的簇（与 hier_labels 的 kept_of 一致）。

只在图库侧（train）使用；query 侧分配逻辑不变。
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import squareform

FORBID = 2.0  # 同机位子簇间的强制距离（cosine distance 上限）


def _labels_by_thr(X, thr, method="average"):
    if len(X) == 1:
        return np.zeros(1, dtype=int)
    Z = linkage(X, method=method, metric="cosine")
    return fcluster(Z, t=thr, criterion="distance")


def camera_aware_labels(V, cams, thr_in=0.12, thr_out=0.20, min_cluster=2,
                        verbose=False, member_level=False):
    """返回 (lab, kept)。lab 为每条轨迹的簇 id（0..K-1 连续），kept 为保留簇列表。"""
    V = np.asarray(V, np.float32)
    cams = np.asarray(cams)
    n = len(V)
    if n == 0:
        return np.zeros(0, dtype=int), []

    # 1) 机位内紧聚类
    sub = np.full(n, -1, dtype=int)
    nxt = 0
    for cam in sorted(set(cams.tolist())):
        idx = np.where(cams == cam)[0]
        lab_c = _labels_by_thr(V[idx], thr_in)
        for c in sorted(set(lab_c.tolist())):
            sub[idx[lab_c == c]] = nxt
            nxt += 1
    n_sub = nxt

    # 2) 跨机位合并（complete-linkage on 子簇质心，同机位禁止合并）
    if n_sub <= 1:
        merged = sub.copy()
    else:
        cents = np.zeros((n_sub, V.shape[1]), np.float32)
        sub_cam = np.empty(n_sub, dtype=object)
        for s in range(n_sub):
            m = sub == s
            v = V[m].mean(0)
            cents[s] = v / (np.linalg.norm(v) + 1e-12)
            sub_cam[s] = cams[m][0]
        if member_level:
            # G 组变体：跨机位合并时用「成员级最大距离」而非子簇质心距离。
            # 质心距离只约束平均位置，一串「中间人」能把两人串成一簇（详见
            # deliverable/data-request-v2/ANALYSIS.md §三.6）。
            Sfull = np.clip(V @ V.T, -1.0, 1.0)
            groups = [np.where(sub == s)[0] for s in range(n_sub)]
            D = np.zeros((n_sub, n_sub), np.float64)
            for s in range(n_sub):
                for t in range(s + 1, n_sub):
                    if sub_cam[s] == sub_cam[t]:
                        v = FORBID
                    else:
                        v = 1.0 - float(Sfull[np.ix_(groups[s], groups[t])].min())
                    D[s, t] = D[t, s] = v
        else:
            sim = np.clip(cents @ cents.T, -1.0, 1.0)
            D = (1.0 - sim).astype(np.float64)
            np.fill_diagonal(D, 0.0)
            same_cam = sub_cam[:, None] == sub_cam[None, :]
            D[same_cam] = FORBID
            np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="complete")
        raw = fcluster(Z, t=thr_out, criterion="distance")
        merged = np.array([raw[sub[i]] for i in range(n)])

    # 3) 连续化 + 过滤小簇
    uniq = {c: i for i, c in enumerate(sorted(set(merged.tolist())))}
    lab = np.array([uniq[c] for c in merged], dtype=int)
    kept = [c for c in sorted(set(lab.tolist())) if (lab == c).sum() >= min_cluster]
    if verbose:
        print(f"[camera-aware] thr_in={thr_in} thr_out={thr_out} "
              f"tracks={n} sub_clusters={n_sub} final_clusters={len(set(lab.tolist()))} "
              f"kept={len(kept)}")
    return lab, kept


def _split_rec(V, cams, idx, thr, depth, max_depth):
    """递归切开：只在「跨机位且距离 > thr」的 MST 最长边上切。"""
    if idx.size < 2 or depth >= max_depth:
        return [idx]
    X = V[idx]
    D = 1.0 - np.clip(X @ X.T, -1.0, 1.0)
    np.fill_diagonal(D, 0.0)
    Mst = minimum_spanning_tree(D).toarray()
    rows, cols = np.nonzero(Mst)
    best = None
    for a, b in zip(rows.tolist(), cols.tolist()):
        if a >= b:
            continue
        if cams[idx[a]] == cams[idx[b]]:
            continue                      # 只切跨机位边，机位内不切
        w = float(Mst[a, b])
        if w > thr and (best is None or w > best[0]):
            best = (w, a, b)
    if best is None:
        return [idx]
    _, a, b = best
    Mst[a, b] = Mst[b, a] = 0.0
    ncomp, labels = connected_components(Mst, directed=False)
    out = []
    for k in range(ncomp):
        out += _split_rec(V, cams, idx[labels == k], thr, depth + 1, max_depth)
    return out


def split_cross_camera(lab, kept, V, cams, thr_split=0.15, max_depth=8,
                       min_size=1, verbose=False):
    """D 组：在每个簇的成员级 MST 上，沿「跨机位且 > thr_split」的最长边递归切开。

    与 camera_aware_labels 的区别只在跨机位环节：
      - 现有实现按**子簇质心**做 complete-linkage，质心近就合并；
      - 这里改成在**成员级 MST** 上看跨机位最长边，超过阈值就断开。
    返回 (new_lab, new_kept)，簇号连续。
    """
    n = len(lab)
    new_lab = np.full(n, -1, dtype=int)
    nxt = 0
    n_split = 0
    for c in kept:
        idx = np.where(lab == c)[0]
        parts = _split_rec(V, cams, idx, thr_split, 0, max_depth)
        if len(parts) > 1:
            n_split += 1
        for p in parts:
            new_lab[p] = nxt
            nxt += 1
    new_kept = [c for c in range(nxt) if (new_lab == c).sum() >= min_size]
    if verbose:
        print(f"[split] thr_split={thr_split} 切开簇数={n_split} "
              f"簇总数 {len(kept)} -> {len(new_kept)}")
    return new_lab, new_kept
