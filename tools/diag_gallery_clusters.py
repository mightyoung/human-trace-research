#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：gallery 聚类为什么把明显不同的人放在一起。

三条线索：
  A. **库内误合并率**：train 段同帧共现（同时在场＝必为不同人）的轨迹对，有多少被分到同一簇。
     这个数以前从未算过——共现一直被当作 test 侧的评估工具，没用来体检图库本身。
  B. **桥接轨迹**：同一条轨迹的 a/b 两片若外观差很多（a/b 余弦低），说明轨迹内部发生了 ID 切换；
     它的均值向量会落在两个人之间，成为层次聚类的桥。
  C. **消融**：把低 a/b 一致性的轨迹剔除后重新聚类，看误合并率、簇数、以及 test 指标怎么变。

只读，不覆盖任何既有产物。
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/gallery_diag.json")


def load(segs):
    keys, A, B = [], [], []
    for s in segs:
        k, a, b = M.load_pair(EMB, s)
        keys += k
        A.append(a)
        B.append(b)
    return keys, np.concatenate(A), np.concatenate(B)


def cluster(V, thr=0.2):
    Z = linkage(V, method="average", metric="cosine")
    raw = fcluster(Z, t=thr, criterion="distance")
    uniq = {c: i for i, c in enumerate(sorted(set(raw.tolist())))}
    return np.array([uniq[c] for c in raw]), Z


def main():
    ktr, Va, Vb = load(S.TRAIN_SEGS)
    V = _norm((Va + Vb) / 2.0)
    ab_cos = (Va * Vb).sum(1)
    lab, Z = cluster(V)
    n = len(ktr)

    # A. 库内共现对
    pairs = []
    keys_ok = set(ktr)
    for seg in S.TRAIN_SEGS:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        pairs += [((seg, u), (seg, v)) for u, v in cooc_pairs(p, keys_ok, seg, 5)]
    kidx = {k: i for i, k in enumerate(ktr)}
    merges = [(u, v) for u, v in pairs if lab[kidx[u]] == lab[kidx[v]]]
    print(f"train tracks={n} clusters={len(set(lab.tolist()))} cooc pairs={len(pairs)} "
          f"库内误合并={len(merges)} ({100*len(merges)/max(1,len(pairs)):.2f}%)")

    # 簇规模
    sizes = Counter(lab.tolist())
    print("簇规模: 中位 %d, 最大 %d, >=10 条的簇 %d 个" %
          (int(np.median(list(sizes.values()))), max(sizes.values()),
           sum(1 for v in sizes.values() if v >= 10)))

    # B. a/b 一致性
    print("a/b 余弦: p1=%.3f p5=%.3f p10=%.3f 中位=%.3f" %
          tuple(np.percentile(ab_cos, [1, 5, 10, 50])))
    bad = ab_cos < 0.5
    print(f"a/b<0.5 的轨迹: {int(bad.sum())} 条 ({100*bad.mean():.1f}%)")

    # 误合并对的 a/b 与相似度
    if merges:
        sim = np.array([float(V[kidx[u]] @ V[kidx[v]]) for u, v in merges])
        abc = np.array([min(ab_cos[kidx[u]], ab_cos[kidx[v]]) for u, v in merges])
        print(f"误合并对: 两轨相似度中位={np.median(sim):.3f}  p10={np.percentile(sim,10):.3f} "
              f"| 两轨 a/b 余弦较小者中位={np.median(abc):.3f} "
              f"| 至少一轨 a/b<0.5 的占比={100*(abc<0.5).mean():.0f}%")

    # 每簇最差一对（暴露链式合并）
    worst = []
    for c, sz in sizes.items():
        idx = np.where(lab == c)[0]
        if len(idx) < 2:
            continue
        pair_sim = (V[idx] @ V[idx].T)
        np.fill_diagonal(pair_sim, 1.0)
        worst.append((float(pair_sim.min()), int(sz), c, int(np.unravel_index(pair_sim.argmin(), pair_sim.shape)[0])))
    worst.sort()
    print("最松散簇（簇内最小相似度从低到高）:")
    for s, sz, c, _ in worst[:5]:
        print(f"  簇{c}: size={sz} 最小对内相似度={s:.3f}")

    # C. 消融：剔除低 a/b 轨迹后重聚类
    res = []
    for t in (0.0, 0.4, 0.5, 0.6, 0.7):
        keep = ab_cos >= t if t > 0 else np.ones(n, bool)
        Vk = _norm(V[keep])
        labk, _ = cluster(Vk)
        kk = [k for k, m in zip(ktr, keep) if m]
        idxk = {k: i for i, k in enumerate(kk)}
        p2 = [(u, v) for u, v in pairs if u in idxk and v in idxk]
        mg2 = [(u, v) for u, v in p2 if labk[idxk[u]] == labk[idxk[v]]]
        res.append({"ab_min": t, "kept": int(keep.sum()), "clusters": int(len(set(labk.tolist()))),
                    "pairs": len(p2), "merges": len(mg2),
                    "merge_rate": round(len(mg2) / max(1, len(p2)), 4)})
        print(f"[ab>={t:.1f}] kept={keep.sum():3d} clusters={len(set(labk.tolist())):3d} "
              f"pairs={len(p2):4d} 误合并={len(mg2):3d} ({100*len(mg2)/max(1,len(p2)):.2f}%)")

    json.dump({"n_train": n, "n_clusters": len(set(lab.tolist())), "n_pairs": len(pairs),
               "n_merges": len(merges), "sizes": dict(sizes), "ablation": res},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
