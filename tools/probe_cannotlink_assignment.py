#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探针：把「同帧共现」从度量改成分配约束，是否值得？

做法（与生产口径同分数、同共现对）：
  1) 复现 yolo8dd 的 Sa/Sb 与共现对；
  2) 对每个 theta：先按现状算 coverage / same / fm；
  3) 再做一次「cannot-link 冲突消解」：把「已接受且落到同簇」的共现对建成冲突图，
     每个连通分量只保留 min(sa,sb) 最高的一条轨迹，其余强制弃判（abstain）；
  4) 重算 coverage / fm，得到「约束版前沿」。

如果约束版在同样 fm 下覆盖更高，说明 R1b（约束式分配）值得做；否则不值得。
只读，不写任何既有产物。
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
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/probe_cannotlink.json")


def build():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    return kte, Sa, Sb, K


def conflicts(kte, Sa, Sb, cooc, thr):
    """返回：接受掩码、簇指派、冲突对（在已接受且同簇的共现对上）。"""
    ia, ib = Sa.argmax(1), Sb.argmax(1)
    sa, sb = Sa.max(1), Sb.max(1)
    acc = (sa >= thr) & (sb >= thr)
    kidx = {k: i for i, k in enumerate(kte)}
    out = []
    for u, v in cooc:
        iu, iv = kidx[u], kidx[v]
        if not (acc[iu] and acc[iv]):
            continue
        if ia[iu] == ib[iv] and ia[iv] == ib[iu]:
            out.append((iu, iv))
        elif ia[iu] == ia[iv] or ib[iu] == ib[iv]:
            out.append((iu, iv))
    return acc, (ia, ib), out


def eval_rows(kte, Sa, Sb, cooc, thr, drop=()):
    """drop = 强制弃判的轨迹下标集合。"""
    kidx = {k: i for i, k in enumerate(kte)}
    drop = set(drop)
    ia, ib = Sa.argmax(1), Sb.argmax(1)
    sa, sb = Sa.max(1), Sb.max(1)
    acca = (sa >= thr) & np.array([i not in drop for i in range(len(kte))])
    accb = (sb >= thr) & np.array([i not in drop for i in range(len(kte))])
    both = acca & accb
    same = float((ia[both] == ib[both]).mean()) if both.any() else float("nan")
    ev = mg = 0
    for u, v in cooc:
        iu, iv = kidx[u], kidx[v]
        if (acca[iu] and accb[iv]) or (acca[iv] and accb[iu]):
            ev += 1
            if (acca[iu] and accb[iv] and ia[iu] == ib[iv]) or \
               (acca[iv] and accb[iu] and ia[iv] == ib[iu]):
                mg += 1
    return {"thr": float(thr), "coverage": float(both.mean()), "same": same,
            "fm": (mg / ev if ev else float("nan")), "evaluable": ev, "merged": mg}


def resolve(kte, Sa, Sb, cooc, thr):
    """冲突消解：每个连通分量保留 min(sa,sb) 最高者，其余弃判。返回被弃判的下标。"""
    acc, _, bad = conflicts(kte, Sa, Sb, cooc, thr)
    if not bad:
        return []
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in bad:
        union(u, v)
    groups = {}
    for i in list(parent):
        groups.setdefault(find(i), []).append(i)
    s = np.minimum(Sa.max(1), Sb.max(1))
    drop = []
    for g in groups.values():
        if len(g) <= 1:
            continue
        keep = max(g, key=lambda i: s[i])
        drop += [i for i in g if i != keep]
    return drop


def main():
    kte, Sa, Sb, K = build()
    cooc = []
    keys_ok = set(kte)
    for seg in S.TEST_SEGS:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        cooc += [((seg, u), (seg, v)) for u, v in cooc_pairs(p, keys_ok, seg, 5)]
    print(f"tracks={len(kte)} K={K} cooc={len(cooc)}")

    rows = []
    for thr in np.round(np.arange(1.50, 2.81, 0.10), 2):
        base = eval_rows(kte, Sa, Sb, cooc, thr)
        drop = resolve(kte, Sa, Sb, cooc, thr)
        con = eval_rows(kte, Sa, Sb, cooc, thr, drop)
        rows.append({"base": base, "constrained": {**con, "n_dropped": len(drop)}})
        print(f"thr={thr:.2f} | base cov={base['coverage']:.3f} same={base['same']:.3f} fm={base['fm']*100:.2f}% "
              f"| cannot-link cov={con['coverage']:.3f} same={con['same']:.3f} fm={con['fm']*100:.2f}% "
              f"(drop {len(drop)})")

    json.dump({"K": K, "n_test": len(kte), "n_cooc": len(cooc), "rows": rows},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
