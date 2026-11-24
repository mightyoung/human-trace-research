#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1–E3：会话对齐 / 每身份校准 / per-camera 多原型的增量评估。

假设来源（docs/reid-competitive-research-2026-10-17.md）：
  E1 会话均值对齐    ← PEA (ICLR 2026) 的平移分量
  E2 每身份 z 校准   ← Identity-Conditioned Score Fusion (arXiv 2610.07366) + EVT 尾部思路
  E3 per-camera 原型 ← Dual-Stage Graph Association (Buildings 2026) / 相机不相交画廊

口径（固定）：
  gallery = 1012 train 4 段（上午）；query = 1012 test 4 段（下午，时间隔离）。
  指标：覆盖（a/b 双侧接受）、稳定（条件一致率）、安全（共现误合并率）。
  工作点：test-oracle —— θ 在 test 上按"误合并≤2%"取覆盖最大，仅用于比较**上界**，不可部署。

用法：python tools/session_selector_v1.py
"""
import json
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
from eval_selector_1012test import load_pair, cooc_pairs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "output_crosscam_local/1012_osnet_emb")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/e1_e3.json")

TRAIN_SEGS = ["305_train_529", "318_train_525", "320_train_514", "323_train_510"]
TEST_SEGS = ["305_test_544", "318_test_534", "320_test_518", "323_test_527"]

HIER_THR = 0.2       # 沿用既有训练报告（hier-avg, cosine）
MIN_CLUSTER = 2
MIN_PERCAM = 2
TAIL_PCT = 90.0
FM_BUDGET = 0.02


def load(segs):
    keys, Va, Vb, cams = [], [], [], []
    for s in segs:
        ks, a, b = load_pair(EMB, s)
        cam = s.split("_")[0]
        keys += ks
        Va.append(a)
        Vb.append(b)
        cams += [cam] * len(ks)
    return keys, np.concatenate(Va), np.concatenate(Vb), np.array(cams)


def hier_labels(V, thr=None):
    Z = linkage(V, method="average", metric="cosine")
    raw = fcluster(Z, t=HIER_THR if thr is None else thr, criterion="distance")
    uniq = {c: i for i, c in enumerate(sorted(set(raw.tolist())))}
    lab = np.array([uniq[c] for c in raw])
    return lab, kept_of(lab)


def kept_of(lab):
    return [c for c in sorted(set(lab.tolist())) if (lab == c).sum() >= MIN_CLUSTER]


def build_gallery(V, lab, cams, kept, percam=False):
    """percam=True 时每簇按相机建原型（≥MIN_PERCAM 才建），无则回退簇均值。"""
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        got = False
        if percam:
            for cam in sorted(set(cams[m].tolist())):
                mm = m & (cams == cam)
                if mm.sum() >= MIN_PERCAM:
                    protos.append(_norm(V[mm].mean(0, keepdims=True))[0])
                    pcl.append(ci)
                    got = True
        if not got:
            protos.append(_norm(V[m].mean(0, keepdims=True))[0])
            pcl.append(ci)
    return np.asarray(protos, np.float32), np.asarray(pcl), list(range(len(kept)))


def cluster_scores(X, protos, pcl, K):
    S = X @ protos.T
    out = np.full((len(X), K), -np.inf, dtype=np.float32)
    for k in range(K):
        cols = np.where(pcl == k)[0]
        if cols.size:
            out[:, k] = S[:, cols].max(1)
    return out


def fit_ident_stats(S_train, lab, kept, pct=TAIL_PCT):
    """每身份冒充者尾部统计 + 经验贝叶斯收缩（向池化统计量）。"""
    K = S_train.shape[1]
    ms, ss, ns = np.full(K, np.nan), np.full(K, np.nan), np.zeros(K)
    for k in range(K):
        s = S_train[lab != kept[k], k]
        if s.size < 10:
            continue
        tail = s[s >= np.percentile(s, pct)]
        ms[k], ss[k], ns[k] = float(tail.mean()), float(tail.std() + 1e-6), float(s.size)
    ok = np.isfinite(ms) & np.isfinite(ss)
    m0 = float(np.median(ms[ok])); s0 = float(np.median(ss[ok]))
    n0 = max(1.0, float(np.median(ns[ok])))
    m_hat = (np.where(ok, ns * np.nan_to_num(ms), 0) + n0 * m0) / (np.where(ok, ns, 0) + n0)
    s_hat = (np.where(ok, ns * np.nan_to_num(ss), 0) + n0 * s0) / (np.where(ok, ns, 0) + n0)
    return m_hat.astype(np.float32), s_hat.astype(np.float32), {"m0": m0, "s0": s0, "n0": n0}


def score_rows(X, protos, pcl, K, mu=None, calib=None):
    Xa = _norm(X - mu) if mu is not None else _norm(X)
    S = cluster_scores(Xa, protos, pcl, K)
    if calib is not None:
        S = (S - calib[0]) / calib[1]
    return S


def _fm_count(acca, accb, ia, ib, kidx, pairs, fixed):
    """统计假合并的（合并数, 分母）。

    fixed=False：历史口径，只有「某方向两半都过阈值」的对才入分母 → 分母随 θ 变化。
    fixed=True ：固定口径，pairs 全部入分母，未过阈值的对按「未合并」计。
    """
    ev = mg = 0
    for u, v in pairs:
        iu, iv = kidx[u], kidx[v]
        dir_uv = acca[iu] and accb[iv]
        dir_vu = acca[iv] and accb[iu]
        if fixed:
            ev += 1
            if (dir_uv and ia[iu] == ib[iv]) or (dir_vu and ia[iv] == ib[iu]):
                mg += 1
        elif dir_uv or dir_vu:
            ev += 1
            if (dir_uv and ia[iu] == ib[iv]) or (dir_vu and ia[iv] == ib[iu]):
                mg += 1
    return mg, ev


def eval_point(Sa, Sb, keys, cooc, thr, eval_pairs=None):
    """单点评估。

    eval_pairs=None（默认）沿用历史口径：fm 的分母是该 θ 下可评估的共现对，
    随 θ 变化。★ 该口径下 fm **不可跨配置比较**——θ 越低分母越大，若新纳入的
    对更容易分开，fm 会被稀释下降，形成「质量更好」的假象
    （见 docs/e21-results-2026-11-09.md §13）。

    给定 eval_pairs（固定共现对集合）时，分母恒为 len(eval_pairs)，与 θ 无关，
    此时 fm 才可跨配置比较。两种口径同时返回，便于交叉核对。

    选择 eval_pairs 的原则：**取全部共现对**（等价于比较绝对误合并对数）最不易被
    操纵；若取「某配置 A 工作点上的可评估对」，则 A 没有决定的对不计入分母，
    接受更宽的配置在这些对上的误合并也看不见 —— 会**系统性地偏袒接受更宽的一方**。
    """
    kidx = {k: i for i, k in enumerate(keys)}
    ia, sa = Sa.argmax(1), Sa.max(1)
    ib, sb = Sb.argmax(1), Sb.max(1)
    acca, accb = sa >= thr, sb >= thr
    both = acca & accb
    same = float((ia[both] == ib[both]).mean()) if both.any() else float("nan")

    if eval_pairs is None:
        mg, ev = _fm_count(acca, accb, ia, ib, kidx, cooc, fixed=False)
        extra = {"evaluable": ev,
                 "false_merge": (mg / ev if ev else float("nan"))}
    else:
        mg, ev = _fm_count(acca, accb, ia, ib, kidx, eval_pairs, fixed=True)
        mg_a, ev_a = _fm_count(acca, accb, ia, ib, kidx, cooc, fixed=False)
        extra = {"evaluable": ev,
                 "false_merge": (mg / ev if ev else float("nan")),
                 "evaluable_adaptive": ev_a,
                 "false_merge_adaptive": (mg_a / ev_a if ev_a else float("nan"))}

    return {"thr": float(thr), "coverage": float(both.mean()), "same": same, **extra}


def sweep(Sa, Sb, keys, cooc, grid):
    return [eval_point(Sa, Sb, keys, cooc, t) for t in grid]


def pick(rows, min_eval=100):
    cand = [r for r in rows if r["evaluable"] >= min_eval
            and np.isfinite(r["false_merge"]) and r["false_merge"] <= FM_BUDGET]
    return max(cand, key=lambda r: r["coverage"]) if cand else None


def main():
    ktr, Vtr_a, Vtr_b, ctr = load(TRAIN_SEGS)
    kte, Vte_a, Vte_b, cte = load(TEST_SEGS)
    Vtr = _norm((Vtr_a + Vtr_b) / 2.0)
    Vte = _norm((Vte_a + Vte_b) / 2.0)
    print(f"train {len(ktr)} tracks / test {len(kte)} tracks")

    cooc_te = []
    for s in TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte), s, 5):
            cooc_te.append(((s, u), (s, v)))
    print(f"cooc test {len(cooc_te)}")

    lab, kept = hier_labels(Vtr)
    K = len(kept)
    print(f"clusters kept {K}")

    mu_g = Vtr.mean(0)      # gallery 侧统计（train）
    mu_q = Vte.mean(0)      # query 侧统计（test，无标签）

    p_single, pcl_single, _ = build_gallery(Vtr, lab, ctr, kept, percam=False)
    p_percam, pcl_percam, _ = build_gallery(Vtr, lab, ctr, kept, percam=True)
    p_single_al = _norm(p_single - mu_g)
    p_percam_al = _norm(p_percam - mu_g)
    print(f"protos: single={len(p_single)} percam={len(p_percam)}")

    plans = [
        # name, protos, pcl, mu_gallery, mu_query, calib?
        ("E0 raw", p_single, pcl_single, None, None, False),
        ("E1 +align", p_single_al, pcl_single, mu_g, mu_q, False),
        ("E2 +align+calib", p_single_al, pcl_single, mu_g, mu_q, True),
        ("E3b +align+percam", p_percam_al, pcl_percam, mu_g, mu_q, False),
        ("E3 +align+calib+percam", p_percam_al, pcl_percam, mu_g, mu_q, True),
    ]

    grid_raw = np.round(np.arange(0.30, 0.96, 0.01), 3)
    grid_z = np.round(np.arange(-2.0, 6.01, 0.05), 3)

    results = {}
    for name, protos, pcl, mg, mq, use_calib in plans:
        calib = None
        if use_calib:
            S_tr = score_rows(Vtr, protos, pcl, K, mu=mg)
            m_hat, s_hat, pinfo = fit_ident_stats(S_tr, lab, kept)
            calib = (m_hat, s_hat)
        grid = grid_z if use_calib else grid_raw
        Sa = score_rows(Vte_a, protos, pcl, K, mu=mq, calib=calib)
        Sb = score_rows(Vte_b, protos, pcl, K, mu=mq, calib=calib)
        rows = sweep(Sa, Sb, kte, cooc_te, grid)
        best = pick(rows)
        results[name] = {"n_protos": int(len(protos)), "n_clusters": int(K),
                         "calibrated": bool(use_calib),
                         "best_test_fm2": best, "sweep": rows}
        b = best or {}
        print(f"[{name:24s}] protos={len(protos):4d} | best@fm<=2%: thr={b.get('thr')} "
              f"cov={b.get('coverage', float('nan')):.3f} same={b.get('same', float('nan')):.3f} "
              f"fm={b.get('false_merge', float('nan')):.4f} (eval={b.get('evaluable')})")

    print("\n== 同覆盖率对照（same / 误合并）==")
    print(f"{'cov':>5} " + " ".join(f"{n:>24s}" for n in results))
    matched = {}
    for target in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        line = f"{target:5.2f} "
        matched[str(target)] = {}
        for name, r in results.items():
            cand = [x for x in r["sweep"] if x["coverage"] >= target and x["evaluable"] >= 100
                    and np.isfinite(x["false_merge"])]
            if not cand:
                line += f"{'n/a':>24s} "
                continue
            x = min(cand, key=lambda z: z["coverage"])
            matched[str(target)][name] = x
            line += f"{x['same']:.3f}/{x['false_merge']*100:5.2f}%      "
        print(line)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"train_tracks": len(ktr), "test_tracks": len(kte),
               "cooc_test": len(cooc_te), "n_clusters": K,
               "n_protos": {"single": int(len(p_single)), "percam": int(len(p_percam))},
               "variants": results, "matched_coverage": matched},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
