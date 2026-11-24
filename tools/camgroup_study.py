#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按摄像头分组聚类/选择器 vs 全局方案：对照研究（2026-10-14）。

研究问题：把向量打上摄像头标签、按摄像头分组独立聚类（=每摄像头一个选择器），
能否提高准确率与可行性？

实验设计（计划，先于实测写定）：
  A. 摄像头信号量化：嵌入空间中摄像头可分性（centroid 分类）、同机/跨机 cos 分布
  B. 聚类层对照：全局 vs 分组（cooc-F1 在【同摄像头子集】上的直接对比、split-half AUC）
  C. 选择器层对照（test 集）：全局选择器 vs 组专用选择器（per-camera same/coll/cov）
  D. 跨段身份保持：mutual-NN 对分同机/跨机；三方案（全局/组专用/物理拼接）keep rate
  E. 组间合并诊断：跨组原型最近邻 cos 分布（判断"跨组比较"本身是否更困难）

数据：1012 切分。train 4 段（305/318/320/323），test 4 段；嵌入 cradio v5 768d。
口径：全部复用 train_selector_cluster.py 的聚类/评估函数（同一代码路径保证可比）。
产物：{out}/study.json + 2 图。
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector import IdentitySelector, _norm  # noqa: E402
from train_selector_cluster import (  # noqa: E402
    load_pair, pair_auc, assign, cluster_mutual_knn, cluster_kmeans, cluster_hier,
    protos_from_labels, consistency, pairwise_f1, pairwise_f1_cooc, cooc_pairs,
)

ROOT = os.path.dirname(HERE)
EMB = os.path.join(ROOT, "output_crosscam", "analysis", "cradio_1012_emb")
JDL = os.path.join(ROOT, "output_crosscam", "1012_final")

TRAIN_SEGS = ["305_train_529", "318_train_525", "320_train_514", "323_train_510"]
TEST_SEGS = ["305_test_544", "318_test_534", "320_test_518", "323_test_527"]
CAMS = ["305", "318", "320", "323"]


def cam_of(seg):
    return seg.split("_")[0]


def load_segs(segs):
    keys, Va, Vb = [], [], []
    for s in segs:
        ks, va, vb = load_pair(EMB, s)
        keys += ks
        Va.append(va)
        Vb.append(vb)
    Va = np.concatenate(Va)
    Vb = np.concatenate(Vb)
    V = _norm((Va + Vb) / 2.0, axis=1)
    return keys, Va, Vb, V


def select_best(V, Va, Vb, keys, cooc, verbose=True):
    """与 train_selector_cluster.main 相同的网格与选择标准（-round(f1,4), ncl）。"""
    N = len(V)
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
        results.append({"method": method, **params, "n_clusters": ncl,
                        "f1": pf["f1"], "prec": pf["prec"], "rec": pf["rec"],
                        "tp": int(pf["tp"]), "fp": int(pf["fp"]),
                        "neg_kind": pf.get("neg_kind"), "same": sm, "coll": cl,
                        "score": sc, "_lab": lab})

    for k in (5, 8, 12):
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            lab, ncl = cluster_mutual_knn(V, k, thr)
            if ncl < 2 or ncl > N * 0.75:
                continue
            _eval("mutual-knn", {"k": k, "thr": thr}, lab, ncl)
    for K in [x for x in (4, 6, 8, 12, 16, 20, 28, 40, 60) if x <= max(2, N * 0.5)]:
        lab, ncl = cluster_kmeans(V, K)
        if lab is None:
            continue
        _eval("kmeans", {"K": K}, lab, ncl)
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        lab, ncl = cluster_hier(V, thr)
        if ncl < 2 or ncl > N * 0.75:
            continue
        _eval("hier-avg", {"thr": thr}, lab, ncl)

    results.sort(key=lambda d: (-round(d["f1"], 4), d["n_clusters"]))
    best = results[0]
    if verbose:
        for d in results[:5]:
            print(f"    {d['method']:11s} ncl={d['n_clusters']:3d} F1={d['f1']:.3f} "
                  f"P={d['prec']:.3f} R={d['rec']:.3f}")
    return best, results


def make_selector(V, Va, Vb, keys, cooc, min_cluster=2):
    """跑网格 → best → 去小簇 → IdentitySelector + thr 校准（组内）。"""
    best, results = select_best(V, Va, Vb, keys, cooc)
    lab = best["_lab"]
    protos, ids = protos_from_labels(V, lab)
    sizes = np.array([(lab == c).sum() for c in ids])
    keep = sizes >= min_cluster
    if not keep.any():
        keep = sizes == sizes.max()
    protos2 = protos[keep]
    # thr 校准（组内）
    pos = np.sum(Va * Vb, axis=1)
    pos_p5 = float(np.percentile(pos, 5))
    kidx = {k: i for i, k in enumerate(keys)}
    cs = [float(V[kidx[u]] @ V[kidx[v]]) for u, v in cooc if u in kidx and v in kidx]
    cooc_p99 = float(np.percentile(cs, 99)) if cs else None
    thr_unknown = max(0.2, (pos_p5 + cooc_p99) / 2) if cooc_p99 else 0.3
    sel = IdentitySelector(protos2, thr_unknown=thr_unknown,
                           meta={"method": best["method"]})
    info = {k: v for k, v in best.items() if k != "_lab"}
    info.update({"n_protos": int(len(protos2)), "n_small_removed": int((~keep).sum()),
                 "pos_p5": pos_p5, "cooc_p99": cooc_p99, "thr_unknown": thr_unknown,
                 "n_cooc": len(cooc)})
    return sel, info, protos2


def eval_test(sel, T_keys, T_Va, T_Vb, cam=None, seed=3, n_rand=20000):
    """选择器在 test（可指定摄像头子集）上的 known-restricted 评估。"""
    m = np.array([cam_of(k[0]) == cam for k in T_keys]) if cam else np.ones(len(T_keys), bool)
    keys = [k for k, mm in zip(T_keys, m) if mm]
    Va = T_Va[m]
    Vb = T_Vb[m]
    N = len(keys)
    if N == 0:
        return None
    SA = sel.sims_all(Va)
    SB = sel.sims_all(Vb)
    idxa = SA.argmax(1)
    idxb = SB.argmax(1)
    sa = SA[np.arange(N), idxa]
    sb = SB[np.arange(N), idxb]
    ia = np.where(sa >= sel.thr_unknown, idxa, -1)
    ib = np.where(sb >= sel.thr_unknown, idxb, -1)
    kn = (ia >= 0) & (ib >= 0)
    cov = float(kn.mean())
    same_known = float((ia[kn] == ib[kn]).mean()) if kn.sum() else None
    rng = np.random.default_rng(seed)
    l = rng.integers(0, N, n_rand)
    r = rng.integers(0, N, n_rand)
    kp = l != r
    il, ir = ia[l[kp]], ib[r[kp]]
    kn2 = (il >= 0) & (ir >= 0)
    coll_known = float((il[kn2] == ir[kn2]).mean()) if kn2.sum() else None
    return {"n": N, "n_known": int(kn.sum()), "coverage": cov, "same_known": same_known,
            "coll_known": coll_known, "both_unknown": float(((ia < 0) & (ib < 0)).mean())}


def f3(v):
    return "None" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output_crosscam",
                                                  "analysis", "camgroup_study"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {"design": {
        "train_segs": TRAIN_SEGS, "test_segs": TEST_SEGS,
        "note": "同一 1012 切分；全部复用 train_selector_cluster 网格与选择标准"}}

    # ---- 载入 ----
    print("== load ==")
    TR_keys, TR_Va, TR_Vb, TR_V = load_segs(TRAIN_SEGS)
    TE_keys, TE_Va, TE_Vb, TE_V = load_segs(TEST_SEGS)
    print(f"train {len(TR_keys)} tracks; test {len(TE_keys)} tracks")
    tr_cam = np.array([cam_of(k[0]) for k in TR_keys])
    te_cam = np.array([cam_of(k[0]) for k in TE_keys])

    # ================= A. 摄像头信号量化 =================
    print("\n== A. 摄像头信号（centroid 分类 + 同机/跨机 cos） ==")
    cents, cnames = [], []
    for c in CAMS:
        v = TR_V[tr_cam == c].mean(0)
        cents.append(_norm(v[None, :])[0])
        cnames.append(c)
    cents = np.asarray(cents)
    Cmat = cents @ cents.T
    cent_sim = {(cnames[i], cnames[j]): float(Cmat[i, j])
                for i in range(4) for j in range(i + 1, 4)}
    # test 轨 → 最近摄像头 centroid（train 拟合）
    ST = TE_V @ cents.T
    pred = ST.argmax(1)
    true = np.array([CAMS.index(c) for c in te_cam])
    cent_acc = float((pred == true).mean())
    # 混淆
    conf = {}
    for i, c in enumerate(CAMS):
        sel_i = true == i
        conf[c] = {CAMS[j]: int(((pred == j) & sel_i).sum()) for j in range(4)}
    # 同机 / 跨机 cos 分布（随机对）
    rng = np.random.default_rng(11)
    n = 30000
    l = rng.integers(0, len(TR_V), n)
    r = rng.integers(0, len(TR_V), n)
    kp = l != r
    same_cam = tr_cam[l[kp]] == tr_cam[r[kp]]
    cos_pairs = np.sum(TR_V[l[kp]] * TR_V[r[kp]], axis=1)
    same_cos = cos_pairs[same_cam]
    cross_cos = cos_pairs[~same_cam]
    report["A_camera_signal"] = {
        "centroid_cos_matrix": {f"{c1}-{c2}": v for (c1, c2), v in cent_sim.items()},
        "test_centroid_classify_acc": cent_acc,
        "test_centroid_confusion": conf,
        "same_cam_cos": {"mean": float(same_cos.mean()), "p50": float(np.percentile(same_cos, 50)),
                         "p95": float(np.percentile(same_cos, 95))},
        "cross_cam_cos": {"mean": float(cross_cos.mean()), "p50": float(np.percentile(cross_cos, 50)),
                          "p95": float(np.percentile(cross_cos, 95))},
    }
    print(f"centroid classify acc (test→train cents): {cent_acc:.3f}")
    print(f"same-cam random cos mean={same_cos.mean():.3f} | cross-cam mean={cross_cos.mean():.3f} "
          f"(diff={same_cos.mean()-cross_cos.mean():+.3f})")

    # ================= B + C-prep: 全局选择器（重跑复现） =================
    print("\n== B/C. 全局方案（重跑复现） ==")
    keys_set = set(TR_keys)
    cooc_all = []
    for s in TRAIN_SEGS:
        for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), keys_set, s, 5):
            cooc_all.append(((s, u), (s, v)))
    print(f"[cooc] {len(cooc_all)} pairs (all cams)")
    gsel, ginfo, gprotos = make_selector(TR_V, TR_Va, TR_Vb, TR_keys, cooc_all)
    print(f"[global] best={ginfo['method']} F1={ginfo['f1']:.4f} ncl={ginfo['n_clusters']} "
          f"protos={ginfo['n_protos']} thr={ginfo['thr_unknown']:.4f}")
    report["B_global"] = ginfo

    # 全局的 per-camera F1（在 cooc 子集上与分组方案对比）
    g_protos = gprotos
    gia, _ = assign(TR_Va, g_protos)
    gib, _ = assign(TR_Vb, g_protos)
    g_ia_map = {k: int(gia[i]) for i, k in enumerate(TR_keys)}
    g_ib_map = {k: int(gib[i]) for i, k in enumerate(TR_keys)}

    # ================= B. 分组聚类 =================
    print("\n== B. 分组聚类（每摄像头独立） ==")
    group_info, group_sels, group_protos = {}, {}, {}
    for c in CAMS:
        m = tr_cam == c
        Vc, Vac, Vbc = TR_V[m], TR_Va[m], TR_Vb[m]
        kc = [k for k, mm in zip(TR_keys, m) if mm]
        ks_c = set(kc)
        cooc_c = []
        for s in [s for s in TRAIN_SEGS if cam_of(s) == c]:
            for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), ks_c, s, 5):
                cooc_c.append(((s, u), (s, v)))
        print(f"  [{c}] n={len(kc)} cooc={len(cooc_c)}")
        sel_c, info_c, protos_c = make_selector(Vc, Vac, Vbc, kc, cooc_c)
        group_info[c] = info_c
        group_sels[c] = sel_c
        group_protos[c] = protos_c
        print(f"    best={info_c['method']} F1={info_c['f1']:.4f} ncl={info_c['n_clusters']} "
              f"protos={info_c['n_protos']} thr={info_c['thr_unknown']:.4f}")
    report["B_group"] = group_info

    # per-camera F1 同子集对比（全局 vs 组专用）
    print("\n  per-camera cooc-F1（同子集：正对=该 cam a/b；负对=该 cam 共现）")
    f1_compare = {}
    for c in CAMS:
        m = tr_cam == c
        kc = [k for k, mm in zip(TR_keys, m) if mm]
        cooc_c = [(u, v) for (u, v) in cooc_all if cam_of(u[0]) == c]
        g_pf = pairwise_f1_cooc(g_ia_map, g_ib_map, cooc_c, kc)
        ga, _ = assign(TR_Va[m], group_protos[c])
        gb, _ = assign(TR_Vb[m], group_protos[c])
        grp_ia = {k: int(ga[i]) for i, k in enumerate(kc)}
        grp_ib = {k: int(gb[i]) for i, k in enumerate(kc)}
        c_pf = pairwise_f1_cooc(grp_ia, grp_ib, cooc_c, kc)
        f1_compare[c] = {"global_f1": g_pf["f1"], "group_f1": c_pf["f1"],
                         "global_prec": g_pf["prec"], "group_prec": c_pf["prec"],
                         "global_rec": g_pf["rec"], "group_rec": c_pf["rec"],
                         "n_pos": len(kc), "n_cooc": len(cooc_c)}
        print(f"    {c}: global F1={g_pf['f1']:.3f} (P={g_pf['prec']:.3f} R={g_pf['rec']:.3f}) | "
              f"group F1={c_pf['f1']:.3f} (P={c_pf['prec']:.3f} R={c_pf['rec']:.3f})")
    report["B_per_camera_f1"] = f1_compare

    # ================= C. test 集选择器对照 =================
    print("\n== C. test 选择器对照（known-restricted） ==")
    c_res = {"global_all": eval_test(gsel, TE_keys, TE_Va, TE_Vb), "per_camera": {}}
    print(f"  global/all: cov={c_res['global_all']['coverage']:.3f} "
          f"same={c_res['global_all']['same_known']:.3f} coll={c_res['global_all']['coll_known']:.3f} "
          f"n={c_res['global_all']['n']}")
    for c in CAMS:
        eg = eval_test(gsel, TE_keys, TE_Va, TE_Vb, cam=c)
        ec = eval_test(group_sels[c], TE_keys, TE_Va, TE_Vb, cam=c)
        c_res["per_camera"][c] = {"global": eg, "group": ec}
        print(f"  [{c}] n={eg['n']:3d} | global: cov={f3(eg['coverage'])} known={eg['n_known']:3d} "
              f"same={f3(eg['same_known'])} coll={f3(eg['coll_known'])} | group: cov={f3(ec['coverage'])} "
              f"known={ec['n_known']:3d} same={f3(ec['same_known'])} coll={f3(ec['coll_known'])}")
    report["C_test_selector"] = c_res

    # ---- C2. kmeans 初始化稳健性（全局 vs 各组，多 seed） ----
    print("\n== C2. kmeans 初始化稳健性（seed 扫描） ==")
    seeds = [0, 7, 42, 123, 2026]
    def kmeans_f1_over_seeds(V, Va, Vb, keys, cooc, K):
        out = []
        for sd in seeds:
            lab, _ = cluster_kmeans(V, K, seed=sd)
            if lab is None:
                out.append(None)
                continue
            protos, _ = protos_from_labels(V, lab)
            ia, _ = assign(Va, protos)
            ib, _ = assign(Vb, protos)
            ia_map = {k: int(ia[i]) for i, k in enumerate(keys)}
            ib_map = {k: int(ib[i]) for i, k in enumerate(keys)}
            pf = pairwise_f1_cooc(ia_map, ib_map, cooc, keys)
            out.append(float(pf["f1"]))
        return out
    robust = {}
    gK = ginfo.get("K", 60)
    rows = [("global", TR_V, TR_Va, TR_Vb, TR_keys, cooc_all, gK)]
    for c in CAMS:
        m = tr_cam == c
        kc = [k for k, mm in zip(TR_keys, m) if mm]
        cooc_c = [(u, v) for (u, v) in cooc_all if cam_of(u[0]) == c]
        rows.append((c, TR_V[m], TR_Va[m], TR_Vb[m], kc, cooc_c, group_info[c].get("K", 60)))
    for name, V_, Va_, Vb_, k_, cc_, K_ in rows:
        f1s = kmeans_f1_over_seeds(V_, Va_, Vb_, k_, cc_, K_)
        vals = [x for x in f1s if x is not None]
        robust[name] = {"K": K_, "seeds": seeds, "f1s": f1s,
                        "mean": float(np.mean(vals)) if vals else None,
                        "std": float(np.std(vals)) if vals else None,
                        "min": float(np.min(vals)) if vals else None,
                        "max": float(np.max(vals)) if vals else None}
        print(f"  {name:7s} K={K_:2d} F1 seeds={[None if x is None else round(x,3) for x in f1s]} "
              f"mean={np.mean(vals):.3f} std={np.std(vals):.3f}")
    report["C2_kmeans_seed_robustness"] = robust

    # ---- C3. 同参数对照：mutual-knn (k=5, thr=0.3) 全局 vs 分组（per-camera 子集） ----
    print("\n== C3. 同参数对照 mutual-knn(k5,thr0.3)：per-camera 子集 F1 ==")
    lab_g, _ = cluster_mutual_knn(TR_V, 5, 0.3)
    protos_g, _ = protos_from_labels(TR_V, lab_g)
    ga_, _ = assign(TR_Va, protos_g)
    gb_, _ = assign(TR_Vb, protos_g)
    ga_map = {k: int(ga_[i]) for i, k in enumerate(TR_keys)}
    gb_map = {k: int(gb_[i]) for i, k in enumerate(TR_keys)}
    c3 = {}
    for c in CAMS:
        m = tr_cam == c
        kc = [k for k, mm in zip(TR_keys, m) if mm]
        cooc_c = [(u, v) for (u, v) in cooc_all if cam_of(u[0]) == c]
        pf_global = pairwise_f1_cooc(ga_map, gb_map, cooc_c, kc)
        lab_c, _ = cluster_mutual_knn(TR_V[m], 5, 0.3)
        protos_c, _ = protos_from_labels(TR_V[m], lab_c)
        ia_, _ = assign(TR_Va[m], protos_c)
        ib_, _ = assign(TR_Vb[m], protos_c)
        ia_map = {k: int(ia_[i]) for i, k in enumerate(kc)}
        ib_map = {k: int(ib_[i]) for i, k in enumerate(kc)}
        pf_group = pairwise_f1_cooc(ia_map, ib_map, cooc_c, kc)
        c3[c] = {"global_mkn_f1": pf_global["f1"], "group_mkn_f1": pf_group["f1"],
                 "n_clusters_global_labeled": int(len(set(ga_map[k] for k in kc))),
                 "n_clusters_group": int(len(set(lab_c)))}
        print(f"  {c}: global-mkn F1={pf_global['f1']:.3f} | group-mkn F1={pf_group['f1']:.3f} "
              f"(group ncl={len(set(lab_c))})")
    report["C3_same_param_mkn"] = c3

    # ================= D. 跨段身份保持（同机/跨机拆分） =================
    print("\n== D. 跨段身份保持（mutual-NN 对） ==")
    S = TE_V @ TR_V.T
    t2r = S.argmax(1)
    S2 = TR_V @ TE_V.T
    r2t = S2.argmax(1)
    mutual = [(ti, ri) for ti, ri in enumerate(t2r) if r2t[ri] == ti]
    print(f"  mutual-NN pairs: {len(mutual)}")
    # 全局分配
    g_ia_te, _ = gsel.predict(TE_V)
    g_ib_tr, _ = gsel.predict(TR_V)
    # 分同机/跨机
    same_cam_pairs = [(ti, ri) for ti, ri in mutual if te_cam[ti] == tr_cam[ri]]
    cross_cam_pairs = [(ti, ri) for ti, ri in mutual if te_cam[ti] != tr_cam[ri]]
    def keep_rate(assign_te, assign_tr, pairs, need_known=True):
        kn = [(ti, ri) for ti, ri in pairs
              if (not need_known) or (assign_te[ti] >= 0 and assign_tr[ri] >= 0)]
        if not kn:
            return {"n": 0, "rate": None}
        keep = sum(1 for ti, ri in kn if assign_te[ti] == assign_tr[ri])
        return {"n": len(kn), "rate": keep / len(kn)}
    # 三方案
    plans = {}
    # ① 全局
    plans["global"] = {
        "same_cam": keep_rate(g_ia_te, g_ib_tr, same_cam_pairs),
        "cross_cam": keep_rate(g_ia_te, g_ib_tr, cross_cam_pairs),
    }
    # ② 组专用（跨机不可比——只能给同机）
    OFF = {c: i * 100000 for i, c in enumerate(CAMS)}  # 拼接用的组偏移
    gte_grp = np.full(len(TE_keys), -1)
    gtr_grp = np.full(len(TR_keys), -1)
    gte_stic, gtr_stic = np.full(len(TE_keys), -1), np.full(len(TR_keys), -1)
    for c in CAMS:
        selc = group_sels[c]
        mt = te_cam == c
        if mt.any():
            ids_t, _ = selc.predict(TE_V[mt])
            gte_grp[mt] = ids_t
            gte_stic[mt] = np.where(np.asarray(ids_t) >= 0, np.asarray(ids_t) + OFF[c], -1)
        mr = tr_cam == c
        if mr.any():
            ids_r, _ = selc.predict(TR_V[mr])
            gtr_grp[mr] = ids_r
            gtr_stic[mr] = np.where(np.asarray(ids_r) >= 0, np.asarray(ids_r) + OFF[c], -1)
    plans["group_only"] = {
        "same_cam": keep_rate(gte_grp, gtr_grp, same_cam_pairs),
        "cross_cam": {"n": len(cross_cam_pairs), "rate": None,
                      "note": "组选择器 ID 空间互不相通，跨机对的‘相等性’无定义（数值碰撞无意义）"},
    }
    plans["stitched"] = {
        "same_cam": keep_rate(gte_stic, gtr_stic, same_cam_pairs),
        "cross_cam": {"n": len(cross_cam_pairs), "rate": 0.0,
                      "note": "物理拼接不改跨机不可通约性"},
    }
    report["D_crossseg"] = {"n_mutual": len(mutual),
                            "n_same_cam": len(same_cam_pairs),
                            "n_cross_cam": len(cross_cam_pairs), "plans": plans}

    # ---- D2. 共同可判子集上的公平对比（global known AND group known） ----
    def fair_compare(pairs):
        both = [(ti, ri) for ti, ri in pairs
                if g_ia_te[ti] >= 0 and g_ib_tr[ri] >= 0
                and gte_grp[ti] >= 0 and gtr_grp[ri] >= 0]
        if not both:
            return {"n": 0, "global_rate": None, "group_rate": None}
        g_keep = sum(1 for ti, ri in both if g_ia_te[ti] == g_ib_tr[ri]) / len(both)
        c_keep = sum(1 for ti, ri in both if gte_grp[ti] == gtr_grp[ri]) / len(both)
        return {"n": len(both), "global_rate": float(g_keep), "group_rate": float(c_keep)}
    fair = {"same_cam": fair_compare(same_cam_pairs),
            "cross_cam": fair_compare(cross_cam_pairs)}
    report["D_crossseg"]["fair_common_subset"] = fair
    fs_ = fair["same_cam"]
    print(f"  [fair] same-cam common subset n={fs_['n']}: global={fs_['global_rate']} vs "
          f"group={fs_['group_rate']}")

    # ================= F. 折中方案：全局聚类 + 按摄像头校准阈值 =================
    print("\n== F. 折中：全局 protos + per-camera thr（保持全局 ID 空间） ==")
    TR_kidx = {k: i for i, k in enumerate(TR_keys)}
    thr_cam = {}
    for c in CAMS:
        m = tr_cam == c
        pos_c = np.sum(TR_Va[m] * TR_Vb[m], axis=1)
        p5 = float(np.percentile(pos_c, 5))
        cs_c = [float(TR_V[TR_kidx[u]] @ TR_V[TR_kidx[v]])
                for (u, v) in cooc_all if cam_of(u[0]) == c]
        p99 = float(np.percentile(cs_c, 99)) if cs_c else None
        thr_cam[c] = max(0.2, (p5 + p99) / 2) if p99 else 0.3
        print(f"  thr[{c}]={thr_cam[c]:.4f} (pos_p5={p5:.3f} cooc_p99="
              f"{None if p99 is None else round(p99,3)} n_cooc={len(cs_c)})")
    # test 打分（全局 protos）
    SA_all = gsel.sims_all(TE_Va)
    SB_all = gsel.sims_all(TE_Vb)
    idxa_all = SA_all.argmax(1)
    idxb_all = SB_all.argmax(1)
    sa_all = SA_all[np.arange(len(TE_keys)), idxa_all]
    sb_all = SB_all[np.arange(len(TE_keys)), idxb_all]
    ia_f = np.full(len(TE_keys), -1)
    ib_f = np.full(len(TE_keys), -1)
    for c in CAMS:
        m = te_cam == c
        t = thr_cam[c]
        ia_f[m] = np.where(sa_all[m] >= t, idxa_all[m], -1)
        ib_f[m] = np.where(sb_all[m] >= t, idxb_all[m], -1)
    f_res = {}
    for c in CAMS:
        m = te_cam == c
        N = int(m.sum())
        ia, ib = ia_f[m], ib_f[m]
        kn = (ia >= 0) & (ib >= 0)
        cov = float(kn.mean())
        same = float((ia[kn] == ib[kn]).mean()) if kn.sum() else None
        rng = np.random.default_rng(3)
        l = rng.integers(0, N, 20000)
        r = rng.integers(0, N, 20000)
        kp = l != r
        il, ir = ia[l[kp]], ib[r[kp]]
        kn2 = (il >= 0) & (ir >= 0)
        coll = float((il[kn2] == ir[kn2]).mean()) if kn2.sum() else None
        f_res[c] = {"coverage": cov, "n_known": int(kn.sum()), "same_known": same,
                    "coll_known": coll, "thr": thr_cam[c]}
        print(f"  [{c}] n={N:3d} cov={f3(cov)} known={int(kn.sum()):3d} same={f3(same)} coll={f3(coll)}")
    report["F_global_percam_thr"] = {"thr_cam": thr_cam, "test": f_res}
    # F2. 同机 keep 对照（global / group / F），统一 V 口径 + 共同可判子集
    SV_te = gsel.sims_all(TE_V)
    SV_tr = gsel.sims_all(TR_V)
    f_ia_te = np.full(len(TE_keys), -1)
    f_ia_tr = np.full(len(TR_keys), -1)
    for c in CAMS:
        t = thr_cam[c]
        mt = te_cam == c
        if mt.any():
            sub = SV_te[mt]
            mx = sub.argmax(1)
            sc = sub[np.arange(len(mx)), mx]
            f_ia_te[mt] = np.where(sc >= t, mx, -1)
        mr = tr_cam == c
        if mr.any():
            sub = SV_tr[mr]
            mx = sub.argmax(1)
            sc = sub[np.arange(len(mx)), mx]
            f_ia_tr[mr] = np.where(sc >= t, mx, -1)
    both = [(ti, ri) for ti, ri in same_cam_pairs
            if g_ia_te[ti] >= 0 and g_ib_tr[ri] >= 0
            and gte_grp[ti] >= 0 and gtr_grp[ri] >= 0
            and f_ia_te[ti] >= 0 and f_ia_tr[ri] >= 0]
    if both:
        gk = sum(1 for ti, ri in both if g_ia_te[ti] == g_ib_tr[ri]) / len(both)
        ck = sum(1 for ti, ri in both if gte_grp[ti] == gtr_grp[ri]) / len(both)
        fk = sum(1 for ti, ri in both if f_ia_te[ti] == f_ia_tr[ri]) / len(both)
    else:
        gk = ck = fk = None
    report["F2_quad_keep"] = {"n": len(both), "global": gk, "group": ck,
                              "global_percam_thr": fk}
    print(f"  [F2] same-cam quad-common n={len(both)}: global={f3(gk)} group={f3(ck)} "
          f"global+percam_thr={f3(fk)}")
    for name, p in plans.items():
        sc = p["same_cam"]
        print(f"  {name:11s} same-cam keep: {sc['rate'] if sc['rate'] is None else round(sc['rate'],4)}"
              f" (n={sc['n']}) | cross-cam keep: "
              f"{p['cross_cam']['rate'] if p['cross_cam']['rate'] is not None else None} "
              f"(n={p['cross_cam']['n']})")

    # ================= E. 组间原型可合并性诊断 =================
    print("\n== E. 组间原型诊断（跨组最近邻 cos） ==")
    allp, p_src = [], []
    for c in CAMS:
        allp.append(group_protos[c])
        p_src += [c] * len(group_protos[c])
    allp = np.vstack(allp)
    p_src = np.array(p_src)
    Sp = allp @ allp.T
    np.fill_diagonal(Sp, -1)
    nn_idx = Sp.argmax(1)
    cross_nn, within_nn = [], []
    for i in range(len(allp)):
        j = nn_idx[i]
        (cross_nn if p_src[i] != p_src[j] else within_nn).append(Sp[i, j])
    e_res = {
        "n_protos_total": int(len(allp)),
        "within_group_top1_cos": {"mean": float(np.mean(within_nn)) if within_nn else None,
                                  "p50": float(np.percentile(within_nn, 50)) if within_nn else None},
        "cross_group_top1_cos": {"mean": float(np.mean(cross_nn)) if cross_nn else None,
                                 "p50": float(np.percentile(cross_nn, 50)) if cross_nn else None,
                                 "p95": float(np.percentile(cross_nn, 95)) if cross_nn else None},
        "n_within_nn": len(within_nn), "n_cross_nn": len(cross_nn),
    }
    report["E_inter_group"] = e_res
    print(f"  protos total={len(allp)}; within-group top1 cos mean="
          f"{e_res['within_group_top1_cos']['mean']:.3f} | cross-group top1 cos mean="
          f"{e_res['cross_group_top1_cos']['mean']:.3f} p95={e_res['cross_group_top1_cos']['p95']:.3f}")

    # ================= 图 =================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        x = np.arange(len(CAMS))
        w = 0.35
        nn = lambda v: np.nan if v is None else v  # noqa: E731
        g_same = [nn(c_res["per_camera"][c]["global"]["same_known"]) for c in CAMS]
        p_same = [nn(c_res["per_camera"][c]["group"]["same_known"]) for c in CAMS]
        g_coll = [nn(c_res["per_camera"][c]["global"]["coll_known"]) for c in CAMS]
        p_coll = [nn(c_res["per_camera"][c]["group"]["coll_known"]) for c in CAMS]
        axes[0].bar(x - w / 2, g_same, w, label="global", color="#4C78A8")
        axes[0].bar(x + w / 2, p_same, w, label="per-camera", color="#F58518")
        axes[0].set_xticks(x); axes[0].set_xticklabels(CAMS)
        axes[0].set_title("test same-known (split-half agreement)")
        axes[0].set_ylim(0, 1.05); axes[0].legend()
        axes[1].bar(x - w / 2, g_coll, w, label="global", color="#4C78A8")
        axes[1].bar(x + w / 2, p_coll, w, label="per-camera", color="#F58518")
        axes[1].set_xticks(x); axes[1].set_xticklabels(CAMS)
        axes[1].set_title("test collision-known (lower=better)")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "per_camera_metrics.png"), dpi=140)
        # 分布图
        fig2, ax = plt.subplots(figsize=(7, 4))
        ax.hist(same_cos, bins=60, alpha=0.6, density=True, label="same-camera random pairs")
        ax.hist(cross_cos, bins=60, alpha=0.6, density=True, label="cross-camera random pairs")
        ax.set_title("cos similarity (train V, random pairs)")
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(a.out, "cos_same_vs_cross_cam.png"), dpi=140)
        print("[fig] saved 2 figures")
    except Exception as e:
        print("[fig] skip:", e)

    json.dump(report, open(os.path.join(a.out, "study.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print(f"\n[done] -> {os.path.join(a.out, 'study.json')}")


if __name__ == "__main__":
    main()
