#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器升级实验（2026-10-14）：折中方案（per-cam 阈值）落地验证 + PCLHD 多原型 + N-ULC 软标签借鉴。

背景：上一轮研究（docs/camgroup-selector-study.md）结论——
  硬隔离分组无系统收益；全局模型 + 局部校准才是正确用法。
本实验把结论推进为可落地方案，并借鉴两种近邻技术（先实验、有效再确认）：

  F（折中）  全局聚类不动，thr_unknown 按摄像头分别校准
  P（PCLHD）多原型：每簇分裂为多个子原型（信息熵更高），判定统计量 = 子原型 top-1
  N（N-ULC）近邻软标签：判定统计量 = 簇内近邻 top-k 均值（k>1 即软化；无需单点原型）

变体（同一基聚类，只改原型库表示与判定统计量）：
  single   单原型（现状基线）
  multi2/4/8  每簇 ≤M 子原型（kmeans 分裂，簇级 max 聚合）
  soft3/5  全向量近邻 + 簇内 top-3/5 均值（近邻软标签）

评估（test 348 轨，known-restricted 口径，与 camgroup_study 完全一致）：
  ① 工作点：legacy thr（向量级中点，观测判定统计量的"膨胀/收缩"效应）
             / selfcal thr（变体自身统计量中点，per-cam 与全局）
     → coverage / same_known / coll_known（overall + per-camera）
  ② 曲线：统一 thr 扫描 → 固定 coverage {0.2,0.4,0.6,0.8} 下 same/coll
     （排除 thr 校准口径差异的公平对比——比较的是判定统计量本身的质量）
  ③ 公平子集：所有变体共同 known 的 test 子集上 same 对照
  ④ 跨段保持：mutual-NN 配对（test↔train）的簇一致率（公共可判对子集）

产物：{out}/study.json + coverage_same_curve.png
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector import _norm  # noqa: E402
from train_selector_cluster import cooc_pairs  # noqa: E402
from camgroup_study import (load_segs, select_best, cam_of, CAMS,  # noqa: E402
                            TRAIN_SEGS, TEST_SEGS, JDL)

ROOT = os.path.dirname(HERE)
COV_TARGETS = (0.2, 0.4, 0.6, 0.8)
MIN_CLUSTER = 2


def build_variants(TR_V, lab, keep_ids, M_list=(2, 4, 8, 16), soft_ks=(3, 5)):
    """基于基聚类 lab（全量）构造各变体 (subprotos, sub2cluster, topk)。"""
    from scipy.cluster.vq import kmeans2
    id2slot = {c: i for i, c in enumerate(keep_ids)}
    C = len(keep_ids)
    variants = {}

    # single：簇均值原型
    protos = np.stack([_norm(TR_V[lab == c].mean(0, keepdims=True))[0] for c in keep_ids])
    variants["single"] = {"subprotos": protos.astype(np.float32),
                          "sub2cluster": np.arange(C), "topk": 1}

    # multi：簇内 kmeans 分裂
    for M in M_list:
        sp, s2c = [], []
        for c in keep_ids:
            Vc = TR_V[lab == c]
            n = len(Vc)
            mc = 1 if n < 4 else min(M, n // 2)
            if mc == 1:
                cen = [_norm(Vc.mean(0, keepdims=True))[0]]
            else:
                cen_k, _ = kmeans2(Vc, mc, iter=40, minit="++", seed=42)
                cen = [_norm(x) for x in cen_k]
            sp += cen
            s2c += [id2slot[c]] * mc
        variants[f"multi{M}"] = {"subprotos": np.asarray(sp, np.float32),
                                 "sub2cluster": np.asarray(s2c), "topk": 1}

    # soft：全向量近邻（簇内 top-k 均值）——N-ULC 软标签直用
    mask = np.isin(lab, keep_ids)
    sp = TR_V[mask].astype(np.float32)
    s2c = np.asarray([id2slot[c] for c in lab[mask]])
    for k in soft_ks:
        variants[f"soft{k}"] = {"subprotos": sp, "sub2cluster": s2c, "topk": k}

    # multi8t3：multi8 子原型 + 簇级 top-3 均值（软聚合对照）
    mv8 = variants.get("multi8")
    if mv8 is not None:
        variants["multi8t3"] = {"subprotos": mv8["subprotos"],
                                "sub2cluster": mv8["sub2cluster"], "topk": 3}
    return variants, C


def score_matrix(Vq, var, n_clusters):
    """Vq @ subprotos → 簇级聚合（topk=1: max；topk>1: 簇内 top-k 均值）。"""
    S = Vq @ var["subprotos"].T
    topk = var["topk"]
    if topk == 1:
        sc = np.empty((len(Vq), n_clusters), np.float32)
        for c in range(n_clusters):
            cols = np.where(var["sub2cluster"] == c)[0]
            sc[:, c] = S[:, cols].max(1)
        return sc
    sc = np.empty((len(Vq), n_clusters), np.float32)
    for c in range(n_clusters):
        cols = np.where(var["sub2cluster"] == c)[0]
        kk = min(topk, len(cols))
        part = np.sort(S[:, cols], axis=1)[:, -kk:]
        sc[:, c] = part.mean(1)
    return sc


def wp_thr(sa, sb, cooc, TR_keys, kidx, cam_mask=None):
    """selfcal 中点：pos=min(sa,sb) p5；neg=cooc 双侧统计量池 p99。"""
    idx = np.arange(len(sa))
    if cam_mask is not None:
        idx = idx[cam_mask]
    pos = np.minimum(sa, sb)[idx]
    if len(pos) == 0:
        return None
    p5 = float(np.percentile(pos, 5))
    cs = []
    for u, v in cooc:
        iu, iv = kidx.get(u), kidx.get(v)
        if iu is None or iv is None:
            continue
        if cam_mask is not None and not cam_mask[iu]:
            continue
        cs += [float(sa[iu]), float(sb[iv]), float(sa[iv]), float(sb[iu])]
    if not cs:
        return None
    p99 = float(np.percentile(cs, 99))
    return {"thr": max(0.2, (p5 + p99) / 2), "pos_p5": p5, "neg_p99": p99,
            "n_pos": int(len(pos)), "n_neg": len(cs)}


def legacy_thr_vec(TR_Va, TR_Vb, TR_V, TR_keys, cooc, kidx, cam_mask=None):
    """v1/F 口径：pos=Va·Vb p5；neg=cooc 向量对 p99；中点。"""
    pos = np.sum(TR_Va * TR_Vb, axis=1)
    if cam_mask is not None:
        pos = pos[cam_mask]
    p5 = float(np.percentile(pos, 5))
    cs = []
    for u, v in cooc:
        iu, iv = kidx.get(u), kidx.get(v)
        if iu is None or iv is None:
            continue
        if cam_mask is not None and not cam_mask[iu]:
            continue
        cs.append(float(TR_V[iu] @ TR_V[iv]))
    if not cs:
        return None
    p99 = float(np.percentile(cs, 99))
    return {"thr": max(0.2, (p5 + p99) / 2), "pos_p5": p5, "neg_p99": p99,
            "n_cooc": len(cs)}


def eval_wp(sa, sb, ca, cb, thr_a, thr_b, cam_arr, seed=3, n_rand=30000):
    """指定 thr（标量或 per-sample 数组）下的 known-restricted 评估。"""
    thr_a = np.full(len(sa), thr_a) if np.isscalar(thr_a) else thr_a
    thr_b = np.full(len(sb), thr_b) if np.isscalar(thr_b) else thr_b
    ia = np.where(sa >= thr_a, ca, -1)
    ib = np.where(sb >= thr_b, cb, -1)
    kn = (ia >= 0) & (ib >= 0)
    cov = float(kn.mean())
    same = float((ia[kn] == ib[kn]).mean()) if kn.sum() else None
    rng = np.random.default_rng(seed)
    L = rng.integers(0, len(sa), n_rand)
    R = rng.integers(0, len(sb), n_rand)
    kp = L != R
    il, ir = ia[L[kp]], ib[R[kp]]
    kn2 = (il >= 0) & (ir >= 0)
    coll = float((il[kn2] == ir[kn2]).mean()) if kn2.sum() else None
    out = {"n": len(sa), "coverage": cov, "n_known": int(kn.sum()),
           "same_known": same, "coll_known": coll}
    # per-camera
    pc = {}
    for c in CAMS:
        m = cam_arr == c
        if not m.any():
            continue
        iac, ibc = ia[m], ib[m]
        knc = (iac >= 0) & (ibc >= 0)
        pc[c] = {"n": int(m.sum()), "coverage": float(knc.mean()),
                 "same_known": float((iac[knc] == ibc[knc]).mean()) if knc.sum() else None,
                 "coll_known": None}
    out["per_cam"] = pc
    return out, {"ia": ia, "ib": ib}


def curve_fixed_cov(sa, sb, ca, cb, cam_arr, seed=3, n_rand=30000,
                    targets=COV_TARGETS):
    """统一 thr 网格扫描 → coverage 网格插值（粗粒度 0.001，足够近）。"""
    rng = np.random.default_rng(seed)
    L = rng.integers(0, len(sa), n_rand)
    R = rng.integers(0, len(sb), n_rand)
    kp = L != R
    Ll, Rl = L[kp], R[kp]
    grid = np.arange(0.40, 0.9995, 0.001)
    covs, sames, colls = [], [], []
    for t in grid:
        ia = np.where(sa >= t, ca, -1)
        ib = np.where(sb >= t, cb, -1)
        kn = (ia >= 0) & (ib >= 0)
        covs.append(kn.mean())
        sames.append((ia[kn] == ib[kn]).mean() if kn.sum() else np.nan)
        il, ir = ia[Ll], ib[Rl]
        kn2 = (il >= 0) & (ir >= 0)
        colls.append((il[kn2] == ir[kn2]).mean() if kn2.sum() else np.nan)
    covs, sames, colls = np.asarray(covs), np.asarray(sames), np.asarray(colls)
    # cov 随 thr 单调降 → 插值到目标 cov
    o = np.argsort(covs)
    c_s, s_s, k_s = covs[o], sames[o], colls[o]
    res = {}
    for tg in targets:
        if tg < c_s[0] or tg > c_s[-1]:
            res[f"{tg:.1f}"] = None
            continue
        res[f"{tg:.1f}"] = {"same": float(np.interp(tg, c_s, s_s)),
                            "coll": float(np.interp(tg, c_s, k_s))}
    return res, {"grid_cov": covs.tolist(), "grid_same": sames.tolist(),
                 "grid_coll": colls.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "output_crosscam",
                                                  "analysis", "selector_1012", "upgrade"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ===== 载入 =====
    TR_keys, TR_Va, TR_Vb, TR_V = load_segs(TRAIN_SEGS)
    TE_keys, TE_Va, TE_Vb, TE_V = load_segs(TEST_SEGS)
    tr_cam = np.array([cam_of(k[0]) for k in TR_keys])
    te_cam = np.array([cam_of(k[0]) for k in TE_keys])
    print(f"[load] train={len(TR_keys)} test={len(TE_keys)}")

    keys_set = set(TR_keys)
    cooc_all = []
    for s in TRAIN_SEGS:
        for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), keys_set, s, 5):
            cooc_all.append(((s, u), (s, v)))
    kidx = {k: i for i, k in enumerate(TR_keys)}
    print(f"[cooc] {len(cooc_all)} pairs")

    # ===== 基聚类 =====
    # 注意：kmeans 对初始化/环境敏感（上轮已确认，波动 0.782-0.850），
    # mutual-knn 确定性强，且现部署 selector.npz 即由它建立——固定用 mutual-knn 保持可比。
    _, results = select_best(TR_V, TR_Va, TR_Vb, TR_keys, cooc_all)
    best = [d for d in results if d["method"] == "mutual-knn"][0]
    lab = best["_lab"]
    print(f"[base] {best['method']} k={best.get('k')} thr={best.get('thr')} "
          f"ncl={best['n_clusters']} F1={best['f1']:.4f}")
    ids = sorted(set(lab))
    sizes = np.array([(lab == c).sum() for c in ids])
    keep_ids = [c for c, s in zip(ids, sizes) if s >= MIN_CLUSTER]
    n_removed = len(ids) - len(keep_ids)
    print(f"[keep] {len(keep_ids)} clusters (removed small: {n_removed}, sizes>=2)")

    variants, C = build_variants(TR_V, lab, keep_ids)
    for name, v in variants.items():
        print(f"[variant] {name:8s} subprotos={v['subprotos'].shape[0]:4d} topk={v['topk']}")

    # ===== 预计算 score 矩阵（train a/b/v, test a/b/v） =====
    sc_cache = {}
    for name, v in variants.items():
        sc_cache[name] = {
            "tr_a": score_matrix(TR_Va, v, C), "tr_b": score_matrix(TR_Vb, v, C),
            "tr_v": score_matrix(TR_V, v, C),
            "te_a": score_matrix(TE_Va, v, C), "te_b": score_matrix(TE_Vb, v, C),
            "te_v": score_matrix(TE_V, v, C)}

    # ===== thr 校准 =====
    thr_legacy = legacy_thr_vec(TR_Va, TR_Vb, TR_V, TR_keys, cooc_all, kidx)
    thr_legacy_cam = {}
    for c in CAMS:
        thr_legacy_cam[c] = legacy_thr_vec(TR_Va, TR_Vb, TR_V, TR_keys, cooc_all,
                                           kidx, cam_mask=(tr_cam == c))
    print(f"[thr-legacy] global={thr_legacy['thr']:.4f} | "
          + " ".join(f"{c}:{thr_legacy_cam[c]['thr']:.4f}" for c in CAMS))

    thr_self = {}
    for name in variants:
        sa_tr = sc_cache[name]["tr_a"].max(1)
        sb_tr = sc_cache[name]["tr_b"].max(1)
        g = wp_thr(sa_tr, sb_tr, cooc_all, TR_keys, kidx)
        cm = {}
        for c in CAMS:
            cm[c] = wp_thr(sa_tr, sb_tr, cooc_all, TR_keys, kidx,
                           cam_mask=(tr_cam == c))
        thr_self[name] = {"global": g, "per_cam": cm}
        print(f"[thr-selfcal] {name:8s} global={g['thr']:.4f} | "
              + " ".join(f"{c}:{cm[c]['thr']:.4f}" for c in CAMS))

    # ===== ① 工作点评估 =====
    report = {"base_cluster": {k: v for k, v in best.items() if k != "_lab"},
              "n_keep": len(keep_ids), "n_small_removed": n_removed,
              "thr_legacy": {"global": thr_legacy, "per_cam": thr_legacy_cam},
              "thr_selfcal": {n: {"global": thr_self[n]["global"],
                                  "per_cam": thr_self[n]["per_cam"]} for n in variants},
              "workpoints": {}, "curves": {}, "fair_subset": {}, "crossseg": {}}
    ia_te_all = {}
    for name in variants:
        sa, ca = sc_cache[name]["te_a"].max(1), sc_cache[name]["te_a"].argmax(1)
        sb, cb = sc_cache[name]["te_b"].max(1), sc_cache[name]["te_b"].argmax(1)
        ea = np.array([thr_legacy_cam[c]["thr"] for c in te_cam])
        eb = np.array([thr_self[name]["per_cam"][c]["thr"] for c in te_cam])
        w = {}
        w["legacy_global"] = eval_wp(sa, sb, ca, cb, thr_legacy["thr"],
                                     thr_legacy["thr"], te_cam)[0]
        w["legacy_percam"] = eval_wp(sa, sb, ca, cb, ea, ea, te_cam)[0]
        w["selfcal_global"] = eval_wp(sa, sb, ca, cb, thr_self[name]["global"]["thr"],
                                      thr_self[name]["global"]["thr"], te_cam)[0]
        res, alloc = eval_wp(sa, sb, ca, cb, eb, eb, te_cam)
        w["selfcal_percam"] = res
        ia_te_all[name] = alloc
        report["workpoints"][name] = w
        r = w["selfcal_percam"]
        print(f"[wp] {name:8s} selfcal-percam: cov={r['coverage']:.3f} "
              f"same={r['same_known'] if r['same_known'] is None else round(r['same_known'],3)} "
              f"coll={r['coll_known'] if r['coll_known'] is None else round(r['coll_known'],3)}")
        rl = w["legacy_percam"]
        print(f"[wp] {name:8s} legacy-percam:   cov={rl['coverage']:.3f} "
              f"same={rl['same_known'] if rl['same_known'] is None else round(rl['same_known'],3)} "
              f"coll={rl['coll_known'] if rl['coll_known'] is None else round(rl['coll_known'],3)}")

    # V 向量分配（供跨段保持用；与 crossseg.py 的 V 口径一致）
    alloc_v = {}
    for name in variants:
        sv = sc_cache[name]["te_v"].max(1)
        cv = sc_cache[name]["te_v"].argmax(1)
        thr_v = np.array([thr_self[name]["per_cam"][c]["thr"] for c in te_cam])
        ia_v = np.where(sv >= thr_v, cv, -1)
        st = sc_cache[name]["tr_v"].max(1)
        ct = sc_cache[name]["tr_v"].argmax(1)
        thr_t = np.array([thr_self[name]["per_cam"][c]["thr"] for c in tr_cam])
        ib_v = np.where(st >= thr_t, ct, -1)
        alloc_v[name] = {"te": ia_v, "tr": ib_v}

    # ===== ② 曲线 =====
    for name in variants:
        sa, ca = sc_cache[name]["te_a"].max(1), sc_cache[name]["te_a"].argmax(1)
        sb, cb = sc_cache[name]["te_b"].max(1), sc_cache[name]["te_b"].argmax(1)
        res, grids = curve_fixed_cov(sa, sb, ca, cb, te_cam)
        report["curves"][name] = res
        report.setdefault("_grids", {})[name] = grids
        print(f"[curve] {name:8s} " + " ".join(
            f"cov={k}: same={v['same']:.3f} coll={v['coll']:.3f}" if v else f"cov={k}: n/a"
            for k, v in res.items()))

    # ===== ③ 公平子集 =====
    common = np.ones(len(TE_keys), bool)
    for name in variants:
        ia = ia_te_all[name]["ia"]
        ib = ia_te_all[name]["ib"]
        common &= (ia >= 0) & (ib >= 0)
    fair = {"n": int(common.sum())}
    for name in variants:
        ia = ia_te_all[name]["ia"][common]
        ib = ia_te_all[name]["ib"][common]
        fair[name] = {"same": float((ia == ib).mean()) if common.sum() else None}
    report["fair_subset"] = fair
    print(f"[fair] common known subset n={fair['n']}: "
          + " ".join(f"{n}={fair[n]['same']:.3f}" if fair[n]['same'] is not None
                     else f"{n}=None" for n in variants))

    # ===== ④ 跨段保持（mutual-NN 配对） =====
    S = TE_V @ TR_V.T
    t2r = S.argmax(1)
    S2 = TR_V @ TE_V.T
    r2t = S2.argmax(1)
    mutual = [(ti, ri) for ti, ri in enumerate(t2r) if r2t[ri] == ti]
    sims = np.array([S[ti, ri] for ti, ri in mutual])
    pairs = [(ti, ri) for ti, ri in mutual if S[ti, ri] >= 0.85]
    print(f"[crossseg] mutual={len(mutual)} pairs(sim>=0.85)={len(pairs)}")
    cs_report = {"n_mutual": len(mutual), "n_pairs85": len(pairs), "variants": {}}
    decided = {}
    for name in variants:
        ia = alloc_v[name]["te"]
        ib = alloc_v[name]["tr"]
        kn = [(ti, ri) for ti, ri in pairs if ia[ti] >= 0 and ib[ri] >= 0]
        keep = sum(1 for ti, ri in kn if ia[ti] == ib[ri])
        rate = keep / len(kn) if kn else None
        decided[name] = set(kn)
        cs_report["variants"][name] = {"n_decided": len(kn), "keep_rate": rate,
                                       "coverage": len(kn) / len(pairs) if pairs else None}
        print(f"[crossseg] {name:8s} decided={len(kn)}/{len(pairs)} keep={rate}")
    common_pairs = None
    for name in variants:
        common_pairs = decided[name] if common_pairs is None else (common_pairs & decided[name])
    if common_pairs:
        cp = sorted(common_pairs)
        for name in variants:
            ia = alloc_v[name]["te"]
            ib = alloc_v[name]["tr"]
            keep = sum(1 for ti, ri in cp if ia[ti] == ib[ri])
            cs_report["variants"][name]["keep_on_common"] = keep / len(cp)
        cs_report["n_common_pairs"] = len(cp)
        print(f"[crossseg-fair] common pairs n={len(cp)}: "
              + " ".join(f"{n}={cs_report['variants'][n]['keep_on_common']:.3f}"
                         for n in variants))

    # ===== 图 =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        colors = {"single": "#4C78A8", "multi2": "#F58518", "multi4": "#54A24B",
                  "multi8": "#E45756", "multi16": "#8C564B", "multi8t3": "#17BECF",
                  "soft3": "#B279A2", "soft5": "#9D755D"}
        for name in variants:
            g = report["_grids"][name]
            cov = np.asarray(g["grid_cov"])
            s = np.asarray(g["grid_same"])
            c = np.asarray(g["grid_coll"])
            m = (~np.isnan(s)) & (~np.isnan(c))
            o = np.argsort(cov[m])
            axes[0].plot(cov[m][o], s[m][o], label=name, color=colors.get(name), lw=1.8)
            axes[1].plot(cov[m][o], c[m][o], label=name, color=colors.get(name), lw=1.8)
        axes[0].set_xlabel("coverage (known fraction)")
        axes[0].set_ylabel("same-known (split-half consistency)")
        axes[0].set_title("coverage vs same (higher = better)")
        axes[1].set_xlabel("coverage (known fraction)")
        axes[1].set_ylabel("collision-known (lower = better)")
        axes[1].set_title("coverage vs collision")
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "coverage_same_curve.png"), dpi=140)
        print(f"[fig] {os.path.join(a.out, 'coverage_same_curve.png')}")
    except Exception as e:
        print("[fig] skip:", e)

    grids = report.pop("_grids", {})  # 曲线网格另存，study.json 只保留摘要
    json.dump(grids, open(os.path.join(a.out, "grids.json"), "w",
                          encoding="utf-8"), ensure_ascii=False)
    json.dump(report, open(os.path.join(a.out, "study.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] -> {os.path.join(a.out, 'study.json')}")


if __name__ == "__main__":
    main()
