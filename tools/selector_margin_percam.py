#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器改造 A：margin 的 per-camera 校准（建议要点②）+ 323 独立时段复核（建议要点①）。

Part A（margin per-cam）：
  δ_c = 中点法(同人分片 margin 下界 p5, 异人 cooc 交叉 margin 上界 p99)
  ——与 thr 校准完全同构，全部在 train 上完成；test 评估对比 全局δ / per-camδ。
Part B（323 复核）：
  用 1012 val 段（午间，独立于建库 train 与评估 test）复核 multi8 在 323 的退化：
  single vs multi8 的 WP 数字 + 统一 thr 曲线方向。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector import IdentitySelector, _norm  # noqa: E402
from selector_upgrade_study import build_variants, score_matrix, MIN_CLUSTER  # noqa: E402
from camgroup_study import (load_segs, select_best, cam_of, CAMS,  # noqa: E402
                            TRAIN_SEGS, JDL)
from train_selector_cluster import cooc_pairs  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012", "upgrade")


def margin_top2(sc):
    srt = np.sort(sc, axis=1)
    return srt[:, -1] - srt[:, -2]


def main():
    TR_keys, TR_Va, TR_Vb, TR_V = load_segs(TRAIN_SEGS)
    tr_cam = np.array([cam_of(k[0]) for k in TR_keys])
    keys_set = set(TR_keys)
    cooc_all = []
    for s in TRAIN_SEGS:
        for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), keys_set, s, 5):
            cooc_all.append(((s, u), (s, v)))
    kidx = {k: i for i, k in enumerate(TR_keys)}

    _, results = select_best(TR_V, TR_Va, TR_Vb, TR_keys, cooc_all)
    best = [d for d in results if d["method"] == "mutual-knn"][0]
    lab = best["_lab"]
    keep_ids = [c for c in sorted(set(lab)) if (lab == c).sum() >= MIN_CLUSTER]
    variants, C = build_variants(TR_V, lab, keep_ids)
    mv = variants["multi8"]

    # ===== Part A: margin per-cam 校准（train） =====
    print("=== Part A: margin per-cam 校准 ===")
    sc_a = score_matrix(TR_Va, mv, C)
    sc_b = score_matrix(TR_Vb, mv, C)
    mga, mgb = margin_top2(sc_a), margin_top2(sc_b)
    pos_m = np.minimum(mga, mgb)
    delta_c = {}
    for c in CAMS:
        m = tr_cam == c
        p5 = float(np.percentile(pos_m[m], 5))
        cs = []
        for u, v in cooc_all:
            iu, iv = kidx.get(u), kidx.get(v)
            if iu is None or iv is None or not m[iu]:
                continue
            cs += [float(mga[iu]), float(mgb[iv]), float(mga[iv]), float(mgb[iu])]
        p99 = float(np.percentile(cs, 99)) if cs else None
        if p99 is None:
            delta_c[c] = None
            continue
        delta_c[c] = max(0.0, (p5 + p99) / 2)
        print(f"  [{c}] pos_min_margin_p5={p5:.4f} neg_margin_p99={p99:.4f} "
              f"-> delta[{c}]={delta_c[c]:.4f} (n_pos={int(m.sum())}, n_neg={len(cs)})")
    delt_glob = 0.005
    print(f"  [global] delta={delt_glob}（现有参考值）")

    # ---- test 评估（multi8 + selfcal-percam thr + 三种 margin） ----
    from camgroup_study import TEST_SEGS
    TE_keys, TE_Va, TE_Vb, TE_V = load_segs(TEST_SEGS)
    te_cam = np.array([cam_of(k[0]) for k in TE_keys])
    study = json.load(open(os.path.join(OUT, "study.json")))
    thr_pc = {c: study["thr_selfcal"]["multi8"]["per_cam"][c]["thr"] for c in CAMS}
    sc_ta = score_matrix(TE_Va, mv, C)
    sc_tb = score_matrix(TE_Vb, mv, C)
    s1a, m1a = sc_ta.max(1), margin_top2(sc_ta)
    s1b, m1b = sc_tb.max(1), margin_top2(sc_tb)
    ca, cb = sc_ta.argmax(1), sc_tb.argmax(1)
    thr = np.array([thr_pc[c] for c in te_cam])
    dvec = np.array([(delta_c[c] if delta_c[c] is not None else delt_glob) for c in te_cam])
    rng = np.random.default_rng(3)
    N = len(TE_keys)
    L = rng.integers(0, N, 30000)
    R = rng.integers(0, N, 30000)
    kp = L != R

    def ev(dvec_, tag):
        ia = np.where((s1a >= thr) & (m1a >= dvec_), ca, -1)
        ib = np.where((s1b >= thr) & (m1b >= dvec_), cb, -1)
        kn = (ia >= 0) & (ib >= 0)
        cov = float(kn.mean())
        same = float((ia[kn] == ib[kn]).mean()) if kn.sum() else None
        il, ir = ia[L[kp]], ib[R[kp]]
        kn2 = (il >= 0) & (ir >= 0)
        coll = float((il[kn2] == ir[kn2]).mean()) if kn2.sum() else None
        pc = {}
        for c in CAMS:
            m = te_cam == c
            knc = ((ia[m] >= 0) & (ib[m] >= 0))
            pc[c] = {"cov": float(knc.mean()),
                     "same": (float((ia[m][knc] == ib[m][knc]).mean())
                              if knc.sum() else None)}
        print(f"  [{tag:12s}] cov={cov:.3f} same={same if same is None else round(same,3)} "
              f"coll={coll if coll is None else round(coll,3)} | "
              + " ".join(f"{c}:{pc[c]['cov']:.2f}/{None if pc[c]['same'] is None else round(pc[c]['same'],2)}"
                         for c in CAMS))
        return {"coverage": cov, "same_known": same, "coll_known": coll, "per_cam": pc}

    res = {"delta_per_cam": delta_c}
    print("\n  test 对比（multi8 + selfcal-percam thr）：")
    res["no_margin"] = ev(np.zeros(N), "no margin")
    res["global_005"] = ev(np.full(N, delt_glob), "global 0.005")
    res["percam"] = ev(dvec, "per-cam δ")
    json.dump(res, open(os.path.join(OUT, "margin_percam.json"), "w"),
              ensure_ascii=False, indent=1)

    # ===== Part B: 323 独立时段复核（1012 val 段） =====
    print("\n=== Part B: 323 val 段（独立时段）复核 ===")
    v_keys, v_Va, v_Vb = [], [], []
    seg = "323_val_538"
    za = np.load(os.path.join(ROOT, "output_crosscam", "analysis", "cradio_1012_emb",
                              f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(ROOT, "output_crosscam", "analysis", "cradio_1012_emb",
                              f"{seg}_emb_b.npz"))
    from train_selector_cluster import tail_digits
    for k in za.files:
        if k in zb.files:
            v_keys.append((seg, tail_digits(k)))
            v_Va.append(za[k])
            v_Vb.append(zb[k])
    v_Va, v_Vb = np.asarray(v_Va, np.float32), np.asarray(v_Vb, np.float32)
    print(f"  [data] {seg}: {len(v_keys)} tracks")

    from selector import IdentitySelector as IS
    sel8 = IS.load(os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012",
                                "selector_v2.npz"))
    sel1 = IS.load(os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012",
                                "selector_v2_single.npz"))
    cam_list = ["323"] * len(v_keys)
    out = {}
    for tag, sel in (("single", sel1), ("multi8", sel8)):
        ia, _ = sel.predict(v_Va, cams=cam_list)
        ib, _ = sel.predict(v_Vb, cams=cam_list)
        kn = (ia >= 0) & (ib >= 0)
        cov = float(kn.mean())
        same = float((ia[kn] == ib[kn]).mean()) if kn.sum() else None
        # 统一 thr 曲线（用 multi8 的 score 统计量两库对比不公平；此处只给 WP + 一个统一点位）
        out[tag] = {"coverage": cov, "same_known": same, "n_known": int(kn.sum())}
        print(f"  [{tag:7s}] WP(自带thr): cov={cov:.3f} same={same if same is None else round(same,3)} "
              f"(n={int(kn.sum())})")
    # 统一统计量对照：对 multi8 score 扫 thr，在相同 cov 下比 same（与 single 同口径统一扫描）
    def curve(sca, scb, ca_, cb_, targets=(0.2, 0.3, 0.4, 0.5)):
        sa_, sb_ = sca.max(1), scb.max(1)
        rows = []
        for t in np.arange(0.5, 0.9995, 0.002):
            ia_ = np.where(sa_ >= t, ca_, -1)
            ib_ = np.where(sb_ >= t, cb_, -1)
            kn_ = (ia_ >= 0) & (ib_ >= 0)
            rows.append((kn_.mean(), (ia_[kn_] == ib_[kn_]).mean() if kn_.sum() else np.nan))
        covs = np.array([r[0] for r in rows])
        sames = np.array([r[1] for r in rows])
        o = np.argsort(covs)
        return {f"{tg}": (float(np.interp(tg, covs[o], sames[o]))
                          if covs[o][0] <= tg <= covs[o][-1] else None) for tg in targets}
    # single 的 score：用 single 库的原型
    sc_a1 = _norm(v_Va) @ sel1.protos.T
    sc_b1 = _norm(v_Vb) @ sel1.protos.T
    cv1 = curve(sc_a1, sc_b1, sc_a1.argmax(1), sc_b1.argmax(1))
    sc_a8 = score_matrix(v_Va, mv, C)
    sc_b8 = score_matrix(v_Vb, mv, C)
    cv8 = curve(sc_a8, sc_b8, sc_a8.argmax(1), sc_b8.argmax(1))
    print(f"  [curve single] " + " ".join(f"c{k}:{'-' if v is None else round(v,2)}" for k, v in cv1.items()))
    print(f"  [curve multi8] " + " ".join(f"c{k}:{'-' if v is None else round(v,2)}" for k, v in cv8.items()))
    out["curve_single"] = cv1
    out["curve_multi8"] = cv8
    json.dump(out, open(os.path.join(OUT, "val323_recheck.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[done] -> margin_percam.json / val323_recheck.json")


if __name__ == "__main__":
    main()
