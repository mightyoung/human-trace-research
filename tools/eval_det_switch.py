#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测器切换评估：任一组表征目录走同一判定管线，输出 full / common 两个口径。

与 eval_two_encodings.run() 同管线（会话对齐 + 每身份 z 校准 + self-cal 定 θ，
a/b 一致率 >=0.90、覆盖优先），因此可与 10-17/10-18 的历史数字直接比较。

与 eval_mask_variants.py 的差别：
  - 变体与段由命令行指定（换检测器时只跑试点段，不必凑齐 8 段）
  - 缺文件的段自动跳过
  - common 口径 = 各变体轨迹交集（公平比较），full = 各自全集（部署视角）

用法：
  python tools/eval_det_switch.py --variant baseline=.../1012_osnet_emb \
      --variant yolo26s=.../1012_yolo_emb --segs 305_train_529,305_test_544
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
from eval_kreciprocal import k_reciprocal  # noqa: E402

ROOT = S.ROOT
LOCAL = os.path.join(ROOT, "output_crosscam_local")


def has_pair(emb_dir, seg):
    return (os.path.exists(os.path.join(emb_dir, f"{seg}_emb_a.npz")) and
            os.path.exists(os.path.join(emb_dir, f"{seg}_emb_b.npz")))


def run(emb_dir, segs_tr, segs_te, cooc_te, allowed=None, min_eval=100, same_min=0.90, fm_max=None,
        cluster_mode="global", thr_in=0.12, thr_out=0.20, hier_thr=None,
        scorer="proto", kr_k1=18, kr_k2=6, kr_lam=0.5, veto_dmax=None):
    ktr, Va_tr, Vb_tr = M.load_split(emb_dir, segs_tr, allowed)
    kte, Va_te, Vb_te = M.load_split(emb_dir, segs_te, allowed)
    if not len(ktr) or not len(kte):
        return None
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    if cluster_mode == "camera":
        cams_tr = np.array([k[0].split("_")[0] for k in ktr])
        lab, kept = CA.camera_aware_labels(Vtr, cams_tr, thr_in, thr_out, S.MIN_CLUSTER)
    else:
        lab, kept = S.hier_labels(Vtr, hier_thr)
    if veto_dmax is not None:
        # 簇纯度否决：簇内最大成对余弦距离 >= veto_dmax 的簇整簇弃用（不让它成为任何身份原型）。
        # 依据：39 个已人工判定簇上 d_max 的 AUC 0.769；留一验证下精度 0.643→0.796。
        drop = []
        for c in kept:
            idx = np.where(lab == c)[0]
            if idx.size >= 2:
                Xc = Vtr[idx]
                Sx = Xc @ Xc.T
                dmax = float(1.0 - Sx[np.triu_indices(idx.size, 1)].min())
            else:
                dmax = 0.0
            if dmax >= veto_dmax:
                drop.append(c)
        if drop:
            ds = set(drop)
            kept = [c for c in kept if c not in ds]
            print(f"[veto] 弃用 {len(drop)}/{len(drop) + len(kept)} 个簇（d_max>={veto_dmax}）")
    K = len(kept)

    if scorer == "krecip":
        # 标准 k-reciprocal 重排（Zhong CVPR'17）：图建在 train+test 并集上，
        # 相似度用原始 cosine（不做均值对齐——实测对齐后邻居结构失效，cov 会掉到 0.4–0.56）。
        def cs(M):
            S0 = np.full((M.shape[0], K), -np.inf, np.float64)
            for ci, c in enumerate(kept):
                idx = np.where(lab == c)[0]
                if idx.size:
                    S0[:, ci] = M[:, idx].max(1)
            return S0

        ntr = len(Vtr)
        X = np.vstack([Vtr, _norm((Va_te + Vb_te) / 2.0)])
        D = (1.0 - X @ X.T).astype(np.float64)
        np.fill_diagonal(D, 0.0)
        Sr = 1.0 - k_reciprocal(D, kr_k1, kr_k2, kr_lam)
        S_tr = cs(Sr[:ntr, :ntr]).astype(np.float32)
        m_hat, s_hat, _ = S.fit_ident_stats(S_tr, lab, kept)
        out = []
        for Vq in (_norm(Va_te), _norm(Vb_te)):
            Xq = np.vstack([Vtr, Vq])
            Dq = (1.0 - Xq @ Xq.T).astype(np.float64)
            np.fill_diagonal(Dq, 0.0)
            Sq = cs(1.0 - k_reciprocal(Dq, kr_k1, kr_k2, kr_lam)[ntr:, :ntr])
            out.append(((Sq - m_hat) / s_hat).astype(np.float32))
        Sa, Sb = out
    else:
        mu_g = Vtr.mean(0)
        protos, pcl = [], []
        for ci, c in enumerate(kept):
            m = lab == c
            if m.sum() == 0:
                continue
            protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
            pcl.append(ci)
        protos = _norm(np.asarray(protos, np.float32) - mu_g)
        pcl = np.asarray(pcl)
        m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
        mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
        Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
        Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    keys_ok = set(kte)
    cooc = [(u, v) for u, v in cooc_te if u in keys_ok and v in keys_ok]
    rows = S.sweep(Sa, Sb, kte, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= same_min
          and r["evaluable"] >= min_eval
          and (fm_max is None or (np.isfinite(r["false_merge"]) and r["false_merge"] <= fm_max))]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    ref = {}
    for tgt in (0.30, 0.40, 0.50):
        cand = [r for r in rows if r["coverage"] >= tgt and r["evaluable"] >= min_eval
                and np.isfinite(r["false_merge"])]
        if cand:
            r = min(cand, key=lambda x: x["coverage"])
            ref[str(tgt)] = {"same": r["same"], "false_merge": r["false_merge"]}
    return {"n_train": len(ktr), "n_test": len(kte), "K": K, "n_cooc": len(cooc),
            "self_cal": best, "at_coverage": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="",
                    help="部署配置 JSON（configs/selector_deploy.json）：提供 variants / cooc_map / same_min 默认值，"
                         "命令行显式给出的项优先")
    ap.add_argument("--variant", action="append", default=[], help="NAME=DIR，可重复")
    ap.add_argument("--segs", default="", help="逗号分隔；留空=8 段全集")
    ap.add_argument("--cooc-map", action="append", default=[],
                    help="NAME=DIR：该变体的轨迹来源（DIR/{seg}/{seg}_detections.jsonl）；"
                         "不给则用 DINO 的 output_crosscam/1012_final/{seg}.jsonl。"
                         "换检测器时 track id 空间不同，必须各自从自己的检测结果取共现对")
    ap.add_argument("--min-eval", type=int, default=None, help="self-cal 要求的最少可评估共现对数")
    ap.add_argument("--same-min", type=float, default=None,
                    help="self-cal 的 a/b 一致率门槛（未指定时：有 profile 取 profile，否则 0.90）")
    ap.add_argument("--fm-max", type=float, default=None,
                    help="self-cal 的误合并上限（可选，默认不约束；profile 有 fm_max 时取之）。"
                         "历史口径只约束 same，不约束 fm，加上后结果会变严。")
    ap.add_argument("--veto-dmax", type=float, default=None,
                    help="簇纯度否决（默认关）：图库簇内最大成对余弦距离 >= 该值的簇整簇弃用。"
                         "依据 docs/cluster-purity-signal-2026-10-19.md。0.30 是留一验证下的候选值。")
    ap.add_argument("--clean-neg", action="store_true",
                    help="从共现负对里剔除「头框轨迹 ↔ 它自己的身体」这类同人重复检测对（假负对）。"
                         "默认关闭＝历史口径；开启后 fm 分母会少 16 对（805→789）。")
    ap.add_argument("--cluster-mode", choices=["global", "camera"], default=None,
                    help="global=4 机位混合做一次 average-linkage（历史口径）；"
                         "camera=先机位内紧聚类再跨机位合并（camera-aware）。"
                         "未指定时：有 profile 取 profile.cluster.mode，否则 global")
    ap.add_argument("--hier-thr", type=float, default=None,
                    help="global 模式的 cosine 距离阈值（默认取 session_selector_v1.HIER_THR=0.2）")
    ap.add_argument("--thr-in", type=float, default=None,
                    help="camera 模式：机位内紧聚类阈值（profile 有则取之，否则 0.12）")
    ap.add_argument("--thr-out", type=float, default=None,
                    help="camera 模式：跨机位合并阈值（profile 有则取之，否则 0.20）")
    ap.add_argument("--scorer", choices=["proto", "krecip"], default=None,
                    help="proto=簇原型余弦 + 每身份 z 校准（生产历史口径）；krecip=k-reciprocal 重排")
    ap.add_argument("--kr-k1", type=int, default=None, help="krecip 的 k1（默认 18）")
    ap.add_argument("--kr-k2", type=int, default=None, help="krecip 的 k2（默认 6）")
    ap.add_argument("--kr-lam", type=float, default=None, help="krecip 的 lambda（默认 0.5）")
    ap.add_argument("--out", default=os.path.join(LOCAL, "selector_osnet_1012/eval_det_switch.json"))
    a = ap.parse_args()

    prof = {}
    if a.profile:
        prof = json.load(open(a.profile, encoding="utf-8"))
        print(f"[profile] {prof.get('name')} detector={prof.get('detector')} "
              f"same_min={prof.get('same_min')} primary={prof.get('primary')}")
    if a.min_eval is None:
        a.min_eval = int(prof.get("min_eval", 100))
    if a.same_min is None:
        a.same_min = float(prof.get("same_min", 0.90))
    if a.fm_max is None and prof.get("fm_max") is not None:
        a.fm_max = float(prof["fm_max"])
    if not a.variant:
        a.variant = [f"{n}={d}" for n, d in prof.get("variants", {}).items()]
    if not a.cooc_map:
        a.cooc_map = [f"{n}={d}" for n, d in prof.get("cooc_map", {}).items()]
    clu = prof.get("cluster", {}) if prof else {}
    if a.cluster_mode is None:
        a.cluster_mode = clu.get("mode", "global")
    if a.hier_thr is None and clu.get("hier_thr") is not None:
        a.hier_thr = float(clu["hier_thr"])
    if a.thr_in is None:
        a.thr_in = float(clu.get("thr_in", 0.12))
    if a.thr_out is None:
        a.thr_out = float(clu.get("thr_out", 0.20))
    if a.veto_dmax is None and clu.get("veto_dmax") is not None:
        a.veto_dmax = float(clu["veto_dmax"])
    sc = prof.get("scorer", {}) if prof else {}
    if a.scorer is None:
        a.scorer = sc.get("mode", "proto")
    if a.kr_k1 is None:
        a.kr_k1 = int(sc.get("k1", 18))
    if a.kr_k2 is None:
        a.kr_k2 = int(sc.get("k2", 6))
    if a.kr_lam is None:
        a.kr_lam = float(sc.get("lambda", 0.5))
    print(f"[cluster] mode={a.cluster_mode} hier_thr={a.hier_thr} "
          f"thr_in={a.thr_in} thr_out={a.thr_out} | scorer={a.scorer} "
          f"k1={a.kr_k1} k2={a.kr_k2} lam={a.kr_lam}")
    if not a.variant:
        raise SystemExit("需要 --variant 或 --profile")

    variants = []
    for spec in a.variant:
        name, _, d = spec.partition("=")
        variants.append((name, d))

    want = [s for s in a.segs.split(",") if s] or (S.TRAIN_SEGS + S.TEST_SEGS)
    segs_tr = [s for s in want if s in S.TRAIN_SEGS]
    segs_te = [s for s in want if s in S.TEST_SEGS]

    avail = [(n, d, [s for s in segs_tr if has_pair(d, s)],
              [s for s in segs_te if has_pair(d, s)]) for n, d in variants]
    for n, d, tr, te in avail:
        print(f"[avail] {n}: train={tr} test={te}")
    if len(segs_te) == 0:
        raise SystemExit("没有可用的测试段")

    sets = {}
    for n, d, tr, te in avail:
        ks = set()
        for s in tr + te:
            k, _va, _vb = M.load_pair(d, s)
            ks |= set(k)
        sets[n] = ks
        print(f"[keys] {n}: {len(ks)}")
    common = set.intersection(*sets.values())
    print(f"[keys] common: {len(common)}")

    # 同帧共现对（同时在场 = 铁定不同人）——每个变体用自己检测结果里的 track id
    cmap = {}
    for spec in a.cooc_map:
        name, _, d = spec.partition("=")
        cmap[name] = d
    cooc_by_var = {}
    for n, d, _tr, te in avail:
        keys_te = set()
        for s in te:
            k, _va, _vb = M.load_pair(d, s)
            keys_te |= set(k)
        pairs = []
        for s in te:
            src = cmap.get(n)
            p = (os.path.join(src, s, f"{s}_detections.jsonl") if src
                 else os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"))
            if os.path.exists(p):
                pairs += [((s, u), (s, v)) for u, v in cooc_pairs(p, keys_te, s, 5)]
        if a.clean_neg:
            from render_head_track_audit import find_head_tracks
            n0 = len(pairs)
            kept_pairs = []
            for s in te:
                src = cmap.get(n)
                p = (os.path.join(src, s, f"{s}_detections.jsonl") if src
                     else os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"))
                hm = {t: v[0] for t, v in find_head_tracks(s, p).items()} if os.path.exists(p) else {}
                for u, v in pairs:
                    if u[0] != s:
                        continue
                    if (u[1] in hm and hm[u[1]] == v[1]) or (v[1] in hm and hm[v[1]] == u[1]):
                        continue
                    kept_pairs.append((u, v))
            pairs = kept_pairs
            print(f"[cooc-clean] {n}: 剔除头框↔宿主 {n0 - len(pairs)} 对")
        cooc_by_var[n] = pairs
        print(f"[cooc] {n}: {len(pairs)} pairs")

    out = {"variants": {n: d for n, d in variants}, "segs": want,
           "same_min": a.same_min, "fm_max": a.fm_max, "min_eval": a.min_eval,
           "clean_neg": bool(a.clean_neg),
           "cluster_mode": a.cluster_mode, "hier_thr": a.hier_thr,
           "thr_in": a.thr_in, "thr_out": a.thr_out, "veto_dmax": a.veto_dmax,
           "scorer": a.scorer, "kr": {"k1": a.kr_k1, "k2": a.kr_k2, "lambda": a.kr_lam},
           "profile": prof.get("name") if prof else None,
           "n_common": len(common), "runs": {}}
    for scope, allowed in (("full", None), ("common", common)):
        for n, d, tr, te in avail:
            if not tr:
                continue
            r = run(d, tr, te, cooc_by_var[n], allowed, a.min_eval, a.same_min, a.fm_max,
                    a.cluster_mode, a.thr_in, a.thr_out, a.hier_thr,
                    a.scorer, a.kr_k1, a.kr_k2, a.kr_lam, a.veto_dmax)
            if r is None:
                continue
            out["runs"][f"{n}@{scope}"] = r
            b = r["self_cal"] or {}
            print(f"[{n:9s}@{scope:6s}] tr={r['n_train']} te={r['n_test']} K={r['K']} "
                  f"pairs={r['n_cooc']} | cov={b.get('coverage', float('nan')):.3f} "
                  f"same={b.get('same', float('nan')):.3f} "
                  f"fm={b.get('false_merge', float('nan'))*100:.2f}%", flush=True)
            for t, v in r["at_coverage"].items():
                print(f"                 @cov>={t}: same={v['same']:.3f} "
                      f"fm={v['false_merge']*100:.2f}%", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
