#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 DINOv3-B 空间构建身份选择器（EVAL-3c 长流实验的前置）。

两个数据源（同一无监督管线，仅换训练段）：
  --source 1012_train   1012-train 4 路（跨 session 参照；实测对 long35 覆盖仅 33%）
  --source long35_ref   long35 前 --ref-end 帧（**同 session bootstrap**，默认用于长流实验）

为什么需要后者：d3b 在 1012-train 上构建的选择器，对另一天的 long35 只覆盖 33%
（同段重叠校验显示 d3b 同人 cos≈0.999、随机对 p99=0.966 → 判别信息充分，
低覆盖是跨日外观漂移所致）。长流实验因此采用"同 session 先 bootstrap 再流式"的部署形态；
跨 session 覆盖崩塌单独作为结果报告。

管线（与 build_selector_v2 完全一致）：
  mutual-knn 聚类 → 共现负对（同帧共现≥5 帧 = 铁定不同人）两档阈值
  → multi8 子原型 → per-cam margin p30 校准

产物：output_crosscam_local/d3b_ab/selector_d3b[_longref].npz（IdentitySelector 兼容）
只读实验：不写任何生产文件。
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from selector import IdentitySelector, _norm  # noqa: E402
from selector_upgrade_study import (build_variants, score_matrix, wp_thr,  # noqa: E402
                                    legacy_thr_vec, MIN_CLUSTER)
from camgroup_study import select_best, cam_of, CAMS  # noqa: E402
from train_selector_cluster import (cooc_pairs, pairwise_f1_cooc,  # noqa: E402
                                    tail_digits)

ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "output_crosscam_local", "d3b_ab")
JDL_1012 = os.path.join(ROOT, "output_crosscam", "1012_final")
JDL_LONG = os.path.join(ROOT, "output_crosscam", "long35")
TRAIN_SEGS = ["305_train_529", "318_train_525", "320_train_514", "323_train_510"]
CRADIO_REF = os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012",
                          "selector_v2.npz")


def _load_ab(path, seg, cams=None, tid_filter=None):
    z = np.load(f"{path}.npz")
    za = np.load(f"{path}_a.npz")
    zb = np.load(f"{path}_b.npz")
    keys, va, vb = [], [], []
    for k in z.files:
        if k not in za.files or k not in zb.files:
            continue
        tid = tail_digits(k)
        if tid_filter is not None and tid not in tid_filter:
            continue
        keys.append((seg, tid))
        va.append(za[k])
        vb.append(zb[k])
    return keys, np.asarray(va, np.float32), np.asarray(vb, np.float32)


def track_midframes(jsonl_path):
    acc = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for t in d.get("track_ids", []) or []:
                if t is None or int(t) < 0:
                    continue
                a = acc.setdefault(int(t), [10 ** 9, -1])
                a[0] = min(a[0], d["frame"])
                a[1] = max(a[1], d["frame"])
    return {k: (v[0] + v[1]) / 2.0 for k, v in acc.items()}


def load_segs_source(source, ref_end):
    keys, Va, Vb, cooc_all, jdls = [], [], [], [], {}
    if source == "1012_train":
        for s in TRAIN_SEGS:
            ks, va, vb = _load_ab(os.path.join(BASE, "1012_train", f"{s}_emb_d3b"), s)
            keys += ks; Va.append(va); Vb.append(vb)
            jdls[s] = os.path.join(JDL_1012, f"{s}.jsonl")
            print(f"  [load] {s}: {len(ks)} tracks", flush=True)
    else:
        for c in CAMS:
            seg = f"{c}_long35"
            jp = os.path.join(JDL_LONG, f"{c}_long35_detections.jsonl")
            mids = track_midframes(jp)
            keep = {t for t, m in mids.items() if m < ref_end}
            ks, va, vb = _load_ab(os.path.join(BASE, "long35", f"{c}_emb_long35_d3b"),
                                  seg, tid_filter=keep)
            keys += ks; Va.append(va); Vb.append(vb)
            jdls[seg] = jp
            print(f"  [load] {seg}: {len(ks)} tracks (mid<{ref_end}, "
                  f"candidates={len(keep)})", flush=True)
    Va = np.concatenate(Va); Vb = np.concatenate(Vb)
    V = _norm((Va + Vb) / 2.0, axis=1)
    keys_set = set(keys)
    for seg, jp in jdls.items():
        pairs = cooc_pairs(jp, keys_set, seg, 5)
        cooc_all += [((seg, u), (seg, v)) for u, v in pairs]
        print(f"  [cooc] {seg}: {len(pairs)} 对", flush=True)
    return keys, Va, Vb, V, cooc_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="long35_ref", choices=["1012_train", "long35_ref"])
    ap.add_argument("--ref-end", type=int, default=15000,
                    help="long35_ref 源：只用中点帧 < ref_end 的轨迹做 bootstrap")
    ap.add_argument("--out-name", default="")
    args = ap.parse_args()

    out_name = args.out_name or ("selector_d3b.npz" if args.source == "1012_train"
                                 else "selector_d3b_longref.npz")
    print(f"[1/5] 加载 {args.source}（d3b a/b）...", flush=True)
    TR_keys, TR_Va, TR_Vb, TR_V, cooc_all = load_segs_source(args.source, args.ref_end)
    print(f"[1/5] 合计 {len(TR_keys)} 轨，dim={TR_V.shape[1]}，共现负对 {len(cooc_all)}",
          flush=True)

    tr_cam = np.array([cam_of(k[0]) for k in TR_keys])
    kidx = {k: i for i, k in enumerate(TR_keys)}

    print("[2/5] 聚类网格选择...", flush=True)
    _, results = select_best(TR_V, TR_Va, TR_Vb, TR_keys, cooc_all)
    best = [d for d in results if d["method"] == "mutual-knn"][0]
    lab = best["_lab"]
    ids = sorted(set(lab))
    keep_ids = [c for c in ids if (lab == c).sum() >= MIN_CLUSTER]
    print(f"[2/5] mutual-knn k={best.get('k')} thr={best.get('thr')} "
          f"ncl={best['n_clusters']} F1={best['f1']:.4f} → keep {len(keep_ids)} "
          f"(median size {np.median([(lab == c).sum() for c in keep_ids]):.0f})", flush=True)

    print("[3/5] multi8 子原型...", flush=True)
    variants, C = build_variants(TR_V, lab, keep_ids)
    v = variants["multi8"]

    print("[4/5] 阈值/margin 校准...", flush=True)
    sc_a = score_matrix(TR_Va, v, C)
    sc_b = score_matrix(TR_Vb, v, C)
    sa, sb = sc_a.max(1), sc_b.max(1)
    g_self = wp_thr(sa, sb, cooc_all, TR_keys, kidx)
    pc_self = {c: wp_thr(sa, sb, cooc_all, TR_keys, kidx, cam_mask=(tr_cam == c))
               for c in CAMS}
    g_loose = legacy_thr_vec(TR_Va, TR_Vb, TR_V, TR_keys, cooc_all, kidx)
    pc_loose = {c: legacy_thr_vec(TR_Va, TR_Vb, TR_V, TR_keys, cooc_all, kidx,
                                  cam_mask=(tr_cam == c)) for c in CAMS}
    ia, ib = sc_a.argmax(1), sc_b.argmax(1)
    pf = pairwise_f1_cooc({k: int(ia[i]) for i, k in enumerate(TR_keys)},
                          {k: int(ib[i]) for i, k in enumerate(TR_keys)},
                          cooc_all, TR_keys)
    srt_a, srt_b = np.sort(sc_a, 1), np.sort(sc_b, 1)
    pos_m = np.minimum(srt_a[:, -1] - srt_a[:, -2], srt_b[:, -1] - srt_b[:, -2])
    margin_pc = {c: {f"p{p}": float(np.percentile(pos_m[tr_cam == c], p))
                     for p in (30, 20, 15, 10)} for c in CAMS}

    print(f"[4/5] F1={pf['f1']:.4f} P={pf['prec']:.3f} R={pf['rec']:.3f} | selfcal "
          + " ".join(f"{c}:{pc_self[c]['thr']:.4f}" for c in CAMS), flush=True)
    print("[4/5] loose " + " ".join(f"{c}:{pc_loose[c]['thr']:.4f}" for c in CAMS), flush=True)
    print("[4/5] margin p30 " + " ".join(f"{c}:{margin_pc[c]['p30']:.5f}" for c in CAMS),
          flush=True)

    info = {"method": best["method"], "params": {"k": best.get("k"), "thr": best.get("thr")},
            "n_clusters": C, "n_subprotos": int(v["subprotos"].shape[0]),
            "topk": v["topk"], "f1_cooc_train": pf["f1"],
            "thr_rule": "selfcal: mid-point(min(sa,sb)_p5, cooc_p99); "
                        "loose: mid-point(Va.Vb_p5, cooc_VV_p99)",
            "selfcal": {"global": g_self["thr"],
                        "per_cam": {c: pc_self[c]["thr"] for c in CAMS}},
            "loose": {"global": g_loose["thr"],
                      "per_cam": {c: pc_loose[c]["thr"] for c in CAMS}},
            "margin_prec_per_cam": {c: round(margin_pc[c]["p30"], 6) for c in CAMS},
            "margin_scan_per_cam": {c: {k: round(vv, 6) for k, vv in margin_pc[c].items()}
                                    for c in CAMS},
            "margin_rule": "per-cam 分位: p30(min(margin_a,margin_b) 同人池) ≈ 保留70%",
            "version": f"d3b-{args.source}-multi8-percam",
            "space": "dinov3b-768",
            "source": args.source, "ref_end": args.ref_end,
            "n_train_tracks": len(TR_keys),
            "note": "与 selector_v2 同一无监督管线，仅换嵌入空间/训练段；不含倒影门"}

    sel = IdentitySelector(
        v["subprotos"], thr_unknown=g_self["thr"], proto_cluster=v["sub2cluster"],
        thr_unknown_per_cam=info["selfcal"]["per_cam"],
        thr_loose=g_loose["thr"], thr_loose_per_cam=info["loose"]["per_cam"],
        margin_prec=0.005, margin_prec_per_cam=info["margin_prec_per_cam"],
        margin_ret=0.7, meta=info)
    path = os.path.join(BASE, out_name)
    sel.save(path)
    print(f"[5/5] saved {path}", flush=True)

    cmp = {"built": dict(n_train=len(TR_keys), n_clusters=C,
                         n_subprotos=int(v["subprotos"].shape[0]), f1=round(pf["f1"], 4),
                         thr_selfcal_g=round(g_self["thr"], 4),
                         thr_per_cam={c: round(pc_self[c]["thr"], 4) for c in CAMS},
                         margin_p30={c: round(margin_pc[c]["p30"], 5) for c in CAMS})}
    if os.path.exists(CRADIO_REF):
        z = np.load(CRADIO_REF, allow_pickle=True)
        m = z["meta"][0]
        cmp["cradio_ref"] = dict(n_clusters=m.get("n_clusters"),
                                 n_subprotos=m.get("n_subprotos"),
                                 f1=round(float(m.get("f1_cooc_train", 0)), 4),
                                 thr_selfcal_g=round(float(m["selfcal"]["global"]), 4))
    with open(os.path.join(BASE, out_name.replace(".npz", "_meta.json")), "w",
              encoding="utf-8") as f:
        json.dump(dict(info=info, compare=cmp), f, ensure_ascii=False, indent=1)
    print("[compare] " + json.dumps(cmp, ensure_ascii=False), flush=True)

    ids_v, sims_v = sel.predict(TR_V, cams=[cam_of(k[0]) for k in TR_keys])
    print(f"[smoke] bootstrap known={float((ids_v >= 0).mean()):.3f} | "
          f"sim median={np.median(sims_v[ids_v >= 0]):.4f}", flush=True)


if __name__ == "__main__":
    main()
