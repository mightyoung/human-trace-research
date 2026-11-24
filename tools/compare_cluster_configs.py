#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""聚类配置的同覆盖点对比 + 图库纯度诊断。

为什么不用 self-cal 单点比较：self-cal 在 θ 网格上取「same>=门槛 且覆盖最大」，
相邻配置常落在不同平台，读数会跳 10pp+（本项目 10-19 已多次踩到）。
本脚本改为在**固定覆盖点**上报 same / fm，并额外算图库自身纯度
（train 同帧共现对 = 铁定不同人，被聚进同一簇的比例）。

用法：
  python tools/compare_cluster_configs.py --emb output_crosscam_local/1012_yolo8_dd_emb \
      --det output_crosscam_local/1012_yolo8_dd
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

ROOT = S.ROOT


def load(emb, segs, allowed=None):
    keys, As, Bs = [], [], []
    for s in segs:
        k, a, b = M.load_pair(emb, s, allowed)
        if len(k):
            keys += k
            As.append(a)
            Bs.append(b)
    return keys, np.concatenate(As), np.concatenate(Bs)


def cooc_from_det(det, segs, keyset, gap=5):
    out = []
    for s in segs:
        p = os.path.join(det, s, f"{s}_detections.jsonl")
        if os.path.exists(p):
            out += [((s, u), (s, v)) for u, v in cooc_pairs(p, keyset, s, gap)]
    return out


def build(lab, kept, Vtr):
    K = len(kept)
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
    return K, mu_g, protos, pcl, (m_hat, s_hat)


def purity(lab, keys, cooc, kept=None):
    idx = {k: i for i, k in enumerate(keys)}
    kept_set = set(kept) if kept is not None else set(lab.tolist())
    ev = mg = 0
    for u, v in cooc:
        i, j = idx.get(u), idx.get(v)
        if i is None or j is None:
            continue
        if lab[i] not in kept_set or lab[j] not in kept_set:
            continue
        ev += 1
        mg += int(lab[i] == lab[j])
    return mg, ev


def at_cov(rows, target, min_eval=100):
    cand = [r for r in rows if r["coverage"] >= target and r["evaluable"] >= min_eval
            and np.isfinite(r["false_merge"]) and np.isfinite(r["same"])]
    if not cand:
        return None
    return min(cand, key=lambda r: r["coverage"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb"))
    ap.add_argument("--det", default=os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/cluster_cmp.json"))
    a = ap.parse_args()

    ktr, Atr, Btr = load(a.emb, S.TRAIN_SEGS)
    kte, Ate, Bte = load(a.emb, S.TEST_SEGS)
    Vtr = _norm((Atr + Btr) / 2.0)
    cams_tr = np.array([k[0].split("_")[0] for k in ktr])
    cooc_tr = cooc_from_det(a.det, S.TRAIN_SEGS, set(ktr))
    cooc_te = cooc_from_det(a.det, S.TEST_SEGS, set(kte))
    print(f"train {len(ktr)} test {len(kte)} | cooc train {len(cooc_tr)} test {len(cooc_te)}")

    mu_q = _norm((Ate + Bte) / 2.0).mean(0)
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)

    configs = [("global thr=0.20", "global", dict(hier_thr=0.20)),
               ("global thr=0.15", "global", dict(hier_thr=0.15)),
               ("global thr=0.12", "global", dict(hier_thr=0.12)),
               ("cam 0.12/0.15", "camera", dict(thr_in=0.12, thr_out=0.15)),
               ("cam 0.12/0.20", "camera", dict(thr_in=0.12, thr_out=0.20)),
               ("cam 0.12/0.25", "camera", dict(thr_in=0.12, thr_out=0.25)),
               ("cam 0.15/0.20", "camera", dict(thr_in=0.15, thr_out=0.20)),
               ("cam 0.15/0.25", "camera", dict(thr_in=0.15, thr_out=0.25)),
               ("cam 0.15/0.30", "camera", dict(thr_in=0.15, thr_out=0.30)),
               ("cam 0.18/0.25", "camera", dict(thr_in=0.18, thr_out=0.25))]
    targets = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    res = {}
    print(f"\n{'config':16s} {'K':>4s} {'库纯度':>14s} {'self-cal cov':>12s} {'same':>6s} {'fm':>7s}")
    for tag, mode, kw in configs:
        if mode == "camera":
            lab, kept = CA.camera_aware_labels(Vtr, cams_tr, kw["thr_in"], kw["thr_out"], S.MIN_CLUSTER)
        else:
            lab, kept = S.hier_labels(Vtr, kw["hier_thr"])
        K, mu_g, protos, pcl, calib = build(lab, kept, Vtr)
        Sa = S.score_rows(Ate, protos, pcl, K, mu=mu_q, calib=calib)
        Sb = S.score_rows(Bte, protos, pcl, K, mu=mu_q, calib=calib)
        rows = S.sweep(Sa, Sb, kte, cooc_te, grid)
        mg, ev = purity(lab, ktr, cooc_tr, kept)
        ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
        best = max(ok, key=lambda r: r["coverage"]) if ok else None
        entry = {"mode": mode, "kw": kw, "K": K, "gallery_fm": (mg / ev if ev else None),
                 "gallery_fm_pairs": f"{mg}/{ev}",
                 "self_cal": best,
                 "at_cov": {str(t): at_cov(rows, t) for t in targets}}
        res[tag] = entry
        print(f"{tag:16s} {K:4d} {mg:6d}/{ev:<6d} {best['coverage']:12.3f} {best['same']:6.3f} {best['false_merge']*100:6.2f}%")

    print(f"\n同覆盖点比较（same / fm）：")
    hdr = "".join(f"{t:>16.2f}" for t in targets)
    print(f"{'config':16s}" + hdr)
    for tag, e in res.items():
        line = f"{tag:16s}"
        for t in targets:
            v = e["at_cov"][str(t)]
            if v is None:
                cell = "—"
            else:
                cell = "%.3f/%.2f%%" % (v["same"], v["false_merge"] * 100)
            line += f"{cell:>16s}"
        print(line)

    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
