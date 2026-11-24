#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""拒识理由报告（face-body-routing 设计里的 P0）。

只做**解释层**，不改覆盖口径、不改选择器判定：
  对每条 test 轨迹，按生产配置给出 a/b 判定与分数，再结合人脸可用性，分成四类：
    accepted            两侧都接受
    face_rejected       有可用人脸，但 body 没匹配上  → 真正可追的缺口
    near_no_face        无可用人脸，但 body 分数接近门槛（θ-0.5 内）→ 差一点
    unjudgeable         无可用人脸且分数远低于门槛 → 明确「看不清」

输出：
  - 终端分机位汇总
  - CSV：selector_osnet_1012/reject_reasons.csv（逐轨迹，含分数/人脸字段）

用法：
  python tools/report_reject_reasons.py --profile configs/selector_deploy.json
"""
import argparse
import collections
import csv
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
LOCAL = os.path.join(ROOT, "output_crosscam_local")


def load_face(seg_filter=None):
    p = os.path.join(LOCAL, "selector_osnet_1012/face_routing/face_track.csv")
    out = {}
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if seg_filter and r["seg"] not in seg_filter:
                continue
            try:
                out[(r["seg"], int(r["tid"]))] = r
            except (KeyError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.path.join(ROOT, "configs/selector_deploy.json"))
    ap.add_argument("--emb", default="")
    ap.add_argument("--det", default="")
    ap.add_argument("--out", default=os.path.join(LOCAL, "selector_osnet_1012/reject_reasons.csv"))
    a = ap.parse_args()

    prof = json.load(open(a.profile, encoding="utf-8"))
    primary = prof.get("primary", "yolo8dd")
    emb = a.emb or os.path.join(ROOT, prof["variants"][primary])
    det = a.det or os.path.join(ROOT, prof["cooc_map"].get(primary, ""))
    clu = prof.get("cluster", {})

    ktr, Atr, Btr = [], [], []
    for s in S.TRAIN_SEGS:
        k, x, y = M.load_pair(emb, s)
        if k:
            ktr += k
            Atr.append(x)
            Btr.append(y)
    kte, Ate, Bte = [], [], []
    for s in S.TEST_SEGS:
        k, x, y = M.load_pair(emb, s)
        if k:
            kte += k
            Ate.append(x)
            Bte.append(y)
    Atr, Btr = np.concatenate(Atr), np.concatenate(Btr)
    Ate, Bte = np.concatenate(Ate), np.concatenate(Bte)

    Vtr = _norm((Atr + Btr) / 2.0)
    if clu.get("mode") == "camera":
        cams = np.array([k[0].split("_")[0] for k in ktr])
        lab, kept = CA.camera_aware_labels(Vtr, cams, clu.get("thr_in", 0.12),
                                          clu.get("thr_out", 0.20), S.MIN_CLUSTER)
    else:
        lab, kept = S.hier_labels(Vtr, clu.get("hier_thr"))

    K = len(kept)
    mu_g = Vtr.mean(0)
    protos = _norm(np.asarray([_norm(Vtr[lab == c].mean(0, keepdims=True))[0] for c in kept],
                              np.float32) - mu_g)
    pcl = np.arange(K)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Ate + Bte) / 2.0).mean(0)
    Sa = S.score_rows(Ate, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Bte, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))

    cooc = []
    for s in S.TEST_SEGS:
        p = os.path.join(det, s, f"{s}_detections.jsonl")
        if os.path.exists(p):
            cooc += [((s, u), (s, v)) for u, v in cooc_pairs(p, set(kte), s, 5)]
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    rows = S.sweep(Sa, Sb, kte, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= prof.get("same_min", 0.90)
          and r["evaluable"] >= prof.get("min_eval", 100)]
    best = max(ok, key=lambda r: r["coverage"])
    thr = best["thr"]
    print(f"[{prof.get('name')}] θ={thr:.2f} cov={best['coverage']:.3f} "
          f"same={best['same']:.3f} fm={best['false_merge']*100:.2f}%")

    face = load_face({k[0] for k in kte})
    near = thr - 0.5
    out, stat = [], collections.defaultdict(lambda: collections.Counter())
    for i, k in enumerate(kte):
        sa, sb = float(Sa[i].max()), float(Sb[i].max())
        score = min(sa, sb)
        cam = k[0].split("_")[0]
        acc = sa >= thr and sb >= thr
        f = face.get(k, {})
        usable = str(f.get("usable", "0")) in ("1", "True", "true")
        det_faces = int(float(f.get("det") or 0)) if f else 0
        if acc:
            cat = "accepted"
        elif usable:
            cat = "face_rejected"
        elif score >= near:
            cat = "near_no_face"
        else:
            cat = "unjudgeable"
        stat[cam][cat] += 1
        stat["ALL"][cat] += 1
        out.append({"seg": k[0], "tid": k[1], "cam": cam,
                    "score_a": round(sa, 3), "score_b": round(sb, 3), "score": round(score, 3),
                    "accepted": int(acc), "category": cat,
                    "face_usable": int(usable), "face_det": det_faces,
                    "face_self_cos": f.get("self_cos", ""), "face_pd_p50": f.get("pd_p50", ""),
                    "n_frames": f.get("n_frames", "")})

    cats = ["accepted", "face_rejected", "near_no_face", "unjudgeable"]
    print(f"\n{'机位':6s} {'轨迹':>4s} " + " ".join(f"{c:>14s}" for c in cats))
    for cam in sorted(stat):
        n = sum(stat[cam].values())
        print(f"{cam:6s} {n:4d} " + " ".join(f"{stat[cam][c]:7d} ({stat[cam][c]/n*100:4.0f}%)" for c in cats))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    n = len(out)
    print(f"\n[out] {a.out}  ({n} 条)")
    print(f"可追缺口（有可用脸但 body 未接受）: {stat['ALL']['face_rejected']} 条")
    print(f"明确看不清（无脸且远低于门槛）  : {stat['ALL']['unjudgeable']} 条 "
          f"（{stat['ALL']['unjudgeable']/n*100:.0f}%）")


if __name__ == "__main__":
    main()
