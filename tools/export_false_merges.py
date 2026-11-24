#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出指定变体在指定 θ 下的"共现误合并"明细，供人工复核（U2 的最小可用输入）。

管线与 eval_det_switch.run() 完全一致（同一 hier_labels / fit_ident_stats / score_rows），
只是把 false_merge 的分子逐对落盘：段、两条轨迹 id、共现帧、两侧分数。

用法：
  python tools/export_false_merges.py --variant yolo26s=output_crosscam_local/1012_yolo8_emb \
      --det-map yolo26s=output_crosscam_local/1012_yolo8 --thr 2.85 \
      --out output_crosscam_local/selector_osnet_1012/fm_pairs_yolo.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm                                  # noqa: E402
import session_selector_v1 as S                             # noqa: E402
from eval_selector_1012test import cooc_pairs                # noqa: E402
import eval_mask_variants as M                               # noqa: E402
import cluster_camera_aware as CA                            # noqa: E402


def build(emb_dir, segs_tr, segs_te, cluster_mode="global", hier_thr=None,
          thr_in=0.12, thr_out=0.20):
    ktr, Va_tr, Vb_tr = M.load_split(emb_dir, segs_tr)
    kte, Va_te, Vb_te = M.load_split(emb_dir, segs_te)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    if cluster_mode == "camera":
        cams_tr = np.array([k[0].split("_")[0] for k in ktr])
        lab, kept = CA.camera_aware_labels(Vtr, cams_tr, thr_in, thr_out, S.MIN_CLUSTER)
    else:
        lab, kept = S.hier_labels(Vtr, hier_thr)
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
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    return kte, Sa, Sb, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="NAME=EMB_DIR")
    ap.add_argument("--det-map", required=True, help="NAME=DET_DIR（取共现对与共现帧）")
    ap.add_argument("--det-pattern", default="{det}/{seg}/{seg}_detections.jsonl",
                    help="检测文件路径模板；DINO 基线用 output_crosscam/1012_final/{seg}.jsonl 时"
                         "传 \"{det}/{seg}.jsonl\" 并把 --det-map 指到 1012_final")
    ap.add_argument("--thr", type=float, required=True, help="self-cal 选中的 θ")
    ap.add_argument("--cluster-mode", choices=["global", "camera"], default="global")
    ap.add_argument("--hier-thr", type=float, default=None)
    ap.add_argument("--thr-in", type=float, default=0.12)
    ap.add_argument("--thr-out", type=float, default=0.20)
    ap.add_argument("--segs", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    name, _, emb = a.variant.partition("=")
    _, _, det = a.det_map.partition("=")
    want = [s for s in a.segs.split(",") if s] or (S.TRAIN_SEGS + S.TEST_SEGS)
    segs_tr = [s for s in want if s in S.TRAIN_SEGS]
    segs_te = [s for s in want if s in S.TEST_SEGS]

    kte, Sa, Sb, K = build(emb, segs_tr, segs_te, a.cluster_mode, a.hier_thr, a.thr_in, a.thr_out)
    kidx = {k: i for i, k in enumerate(kte)}
    ia, sa = Sa.argmax(1), Sa.max(1)
    ib, sb = Sb.argmax(1), Sb.max(1)
    acca, accb = sa >= a.thr, sb >= a.thr

    # 共现帧索引：{(seg,u,v) -> first frame}
    first_co = {}
    for s in segs_te:
        p = a.det_pattern.format(det=det, seg=s)
        if not os.path.exists(p):
            continue
        keys_te = {k for k in kte if k[0] == s}
        for u, v in cooc_pairs(p, keys_te, s, 5):
            first_co.setdefault((s, min(u, v), max(u, v)), None)
        # 逐帧记录共现
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            ids = sorted(set(d.get("track_ids") or []))
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    k = (s, ids[i], ids[j])
                    if k in first_co and first_co[k] is None:
                        first_co[k] = d.get("frame")

    rows = []
    for (s, u, v), frame in first_co.items():
        if (s, u) not in kidx or (s, v) not in kidx:
            continue
        iu, iv = kidx[(s, u)], kidx[(s, v)]
        hit = False
        if acca[iu] and accb[iv] and ia[iu] == ib[iv]:
            hit, who = True, "a_u=b_v"
        elif acca[iv] and accb[iu] and ia[iv] == ib[iu]:
            hit, who = True, "a_v=b_u"
        if hit:
            rows.append({"seg": s, "id_u": u, "id_v": v, "frame": frame,
                         "cluster": int(ia[iu] if who == "a_u=b_v" else ia[iv]),
                         "score": round(float(max(Sa[iu].max(), Sa[iv].max())), 2),
                         "dir": who})
    rows.sort(key=lambda r: (r["seg"], r["frame"] or 0))
    print(f"[{name}] θ={a.thr} K={K} 误合并 {len(rows)} 对")
    for r in rows[:50]:
        print(f"  {r['seg']:14s} f={r['frame']} id={r['id_u']},{r['id_v']} "
              f"cluster={r['cluster']} score={r['score']} {r['dir']}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                               ["seg", "id_u", "id_v", "frame", "cluster", "score", "dir"])
            w.writeheader()
            w.writerows(rows)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
