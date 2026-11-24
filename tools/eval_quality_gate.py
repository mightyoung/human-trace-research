#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""质量门：把「不可能判定」的轨迹排除后，指标怎么变。

两条口径都给：
  A. 分母不变（229 条）：质量门 = 强制弃判，覆盖必然下降，衡量"少说错话"的代价；
  B. 分母改为可用轨迹：衡量"在可判定的轨迹里能覆盖多少"。

同时报告：门拦掉了多少条已接受轨迹、7 条已复核样本里拦掉哪几条。
"""
import csv
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
QCSV = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/track_quality.csv")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/quality_gate.json")
GRID = np.round(np.arange(-2.0, 8.01, 0.05), 3)


def build():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g, mu_q = Vtr.mean(0), _norm((Va_te + Vb_te) / 2.0).mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    mh, sh, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(mh, sh))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(mh, sh))
    return kte, Sa, Sb, K


def main():
    qte, Sa, Sb, K = build()
    qm = {}
    with open(QCSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            qm[(r["seg"], int(r["tid"]))] = SimpleNamespace(
                n_obs=int(r["n_obs"]), h_px_p50=float(r["h_px_p50"]),
                h_frac_p50=float(r["h_frac_p50"]), aspect_p50=float(r["aspect_p50"]),
                score_p50=float(r["score_p50"]))
    feat = [qm[k] for k in qte]

    cooc = []
    keys_ok = set(qte)
    for seg in S.TEST_SEGS:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        cooc += [((seg, u), (seg, v)) for u, v in cooc_pairs(p, keys_ok, seg, 5)]

    rules = {
        "none": lambda f: True,
        "R1 n>=8 & hpx>=64": lambda f: (f.n_obs >= 8) and (f.h_px_p50 >= 64),
        "R1+score>=0.30": lambda f: (f.n_obs >= 8) and (f.h_px_p50 >= 64) and (f.score_p50 >= 0.30),
        "R2 n>=10 & hpx>=96": lambda f: (f.n_obs >= 10) and (f.h_px_p50 >= 96),
        "R3 n>=8 & hpx>=96 & sc>=0.30": lambda f: (f.n_obs >= 8) and (f.h_px_p50 >= 96) and (f.score_p50 >= 0.30),
    }

    checked = {("323_test_527", 6010): "可能对", ("323_test_527", 1112): "可能对",
               ("318_test_534", 882): "对", ("305_test_544", 892): "不可判定",
               ("320_test_518", 828): "可能错", ("320_test_518", 1225): "不确定",
               ("320_test_518", 1050): "错"}

    res = {}
    base_acc = None
    for name, fn in rules.items():
        keep = np.array([bool(fn(f)) for f in feat])
        Ska, Skb = Sa.copy(), Sb.copy()
        Ska[~keep] = -1e9
        Skb[~keep] = -1e9
        rows = S.sweep(Ska, Skb, qte, cooc, GRID)
        ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
        best = max(ok, key=lambda r: r["coverage"]) if ok else None
        acc = (Sa.max(1) >= 2.5) & (Sb.max(1) >= 2.5)
        if name == "none":
            base_acc = acc
        entry = {"n_usable": int(keep.sum()), "self_cal": best,
                 "n_acc_at_2.5": int((acc & keep).sum()),
                 "cov_over_usable": round(float((acc & keep).sum() / max(keep.sum(), 1)), 4),
                 "dropped_accepted": int((acc & ~keep).sum())}
        # 7 条已复核样本
        hit = []
        for k, v in checked.items():
            i = qte.index(k) if k in qte else None
            if i is not None and not keep[i]:
                hit.append(f"{k[0]}#{k[1]}({v})")
        entry["blocked_checked"] = hit
        res[name] = entry
        b = best or {}
        print(f"[{name:26s}] usable={keep.sum():3d} | thr={b.get('thr')} cov={b.get('coverage', float('nan')):.3f} "
              f"same={b.get('same', float('nan')):.3f} fm={b.get('false_merge', float('nan'))*100:.2f}% "
              f"| cov/usable={entry['cov_over_usable']:.3f} | 拦掉已接受 {entry['dropped_accepted']}")
        if hit:
            print(f"    拦掉已复核: {', '.join(hit)}")

    json.dump({"n_test": len(qte), "K": K, "rules": res},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
