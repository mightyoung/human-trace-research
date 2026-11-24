#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同一判定管线下的两组表征对照（基线 vs A+C）。

只换"轨迹表征怎么来"，其余完全一致：
  会话对齐 + 每身份 z 校准 + self-cal 定 θ（覆盖最大且 a/b 一致率 ≥0.90）。
指标：覆盖 / 稳定 / 共现误合并。

用法：
  python tools/eval_two_encodings.py \
     --a output_crosscam_local/1012_osnet_emb \
     --b output_crosscam_local/1012_osnet_crops_ac
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


def load_pair(emb_dir, seg):
    za = np.load(os.path.join(emb_dir, f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(emb_dir, f"{seg}_emb_b.npz"))
    keys, va, vb = [], [], []
    for k in za.files:
        if k not in zb.files:
            continue
        keys.append((seg, int(str(k).rsplit("#", 1)[1])))
        va.append(za[k])
        vb.append(zb[k])
    return keys, np.asarray(va, np.float32), np.asarray(vb, np.float32)


def load_split(emb_dir, segs):
    keys, Va, Vb = [], [], []
    for s in segs:
        k, a, b = load_pair(emb_dir, s)
        keys += k
        Va.append(a)
        Vb.append(b)
    return keys, np.concatenate(Va), np.concatenate(Vb)


def run(emb_dir, cooc_te):
    ktr, Va_tr, Vb_tr = load_split(emb_dir, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = load_split(emb_dir, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
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
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    keys_ok = set(kte)
    cooc = [(u, v) for u, v in cooc_te if u in keys_ok and v in keys_ok]
    rows = S.sweep(Sa, Sb, kte, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90
          and r["evaluable"] >= 100]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    # 同覆盖率参考点
    ref = {}
    for tgt in (0.30, 0.40, 0.50):
        cand = [r for r in rows if r["coverage"] >= tgt and r["evaluable"] >= 100
                and np.isfinite(r["false_merge"])]
        if cand:
            r = min(cand, key=lambda x: x["coverage"])
            ref[str(tgt)] = {"same": r["same"], "false_merge": r["false_merge"]}
    return {"n_train": len(ktr), "n_test": len(kte), "K": K, "self_cal": best,
            "at_coverage": ref, "sweep": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="基线嵌入目录")
    ap.add_argument("--b", required=True, help="A+C 嵌入目录")
    ap.add_argument("--out", default=os.path.join(S.ROOT,
                    "output_crosscam_local/selector_osnet_1012/eval_ac.json"))
    args = ap.parse_args()

    kte_guess = S.load(S.TEST_SEGS)[0]
    cooc_te = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(S.ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte_guess), s, 5):
            cooc_te.append(((s, u), (s, v)))

    out = {"cooc_pairs": len(cooc_te)}
    for tag, d in (("baseline", args.a), ("A+C", args.b)):
        r = run(d, cooc_te)
        out[tag] = {k: v for k, v in r.items() if k != "sweep"}
        out[tag]["sweep"] = r["sweep"]
        b = r["self_cal"] or {}
        print(f"[{tag:8s}] tr={r['n_train']} te={r['n_test']} K={r['K']} | "
              f"cov={b.get('coverage', float('nan')):.3f} same={b.get('same', float('nan')):.3f} "
              f"fm={b.get('false_merge', float('nan'))*100:.2f}%")
        for t, v in r["at_coverage"].items():
            print(f"           @cov>={t}: same={v['same']:.3f} fm={v['false_merge']*100:.2f}%")

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
