#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按簇分层抽第二批基线精度样本（每簇 1 条）。

为什么要按簇抽：第一批 25 条只落在 16 个簇上，且 7 个多成员簇 7/7 全对或全错
（错误是簇级现象），轨迹之间不独立 —— 有效样本量是簇数不是轨迹数。

抽样设计：
  - 逐机位（305 / 318 / 323）在"尚未审过的簇"里**均匀随机抽簇**，每簇只抽 1 条轨迹；
  - 320 已全查（6/6），不补；
  - 抽样结果同时支持两个估计量：
      * 簇级精度 = 抽样簇中判对的簇占比（简单平均）
      * 轨迹级精度 = 按簇大小加权（Horvitz-Thompson），与第一批口径可比

前置校验（不通过则不写文件）：
  1) 重算的 score_a / score_b / accepted 必须与 reject_reasons.csv 逐条一致（229 条）
  2) 重算的簇号必须与 baseline_audit_sample.csv 的 25 条一致

用法：
  python tools/sample_cluster_audit.py --n 305:10,318:8,323:8 --seed 20261019
输出：
  output_crosscam_local/selector_osnet_1012/baseline_audit2_sample.csv
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
import cluster_camera_aware as CA  # noqa: E402
from selector import _norm  # noqa: E402

EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
REJ = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/reject_reasons.csv")
PREV = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/baseline_audit_sample.csv")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/baseline_audit2_sample.csv")

THR = 1.85
THR_IN, THR_OUT = 0.15, 0.25


def build():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    cams_tr = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams_tr, THR_IN, THR_OUT, S.MIN_CLUSTER)
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

    csize = {int(c): int((lab == c).sum()) for c in kept}
    recs = []
    for i, k in enumerate(kte):
        sa, sb = float(Sa[i].max()), float(Sb[i].max())
        ia, ib = int(Sa[i].argmax()), int(Sb[i].argmax())
        recs.append({
            "seg": k[0], "tid": int(k[1]), "cam": k[0].split("_")[0],
            "score_a": sa, "score_b": sb,
            "acc": sa >= THR and sb >= THR,
            # 注意：历史 csv 的 cluster 列是 kept 的下标（0..K-1），不是原始 fcluster 标签。
            # 两者差 1.6 倍（K=111 vs uniq=177），混用会导致抽样落在错的簇上。
            "ident_a": ia, "ident_b": ib,
            "raw_label": int(kept[ia]),
            "ab_same": ia == ib,
            "size": csize[int(kept[ia])],
        })
    return recs, K


def verify(recs):
    rej = {(r["seg"], int(r["tid"])): r for r in
           csv.DictReader(open(REJ, encoding="utf-8-sig"))}
    bad = 0
    for r in recs:
        o = rej.get((r["seg"], r["tid"]))
        if o is None:
            print("[ERR] 缺 reject_reasons 记录", r["seg"], r["tid"])
            bad += 1
            continue
        if abs(float(o["score_a"]) - r["score_a"]) > 5e-3 or \
           abs(float(o["score_b"]) - r["score_b"]) > 5e-3:
            bad += 1
            if bad <= 5:
                print(f"[ERR] 分数不一致 {r['seg']}#{r['tid']} "
                      f"recalc {r['score_a']:.4f}/{r['score_b']:.4f} "
                      f"csv {o['score_a']}/{o['score_b']}")
        if (o["accepted"] == "1") != r["acc"]:
            bad += 1
            if bad <= 5:
                print(f"[ERR] accepted 不一致 {r['seg']}#{r['tid']}")
    print(f"[verify] 229 条打分/accepted：{'全部一致' if bad == 0 else f'{bad} 处不一致'}")
    if bad:
        raise SystemExit(1)

    prev = list(csv.DictReader(open(PREV, encoding="utf-8-sig")))
    idx = {(r["seg"], r["tid"]): r for r in recs}
    miss = 0
    for r in prev:
        p = idx.get((r["seg"], int(r["tid"])))
        if p is None or p["ident_a"] != int(r["assigned_cluster"]):
            miss += 1
            print(f"[ERR] 簇号不一致 {r['seg']}#{r['tid']} "
                  f"recalc {p and p['ident_a']} csv {r['assigned_cluster']}")
    print(f"[verify] 25 条簇号：{'全部一致' if miss == 0 else f'{miss} 处不一致'}")
    if miss:
        raise SystemExit(1)


def sample(recs, want, seed):
    prev = {(r["seg"], int(r["tid"])) for r in
            csv.DictReader(open(PREV, encoding="utf-8-sig"))}
    done_clusters = {int(r["assigned_cluster"]) for r in
                     csv.DictReader(open(PREV, encoding="utf-8-sig"))}
    rng = random.Random(seed)
    picked = []
    for cam, n in want:
        pool = [r for r in recs if r["acc"] and r["cam"] == cam and r["ab_same"]]
        by = defaultdict(list)
        for r in pool:
            by[r["ident_a"]].append(r)
        cand = [c for c in by if c not in done_clusters]
        cand.sort()
        if len(cand) < n:
            print(f"[warn] {cam} 可用簇只有 {len(cand)} 个，少于目标 {n}")
        take = rng.sample(cand, min(n, len(cand)))
        print(f"[{cam}] 可抽簇 {len(cand)}，抽 {len(take)}；"
              f"簇大小 {sorted(len(by[c]) for c in take)}")
        for c in take:
            mem = [r for r in by[c] if (r["seg"], r["tid"]) not in prev]
            if not mem:
                mem = by[c]
            picked.append(rng.choice(mem))
    picked.sort(key=lambda r: (r["cam"], -r["ident_a"]))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="305:10,318:8,323:8")
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args()
    want = []
    for p in a.n.split(","):
        c, n = p.split(":")
        want.append((c, int(n)))

    recs, K = build()
    print(f"[build] K={K}  test={len(recs)}  accepted={sum(r['acc'] for r in recs)}")
    verify(recs)
    picked = sample(recs, want, a.seed)

    hdr = ["seg", "tid", "cam", "abs_cos_a", "abs_cos_b", "assigned_cluster",
           "cluster_size", "cluster_tracks"]
    acc_by = defaultdict(int)
    for r in recs:
        if r["acc"]:
            acc_by[r["ident_a"]] += 1
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in picked:
            w.writerow([r["seg"], r["tid"], r["cam"],
                        f"{r['score_a']:.3f}", f"{r['score_b']:.3f}",
                        r["ident_a"], r["size"], acc_by[r["ident_a"]]])
    print(f"\n[out] {OUT}  ({len(picked)} 条 / {len(set(r['ident_a'] for r in picked))} 个簇)")
    for r in picked:
        print(f"  {r['cam']} {r['seg']}#{r['tid']:<5} cluster={r['ident_a']:<4} "
              f"size={r['size']:<3} acc_in_cluster={acc_by[r['ident_a']]}  "
              f"score={r['score_a']:.2f}/{r['score_b']:.2f}")


if __name__ == "__main__":
    main()
