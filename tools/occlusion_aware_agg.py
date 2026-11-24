#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""遮挡感知的轨迹聚合：量化"他人框覆盖"对表征的污染，并给出解法。

背景（2026-10-18 复盘中确认）：
  - 帧内重复框（同一目标多尺度）已由 tools/containment_nms.py 解决：
    1012 数据里 containment>0.9 的框对为 0，检测阶段删掉了 18.4% 的原始框。
  - 但**不同人之间的投影重叠**（前后站位、俯视工位）仍存在：A 的框里含 B 的像素。
    这部分会污染 A 的裁剪嵌入，现有链路没有任何处理。

本脚本：
  ① 对每条轨迹的每张采样裁剪，计算 occlusion = max(其他框 ∩ 本框) / 本框面积；
  ② 比较三种聚合：
     A0 uniform       现状（等权）
     A4 drop-occ      丢弃 occlusion>tau 的裁剪，剩余等权（不足 3 张则回退全部）
     A5 weight-occ    按 (1−occlusion) 加权
  ③ 在 E2 判定管线上评估（对齐 + 每身份 z 校准 + self-cal θ，稳定率≥0.90）。

用法：python tools/occlusion_aware_agg.py [--tau 0.15]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
CROP_DIR = os.path.join(ROOT, "output_crosscam_local/1012_osnet_crops")
AB = os.path.join(ROOT, "output_crosscam_local/1012_ab")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/occlusion_agg.json")
SAMPLES = 8


def track_frames(seg):
    """{(ab,tid): [(frame, box), ...]}，与编码器的采样顺序一致。"""
    t = {}
    for ab in ("a", "b"):
        d = defaultdict(list)
        for line in open(os.path.join(AB, f"{seg}_{ab}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            tids = r.get("track_ids") or []
            for i, b in enumerate(r.get("boxes") or []):
                if i < len(tids) and tids[i] is not None and int(tids[i]) >= 0:
                    d[int(tids[i])].append((int(r["frame"]), b))
        for k in d:
            d[k].sort()
        for k, v in d.items():
            t[(ab, k)] = v
    return t


def frame_boxes(seg):
    """{frame: [(tid, box), ...]} 用于算遮挡。"""
    fb = defaultdict(list)
    for ab in ("a", "b"):
        for line in open(os.path.join(AB, f"{seg}_{ab}.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            tids = r.get("track_ids") or []
            for i, b in enumerate(r.get("boxes") or []):
                if i < len(tids) and tids[i] is not None and int(tids[i]) >= 0:
                    fb[int(r["frame"])].append((int(tids[i]), b))
    return fb


def occlusion_of(box, others):
    """本框被其他框覆盖的最大比例。"""
    ax1, ay1, ax2, ay2 = [float(v) for v in box]
    own = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    worst = 0.0
    for _t, ob in others:
        bx1, by1, bx2, by2 = [float(v) for v in ob]
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        worst = max(worst, (iw * ih) / own)
    return float(min(worst, 1.0))


def compute_occ(seg):
    """{(ab,tid): [occ per sampled crop]}"""
    tf = track_frames(seg)
    fb = frame_boxes(seg)
    out = {}
    for (ab, tid), lst in tf.items():
        if len(lst) < 2:
            continue
        idxs = np.linspace(0, len(lst) - 1, min(SAMPLES, len(lst))).astype(int)
        occ = []
        for k in idxs:
            fnum, box = lst[k]
            others = [(t, b) for t, b in fb.get(fnum, []) if t != tid]
            occ.append(occlusion_of(box, others))
        out[(ab, tid)] = occ
    return out


def load_emb(seg):
    d = np.load(os.path.join(CROP_DIR, f"{seg}_crops.npz"))
    eids, emb = d["eids"], d["emb"]
    buck = defaultdict(list)
    for i, e in enumerate(eids):
        ab, tid, k = str(e).split("#")
        buck[(ab, int(tid))].append((int(k), i))
    for k in buck:
        buck[k].sort()
    return buck, emb


def aggregate(buck, emb, occ, seg, mode, tau):
    """返回 {tid: (va, vb)}"""
    out = {}
    tids = {t for (_ab, t) in buck}
    for t in sorted(tids):
        ka, kb = buck.get(("a", t)), buck.get(("b", t))
        if not ka or not kb:
            continue
        ves = []
        for ab, ks in (("a", ka), ("b", kb)):
            idx = [i for _k, i in ks]
            V = emb[idx]
            o = np.asarray((occ.get((ab, t)) or [0.0] * len(idx))[:len(idx)], np.float32)
            if len(o) < len(idx):
                o = np.pad(o, (0, len(idx) - len(o)), constant_values=0.0)
            if mode == "uniform":
                w = np.ones(len(idx), np.float32)
            elif mode == "drop":
                keep = o <= tau
                if keep.sum() < min(3, len(idx)):
                    keep = np.ones(len(idx), bool)
                V, o = V[keep], o[keep]
                w = np.ones(len(V), np.float32)
            else:  # weight
                w = np.clip(1.0 - o, 0.05, 1.0)
            m = (V * w[:, None]).sum(0) / w.sum()
            ves.append(m / (np.linalg.norm(m) + 1e-9))
        out[t] = (ves[0], ves[1])
    return out


def evaluate(agg_tr, agg_te, cooc_te, kte):
    keys_tr = sorted(agg_tr)
    Va_tr = np.asarray([agg_tr[t][0] for t in keys_tr], np.float32)
    Vb_tr = np.asarray([agg_tr[t][1] for t in keys_tr], np.float32)
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
    S_tr = S.score_rows(Vtr, protos, pcl, K, mu=mu_g)
    m_hat, s_hat, _ = S.fit_ident_stats(S_tr, lab, kept)

    keys_te = sorted(agg_te)
    Va = np.asarray([agg_te[t][0] for t in keys_te], np.float32)
    Vb = np.asarray([agg_te[t][1] for t in keys_te], np.float32)
    mu_q = _norm((Va + Vb) / 2.0).mean(0)
    Sa = S.score_rows(Va, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    rows = S.sweep(Sa, Sb, keys_te, cooc_te, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90
          and r["evaluable"] >= 100]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    return {"n_tr": len(keys_tr), "n_te": len(keys_te), "K": K, "self_cal": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.15)
    a = ap.parse_args()

    # 共现负对
    kte_guess = S.load(S.TEST_SEGS)[0]
    cooc_te = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte_guess), s, 5):
            cooc_te.append(((s, u), (s, v)))

    # 逐段统计遮挡
    occ_stats = {}
    occ_all = {"train": [], "test": []}
    for split, segs in (("train", S.TRAIN_SEGS), ("test", S.TEST_SEGS)):
        for s in segs:
            occ = compute_occ(s)
            vals = [v for vs in occ.values() for v in vs]
            occ_stats[s] = {"n_crop": len(vals),
                            "mean": float(np.mean(vals)),
                            "p50": float(np.median(vals)),
                            "p90": float(np.percentile(vals, 90)),
                            "frac_gt_tau": float(np.mean(np.asarray(vals) > a.tau))}
            occ_all[split] += vals
            print(f"[occ] {s}: crops={len(vals)} mean={np.mean(vals):.3f} "
                  f"p90={np.percentile(vals,90):.3f} frac>{a.tau}={np.mean(np.asarray(vals)>a.tau):.3f}")

    results = {}
    for mode in ("uniform", "drop", "weight"):
        agg_tr, agg_te = {}, {}
        for s in S.TRAIN_SEGS:
            buck, emb = load_emb(s)
            occ = compute_occ(s)
            for t, v in aggregate(buck, emb, occ, s, mode, a.tau).items():
                agg_tr[(s, t)] = v
        for s in S.TEST_SEGS:
            buck, emb = load_emb(s)
            occ = compute_occ(s)
            for t, v in aggregate(buck, emb, occ, s, mode, a.tau).items():
                agg_te[(s, t)] = v
        # 只保留 a/b 都存在且共现键一致的
        keys_te = set(agg_te)
        cooc_f = [(u, v) for u, v in cooc_te if u in keys_te and v in keys_te]
        r = evaluate(agg_tr, agg_te, cooc_f, keys_te)
        results[mode] = r
        b = r["self_cal"] or {}
        print(f"[{mode:8s}] tr={r['n_tr']} te={r['n_te']} K={r['K']} | cov={b.get('coverage', float('nan')):.3f} "
              f"same={b.get('same', float('nan')):.3f} fm={b.get('false_merge', float('nan'))*100:.2f}%")

    json.dump({"tau": a.tau, "occ_stats": occ_stats,
               "split_mean": {k: float(np.mean(v)) for k, v in occ_all.items()},
               "results": results}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
