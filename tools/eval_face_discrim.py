#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人脸嵌入在本数据集上的判别力（用免费真值，不需要人工标注）。

正对：同一条轨迹的 a/b 两半各自的人脸原型（同一人，跨时段/跨姿态）
负对：同帧共现的两条轨迹（同时在场 → 必为不同人）

与 body（OSNet）同口径对比，回答：人脸值不值得接进主链？
用法：python tools/eval_face_discrim.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

VEC = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/face_routing/face_vectors.npz")
DET_DIR = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.dirname(VEC)


def cooc_pairs(seg, stride=5):
    """同帧共现的轨迹对（间隔≥stride 帧才计一次，避免同一对反复计数）"""
    p = os.path.join(DET_DIR, seg, f"{seg}_detections.jsonl")
    pairs = set()
    last = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            tids = sorted(int(t) for t in o["track_ids"])
            fr = int(o["frame"])
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    k = (tids[i], tids[j])
                    if fr - last.get(k, -10 ** 9) >= stride:
                        pairs.add(k)
                        last[k] = fr
    return pairs


def auc(pos, neg):
    v = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(v)
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    z = np.load(VEC)
    vec = {k: v for k, v in zip(z["keys"].tolist(), z["vecs"])}
    print(f"vectors: {len(vec)}")

    pos = []
    for k in vec:
        if k.endswith("|a"):
            kb = k[:-1] + "b"
            if kb in vec:
                pos.append(float(vec[k] @ vec[kb]))
    pos = np.asarray(pos)

    neg = []
    n_pairs = 0
    segs = sorted({k.split("|")[0] for k in vec})
    for seg in segs:
        pairs = cooc_pairs(seg)
        n_pairs += len(pairs)
        for u, v in pairs:
            ku, kv = f"{seg}|{u}|full", f"{seg}|{v}|full"
            if ku in vec and kv in vec:
                neg.append(float(vec[ku] @ vec[kv]))
    neg = np.asarray(neg)

    res = dict(n_pos=len(pos), n_neg=len(neg), n_cooc_pairs=n_pairs)
    if len(pos) >= 3:
        res["pos"] = dict(p5=round(float(np.percentile(pos, 5)), 3),
                          p25=round(float(np.percentile(pos, 25)), 3),
                          p50=round(float(np.percentile(pos, 50)), 3))
    if len(neg) >= 3:
        res["neg"] = dict(p50=round(float(np.percentile(neg, 50)), 3),
                          p95=round(float(np.percentile(neg, 95)), 3),
                          p99=round(float(np.percentile(neg, 99)), 3),
                          max=round(float(neg.max()), 3))
    if len(pos) >= 3 and len(neg) >= 3:
        res["auc"] = round(auc(pos, neg), 4)
        sweep = []
        for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
            tpr = float((pos >= t).mean())
            fpr = float((neg >= t).mean())
            sweep.append(dict(thr=t, tpr=round(tpr, 3), fpr=round(fpr, 4),
                              nmiss=int((pos < t).sum()), nfp=int((neg >= t).sum())))
        res["sweep"] = sweep
    with open(os.path.join(OUT, "face_discrim.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in res.items() if k != "sweep"}, ensure_ascii=False, indent=2))
    for s in res.get("sweep", []):
        print("  thr=%.2f  同人通过 %5.1f%%  异人通过 %5.2f%%  (漏 %d / 误 %d)"
              % (s["thr"], 100 * s["tpr"], 100 * s["fpr"], s["nmiss"], s["nfp"]))


if __name__ == "__main__":
    main()
