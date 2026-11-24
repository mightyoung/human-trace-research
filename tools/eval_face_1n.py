#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人脸 vs 整体：同一批轨迹上的 1:N 检索对比（Rank-1 / Rank-5）。

协议（同一条轨迹的两半互查，避免用 gallery 真值）：
    query = 轨迹 i 的前半原型 (a)
    gallery = 所有轨迹的后半原型 (b)，i 自己的 b 也在其中
    命中 = Rank-1 是 i 自己的 b（等价于"在两片之间把这个人找回来"）

意义：这比 1:1 更接近业务（我们要在 109 个簇里挑一个），而 1:N 对质量更敏感。
用法：python tools/eval_face_1n.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

FR = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/face_routing")
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
OUT = os.path.join(FR, "face_1n.json")


def load_face():
    """seg -> {tid: {'a':v,'b':v,'full':v}}"""
    res = {}
    for p in glob.glob(os.path.join(FR, "face_vecs_*.npz")):
        z = np.load(p)
        d = {}
        for k, v in zip(z["keys"].tolist(), z["vecs"]):
            seg, tid, tag = k.split("|")
            d.setdefault(tid, {})[tag] = v
        seg = p.split("face_vecs_")[1][:-4]
        res[seg] = d
    return res


def load_body():
    """seg -> {tid: {'a':v,'b':v}}（生产 YOLO 表征）"""
    res = {}
    for p in glob.glob(os.path.join(EMB, "*_emb_a.npz")):
        seg = os.path.basename(p)[:-len("_emb_a.npz")]
        za = np.load(p)
        zb = np.load(os.path.join(EMB, f"{seg}_emb_b.npz"))
        d = {}
        kb = set(zb.files)
        for k in za.files:
            if k not in kb:
                continue
            tid = str(k).split("#")[-1]
            va, vb = za[k], zb[k]
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na > 1e-6 and nb > 1e-6:
                d[tid] = {"a": va / na, "b": vb / nb}
        res[seg] = d
    return res


def ranks(mats_a, mats_b):
    """返回每条 query 的 rank（1-based）；gallery 含自己。"""
    out = []
    for i, (va, _) in enumerate(mats_a):
        sims = np.array([float(va @ vb) for _, vb in mats_b])
        order = np.argsort(-sims)
        out.append(int(np.where(order == i)[0][0]) + 1)
    return out


def summarize(rk, n):
    rk = np.asarray(rk)
    return dict(n=n,
                rank1=int((rk <= 1).sum()), rank1_rate=round(float((rk <= 1).mean()), 3),
                rank5=int((rk <= 5).sum()), rank5_rate=round(float((rk <= 5).mean()), 3),
                rank_med=float(np.median(rk)))


def main():
    F = load_face()
    B = load_body()
    res = {"protocol": "query=a, gallery=all b (含自己), 命中=rank1 指向自己的 b", "by_seg": {},
           "note": "样本量小（每段 4–20 条），置信区间很宽；只用于同条件下的人脸/body 对比，不作绝对结论"}
    agg_f, agg_b = [], []
    for seg in sorted(F):
        face = F[seg]
        tids = sorted(t for t, d in face.items() if "a" in d and "b" in d)
        if len(tids) < 3:
            continue
        fa = [(face[t]["a"].astype(np.float32), None) for t in tids]
        fb = [(None, face[t]["b"].astype(np.float32)) for t in tids]
        rf = ranks(fa, fb)
        body = B.get(seg, {})
        bt = [t for t in tids if t in body]
        if len(bt) >= 3:
            ba = [(body[t]["a"], None) for t in bt]
            bb = [(None, body[t]["b"]) for t in bt]
            rb = ranks(ba, bb)
        else:
            rb = []
        res["by_seg"][seg] = dict(face=summarize(rf, len(tids)),
                                  body=summarize(rb, len(bt)) if rb else None)
        agg_f += rf
        agg_b += rb
    res["overall"] = dict(face=summarize(agg_f, len(agg_f)) if agg_f else None,
                          body=summarize(agg_b, len(agg_b)) if agg_b else None)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res["overall"], ensure_ascii=False))
    for seg, d in res["by_seg"].items():
        print(seg, "face", d["face"], "| body", d["body"])
    print("[out]", OUT)


if __name__ == "__main__":
    main()
