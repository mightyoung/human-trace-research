# -*- coding: utf-8 -*-
"""E21 跨天矩阵：用逐查询分数在**固定 θ** 上重算三指标。

为什么不能直接用 best.json 的 fm
--------------------------------
d7_sweep 的 θ 是自校准选出来的，9 个格子跨了 0.90–2.75，可评估的共现对数
从 18 到 523 —— 分母差 29 倍，fm 不可比。queries/*.csv 里存了 score_a/score_b，
所以可以在任意 θ 上重放接受判定，把分母钉死。

口径
----
coverage  双视图分数都 ≥θ 的查询占比
same      在双视图都接受的查询里，两视图 argmax 一致性
fm_geom   在「双端都被接受 且 两框中位框 IoU<0.30」的共现对上，被判同簇的比例
          （IoU≥0.30 视为重复框，不算两个人；详见 e21_xday_audit.py）

静态假框过多的段会被标记 invalid_seg 并单独统计（默认不剔除，但列出）

用法：
  python tools/e21_xday_table.py
  python tools/e21_xday_table.py --theta 0.9,1.5,2.0 --out tmp/e21_xday/matrix.json
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

ROOT = "tmp/e21_xday"
DET = "tmp/e21_local/det_gpu"
IOU_DUP = 0.30
STAT_CV = 0.04
STAT_MIN_LEN = 200


def track_geom(seg):
    p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
    if not os.path.exists(p):
        return None
    box = collections.defaultdict(list)
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for t, b in zip(d["track_ids"], d["boxes"]):
                box[t].append(b)
    out = {}
    for t, bl in box.items():
        a = np.asarray(bl, np.float32)
        w = float(np.median(a[:, 2] - a[:, 0]))
        h = float(np.median(a[:, 3] - a[:, 1]))
        n = len(a)
        cv = (float(0.5 * (a[:, 2] - a[:, 0]).std() / max(w, 1e-6)
                    + 0.5 * (a[:, 3] - a[:, 1]).std() / max(h, 1e-6))
              if n >= 5 else 1.0)
        out[t] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])),
                  float(np.median(a[:, 2])), float(np.median(a[:, 3])), cv, n)
    return out


def iou(b1, b2):
    x0, y0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x1, y1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-9)


def load_cell(cell, cache):
    d = os.path.join(ROOT, cell)
    q = {}
    with open(os.path.join(d, "A_baseline.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            q[r["query_id"]] = (float(r["score_a"]), float(r["score_b"]),
                                r["cluster_a_local"], r["cluster_b_local"],
                                r["final_cluster_local"])
    with open(os.path.join(d, "A_baseline.json"), encoding="utf-8") as f:
        pairs = json.load(f)
    keep, geom_info = [], []
    for u, v in pairs:
        su, tu, sv, tv = u[0], u[1], v[0], v[1]
        ku, kv = f"{su}#{tu}", f"{sv}#{tv}"
        if ku not in q or kv not in q:
            continue
        if su not in cache:
            cache[su] = track_geom(su)
        g = cache[su]
        if not g or tu not in g or tv not in g:
            continue
        b1, b2 = g[tu], g[tv]
        static = ((b1[4] < STAT_CV and b1[5] > STAT_MIN_LEN) or
                  (b2[4] < STAT_CV and b2[5] > STAT_MIN_LEN))
        keep.append((ku, kv, iou(b1, b2) < IOU_DUP, static))
        geom_info.append((su, iou(b1, b2), static))
    seg_stat = collections.Counter()
    for su, _, static in geom_info:
        seg_stat[su] = seg_stat.get(su, 0)
    seg_n, seg_s = collections.Counter(), collections.Counter()
    for su, _, static in geom_info:
        seg_n[su] += 1
        seg_s[su] += int(static)
    invalid = {s for s in seg_n if seg_n[s] >= 5
               and seg_s[s] / seg_n[s] > 0.5}
    return q, keep, invalid


def metrics(q, keep, theta, drop_invalid):
    acc = {k: (v[0] >= theta and v[1] >= theta) for k, v in q.items()}
    cov = sum(acc.values()) / max(len(acc), 1)
    accd = [k for k in acc if acc[k]]
    same = (sum(1 for k in accd if q[k][2] == q[k][3]) / len(accd)) if accd else float("nan")
    ev = mg = 0
    for ku, kv, geom_ok, static in keep:
        if not geom_ok:
            continue
        if drop_invalid and (ku.split("#")[0] in drop_invalid or
                             kv.split("#")[0] in drop_invalid):
            continue
        if acc[ku] and acc[kv]:
            ev += 1
            if q[ku][4] == q[kv][4]:
                mg += 1
    return {"coverage": cov, "same": same, "ev": ev, "mg": mg,
            "fm": (mg / ev if ev else float("nan"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--theta", default="0.90,1.20,1.50,2.00,2.50")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    TH = [float(x) for x in a.theta.split(",")]
    cells = sorted(c for c in os.listdir(a.root)
                   if os.path.isdir(os.path.join(a.root, c)))
    cache = {}
    data = {c: load_cell(c, cache) for c in cells}
    inv_all = set()
    for c in cells:
        inv_all |= data[c][2]
    if inv_all:
        print("静态假框段（共现对 >50% 含静止轨，建议剔除）：")
        for s in sorted(inv_all):
            print("   " + s)
        print()

    res = {}
    for drop in (False, True):
        tag = "剔除无效段" if drop else "全量"
        print("=" * 96)
        print(f"固定 θ 矩阵（fm 只用 IoU<{IOU_DUP} 的共现对）—— {tag}")
        print("%-26s %-13s" % ("cell", "查询日")
              + "".join("%20s" % f"θ={t:.1f}" for t in TH))
        for c in cells:
            q, keep, inv = data[c]
            row = []
            for t in TH:
                m = metrics(q, keep, t, inv if drop else set())
                row.append("%.3f/%.3f/%.1f%%" % (m["coverage"], m["same"], 100 * m["fm"]))
                res.setdefault((tag, c, t), m)
            print("%-26s %-13s" % (c, c.split("__to__")[1])
                  + "".join("%20s" % x for x in row))
        print()
        # 分母
        print("可评估共现对数（几何过滤后）")
        print("%-26s" % "cell" + "".join("%12s" % f"θ={t:.1f}" for t in TH))
        for c in cells:
            q, keep, inv = data[c]
            print("%-26s" % c + "".join(
                "%12d" % metrics(q, keep, t, inv if drop else set())["ev"]
                for t in TH))
        print()
    if a.out:
        out = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"invalid_segs": sorted(inv_all), "metrics": out},
                      f, ensure_ascii=False, indent=1)
        print("-> " + a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
