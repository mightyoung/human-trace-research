# -*- coding: utf-8 -*-
"""E21 跨天结果的几何审计：共现对到底是「两个人」还是「同一个框的重复」。

背景
----
d7_sweep 的 false_merge 把「同帧不同轨」当作已知负例（两个人同时在画）。
这条假设在两种情况下失效，实测两种都撞上了：

  1) 重复框：同一人出两个近乎重合的框 —— IoU 高，外观 cos 常 >0.99。
  2) 静态假框：检测器锁在灯/椅/显示器上，框坐标常年不动（如 2026-10-23
     办公区2 的 tid=6，9510 帧同一位置 (896,249)-(973,344)）。

这两类对一旦被算进分母，出现「同一机位 fm=100%」这种不可能的数字
（63/63），而它既不是模型错、也不是跨天错，是**负例集被污染**。

本脚本对每个共现对算几何证据，给出三层 fm：
  fm_all  原口径全部可评估对
  fm_geom 只保留 IoU<0.3 的对（两个框确实分得开）
  fm_stat 在 fm_geom 基础上再剔除含静态轨（框几乎不动）的对

用法：
  python tools/e21_xday_audit.py            # 读 tmp/e21_xday/*/
  python tools/e21_xday_audit.py --json tmp/e21_xday/audit.json
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

IOU_DUP = 0.30      # ≥ 此值视为重复框，不算「两个人」
STAT_CV = 0.04      # 框高一/宽一相对标准差 < 此值且寿命 >200 帧 → 静态
STAT_MIN_LEN = 200


def track_geom(seg):
    """每轨的中位框、框尺寸变异系数、寿命。"""
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
        if n >= 5 and w > 0 and h > 0:
            cv = float(0.5 * (a[:, 2] - a[:, 0]).std() / w
                       + 0.5 * (a[:, 3] - a[:, 1]).std() / h)
        else:
            cv = 1.0
        out[t] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])),
                  float(np.median(a[:, 2])), float(np.median(a[:, 3])),
                  cv, n)
    return out


def iou(b1, b2):
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-9)


def audit_cell(cell, cache):
    d = os.path.join(ROOT, cell)
    acc, cl = {}, {}
    qcam = collections.Counter()
    with open(os.path.join(d, "A_baseline.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = r["query_id"]
            acc[k] = (r["accept_a"] == "1" and r["accept_b"] == "1")
            cl[k] = r["final_cluster_local"]
            qcam[k.split("_")[1]] += 1
    with open(os.path.join(d, "A_baseline.json"), encoding="utf-8") as f:
        pairs = json.load(f)

    st = {k: {"ev": 0, "mg": 0,
              "cam": collections.Counter(),
              "cam_mg": collections.Counter()} for k in ("all", "geom", "stat")}
    per_cam = collections.defaultdict(lambda: collections.Counter())
    dup_iou, n_iou = 0, 0
    for u, v in pairs:
        su, tu = u[0], u[1]
        sv, tv = v[0], v[1]
        ku, kv = f"{su}#{tu}", f"{sv}#{tv}"
        if ku not in acc or kv not in acc or not (acc[ku] and acc[kv]):
            continue
        cam = su.split("_")[1]
        if su not in cache:
            cache[su] = track_geom(su)
        g = cache[su]
        if not g or tu not in g or tv not in g:
            continue
        b1, b2 = g[tu], g[tv]
        i = iou(b1, b2)
        static = ((b1[4] < STAT_CV and b1[5] > STAT_MIN_LEN) or
                  (b2[4] < STAT_CV and b2[5] > STAT_MIN_LEN))
        merged = cl[ku] == cl[kv]
        n_iou += 1
        if i >= IOU_DUP:
            dup_iou += 1
        for name, ok in (("all", True),
                         ("geom", i < IOU_DUP),
                         ("stat", i < IOU_DUP and not static)):
            if ok:
                st[name]["ev"] += 1
                st[name]["mg"] += int(merged)
                st[name]["cam"][cam] += 1
                if merged:
                    st[name]["cam_mg"][cam] += 1
    res = {"cell": cell, "n_pairs_raw": len(pairs), "n_iou": n_iou,
           "dup_rate": dup_iou / n_iou if n_iou else float("nan"),
           "qcam": dict(qcam)}
    for name in ("all", "geom", "stat"):
        ev, mg = st[name]["ev"], st[name]["mg"]
        res[name] = {"ev": ev, "mg": mg,
                     "fm": mg / ev if ev else float("nan"),
                     "cam": {c: [st[name]["cam"][c], st[name]["cam_mg"][c]]
                             for c in sorted(st[name]["cam"])}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cells = sorted(c for c in os.listdir(a.root)
                   if os.path.isdir(os.path.join(a.root, c)))
    cache = {}
    out = {}
    print("%-26s %6s %6s %6s %6s %6s %6s"
          % ("cell", "ev", "fm_all", "ev2", "fm_geom", "ev3", "fm_stat"))
    for c in cells:
        r = audit_cell(c, cache)
        out[c] = r
        print("%-26s %6d %5.1f%% %6d %5.1f%% %6d %5.1f%%"
              % (c, r["all"]["ev"], 100 * r["all"]["fm"],
                 r["geom"]["ev"], 100 * r["geom"]["fm"],
                 r["stat"]["ev"], 100 * r["stat"]["fm"]))
    print()
    print("判为重复框(IoU>=%.2f) 的比例：" % IOU_DUP)
    for c in cells:
        print("  %-26s %.1f%%  (n=%d)" % (c, 100 * out[c]["dup_rate"],
                                          out[c]["n_iou"]))
    print()
    print("逐机位（fm_geom｜fm_stat）")
    for c in cells:
        parts = []
        for cam, (ev, mg) in out[c]["geom"]["cam"].items():
            ev3, mg_ = out[c]["stat"]["cam"].get(cam, [0, 0])
            parts.append("%s %.0f%%(%d)/%.0f%%(%d)"
                         % (cam, 100 * mg / ev if ev else 0, ev,
                            100 * mg_ / ev3 if ev3 else 0, ev3))
        print("  %-26s %s" % (c, " | ".join(parts)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("\n-> " + a.json)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
