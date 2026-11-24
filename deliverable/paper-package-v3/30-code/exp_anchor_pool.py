#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人脸锚点「暂存类」对身份选择器的增益测量。

用户提案
--------
检测全身时同时检测人脸；人脸若匹配到库中某个身份，就把该人脸的全身目标放进
一个**暂存类**（不参与聚类），并在身份选择器里给它所属身份追加一份**实时特征值**，
用于辅助判定（实时与历史训练都用）。

与已有「人脸锚点合并聚类」的区别
--------------------------------
聚类里合是**不可逆**的：一条错脸会把整簇拉脏。选择器注入是**查询时**的、
可撤销的，所以它还能承担「否决」而不仅是「合并」。这是提案真正新增的能力。

本次测什么（不做假设，只量三件事）
----------------------------------
1. **锚点库存**：可用人脸的轨有多少、覆盖多少身份、每机位每 30 min 够不够。
2. **漂移能否被抵消**：上午建原型库 → 下午推理（跨时段），
   比较「库原型」与「同机位 ±W 分钟内的锚点原型」对**本人**的余弦。
   若锚点把 ≥0.80 命中从 36% 抬到 80%+，说明机制成立。
3. **覆盖天花板**：能被锚点改善的查询占多少 → 这是端到端增益的上限。

口径
----
- 库原型 = 上午 4 段（train）里该身份轨嵌入的均值；查询 = 下午 4 段（test）。
- 锚点原型 = 同机位、|Δt|≤W、有可用脸、同身份的其它轨均值（不含查询自身）。
  这里**假设人脸匹配 100% 正确**（上限口径），匹配误差另计。
- 指标：own 余弦 中位 / ≥0.80 命中率；top-1 准确率（候选=该机位有原型的身份）。
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

FPS = 25.0
# 秒级 OSD 起录（见 data-sync-audit-2026-11-09.md §2）
OSD0 = {
    "320_train_514": "08:03:54", "318_train_525": "08:22:57",
    "323_train_510": "08:35:01", "305_train_529": "09:28:41",
    "323_test_527": "13:45:36", "318_test_534": "14:22:55",
    "305_test_544": "14:30:00",   # ★ 秒位被绿植遮挡，真值在 14:30:00–14:30:59；
                                   #   本实验只用它做 ±W 分钟窗（W≥15），59 s 误差可忽略
    "320_test_518": "15:02:58",
}
MORNING = ["305_train_529", "318_train_525", "320_train_514", "323_train_510"]
AFTERNOON = ["305_test_544", "318_test_534", "320_test_518", "323_test_527"]


def hms(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def cam_of(k):
    return k.rsplit("#", 1)[0].split("_")[0]


def seg_of(k):
    return k.rsplit("#", 1)[0]


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def load_face(face_dir):
    """→ {(seg,tid): usable}"""
    out = {}
    for p in glob.glob(os.path.join(face_dir, "face_track_*.csv")):
        seg = os.path.basename(p)[len("face_track_"):-4]
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    out[(seg, int(float(r["tid"])))] = int(float(r.get("usable") or 0))
                except (TypeError, ValueError):
                    continue
    return out


def load_spans(det_dir, seg, keys):
    path = os.path.join(det_dir, seg, seg + "_detections.jsonl")
    acc = collections.defaultdict(lambda: [1 << 60, -1])
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fr = int(r["frame"])
            for t in (r.get("track_ids") or []):
                if t is None or int(t) < 0:
                    continue
                k = "%s#%d" % (seg, int(t))
                if k in keys:
                    a = acc[k]
                    a[0] = min(a[0], fr)
                    a[1] = max(a[1], fr)
    t0 = hms(OSD0[seg])
    return {k: (t0 + a / FPS, t0 + b / FPS) for k, (a, b) in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--window", type=float, nargs="+", default=[15, 30, 60],
                    help="锚点时间窗（分钟）")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cluster_pos import load_emb

    segs = MORNING + AFTERNOON
    E, keys = load_emb(args.emb_dir, segs)
    E = l2n(E)
    eidx = {k: i for i, k in enumerate(keys)}
    gt = json.load(open(args.gt, encoding="utf-8"))
    gt = {k: v for k, v in gt.items() if k in eidx}
    face = load_face(args.face)

    span = {}
    for s in segs:
        ks = {k for k in gt if seg_of(k) == s}
        span.update(load_spans(args.det, s, ks))

    rows = []
    for k in gt:
        if k not in span:
            continue
        t0, t1 = span[k]
        rows.append({"k": k, "pid": gt[k], "cam": cam_of(k), "seg": seg_of(k),
                     "tm": 0.5 * (t0 + t1), "t0": t0, "t1": t1,
                     "face": face.get((seg_of(k), int(k.rsplit("#", 1)[1])), 0),
                     "i": eidx[k]})
    print("轨 %d（上午 %d / 下午 %d）；有可用脸 %d（%.1f%%）"
          % (len(rows), sum(1 for r in rows if r["seg"] in MORNING),
             sum(1 for r in rows if r["seg"] in AFTERNOON),
             sum(1 for r in rows if r["face"] >= 1),
             100.0 * sum(1 for r in rows if r["face"] >= 1) / max(1, len(rows))))

    # ---- 1. 锚点库存：每机位每 30 min
    print("\n== 锚点库存（有可用脸的轨）==")
    print("%-6s %6s %6s %6s %8s" % ("机位", "轨数", "带脸", "占比", "身份数(带脸)"))
    for cam in sorted({r["cam"] for r in rows}):
        sub = [r for r in rows if r["cam"] == cam]
        fa = [r for r in sub if r["face"] >= 1]
        print("%-6s %6d %6d %5.1f%% %8d"
              % (cam, len(sub), len(fa), 100.0 * len(fa) / max(1, len(sub)),
                 len({r["pid"] for r in fa})))
    for cam in sorted({r["cam"] for r in rows}):
        for seg in AFTERNOON:
            sub = [r for r in rows if r["cam"] == cam and r["seg"] == seg]
            if not sub:
                continue
            bins = collections.defaultdict(lambda: [0, 0])
            for r in sub:
                b = int(r["tm"] // 1800)
                bins[b][0] += 1
                bins[b][1] += int(r["face"] >= 1)
            detail = " ".join("%d/%d" % (v[1], v[0]) for _, v in sorted(bins.items()))
            print("   下午 %-14s 每 30min 带脸/总：%s" % (seg, detail))

    # ---- 2/3. 库原型 vs 锚点原型
    lib = {}
    by_pid = collections.defaultdict(list)
    for r in rows:
        if r["seg"] in MORNING:
            by_pid[r["pid"]].append(r["i"])
    for pid, ii in by_pid.items():
        lib[pid] = E[ii].mean(0)
    lib = {p: v / max(np.linalg.norm(v), 1e-9) for p, v in lib.items()}

    queries = [r for r in rows if r["seg"] in AFTERNOON and r["pid"] in lib]
    print("\n== 查询集：下午轨 %d（其身份在上午库里有原型）==" % len(queries))

    # ---- 选择器口径：候选 = 全库身份；有锚点的身份用「库+锚点」均值作原型
    print("\n== 选择器口径（τ 扫描；coverage=敢判比例，same=敢判时 top-1 正确率）==")
    print("%-6s %-10s %-22s %-22s" % ("窗", "n_cand", "全库原型 cov / same", "库+锚点 cov / same"))
    for W in args.window:
        Wsec = W * 60
        recs = []
        for q in queries:
            anc = collections.defaultdict(list)
            for r in rows:
                if r["cam"] != q["cam"] or r["face"] < 1 or r["k"] == q["k"]:
                    continue
                if abs(r["tm"] - q["tm"]) <= Wsec:
                    anc[r["pid"]].append(r["i"])
            s_lib = {p: float(E[q["i"]] @ v) for p, v in lib.items()}
            s_aug = dict(s_lib)
            for p, ii in anc.items():
                v = E[ii].mean(0)
                v = v / max(np.linalg.norm(v), 1e-9)
                if p in lib:                      # 有库原型 → 取均值（实时值与历史值合并）
                    m = lib[p] + v
                    v = m / max(np.linalg.norm(m), 1e-9)
                s_aug[p] = float(E[q["i"]] @ v)
            recs.append((q["pid"], s_lib, s_aug, len(anc)))
        ncs = [r[3] for r in recs]
        line = []
        for key in (1, 2):
            out = []
            for tau in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
                ans = [(pid, sc) for pid, sl, sa, _ in recs
                       for sc in [sl if key == 1 else sa]
                       if max(sc.values()) >= tau]
                if not ans:
                    out.append("τ%.2f  0%%/—" % tau)
                    continue
                same = np.mean([max(sc, key=sc.get) == pid for pid, sc in ans])
                out.append("τ%.2f %4.0f%%/%4.1f%%"
                           % (tau, 100.0 * len(ans) / len(recs), 100 * same))
            line.append("  ".join(out))
        print("%-6s %-10s %s" % ("±%dm" % W,
                                 "中位%d 最大%d" % (int(np.median(ncs)), max(ncs)), ""))
        print("        全库   ", line[0])
        print("        库+锚点", line[1])

    for W in args.window[:1]:
        Wsec = W * 60
        stat = collections.defaultdict(list)
        top1_lib = top1_aug = n_aug = top1_lib_sub = n_eval = 0
        for q in queries:
            own_lib = float(E[q["i"]] @ lib[q["pid"]])
            stat["lib"].append(own_lib)
            # 同机位 ±W 内的锚点（同身份 = 上限口径）
            anc_own = [r for r in rows
                       if r["cam"] == q["cam"] and r["pid"] == q["pid"]
                       and r["face"] >= 1 and r["k"] != q["k"]
                       and abs(r["tm"] - q["tm"]) <= Wsec]
            anc_any = collections.defaultdict(list)
            for r in rows:
                if r["cam"] != q["cam"] or r["face"] < 1 or r["k"] == q["k"]:
                    continue
                if abs(r["tm"] - q["tm"]) > Wsec:
                    continue
                anc_any[r["pid"]].append(r["i"])
            # top-1：候选 = 该机位±W 有锚点的身份
            if anc_any:
                cand = {p: E[ii].mean(0) for p, ii in anc_any.items()}
                cand = {p: v / max(np.linalg.norm(v), 1e-9) for p, v in cand.items()}
                best = max(cand, key=lambda p: float(E[q["i"]] @ cand[p]))
                top1_aug += int(best == q["pid"])
                n_aug += 1
                if q["pid"] in cand:          # ★ 只在「本人也在候选里」时才算可判，
                    n_eval += 1               #   否则是「没锚点」不是「判错」
                    top1_lib_sub += int(best_lib == q["pid"])
                    stat["aug"].append(float(E[q["i"]] @ cand[q["pid"]]))
                    stat["lib_sub"].append(own_lib)
            # 对照：库原型 top-1（候选 = 该机位所有有库原型的身份）
            cand_lib = {p: v for p, v in lib.items()}
            best_lib = max(cand_lib, key=lambda p: float(E[q["i"]] @ cand_lib[p]))
            top1_lib += int(best_lib == q["pid"])
            if anc_own:
                stat["own_anchor"].append(1)
            else:
                stat["own_anchor"].append(0)

        def desc(v):
            if not v:
                return "—"
            a = np.array(v)
            return "中位 %.3f ｜ ≥0.80 %.1f%% ｜ n=%d" % (
                float(np.median(a)), 100.0 * float((a >= 0.80).mean()), len(a))

        cov = float(np.mean(stat["own_anchor"])) if stat["own_anchor"] else 0.0
        print("\n-- 锚点窗 ±%d min --" % W)
        print("  本人余弦·上午库原型   %s" % desc(stat["lib"]))
        print("  本人余弦·同机位锚点   %s" % desc(stat["aug"]))
        print("  有本人锚点的查询占比  %.1f%%（%d/%d）"
              % (100 * cov, int(sum(stat["own_anchor"])), len(stat["own_anchor"])))
        print("  top-1 全库原型（全体 %d 条）  %.1f%%"
              % (len(queries), 100.0 * top1_lib / max(1, len(queries))))
        print("  同子集（有本人锚点的 %d 条）本人余弦：库原型 %s"
              % (len(stat["lib_sub"]), desc(stat["lib_sub"])))
        print("  top-1 同子集（%d 条可判）  全库 %.1f%% ｜ 锚点 %.1f%%"
              % (n_eval, 100.0 * top1_lib_sub / max(1, n_eval),
                 100.0 * top1_aug / max(1, n_eval)))


if __name__ == "__main__":
    main()
