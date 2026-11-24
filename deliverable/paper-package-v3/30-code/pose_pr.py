#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补两处口径（评审指出）：

Q1 【等判对率比覆盖】形态加权会把所有候选的分数整体抬高（加权均值更靠近查询），
    固定 τ=0.80 下的覆盖率上升可能是「分数膨胀」而非判别力提升。
    → 扫 τ 得 PR 曲线，报 **precision = P 时的 recall**。

Q2 【无库查询的误接受率】主实验分母只算「本人上午在该机位有库」的 179 条，
    另 29 条（来自 8 个身份，库里没有他们）被排除。形态加权抬高分数后，
    这些人被强行判给库内某人的比例会上升。
    → 报这些查询在各方法下的「误接受率」（score >= τ 即被误判为库内某人）。

用法：
  python pose_pr.py --emb-dir emb12 --face face8 --det old8 --gt gt_8seg.json --segs "..." 
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_match import PoseTemplates, AnchorPool, l2            # noqa: E402
from cluster_pos import load_emb                               # noqa: E402
from exp_pose_pool import load_det_meta, load_face             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--segs", required=True)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--pool-max", type=int, default=8)
    ap.add_argument("--dump", default="", help="逐查询明细（含 s0/s5/s1 全部分数），供评审复算")
    args = ap.parse_args()

    segs = args.segs.split()
    E64, keys = load_emb(args.emb_dir, segs)
    E = l2(E64)
    eidx = {k: i for i, k in enumerate(keys)}
    gt = json.load(open(args.gt, encoding="utf-8"))
    face = load_face(args.face)
    start, shape = load_det_meta(args.det, segs, set(eidx))

    rows = [dict(k=k, i=eidx[k], pid=gt[k], cam=k.rsplit("#", 1)[0].split("_")[0],
                 tr=("_train_" in k), t=start.get(k, 0), face=(k in face),
                 h=shape.get(k, (-1.0, -1.0))[0]) for k in keys if k in gt]
    tpl = PoseTemplates(E, sigma=args.sigma).build(
        [(r["cam"], r["pid"], r["i"], r["h"]) for r in rows if r["tr"]])

    queries = sorted([r for r in rows if not r["tr"]], key=lambda r: (r["cam"], r["t"], r["k"]))
    pool = AnchorPool(cap=args.pool_max)
    rec = []
    for q in queries:
        cands = tpl.pids(q["cam"])
        if not cands:
            continue
        s0 = tpl.score(q["i"], q["h"], q["cam"], sigma=0)
        s5 = tpl.score(q["i"], q["h"], q["cam"], sigma=args.sigma)
        sp = pool.score(E, q["i"], q["h"], q["cam"], cands, k=args.pool_k)
        s1 = {p: max(s5[p], sp[p]) for p in cands}
        own = s0.get(q["pid"], -1.0)
        rec.append(dict(q=q, own0=own, s0=s0, s5=s5, s1=s1))
        if q["face"] and q["pid"] in cands:
            pool.add(q["cam"], q["pid"], q["i"], q["h"], q["t"])

    METHS = [("S0 库均值", "s0"), ("S5 形态加权", "s5"), ("S5+池", "s1")]
    ev = [r for r in rec if r["own0"] >= 0]
    nolib = [r for r in rec if r["own0"] < 0]
    print("查询 %d ｜ 有库可评 %d ｜ **无库 %d**（评审指出的那批）"
          % (len(rec), len(ev), len(nolib)))

    # ---------------- Q2：无库查询的误接受率
    print("\n===== Q2 无库查询（真值不在该机位候选里，任何判定都是错的）=====")
    if nolib:
        by = collections.Counter(r["q"]["pid"] for r in nolib)
        print("  身份分布：%s" % "，".join("%s %d" % (p, n) for p, n in by.most_common()))
        print("  %-14s %-10s %-10s %-10s %-10s" % ("方法", "τ=.70", "τ=.75", "τ=.80", "τ=.85"))
        for name, mk in METHS:
            line = []
            for tau in (0.70, 0.75, 0.80, 0.85):
                acc = sum(1 for r in nolib if max(r[mk].values()) >= tau)
                line.append("%d/%d=%.0f%%" % (acc, len(nolib), 100.0 * acc / len(nolib)))
            print("  %-14s %-10s %-10s %-10s %-10s" % (name, *line))
        print("  被误判成的身份（τ=.80，S5）：")
        c = collections.Counter(max(r["s5"], key=r["s5"].get) for r in nolib
                                if max(r["s5"].values()) >= 0.80)
        print("    %s" % ("，".join("%s %d" % (p, n) for p, n in c.most_common(6)) or "无"))

    # ---------------- Q1：等判对率下的覆盖
    print("\n===== Q1 等判对率下的覆盖（扫 τ，precision = 通过判定中判对的比例）=====")
    grid = np.round(np.arange(0.50, 0.981, 0.005), 4)
    curves = {}
    for name, mk in METHS:
        pts = []
        for tau in grid:
            ans = [r for r in ev if max(r[mk].values()) >= tau]
            if not ans:
                continue
            hit = sum(1 for r in ans if max(r[mk], key=r[mk].get) == r["q"]["pid"])
            pts.append(dict(tau=float(tau), cov=len(ans) / len(ev),
                            prec=hit / len(ans), rec=hit / len(ev), n=len(ans)))
        curves[mk] = pts

    print("  分母 = %d 条有库可评查询。P 是「已判定里的判对率」，R 是「全体的判对率」。" % len(ev))
    print("  %-14s %-22s %-22s %-22s" % ("方法", "P≥0.93", "P≥0.95", "P≥0.97"))
    for name, mk in METHS:
        cells = []
        for P in (0.93, 0.95, 0.97):
            ok = [p for p in curves[mk] if p["prec"] >= P]
            if not ok:
                cells.append("%-22s" % "—")
                continue
            b = max(ok, key=lambda p: p["rec"])
            cells.append("%-22s" % ("R %.1f%% (cov %.0f%% τ%.2f n%d)"
                                    % (100 * b["rec"], 100 * b["cov"], b["tau"], b["n"])))
        print("  %-14s %s" % (name, "".join(cells)))

    print("\n  == 参考：固定 τ=0.80（旧报告口径）==")
    print("  %-14s %-8s %-8s %-8s" % ("方法", "覆盖", "判对率", "召回"))
    for name, mk in METHS:
        ans = [r for r in ev if max(r[mk].values()) >= 0.80]
        hit = sum(1 for r in ans if max(r[mk], key=r[mk].get) == r["q"]["pid"])
        print("  %-14s %5.0f%%   %5.1f%%   %5.1f%%"
              % (name, 100.0 * len(ans) / len(ev), 100.0 * hit / max(len(ans), 1),
                 100.0 * hit / len(ev)))

    # 逐查询明细：把加权分数一并落盘，评审可自行复算任何口径
    if args.dump:
        out = []
        for r in rec:
            out.append(dict(k=r["q"]["k"], cam=r["q"]["cam"], pid=r["q"]["pid"],
                            h=round(float(r["q"]["h"]), 1), t=round(float(r["q"]["t"]), 1),
                            face=bool(r["q"]["face"]), own0=round(float(r["own0"]), 4),
                            has_lib=bool(r["own0"] >= 0),
                            s0={p: round(v, 4) for p, v in r["s0"].items()},
                            s5={p: round(v, 4) for p, v in r["s5"].items()},
                            s1={p: round(v, 4) for p, v in r["s1"].items()}))
        json.dump(dict(n=len(out), n_lib=len(ev), n_nolib=len(nolib),
                       methods=["s0 库均值", "s5 形态加权", "s1 s5+池"], rows=out),
                  open(args.dump, "w"), ensure_ascii=False, indent=1)
        print("\n逐查询明细已写 %s（%d 条，含 s0/s5/s1 全部分数）" % (args.dump, len(out)))

    # PR 曲线原始点（便于画图/复核）
    out = os.environ.get("POSE_PR_OUT", "")
    if out:
        json.dump({k: v for k, v in curves.items()},
                  open(out, "w"), ensure_ascii=False)
        print("\nPR 曲线原始点已写 %s" % out)


if __name__ == "__main__":
    main()
