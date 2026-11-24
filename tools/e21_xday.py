# -*- coding: utf-8 -*-
"""E21 跨天结构验证驱动：Day A 建库 → Day B 查询。

与 e21_eval.py 的唯一差别：图库段与查询段来自**不同日期**。
  e21_eval.py : tr(A) -> te(A)   ← 日内自洽（复现性）
  e21_xday.py : tr(A) -> te(B)   ← 跨天（本条线要验的）

★ 三个指标都不依赖真值，是跨天唯一能诚实报告的东西：
  coverage   双视图(a/b)都过 θ 的查询占比 —— 受图库规模与人群重合度共同影响，
             **不能单独当召回读**（新人不重合时必然掉）。
  same       双视图 argmax 一致率 —— 域漂移敏感，真值无关。
  false_merge 共现对（同帧两人=已知负例）被判同簇的比例 —— 阈值代价，真值无关。

用法（远端 W 目录下）：
  python3 e21_xday.py --start 2026-10-22 --end 2026-10-23
  python3 e21_xday.py --pairs 2026-10-22>2026-10-23,2026-10-23>2026-10-26
  python3 e21_xday.py --start 2026-10-22 --end 2026-10-23 --grid 0.50,4.50,0.05
"""
import argparse
import os
import subprocess
import sys
import time

W = "/share/data4/yhh/human-count"
E = "/share/data4/yhh/data/human-count/e21"
PY = sys.executable
GRID = "0.50,4.50,0.05"

DAY_CAMS = {
    "2026-10-22": ["办公区1", "办公区2", "办公区球机", "茶水间", "行政走廊通道", "电梯间"],
    "2026-10-23": ["办公区1", "办公区2", "办公区球机", "茶水间", "走廊通道", "电梯间"],
    "2026-10-26": ["办公区1", "办公区2", "办公区球机", "茶水间", "行政走廊通道", "电梯间"],
}


def segs(day, split):
    return ",".join(f"{day}_{c}_{split}" for c in DAY_CAMS[day])


def run_pair(a, b, only, grid, out_root, fm_eval="all"):
    # ★ 同一天时 tr/te 都来自该日（对角线＝日内对照）；跨天时 tr 取 A、te 取 B
    out = os.path.join(out_root, f"{a}__to__{b}")
    cmd = [PY, os.path.join(W, "tools/d7_sweep.py"),
           "--emb", os.path.join(E, "emb"), "--det", os.path.join(E, "dd"),
           "--train-segs", segs(a, "tr"), "--test-segs", segs(b, "te"),
           "--out", out, "--grid", grid]
    if fm_eval:
        # all＝分母固定为全部共现对，fm 才可跨天/跨 θ 比较（默认 adaptive 会漂移）
        cmd += ["--fm-eval", fm_eval]
    if only:
        cmd += ["--only", only]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=W, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(f"=== {a} -> {b}  rc={p.returncode}  {time.time()-t0:.0f}s ===", flush=True)
    for ln in (p.stdout or "").splitlines():
        if ln.startswith("[") and "]" in ln:
            print("  " + ln, flush=True)
    if p.returncode != 0:
        print((p.stderr or "")[-800:], flush=True)
    return p.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--pairs", default="", help="形如 A>B,C>D")
    ap.add_argument("--only", default="A_baseline")
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--fm-eval", default="all",
                    help="all=分母固定（跨天可比）；adaptive=历史口径（分母随 θ 漂）")
    ap.add_argument("--diag", action="store_true",
                    help="额外跑对角线（A->A），作为同一管道下的日内对照")
    ap.add_argument("--out-root", default=os.path.join(E, "results_xday"))
    a = ap.parse_args()

    pairs = []
    if a.pairs:
        # 分隔符用 : 而非 >（> 会被 shell 当重定向）
        for x in a.pairs.split(","):
            if ":" in x:
                pairs.append(tuple(x.split(":")))
    elif a.start and a.end:
        pairs = [(a.start, a.end)]
    else:
        raise SystemExit("need --pairs or --start/--end")
    if a.diag:
        days = sorted({d for p in pairs for d in p})
        pairs = [(d, d) for d in days] + pairs

    os.makedirs(a.out_root, exist_ok=True)
    bad = 0
    for x, y in pairs:
        bad += run_pair(x, y, a.only, a.grid, a.out_root, a.fm_eval) != 0
    print(f"done, failed={bad}", flush=True)
