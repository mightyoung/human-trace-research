#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F2：过渡时间分布 + 联合评分（Relay v2 / S2）

依据（docs/relay-v2-plan.md §5 F2；docs/cam-handover-neighbor-research.md §2.2）：
  确认边 dt 呈"秒级主峰+长尾"双模态（p25=1.2s / med=4.8s / max=59s）；
  纯 sim 排序无法区分"真同人但外观中等 + dt 秒级"与"异人但外观高 + dt 长尾"
  （实例：343:E042 → 316:E046 真同人 sim=0.9445 dt=1.0 vs → 316:E050 异人 sim=0.9499 dt=47.6）。

本脚本（增量，不动 v5 产物）：
  Step 1 fit 集 —— primary 边剔除"已被人工判定过的"（防自证；provenance 显式记录）
  Step 2 Parzen 非参估计 f(dt)（高斯核，Silverman 带宽；样本<20 → 均匀兜底）
  Step 3 联合分 score = sim + λ·log(w_dt)，w_dt = f(dt)/f_max（λ 默认 0.01：先验值，
        使主峰-长尾贡献差 ~0.005-0.02，与 sim 噪声尺度匹配；正式校准待 241 判定后留出执行）
  Step 4 输出 dist_v1.json（含 quantiles/敏感性表）+ scored_v2.json（2618 候选全量加 w_dt/score）

用法: python tools/relay_transition_dist.py [--lambda 0.01]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relay_expand import expand_assign  # noqa: E402
from relay_triage import mutual_best_flags  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
SCORED = os.path.join(ANA, "relay_clean2", "relay_scored_cradio_v5.json")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
OUT_DIR = os.path.join(ANA, "relay_v2")
GRID = np.arange(-5.0, 65.01, 0.5)


def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def fit_bandwidth(dts):
    n = len(dts)
    if n < 2:
        return 5.0
    sigma = float(np.std(dts, ddof=1))
    h = 1.06 * sigma * n ** (-0.2)
    return float(np.clip(h, 2.0, 10.0))


def parzen(grid, dts, h):
    z = (grid[:, None] - dts[None, :]) / h
    k = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * h)
    return k.mean(axis=1)


def wdt_of(dt, f_grid, fmax):
    """f 在格点上 → 线性插值（clip 到格点范围）。"""
    x = float(np.clip(dt, GRID[0], GRID[-1]))
    i = int(np.searchsorted(GRID, x) - 1)
    i = max(0, min(i, len(GRID) - 2))
    t = (x - GRID[i]) / (GRID[i + 1] - GRID[i])
    f = f_grid[i] * (1 - t) + f_grid[i + 1] * t
    return float(max(f, 1e-9) / fmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-default", type=float, default=0.01, dest="lam")
    ap.add_argument("--scored", default=SCORED)
    ap.add_argument("--labeled", default=LABELED)
    args = ap.parse_args()

    d = json.load(open(args.scored, encoding="utf-8"))
    cands = d["candidates"]

    # ---- Step 1: fit 集（primary 边 − 已判定）----
    mb = mutual_best_flags(cands)
    primary = [r for r in cands if mb[id(r)] and r["sim"] >= 0.943]

    rows = list(csv.DictReader(open(args.labeled, encoding="utf-8-sig")))
    judged_keys = set()
    for r in rows:
        if r.get("human_same_person", "").strip() != "":
            judged_keys.add(parse_pid(r["pair_id"]))

    fit_dts, excluded = [], []
    for r in primary:
        if (r["a"], r["b"]) in judged_keys:
            excluded.append(r)
        else:
            fit_dts.append(r["dt"])
    fit_dts = np.asarray(fit_dts, dtype=float)
    print(f"[fit] primary {len(primary)} 边，剔除已判定 {len(excluded)}，"
          f"fit 样本 {len(fit_dts)} 条（防自证 provenance 见 dist_v1.json）")

    # ---- Step 2: Parzen ----
    if len(fit_dts) >= 20:
        h = fit_bandwidth(fit_dts)
        f_grid = parzen(GRID, fit_dts, h)
        mode = "parzen"
    else:  # 兜底：均匀
        h = None
        f_grid = np.ones_like(GRID)
        mode = "uniform_fallback"
    fmax = float(f_grid.max())
    qs = {f"p{q}": float(np.percentile(fit_dts, q)) for q in (5, 25, 50, 75, 95)}
    # 主峰位置（格点最大处）
    peak = float(GRID[int(f_grid.argmax())])
    print(f"[dist] mode={mode} h={h and round(h,2)} n={len(fit_dts)} | "
          f"quantiles p25={qs['p25']:.1f} p50={qs['p50']:.1f} p95={qs['p95']:.1f} | peak@{peak:.1f}s")
    for probe in (1.0, 2.6, 10.0, 41.6, 47.6, 59.0):
        print(f"   w_dt({probe:5.1f}s) = {wdt_of(probe, f_grid, fmax):.4f}")

    # ---- Step 3: 联合分 ----
    lam = args.lam
    scored2 = []
    for c in cands:
        w = wdt_of(c["dt"], f_grid, fmax)
        rec = dict(c)
        rec["w_dt"] = round(w, 5)
        rec["score"] = round(c["sim"] + lam * float(np.log(w + 1e-9)), 5)
        scored2.append(rec)

    # ---- Step 4: λ 敏感性（展示用；正式校准待 241 判定留出）----
    sensitivity = {}
    for lam_i in (0.0, 0.005, 0.01, 0.02, 0.05):
        pool_i = []
        for c in cands:
            w = wdt_of(c["dt"], f_grid, fmax)
            rec = dict(c)
            rec["w_dt"] = w
            rec["score"] = c["sim"] + lam_i * float(np.log(w + 1e-9))
            pool_i.append(rec)
        # 用 score 排序跑分配（sort_key 由 expand_assign 扩展支持）
        res = expand_assign(pool_i, 0.943, 3, 2, 0.95, sort_key="score")
        acc = {(r["a"], r["b"]) for r in res["primary"] + res["secondary"]}
        # 19 判定回放（只读评估，供敏感性展示）
        mist = [("343:343-E010", "316:316-E014"), ("343:343-E011", "316:316-E013"),
                ("343:343-E042", "316:316-E046"), ("343:343-E042", "316:316-E048"),
                ("343:343-E038", "316:316-E043")]
        fp2 = [("343:343-E042", "316:316-E050"), ("343:343-E042", "316:316-E049")]
        recovered = [f"{a}<->{b}" for a, b in mist if (a, b) in acc]
        fp_hit = [f"{a}<->{b}" for a, b in fp2 if (a, b) in acc]
        sensitivity[f"lambda_{lam_i}"] = {
            "n_secondary": len(res["secondary"]), "recovered": recovered,
            "n_recovered": len(recovered), "fp_audit_or_active": fp_hit}
        print(f"[λ={lam_i:5.3f}] secondary={len(res['secondary']):3d} | 恢复 {len(recovered)}/5 "
              f"| 异人入边 {len(fp_hit)}  ({', '.join(recovered)})")

    # ---- 保存 ----
    dist = {
        "version": "relay_v2/dist/1",
        "fit": {"source": "primary_edges_minus_judged",
                "n_primary": len(primary), "n_excluded_judged": len(excluded),
                "n_fit": int(len(fit_dts)),
                "excluded": [f"{r['a']}->{r['b']}" for r in excluded],
                "provenance": "fit 集剔除全部已人工判定行（含存疑）；评估样本不参与拟合"},
        "mode": mode, "h": h, "n": int(len(fit_dts)),
        "quantiles": qs, "peak_s": peak,
        "lambda_default": lam,
        "lambda_note": "先验值（主峰-长尾贡献差与 sim 尺度匹配）；正式校准待 241 判定后按留出纪律执行",
        "grid_x": GRID.round(2).tolist(), "grid_f": f_grid.round(8).tolist(),
        "sensitivity": sensitivity,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(dist, open(os.path.join(OUT_DIR, "dist_v1.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"lam": lam, "based_on": {"scored": os.path.relpath(args.scored, ROOT).replace("\\", "/")},
               "n": len(scored2), "candidates": scored2},
              open(os.path.join(OUT_DIR, "scored_v2.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[out] {OUT_DIR}/dist_v1.json + scored_v2.json")


if __name__ == "__main__":
    main()
