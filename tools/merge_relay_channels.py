# -*- coding: utf-8 -*-
"""P0：把「门户驱动通道」（gate channel）并入 relay 候选，独立标 channel，不覆盖 v2。

背景（见 docs/s1-305-egress-findings.md）：
  relay v2 的候选来自 v5 池（47 边）+ secondary/weak，305 只有 2 条进入池；
  门户驱动通道（tools/gil_gate_candidates.py）用完全不同的机制
  （出口门 + 0≤dt≤30s + cos≥0.94 + 每出口只留最佳一条）独立生成 57 条候选，
  其中 14 条是既有接力图里没有的，且**逐条复现了人工判定的全部 relay 命中**。

本脚本是「合并」这一步，纪律：
  · **不改** relay_v2/edges_v2.json（源码真相）；v2 的 182 条原样保留，仅补一个
    字段 `channel="main"`（便于下游按通道拆账）。
  · 新增 14 条标 `channel="gate"` / `status="gate"` / `edge_type="gate"` / `tier="gate"`。
  · 打分口径与 v2 完全一致：`score = sim + λ·ln w_dt`，λ=0.01，w_dt 由**冻结**的
    relay_v2/dist_v1.json 插值（不重拟合）。
  · 冲突处理：若某条 gate 候选的 (a,b) 已在 v2 中，则**丢弃**（v2 优先，通道不覆盖）。

输出：
  output_crosscam/analysis/relay_v2/edges_v2_gated.json
用法：
  python tools/merge_relay_channels.py
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RV2 = os.path.join(ROOT, "output_crosscam/analysis/relay_v2")
GATE = os.path.join(ROOT, "output_crosscam/analysis/gil/gate_candidates.json")


def w_dt_interp(dist):
    """冻结分布插值：与 relay_weak_candidates.py 完全同式。"""
    GRID = np.asarray(dist["grid_x"], dtype=float)
    FGRID = np.asarray(dist["grid_f"], dtype=float)
    FMAX = float(FGRID.max())
    lam = float(dist.get("lambda_default", 0.01))

    def wdt_of(dt):
        x = float(np.clip(dt, GRID[0], GRID[-1]))
        i = int(np.searchsorted(GRID, x) - 1)
        i = max(0, min(i, len(GRID) - 2))
        t = (x - GRID[i]) / (GRID[i + 1] - GRID[i])
        f = FGRID[i] * (1 - t) + FGRID[i + 1] * t
        return float(max(f, 1e-9) / FMAX)
    return wdt_of, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", default=os.path.join(RV2, "edges_v2.json"))
    ap.add_argument("--gate", default=GATE)
    ap.add_argument("--dist", default=os.path.join(RV2, "dist_v1.json"))
    ap.add_argument("--out", default=os.path.join(RV2, "edges_v2_gated.json"))
    a = ap.parse_args()

    d = json.load(open(a.v2, encoding="utf-8"))
    gc = json.load(open(a.gate, encoding="utf-8"))
    dist = json.load(open(a.dist, encoding="utf-8"))
    wdt_of, lam = w_dt_interp(dist)

    # ---- 1) v2 原样（只补 channel 字段）----
    v2_edges = []
    for e in d["edges"]:
        e = dict(e)
        e["channel"] = "main"
        v2_edges.append(e)
    have = {(e["a"], e["b"]) for e in v2_edges}
    print(f"[merge] v2 边 {len(v2_edges)}（status={dict(Counter(e['status'] for e in v2_edges))}）")

    # ---- 2) gate 候选：只保留 v2 里没有的 ----
    gate_new, skipped = [], []
    for c in gc["candidates"]:
        key = (c["a"], c["b"])
        if key in have:
            skipped.append(key)
            continue
        w = wdt_of(c["dt"])
        gate_new.append(dict(
            a=c["a"], b=c["b"], dir=f"{c['a'].split(':')[0]}->{c['b'].split(':')[0]}",
            dt=c["dt"], sim=c["sim"],
            a_end=c.get("a_out"), b_start=c.get("b_in"),
            a_tlast=c.get("a_tlast"), b_tfirst=c.get("b_tfirst"),
            a_frag_last=None, b_frag_first=None,
            edge_type="gate", status="gate", budget_rank=None, tier="gate",
            w_dt=round(w, 5), score=round(c["sim"] + lam * float(np.log(w + 1e-9)), 5),
            channel="gate",
        ))
    print(f"[merge] gate 候选 {len(gc['candidates'])} → v2 已含 {len(skipped)}，新增 {len(gate_new)}")

    merged = v2_edges + gate_new
    # ---- 3) 一致性校验：无重复有向对 ----
    pairs = Counter((e["a"], e["b"]) for e in merged)
    dup = [k for k, v in pairs.items() if v > 1]
    assert not dup, f"重复边: {dup[:5]}"

    # ---- 4) 方向矩阵 / 度分布 ----
    dir_mat = Counter(e["dir"] for e in merged)
    dir_mat_main = Counter(e["dir"] for e in v2_edges)
    out_deg = Counter(e["a"] for e in merged)
    in_deg = Counter(e["b"] for e in merged)
    out_deg_main = Counter(e["a"] for e in v2_edges)

    def per_cam(deg):
        c = Counter()
        for k, v in deg.items():
            c[k.split(":")[0]] += v
        return {k: c[k] for k in sorted(c)}

    print("\n[方向矩阵] main → +gate 增量:")
    for dr in sorted(set(dir_mat) | set(dir_mat_main)):
        m, g = dir_mat_main.get(dr, 0), dir_mat.get(dr, 0)
        print(f"  {dr}: {m} → {g}  (+{g - m})")
    print("\n[出度按机位] main → merged:")
    om, og = per_cam(out_deg_main), per_cam(out_deg)
    for cam in sorted(set(om) | set(og)):
        print(f"  {cam}: {om.get(cam, 0)} → {og.get(cam, 0)}  (+{og.get(cam, 0) - om.get(cam, 0)})")

    print("\n[新增 14 条明细]")
    for e in gate_new:
        print(f"  {e['a']:14} -> {e['b']:14} dt={e['dt']:6.1f}s sim={e['sim']:.4f} "
              f"score={e['score']:.4f}")

    out = dict(
        version="relay_v2/edges/1+gated",
        based_on=dict(main=os.path.relpath(a.v2, ROOT),
                      gate=os.path.relpath(a.gate, ROOT),
                      dist=os.path.relpath(a.dist, ROOT)),
        lam=lam,
        params=dict(**d.get("params", {}), channels=["main", "gate"]),
        summary=dict(n_total=len(merged),
                     n_main=len(v2_edges), n_gate=len(gate_new),
                     n_main_active=sum(1 for e in v2_edges if e["status"] == "active"),
                     n_main_audit=sum(1 for e in v2_edges if e["status"] == "audit"),
                     direction_matrix=dict(dir_mat),
                     out_degree_by_cam=og),
        edges=merged,
    )
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {os.path.relpath(a.out, ROOT)}  共 {len(merged)} 边 "
          f"= {len(v2_edges)} main + {len(gate_new)} gate")


if __name__ == "__main__":
    main()
