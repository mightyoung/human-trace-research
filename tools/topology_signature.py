#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""六机位拓扑显著性检验（探索性，第一版）

目标：从"轨迹出画/入画事件"中检验机位对之间是否存在真实的**转移显著性**，
      并估计典型通行时间（dt 中位数），产出可复核的有向加权拓扑图。

方法（可辩护）：
  1) 事件提取：每条"移动"轨迹取 t_first / t_last（墙钟，OSD 对齐后的相对秒）。
  2) 观测统计：配对 (A 出画, B 入画) 且 dt = t_first(B) - t_last(A) ∈ [DT_MIN, DT_MAX] 的计数 N_obs。
  3) 原假设用**环形移位置换（circular-shift surrogate）**：把 B 的入画时刻整体随机平移
     （在观测窗内回绕），保持 B 的边缘分布与聚集结构，重复 S 次得到 N 的零分布
     → z=(N_obs-mean)/std，经验 p 值。该检验对"两边都活跃"造成的偶然配对有免疫力。
  4) 只保留 z>=Z_MIN 且 N_obs>=N_MIN 的边；dt 中位数 = 通行耗时估计。

性质声明：**探索性结果**，非 GT 验证。数据 = 10-10 六路同步 10 分钟切片（clean/dualclean 视图）。
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TC = os.path.join(ROOT, "output_crosscam", "analysis", "track_cleaning")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "topology")

CAMS = ["305", "316", "318", "320", "323", "343"]
SCENE = {"305": "办公区1", "316": "行政走廊", "318": "办公区2",
         "320": "办公区球机", "323": "茶水间", "343": "电梯间"}
RES = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
# 相对 08:55:00 的首帧墙钟偏移（秒）——与 tools/crosscam_relay.py 一致
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0
# 316/323 用 dualclean（掩膜+运动学），其余用 clean
FILE = {c: (f"{c}_masked_cleaned.jsonl" if c in ("316", "323") else f"{c}_cleaned.jsonl")
        for c in CAMS}


def load_tracks(cam):
    path = os.path.join(TC, FILE[cam])
    W, H = RES[cam]
    acc = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for tid, b in zip(d.get("track_ids", []), d.get("boxes", [])):
                if tid is None or tid < 0:
                    continue
                x1, y1, x2, y2 = b
                acc[tid].append((fr, (x1 + x2) / 2 / W, y2 / H, y2 - y1))
    out = {}
    for tid, rows in acc.items():
        rows.sort()
        fr = np.array([r[0] for r in rows], float)
        u = np.array([r[1] for r in rows])
        v = np.array([r[2] for r in rows])
        h = np.array([r[3] for r in rows])
        out[tid] = (fr, u, v, h)
    return out


def events(cam):
    """返回 (exits, entries)；每条 = dict(t, u, v, h, dur, span, tid)"""
    ex, en = [], []
    for tid, (fr, u, v, h) in load_tracks(cam).items():
        dur = (fr[-1] - fr[0]) / FPS
        span = float(np.hypot(u.max() - u.min(), v.max() - v.min()))
        if not (span >= 0.05 and dur >= 1.0):   # 只保留"移动"轨迹
            continue
        t0 = T0[cam] + fr[0] / FPS
        t1 = T0[cam] + fr[-1] / FPS
        en.append(dict(t=t0, u=float(u[0]), v=float(v[0]), h=float(h[0]), tid=tid, dur=dur))
        ex.append(dict(t=t1, u=float(u[-1]), v=float(v[-1]), h=float(h[-1]), tid=tid, dur=dur))
    return ex, en


def pair_count(ta, tb, dmin, dmax):
    """#{(a,b): dt=tb-ta ∈ [dmin,dmax]}"""
    if len(ta) == 0 or len(tb) == 0:
        return 0
    tb = np.sort(np.asarray(tb, float))
    n = 0
    for a in ta:
        lo = np.searchsorted(tb, a + dmin, "left")
        hi = np.searchsorted(tb, a + dmax, "right")
        n += hi - lo
    return int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmin", type=float, default=0.5)
    ap.add_argument("--dmax", type=float, default=60.0)
    ap.add_argument("--shifts", type=int, default=600)
    ap.add_argument("--zmin", type=float, default=3.0)
    ap.add_argument("--nmin", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    EV = {}
    for c in CAMS:
        ex, en = events(c)
        EV[c] = (ex, en)
        print(f"[{c} {SCENE[c]}] 移动轨迹: 出画 {len(ex)} / 入画 {len(en)} "
              f"(t 范围 {min(e['t'] for e in ex):.0f}~{max(e['t'] for e in ex):.0f}s)")

    # 统一观测窗
    allt = [e["t"] for c in CAMS for e in EV[c][0]]
    W = max(allt) - min(allt)
    print(f"\n[窗] 统一观测窗 ≈ {W:.0f}s，dt 窗 = [{args.dmin}, {args.dmax}]s，置换 {args.shifts} 次")

    rows = []
    for A in CAMS:
        for B in CAMS:
            if A == B:
                continue
            ta = [e["t"] for e in EV[A][0]]          # A 出画
            tb = [e["t"] for e in EV[B][1]]          # B 入画
            if len(ta) < 2 or len(tb) < 2:
                continue
            n_obs = pair_count(ta, tb, args.dmin, args.dmax)
            # 环形移位置换零分布
            tb0 = np.array(tb, float)
            null = np.empty(args.shifts, int)
            for i in range(args.shifts):
                s = rng.uniform(0, W)
                tbs = ((tb0 + s) % W) + min(allt)
                null[i] = pair_count(ta, tbs, args.dmin, args.dmax)
            mu, sd = float(null.mean()), float(null.std(ddof=0)) or 1e-9
            z = (n_obs - mu) / sd
            p = (1 + int((null >= n_obs).sum())) / (args.shifts + 1)
            # 收集贡献对的 dt（用于通行时间估计）
            dts = []
            tbs = np.sort(tb0)
            for a in ta:
                lo = np.searchsorted(tbs, a + args.dmin, "left")
                hi = np.searchsorted(tbs, a + args.dmax, "right")
                dts += list(tbs[lo:hi] - a)
            rows.append(dict(A=A, B=B, n_obs=n_obs, null_mean=round(mu, 2),
                             null_sd=round(sd, 2), z=round(z, 2), p=round(p, 4),
                             n_exit_A=len(ta), n_entry_B=len(tb),
                             dt_med=round(float(np.median(dts)), 1) if dts else None))
    rows.sort(key=lambda r: -r["z"])

    print(f"\n{'A->B':>10} {'N_obs':>6} {'null_mu':>8} {'z':>7} {'p':>7} {'dt_med':>7}")
    sig = []
    for r in rows:
        if r["z"] >= args.zmin and r["n_obs"] >= args.nmin:
            sig.append(r)
            print(f"  {r['A']:>3}->{r['B']:<3} {r['n_obs']:>6} {r['null_mean']:>8} "
                  f"{r['z']:>7} {r['p']:>7} {str(r['dt_med']):>7}   <== 显著")

    # 无向汇总（同一对取两向之和的 z 上限）
    und = {}
    for r in sig:
        k = tuple(sorted((r["A"], r["B"])))
        if k not in und or r["z"] > und[k]["z"]:
            und[k] = r
    print(f"\n[显著有向边] {len(sig)} 条 / [显著无向对] {len(und)} 对")

    out = dict(note="探索性：环形移位置换显著性检验（非 GT 验证）",
               window_s=round(W, 1), dt_window=[args.dmin, args.dmax],
               shifts=args.shifts, z_min=args.zmin, cam_scene=SCENE,
               edges_all=rows, edges_significant=sig,
               undirected_significant={f"{k[0]}-{k[1]}": v for k, v in und.items()})
    with open(os.path.join(OUT, "topology_signature.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 可视化 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
                break
        plt.rcParams["axes.unicode_minus"] = False

        # 左：有序机位对 z 值排序（含 3σ 显著性门限）
        top = rows[:18][::-1]
        lbl = [f"{r['A']}→{r['B']} (N={r['n_obs']})" for r in top]
        zs = [r["z"] for r in top]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.6),
                                 gridspec_kw={"width_ratios": [1, 1]})
        ax = axes[0]
        ax.barh(range(len(top)), zs,
                color=["#d62728" if z >= 3 else "#1f77b4" for z in zs])
        ax.axvline(3.0, color="#d62728", ls="--", lw=1.2)
        ax.text(3.05, 0.2, "3σ 显著性门限", color="#d62728", fontsize=9)
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(lbl, fontsize=8)
        ax.set_xlabel("显著性 z（环形移位置换零模型，600 次）")
        ax.set_title("转移显著性（Top18，10 分钟数据）")
        for i, z in enumerate(zs):
            ax.text(z + 0.03 if z >= 0 else z - 0.03, i, f"{z:.2f}", va="center",
                    ha="left" if z >= 0 else "right", fontsize=7)

        # 右：活动相关性（来自 multicam_overview 的 30s 分箱）
        ax = axes[1]
        ov = json.load(open(os.path.join(ROOT, "output_crosscam", "analysis",
                                         "multicam_overview.json"), encoding="utf-8"))
        M = np.zeros((6, 6))
        for i, a in enumerate(CAMS):
            for j, b in enumerate(CAMS):
                if a not in ov or b not in ov:
                    M[i, j] = np.nan; continue
                xa = np.array(ov[a]["active_per_30s"], float)
                xb = np.array(ov[b]["active_per_30s"], float)
                n = min(len(xa), len(xb))
                M[i, j] = np.corrcoef(xa[:n], xb[:n])[0, 1] if n > 2 else np.nan
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CAMS); ax.set_yticklabels(CAMS)
        for i in range(6):
            for j in range(6):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8,
                            color="k" if abs(M[i, j]) < 0.6 else "w")
        ax.set_title("机位活动相关性（30s 分箱，r）")
        fig.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        png = os.path.join(OUT, "topology_signature.png")
        plt.savefig(png, dpi=120)
        print(f"\n[fig] {png}")
    except Exception as e:
        print(f"[fig] 失败: {e}")

    print(f"[out] {os.path.join(OUT, 'topology_signature.json')}")


if __name__ == "__main__":
    main()
