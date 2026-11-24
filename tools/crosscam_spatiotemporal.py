#!/usr/bin/env python
"""时空一致性判据（位置映射 + 轨迹连续性）——第三道独立信号。

原理（用户建议的判据）：
  同一个人在两路中的落脚点轨迹，经 A→B 空间映射后，在同一时刻应当重合：
      r(t) = | M(A_foot(t)) - B_foot(t + shift) |
  真配对：r(t) 小且连续平稳；假配对：r(t) 大或剧烈跳变。

映射 M 用已确认对的对应点拟合（二次多项式）；对训练对做留一交叉（LOO）
以避免自证，对未参与拟合的候选对用全量映射（无污染）。

用法:
    python tools/crosscam_spatiotemporal.py \
        --a-jsonl ... --b-jsonl ... --pairs-csv .../confirmed_pairs.csv \
        --out .../spatiotemporal_scores.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

# 已判读集合（8 正 / 24 负）——120:23 经时空判据+复核订正为真；98:26/142:26/53:11 经时空判据定案为伪；
# 123:18/111:18 为 2026-10-12 流水线压力测试新发现（久坐男子×步行女士，球机正下方压缩区误放）
POSITIVE = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEGATIVE = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1),
            (146, 21), (127, 19), (123, 19), (57, 11), (131, 23),
            (124, 18), (131, 24), (3, 4), (126, 18), (43, 11), (10, 1), (131, 28),
            (98, 26), (142, 26), (53, 11), (123, 18), (111, 18)]
# 用于拟合映射的"高置信"训练对（37:11 待用户复核，不参与拟合、仅作测试正样本）
TRAIN_PAIRS = [(1, 1), (14, 3), (69, 14), (120, 18), (139, 24), (146, 28)]
SHIFT = 20


def load_tracks(jsonl_path):
    """tid -> (frames[], u[], v[]) 归一化落脚点"""
    raw = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    raw.setdefault(int(tids[i]), []).append((r["frame"], b))
    out = {}
    for tid, lst in raw.items():
        lst.sort()
        fr = np.array([x[0] for x in lst], dtype=float)
        u = np.array([(x[1][0] + x[1][2]) / 2 for x in lst])
        v = np.array([x[1][3] for x in lst])
        out[tid] = (fr, u, v)
    return out


def norm_uv(tracks, W, H):
    for tid, (fr, u, v) in tracks.items():
        tracks[tid] = (fr, u / W, v / H)
    return tracks


def poly_feat(u, v):
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)


def fit_map(points):
    """points: list of (u_a, v_a, u_b, v_b) -> coef (6,2)"""
    A = np.array([[p[0], p[1]] for p in points])
    B = np.array([[p[2], p[3]] for p in points])
    X = poly_feat(A[:, 0], A[:, 1])
    coef, *_ = np.linalg.lstsq(X, B, rcond=None)
    return coef


def apply_map(coef, u, v):
    return poly_feat(np.asarray(u), np.asarray(v)) @ coef


def load_train_points(csv_path, pairs):
    pts = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pa, pb = (int(x) for x in r["pair"].split(":"))
            if (pa, pb) in pairs:
                pts.append((float(r["u_a"]), float(r["v_a"]),
                            float(r["u_b"]), float(r["v_b"])))
    return pts


def residual_series(coef, ta, tb, step=10):
    """映射后残差序列：r(t)=|M(A(t))-B(t+shift)|，仅统计双方可见的公共栅格。"""
    fra, ua, va = ta
    frb, ub, vb = tb
    lo = max(fra[0], frb[0] - SHIFT)
    hi = min(fra[-1], frb[-1] - SHIFT)
    if hi - lo < 25:
        return None, 0
    grid = np.arange(lo, hi + 1, step)
    gb = grid + SHIFT
    ua_i = np.interp(grid, fra, ua)
    va_i = np.interp(grid, fra, va)
    ub_i = np.interp(gb, frb, ub)
    vb_i = np.interp(gb, frb, vb)
    pred = apply_map(coef, ua_i, va_i)
    r = np.hypot(pred[:, 0] - ub_i, pred[:, 1] - vb_i)
    return r, len(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--a-wh", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--b-wh", nargs=2, type=int, default=[2560, 1440])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ta = norm_uv(load_tracks(args.a_jsonl), *args.a_wh)
    tb = norm_uv(load_tracks(args.b_jsonl), *args.b_wh)

    all_pts = load_train_points(args.pairs_csv, set(TRAIN_PAIRS))
    coef_full = fit_map(all_pts)
    print(f"[map] 训练点 {len(all_pts)} 个（{len(TRAIN_PAIRS)} 对确认）")

    rows = []
    # 正样本：训练对用 LOO 映射；37:11 用全量映射
    for (a, b) in POSITIVE:
        if (a, b) in TRAIN_PAIRS:
            pts = load_train_points(args.pairs_csv, set(TRAIN_PAIRS) - {(a, b)})
            coef = fit_map(pts)
            mode = "LOO"
        else:
            coef = coef_full
            mode = "out-of-fit"
        if a not in ta or b not in tb:
            continue
        r, n = residual_series(coef, ta[a], tb[b])
        rows.append({"lab": "P", "a": a, "b": b, "mode": mode,
                     "median_r": float(np.median(r)) if r is not None else None,
                     "p90_r": float(np.percentile(r, 90)) if r is not None else None,
                     "n": n})
    # 负样本：全量映射
    for (a, b) in NEGATIVE:
        if a not in ta or b not in tb:
            continue
        r, n = residual_series(coef_full, ta[a], tb[b])
        rows.append({"lab": "N", "a": a, "b": b, "mode": "out-of-fit",
                     "median_r": float(np.median(r)) if r is not None else None,
                     "p90_r": float(np.percentile(r, 90)) if r is not None else None,
                     "n": n})

    P = [r for r in rows if r["lab"] == "P" and r["median_r"] is not None]
    N = [r for r in rows if r["lab"] == "N" and r["median_r"] is not None]
    print(f"\n有效样本: {len(P)} 正 / {len(N)} 负")

    print(f"\n{'pair':>10} {'mode':>10} {'median_r':>9} {'p90_r':>8} {'n':>4}")
    for r in sorted(rows, key=lambda r: (r["median_r"] if r["median_r"] is not None else 9)):
        mr = f"{r['median_r']:.4f}" if r["median_r"] is not None else "None"
        p9 = f"{r['p90_r']:.4f}" if r["p90_r"] is not None else "None"
        print(f"A{r['a']:>4}:{r['b']:<4} {r['mode']:>10} {mr:>9} {p9:>8} {r['n']:>4}  {r['lab']}")

    # AUC（残差越小越可能同人 → 取负号）
    if P and N:
        c = 0.0
        for p in P:
            for n in N:
                if p["median_r"] < n["median_r"]:
                    c += 1
                elif p["median_r"] == n["median_r"]:
                    c += 0.5
        auc = c / (len(P) * len(N))
        print(f"\nmedian_r：min(P)={min(r['median_r'] for r in P):.4f}  "
              f"max(N)={max(r['median_r'] for r in N):.4f}  AUC={auc:.3f}")

    Path(args.out).write_text(
        json.dumps({"shift": SHIFT, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
