#!/usr/bin/env python
"""对已人工确认的跨画面轨迹对，提取对应点并做精细时间对齐。

与自动配对不同：这里 tra_id 对是"人工确认"过的（证据：裁剪核对图），
用于生成交付级的对应点数据。

附加：用确认对自身的速度曲线做 shift 精调（在自动对齐的粗结果上收敛）。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_tracks(jsonl_path, W, H):
    tracks = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i >= len(tids) or tids[i] is None or tids[i] < 0:
                    continue
                tid = int(tids[i])
                x1, y1, x2, y2 = b
                tracks.setdefault(tid, []).append((r["frame"], (x1 + x2) / 2 / W, y2 / H))
    for k in tracks:
        tracks[k].sort()
    return tracks


def series(track, grid):
    fr = np.array([p[0] for p in track], dtype=float)
    u = np.array([p[1] for p in track])
    v = np.array([p[2] for p in track])
    ok = (grid >= fr[0]) & (grid <= fr[-1])
    uu = np.full(len(grid), np.nan)
    vv = np.full(len(grid), np.nan)
    uu[ok] = np.interp(grid[ok], fr, u)
    vv[ok] = np.interp(grid[ok], fr, v)
    return uu, vv, ok


def speed_profile(u, v, ok):
    sp = np.full(len(u), np.nan)
    for i in range(1, len(u)):
        if ok[i] and ok[i - 1]:
            sp[i] = np.hypot(u[i] - u[i - 1], v[i] - v[i - 1])
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-wh", nargs=2, type=int, required=True)
    ap.add_argument("--b-wh", nargs=2, type=int, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--grid-step", type=int, default=25, help="对应点采样步长(帧)")
    ap.add_argument("--shift", type=int, required=True,
                    help="初始 shift（frame_b = frame_a + shift）")
    ap.add_argument("--refine", action="store_true",
                    help="用确认对速度曲线精调 shift")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="确认的轨迹对，格式 a:b，如 1:1 139:24")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    WA, HA = args.a_wh
    WB, HB = args.b_wh
    ta = load_tracks(args.a_jsonl, WA, HA)
    tb = load_tracks(args.b_jsonl, WB, HB)

    pairs = [tuple(int(x) for x in p.split(":")) for p in args.pairs]
    for pa, pb in pairs:
        if pa not in ta:
            print(f"[warn] A tid {pa} not found")
        if pb not in tb:
            print(f"[warn] B tid {pb} not found")

    shift = args.shift
    if args.refine:
        print("[refine] 尝试用确认对速度曲线精调 shift ...")
        best = (None, -2.0)
        for s in range(max(0, shift - 60), shift + 61, 5):
            grid = np.arange(0, 14916, 25)
            gb = grid + s
            m = (gb >= 0) & (gb < 14906)
            g2, gb2 = grid[m], gb[m]
            rs = []
            for pa, pb in pairs:
                if pa not in ta or pb not in tb:
                    continue
                ua, va, oka = series(ta[pa], g2)
                ub, vb, okb = series(tb[pb], gb2)
                spa = speed_profile(ua, va, oka)
                spb = speed_profile(ub, vb, okb)
                msk = ~np.isnan(spa) & ~np.isnan(spb)
                if msk.sum() >= 8 and spa[msk].std() > 1e-9 and spb[msk].std() > 1e-9:
                    rs.append(np.corrcoef(spa[msk], spb[msk])[0, 1])
            if rs:
                r = float(np.mean(rs))
                print(f"   shift={s:+4d} ({s/args.fps:+.1f}s)  mean_r={r:.3f}  "
                      f"n_pairs={len(rs)}")
                if r > best[1]:
                    best = (s, r)
        if best[0] is not None:
            print(f"[refine] best shift = {best[0]} ({best[0]/args.fps:+.2f}s), "
                  f"r={best[1]:.3f}  (初始 {shift})")
            shift = best[0]

    # 导出对应点
    rows = []
    grid = np.arange(0, 14916, args.grid_step)
    gb = grid + shift
    m = (gb >= 0) & (gb < 14906)
    g2, gb2 = grid[m], gb[m]
    for pa, pb in pairs:
        if pa not in ta or pb not in tb:
            continue
        ua, va, oka = series(ta[pa], g2)
        ub, vb, okb = series(tb[pb], gb2)
        both = oka & okb
        for i in np.where(both)[0]:
            rows.append({
                "pair": f"{pa}:{pb}",
                "t_sec": round(float(g2[i]) / args.fps, 2),
                "u_a": round(float(ua[i]), 5), "v_a": round(float(va[i]), 5),
                "u_b": round(float(ub[i]), 5), "v_b": round(float(vb[i]), 5),
            })
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "t_sec", "u_a", "v_a", "u_b", "v_b"])
        w.writeheader()
        w.writerows(rows)
    n_by = {}
    for r in rows:
        n_by[r["pair"]] = n_by.get(r["pair"], 0) + 1
    print(f"[out] {args.out_csv}  shift={shift}  points={len(rows)}  by_pair={n_by}")


if __name__ == "__main__":
    main()
