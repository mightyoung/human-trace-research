#!/usr/bin/env python
"""跨摄像头共现分析：318(办公区2) × 320(办公区球机)。

输入：两路 infer_supervision.py 产出的 jsonl（逐帧 track_ids + boxes）。
由于两路有重叠视野，且存在运动目标（走动人员），采用"用'人'当标定物"的
FOV 共现法（HANDOFF §6 路线 A）建立两路空间关系候选。

处理链条：
  1. 载入 jsonl，提取轨迹（track_id -> 落脚点序列；落脚点=框底边中点）
  2. 时间对齐：用全画面"运动能量"信号做互相关，确定两路帧偏移
  3. 筛出运动轨迹（走动的人）
  4. 配对评分：运动同步相关性 + 唯一运动目标强制匹配
  5. 输出对应点对 + 可视化（视野交叠图 + 对应散点）

用法:
    python tools/crosscam_analysis.py \
        --a output_crosscam/318_detections.jsonl --a-wh 1920 1080 --a-t0 08:55:20 \
        --b output_crosscam/320_detections.jsonl --b-wh 2560 1440 --b-t0 08:55:19 \
        --fps 25 --out-dir output_crosscam/analysis
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- loading

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_tracks(rows, W, H):
    """track_id -> {frames, feet(归一化), boxes(像素), scores}"""
    tracks = {}
    for r in rows:
        f = r["frame"]
        tids = r.get("track_ids", []) or []
        boxes = r.get("boxes", []) or []
        scores = r.get("scores", []) or []
        for i, b in enumerate(boxes):
            tid = tids[i] if i < len(tids) else -1
            if tid is None or tid < 0:
                continue
            x1, y1, x2, y2 = b
            u = (x1 + x2) / 2.0 / W
            v = y2 / H
            t = tracks.setdefault(tid, {"frames": [], "feet": [], "boxes": [],
                                        "scores": []})
            t["frames"].append(f)
            t["feet"].append((u, v))
            t["boxes"].append(b)
            t["scores"].append(scores[i] if i < len(scores) else 0.0)
    for t in tracks.values():
        order = np.argsort(t["frames"])
        t["frames"] = [t["frames"][i] for i in order]
        t["feet"] = [t["feet"][i] for i in order]
        t["boxes"] = [t["boxes"][i] for i in order]
    return tracks


def track_stats(t):
    fr = np.asarray(t["frames"])
    feet = np.asarray(t["feet"])
    if len(fr) == 0:
        return None
    d = np.diff(feet, axis=0) if len(feet) > 1 else np.zeros((0, 2))
    seg = np.sqrt((d ** 2).sum(axis=1)) if len(d) else np.zeros(0)
    duration_frames = fr[-1] - fr[0]
    return {
        "n": len(fr),
        "first": int(fr[0]),
        "last": int(fr[-1]),
        "duration_frames": int(duration_frames),
        "path_len": float(seg.sum()),
        "net_disp": float(np.sqrt(((feet[-1] - feet[0]) ** 2).sum())),
        "mean_speed": float(seg.mean()) if len(seg) else 0.0,
        "max_step": float(seg.max()) if len(seg) else 0.0,
        "median_foot": feet[len(feet) // 2].tolist(),
    }


# ---------------------------------------------------------------- alignment

def activity_signal(tracks, n_frames, stride, bin_frac=0.1):
    """全画面运动能量信号：每帧所有轨迹落脚点位移之和。"""
    sig = np.zeros(n_frames, dtype=float)
    for t in tracks.values():
        fr = np.asarray(t["frames"])
        feet = np.asarray(t["feet"])
        for k in range(1, len(fr)):
            step = fr[k] - fr[k - 1]
            d = np.sqrt(((feet[k] - feet[k - 1]) ** 2).sum())
            if step > 0:
                sig[fr[k]] += d / step * stride  # 折算回"每原始帧"位移
    return sig


def cross_correlate_shift(sig_a, sig_b, max_shift=100):
    """找 best shift：使 sig_a[f] ≈ sig_b[f + shift]（帧索引粒度）。
    返回 (best_shift, best_score, scores_curve)。
    """
    L = min(len(sig_a), len(sig_b))
    a = sig_a[:L] - sig_a[:L].mean()
    b = sig_b[:L] - sig_b[:L].mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0, 0.0, {}
    scores = {}
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            aa = a[: L - s]
            bb = b[s:]
        else:
            aa = a[-s:]
            bb = b[: L + s]
        if len(aa) < 100:
            continue
        c = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
        scores[s] = c
    if not scores:
        return 0, 0.0, scores
    best = max(scores, key=scores.get)
    return best, scores[best], scores


# ---------------------------------------------------------------- matching

def interp_feet(frames, feet, grid):
    """把轨迹落脚点插值到公共时间栅格（帧索引）。返回 (N,2) 与可见掩码。"""
    fr = np.asarray(frames, dtype=float)
    fe = np.asarray(feet, dtype=float)
    out = np.full((len(grid), 2), np.nan)
    ok = np.zeros(len(grid), dtype=bool)
    for i, g in enumerate(grid):
        # 该栅格点落在轨迹可见区间内（前后各留半格容忍）
        if len(fr) == 0:
            continue
        if g < fr[0] or g > fr[-1]:
            continue
        u = np.interp(g, fr, fe[:, 0])
        v = np.interp(g, fr, fe[:, 1])
        out[i] = (u, v)
        ok[i] = True
    return out, ok


def speed_on_grid(raw_frames, raw_feet, grid, lag_frames):
    """固定时间滞后的速度序列：sp[i] = |p(t+lag) - p(t)| / (lag/fps)。

    与采样步长解耦：无论 grid 步长多大，都用同一时间窗口测位移，
    避免"相邻样本差"在细采样时被检测抖动噪声淹没（实证：同一对
    在 0.4s 采样下 r=0.34，1s 下 r=0.79）。
    """
    tf = np.asarray(raw_frames, dtype=float)
    fe = np.asarray(raw_feet, dtype=float)
    t0 = grid.astype(float)
    t1 = t0 + lag_frames
    in0 = (t0 >= tf[0]) & (t0 <= tf[-1])
    in1 = (t1 >= tf[0]) & (t1 <= tf[-1])
    u0 = np.interp(t0, tf, fe[:, 0])
    v0 = np.interp(t0, tf, fe[:, 1])
    u1 = np.interp(t1, tf, fe[:, 0])
    v1 = np.interp(t1, tf, fe[:, 1])
    sp = np.hypot(u1 - u0, v1 - v0)
    ok = in0 & in1
    sp[~ok] = np.nan
    return sp, ok


def motion_sync(fa, oka, fb, okb, spa, spb):
    """运动同步分：共同可测栅格上速度序列的皮尔逊相关。"""
    msk = ~np.isnan(spa) & ~np.isnan(spb)
    if msk.sum() < 5:
        return None, 0, int(msk.sum())
    a = spa[msk]
    b = spb[msk]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0, 0, int(msk.sum())
    r = float(np.corrcoef(a, b)[0, 1])
    n_both_visible = int((oka & okb).sum())
    return r, int(msk.sum()), n_both_visible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 路 jsonl（318）")
    ap.add_argument("--b", required=True, help="B 路 jsonl（320）")
    ap.add_argument("--a-wh", nargs=2, type=int, required=True)
    ap.add_argument("--b-wh", nargs=2, type=int, required=True)
    ap.add_argument("--a-t0", default="", help="A 路切片起点 OSD（HH:MM:SS）")
    ap.add_argument("--b-t0", default="")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    # 运动目标筛选阈值
    ap.add_argument("--min-path", type=float, default=0.06,
                    help="轨迹路径长度阈值（归一化，相对画面短边比例）")
    ap.add_argument("--min-duration", type=float, default=2.0,
                    help="最短可见时长（秒）")
    ap.add_argument("--align-max-shift", type=int, default=100,
                    help="对齐搜索最大帧偏移")
    ap.add_argument("--force-shift", type=int, default=None,
                    help="跳过活动信号对齐，直接用指定帧偏移（已通过确认对精调时用）")
    ap.add_argument("--grid-step", type=int, default=10,
                    help="公共时间栅格步长（帧）")
    ap.add_argument("--speed-lag-sec", type=float, default=1.0,
                    help="速度测量的固定时间滞后（秒），与采样步长解耦")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    WA, HA = args.a_wh
    WB, HB = args.b_wh

    rows_a = load_jsonl(args.a)
    rows_b = load_jsonl(args.b)
    tracks_a = build_tracks(rows_a, WA, HA)
    tracks_b = build_tracks(rows_b, WB, HB)

    n_frames_a = max(r["frame"] for r in rows_a) + args.stride
    n_frames_b = max(r["frame"] for r in rows_b) + args.stride

    print(f"[load] A: {len(rows_a)} frames, {len(tracks_a)} tracks, "
          f"span={n_frames_a}")
    print(f"[load] B: {len(rows_b)} frames, {len(tracks_b)} tracks, "
          f"span={n_frames_b}")

    # ---------- 1) 时间对齐 ----------
    sig_a = activity_signal(tracks_a, n_frames_a, args.stride)
    sig_b = activity_signal(tracks_b, n_frames_b, args.stride)
    shift, score, curve = cross_correlate_shift(sig_a, sig_b,
                                                args.align_max_shift)
    if args.force_shift is not None:
        print(f"[align] override: shift {shift} -> {args.force_shift} (forced)")
        shift = args.force_shift
    # 约定：sig_a[f] ≈ sig_b[f + shift]
    # 即 B 的帧索引 = A 的帧索引 + shift 时同一时刻。
    print(f"[align] best shift = {shift} frames "
          f"({shift/args.fps:+.2f}s), corr={score:.3f}")
    top5 = sorted(curve.items(), key=lambda kv: -kv[1])[:5]
    print("[align] top-5 shifts:", [(s, round(c, 3)) for s, c in top5])

    # ---------- 2) 筛运动轨迹 ----------
    def movers(tracks):
        out = {}
        for tid, t in tracks.items():
            st = track_stats(t)
            if st is None:
                continue
            if st["path_len"] >= args.min_path and \
               st["duration_frames"] / args.fps >= args.min_duration:
                out[tid] = st
        return out

    ma = movers(tracks_a)
    mb = movers(tracks_b)
    print(f"[movers] A: {len(ma)} movers {sorted(ma)}")
    print(f"[movers] B: {len(mb)} movers {sorted(mb)}")

    # ---------- 3) 公共栅格上配对 ----------
    ALIGN = shift  # frame_b = frame_a + ALIGN 为同一时刻
    grid = np.arange(0, n_frames_a, args.grid_step)
    grid_b = grid + ALIGN
    grid_b = grid_b[(grid_b >= 0) & (grid_b < n_frames_b)]
    grid = grid[(grid + ALIGN >= 0) & (grid + ALIGN < n_frames_b)]
    lag_frames = int(round(args.speed_lag_sec * args.fps))

    results = []
    for tid_a, sta in ma.items():
        tA = tracks_a[tid_a]
        fa, oa = interp_feet(tA["frames"], tA["feet"], grid)
        for tid_b, stb in mb.items():
            tB = tracks_b[tid_b]
            fb, ob = interp_feet(tB["frames"], tB["feet"], grid_b)
            both = oa & ob
            covis = int(both.sum()) * args.grid_step / args.fps
            # 时间交集（用可见区间）
            oa_inter = (tA["frames"][0] <= grid) & (grid <= tA["frames"][-1])
            ob_inter = (tB["frames"][0] <= grid_b) & (grid_b <= tB["frames"][-1])
            inter_sec = float((oa_inter & ob_inter).sum()) * args.grid_step / args.fps
            if inter_sec < 2.0:
                continue
            spa, _ = speed_on_grid(tA["frames"], tA["feet"], grid, lag_frames)
            spb, _ = speed_on_grid(tB["frames"], tB["feet"], grid_b, lag_frames)
            r, n_seg, n_covis = motion_sync(fa, oa, fb, ob, spa, spb)
            results.append({
                "tid_a": int(tid_a), "tid_b": int(tid_b),
                "inter_sec": round(inter_sec, 1),
                "covis_sec": round(covis, 1),
                "sync_r": None if r is None else round(r, 3),
                "sync_n": n_seg,
                "a_path": round(sta["path_len"], 4),
                "b_path": round(stb["path_len"], 4),
                "a_first": sta["first"], "a_last": sta["last"],
                "b_first": stb["first"], "b_last": stb["last"],
            })

    results.sort(key=lambda r: (-(r["sync_r"] or -2), -r["covis_sec"]))
    print(f"[pairs] {len(results)} candidate pairs (inter>=2s)")
    for r in results[:12]:
        print("   ", r)

    # ---------- 4) 导出 ----------
    (out_dir / "crosscam_candidates.json").write_text(
        json.dumps({
            "align": {"shift_frames": shift, "shift_sec": shift / args.fps,
                      "corr": score, "forced": args.force_shift is not None,
                      "top5": [[int(s), round(c, 3)] for s, c in top5]},
            "n_frames_a": n_frames_a, "n_frames_b": n_frames_b,
            "grid_step": args.grid_step, "fps": args.fps,
            "movers_a": {int(k): v for k, v in ma.items()},
            "movers_b": {int(k): v for k, v in mb.items()},
            "pairs": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 对应点对：仅对同步分最高的若干对导出
    pair_rows = []
    for r in results[:6]:
        if r["sync_r"] is None:
            continue
        tA = tracks_a[r["tid_a"]]
        tB = tracks_b[r["tid_b"]]
        fa, oa = interp_feet(tA["frames"], tA["feet"], grid)
        fb, ob = interp_feet(tB["frames"], tB["feet"], grid_b)
        both = oa & ob
        for i in np.where(both)[0]:
            t_wall = grid[i] / args.fps
            pair_rows.append({
                "pair": f"{r['tid_a']}-{r['tid_b']}",
                "sync_r": r["sync_r"],
                "t_sec": round(float(t_wall), 2),
                "u_a": round(float(fa[i, 0]), 5), "v_a": round(float(fa[i, 1]), 5),
                "u_b": round(float(fb[i, 0]), 5), "v_b": round(float(fb[i, 1]), 5),
            })
    if pair_rows:
        with (out_dir / "correspondence_pairs.csv").open("w", newline="",
                                                         encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)
    print(f"[out] {out_dir}/crosscam_candidates.json")
    if pair_rows:
        print(f"[out] {out_dir}/correspondence_pairs.csv "
              f"({len(pair_rows)} rows)")


if __name__ == "__main__":
    main()
