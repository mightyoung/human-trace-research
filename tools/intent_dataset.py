#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务二数据：轨迹运动意图数据集构建（历史窗 → 未来运动分类）。

输入：检测 jsonl（GroundingDINO+ByteTrack）——支持两种数据源：
  - 1012 切段（fps=25, stride=5 → 检测帧间隔 0.2s）
  - long35（fps=30, stride=5 → 检测帧间隔 0.1667s）
输出：npz {X:[N,L_in,F], y:[N], v_fut:[N], theta:[N], meta} + stats json

标签（检查时刻 t = 输入窗末尾）：
  stop      —— 未来窗平均速度 v_f < thr_stop
  backward  —— 未来方向 vs 输入方向变化 |θ| ≥ 140°（折返）
  turn_left / turn_right —— 40° ≤ |θ| < 140°（图像坐标 y 向下：θ>0 = 视觉顺时针/右转）
  forward   —— 其余（|θ| < 40°）

坐标归一化按各机位真实分辨率（2560×1440 或 1920×1080）。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RES = {"305": (2560, 1440), "320": (2560, 1440), "316": (2560, 1440), "343": (2560, 1440),
       "318": (1920, 1080), "323": (1920, 1080)}

CLASS_NAMES = {0: "stop", 1: "forward", 2: "backward", 3: "turn_left", 4: "turn_right"}


def load_tracks(jsonl_path):
    acc = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fr = r["frame"]
            for t, b in zip(r.get("track_ids") or [], r.get("boxes") or []):
                if t is None or t < 0:
                    continue
                acc.setdefault(int(t), []).append((int(fr), [float(x) for x in b]))
    for t in acc:
        acc[t].sort(key=lambda x: x[0])
    return acc


def foot(box):
    return ((box[0] + box[2]) / 2.0, box[3])


def wrap_deg(a):
    while a > 180.0:
        a -= 360.0
    while a <= -180.0:
        a += 360.0
    return a


def classify(v_f, theta, thr_stop):
    if v_f < thr_stop:
        return 0
    if abs(theta) >= 140.0:
        return 2
    if theta >= 40.0:
        return 4
    if theta <= -40.0:
        return 3
    return 1


def build_from_srcs(srcs, out_npz, stats_path, fps=25.0, stride=5, L_in=12, L_fut=5,
                    win_stride=2, thr_stop=0.01, dt=None, thr_move=0.01):
    """srcs: [(seg, cam, jsonl_path), ...]
    thr_move: 仅保留"检查时刻处于移动"的样本（v_in ≥ thr_move），0=不过滤。
    理由：办公区静止目标（坐姿）占比 ~88%，全量训练会退化为常数预测；
    预测对象应为"移动中的行人"（与用户语义一致：预判继续走/停下/转向）。
    """
    DT = dt or (stride / fps)
    min_len = L_in + L_fut + 2
    X, Y, VF, TH, VI, META, XF = [], [], [], [], [], [], []
    seg_stats = {}
    for seg, cam, jp in srcs:
        if not os.path.exists(jp):
            print(f"[skip] missing {jp}")
            continue
        W, H = RES[cam]
        tracks = load_tracks(jp)
        n_samp, n_track_used, n_static_skip = 0, 0, 0
        for tid, seq in tracks.items():
            if len(seq) < min_len:
                continue
            frames = [f for f, _ in seq]
            t_ok = all((frames[i + 1] - frames[i]) <= stride * 1.5
                       for i in range(len(frames) - 1))
            if not t_ok:
                blocks, cur = [], [0]
                for i in range(len(frames) - 1):
                    if frames[i + 1] - frames[i] <= stride * 1.5:
                        cur.append(i + 1)
                    else:
                        blocks.append(cur)
                        cur = [i + 1]
                blocks.append(cur)
                blocks.sort(key=len, reverse=True)
                if not blocks or len(blocks[0]) < min_len:
                    continue
                seq = [seq[i] for i in blocks[0]]
            n_track_used += 1
            feats = []
            prev_uv = None
            for f, b in seq:
                u, v = foot(b)
                u, v = u / W, v / H
                w = (b[2] - b[0]) / W
                h = (b[3] - b[1]) / H
                if prev_uv is None:
                    du = dv = 0.0
                else:
                    du = u - prev_uv[0]
                    dv = v - prev_uv[1]
                feats.append([u, v, du, dv, w, h, DT])
                prev_uv = (u, v)
            feats = np.asarray(feats, dtype=np.float32)
            pts = feats[:, :2]
            n = len(feats)
            last_start = n - (L_in + L_fut)
            for s in range(0, last_start + 1, win_stride):
                xi = s + L_in - 1
                xf = s + L_in + L_fut - 1
                # 输入速度（最近 4 步窗，≈0.8s@25fps / 0.67s@30fps）
                back = max(s, xi - 4)
                d_in = pts[xi] - pts[back]
                v_in = float(np.hypot(d_in[0], d_in[1]) / (max(1, xi - back) * DT))
                if thr_move > 0 and v_in < thr_move:
                    n_static_skip += 1
                    continue
                d = pts[xf] - pts[xi]
                v_f = float(np.hypot(d[0], d[1]) / (L_fut * DT))
                n_in = np.hypot(d_in[0], d_in[1])
                n_f = np.hypot(d[0], d[1])
                if n_in < 1e-8 or n_f < 1e-8:
                    theta = 0.0
                else:
                    ang_in = math.atan2(d_in[1], d_in[0])
                    ang_f = math.atan2(d[1], d[0])
                    theta = wrap_deg(math.degrees(ang_f - ang_in))
                y = classify(v_f, theta, thr_stop)
                X.append(feats[s:xi + 1].copy())
                XF.append(feats[xi + 1:xf + 1, :2].copy())   # 未来窗 u,v（可视化/分析用）
                Y.append(y)
                VF.append(v_f)
                TH.append(theta)
                VI.append(v_in)
                META.append((seg, cam, tid, int(seq[xi][0])))
                n_samp += 1
        seg_stats[seg] = {"tracks": len(tracks), "tracks_used": n_track_used,
                          "samples": n_samp, "static_skipped": n_static_skip}
        print(f"[seg] {seg}: tracks={len(tracks)} used={n_track_used} "
              f"samples={n_samp} static_skip={n_static_skip}")
    if not X:
        print("[error] no samples")
        return None
    X = np.stack(X).astype(np.float32)
    XF = np.stack(XF).astype(np.float32)
    Y = np.asarray(Y, dtype=np.int64)
    VF = np.asarray(VF, dtype=np.float32)
    TH = np.asarray(TH, dtype=np.float32)
    VI = np.asarray(VI, dtype=np.float32)
    np.savez_compressed(out_npz, X=X, Xf=XF, y=Y, v_fut=VF, theta=TH, v_in=VI,
                        meta=np.asarray(META, dtype=object))
    cnt = {CLASS_NAMES[k]: int((Y == k).sum()) for k in range(5)}
    stats = {"n": int(len(Y)), "L_in": L_in, "L_fut": L_fut, "win_stride": win_stride,
             "thr_stop": thr_stop, "thr_move": thr_move, "fps": fps, "stride": stride,
             "class_counts": cnt, "segs": seg_stats,
             "v_fut_pct": {str(p): float(np.percentile(VF, p)) for p in [1, 5, 25, 50, 75, 90, 95, 99]},
             "v_in_pct": {str(p): float(np.percentile(VI, p)) for p in [1, 5, 25, 50, 75, 90, 95, 99]}}
    json.dump(stats, open(stats_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] {len(Y)} samples -> {os.path.basename(out_npz)}")
    print(f"[dist] {cnt}")
    return stats


DEF_SEGS_1012 = {
    "train": ["320_train_514", "318_train_525", "323_train_510", "305_train_529"],
    "val":   ["320_val_519", "318_val_525", "323_val_538", "305_val_517"],
    "test":  ["320_test_518", "318_test_534", "323_test_527", "305_test_544"],
}


def srcs_1012(segs):
    out = []
    for s in segs:
        cam = s.split("_")[0]
        jp = os.path.join(ROOT, "output_crosscam", "1012_final", f"{s}.jsonl")
        out.append((s, cam, jp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", choices=["1012", "long35"], default="1012")
    ap.add_argument("--segs", default="", help="逗号分隔（1012: seg 名; long35: cam 名）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--thr-stop", type=float, default=0.01)
    ap.add_argument("--thr-move", type=float, default=0.01,
                    help="仅保留移动样本（v_in≥thr）；0=不过滤")
    ap.add_argument("--l-in", type=int, default=12)
    ap.add_argument("--l-fut", type=int, default=5)
    ap.add_argument("--win-stride", type=int, default=2)
    a = ap.parse_args()
    if a.src == "1012":
        segs = [s for s in a.segs.split(",") if s]
        srcs = srcs_1012(segs)
        fps, stride = 25.0, 5
    else:  # long35
        cams = [c for c in (a.segs or "316,318,320").split(",") if c]
        d = os.path.join(ROOT, "output_crosscam", "long35")
        # 316 dualclean；318 修正后 clean=raw（用原始检测）；320 cleaned（0 剔除）
        mp = {"316": os.path.join("cleaning", "316_masked_cleaned.jsonl"),
              "318": "318_long35_detections.jsonl",
              "320": os.path.join("cleaning", "320_cleaned.jsonl")}
        srcs = [(f"long35_{c}", c, os.path.join(d, mp[c])) for c in cams]
        fps, stride = 30.0, 5
    stats = a.stats or a.out.replace(".npz", "_stats.json")
    build_from_srcs(srcs, a.out, stats, fps=fps, stride=stride, L_in=a.l_in,
                    L_fut=a.l_fut, win_stride=a.win_stride, thr_stop=a.thr_stop,
                    thr_move=a.thr_move)


if __name__ == "__main__":
    main()
