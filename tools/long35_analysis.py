# -*- coding: utf-8 -*-
"""多目标轨迹追踪分析（长窗研究版）。

输入：检测+跟踪 jsonl（{cam}*_detections.jsonl，字段 frame/track_ids/boxes/scores/count）。
输出：并发度曲线、轨迹统计、碎片/串号代理指标、跨机位活跃度对照。

用法：
  python tools/long35_analysis.py --tag v10        # 现有 10 分钟数据
  python tools/long35_analysis.py --tag long35     # 35 分钟长窗数据（回传后）

时间轴：t = T0[cam] + frame / fps（与全项目一致）。
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis", "long35")
os.makedirs(ANA, exist_ok=True)

CAMS = ["316", "318", "320"]
T0 = {"316": 4.0, "318": 20.0, "320": 19.0}
FPS = {"316": 24.999, "318": 24.991, "320": 25.003}
W = {"316": 2560, "318": 1920, "320": 2560}
H = {"316": 1440, "318": 1080, "320": 1440}

# ---- 输入路径（tag: v10=现有 10 分钟；long35=长窗回传）----
DET_PATHS = {
    "v10":    {c: os.path.join(ROOT, "output_crosscam", f"{c}_detections.jsonl") for c in CAMS},
    "long35": {c: os.path.join(ROOT, "output_crosscam", "long35", f"{c}_long35_detections.jsonl") for c in CAMS},
}

# ---- 代理指标阈值 ----
CFG = dict(
    frag_gap_frames=60,      # 碎片重连：旧轨结束后 ≤60 帧（2.4s）出现新轨
    frag_dist_norm=0.12,     # 碎片重连：中心距 < 0.12×画面宽
    frag_h_ratio=(0.6, 1.6), # 碎片重连：框高比范围
    jump_norm=0.10,          # 空间跳跃：相邻检测帧位移 > 0.10×画面宽
    overlap_iou=0.5,         # 同帧双轨 IoU 冲突阈值
)


def load_det(path):
    """读 jsonl → rows 列表（frame, tids, boxes, count, t）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(d)
    return rows


def build_tracks(rows, cam):
    """tid → 轨迹（帧列表 + 框列表）。只保留 tid>=0。"""
    tracks = defaultdict(list)  # tid -> [(frame, box)]
    for r in rows:
        for t, b in zip(r["track_ids"], r["boxes"]):
            if t is not None and int(t) >= 0:
                tracks[int(t)].append((r["frame"], b))
    return tracks


def track_stats(tracks, cam):
    """轨迹级统计。"""
    fps = FPS[cam]
    out = []
    for tid, seq in tracks.items():
        seq.sort()
        f0, b0 = seq[0]
        f1, b1 = seq[-1]
        # 平均框高（归一化）
        hh = np.mean([(b[3] - b[1]) for _, b in seq])
        out.append(dict(tid=tid, n=len(seq), f0=f0, f1=f1,
                        dur=(f1 - f0) / fps, h_norm=hh / H[cam]))
    return out


def concurrency(rows, cam):
    """并发度：每检测帧活跃数 + 峰值区间。"""
    fps = FPS[cam]
    n_act = np.array([r["count"] for r in rows], dtype=float)
    frames = np.array([r["frame"] for r in rows])
    ts = T0[cam] + frames / fps
    # 60s 桶
    if len(ts):
        t_start, t_end = ts[0], ts[-1]
        buckets = np.arange(np.floor(t_start / 60) * 60, t_end + 60, 60)
        bmean, bmax, bcent = [], [], []
        for i in range(len(buckets) - 1):
            m = (ts >= buckets[i]) & (ts < buckets[i + 1])
            if m.sum() == 0:
                continue
            bmean.append(float(n_act[m].mean()))
            bmax.append(int(n_act[m].max()))
            bcent.append(float((buckets[i] + buckets[i + 1]) / 2))
    else:
        bmean, bmax, bcent = [], [], []
    # 高并发区间（≥3 连续 ≥20s）
    hi_intervals = []
    cur = None
    for r in rows:
        flag = r["count"] >= 3
        t = T0[cam] + r["frame"] / fps
        if flag and cur is None:
            cur = [t, t]
        elif flag:
            cur[1] = t
        elif cur is not None:
            if cur[1] - cur[0] >= 20:
                hi_intervals.append(cur)
            cur = None
    if cur is not None and cur[1] - cur[0] >= 20:
        hi_intervals.append(cur)
    hist, edges = np.histogram(n_act, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 12.5])
    return dict(
        n_frames=len(rows),
        mean=float(n_act.mean()), max=float(n_act.max()),
        p50=float(np.percentile(n_act, 50)), p95=float(np.percentile(n_act, 95)),
        hist={f"{int(edges[i])}-{int(edges[i+1]-1)}": int(hist[i]) for i in range(len(hist))},
        bucket_60s=dict(t=bcent, mean=bmean, max=bmax),
        hi_intervals=[[round(a, 1), round(b, 1)] for a, b in hi_intervals[:30]],
    )


def proxies(rows, tracks, cam):
    """碎片重连 / 空间跳跃 / 同帧重叠 三类代理指标。"""
    fps, w, h = FPS[cam], W[cam], H[cam]
    # 每检测帧的 tid -> box
    by_frame = {}
    for r in rows:
        m = {}
        for t, b in zip(r["track_ids"], r["boxes"]):
            if t is not None and int(t) >= 0:
                m[int(t)] = b
        by_frame[r["frame"]] = m

    # C1 疑似碎片重连
    frag_events = []
    # tid 首末检测
    starts, ends = {}, {}
    for tid, seq in tracks.items():
        seq.sort()
        starts[tid] = (seq[0][0], seq[0][1])
        ends[tid] = (seq[-1][0], seq[-1][1])
    # 按结束帧排序，便于窗口查找
    end_list = sorted(((f1, tid) for tid, (f1, _) in ends.items()))
    for tid, (f0, b0) in starts.items():
        cx0, cy0 = (b0[0] + b0[2]) / 2 / w, (b0[1] + b0[3]) / 2 / h
        h0 = (b0[3] - b0[1]) / h
        # 在 [f0-60, f0) 内找结束的轨迹
        for f1, other in end_list:
            if f1 >= f0:
                break
            if f0 - f1 > CFG["frag_gap_frames"]:
                continue
            b1 = ends[other][1]
            cx1, cy1 = (b1[0] + b1[2]) / 2 / w, (b1[1] + b1[3]) / 2 / h
            h1 = (b1[3] - b1[1]) / h
            dist = ((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2) ** 0.5
            if dist < CFG["frag_dist_norm"] and \
               CFG["frag_h_ratio"][0] <= (h0 / (h1 + 1e-9)) <= CFG["frag_h_ratio"][1]:
                frag_events.append(dict(new=f"t{tid}", old=f"t{other}",
                                        gap_s=round((f0 - f1) / fps, 2),
                                        dist=round(dist, 3)))

    # C2 同 tid 空间跳跃
    jump_events = []
    moves = []
    for tid, seq in tracks.items():
        seq.sort()
        for i in range(1, len(seq)):
            f_a, b_a = seq[i - 1]
            f_b, b_b = seq[i]
            ca = ((b_a[0] + b_a[2]) / 2 / w, (b_a[1] + b_a[3]) / 2 / h)
            cb = ((b_b[0] + b_b[2]) / 2 / w, (b_b[1] + b_b[3]) / 2 / h)
            d = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
            dt = (f_b - f_a) / fps
            dm = d / dt if dt > 0 else 0
            moves.append(dm)
            if d > CFG["jump_norm"]:
                jump_events.append(dict(tid=f"t{tid}", f_a=f_a, f_b=f_b,
                                        disp=round(d, 3), dt=round(dt, 2)))
    moves = np.array(moves) if moves else np.zeros(1)

    # C3 同帧双轨重叠
    overlap_events = 0
    for f, m in by_frame.items():
        tids = list(m.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                b1, b2 = m[tids[i]], m[tids[j]]
                x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                iw, ih = max(0, x2 - x1), max(0, y2 - y1)
                inter = iw * ih
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / (a1 + a2 - inter + 1e-9)
                if iou > CFG["overlap_iou"]:
                    overlap_events += 1
    return dict(
        frag_reconnects=frag_events,
        n_frag=len(frag_events),
        jump_events=jump_events[:100],
        n_jump=len(jump_events),
        move_speed_norm=dict(p50=float(np.percentile(moves, 50)),
                             p90=float(np.percentile(moves, 90)),
                             p99=float(np.percentile(moves, 99)),
                             max=float(moves.max())),
        same_frame_overlap=overlap_events,
    )


def time_segments(rows, tracks, cam, frag_events):
    """时间三分段对照：新增轨迹 / 活跃轨迹 / 碎片重连 / 并发，检验长时间稳定性。"""
    fps = FPS[cam]
    frames = np.array([r["frame"] for r in rows])
    ts = T0[cam] + frames / fps
    t_lo, t_hi = float(ts[0]), float(ts[-1])
    seg_len = (t_hi - t_lo) / 3
    out = []
    for k in range(3):
        a, b = t_lo + k * seg_len, t_lo + (k + 1) * seg_len
        # 新增轨迹（首检在窗口内）
        n_new = sum(1 for tid, seq in tracks.items()
                    if a <= T0[cam] + min(f for f, _ in seq) / fps < b)
        # 活跃轨迹（与窗口有重叠）
        n_active = sum(1 for tid, seq in tracks.items()
                       if (min(f for f, _ in seq) / fps + T0[cam] < b) and
                          (max(f for f, _ in seq) / fps + T0[cam] >= a))
        # 碎片重连（新轨首帧在窗口内）
        n_frag = 0
        for ev in frag_events:
            # ev["new"] = "t123"，需要找该 tid 首帧
            tid = int(ev["new"][1:])
            if tid in tracks:
                f0 = min(f for f, _ in tracks[tid])
                if a <= T0[cam] + f0 / fps < b:
                    n_frag += 1
        # 并发 mean（窗口内检测帧）
        m = (ts >= a) & (ts < b)
        cmean = float(np.mean([r["count"] for r, t in zip(rows, ts) if a <= t < b])) if m.sum() else 0
        out.append(dict(seg=k, t0=round(a, 1), t1=round(b, 1),
                        n_new=n_new, n_active=n_active, n_frag=n_frag,
                        concurrency_mean=round(cmean, 2)))
    return out


def analyze_cam(cam, tag):
    path = DET_PATHS[tag][cam]
    if not os.path.exists(path):
        print(f"[skip] {cam}: {path} 不存在")
        return None
    rows = load_det(path)
    tracks = build_tracks(rows, cam)
    ts = track_stats(tracks, cam)
    con = concurrency(rows, cam)
    prox = proxies(rows, tracks, cam)
    tseg = time_segments(rows, tracks, cam, prox["frag_reconnects"])

    durs = np.array([s["dur"] for s in ts])
    n_tracks = len(ts)
    seg = dict(
        lt2s=int((durs < 2).sum()), lt10s=int(((durs >= 2) & (durs < 10)).sum()),
        lt60s=int(((durs >= 10) & (durs < 60)).sum()), ge60s=int((durs >= 60).sum()))
    stat = dict(
        cam=cam, tag=tag, n_detect_frames=len(rows),
        n_tracks=n_tracks,
        dur_p50=float(np.percentile(durs, 50)), dur_p90=float(np.percentile(durs, 90)),
        dur_max=float(durs.max()), dur_mean=float(durs.mean()),
        seg=seg,
        frag_rate=round(prox["n_frag"] / max(n_tracks, 1), 3),
        jump_rate=round(prox["n_jump"] / max(len(rows), 1), 4),
        overlap=prox["same_frame_overlap"],
    )
    print(f"\n===== {cam} ({tag}) =====")
    print(f"检测帧 {len(rows)} | 轨迹 {n_tracks} | 时长 p50={stat['dur_p50']:.1f}s "
          f"p90={stat['dur_p90']:.1f}s max={stat['dur_max']:.0f}s")
    print(f"分段: <2s {seg['lt2s']} | 2-10s {seg['lt10s']} | 10-60s {seg['lt60s']} | >60s {seg['ge60s']}")
    print(f"并发: mean={con['mean']:.2f} max={con['max']:.0f} | 直方 {con['hist']}")
    print(f"高并发(≥3人)区间数: {len(con['hi_intervals'])}  前 5: {con['hi_intervals'][:5]}")
    print(f"代理: 碎片重连 {prox['n_frag']} (率 {stat['frag_rate']}) | "
          f"空间跳跃 {prox['n_jump']} | 同帧重叠 {prox['same_frame_overlap']}")
    print(f"运动速度(画面宽/秒) p50={prox['move_speed_norm']['p50']:.3f} "
          f"p90={prox['move_speed_norm']['p90']:.3f} p99={prox['move_speed_norm']['p99']:.3f} "
          f"max={prox['move_speed_norm']['max']:.3f}")
    print("时间三段（前/中/后 1/3）: 新轨 | 活跃 | 碎片重连 | 并发mean")
    for s in tseg:
        print(f"  [{s['t0']:.0f}-{s['t1']:.0f}s] {s['n_new']} | {s['n_active']} | "
              f"{s['n_frag']} | {s['concurrency_mean']}")
    return dict(stat=stat, concurrency=con, proxies={k: v for k, v in prox.items()
                                                     if k not in ("jump_events", "frag_reconnects")},
                frag_reconnects=prox["frag_reconnects"][:50],
                time_segments=tseg,
                track_list=sorted(ts, key=lambda s: -s["dur"])[:50])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="long35", choices=["v10", "long35"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {}
    for cam in CAMS:
        r = analyze_cam(cam, args.tag)
        if r:
            res[cam] = r
    out = args.out or os.path.join(ANA, f"analysis_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
