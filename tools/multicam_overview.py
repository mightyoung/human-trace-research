#!/usr/bin/env python
"""六路覆盖总览（多机首轮分析）

输入：output_crosscam/{305,316,318,320,323,343}_detections.jsonl
输出：
  - multicam_overview.json  每路统计（唯一轨迹/稳定轨迹/活动曲线）
  - fig_multicam_timeline.png  六路"在场人数"时间线（30s 分箱，按唯一轨迹计）
用法:
  python tools/multicam_overview.py --dir output_crosscam --out-dir output_crosscam/analysis
"""
import argparse
import json
import os

import numpy as np

CAMERAS = [
    ("316", "行政走廊通道"),
    ("305", "办公区1"),
    ("318", "办公区2"),
    ("320", "办公区球机"),
    ("323", "茶水间"),
    ("343", "电梯间"),
]
BIN_SEC = 30
FPS = 25


def load_jsonl(path):
    frames, counts, tids_by_frame = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fr = int(r["frame"])
            ts = [int(t) for t in (r.get("track_ids", []) or [])
                  if t is not None and int(t) >= 0]
            frames.append(fr)
            counts.append(len(r.get("boxes", []) or []))
            tids_by_frame.append(ts)
    return frames, counts, tids_by_frame


def analyze(path):
    frames, counts, tids_by_frame = load_jsonl(path)
    if not frames:
        return None
    frames = np.asarray(frames)
    counts = np.asarray(counts)
    all_tids = {}
    for fr, ts in zip(frames, tids_by_frame):
        for t in ts:
            all_tids.setdefault(t, 0)
            all_tids[t] += 1
    uniq = len(all_tids)
    stable = sum(1 for t, n in all_tids.items() if n >= 10)  # ≥2s
    long_tracks = sum(1 for t, n in all_tids.items() if n >= 25)  # ≥5s

    lo, hi = int(frames.min()), int(frames.max())
    bins = np.arange(lo, hi + 1, BIN_SEC * FPS)
    act = []
    for i, b0 in enumerate(bins):
        b1 = b0 + BIN_SEC * FPS
        ids = set()
        for fr, ts in zip(frames, tids_by_frame):
            if b0 <= fr < b1:
                ids.update(ts)
        act.append(len(ids))
    return {
        "rows": len(frames),
        "count_mean": round(float(counts.mean()), 2),
        "count_max": int(counts.max()),
        "unique_tracks": uniq,
        "tracks_ge_2s": stable,
        "tracks_ge_5s": long_tracks,
        "active_per_30s": act,
        "t_lo_sec": round(lo / FPS, 1),
        "t_hi_sec": round(hi / FPS, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output_crosscam")
    ap.add_argument("--out-dir", default="output_crosscam/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out = {}
    print(f"{'cam':>4} {'scene':<8} {'count_mean':>10} {'count_max':>9} "
          f"{'uniq':>5} {'ge2s':>5} {'ge5s':>5}")
    for cid, name in CAMERAS:
        p = os.path.join(args.dir, f"{cid}_detections.jsonl")
        if not os.path.exists(p):
            print(f"{cid:>4} {name:<8}  (jsonl 缺失)")
            continue
        st = analyze(p)
        if st is None:
            continue
        out[cid] = {"scene": name, **st}
        print(f"{cid:>4} {name:<8} {st['count_mean']:>10} {st['count_max']:>9} "
              f"{st['unique_tracks']:>5} {st['tracks_ge_2s']:>5} {st['tracks_ge_5s']:>5}")

    with open(os.path.join(args.out_dir, "multicam_overview.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[out] {os.path.join(args.out_dir, 'multicam_overview.json')}")

    # 时间线图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), dpi=120, sharex=True)
        for ax, (cid, name) in zip(axes.ravel(), CAMERAS):
            if cid not in out:
                ax.set_title(f"{name} ({cid}) - no data", fontsize=10)
                continue
            act = out[cid]["active_per_30s"]
            t = np.arange(len(act)) * BIN_SEC / 60.0
            ax.fill_between(t, act, color="#4c72b0", alpha=0.55)
            ax.plot(t, act, color="#1f4e79", lw=1.3)
            ax.set_title(f"{name} ({cid})", fontsize=10)
            ax.grid(alpha=0.25)
            ax.set_ylabel("在场轨迹数", fontsize=8.5)
        for ax in axes[-1]:
            ax.set_xlabel("time within clip (min)", fontsize=8.5)
        fig.suptitle("六路'在场轨迹数'时间线（30s 分箱，08:55–09:05 同步切片）", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_multicam_timeline.png"), facecolor="white")
        print(f"[out] {os.path.join(args.out_dir, 'fig_multicam_timeline.png')}")
    except Exception as e:  # noqa: BLE001
        print("[warn] figure failed:", str(e)[:200])


if __name__ == "__main__":
    main()
