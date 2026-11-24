#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹统计：每段的人数/帧、轨迹条数、观测数分布（用于检测器横比时剥离"碎轨迹"效应）。

只做计数，不涉及任何判定指标；输入为逐帧 jsonl（frame/track_ids/boxes）。
用法：
  python tools/track_stats.py --name dino --glob "output_crosscam/1012_final/{seg}.jsonl" \
      --segs 305_train_529,...
  python tools/track_stats.py --name yolo --glob "output_crosscam_local/1012_yolo8/{seg}/{seg}_detections.jsonl"
"""
import argparse
import json
import os
import statistics as st


def stats(path):
    frames = 0
    boxes = 0
    per_frame = []
    obs = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        ids = d.get("track_ids") or []
        frames += 1
        boxes += len(ids)
        per_frame.append(len(ids))
        for i in ids:
            obs[i] = obs.get(i, 0) + 1
    if frames == 0:
        return None
    lens = sorted(obs.values())
    n = len(lens)
    med = st.median(lens) if lens else 0
    mean_pf = boxes / frames
    sd = st.pstdev(per_frame) if frames > 1 else 0.0
    short = sum(1 for x in lens if x < 6)
    return {"frames": frames, "boxes": boxes, "persons_per_frame": round(mean_pf, 2),
            "cv": round(sd / mean_pf, 3) if mean_pf else None,
            "tracks": n, "median_obs": med,
            "p10_obs": lens[max(0, int(0.1 * n) - 1)] if n else 0,
            "short_tracks_lt6": short,
            "short_share": round(short / n, 3) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--glob", required=True, help="含 {seg} 占位符的路径模板")
    ap.add_argument("--segs", default="305_train_529,318_train_525,320_train_514,323_train_510,"
                                     "305_test_544,318_test_534,320_test_518,323_test_527")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {}
    print(f"{'seg':16s} {'frames':>7s} {'p/frame':>8s} {'cv':>6s} {'tracks':>7s} "
          f"{'medObs':>7s} {'p10':>5s} {'<6obs':>7s}")
    for seg in a.segs.split(","):
        p = a.glob.format(seg=seg)
        if not os.path.exists(p):
            print(f"{seg:16s} MISSING {p}")
            continue
        s = stats(p)
        if not s:
            print(f"{seg:16s} EMPTY")
            continue
        res[seg] = s
        print(f"{seg:16s} {s['frames']:7d} {s['persons_per_frame']:8.2f} {s['cv']:6.3f} "
              f"{s['tracks']:7d} {s['median_obs']:7.0f} {s['p10_obs']:5d} "
              f"{s['short_tracks_lt6']:3d}({s['short_share']*100:.0f}%)")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"name": a.name, "stats": res}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
