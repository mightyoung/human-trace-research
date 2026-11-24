#!/usr/bin/env python
"""扫描「持续嵌套轨迹」——同一个人的头肩框/全身框被检测成两条轨迹。

判据（纯框几何，不依赖掩膜）：
  框 b 有 >=CONTAIN 的比例落在另一个**不同轨迹**的框 big 内，且该情况占 b 全部观测的
  >=RATIO 帧，则 b 视为「被嵌套轨迹」。这类轨迹会让覆盖率虚高（同一个人被算成两三个身份），
  必须从任何「覆盖增益」里剔除后再看。

用法：
  PY tools/scan_nested_tracks.py \
      --det dir1=output_crosscam_local/1012_yolo8_dd \
      --det dir2=output_crosscam_local/1012_yolol_all_dd \
      --segs 305_test_544,323_test_527 [--json out.json]

  不给 --segs 则扫该目录下全部段。
"""
import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def contain(a, b):
    """b 被 a 覆盖的面积比例。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return i / ab if ab > 0 else 0.0


def scan_seg(path, contain_thr=0.8, ratio_thr=0.7):
    tot = collections.Counter()
    nest = collections.Counter()
    hrs = collections.defaultdict(list)   # 被嵌套框高度 / 宿主框高度
    frames = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        bs = d.get("boxes") or []
        ids = d.get("track_ids") or []
        if len(ids) != len(bs):
            continue
        frames += 1
        for i, b in enumerate(bs):
            tot[ids[i]] += 1
            for j, big in enumerate(bs):
                if j == i or ids[j] == ids[i]:
                    continue
                if contain(big, b) >= contain_thr:
                    nest[ids[i]] += 1
                    hb = b[3] - b[1]
                    hg = big[3] - big[1]
                    if hg > 0:
                        hrs[ids[i]].append(hb / hg)
                    break
    ratio = {t: nest[t] / tot[t] for t in tot if tot[t]}
    hi = [t for t, v in ratio.items() if v >= ratio_thr]
    # 只看「长期存在」的嵌套轨迹：观测数 >= 20 才有统计意义
    hi_long = [t for t in hi if tot[t] >= 20]
    med_h = st.median([st.median(hrs[t]) for t in hi if hrs[t]]) if hi else 0.0
    med_obs = st.median([tot[t] for t in hi]) if hi else 0.0
    return {
        "frames": frames,
        "tracks": len(tot),
        "nested_tracks": len(hi),
        "nested_ratio": round(len(hi) / len(tot), 4) if tot else 0.0,
        "nested_tracks_obs_ge20": len(hi_long),
        "nested_ratio_obs_ge20": round(len(hi_long) / len(tot), 4) if tot else 0.0,
        "height_ratio_median": round(float(med_h), 3),
        "nested_obs_median": float(med_obs),
        "top": sorted(((t, tot[t], round(nest[t] / tot[t], 2)) for t in hi),
                      key=lambda x: -x[1])[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", action="append", required=True,
                    help="NAME=DIR；读 DIR/{seg}/{seg}_detections.jsonl")
    ap.add_argument("--segs", default="", help="逗号分隔；留空=该目录下全部段")
    ap.add_argument("--contain", type=float, default=0.8)
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    specs = []
    for s in a.det:
        n, _, d = s.partition("=")
        specs.append((n, d))

    print(f"判据: 框被另一轨迹框包含 >= {a.contain} 且占该轨迹 >= {a.ratio:.0%} 的帧")
    print(f"{'variant':10s} {'seg':16s} {'轨迹':>5s} {'嵌套':>5s} {'占比':>6s} "
          f"{'≥20观测的嵌套':>12s} {'高占比中位':>10s}")
    out = {}
    for name, d in specs:
        if a.segs:
            segs = [x.strip() for x in a.segs.split(",") if x.strip()]
        else:
            segs = sorted(os.path.basename(os.path.dirname(p))
                          for p in glob.glob(os.path.join(d, "*", "*_detections.jsonl")))
        out[name] = {}
        for seg in segs:
            p = os.path.join(d, seg, f"{seg}_detections.jsonl")
            if not os.path.exists(p):
                print(f"{name:10s} {seg:16s} 缺失")
                continue
            r = scan_seg(p, a.contain, a.ratio)
            out[name][seg] = r
            print(f"{name:10s} {seg:16s} {r['tracks']:5d} {r['nested_tracks']:5d} "
                  f"{r['nested_ratio']*100:5.1f}% {r['nested_tracks_obs_ge20']:12d} "
                  f"{r['height_ratio_median']:10.2f}")
    if a.json:
        p = a.json if os.path.isabs(a.json) else os.path.join(ROOT, a.json)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[out] {p}")


if __name__ == "__main__":
    main()
