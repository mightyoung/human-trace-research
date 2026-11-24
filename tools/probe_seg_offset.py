#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定位 8 段在卷视频中的帧偏移。

原理：段检测是「切片后重跑」的，帧号从 0 开始；但框坐标与同帧的卷检测完全一致
（同一检测器、同一像素）。所以拿段的第 0 帧框集合，在卷检测里逐帧找完全一致的
框集合，命中处即为偏移。

用法（远端）：
  python probe_seg_offset.py
输出：{seg: (vol, offset_frame, matched_n_boxes)}
"""
import glob
import json
import os

BASE = "/home/r4900g3/yhh/full1012"
SEGS = ["305_test_544", "305_train_529", "318_test_534", "318_train_525",
        "320_test_518", "320_train_514", "323_test_527", "323_train_510"]
VOLS = ["v2518", "v2526", "v2534", "v2540", "v2545", "v2559"]


def norm_boxes(bxs):
    return tuple(sorted(tuple(round(float(x), 1) for x in b) for b in bxs))


def seg_probe(seg):
    """取该段第一帧的框集合（frame 最小）。"""
    cands = glob.glob(os.path.join(BASE, "out8", seg, "*_detections.jsonl"))
    if not cands:
        cands = glob.glob(os.path.join(BASE, "out8", seg, "*.jsonl"))
    if not cands:
        return None
    best = None
    with open(cands[0], encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fr = int(r.get("frame", 0))
            if best is None or fr < best[0]:
                best = (fr, norm_boxes(r.get("boxes") or []))
            if fr > 200:
                break
    return best


def main():
    probes = {}
    for seg in SEGS:
        p = seg_probe(seg)
        if p:
            probes[seg] = p
            print("[probe] %-16s frame=%-6d boxes=%d" % (seg, p[0], len(p[1])))
        else:
            print("[probe] %-16s 缺检测文件" % seg)

    target = {seg: p[1] for seg, p in probes.items()}
    found = {}
    for vol in VOLS:
        path = os.path.join(BASE, "out", vol, vol + "_detections.jsonl")
        if not os.path.exists(path):
            print("[skip] 缺 %s" % path)
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                nb = norm_boxes(r.get("boxes") or [])
                if len(nb) < 2:
                    continue
                for seg, tb in target.items():
                    if seg in found:
                        continue
                    # 卷帧通常包含更多/更少框：做子集匹配
                    if set(nb) == set(tb) or (len(tb) >= 3 and set(tb).issubset(set(nb))):
                        found[seg] = (vol, int(r.get("frame", 0)), len(tb), len(nb))
                n += 1
        print("[scan] %s frames=%d 累计命中 %d" % (vol, n, len(found)))

    print("\n=== 结果 ===")
    for seg in SEGS:
        if seg in found:
            vol, fr, a, b = found[seg]
            print("%-16s -> %s  offset=%d  seg_boxes=%d vol_boxes=%d" % (seg, vol, fr, a, b))
        else:
            print("%-16s -> 未命中" % seg)


if __name__ == "__main__":
    main()
