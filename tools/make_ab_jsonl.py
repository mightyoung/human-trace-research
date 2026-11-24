#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为选择器一致性评估生成轨迹分片 jsonl：每条轨迹按帧序列中点切成 前半(_a)/后半(_b)。

用途：两半各自独立编码后：
  - 同轨迹 (a,b) 的 cos = "分片判别力"正对基线
  - 选择器对 a/b 的分配是否一致 = 无标签核心指标

用法：python tools/make_ab_jsonl.py --in-jsonl <path> --out-prefix <dir/{seg}> --min-frames 6
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-prefix", required=True, help="输出前缀，如 .../323_train_510")
    ap.add_argument("--min-frames", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)

    rows = []
    with open(a.in_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    tid_frames = {}
    for r in rows:
        for t in (r.get("track_ids") or []):
            if t is not None and t >= 0:
                tid_frames.setdefault(int(t), []).append(r["frame"])

    half = {}
    for t, frs in tid_frames.items():
        if len(frs) < a.min_frames:
            continue
        frs = sorted(frs)
        k = len(frs) // 2
        half[t] = (set(frs[:k]), set(frs[k:]))   # (前半帧集, 后半帧集)
    n_tracks = len(half)

    fa = open(a.out_prefix + "_a.jsonl", "w", encoding="utf-8")
    fb = open(a.out_prefix + "_b.jsonl", "w", encoding="utf-8")
    n_a = n_b = 0
    for r in rows:
        tids, boxes = r.get("track_ids") or [], r.get("boxes") or []
        for tag, out in (("a", fa), ("b", fb)):
            sel_t, sel_b = [], []
            for i, t in enumerate(tids):
                if t is None or int(t) not in half:
                    continue
                if r["frame"] in half[int(t)][0 if tag == "a" else 1]:
                    sel_t.append(int(t))
                    sel_b.append(boxes[i])
            if sel_t:
                out.write(json.dumps({"frame": r["frame"], "track_ids": sel_t,
                                      "boxes": sel_b}, ensure_ascii=False) + "\n")
                if tag == "a":
                    n_a += 1
                else:
                    n_b += 1
    fa.close()
    fb.close()
    print(f"[done] {a.out_prefix}: tracks={n_tracks} a_rows={n_a} b_rows={n_b}")


if __name__ == "__main__":
    main()
