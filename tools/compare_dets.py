#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两个检测结果在**同一批帧**上的框级一致性：谁独有、独有框多大。

用来回答"换更大的模型能不能召回 s 漏掉的人"——不需要编码与评估。
两侧都先按 IoU>=0.5 去重（消除同一人双框的干扰），再一对一贪心匹配。

用法：
  python tools/compare_dets.py --a output_crosscam_local/_screen/s-seg_iou70/305_test_544/305_test_544_detections.jsonl \
      --b output_crosscam_local/_screen/m-seg_iou70/305_test_544/305_test_544_detections.jsonl
"""
import argparse
import json
import statistics as st


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def dedup(boxes, scores, thr=0.5):
    order = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(i)
    return [boxes[i] for i in sorted(keep)]


def load(p):
    out = {}
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        bs = d.get("boxes") or []
        sc = d.get("scores") or [1.0] * len(bs)
        if len(sc) != len(bs):
            sc = [1.0] * len(bs)
        out[d["frame"]] = dedup(bs, sc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--match", type=float, default=0.5)
    ap.add_argument("--area", type=float, default=1920 * 1080, help="画面像素数，用于面积占比")
    a = ap.parse_args()
    A, B = load(a.a), load(a.b)
    frames = sorted(set(A) & set(B))
    na = nb = ma = mb = only_a = only_b = 0
    area_a, area_b = [], []
    for f in frames:
        ba, bb = A[f], B[f]
        na += len(ba)
        nb += len(bb)
        used = set()
        for x in ba:
            best, bi = 0.0, -1
            for j, y in enumerate(bb):
                if j in used:
                    continue
                v = iou(x, y)
                if v > best:
                    best, bi = v, j
            if best >= a.match:
                used.add(bi)
                ma += 1
            else:
                only_a += 1
                area_a.append((x[2] - x[0]) * (x[3] - x[1]) / a.area * 100)
        for j in range(len(bb)):
            if j not in used:
                only_b += 1
                y = bb[j]
                area_b.append((y[2] - y[0]) * (y[3] - y[1]) / a.area * 100)
    n = max(1, len(frames))
    print(f"共同帧 {len(frames)}  A框/帧={na/n:.2f}  B框/帧={nb/n:.2f}  匹配率(A被B覆盖)={ma/max(1,na)*100:.1f}%")
    print(f"A 独有 {only_a} ({only_a/n:.2f}/帧) 中位面积 {st.median(area_a) if area_a else 0:.3f}%")
    print(f"B 独有 {only_b} ({only_b/n:.2f}/帧) 中位面积 {st.median(area_b) if area_b else 0:.3f}%")
    tiny = sum(1 for x in area_b if x < 0.05)
    print(f"B 独有里 <0.05% 画面 的小框占比 {tiny/max(1,len(area_b))*100:.0f}%")


if __name__ == "__main__":
    main()
