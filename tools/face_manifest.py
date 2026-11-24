#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出「人脸裁剪清单」给远程编码器（双线融合用）。

从 face_shot_{tag}.json 里按 (cam,tid) 聚合 usable 人脸，取 q 最高 top-K 帧，
输出清单 JSON —— 远程脚本据此读视频帧、裁人脸、送 C-RADIO→Gemma 桥接编码，
得到与主链身体向量**同空间**的 768d 人脸向量。

用法：
    python tools/face_manifest.py --tag p0c --topk 3 \
        --out output_crosscam/analysis/face/face_manifest_p0c.json
"""
import argparse
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "face")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0c")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min-q", type=float, default=0.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(os.path.join(FACE_DIR, f"face_shot_{args.tag}.json"), encoding="utf-8") as f:
        shots = json.load(f)["shots"]

    groups = defaultdict(list)
    for s in shots:
        if s.get("usable") and s.get("tid") is not None and s.get("q", 0) >= args.min_q:
            groups[(s["cam"], s["tid"])].append(s)

    items = []
    for (cam, tid), lst in sorted(groups.items()):
        lst.sort(key=lambda s: -s["q"])
        items.append(dict(key=f"{cam}|{tid}", cam=cam, tid=tid,
                          shots=[dict(frame=int(s["frame"]),
                                      face=[round(float(v), 1) for v in s["face"]],
                                      q=round(float(s["q"]), 4)) for s in lst[: args.topk]]))

    out = dict(version="face_manifest_v1", tag=args.tag, topk=args.topk,
               n_tids=len(items), n_shots=sum(len(i["shots"]) for i in items),
               note="供 face_gemma_encode_remote.py 消费：人脸裁剪 → RADIO→Gemma 桥接 → 768d",
               items=items)
    path = args.out or os.path.join(FACE_DIR, f"face_manifest_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[manifest] tids={out['n_tids']} shots={out['n_shots']} -> {path}")


if __name__ == "__main__":
    main()
