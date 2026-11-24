#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""编码设备对照：同一批裁剪分别在 CPU 与 DirectML 上过 OSNet，量化设备漂移。

用途：本次掩膜变体（encode_mask_reid.py）在 CPU 上编码，而 10-17 的基线嵌入在
DML 上编码。若两者特征本身差异不可忽略，掩膜效果就无法与基线直接比较。本脚本
给出一个可直接引用的数量级：同裁剪 cosine(CPU, DML)。

用法：python tools/osnet_device_control.py --n 240
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encode_1012_osnet_seq as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="318_test_534")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default=str(E.ROOT / "output_crosscam_local/1012_osnet_crops_mask/device_control.json"))
    a = ap.parse_args()

    tracks = E.load_ab(E.AB_DIR / f"{a.seg}_a.jsonl")
    reqs = []
    for tid in sorted(tracks):
        tf = tracks[tid]
        if len(tf) < 2:
            continue
        for k in np.linspace(0, len(tf) - 1, min(8, len(tf))).astype(int):
            reqs.append((int(tf[k][0]), tf[k][1]))
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(reqs), size=min(a.n, len(reqs)), replace=False).tolist())

    cap = cv2.VideoCapture(str(E.VID_DIR / f"{a.seg}.mp4"))
    crops, cur, frame = [], -1, None
    for i in pick:
        fnum, box = reqs[i]
        while cur < fnum:
            if not cap.grab():
                break
            cur += 1
        if cur < fnum:
            break
        ok, frame = cap.retrieve()
        if not ok:
            continue
        c = E.crop_raw(frame, box)
        if c is not None:
            crops.append(c)
    cap.release()
    print(f"crops={len(crops)}", flush=True)

    V = {}
    for dev in ("cpu", "dml"):
        V[dev] = E.OSNetEnc(dev).encode(crops)
    cos = np.sum(V["cpu"] * V["dml"], axis=1)
    res = {"seg": a.seg, "n": int(len(crops)),
           "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
           "cos_p01": float(np.percentile(cos, 1)),
           "rel_l2_mean": float(np.mean(np.linalg.norm(V["cpu"] - V["dml"], axis=1)))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
