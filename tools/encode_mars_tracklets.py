#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 MARS 轨迹裁图编码成下游要的 `<seg>_emb_a.npz` / `<seg>_emb_b.npz`。

口径与现场一致：
  - 每条轨迹按帧序对半切（a=前半，b=后半），两半各取至多 --samples 帧
  - 每半的特征 = 这些帧 OSNet 特征的均值，再 L2 归一化
  - 键 = `<seg>#<tid>`，与 make_ab_jsonl / d7_sweep 约定一致
  - 默认 BGR（cv2 读图直送 torchreid），与冻结基线同口径；--rgb 只用于敏感性实验

用法：
  PY tools/encode_mars_tracklets.py --pkg validation/mars --device dml
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_1012_osnet_seq import OSNetEnc  # noqa: E402

CHUNK = 256


def pick(idx, k):
    """在 [0, len) 里均匀取至多 k 个下标（含首尾）"""
    n = len(idx)
    if n <= k:
        return idx
    pos = np.linspace(0, n - 1, k).round().astype(int)
    return [idx[p] for p in sorted(set(pos.tolist()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="validation/mars")
    ap.add_argument("--segs", default="", help="只编这些段（逗号分隔）；默认 manifest 全量")
    ap.add_argument("--samples", type=int, default=8, help="每半最多取几帧（现场口径 8）")
    ap.add_argument("--device", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--rgb", action="store_true")
    a = ap.parse_args()

    pkg = os.path.abspath(a.pkg)
    mdir = os.path.join(pkg, "manifest")
    segs = ([x.strip() for x in a.segs.split(",") if x.strip()] if a.segs
            else sorted(f[:-6] for f in os.listdir(mdir) if f.endswith(".jsonl")))
    out_dir = os.path.join(pkg, "emb")
    os.makedirs(out_dir, exist_ok=True)
    enc = OSNetEnc(a.device, rgb=a.rgb)

    t_all = time.time()
    tot_crop = 0
    for seg in segs:
        t0 = time.time()
        rows = [json.loads(l) for l in open(os.path.join(mdir, f"{seg}.jsonl"), encoding="utf-8")]
        res = {"a": {}, "b": {}}
        buf, meta = [], []

        def flush():
            nonlocal tot_crop
            if not buf:
                return
            V = enc.encode(buf)
            tot_crop += len(buf)
            # 按 (ab,tid) 聚合
            agg = {}
            for (ab, tid, _k), v in zip(meta, V):
                agg.setdefault((ab, tid), []).append(v)
            for (ab, tid), vs in agg.items():
                m = np.asarray(vs, np.float32).mean(0)
                m = m / (np.linalg.norm(m) + 1e-9)
                res[ab][f"{seg}#{tid}"] = m.astype(np.float32)
            buf.clear()
            meta.clear()

        for r in rows:
            frs = r.get("frames") or list(range(r["f0"], r["f0"] + int(r["n_frames"])))
            k = len(frs) // 2
            halves = {"a": frs[:k], "b": frs[k:]}
            for ab, frames in halves.items():
                for fidx in pick(frames, a.samples):
                    p = os.path.join(r["dir"], f"{r['prefix']}{fidx:03d}.jpg")
                    im = cv2.imread(p)
                    if im is None:
                        continue
                    buf.append(im)
                    meta.append((ab, int(r["tid"]), fidx))
                    if len(buf) >= CHUNK:
                        flush()
        flush()
        np.savez(os.path.join(out_dir, f"{seg}_emb_a.npz"), **res["a"])
        np.savez(os.path.join(out_dir, f"{seg}_emb_b.npz"), **res["b"])
        print(f"[{seg}] {len(rows)} 轨 -> a={len(res['a'])} b={len(res['b'])} "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[done] {len(segs)} 段 / {tot_crop} 裁图 / {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
