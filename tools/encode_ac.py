#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A+C：面积分位收紧框 + 人占比高的样本选择，重编码轨迹表征。

A（收紧框）：对每条轨迹统计全帧框面积，取 p30 作为"人实际占据"的参考面积，
  以框中心缩放 scale = clip(sqrt(p30/median), min_scale, 1.0)。目的是削掉
  俯视工位场景里框边缘的座椅/显示器。
C（选样）：在收紧后的裁剪中，按"人占比代理分" = 面积占比 × clip(清晰度/300, 0, 1)
  取前 K 张（默认 4），替代原来的 8 张等权平均。

口径与既有编码保持一致：同一批采样帧（np.linspace 8 点）、同一 OSNet 权重、
同一 a/b 分半、L2 归一后按轨迹求均值。`--tighten off --topk 0` 必须复现既有嵌入。

用法：
  python tools/encode_ac.py --seg 320_test_518 --tighten --topk 4 \
      --out-dir output_crosscam_local/1012_osnet_crops_ac
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encode_1012_osnet_seq as E  # noqa: E402

ROOT = E.ROOT
SAMPLES = 8


def crop_tight(frame, box, scale):
    if scale >= 0.999:
        return E.crop_raw(frame, box)
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return E.crop_raw(frame, [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--device", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--tighten", action="store_true")
    ap.add_argument("--min-scale", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=0, help="每条轨迹保留的张数，0=全部 8 张")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    seg = a.seg
    va = E.VID_DIR / f"{seg}.mp4"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_a, out_b = out_dir / f"{seg}_emb_a.npz", out_dir / f"{seg}_emb_b.npz"
    if out_a.exists() and out_b.exists():
        print(f"[skip] {seg}")
        return

    smap = E.load_scores(seg)

    # ---------- 第一遍：只算几何量（不解码），选样 ----------
    # A：scale = clip(sqrt(p30/median), min_scale, 1)。只依赖框，不需要像素。
    # C：按收紧后的面积占比取前 K 张（人越大 => 越可能是清晰主体）。
    plan = defaultdict(list)    # (ab,tid) -> [(k, frame, box, scale)]
    scales = []
    for ab, p in (("a", E.AB_DIR / f"{seg}_a.jsonl"), ("b", E.AB_DIR / f"{seg}_b.jsonl")):
        tracks = E.load_ab(p)
        for tid, tf in tracks.items():
            if len(tf) < 2:
                continue
            areas = np.array([max(1.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
                              for _f, b, _s in tf])
            med = float(np.median(areas))
            p30 = float(np.percentile(areas, 30))
            sc_scale = float(np.clip(np.sqrt(p30 / max(med, 1e-6)), a.min_scale, 1.0)) \
                if a.tighten else 1.0
            scales.append(sc_scale)
            idxs = np.linspace(0, len(tf) - 1, min(SAMPLES, len(tf))).astype(int)
            cand = []
            for k, ix in enumerate(idxs):
                fnum, box, sc = tf[ix]
                area = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])) * sc_scale ** 2
                cand.append((area, k, int(fnum), box, sc_scale, float(smap.get((int(fnum), int(tid)), sc))))
            keep = sorted(cand, key=lambda x: -x[0])[:a.topk] if a.topk > 0 else cand
            for _area, k, fnum, box, scl, conf in sorted(keep, key=lambda x: x[1]):
                plan[(ab, int(tid))].append((k, fnum, box, scl, conf))

    reqs = []
    for key, items in plan.items():
        for k, fnum, box, scl, conf in items:
            reqs.append((fnum, key[0], key[1], k, box, conf, scl))
    reqs.sort(key=lambda x: x[0])
    n_keep = len(reqs)
    print(f"[{seg}] tracks={len(plan)} selected_crops={n_keep} "
          f"(tighten={a.tighten}, topk={a.topk}, mean_scale={np.mean(scales) if scales else 1:.3f})",
          flush=True)

    # ---------- 第二遍：只解码被选中的裁剪，流式编码（内存常数） ----------
    enc = E.OSNetEnc(a.device)
    cap = cv2.VideoCapture(str(va))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {va}")

    CHUNK = 256
    feats_by_key = defaultdict(list)
    qrows, eids, owners = [], [], []
    buf, buf_keys, buf_q = [], [], []
    cur, frame = -1, None
    H = W = None
    t0 = time.time()

    def flush():
        nonlocal buf, buf_keys, buf_q
        if not buf:
            return
        V = enc.encode(buf)
        for key, vec in zip(buf_keys, V):
            feats_by_key[key].append(vec)
        buf, buf_keys, buf_q = [], [], []

    for fnum, ab, tid, k, box, conf, scl in reqs:
        if fnum > cur:
            ok = True
            while cur < fnum:
                ok = cap.grab()
                if not ok:
                    break
                cur += 1
            if not ok:
                break
            ok, frame = cap.retrieve()
            if not ok:
                break
            if H is None:
                H, W = frame.shape[:2]
        if frame is None:
            continue
        c = crop_tight(frame, box, scl)
        if c is None:
            continue
        bw = max(1.0, (float(box[2]) - float(box[0])) * scl)
        bh = max(1.0, (float(box[3]) - float(box[1])) * scl)
        area = (bw * bh) / float((W or 1) * (H or 1))
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        buf.append(c)
        buf_keys.append((ab, int(tid)))
        buf_q.append((conf, area, sharp))
        eids.append(f"{ab}#{tid}#{k}")
        owners.append((ab, int(tid)))
        qrows.append((conf, area, sharp))
        if len(buf) >= CHUNK:
            flush()
    flush()
    cap.release()
    feats = (np.concatenate([np.stack(v) for v in feats_by_key.values() if v], axis=0)
             if feats_by_key else np.zeros((0, 512), np.float32))
    print(f"[{seg}] decoded+encoded {len(feats)} crops in {time.time()-t0:.0f}s", flush=True)

    np.savez(out_dir / f"{seg}_crops.npz", eids=np.asarray(eids),
             emb=feats.astype(np.float32),
             q=np.asarray(qrows, np.float32).reshape(-1, 3))

    res = {"a": {}, "b": {}}
    for key, vs in feats_by_key.items():
        if not vs:
            continue
        V = np.stack(vs)
        m = V.mean(0)
        m = m / (np.linalg.norm(m) + 1e-9)
        res[key[0]][f"{seg}#{key[1]}"] = m.astype(np.float32)
    np.savez(out_a, **res["a"])
    np.savez(out_b, **res["b"])
    print(f"[done] {seg}: a={len(res['a'])} b={len(res['b'])} tracks, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
