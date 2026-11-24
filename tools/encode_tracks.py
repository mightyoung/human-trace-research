#!/usr/bin/env python
"""通用轨迹外观编码器 → npz（keys: {prefix}{tid}）。

为跨机位去重链路批量编码某机位 jsonl 中的全部轨迹（或指定轨迹），
输出可直接供 crosscam_pipeline.py 消费的 embedding 文件。

用法（远程 GPU 例）:
  ~/miniconda3/envs/rfdetr/bin/python tools/encode_tracks.py \
    --jsonl crosscam/out/318_detections.jsonl --prefix a \
    --video crosscam/in/318.mp4 --wh 1920 1080 \
    --out run/reid_emb_318.npz --samples 8 --backend dinov3b --device cuda

依赖：与 crosscam_reid.py 相同（torch/torchvision/timm；权重见 models/）。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscam_reid import Embedder, prep_crop  # noqa: E402


def load_tracks(jsonl_path):
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((int(r["frame"]), b))
    for k in t:
        t[k].sort()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=None, help="检测 jsonl 路径（与 --det-cam 二选一）")
    ap.add_argument("--det-cam", default=None,
                    help="机位号：走 det_view 解析检测源（DET_VIEW=raw|clean）；"
                         "video 默认 {DET_DATA}/video-6way/{cam}.mp4")
    ap.add_argument("--video", default=None)
    ap.add_argument("--prefix", default="t", help="key 前缀（318→a, 320→b）")
    ap.add_argument("--wh", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--min-frames", type=int, default=1,
                    help="跳过帧数少于该值的轨迹")
    ap.add_argument("--backend", default="dinov3b",
                    choices=["dinov2", "dinov3s", "dinov3b"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8, help="每轨迹批大小")
    args = ap.parse_args()

    if args.det_cam:
        from det_view import remote_det, describe
        print(describe(), flush=True)
        data = os.environ.get("DET_DATA", "/share/data4/yhh/data/human-count")
        args.jsonl = args.jsonl or remote_det(args.det_cam, data)
        args.video = args.video or f"{data}/video-6way/{args.det_cam}.mp4"
    if not args.jsonl:
        ap.error("需要 --jsonl 或 --det-cam")
    if not args.video:
        ap.error("需要 --video 或 --det-cam")

    tracks = load_tracks(args.jsonl)
    tids = sorted(tid for tid, v in tracks.items()
                  if len(v) >= args.min_frames)
    print(f"[encode] {args.jsonl}: {len(tracks)} tracks total, "
          f"{len(tids)} with >= {args.min_frames} frames", flush=True)

    emb = Embedder(device=args.device, backend=args.backend)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"

    out = {}
    n_ok = n_crop_fail = 0
    t0 = time.time()
    for i, tid in enumerate(tids):
        tf = tracks[tid]
        idxs = np.linspace(0, len(tf) - 1, min(args.samples, len(tf))).astype(int)
        crops = []
        for k in idxs:
            fnum, box = tf[k]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = prep_crop(frame, box)
            if rgb is None:
                continue
            crops.append(rgb)
        if not crops:
            n_crop_fail += 1
            continue
        vecs = []
        for s in range(0, len(crops), args.batch):
            vecs.append(emb.embed_batch(crops[s:s + args.batch]))
        M = np.concatenate(vecs, axis=0)
        m = M.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-9)
        out[f"{args.prefix}{tid}"] = m.astype(np.float32)
        n_ok += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(tids)}] elapsed {time.time()-t0:.0f}s", flush=True)
    cap.release()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"[encode] done: {n_ok} tracks encoded ({n_crop_fail} crop-fail), "
          f"{time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
