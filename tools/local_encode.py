#!/usr/bin/env python
"""本机版嵌入环节（DINOv3-B + torch-directml）。

定位：替代远程 GPU（CUDA）版 tools/encode_tracks.py——在本机
（AMD RX 5700 XT / DirectML）对检测 jsonl 中的轨迹裁剪做外观编码，
输出与远程历史产物（analysis/relay/remote_emb/reid_emb_*_d3b.npz）同构的 npz。

口径对齐（与远程 encode_tracks.py + crosscam_reid.py 逐项一致）：
  - 模型：timm `vit_base_patch16_dinov3`（global_pool="token", num_classes=0）
    + 本地权重 models/dinov3_vitb16_timm/model.safetensors（离线加载）
  - 预处理：prep_crop（pad=0.08，固定高 224，宽超则居中裁剪、不足则补边，
    BGR→RGB）→ ToTensor + ImageNet 归一化（与 crosscam_reid.Embedder 相同）
  - 采样：每轨迹最多 --samples（默认 8，与 _remote_encode6.sh 一致）张，
    np.linspace 均匀取；逐帧 cv2 seek+read
  - 聚合：单张嵌入 L2 归一化 → 均值 → 再 L2 归一化（跨设备逐位对齐）
  - 输出：np.savez，keys = `{prefix}{tid}`（默认 prefix=v{cam}_，与远程一致）

设备：torch-directml（privateuseone）优先；--device cpu 可切（对照用）。

用法:
  python tools/local_encode.py \
      --jsonl output_crosscam_local/detect/305_detections.jsonl \
      --video media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4 \
      --prefix v305_ --out output_crosscam_local/embed/reid_emb_305_d3b_local.npz
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


# ---------------- 与 crosscam_reid.py 逐行一致的 crop 逻辑 ----------------
def prep_crop(frame, box, out_h=224, pad=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad))
    Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad))
    Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
        return None
    s = out_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, out_h))
    if nw >= out_h:
        off = (nw - out_h) // 2
        c = c[:, off:off + out_h]
    else:
        padw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2,
                               cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def load_tracks(jsonl_path):
    """tid -> list[(frame, box)]（与 encode_tracks.py 相同）。"""
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


class EmbedderLocal:
    """DINOv3-B 外观嵌入（timm 本地权重；device: dml / cpu）。"""

    def __init__(self, weights, device="dml"):
        import timm
        from safetensors.torch import load_file

        if device == "dml":
            import torch_directml
            self.device = torch_directml.device()
        else:
            self.device = torch.device("cpu")
        arch = "vit_base_patch16_dinov3"
        print(f"[reid] backend=dinov3b device={device} ({self.device})", flush=True)
        m = timm.create_model(arch, pretrained=False, num_classes=0,
                              global_pool="token")
        sd = load_file(str(weights))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[reid] dinov3-B loaded: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
        self.model = m.eval().to(self.device)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def embed_batch(self, rgb_list):
        xs = torch.stack([self.tf(im) for im in rgb_list]).to(self.device)
        f = self.model(xs)
        f = torch.nn.functional.normalize(f, dim=1)
        if str(self.device).startswith("privateuseone"):
            import torch_directml  # noqa: F401  确保 dml 同步语义
        return f.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="本机版 DINOv3-B 轨迹编码")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--prefix", default="v", help="key 前缀（如 v305_）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-frames", type=int, default=1)
    ap.add_argument("--weights", default="models/dinov3_vitb16_timm/model.safetensors")
    ap.add_argument("--device", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--ab", action="store_true",
                    help="额外输出 {out}_a.npz / {out}_b.npz（偶/奇采样子集均值），"
                         "用于自一致性指标 ab_same，口径对应 cradio 的 emb_a/emb_b")
    args = ap.parse_args()

    tracks = load_tracks(args.jsonl)
    tids = sorted(tid for tid, v in tracks.items() if len(v) >= args.min_frames)
    print(f"[encode] {args.jsonl}: {len(tracks)} tracks total, "
          f"{len(tids)} with >= {args.min_frames} frames", flush=True)

    emb = EmbedderLocal(args.weights, args.device)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"

    out = {}
    out_a, out_b = {}, {}
    n_crop_fail = 0
    t0 = time.time()
    for i, tid in enumerate(tids):
        tf = tracks[tid]
        idxs = np.linspace(0, len(tf) - 1, min(args.samples, len(tf))).astype(int)
        crops, kept = [], []
        for j, k in enumerate(idxs):
            fnum, box = tf[k]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = prep_crop(frame, box)
            if rgb is None:
                continue
            crops.append(rgb)
            kept.append(j)
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
        if args.ab and len(M) >= 2:
            ka = [x for x, j in enumerate(kept) if j % 2 == 0]
            kb = [x for x, j in enumerate(kept) if j % 2 == 1]
            if ka and kb:
                for dst, sel in ((out_a, ka), (out_b, kb)):
                    mm = M[sel].mean(axis=0)
                    dst[f"{args.prefix}{tid}"] = (mm / (np.linalg.norm(mm) + 1e-9)
                                                  ).astype(np.float32)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(tids)}] elapsed {time.time()-t0:.0f}s", flush=True)
    cap.release()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    if args.ab and out_a:
        stem = args.out[:-4] if args.out.endswith(".npz") else args.out
        np.savez(f"{stem}_a.npz", **out_a)
        np.savez(f"{stem}_b.npz", **out_b)
        print(f"[encode] ab 子集输出: {stem}_a.npz / {stem}_b.npz "
              f"(n={len(out_a)})", flush=True)
    print(f"[encode] done: {len(out)} tracks encoded ({n_crop_fail} crop-fail), "
          f"{time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
