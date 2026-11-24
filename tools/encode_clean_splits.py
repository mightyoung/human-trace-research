#!/usr/bin/env python
"""按 splits_v5 的 train/test 轨迹，从 cleanset_v2 + 视频编码（OSNet / DINOv3-B）。

为什么需要这个脚本：
  本地检测（local_detect）的 track_id 与 splits_v5（远程 clean2）的 tid **编号分叉**
  （10-17 对账记录：frame-exact 0.5%–63%）。直接用本地 npz 去对 splits 的 tid
  只有 ~27% 命中，评估无效。本脚本用 cleanset_v2 的轨迹定义重新编码，保证键对齐。

口径：
  - 轨迹：splits_v5.{split}.tracks[cam] -> cleanset_v2/{cam}_detections.jsonl 的 tid
  - 采样：每轨 <=6 框（np.linspace），跳过 w<24 或 h<48
  - 裁剪：OSNet 原始框（torchreid 256×128）；DINOv3 prep_crop(pad 0.08, h=224)
  - 聚合：per-crop L2 -> 轨迹均值 -> L2
  - 输出键：{cam}_{tid}（与 splits 对齐）

用法：
  python tools/encode_clean_splits.py --encoder osnet --split train --device dml
  python tools/encode_clean_splits.py --encoder d3b --split test --device dml
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "output_crosscam/analysis/splits_v5.json"
DET_DIR = ROOT / "output_crosscam/cleanset_v2"
VID_DIR = ROOT / "media-resource/multi-cemra/synced-10min"
VIDEO = {
    "316": "01_行政走廊通道_316/316_085500-090500.mp4",
    "305": "02_办公区1_305/305_085500-090500.mp4",
    "318": "03_办公区2_318/318_085500-090500.mp4",
    "320": "04_办公区球机_320/320_085500-090500.mp4",
    "323": "05_茶水间_323/323_085500-090500.mp4",
    "343": "06_电梯间_343/343_085500-090500.mp4",
}
OSNET_W = ("models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_"
           "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
DINO_W = "models/dinov3_vitb16_timm/model.safetensors"
MIN_W, MIN_H = 24, 48
SAMPLES = 6


def load_tracks(cam):
    p = DET_DIR / f"{cam}_detections.jsonl"
    t = {}
    with open(p, encoding="utf-8") as f:
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


def crop_raw(frame, box):
    x1, y1, x2, y2 = [int(float(v)) for v in box]
    X1, Y1 = max(0, x1), max(0, y1)
    X2, Y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < MIN_H or c.shape[1] < MIN_W:
        return None
    return c


def prep_crop_rgb(frame, box, out_h=224, pad=0.08):
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad)); Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad)); Y2 = min(frame.shape[0], int(y2 + h * pad))
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
        pw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, pw // 2, pw - pw // 2, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


class OSNetEnc:
    def __init__(self, device="auto"):
        from torchreid.reid.utils import FeatureExtractor
        dev = "cpu"
        if device != "cpu":
            try:
                import torch_directml  # noqa: F401
                dev = str(torch_directml.device())
            except Exception:
                if device == "dml":
                    raise
        if dev == "cpu":
            torch.set_flush_denormal(True)
        print(f"[osnet] device={dev}", flush=True)
        self.fe = FeatureExtractor(model_name="osnet_x1_0",
                                   model_path=str(ROOT / OSNET_W),
                                   device=dev, verbose=False)

    @torch.no_grad()
    def encode(self, bgr_list):
        v = self.fe(bgr_list).detach().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)


class DinoEnc:
    def __init__(self, device="auto"):
        import timm
        from safetensors.torch import load_file
        from torchvision import transforms
        dev = "cpu"
        if device != "cpu":
            try:
                import torch_directml
                dev = torch_directml.device()
            except Exception:
                if device == "dml":
                    raise
        print(f"[dinov3b] device={dev}", flush=True)
        m = timm.create_model("vit_base_patch16_dinov3", pretrained=False,
                              num_classes=0, global_pool="token")
        m.load_state_dict(load_file(str(ROOT / DINO_W)), strict=False)
        self.device = dev
        self.m = m.eval().to(dev)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])


    @torch.no_grad()
    def encode(self, rgb_list):
        xs = torch.stack([self.tf(im) for im in rgb_list]).to(self.device)
        f = self.m(xs)
        f = torch.nn.functional.normalize(f, dim=1)
        return f.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["osnet", "d3b"])
    ap.add_argument("--split", required=True, choices=["train", "test", "val"])
    ap.add_argument("--device", default="auto", choices=["auto", "dml", "cpu"])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    splits = json.load(open(SPLITS, encoding="utf-8"))
    tracks_by_cam = splits[args.split]["tracks"]
    enc = OSNetEnc(args.device) if args.encoder == "osnet" else DinoEnc(args.device)

    out_path = Path(args.out) if args.out else (
        ROOT / f"output_crosscam_local/selector_clean/emb_{args.split}_{args.encoder}.npz")

    caps = {}
    dets = {}
    out = {}
    t0 = time.time()
    n_track = 0
    n_crop = 0
    for cam, tids in tracks_by_cam.items():
        if not tids:
            continue
        if cam not in caps:
            caps[cam] = cv2.VideoCapture(str(VID_DIR / VIDEO[cam]))
            dets[cam] = load_tracks(cam)
        cap = caps[cam]
        for tid in tids:
            tf = dets[cam].get(int(tid), [])
            if not tf:
                continue
            idxs = np.linspace(0, len(tf) - 1, min(SAMPLES, len(tf))).astype(int)
            crops = []
            for k in idxs:
                fnum, box = tf[k]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                ok, frame = cap.read()
                if not ok:
                    continue
                c = crop_raw(frame, box) if args.encoder == "osnet" else prep_crop_rgb(frame, box)
                if c is not None:
                    crops.append(c)
            if len(crops) < 2:
                continue
            V = enc.encode(crops)
            n_crop += V.shape[0]
            m = V.mean(0)
            m = m / (np.linalg.norm(m) + 1e-9)
            out[f"{cam}_{int(tid)}"] = m.astype(np.float32)
            n_track += 1
            if n_track % 50 == 0:
                print(f"  [{n_track}] {time.time()-t0:.0f}s", flush=True)
    for c in caps.values():
        c.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"[done] {args.encoder}/{args.split}: {n_track} tracks, {n_crop} crops, "
          f"{time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
