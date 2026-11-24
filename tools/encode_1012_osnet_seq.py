#!/usr/bin/env python
"""1012 段 OSNet 编码（顺序读取版）——避免随机 seek 导致的超长耗时。

问题：1012 train 段 18000 帧（12 min，1GB+），810 轨 × 8 帧 = 6480 次随机 seek，
      单分片 >10 分钟。本脚本对一段视频**只顺序读一遍**，同时处理 a/b 两个分片的
      所有帧请求（按帧号排序推进），每段约 2-4 分钟。

用法：
  python tools/encode_1012_osnet_seq.py --seg 305_train_529
输出：
  output_crosscam_local/1012_osnet_emb/{seg}_emb_a.npz
  output_crosscam_local/1012_osnet_emb/{seg}_emb_b.npz
  键 = "{seg}#{tid}"
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
AB_DIR = ROOT / "output_crosscam_local/1012_ab"
EMB_DIR = ROOT / "output_crosscam_local/1012_osnet_emb"
VID_DIR = ROOT / "media-resource/1012/segments"
OSNET_W = ("models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_"
           "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
SAMPLES = 8
MIN_W, MIN_H = 24, 48


def load_ab(path):
    t = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids") or []
            scores = r.get("scores") or []
            for i, b in enumerate(r.get("boxes") or []):
                if i < len(tids) and tids[i] is not None and int(tids[i]) >= 0:
                    sc = float(scores[i]) if i < len(scores) else 0.0
                    t[int(tids[i])].append((int(r["frame"]), b, sc))
    for k in t:
        t[k].sort()
    return t


def load_scores(seg):
    """从原始检测 jsonl 回查每 (frame, tid) 的检测置信度。
    a/b 分片 jsonl 只保留 frame/track_ids/boxes，scores 会丢失，故需回源。"""
    p = ROOT / "output_crosscam/1012_final" / f"{seg}.jsonl"
    smap = {}
    if not p.exists():
        return smap
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids") or []
            scores = r.get("scores") or []
            fr = int(r["frame"])
            for i, t in enumerate(tids):
                if t is None or int(t) < 0:
                    continue
                smap[(fr, int(t))] = float(scores[i]) if i < len(scores) else 0.0
    return smap


def crop_raw(frame, box):
    x1, y1, x2, y2 = [int(float(v)) for v in box]
    X1, Y1 = max(0, x1), max(0, y1)
    X2, Y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < MIN_H or c.shape[1] < MIN_W:
        return None
    return c


class OSNetEnc:
    def __init__(self, device="dml", rgb=False):
        # rgb=False 保持历史口径（cv2 的 BGR 直接喂 torchreid，torchreid 用 ToPILImage 按 RGB 解读 → 通道颠倒）。
        # rgb=True 为正确口径；切换后所有阈值（thr_in/thr_out/θ/same）需重新标定。
        self.rgb = rgb
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
        outs = []
        for i in range(0, len(bgr_list), 32):
            batch = bgr_list[i:i + 32]
            if self.rgb:
                batch = [np.ascontiguousarray(c[:, :, ::-1]) for c in batch]
            v = self.fe(batch).detach().cpu().numpy().astype(np.float32)
            outs.append(v)
        V = np.concatenate(outs, axis=0)
        n = np.linalg.norm(V, axis=1, keepdims=True)
        return V / np.maximum(n, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--device", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--out-dir", default=str(EMB_DIR), help="输出目录（默认 EMB_DIR，可指向临时目录做校验）")
    ap.add_argument("--save-crops", action="store_true",
                    help="另存逐裁特征+质量（{seg}_crops.npz），用于质量/多样性加权聚合实验")
    ap.add_argument("--ab-dir", default=str(AB_DIR),
                    help="轨迹 jsonl 目录（默认 1012_ab；换检测器时指向新的 _a/_b 目录）")
    ap.add_argument("--video", default="", help="直接指定 mp4（默认 <VID_DIR>/<seg>.mp4），用于原卷局部区间")
    ap.add_argument("--rgb", action="store_true",
                    help="裁图先 BGR→RGB 再送 torchreid（正确口径）。默认关，保证冻结基线逐位可复现")
    args = ap.parse_args()

    seg = args.seg
    pa, pb = Path(args.ab_dir) / f"{seg}_a.jsonl", Path(args.ab_dir) / f"{seg}_b.jsonl"
    va = Path(args.video) if args.video else VID_DIR / f"{seg}.mp4"
    for p in (pa, pb, va):
        if not p.exists():
            raise SystemExit(f"missing: {p}")
    emb_dir = Path(args.out_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    out_a, out_b = emb_dir / f"{seg}_emb_a.npz", emb_dir / f"{seg}_emb_b.npz"
    if out_a.exists() and out_b.exists():
        print(f"[skip] {seg} exists")
        return

    # 请求：每轨 np.linspace 取 samples 帧（置信度从原始检测回查）
    smap = load_scores(seg) if getattr(args, "save_crops", False) else {}
    reqs = []
    for ab, p in (("a", pa), ("b", pb)):
        tracks = load_ab(p)
        for tid, tf in tracks.items():
            if len(tf) < 2:
                continue
            idxs = np.linspace(0, len(tf) - 1, min(args.samples, len(tf))).astype(int)
            for k in idxs:
                fnum, box, sc = tf[k]
                if smap:
                    sc = smap.get((int(fnum), int(tid)), sc)
                reqs.append((int(fnum), ab, int(tid), box, float(sc)))
    reqs.sort(key=lambda x: x[0])
    print(f"[{seg}] requests={len(reqs)} tracks_a/b", flush=True)

    enc = OSNetEnc(args.device, rgb=args.rgb)
    cap = cv2.VideoCapture(str(va))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {va}")

    # 分块流式编码：不再一次性缓存全部裁剪（305 段曾因 546MB 累积触发 OpenCV OOM），
    # 而是每 CHUNK 张就编码一次，内存占用变常数。
    CHUNK = 512
    counts = defaultdict(int)
    idx_by_key = defaultdict(list)
    eids, qrows = [], []
    chunks = []          # 按到达顺序的编码结果块
    buf_crops = []
    cur = -1
    frame = None
    t0 = time.time()
    n_crop = 0
    H, W = None, None
    t_enc_total = 0.0

    def flush():
        nonlocal buf_crops, t_enc_total
        if not buf_crops:
            return
        t1 = time.time()
        chunks.append(enc.encode(buf_crops))
        t_enc_total += time.time() - t1
        buf_crops = []

    for fnum, ab, tid, box, sc in reqs:
        if fnum > cur:
            ok = True
            while cur < fnum:
                ok = cap.grab()
                if not ok:
                    break
                cur += 1
            if not ok:
                break
            try:
                ok, frame = cap.retrieve()
            except cv2.error as e:
                print(f"[{seg}] retrieve failed at frame {cur}: {e}", flush=True)
                break
            if not ok:
                break
            if H is None:
                H, W = frame.shape[:2]
        if frame is None:
            continue
        c = crop_raw(frame, box)
        if c is not None:
            key = (ab, tid)
            k = counts[key]
            counts[key] += 1
            idx_by_key[key].append(len(eids))
            eids.append(f"{ab}#{tid}#{k}")
            bw = max(1.0, float(box[2]) - float(box[0]))
            bh = max(1.0, float(box[3]) - float(box[1]))
            area = (bw * bh) / float((W or 1) * (H or 1))
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            qrows.append((sc, area, sharp))
            buf_crops.append(c)
            n_crop += 1
            if len(buf_crops) >= CHUNK:
                flush()
    flush()
    cap.release()
    print(f"[{seg}] decoded+{n_crop} crops in {time.time()-t0:.0f}s", flush=True)

    feats = (np.concatenate(chunks, axis=0) if chunks
             else np.zeros((0, 512), np.float32))
    print(f"[{seg}] encoded {len(feats)} crops in {t_enc_total:.0f}s "
          f"({t_enc_total/max(1,len(feats))*1000:.1f} ms/crop)", flush=True)

    # 可选：保存逐裁特征 + 质量三要素（conf, 面积占比, 拉普拉斯清晰度），供质量/多样性加权聚合
    if getattr(args, "save_crops", False):
        crop_path = emb_dir / f"{seg}_crops.npz"
        np.savez(crop_path, eids=np.asarray(eids), emb=feats.astype(np.float32),
                 q=np.asarray(qrows, np.float32).reshape(-1, 3))
        print(f"[{seg}] crops -> {crop_path} ({len(eids)} rows)", flush=True)

    res = {"a": {}, "b": {}}
    for key, idxs in idx_by_key.items():
        ab, tid = key
        V = feats[np.asarray(idxs, int)]
        m = V.mean(0)
        m = m / (np.linalg.norm(m) + 1e-9)
        res[ab][f"{seg}#{tid}"] = m.astype(np.float32)
    np.savez(out_a, **res["a"])
    np.savez(out_b, **res["b"])
    print(f"[done] {seg}: a={len(res['a'])} b={len(res['b'])} tracks, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
