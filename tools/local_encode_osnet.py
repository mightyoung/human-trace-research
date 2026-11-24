#!/usr/bin/env python
"""本机版 OSNet 轨迹编码（专用 ReID 模型，MSMT17 微调）。

定位：与 tools/local_encode.py（DINOv3-B）同采样/同聚合口径的替代编码器，
用于验证「embedding 从通用特征换成专用 ReID」在本机链路上的效果。

口径：
  - 模型：torchreid FeatureExtractor(model_name=osnet_x1_0, model_path=本地权重)
          权重 models/reid_osnet/osnet_x1_0_msmt17_combineall_*.pth（2.17M 参数）
  - 裁剪：原始检测框（无 pad；torchreid 内部 256×128 resize + ImageNet 归一化）
  - 采样：每轨迹 <= --samples（默认 8）张，np.linspace 均匀取（与 local_encode 一致）
  - 聚合：per-crop L2 -> 均值 -> 再 L2（与 local_encode 一致）
  - 输出：np.savez，keys=`{prefix}{tid}`，512 维 float32

加速：--ftz（默认开）启用 torch.set_flush_denormal(True)。
  msmt17 权重含 ~35 万次正规数（subnormal），x86 CPU 走慢路径；
  exp8 实测 FTZ 后 653ms -> 72.7ms/裁剪（docs/commit-015/verify_work/exp8_osnet/report.md §9）。

用法:
  python tools/local_encode_osnet.py \
      --jsonl output_crosscam_local/detect/343_detections.jsonl \
      --video media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4 \
      --prefix v343_ --out output_crosscam_local/embed_osnet/reid_emb_343_osnet_local.npz
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

OSNET_MSMT17 = ("models/reid_osnet/"
                "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
                "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
OSNET_AIN = ("models/reid_osnet/"
             "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_"
             "b64_fb10_softmax_labsmth_flip_jitter.pth")


def load_tracks(jsonl_path):
    """tid -> list[(frame, box)]（与 local_encode.py / encode_tracks.py 相同）。"""
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


def crop_box(frame, box, pad=0.0):
    """原始框裁剪（BGR uint8）。pad=0 与 exp8 的 OSNet 口径一致。

    仅做合法性检查与最小尺寸过滤；不 resize（torchreid 内部处理）。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if pad > 0:
        x1 -= w * pad
        y1 -= h * pad
        x2 += w * pad
        y2 += h * pad
    X1 = max(0, int(x1))
    Y1 = max(0, int(y1))
    X2 = min(frame.shape[1], int(x2))
    Y2 = min(frame.shape[0], int(y2))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 24 or c.shape[1] < 12:
        return None
    return c


class OSNetEmbedder:
    """OSNet 外观嵌入（torchreid FeatureExtractor；CPU 或 DirectML）。"""

    def __init__(self, weights, model_name="osnet_x1_0", ftz=True, device="auto"):
        from torchreid.reid.utils import FeatureExtractor
        dev = "cpu"
        if device != "cpu":
            try:
                import torch_directml  # noqa: F401  注册 privateuseone 后端
                dev = str(torch_directml.device())
                print(f"[reid-osnet] device=dml ({dev})", flush=True)
            except Exception as e:
                if device == "dml":
                    raise
                print(f"[reid-osnet] torch_directml unavailable ({e}) -> cpu",
                      flush=True)
        if ftz and dev == "cpu":
            torch.set_flush_denormal(True)
        print(f"[reid-osnet] backend={model_name} weights={weights} "
              f"device={dev} ftz={ftz and dev == 'cpu'}", flush=True)
        self.fe = FeatureExtractor(model_name=model_name, model_path=str(weights),
                                   device=dev, verbose=False)
        self.dim = 512
        self.device = dev

    @torch.no_grad()
    def embed_batch(self, bgr_list):
        v = self.fe(list(bgr_list))
        v = v.detach().cpu().numpy().astype(np.float32)
        # per-crop L2（与 local_encode 的 torch.nn.functional.normalize 等价）
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)


def plan_crops(tracks, tids, samples):
    """预先算出「哪些帧需要裁哪些框」，按帧号聚合。

    长视频（如 35min）用随机 seek 取帧时，每次 `CAP_PROP_POS_FRAMES` 都要从关键帧
    重新解码，实测 305 ms/crop（vc 只占 ~70ms）；顺序读一遍则整段只解码一次。
    """
    need = {}
    for tid in tids:
        tf = tracks[tid]
        idxs = np.linspace(0, len(tf) - 1, min(samples, len(tf))).astype(int)
        for k in idxs:
            fnum, box = tf[k]
            need.setdefault(int(fnum), []).append((tid, box))
    return need


def encode_sequential(cap, need, emb, batch, prefix, samples, min_frames, tracks, tids,
                      log_every=20000, flush_every=64):
    """顺序解码一遍视频，边解码边编码 → 每轨聚合。

    ★ 不能把所有 crop 攒在内存里：35min / 6 路 × 每轨 8 张 2560×1440 裁片会直接 OOM
      （实测第 16267 帧 alloc 失败）。这里维护一个固定大小的缓冲，满一批就前向一次，
      只保留每轨的「向量和 + 计数」。

    与 `--seek random` 的口径差异：帧号由「读到的第 n 帧」确定，而不是 seek 定位；
    对同一人的框裁剪无实质影响（ReID 只看外观），但两次结果不会逐位相同。
    """
    acc: dict[int, tuple[np.ndarray, int]] = {}
    buf: list[tuple[int, np.ndarray]] = []
    maxf = max(need) if need else 0
    f = 0
    t0 = time.time()
    n_err = 0
    n_emb = 0

    def flush() -> None:
        nonlocal buf, n_emb
        if not buf:
            return
        vecs = emb.embed_batch([c for _, c in buf])
        for (tid, _), v in zip(buf, vecs):
            s, n = acc.get(tid, (np.zeros_like(v), 0))
            acc[tid] = (s + v, n + 1)
        n_emb += len(buf)
        buf = []

    while f <= maxf:
        try:
            ok, frame = cap.read()
            if not ok:
                break
        except Exception as e:      # 个别帧解码失败（HEVC 参考帧缺失）不该中断整段
            n_err += 1
            if n_err <= 3:
                print(f"  [warn] frame {f} 解码异常 {type(e).__name__}: {e}", flush=True)
            f += 1
            if n_err > 50:
                print("  [warn] 连续解码异常过多，停止顺序解码", flush=True)
                break
            continue
        for tid, box in need.get(f, ()):
            c = crop_box(frame, box)
            if c is not None:
                buf.append((tid, c))
        if len(buf) >= flush_every:
            flush()
        f += 1
        if f % log_every == 0:
            print(f"  [decode] frame {f}/{maxf} {time.time()-t0:.0f}s "
                  f"({len(acc)} tracks, {n_emb} crops, err={n_err})", flush=True)
    flush()
    print(f"  [decode] done frames=0..{f-1} {time.time()-t0:.0f}s "
          f"err={n_err} emb={n_emb}", flush=True)

    out, n_crops, n_fail = {}, 0, 0
    for tid in tids:
        got = acc.get(tid)
        if not got or got[1] == 0:
            n_fail += 1
            continue
        m = got[0] / got[1]
        m = m / (np.linalg.norm(m) + 1e-9)
        out[f"{prefix}{tid}"] = m.astype(np.float32)
        n_crops += got[1]
    return out, n_crops, n_fail


def main():
    ap = argparse.ArgumentParser(description="本机版 OSNet 轨迹编码")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--prefix", default="v", help="key 前缀（如 v305_）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-frames", type=int, default=1)
    ap.add_argument("--pad", type=float, default=0.0,
                    help="框外扩比例（默认 0，与 exp8 OSNet 口径一致）")
    ap.add_argument("--weights", default=OSNET_MSMT17)
    ap.add_argument("--model-name", default="osnet_x1_0")
    ap.add_argument("--device", default="auto", choices=["auto", "dml", "cpu"],
                    help="auto=有 torch_directml 则用 DirectML，否则 CPU")
    ap.add_argument("--no-ftz", action="store_true")
    ap.add_argument("--seek", default="sequential", choices=["sequential", "random"],
                    help="取帧方式：sequential=顺序解码一遍（长视频快数倍）；"
                         "random=逐 crop seek（历史行为，结果可逐位复现）")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU 线程数（默认 4；16 核全开时内存峰值过高会 OOM）")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    root = Path(__file__).resolve().parent.parent
    wpath = Path(args.weights)
    if not wpath.is_absolute():
        wpath = root / wpath
    assert wpath.exists(), f"weights not found: {wpath}"

    tracks = load_tracks(args.jsonl)
    tids = sorted(tid for tid, v in tracks.items() if len(v) >= args.min_frames)
    print(f"[encode-osnet] {args.jsonl}: {len(tracks)} tracks total, "
          f"{len(tids)} with >= {args.min_frames} frames", flush=True)

    emb = OSNetEmbedder(wpath, args.model_name, ftz=not args.no_ftz,
                        device=args.device)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"cannot open {args.video}"

    out = {}
    n_crop_fail = 0
    n_crops = 0
    t0 = time.time()

    if args.seek == "sequential":
        need = plan_crops(tracks, tids, args.samples)
        out, n_crops, n_crop_fail = encode_sequential(
            cap, need, emb, args.batch, args.prefix, args.samples,
            args.min_frames, tracks, tids)
        cap.release()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **out)
        el = time.time() - t0
        print(f"[encode-osnet] done(seq): {len(out)} tracks / {n_crops} crops "
              f"({n_crop_fail} crop-fail), {el:.0f}s "
              f"({el/max(n_crops,1)*1000:.0f} ms/crop) -> {out_path}", flush=True)
        return

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
            c = crop_box(frame, box, pad=args.pad)
            if c is None:
                continue
            crops.append(c)
        if not crops:
            n_crop_fail += 1
            continue
        vecs = []
        for s in range(0, len(crops), args.batch):
            vecs.append(emb.embed_batch(crops[s:s + args.batch]))
        M = np.concatenate(vecs, axis=0)
        n_crops += M.shape[0]
        m = M.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-9)
        out[f"{args.prefix}{tid}"] = m.astype(np.float32)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(tids)}] {el:.0f}s "
                  f"({n_crops} crops, {el/max(n_crops,1)*1000:.0f} ms/crop)",
                  flush=True)
    cap.release()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    el = time.time() - t0
    print(f"[encode-osnet] done: {len(out)} tracks / {n_crops} crops "
          f"({n_crop_fail} crop-fail), {el:.0f}s "
          f"({el/max(n_crops,1)*1000:.0f} ms/crop) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
