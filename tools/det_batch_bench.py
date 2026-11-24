# -*- coding: utf-8 -*-
"""det_batch_bench.py — GroundingDINO 批量推理吞吐基准（16 路实时能力评估用）。

测什么：
  - batch = 1/2/4/8 的「processor 前处理 / 模型前向 / 后处理」三段耗时与吞吐
  - fp32 vs fp16 两档
  - 用于推导：单卡可支撑的 16 路实时（stride=5）路数

用法（远程 GPU）：
  ~/miniconda3/envs/rfdetr/bin/python det_batch_bench.py \
      --video /share/data4/yhh/data/human-count/video-6way/318.mp4 \
      --out det_bench.json --device cuda:0
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sample_frames(video, n=32):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = np.linspace(0, max(total - 2, 0), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, dict(total=total, fps=fps, w=w, h=h)


def time_batches(model, processor, frames, batch, device, rounds=3):
    """返回 (proc_s, fwd_s, post_s) 中值（单批 B 张口径）。"""
    ps, fs, qs = [], [], []
    for r in range(rounds):
        imgs = frames[:batch]
        pils = [Image.fromarray(im) for im in imgs]
        texts = ["person."] * len(pils)

        t0 = time.time()
        inputs = processor(images=pils, text=texts, return_tensors="pt").to(device)
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.time()

        with torch.no_grad():
            outputs = model(**inputs)
        if device != "cpu":
            torch.cuda.synchronize()
        t2 = time.time()

        _ = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=0.30, text_threshold=0.25,
            target_sizes=[p.size[::-1] for p in pils])
        if device != "cpu":
            torch.cuda.synchronize()
        t3 = time.time()

        ps.append(t1 - t0)
        fs.append(t2 - t1)
        qs.append(t3 - t2)
    return float(np.median(ps)), float(np.median(fs)), float(np.median(qs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="/share/data4/yhh/models/dino-model")
    ap.add_argument("--out", default="det_bench.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-frames", type=int, default=32)
    args = ap.parse_args()

    device = args.device
    torch.cuda.set_device(int(device.split(":")[1]))
    print(f"[env] {torch.__version__} {torch.cuda.get_device_name(device)}", flush=True)

    frames, meta = sample_frames(args.video, args.n_frames)
    print(f"[video] {meta} frames_sampled={len(frames)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    print("[model] loaded", flush=True)

    result = {"meta": meta, "device": device, "rows": []}

    for dtype_name in ["fp32", "fp16"]:
        try:
            if dtype_name == "fp16":
                model.half()
            else:
                model.float()

            # warmup（装载 kernel / 缓存显存池）
            with torch.no_grad():
                pils = [Image.fromarray(im) for im in frames[:2]]
                inp = processor(images=pils, text=["person."] * 2,
                                return_tensors="pt").to(device)
                if dtype_name == "fp16":
                    inp = {k: (v.half() if v.dtype == torch.float32 else v)
                           for k, v in inp.items()}
                _ = model(**inp)
                torch.cuda.synchronize()

            for batch in [1, 2, 4, 8]:
                p_s, f_s, q_s = time_batches(model, processor, frames, batch, device)
                row = {
                    "dtype": dtype_name, "batch": batch,
                    "proc_ms": round(p_s * 1000, 1),
                    "fwd_ms": round(f_s * 1000, 1),
                    "post_ms": round(q_s * 1000, 1),
                    "total_ms": round((p_s + f_s + q_s) * 1000, 1),
                    "per_img_ms": round((p_s + f_s + q_s) / batch * 1000, 1),
                    "fps_end2end": round(batch / (p_s + f_s + q_s), 2),
                    "fps_fwd_only": round(batch / f_s, 2),
                    "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                }
                torch.cuda.reset_peak_memory_stats()
                result["rows"].append(row)
                print(f"[bench] {row}", flush=True)
        except Exception as e:
            print(f"[bench] {dtype_name} FAILED: {type(e).__name__}: {e}", flush=True)
            result["rows"].append({"dtype": dtype_name, "error": str(e)[:200]})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
