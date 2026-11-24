# -*- coding: utf-8 -*-
"""det_batch_bench2.py — GroundingDINO 补充基准：fp16 autocast + 输入分辨率对比。

背景：首轮基准显示 fp32 下 batch 8 吞吐仅 6.5 fps（批处理无效），fp16 因输入
dtype 问题失败。本脚本：
  - 用 autocast（而非 model.half()）测 fp16
  - 对比 shortest_edge 800 vs 640 两档分辨率
  - batch = 1/4/8
用于推导 16 路实时场景下检测环节的真实可达吞吐与所需卡数。
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sample_frames(video, n=24):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = np.linspace(0, max(total - 2, 0), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def run_once(model, processor, frames, batch, device, edge, use_fp16, rounds=3):
    ps, fs, qs = [], [], []
    for _ in range(rounds):
        pils = [Image.fromarray(im) for im in frames[:batch]]
        texts = ["person."] * len(pils)
        t0 = time.time()
        try:
            inputs = processor(images=pils, text=texts, return_tensors="pt",
                               size={"shortest_edge": edge}).to(device)
        except Exception:
            inputs = processor(images=pils, text=texts, return_tensors="pt").to(device)
        if device != "cpu":
            torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
        if device != "cpu":
            torch.cuda.synchronize()
        t2 = time.time()
        _ = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.30, text_threshold=0.25,
            target_sizes=[p.size[::-1] for p in pils])
        if device != "cpu":
            torch.cuda.synchronize()
        t3 = time.time()
        ps.append(t1 - t0); fs.append(t2 - t1); qs.append(t3 - t2)
    return float(np.median(ps)), float(np.median(fs)), float(np.median(qs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="/share/data4/yhh/models/dino-model")
    ap.add_argument("--out", default="det_bench2.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-frames", type=int, default=24)
    args = ap.parse_args()

    device = args.device
    torch.cuda.set_device(int(device.split(":")[1]))
    frames = sample_frames(args.video, args.n_frames)
    print(f"[video] frames={len(frames)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    print("[model] loaded", flush=True)

    res = {"rows": []}
    for dtype_name, use_fp16 in [("fp32", False), ("fp16", True)]:
        for edge in [800, 640]:
            for batch in [1, 4, 8]:
                try:
                    with torch.no_grad():
                        pils = [Image.fromarray(im) for im in frames[:2]]
                        try:
                            inp = processor(images=pils, text=["person."] * 2,
                                            return_tensors="pt",
                                            size={"shortest_edge": edge}).to(device)
                        except Exception:
                            inp = processor(images=pils, text=["person."] * 2,
                                            return_tensors="pt").to(device)
                        if use_fp16:
                            with torch.autocast("cuda", dtype=torch.float16):
                                _ = model(**inp)
                        else:
                            _ = model(**inp)
                        torch.cuda.synchronize()

                    p_s, f_s, q_s = run_once(model, processor, frames, batch,
                                             device, edge, use_fp16)
                    row = {"dtype": dtype_name, "edge": edge, "batch": batch,
                           "proc_ms": round(p_s * 1000, 1),
                           "fwd_ms": round(f_s * 1000, 1),
                           "post_ms": round(q_s * 1000, 1),
                           "total_ms": round((p_s + f_s + q_s) * 1000, 1),
                           "per_img_ms": round((p_s + f_s + q_s) / batch * 1000, 1),
                           "fps_end2end": round(batch / (p_s + f_s + q_s), 2)}
                    res["rows"].append(row)
                    print(f"[bench] {row}", flush=True)
                    torch.cuda.reset_peak_memory_stats()
                except Exception as e:
                    print(f"[bench] {dtype_name} edge={edge} b={batch} FAILED: "
                          f"{type(e).__name__}: {str(e)[:160]}", flush=True)
                    res["rows"].append({"dtype": dtype_name, "edge": edge,
                                        "batch": batch,
                                        "error": f"{type(e).__name__}: {str(e)[:160]}"})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
