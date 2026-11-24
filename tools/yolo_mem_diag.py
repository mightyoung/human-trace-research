# -*- coding: utf-8 -*-
"""yolo_mem_diag.py — YOLO 评估显存异常诊断（GPU1）。

背景：eval_det.run_yolo 在 602 帧 val 上 warmup 阶段 OOM（进程已占 31GB）。
本脚本用小样本分级验证：单图 → batch4 → batch16 stream，打印分辨率/峰值显存。
"""
import glob
import time

import cv2
import torch
from ultralytics import YOLO

D = "/share/data4/yhh/data/human-count/det-bench"
W = f"{D}/runs/yolo26s/weights/best.pt"
files = sorted(glob.glob(f"{D}/raw/val/*.jpg"))
print("n_files", len(files), flush=True)

im0 = cv2.imread(files[0])
print("img0", files[0], im0.shape, flush=True)
sizes = {}
for f in files[::60][:10]:
    im = cv2.imread(f)
    key = str(im.shape) if im is not None else "READ_FAIL"
    sizes[key] = sizes.get(key, 0) + 1
print("sizes sample:", sizes, flush=True)

dev = 1
m = YOLO(W)
print("names", m.names, flush=True)
try:
    ta = m.ckpt.get("train_args", {})
    print("train imgsz", ta.get("imgsz"), "batch", ta.get("batch"), flush=True)
except Exception as e:  # noqa: BLE001
    print("ckpt meta err", e, flush=True)
print("after load mem %.2f GB" % (torch.cuda.memory_allocated(dev) / 1e9), flush=True)

t0 = time.time()
r = m.predict(files[0], imgsz=960, conf=0.05, device=dev, verbose=False)
print("single ok n=%d %.2fs mem %.2f GB" % (
    len(r[0].boxes), time.time() - t0, torch.cuda.memory_allocated(dev) / 1e9), flush=True)
try:
    print("predictor imgsz:", getattr(m.predictor, "imgsz", "?"), flush=True)
except Exception:  # noqa: BLE001
    pass

torch.cuda.reset_peak_memory_stats(dev)
t0 = time.time()
r = m.predict(files[:4], imgsz=960, conf=0.05, device=dev, batch=4, verbose=False)
print("batch4 ok n=%d %.2fs peak %.2f GB" % (
    sum(len(x.boxes) for x in r), time.time() - t0,
    torch.cuda.max_memory_allocated(dev) / 1e9), flush=True)

torch.cuda.reset_peak_memory_stats(dev)
t0 = time.time()
gen = m.predict(files[:16], imgsz=960, conf=0.05, device=dev, batch=16, verbose=False, stream=True)
cnt = sum(len(x.boxes) for x in gen)
print("batch16 stream ok n=%d %.2fs peak %.2f GB" % (
    cnt, time.time() - t0, torch.cuda.max_memory_allocated(dev) / 1e9), flush=True)

torch.cuda.reset_peak_memory_stats(dev)
t0 = time.time()
gen = m.predict(files[:32], imgsz=960, conf=0.05, device=dev, batch=16, verbose=False, stream=True)
cnt = sum(len(x.boxes) for x in gen)
print("batch16x32 stream ok n=%d %.2fs peak %.2f GB" % (
    cnt, time.time() - t0, torch.cuda.max_memory_allocated(dev) / 1e9), flush=True)
print("DONE", flush=True)
