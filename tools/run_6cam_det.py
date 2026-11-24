#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""6 机位同步窗检测（multi-cemra long35，生产 YOLO26s-seg）。

对齐依据：六路 long35 视频首帧 OSD 实测
  343 08:53:19 / 305 08:54:5? / 316 08:55:04 / 320 08:55:19 / 318 08:55:20 / 323 08:55:18
以最晚的 318（08:55:20）为公共起点，各路裁掉自身偏移。

切分（公共窗 33 min）：
  train  0–720 s（12 min）
  gap    720–810 s（1.5 min，时间隔离缓冲）
  test   810–1410 s（10 min）
  imgsz 960（实测检出与 1280 持平：193 vs 197 框，速度快 1.6×）

用法：PY tools/run_6cam_det.py [--workers 2] [--only 305,343]
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
VID = os.path.join(ROOT, "media-resource/multi-cemra/long-35min")
OUT = os.path.join(ROOT, "output_crosscam_local/6cam_yolo")

# cam -> 相对 08:55:20 的起始偏移（秒）
OFFSET = {"305": 25, "316": 16, "318": 0, "320": 1, "323": 2, "343": 121}
TRAIN_DUR, TEST_START, TEST_DUR = 720, 810, 600
IMGSZ = 960
STRIDE_TR, STRIDE_TE = 10, 5


def run_cam(cam):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.environ.get("HC_PYTHONPATH", "D:/human-count/pylibs")
    env["TMPDIR"] = env["TEMP"] = os.environ.get("HC_TMPDIR", "D:/human-count/tmp")
    log = []
    for split, ss, dur, stride in (
        ("tr", OFFSET[cam], TRAIN_DUR, STRIDE_TR),
        ("te", OFFSET[cam] + TEST_START, TEST_DUR, STRIDE_TE),
    ):
        seg = f"{cam}_6cam_{split}"
        outp = os.path.join(OUT, seg, f"{seg}_detections.jsonl")
        if os.path.exists(outp):
            n = sum(1 for _ in open(outp, encoding="utf-8"))
            os.remove(outp)          # 冒烟残留，正式重跑
        cmd = [PY, os.path.join(ROOT, "tools/yolo_track_det.py"),
               "--seg", seg, "--video", os.path.join(VID, f"{cam}_long35.mp4"),
               "--start-sec", str(ss), "--dur-sec", str(dur), "--stride", str(stride),
               "--imgsz", str(IMGSZ),
               "--out-dir", OUT, "--force"]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        tail = [ln for ln in (p.stdout or "").splitlines() if "DONE" in ln or "Error" in ln]
        log.append(f"[{seg}] {time.time()-t0:.0f}s rc={p.returncode} {' | '.join(tail[-1:])}")
        if p.returncode != 0:
            log.append(f"[{seg}] STDERR: {(p.stderr or '')[-600:]}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    cams = [c.strip() for c in a.only.split(",") if c.strip()] or list(OFFSET)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(run_cam, cams):
            for ln in res:
                print(ln, flush=True)
    print(f"[ALL DONE] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
