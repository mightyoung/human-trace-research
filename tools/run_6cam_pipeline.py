#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""6 机位数据集后端流水线：去重 → a/b 分片 → OSNet 编码。

前置：tools/run_6cam_det.py 已产出 output_crosscam_local/6cam_yolo/<seg>/<seg>_detections.jsonl

用法：
  PY tools/run_6cam_pipeline.py --step dedup
  PY tools/run_6cam_pipeline.py --step ab
  PY tools/run_6cam_pipeline.py --step enc
  PY tools/run_6cam_pipeline.py --step all
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DET = os.path.join(ROOT, "output_crosscam_local/6cam_yolo")
DD = os.path.join(ROOT, "output_crosscam_local/6cam_dd")
AB = os.path.join(ROOT, "output_crosscam_local/6cam_ab")
EMB = os.path.join(ROOT, "output_crosscam_local/6cam_emb")
VID = os.path.join(ROOT, "media-resource/multi-cemra/long-35min")
CAMS = ["305", "316", "318", "320", "323", "343"]
SEGS = [f"{c}_6cam_{s}" for c in CAMS for s in ("tr", "te")]


def sh(cmd):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.environ.get("HC_PYTHONPATH", "D:/human-count/pylibs")
    env["TMPDIR"] = env["TEMP"] = os.environ.get("HC_TMPDIR", "D:/human-count/tmp")
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print("  FAILED:", " ".join(os.path.basename(c) for c in cmd[:2]))
        print((p.stderr or "")[-800:])
    return p.returncode, (p.stdout or "")


def step_dedup():
    os.makedirs(DD, exist_ok=True)
    rc, out = sh([PY, os.path.join(ROOT, "tools/dedup_detections.py"),
                  "--in-dir", DET, "--out-dir", DD, "--iou", "0.5",
                  "--segs", ",".join(SEGS)])
    print("[dedup]", out.strip()[-400:])
    return rc


def step_ab():
    os.makedirs(AB, exist_ok=True)
    for seg in SEGS:
        src = os.path.join(DD, seg, f"{seg}_detections.jsonl")
        if not os.path.exists(src):
            print(f"[ab] 缺 {seg}")
            continue
        rc, out = sh([PY, os.path.join(ROOT, "tools/make_ab_jsonl.py"),
                      "--in-jsonl", src, "--out-prefix", os.path.join(AB, seg),
                      "--min-frames", "6"])
        print(out.strip())
    return 0


def step_enc():
    os.makedirs(EMB, exist_ok=True)
    t0 = time.time()
    for seg in SEGS:
        cam = seg.split("_")[0]
        cmd = [PY, os.path.join(ROOT, "tools/encode_1012_osnet_seq.py"),
               "--seg", seg, "--device", "cpu", "--ab-dir", AB, "--out-dir", EMB,
               "--video", os.path.join(VID, f"{cam}_long35.mp4")]
        t1 = time.time()
        rc, out = sh(cmd)
        tail = [ln for ln in out.splitlines() if "[done]" in ln or "encoded" in ln]
        print(f"[enc] {seg} {time.time()-t1:.0f}s {' | '.join(tail[-2:])}", flush=True)
    print(f"[enc ALL] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["dedup", "ab", "enc", "all"])
    a = ap.parse_args()
    if a.step in ("dedup", "all"):
        step_dedup()
    if a.step in ("ab", "all"):
        step_ab()
    if a.step in ("enc", "all"):
        step_enc()
