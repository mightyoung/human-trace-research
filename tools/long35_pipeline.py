#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""long35 全链路驱动：编码 → 同机位拼接 → 实体图（clean2 口径）。

为什么单独写驱动，而不是逐条手工命令：
  * 六路要按同一套参数跑，手敲 6 条命令行容易漏（317/343 打错一次就是半小时）
  * 每步产物要能被 `ops_run --out` 一次扫到，run 记录才完整
  * 单步失败要能只重跑那一步（`--only encode` / `--only stitch`）

链路（与 10min 生产口径一致，仅换数据源与编码器）：

  cleaning/{cam}_cleaned.jsonl  ──┐
                                  ├─ encode(O SNet msmt17) → embs_osnet/{cam}_emb_long35_osnet.npz
  long-35min/{cam}_long35.mp4   ──┘
                                        │
                                        ├─ stitch(track_stitching) → stitch_long35/stitch_{cam}.csv
                                        │
                                        └─ entities(rebuild_clean2_entities) → stitch_long35/net_tracks_{cam}.json

用法：
    python tools/long35_pipeline.py --only encode --cams 316
    python tools/long35_pipeline.py --only encode
    python tools/long35_pipeline.py --only stitch
    python tools/long35_pipeline.py --only entities
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

CAMS = ["305", "316", "318", "320", "323", "343"]
RES = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}

SRC = ROOT / "output_crosscam"
LONG = SRC / "long35"
VIDEO_DIR = ROOT / "media-resource" / "multi-cemra" / "long-35min"
CLEAN_DIR = LONG / "cleaning"
EMB_DIR = LONG / "embs_osnet"
STITCH_DIR = LONG / "stitch_long35"


def py() -> str:
    return sys.executable


def jsonl_for(cam: str) -> Path:
    """优先 C 配置产物（hybrid NMS + min-h 200，2026-11-10），其次清洗版，最后 raw 检测。

    C 只重跑了 305：其余 5 路的 C 配置与 base 差异 ≤2 个断点（318/323 完全等同），
    为省 9 小时解码没有重跑，沿用旧产物——混用不影响下游（按机位各自读文件）。
    """
    p_c = LONG / f"{cam}_long35_detections_c.jsonl"
    if p_c.exists():
        return p_c
    for name in (f"{cam}_masked_cleaned_v2.jsonl", f"{cam}_masked_cleaned.jsonl",
                 f"{cam}_cleaned.jsonl"):
        p = CLEAN_DIR / name
        if p.exists():
            return p
    return LONG / f"{cam}_long35_detections.jsonl"


def video_for(cam: str) -> Path:
    return VIDEO_DIR / f"{cam}_long35.mp4"


def run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def step_encode(cams: list[str], samples: int, device: str, threads: int = 4,
                seek: str = "random") -> int:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    bad = 0
    for cam in cams:
        out = EMB_DIR / f"{cam}_emb_long35_osnet.npz"
        if out.exists() and out.stat().st_size > 1024:
            print(f"[skip] {cam} 已有 {out.name}（{out.stat().st_size/1e6:.2f} MB）", flush=True)
            continue
        j, v = jsonl_for(cam), video_for(cam)
        if not v.exists():
            print(f"[warn] 缺视频 {v}", flush=True)
            bad += 1
            continue
        t0 = time.time()
        rc = run([py(), str(TOOLS / "local_encode_osnet.py"),
                  "--jsonl", str(j), "--video", str(v),
                  "--prefix", f"v{cam}_", "--out", str(out),
                  "--samples", str(samples), "--device", device,
                  "--threads", str(threads), "--seek", seek])
        dt = time.time() - t0
        print(f"[{'ok' if rc == 0 else 'FAIL'}] {cam} {dt:.1f}s  jsonl={j.name}", flush=True)
        bad += (rc != 0)
    return bad


def step_stitch(cams: list[str]) -> int:
    STITCH_DIR.mkdir(parents=True, exist_ok=True)
    bad = 0
    for cam in cams:
        emb = EMB_DIR / f"{cam}_emb_long35_osnet.npz"
        if not emb.exists():
            print(f"[warn] {cam} 缺特征 {emb.name}", flush=True)
            bad += 1
            continue
        out = STITCH_DIR / f"stitch_{cam}.csv"
        w, h = RES[cam]
        rc = run([py(), str(TOOLS / "track_stitching.py"),
                  "--jsonl", str(jsonl_for(cam)), "--wh", str(w), str(h),
                  "--emb", str(emb), "--prefix", f"v{cam}_", "--out-csv", str(out)])
        n = sum(1 for _ in open(out, encoding="utf-8")) - 1 if out.exists() else 0
        print(f"[{'ok' if rc == 0 else 'FAIL'}] {cam} stitch rows={n}", flush=True)
        bad += (rc != 0)
    return bad


def step_entities(cams: list[str]) -> int:
    """实体图：复用 rebuild_clean2_entities 的算法，输入重定向到 long35。

    该脚本的 `view_path()` 读 `SRC/cleanset_v2/{cam}_detections.jsonl`、`clean2_links()`
    读 `STITCH/stitch_{cam}.csv`，两者都是模块常量。为了不改算法，这里
    `--src <long35 根>` + `--stitch <long35 stitch 目录>`，并把清洗版布置成
    `long35/cleanset_v2/{cam}_detections.jsonl`（与生产同构）。
    """
    STITCH_DIR.mkdir(parents=True, exist_ok=True)
    stage = LONG / "cleanset_v2"
    stage.mkdir(parents=True, exist_ok=True)
    # 布置 clean2 轨迹源（用软链接避免再占一份 60MB；Windows 无权限时退回复制）
    import shutil
    for cam in cams:
        src = jsonl_for(cam)
        dst = stage / f"{cam}_detections.jsonl"
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        print(f"[stage] {cam} ← {src.name}", flush=True)
    t0 = time.time()
    rc = run([py(), str(TOOLS / "rebuild_clean2_entities.py"),
              "--view", "clean2", "--out", str(STITCH_DIR),
              "--src", str(LONG), "--stitch", str(STITCH_DIR)])
    print(f"[{'ok' if rc == 0 else 'FAIL'}] entities {time.time()-t0:.1f}s", flush=True)
    return 0 if rc == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["encode", "stitch", "entities", "all"], default="all")
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seek", default="random", choices=["random", "sequential"],
                    help="取帧方式：random=逐 crop seek（默认，长视频实测更稳）；"
                         "sequential=顺序解码一遍（crop 密集时更快，稀疏时更慢）")
    a = ap.parse_args()
    cams = [c.strip() for c in a.cams.split(",") if c.strip()]

    print(f"[long35] cams={cams} only={a.only} samples={a.samples} device={a.device}", flush=True)
    bad = 0
    if a.only in ("encode", "all"):
        print("== encode ==", flush=True)
        bad += step_encode(cams, a.samples, a.device, a.threads, a.seek)
    if a.only in ("stitch", "all"):
        print("== stitch ==", flush=True)
        bad += step_stitch(cams)
    if a.only in ("entities", "all"):
        print("== entities ==", flush=True)
        bad += step_entities(cams)
    print(f"[long35] done, failures={bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
