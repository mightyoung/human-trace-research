#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测完成后处理：下载 → 清洗(323) → final 集 → a/b 分片 → 上传（本地运行）。

依据全量标记审计结论（docs/reflection-cleaning-integration-report.md §0.c）：
  - 323（茶水间，有镜+地板倒影）：运动学清洗 + 玻璃掩膜（dualclean）
  - 320/318/305（无可靠倒影证据；318 运动学全 FP 史）：保持 raw
"""
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = r"C:\Users\might\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
REMOTE = os.path.join(ROOT, "tools", "remote_run.py")

SEGS = [
    "320_train_514", "318_train_525", "323_train_510", "305_train_529",
    "320_val_519", "318_val_525", "323_val_538", "305_val_517",
    "320_test_518", "318_test_534", "323_test_527", "305_test_544",
]
REMOTE_BASE = "/share/data4/yhh/data/human-count"
DOWN = os.path.join(ROOT, "output_crosscam", "inference-out-1012")
FINAL = os.path.join(ROOT, "output_crosscam", "1012_final")
AB = os.path.join(FINAL, "ab")


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("[CMD FAIL]", cmd[:120], "\n", r.stderr[-500:])
        sys.exit(1)
    return r


def step_download():
    os.makedirs(DOWN, exist_ok=True)
    for seg in SEGS:
        d = os.path.join(DOWN, seg)
        os.makedirs(d, exist_ok=True)
        for f in (f"{seg}_detections.jsonl", f"{seg}_summary.json"):
            dst = os.path.join(d, f)
            if os.path.exists(dst) and os.path.getsize(dst) > 1000:
                continue
            env = dict(os.environ, MSYS_NO_PATHCONV="1")
            r = subprocess.run([sys.executable if sys.executable else PY, REMOTE, "--get",
                                f"{REMOTE_BASE}/inference-out-1012/{seg}/{f}", dst],
                               capture_output=True, text=True, env=env, cwd=ROOT)
            ok = os.path.exists(dst) and os.path.getsize(dst) > 1000
            print(f"[download] {seg}/{f} {'OK' if ok else 'FAIL'}")


def step_clean_323():
    """323 四段：运动学清洗 + 玻璃掩膜。"""
    outd = os.path.join(ROOT, "output_crosscam", "1012_cleaning")
    os.makedirs(outd, exist_ok=True)
    segs323 = [s for s in SEGS if s.startswith("323")]
    jobs = ",".join(f"323={os.path.join(DOWN, s, s + '_detections.jsonl')}" for s in segs323)
    # track_cleaning 对多任务同名会冲突——逐段跑
    for s in segs323:
        jp = os.path.join(DOWN, s, f"{s}_detections.jsonl")
        r = sh(f'"{PY}" "{os.path.join(ROOT, "tools", "track_cleaning.py")}" '
               f'--jobs "323={jp}" --out-dir "{outd}/{s}"')
        tail = [l for l in r.stdout.splitlines() if "轨迹" in l or "倒影" in l or "[out]" in l]
        print(f"[clean] {s}: " + " | ".join(tail[-3:]) if tail else f"[clean] {s} done")
    # 掩膜
    masks = os.path.join(ROOT, "output_crosscam", "glass_mask", "glass_masks.json")
    for s in segs323:
        inp = os.path.join(outd, s, "323_cleaned.jsonl")
        outp = os.path.join(outd, s, "323_masked_cleaned.jsonl")
        if not os.path.exists(inp):
            print(f"[mask] {s}: no cleaned file, skip"); continue
        r = sh(f'"{PY}" "{os.path.join(ROOT, "tools", "glass_mask.py")}" --apply '
               f'--masks "{masks}" --cam 323 --in-path "{inp}" --out_path "{outp}" '
               f'--wh 1920 1080', check=False)
        print(f"[mask] {s}: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "done"))


def step_final_and_ab():
    os.makedirs(FINAL, exist_ok=True)
    os.makedirs(AB, exist_ok=True)
    for seg in SEGS:
        dst = os.path.join(FINAL, f"{seg}.jsonl")
        if seg.startswith("323"):
            src = os.path.join(ROOT, "output_crosscam", "1012_cleaning", seg,
                               "323_masked_cleaned.jsonl")
            if not os.path.exists(src):
                src = os.path.join(DOWN, seg, f"{seg}_detections.jsonl")
        else:
            src = os.path.join(DOWN, seg, f"{seg}_detections.jsonl")
        shutil.copyfile(src, dst)
        r = sh(f'"{PY}" "{os.path.join(ROOT, "tools", "make_ab_jsonl.py")}" '
               f'--in-jsonl "{dst}" --out-prefix "{os.path.join(AB, seg)}" --min-frames 6')
        print("[ab]", r.stdout.strip().splitlines()[-1])


def step_upload_ab():
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    files = sorted(glob.glob(os.path.join(AB, "*.jsonl")))
    for f in files:
        name = os.path.basename(f)
        subprocess.run([sys.executable, REMOTE, "--put", f,
                        f"{REMOTE_BASE}/inference-out-1012-ab/{name}"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
        print(f"[upload] {name}")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["download", "clean", "final", "upload"]
    if "download" in steps:
        step_download()
    if "clean" in steps:
        step_clean_323()
    if "final" in steps:
        step_final_and_ab()
    if "upload" in steps:
        step_upload_ab()
    print("[post-detect done]")
