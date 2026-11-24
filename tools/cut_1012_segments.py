#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1012 长素材切分：train/val/test 三集切片（跨时段防泄漏设计）。

设计（2026-10-13，任务：身份选择器 + 轨迹意图预测的数据底座）：
- 4 点位（320 球机 / 318 办公区2 / 323 茶水间 / 305 办公区1），每点位 3 段：
  train = 上午段 60min（08:00-12:00 时段）
  val   = 午间段 15min（12:00-13:30 时段）
  test  = 下午段 15min（13:30-18:00 时段）
- 合计 6h（train 4h / val 1h / test 1h）；训练集占比 67%（后续可依证据扩量）。
- 切片用 ffmpeg -ss(输入 seek) + -c copy（无损、快）；时间基准为卷内秒偏移。
  注：卷起始 OSD 为近似值（README），偏移取整分钟；段边界误差数秒可接受。

输出：
- media-resource/1012/segments/{cam}_{split}_{cc}.mp4
- media-resource/1012/segments/segments_meta.json
"""
import json
import os
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "media-resource", "1012")
OUT = os.path.join(SRC, "segments")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# (点位, split, 摄像头号, 卷文件, 卷起录 OSD, 段偏移秒, 段时长秒)
SEGS = [
    # ---- 320 办公区球机 ----
    ("320", "train", "514", "0100000251400000000.mp4", "08:03",          0, 60 * 60),
    ("320", "val",   "519", "0100000251900000000.mp4", "12:39",          0, 15 * 60),
    ("320", "test",  "518", "0100000251800000000.mp4", "15:03",          0, 15 * 60),
    # ---- 318 办公区2 ----
    ("318", "train", "525", "0100000252500000000.mp4", "08:22",          0, 60 * 60),
    ("318", "val",   "525", "0100000252500000000.mp4", "08:22",  4 * 3600, 15 * 60),  # 12:22
    ("318", "test",  "534", "0100000253400000000.mp4", "14:22",          0, 15 * 60),
    # ---- 323 茶水间 ----
    ("323", "train", "510", "0100000251000000000.mp4", "08:35",          0, 60 * 60),
    ("323", "val",   "538", "0100000253800000000.mp4", "11:51",    24 * 60, 15 * 60),  # 12:15
    ("323", "test",  "527", "0100000252700000000.mp4", "12:57",    48 * 60, 15 * 60),  # 13:45
    # ---- 305 办公区1 ----
    ("305", "train", "529", "0100000252900000000.mp4", "09:28",          0, 60 * 60),
    ("305", "val",   "517", "0100000251700000000.mp4", "11:24",    66 * 60, 15 * 60),  # 12:30
    ("305", "test",  "544", "0100000254400000000.mp4", "13:28",    62 * 60, 15 * 60),  # 14:30
]


def cut_one(cam, split, cc, vol, osd, ss, dur):
    src = os.path.join(SRC, vol)
    dst = os.path.join(OUT, f"{cam}_{split}_{cc}.mp4")
    if not os.path.exists(src):
        print(f"[MISS] {src}")
        return None
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print(f"[SKIP] exists {dst}")
        return dst
    cmd = [FFMPEG, "-y", "-ss", str(ss), "-i", src, "-t", str(dur),
           "-c", "copy", "-avoid_negative_ts", "make_zero", dst]
    t = __import__("time").time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    el = __import__("time").time() - t
    if r.returncode != 0:
        print(f"[FAIL] {cam}/{split} rc={r.returncode}\n{r.stderr[-600:]}")
        return None
    sz = os.path.getsize(dst) / 1e6
    print(f"[OK] {cam}_{split}_{cc}.mp4  {sz:.0f} MB  ({el:.0f}s)  src={vol} ss={ss}s dur={dur}s")
    return dst


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for cam, split, cc, vol, osd, ss, dur in SEGS:
        dst = cut_one(cam, split, cc, vol, osd, ss, dur)
        if dst:
            meta.append({
                "seg": f"{cam}_{split}_{cc}", "cam": cam, "split": split,
                "cc": cc, "vol": vol, "osd_start": osd,
                "ss": ss, "dur": dur, "file": os.path.basename(dst),
            })
    mp = os.path.join(OUT, "segments_meta.json")
    json.dump({"note": "1012 train/val/test 切分（跨时段设计）", "segments": meta},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[DONE] {len(meta)}/12 segments -> {mp}")


if __name__ == "__main__":
    sys.exit(main())
