#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 long35 的 6 路高清源转成浏览器可播的 540p 版本（对齐现有 10min 媒体的口径）。

为什么需要：`media-resource/multi-cemra/long-35min/*_long35.mp4` 是 2560×1440 原片
（单路 126–527 MB），直接给浏览器会卡；现有 10min 素材是 960×540 / ~13–24 MB。

做法：
  1. 用 imageio_ffmpeg 自带的 ffmpeg（本机没有系统 ffmpeg）；
  2. 先备份现有 `data/media_{cam}.mp4` 到 `data/media_10min/`（回滚用）；
  3. 转码到临时文件 → ffprobe 校验时长/分辨率 → 原子替换；
  4. 幂等：已存在且时长匹配则跳过。

用法：
    python tools/long35_media.py --probe            # 只打印时长/大小
    python tools/long35_media.py --cams 305 --limit 30   # 先转 30 秒试速度
    python tools/long35_media.py                    # 6 路全转
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "media-resource" / "multi-cemra" / "long-35min"
DST = ROOT / "webapp" / "backend" / "data" / "media"
BAK = ROOT / "webapp" / "backend" / "data" / "media_10min"
CAMS = ["305", "316", "318", "320", "323", "343"]
W, H = 960, 540
CRF = "26"
PRESET = "veryfast"


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: Path) -> tuple[float, int, int]:
    """返回 (duration_s, w, h)；用 ffmpeg -i 的 stderr 解析（无 ffprobe 时也可用）。"""
    exe = ffmpeg()
    p = subprocess.run([exe, "-hide_banner", "-i", str(path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    txt = (p.stderr or "") + (p.stdout or "")
    dur, w, h = 0.0, 0, 0
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", txt)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m2 = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", txt)
    if m2:
        w, h = int(m2.group(1)), int(m2.group(2))
    return dur, w, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--limit", type=float, default=0, help="只转前 N 秒（测速用）")
    ap.add_argument("--crf", default=CRF)
    ap.add_argument("--dst-dir", default="",
                    help="输出目录（默认 webapp/backend/data/media）；建议先转 media_long35/ 再整体切换")
    ap.add_argument("--preset", default=PRESET)
    a = ap.parse_args()
    cams = [c.strip() for c in a.cams.split(",") if c.strip()]
    global DST, BAK
    if a.dst_dir:
        DST = Path(a.dst_dir).resolve()
    DST.mkdir(parents=True, exist_ok=True)

    print(f"[ffmpeg] {ffmpeg()}")
    print(f"{'cam':>4} {'src_min':>8} {'src_MB':>8} {'dst_min':>8} {'dst_MB':>8}  状态")
    for cam in cams:
        src = SRC / f"{cam}_long35.mp4"
        dst = DST / f"{cam}.mp4"
        if not src.exists():
            print(f"{cam:>4}  缺源：{src}")
            continue
        sdur, sw, sh = probe(src)
        ddur, dw, dh = probe(dst) if dst.exists() else (0, 0, 0)
        smb = src.stat().st_size / 1e6
        dmb = dst.stat().st_size / 1e6 if dst.exists() else 0
        # 幂等：已转好（时长接近且是 540p）则跳过
        ok = dst.exists() and abs(ddur - sdur) < 2 and (dw, dh) == (W, H)
        state = "已转" if ok else "待转"
        print(f"{cam:>4} {sdur/60:8.2f} {smb:8.1f} {ddur/60:8.2f} {dmb:8.1f}  {state}"
              f"  src={sw}x{sh} dst={dw}x{dh}")
        if a.probe or ok:
            continue

        if dst.exists() and not BAK.exists():
            BAK.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not (BAK / f"{cam}.mp4").exists():
            shutil.copy2(dst, BAK / f"{cam}.mp4")
            print(f"      备份 → {BAK.name}/{cam}.mp4")

        # ★ 试速模式（--limit）只产出旁路样本，绝不替换正式文件（曾误把 305.mp4 覆盖成 30s）
        if a.limit:
            tmp = dst.with_name(f"{cam}.limit{int(a.limit)}s.mp4")
        else:
            tmp = dst.with_suffix(".tmp.mp4")
        cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error"]
        if a.limit:
            cmd += ["-t", str(a.limit)]
        cmd += ["-i", str(src), "-vf", f"scale={W}:{H}", "-c:v", "libx264",
                "-preset", a.preset, "-crf", a.crf, "-pix_fmt", "yuv420p",
                "-an", "-movflags", "+faststart", str(tmp)]
        t0 = time.time()
        rc = subprocess.call(cmd)
        el = time.time() - t0
        if rc != 0 or not tmp.exists():
            print(f"      FAIL rc={rc}")
            tmp.unlink(missing_ok=True)
            continue
        tdur, tw, th = probe(tmp)
        if abs(tdur - (a.limit or sdur)) > 2 or (tw, th) != (W, H):
            print(f"      校验失败：{tdur:.1f}s {tw}x{th}（期望 {(a.limit or sdur):.1f}s {W}x{H}）")
            continue
        os.replace(tmp, dst)
        print(f"      ok {el:.0f}s → {dst.name} {tdur/60:.2f}min "
              f"{dst.stat().st_size/1e6:.1f}MB ({el/max(tdur,1):.2f}× 实时)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
