#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核对预注册 §7 列出的文件 sha256 与本机是否一致。

用法（仓库根目录或任意目录均可）：
    python tools/verify_preeg_hashes.py

退出码：0 全部一致；1 存在不一致（按预注册 §6 视为偏离，须记录后再处理）。

独立性说明（重要）：
  §7 表、JSON 里的哈希、以及导出包 MANIFEST.md 的前 16 位，都是从**导出包内的存档副本**
  算出来的，属同源记录，只做一致性检查，不是独立证据。
  真正独立的对比只有一组：**仓库工作副本 ↔ 导出包内的存档副本**（本脚本第 3 块，逐字节）。
"""
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs/PREREGISTRATION_holdout_2026-10-24.md"
JS = ROOT / "docs/preregistration_holdout_2026-10-24.json"
EXPORT_MANIFEST = ROOT / "dist/export-2026-10-24/MANIFEST.md"

FILES = [
    ("检测权重 yolo26s-seg.pt",
     "models/Ultralytics--YOLO26/snapshots/master/yolo26s-seg.pt"),
    ("编码权重 osnet_x1_0_...pth",
     "models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"),
    ("tools/yolo_track_det.py", "tools/yolo_track_det.py"),
    ("tools/bytetrack_1012.yaml", "tools/bytetrack_1012.yaml"),
    ("tools/dedup_detections.py", "tools/dedup_detections.py"),
]

JSON_KEYS = {
    "tools/yolo_track_det.py": ("pipeline.detector", None),
    "tools/bytetrack_1012.yaml": ("pipeline.tracker.config_sha256", None),
    "tools/dedup_detections.py": ("pipeline.same_frame_dedup.script_sha256", None),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    md = io.open(MD, encoding="utf-8").read()
    sec7 = md.split("## 7 ")[1].split("## 8 ")[0]
    declared = re.findall(r"[a-f0-9]{64}", sec7)

    if len(declared) != 5:
        print(f"[警告] §7 中解析出 {len(declared)} 个 64 位哈希，预期 5 个；请检查表格格式")
        return 1

    js = json.load(io.open(JS, encoding="utf-8"))

    def json_get(dotted):
        cur = js
        for part in dotted.split("."):
            cur = cur[part]
        return cur

    print("=" * 104)
    print(f"{'文件':<34}{'大小(B)':>12}  结论  sha256")
    print("=" * 104)

    ok_all = True
    for (label, rel), dec in zip(FILES, declared):
        p = ROOT / rel
        if not p.exists():
            print(f"{label:<34}{'MISSING':>12}  FAIL")
            ok_all = False
            continue
        got = sha256_file(p)
        ok = got == dec
        ok_all &= ok
        print(f"{label:<34}{p.stat().st_size:>12,}  {'OK  ' if ok else 'FAIL'}  {got}")
        if not ok:
            print(f"{'':<34}{'':>12}        §7 声明: {dec}")

    print("\n-- JSON 与 §7 是否自洽 --")
    for rel, (dotted, _) in JSON_KEYS.items():
        idx = [r for _, r in FILES].index(rel)
        val = json_get(dotted)
        m = re.search(r"[a-f0-9]{64}", val) if isinstance(val, str) else None
        val = m.group(0) if m else val
        same = val == declared[idx]
        ok_all &= same
        print(f"  {rel:<32} JSON {str(val)[:16]}…  §7 {declared[idx][:16]}…  {'一致' if same else '不一致'}")

    print("\n-- 仓库副本 ↔ 导出包存档副本（唯一独立对比，完整哈希） --")
    arch_root = ROOT / "dist/export-2026-10-24/07_code/tools"
    if arch_root.exists():
        for (label, rel), dec in list(zip(FILES, declared))[2:]:
            base = os.path.basename(rel)
            ap = arch_root / base
            if not ap.exists():
                print(f"  {base:<28} 存档副本缺失，跳过")
                continue
            ah = sha256_file(ap)
            same = ah == dec
            ok_all &= same
            print(f"  {base:<28} 存档 {ah[:16]}…  {'与仓库逐字节相同' if same else '与仓库不同'}")
    else:
        print("  [跳过] 未找到存档副本目录")

    print("\n-- 与导出包 MANIFEST.md 记录比对（同源记录，仅作一致性检查） --")
    if EXPORT_MANIFEST.exists():
        man = io.open(EXPORT_MANIFEST, encoding="utf-8").read()
        for (label, rel), dec in list(zip(FILES, declared))[2:]:
            base = os.path.basename(rel)
            m = re.search(r"07_code/tools/" + re.escape(base) +
                          r"`\s*\|\s*([\d,]+)\s*\|\s*`([a-f0-9]{16})`", man)
            if m:
                ok = dec.startswith(m.group(2))
                ok_all &= ok
                print(f"  {base:<28} 导出记录 {m.group(2)}…  大小 {m.group(1):>6}  {'一致' if ok else '不一致'}")
            else:
                print(f"  {base:<28} 未在 MANIFEST 中找到")
    else:
        print("  [跳过] 未找到导出包 MANIFEST.md（不影响前两项结论）")

    print("\n" + "=" * 104)
    print("结论:", "全部一致，未偏离" if ok_all else "存在不一致 → 按预注册 §6 记录偏离，不得静默运行")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
