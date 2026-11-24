#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出实验数据 + 核心代码（**去图版**）。

硬约束（用户要求）：不包含真实录像的任何图像信息。
因此：
- 任何 .jpg/.jpeg/.png/.bmp/.tif/.mp4/.avi/.mov/.mkv 一律不打包；
- 任何 .npz 特征向量也不打包（生物特征模板，且体积大，可用代码+原始素材重算）；
- 视频目录（media-resource/、datasets/CHIRLA/videos/、datasets/bbox_test/）整目录不进入。

只收白名单扩展名，其余一律跳过，从机制上保证「不会漏进去」。

用法：
  PY tools/export_bundle.py --out dist/export-2026-10-24
产物：
  <out>/README.md  EXCLUDED.md  MANIFEST.md  以及各分类目录；同名 .zip 一并生成。
"""
import argparse
import hashlib
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WHITELIST = {".md", ".csv", ".json", ".jsonl", ".py", ".sh", ".ps1", ".txt", ".tsv", ".yaml", ".yml", ".mat"}
# 目录级排除：图像/特征/日志/缓存一律不进
DENY_DIRS = {"crops", "emb", "emb_time", "emb_rgb", "__pycache__", "node_modules", ".git",
             "log", "logs", "tmp", ".workbuddy", "media-resource", "videos", "bbox_test",
             "dist", "det_cache"}

# (源目录, 目标子目录, 是否递归)
SOURCES = [
    ("deliverable/data-request-v2", "01_onsite_1012", True),
    ("validation/mars", "02_validation_mars", True),
    ("validation/chirla", "03_validation_chirla", True),
    ("validation/chirla_time", "04_validation_chirla_time", True),
    ("validation/chirla_s3", "05_validation_chirla_s3", True),
    ("docs", "06_reports", True),
    ("tools", "07_code/tools", False),
    ("configs", "07_code/configs", True),
    ("datasets", "08_datasets_docs", False),
    ("datasets/MARS_info", "08_datasets_docs/MARS_info", True),
]


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def want(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in WHITELIST:
        return False
    parts = {p.lower() for p in path.replace("\\", "/").split("/")}
    return not (parts & DENY_DIRS)


def collect():
    items = []
    for src, dst, rec in SOURCES:
        base = os.path.join(ROOT, src)
        if not os.path.isdir(base):
            print(f"[skip] 源目录不存在：{src}")
            continue
        if rec:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if d.lower() not in DENY_DIRS)
                for fn in sorted(filenames):
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, base)
                    if want(rel):
                        items.append((full, os.path.join(dst, rel)))
        else:
            for fn in sorted(os.listdir(base)):
                full = os.path.join(base, fn)
                if os.path.isfile(full) and want(fn):
                    items.append((full, os.path.join(dst, fn)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/export-2026-10-24")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    items = collect()
    seen = set()
    rows = []
    for src, rel in items:
        if rel in seen:
            print(f"[dup] 跳过重复目标 {rel}")
            continue
        seen.add(rel)
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        rows.append((rel.replace("\\", "/"), os.path.getsize(dst), sha16(dst)))

    total = sum(r[1] for r in rows)
    groups = {}
    for rel, size, _ in rows:
        groups.setdefault(rel.split("/")[0], [0, 0])
        groups[rel.split("/")[0]][0] += 1
        groups[rel.split("/")[0]][1] += size

    # ---- MANIFEST
    lines = ["# MANIFEST（去图版导出）", "",
             f"- 文件总数 **{len(rows)}**｜合计 **{total / 1024 / 1024:.1f} MiB**",
             f"- 生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
             "- `sha256` 为前 16 位；排序按路径。", "",
             "| 分组 | 文件数 | MiB |", "|---|---|---|"]
    for g in sorted(groups):
        lines.append(f"| {g} | {groups[g][0]} | {groups[g][1] / 1024 / 1024:.1f} |")
    lines += ["", "| 文件 | 字节 | sha256[:16] |", "|---|---|---|"]
    lines += [f"| `{rel}` | {size} | `{h}` |" for rel, size, h in rows]
    open(os.path.join(out, "MANIFEST.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---- 机械化校验：包内不得有任何图像/视频/特征文件
    bad = [r for r, _, _ in rows if os.path.splitext(r)[1].lower() in
           {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp",
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".npz", ".npy", ".pt", ".pth", ".zip"}]
    if bad:
        raise SystemExit(f"发现不该打包的文件：{bad[:5]}")

    print(f"[out] {out}")
    print(f"[files] {len(rows)}｜{total / 1024 / 1024:.1f} MiB")
    for g in sorted(groups):
        print(f"   {g:32s} {groups[g][0]:4d} 个  {groups[g][1] / 1024 / 1024:7.1f} MiB")

    if not a.no_zip:
        zp = out + ".zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for dirpath, _, filenames in os.walk(out):
                for fn in sorted(filenames):
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, os.path.dirname(out)))
        print(f"[zip] {zp}  {os.path.getsize(zp) / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
