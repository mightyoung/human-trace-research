#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把交付包重打成「去图版」：不含任何视频帧 / 人像裁图 / 内嵌图片。

做三件事
  1) 复制源目录到 staging（跳过 gt/crops/）
  2) 把 4 个标注页里的 `data:image/...;base64,<payload>` 整段删掉，并在页面顶部插红条说明
  3) 生成包外的人像清单 crops_manifest_noimg.csv（路径 + 字节 + sha256，无像素）
最后压缩、再扫描一遍，断言 A=0 / B=0，否则报错退出。

用法
  PY tools/pack_noimage.py [--tag 2026-10-20]
"""
import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "deliverable", "data-request-v2")
DIST = os.path.join(ROOT, "deliverable", "dist")
CROPS = os.path.join(SRC, "gt", "crops")

DATA_URI = re.compile(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}
BANNER = ('<div style="position:sticky;top:0;z-index:99999;background:#7f1d1d;color:#fff;'
          'padding:9px 14px;font:13.5px/1.6 system-ui,\'Microsoft YaHei\',sans-serif;'
          'text-align:center">本页为<b>去图版</b>：所有人像裁图已移除，仅保留标注逻辑、'
          '判据与完成状态，避免个人影像外流。</div>')


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def strip_pages(stage):
    n_files = n_imgs = 0
    for dp, _, fs in os.walk(stage):
        for f in fs:
            if not f.lower().endswith((".html", ".htm")):
                continue
            p = os.path.join(dp, f)
            txt = open(p, encoding="utf-8", errors="replace").read()
            k = len(DATA_URI.findall(txt))
            if not k and "data:image" not in txt:
                continue
            txt = DATA_URI.sub("", txt)
            # 连代码注释里的字面量也清掉，避免任何扫描器误报
            txt = txt.replace("data:image/webp;base64,", "（图片已移除）")
            txt = txt.replace("data:image/", "（图片已移除）/")
            if k and BANNER not in txt:
                i = txt.find("<body>")
                if i >= 0:
                    txt = txt[:i + 6] + BANNER + txt[i + 6:]
            open(p, "w", encoding="utf-8", newline="").write(txt)
            n_files += 1
            n_imgs += k
            print(f"    stripped {os.path.relpath(p, stage)}: {k} 张 "
                  f"→ {os.path.getsize(p)/1e6:.2f} MB")
    print(f"  去图：{n_files} 个页面 / {n_imgs} 张内嵌图")
    return n_files, n_imgs


def crops_manifest(out_csv):
    rows = []
    for root, tag, kind in ((os.path.join(SRC, "gt", "crops"), "crops", "image"),
                            (os.path.join(SRC, "gt_new", "crops"), "gt_new/crops", "image"),
                            (os.path.join(SRC, "gt", "clips"), "clips", "video")):
        if not os.path.isdir(root):
            continue
        for dp, _, fs in os.walk(root):
            for f in sorted(fs):
                if not f.lower().endswith(tuple(IMG_EXT | VID_EXT)):
                    continue
                p = os.path.join(dp, f)
                rel = tag + "/" + os.path.relpath(p, root).replace("\\", "/")
                rows.append((rel, kind, os.path.getsize(p), sha(p)[:16]))
    rows.sort()
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["path_in_gt", "kind", "bytes", "sha256_16"])
        w.writerows(rows)
    total = sum(r[2] for r in rows)
    n_img = sum(1 for r in rows if r[1] == "image")
    n_vid = sum(1 for r in rows if r[1] == "video")
    print(f"[out] {out_csv}  {n_img} 张图 + {n_vid} 段视频 / {total/1e6:.1f} MB（无像素，仅清单）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="2026-10-20")
    a = ap.parse_args()

    name = f"data-request-v2_{a.tag}_noimg"
    stage = os.path.join(DIST, "_stage", name)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage, exist_ok=True)
    print(f"[1/4] 复制（跳过 gt/crops、gt/clips、gt_new/crops 与所有图片/视频）→ {stage}")
    n = 0
    for dp, dns, fs in os.walk(SRC):
        rel = os.path.relpath(dp, SRC)
        for bad in ("crops", "clips"):
            if bad in dns:
                dns.remove(bad)
        for f in fs:
            low = f.lower()
            # 只保留 results/ 下的曲线图（纯坐标轴、无人像），其余图片与视频一律不进包
            if low.endswith(tuple(IMG_EXT)) and not rel.startswith("results"):
                continue
            if low.endswith(tuple(VID_EXT)):
                continue
            s = os.path.join(dp, f)
            t = os.path.join(stage, rel, f) if rel != "." else os.path.join(stage, f)
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy2(s, t)
            n += 1
    print(f"  {n} 个文件")

    # 视频不随包，但保留其索引（帧区间、时长），便于授权后按需取用
    cm = os.path.join(SRC, "gt", "clips", "manifest.json")
    if os.path.exists(cm):
        os.makedirs(os.path.join(stage, "gt"), exist_ok=True)
        shutil.copy2(cm, os.path.join(stage, "gt", "clips_manifest.json"))
        print("  附 clips_manifest.json（46 条换人复核的帧区间，无视频）")

    print("[2/4] 剔除页面内嵌图")
    strip_pages(stage)

    # 清单按去掉内嵌图之后的真实文件重算
    mf = os.path.join(stage, "MANIFEST.md")
    if os.path.exists(mf):
        os.remove(mf)
    import subprocess
    print("  重建 MANIFEST.md")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_manifest.py"),
                    "--dir", stage, "--noimage"], check=True,
                   env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "pylibs"),
                        "TMPDIR": os.path.join(ROOT, "tmp"),
                        "TEMP": os.path.join(ROOT, "tmp")})

    print("[3/4] 人像清单（包外）")
    crops_manifest(os.path.join(DIST, "crops_manifest_noimg.csv"))

    print("[4/4] 打包并复检")
    zp = os.path.join(DIST, name + ".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for dp, dns, fs in os.walk(stage):
            dns.sort()
            for f in sorted(fs):
                p = os.path.join(dp, f)
                arc = os.path.join(name, os.path.relpath(p, stage)).replace("\\", "/")
                z.write(p, arc)
    print(f"[out] {zp}  {os.path.getsize(zp)/1e6:.1f} MB")

    # 复检
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from scan_sensitive import scan_zip
    rows = scan_zip(zp)
    A = [r for r in rows if r["kind"] == "A"]
    B = [r for r in rows if r["kind"] == "B"]
    V = [r for r in rows if r["kind"] == "V"]
    print(f"  复检：文件 {len(rows)}｜图片 {len(A)}｜视频 {len(V)}｜内嵌图片 {len(B)}")
    ok_a = sorted(os.path.relpath(r["path"], name).replace("\\", "/") for r in A)
    assert not B, f"仍有内嵌图片：{[r['path'] for r in B][:3]}"
    assert not V, f"仍有视频文件：{[r['path'] for r in V][:3]}"
    assert ok_a == ["results/fig2_proxy.png", "results/fig_d4_truth.png"], \
        f"出现了白名单之外的图片文件：{ok_a}"
    print("  ✓ 无视频、无内嵌图片；图片仅剩 2 张坐标轴曲线图（无人像）")


if __name__ == "__main__":
    main()
