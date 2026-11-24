#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描 zip / 目录，找出所有「含人像像素」的文件。

判定三类：
  A 图片文件本身（jpg/png/webp/bmp/gif/tif）
  B 文件内嵌图片（HTML 的 `data:image/...;base64,`，CSV/JSON/MD 里同理）
  C 干净

用法
  PY tools/scan_sensitive.py <zip 或目录> [--json out.json]
  PY tools/scan_sensitive.py --clean <zip> --out <新zip>   # 生成去图版（仅 HTML 内嵌可清）

B 类的 payload 总字节数会一并统计，便于判断包体里有多少是影像。
"""
import argparse
import json
import os
import re
import sys
import zipfile

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}
MAGIC = [b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"RIFF"]
DATA_URI = re.compile(rb"data:image/[a-zA-Z]+;base64,([A-Za-z0-9+/=\s]+)")

PLACEHOLDER = ("data:image/gif;base64,"
               "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
BANNER = ("<div style=\"position:sticky;top:0;z-index:99999;background:#7f1d1d;color:#fff;"
          "padding:10px 14px;font:14px/1.6 system-ui,\\\"Microsoft YaHei\\\";text-align:center\">"
          "本页为<b>去图版</b>：所有人像裁图已移除，仅保留标注逻辑与判据，避免个人影像外流。"
          "</div>")


def scan_bytes(b):
    """返回 (内嵌图片数, payload 字节数)"""
    n, total = 0, 0
    for m in DATA_URI.finditer(b):
        n += 1
        total += len(m.group(1))
    return n, total


def check(path, data):
    ext = os.path.splitext(path)[1].lower()
    if ext in VID_EXT or data[4:8] == b"ftyp":
        return "V", len(data), 0
    if ext in IMG_EXT:
        return "A", len(data), 0
    n, total = scan_bytes(data)
    if n:
        return "B", n, total
    # 兜底：裸二进制魔数（排除纯文本里的 RIFF 字样）
    if ext not in {".csv", ".md", ".json", ".txt", ".html", ".bat", ".py"}:
        for mg in MAGIC:
            if data.startswith(mg):
                return "A", len(data), 0
    return "C", 0, 0


def scan_zip(p):
    rows = []
    with zipfile.ZipFile(p) as z:
        for i in z.infolist():
            if i.is_dir():
                continue
            data = z.read(i.filename)
            kind, n, total = check(i.filename, data)
            rows.append(dict(path=i.filename, kind=kind, n=n, payload=total,
                             size=len(data)))
    return rows


def scan_dir(p):
    rows = []
    for root, _, files in os.walk(p):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, p).replace(os.sep, "/")
            data = open(full, "rb").read()
            kind, n, total = check(rel, data)
            rows.append(dict(path=rel, kind=kind, n=n, payload=total,
                             size=len(data)))
    return rows


def report(rows, title):
    a = [r for r in rows if r["kind"] == "A"]
    b = [r for r in rows if r["kind"] == "B"]
    v = [r for r in rows if r["kind"] == "V"]
    c = [r for r in rows if r["kind"] == "C"]
    pay = sum(r["payload"] for r in b)
    print(f"=== {title} ===")
    print(f"  文件 {len(rows)}｜A 图片 {len(a)}（{sum(r['size'] for r in a)/1e6:.1f} MB）"
          f"｜V 视频 {len(v)}（{sum(r['size'] for r in v)/1e6:.1f} MB）"
          f"｜B 内嵌图片 {len(b)} 个文件 / {sum(r['n'] for r in b)} 张（payload {pay/1e6:.1f} MB）"
          f"｜C 干净 {len(c)}")
    if v:
        by_dir = {}
        for r in v:
            d = os.path.dirname(r["path"]) or "."
            by_dir[d] = by_dir.get(d, 0) + 1
        for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    [V] {d}/ … {n} 个")
    if a:
        by_dir = {}
        for r in a:
            d = os.path.dirname(r["path"]) or "."
            by_dir[d] = by_dir.get(d, 0) + 1
        for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    [A] {d}/ … {n} 个")
    for r in sorted(b, key=lambda x: -x["payload"]):
        print(f"    [B] {r['path']}  {r['n']} 张 / {r['payload']/1e6:.1f} MB payload")
    return a, b, c


def clean_zip(src, dst):
    """把 HTML 里的 data:image 全部换成 1x1 占位，并加顶部横幅；图片文件直接丢弃。"""
    kept, dropped, objs = 0, 0, 0
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
            dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
        for i in zin.infolist():
            if i.is_dir():
                continue
            data = zin.read(i.filename)
            ext = os.path.splitext(i.filename)[1].lower()
            if ext in IMG_EXT:
                dropped += 1
                continue
            n, _ = scan_bytes(data)
            if n:
                objs += n
                if ext in {".html", ".htm"}:
                    txt = data.decode("utf-8", "replace")
                    txt = DATA_URI.sub(PLACEHOLDER.encode(), txt.encode()).decode()
                    for marker in ("<body>", "<body ", "<BODY>"):
                        k = txt.find(marker)
                        if k >= 0:
                            end = txt.find(">", k) + 1
                            txt = txt[:end] + BANNER + txt[end:]
                            break
                    data = txt.encode()
                else:
                    data = DATA_URI.sub(PLACEHOLDER.encode(), data)
            zout.writestr(i.filename, data)
            kept += 1
    print(f"[clean] {dst}\n  保留 {kept} 个文件（其中 {objs} 张内嵌图已替换为占位）"
          f"｜丢弃图片文件 {dropped} 个｜{os.path.getsize(dst)/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.clean:
        assert a.out, "--clean 需要 --out"
        clean_zip(a.target, a.out)
        return
    rows = scan_zip(a.target) if a.target.endswith(".zip") else scan_dir(a.target)
    report(rows, a.target)
    if a.json:
        json.dump(rows, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  [json] {a.json}")


if __name__ == "__main__":
    main()
