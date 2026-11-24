#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 10min 与 35min 两套 Web 数据之间切换（可回滚）。

背景：控制台的数据包（cams.json / tracks/ / entities.json / chains.json）与媒体
（data/media/*.mp4）必须**成套**换，否则会出现"时间轴 10min、视频 35min"这类错配。

本脚本：
  * `snapshot`  —— 把当前 data/ 的快照存到 `data_snap_<tag>/`（首次切换前必做一次）
  * `use <tag>` —— 把 `data_<tag>/` 的内容覆盖到 `data/`，并记录当前 tag
  * `status`    —— 打印当前用的是哪套、媒体时长

约定目录：
  webapp/backend/data/            线上（服务读这里）
  webapp/backend/data_35min/      build_web_data --out-dir 产物
  webapp/backend/data_snap_10min/ 10min 快照
  webapp/backend/data/media_long35/  35min 转码媒体（long35_media.py --dst-dir）

用法：
    python tools/long35_switch.py status
    python tools/long35_switch.py snapshot --tag 10min
    python tools/long35_switch.py use 35min
    python tools/long35_switch.py use snap_10min      # 回滚
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "webapp" / "backend" / "data"
PACK_FILES = ["cams.json", "entities.json", "chains.json", "metrics.json",
              "masks.json", "mirror.json", "pairs.json", "gil.json", "suspects.json"]
PACK_DIRS = ["tracks"]

# 标注类文件**不随片段切换**：判定属于"人"，不属于"哪一段"。
# 且 long35 前 10min 的 track id 与 10min 版逐位一致（已核 6 路），
# 所以 10min 期间人工判的条目在 35min 里仍然指向同一个人。
KEEP_FILES = {"labels.json", "mirror.json", "pairs.json"}


def read_session() -> dict:
    p = DATA / "cams.json"
    if not p.exists():
        return {}
    return (json.loads(p.read_text(encoding="utf-8")) or {}).get("session", {})


def status() -> None:
    s = read_session()
    media = sorted((DATA / "media").glob("*.mp4"))
    print(f"数据目录: {DATA}")
    print(f"session : window_sec={s.get('window_sec')} base_wall={s.get('base_wall')}")
    if media:
        print(f"媒体    : {len(media)} 路，共 {sum(p.stat().st_size for p in media)/1e6:.1f} MB")
    marker = DATA / ".pack_tag"
    print(f"当前标记: {marker.read_text(encoding='utf-8').strip() if marker.exists() else '(无)'}")


def snapshot(tag: str) -> None:
    dst = DATA.parent / f"data_snap_{tag}"
    if dst.exists():
        print(f"快照已存在，跳过: {dst}")
        return
    dst.mkdir(parents=True)
    for f in PACK_FILES:
        if (DATA / f).exists():
            shutil.copy2(DATA / f, dst / f)
    for d in PACK_DIRS:
        if (DATA / d).exists():
            shutil.copytree(DATA / d, dst / d)
    if (DATA / "media").exists():
        (dst / "media").mkdir(exist_ok=True)
        for p in (DATA / "media").glob("*.mp4"):
            shutil.copy2(p, dst / "media" / p.name)
    (dst / "SESSION.json").write_text(json.dumps({"session": read_session()}, indent=1),
                                      encoding="utf-8")
    print(f"快照完成: {dst}（{sum(p.stat().st_size for p in dst.rglob('*') if p.is_file())/1e6:.1f} MB）")


def use(tag: str, media_from: str = "") -> None:
    src = DATA.parent / (f"data_{tag}" if not tag.startswith("snap_") else f"data_{tag}")
    if not src.exists():
        raise SystemExit(f"找不到源目录: {src}")
    if not (src / "cams.json").exists():
        raise SystemExit(f"{src} 里没有 cams.json，先跑 build_web_data --out-dir {src}")
    kept = []
    for f in PACK_FILES:
        if f in KEEP_FILES:
            if (DATA / f).exists():
                kept.append(f)
            continue
        if (src / f).exists():
            shutil.copy2(src / f, DATA / f)
        elif f == "suspects.json" and (DATA / f).exists():
            # 源包没有同人候选（老包/未生成）→ 必须删掉，否则残留的是上一片段的 track id
            (DATA / f).unlink()
            print(f"源包无 {f}，已移除旧文件（避免跨片段残留）")
    if kept:
        print(f"保留标注文件（不随片段切换）: {', '.join(kept)}")
    for d in PACK_DIRS:
        if (src / d).exists():
            shutil.rmtree(DATA / d, ignore_errors=True)
            shutil.copytree(src / d, DATA / d)
    # 媒体：优先从 data/media_long35（转码产物），或快照里的 media
    mdir = Path(media_from) if media_from else (DATA.parent / "data" / "media_long35")
    if not mdir.exists():
        mdir = src / "media"
    if mdir.exists():
        n = 0
        for p in mdir.glob("*.mp4"):
            shutil.copy2(p, DATA / "media" / p.name)
            n += 1
        print(f"媒体: {n} 路 ← {mdir}")
    (DATA / ".pack_tag").write_text(tag, encoding="utf-8")
    print(f"已切到 {tag}；session={read_session()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "snapshot", "use"])
    ap.add_argument("tag", nargs="?", default="")
    ap.add_argument("--media-from", default="")
    a = ap.parse_args()
    if a.cmd == "status":
        status()
    elif a.cmd == "snapshot":
        snapshot(a.tag or "10min")
    else:
        if not a.tag:
            raise SystemExit("use 需要 tag，例如 `use 35min`")
        use(a.tag, a.media_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
