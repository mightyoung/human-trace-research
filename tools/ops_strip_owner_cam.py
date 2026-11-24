# -*- coding: utf-8 -*-
"""剔除去重身份表（gil_owner_*.json）里某机位的映射 —— 止损用。

背景
----
C 方案（hybrid NMS + 尺度条件化）上线时重跑了 305 的 detections，
track_id 全部重编号：旧 367 条（1..~400），新 354 条（1..1156）。

但 `gil_owner_*.json` 仍是 2026-10-14 生成的静态文件，里面的
`tid2g` / `owners` 按**旧 tid** 索引（305 只有 1..107，时间只到 594s）。
于是：
  * 新 tid > 107 → 查不到 gid → 回落显示 `#tid`（正确行为）
  * 新 tid <= 107 → 撞上旧编号 → 绑到**完全无关的旧人** → 显示错误 `gXXXX`

本脚本把指定机位从三份身份表里整体剔除，让页面不再显示错的人。
代价：该机位在去重视图/按人着色下不再有全局人编号（显示 `#tid`），
      且不参与跨机位去重统计。要恢复必须重建 owner 层（见 README 待办）。

用法
----
    python tools/ops_strip_owner_cam.py                 # 预览
    python tools/ops_strip_owner_cam.py --apply         # 备份后写入
    python tools/ops_strip_owner_cam.py --cam 305 --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

FILES = [
    "webapp/frontend/public/gil_owner_A.json",
    "webapp/frontend/public/gil_owner_deduprefl.json",
    "webapp/frontend/public/gil_owner_human.json",
    "webapp/frontend/dist/gil_owner_A.json",
    "webapp/frontend/dist/gil_owner_deduprefl.json",
    "webapp/frontend/dist/gil_owner_human.json",
]


def strip(path: Path, cam: str, apply: bool) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    t2g = d.get("tid2g") or {}
    owners = d.get("owners") or []

    keep_t2g = {k: v for k, v in t2g.items() if not k.startswith(f"{cam}:")}
    keep_owners = [r for r in owners if r[1] != cam]

    n_t2g = len(t2g) - len(keep_t2g)
    n_own = len(owners) - len(keep_owners)

    if apply and (n_t2g or n_own):
        bak = path.with_suffix(path.suffix + f".bak_strip{cam}")
        if not bak.exists():
            shutil.copy2(path, bak)
        d["tid2g"] = keep_t2g
        d["owners"] = keep_owners
        meta = d.setdefault("meta", {})
        meta.setdefault("stripped_cams", {})[cam] = {
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "C 方案重跑 detections 后 track_id 重编号，旧 tid2g 会错绑",
            "n_tid2g": n_t2g,
            "n_owners": n_own,
        }
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    return {"path": str(path), "n_tid2g": n_t2g, "n_owners": n_own,
            "left_t2g": len(keep_t2g), "left_owners": len(keep_owners)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="305")
    ap.add_argument("--apply", action="store_true", help="真正写入（默认只预览）")
    a = ap.parse_args()

    print(f"[{'APPLY' if a.apply else 'DRY-RUN'}] 剔除机位 {a.cam}")
    for f in FILES:
        p = Path(f)
        if not p.exists():
            print(f"  skip  {f}（不存在）")
            continue
        r = strip(p, a.cam, a.apply)
        print(f"  {r['path'].split('/')[-1]:32s} -tid2g {r['n_tid2g']:3d} "
              f"-owners {r['n_owners']:3d} → 剩 {r['left_t2g']}/{r['left_owners']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
