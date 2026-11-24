#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 C 配置（hybrid + min-h 200）从原始框生成 6 路 detections。

为什么要单独一步，而不是直接改 `infer_supervision.py` 重跑检测：
  原始框（NMS 前）已经全量落盘，离线重算 NMS+ByteTrack 只需几秒/路，
  重跑 GPU 检测要几小时。两条路产出应当一致（`check_nms_equiv.py` 已验证逐帧等价）。

输出：`output_crosscam/long35/{cam}_long35_detections_c.jsonl`
      —— 与生产 jsonl 同构，额外补 `labels`（全 "person"）和 `latency_ms: null`，
      避免下游按字段解析时 KeyError。

用法:
  python tools/gen_detections_c.py                 # 6 路全跑（缺 raw 的跳过）
  python tools/gen_detections_c.py --cams 305
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "output_crosscam" / "long35"
RAW_NAMES = ["{cam}_raw10.jsonl", "{cam}_raw10.full.jsonl", "{cam}_long35_raw10.jsonl"]


def find_raw(cam: str) -> Path | None:
    for base in (ROOT / "probe_out", LONG / "raw"):
        for pat in RAW_NAMES:
            p = base / pat.format(cam=cam)
            if p.exists() and p.stat().st_size > 0:
                return p
    return None


def normalize(src: Path, dst: Path) -> tuple[int, int]:
    """补 labels / latency_ms 字段（下游可能按 schema 解析）。"""
    n = 0
    nb = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as g:
        for ln in f:
            d = json.loads(ln)
            nb += len(d.get("boxes") or [])
            d.setdefault("labels", ["person"] * len(d.get("boxes") or []))
            d.setdefault("latency_ms", None)
            # 字段顺序对齐生产：frame,count,deduped_count,raw_count,removed,track_ids,latency_ms,boxes,scores,labels
            out = {k: d.get(k) for k in ("frame", "count", "deduped_count", "raw_count",
                                         "removed", "track_ids", "latency_ms",
                                         "boxes", "scores", "labels")}
            g.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n, nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="305 316 318 320 323 343")
    ap.add_argument("--hy-score-k", default="0.8")
    ap.add_argument("--hy-min-h", default="200")
    ap.add_argument("--tracker", default="cont", choices=["std", "cont"],
                    help="cont = 包含匹配 + 轨迹融合（默认）；std = 上游 ByteTrack")
    ap.add_argument("--suffix", default="_long35_detections_c.jsonl",
                    help="输出文件名后缀（换 tracker 时用不同后缀便于对比/回退）")
    a = ap.parse_args()

    py = sys.executable
    ok, skip = [], []
    for cam in a.cams.split():
        raw = find_raw(cam)
        if raw is None:
            skip.append(cam)
            continue
        tmp = LONG / f"_tmp_{cam}_c.jsonl"
        rc = subprocess.call([py, str(ROOT / "tools" / "retrack_from_raw.py"),
                              "--raw", str(raw), "--out", str(tmp),
                              "--policy", "hybrid",
                              "--hy-score-k", a.hy_score_k,
                              "--hy-min-h", a.hy_min_h,
                              "--tracker", a.tracker])
        if rc != 0:
            print(f"[FAIL] {cam} retrack rc={rc}")
            skip.append(cam)
            continue
        dst = LONG / f"{cam}{a.suffix}"
        n, nb = normalize(tmp, dst)
        tmp.unlink(missing_ok=True)
        print(f"[ok] {cam} {n} 帧 / {nb} 框 -> {dst.name}")
        ok.append(cam)

    print(f"\n完成 {len(ok)} 路: {' '.join(ok)}")
    if skip:
        print(f"跳过（缺原始框）: {' '.join(skip)}")
    return 0 if not skip else 2


if __name__ == "__main__":
    raise SystemExit(main())
