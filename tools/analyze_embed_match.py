#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""嵌入对账的深度分析：重叠度分桶 + 低cos配对审计。

用法:
  python tools/analyze_embed_match.py

输入:
  output_crosscam_local/reconcile_embed_all.json（由 local_check.py embed 产出，
  含 matched_all 全量配对）
  output_crosscam_local/detect/*_detections.jsonl（本地 track 帧集合）
  output_crosscam/*_detections.jsonl（远程 track 帧集合）
  output_crosscam_local/embed/*.npz / output_crosscam/analysis/relay/remote_emb/*.npz

输出（stdout）:
  1) cos × overlap 分桶表（6 路合计 + 逐路）——分离"track 切分差异"与"链路差异"
  2) 低cos配对审计——cos<0.9 的配对中，是否存在替代高cos配对
     （存在 → 匹配器贪心错配/碎片化；不存在 → 深查样本特征）

结论口径（2026-10-17 实测）:
  overlap≥0.95 的配对 cos≈0.995（≈同内容）→ 链路本身差异在千分位；
  低cos尾部集中在低重叠/短轨迹区与贪心错配，属对账方法学副产品。
"""
import json
import re
from pathlib import Path

import numpy as np

BASE = Path("output_crosscam_local")
REM = Path("output_crosscam")
CAMS = ["305", "316", "318", "320", "323", "343"]


def kid(k: str) -> int:
    m = re.search(r"(\d+)$", k)
    return int(m.group(1)) if m else -1


def load_tracks(p: Path) -> dict:
    t: dict = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for tid in r.get("track_ids", []):
                t.setdefault(int(tid), set()).add(r["frame"])
    return t


def main() -> None:
    emb = json.load(open(BASE / "reconcile_embed_all.json", encoding="utf-8"))

    # ---- 1) 分桶 ----
    buckets = [(0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
    allpairs = []
    print("== cos × overlap 分桶 ==")
    for cam in CAMS:
        pairs = emb.get(cam, {}).get("matched_all", [])
        allpairs.extend(pairs)
    coss = np.array([p["cos"] for p in allpairs])
    ovs = np.array([p.get("overlap", 0) for p in allpairs])
    print(f"6 路合计 matched={len(coss)}  cos mean={coss.mean():.4f}  "
          f"median={np.median(coss):.4f}  ≥0.95={np.mean(coss >= 0.95):.1%}  "
          f"≥0.99={np.mean(coss >= 0.99):.1%}")
    for a, b in buckets:
        sel = coss[(ovs >= a) & (ovs < b)]
        if len(sel):
            print(f"  overlap [{a:.2f},{b:.2f}): n={len(sel):>3}  "
                  f"cos mean={sel.mean():.4f}  ≥0.95={np.mean(sel >= 0.95):.0%}  "
                  f"≥0.99={np.mean(sel >= 0.99):.0%}")

    # ---- 2) 低cos审计 ----
    print("\n== 低cos配对审计（cos<0.9 是否存在替代高cos配对）==")
    tot = low = low_with_alt = 0
    no_alt_rows = []
    for cam in CAMS:
        tl = load_tracks(BASE / "detect" / f"{cam}_detections.jsonl")
        tr = load_tracks(REM / f"{cam}_detections.jsonl")
        loc = np.load(BASE / "embed" / f"reid_emb_{cam}_d3b_local.npz")
        rem = np.load(REM / "analysis" / "relay" / "remote_emb" / f"reid_emb_{cam}_d3b.npz")
        for m in emb.get(cam, {}).get("matched_all", []):
            tot += 1
            if m["cos"] >= 0.9:
                continue
            low += 1
            fl = tl[kid(m["local"])]
            le = loc[m["local"]]
            has_alt = False
            for rkey in rem.files:
                if rkey == m["remote"]:
                    continue
                fr = tr.get(kid(rkey), set())
                if not fr:
                    continue
                ov = len(fl & fr) / len(fl | fr)
                if ov < 0.15:
                    continue
                if float(np.dot(le, rem[rkey])) >= 0.95:
                    has_alt = True
                    break
            if has_alt:
                low_with_alt += 1
            else:
                no_alt_rows.append(m)
    print(f"总配对 {tot}; cos<0.9 共 {low} 对")
    print(f"  存在替代高cos(≥0.95)配对: {low_with_alt} 对 → 匹配器贪心错配/碎片化")
    print(f"  无替代: {low - low_with_alt} 对 → 深查（通常为低重叠/短轨迹）")
    for m in sorted(no_alt_rows, key=lambda x: x["cos"])[:10]:
        print(f"    {m['local']:>12} vs {m['remote']:<10} cos={m['cos']:.3f} ov={m.get('overlap', 0):.2f}")


if __name__ == "__main__":
    main()
