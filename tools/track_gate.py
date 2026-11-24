#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测级门控（建议 2 落地）：用 YOLO 掩膜命中率给 DINO 轨迹打"是否真人"分。

只记录，不删除：
  - 每条轨迹的 8 个采样帧里，有多少帧能在框内找到 YOLO 人体掩膜
  - 命中率 >=0.5 → keep；0.2~0.5 → review；<0.2 → drop（幽灵轨迹）
  - 掩膜占画面比 p50 <0.15 → 追加 review 标记（远小目标/局部可见，掩膜本身不可靠）

数据来源全部复用已有产物，不重新推理：
  output_crosscam_local/1012_osnet_emb_maskfill/{seg}_emb_{a,b}.npz   (eids, q[:,1]=mask_frac)
  output_crosscam_local/1012_ab/{seg}_{a,b}.jsonl                     (轨迹真实帧跨度)

用法：
  python tools/track_gate.py --all
  python tools/track_gate.py --seg 305_test_544
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import session_selector_v1 as S  # noqa: E402

MASK_EMB = ROOT / "output_crosscam_local/1012_osnet_emb_maskfill"
MASK_CROPS = ROOT / "output_crosscam_local/1012_osnet_crops_mask"
AB_DIR = ROOT / "output_crosscam_local/1012_ab"
OUT = ROOT / "output_crosscam_local/gate"

KEEP, REVIEW, DROP = "keep", "review", "drop"


def load_hits(seg):
    """返回 {(ab, tid): {'n':k, 'hits':h, 'mf':[...]}}

    crops npz 的 q 列为 (iou, mask_frac, used)：
      used=1 表示该帧框内匹配到 YOLO 人体掩膜并通过最小像素门槛
      mask_frac = 掩膜面积 / 框面积（人体在框内的占比）
    """
    agg = defaultdict(lambda: {"n": 0, "hits": 0, "mf": []})
    p = MASK_CROPS / f"{seg}_crops_maskfill.npz"
    if not p.exists():
        return agg
    z = np.load(p, allow_pickle=False)
    eids, q = z["eids"], z["q"]
    for i, e in enumerate(eids.tolist()):
        e = str(e)
        try:
            a, t, _k = e.split("#")
        except ValueError:
            continue
        rec = agg[(a, int(t))]
        rec["n"] += 1
        rec["hits"] += 1 if float(q[i][2]) > 0.5 else 0
        rec["mf"].append(float(q[i][1]))
    return agg


def load_span(seg):
    """轨迹帧跨度与观测数，用于报告规模。"""
    span = {}
    for ab in ("a", "b"):
        p = AB_DIR / f"{seg}_{ab}.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for t in r.get("track_ids") or []:
                k = (ab, int(t))
                d = span.setdefault(k, {"first": r["frame"], "last": r["frame"], "obs": 0})
                d["first"] = min(d["first"], r["frame"])
                d["last"] = max(d["last"], r["frame"])
                d["obs"] += 1
    return span


def run_seg(seg, force=False):
    outp = OUT / f"{seg}_gate.json"
    if outp.exists() and not force:
        print(f"[skip] {seg}")
        return json.load(open(outp, encoding="utf-8"))

    hits = load_hits(seg)
    span = load_span(seg)
    if not hits:
        print(f"[warn] {seg} 无 maskfill 特征，跳过（先跑 encode_mask_reid.py）")
        return None

    rows = []
    for (ab, tid), rec in sorted(hits.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = rec["n"]
        hr = rec["hits"] / n if n else 0.0
        mf = float(np.median(rec["mf"])) if rec["mf"] else 0.0
        flags = []
        if hr >= 0.5:
            dec = KEEP
        elif hr >= 0.2:
            dec = REVIEW
            flags.append("低命中率")
        else:
            dec = DROP
            flags.append("疑似幽灵轨迹")
        if mf < 0.15:
            flags.append("掩膜过小")
            if dec == KEEP:
                dec = REVIEW
        sp = span.get((ab, tid), {})
        rows.append({
            "seg": seg, "ab": ab, "track": tid,
            "n": n, "hits": rec["hits"], "hit_rate": round(hr, 4),
            "mask_frac_p50": round(mf, 4),
            "obs": sp.get("obs"), "first": sp.get("first"), "last": sp.get("last"),
            "decision": dec, "flags": flags,
        })

    cnt = defaultdict(int)
    for r in rows:
        cnt[r["decision"]] += 1
    summary = {
        "seg": seg, "tracks": len(rows),
        "keep": cnt[KEEP], "review": cnt[REVIEW], "drop": cnt[DROP],
        "drop_ratio": round(cnt[DROP] / max(1, len(rows)), 4),
        "kept_hit_rate_mean": round(float(np.mean([r["hit_rate"] for r in rows
                                                   if r["decision"] != DROP] or [0])), 4),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "tracks": rows}, f, ensure_ascii=False, indent=1)
    print(f"[{seg}] tracks={len(rows)} keep={summary['keep']} review={summary['review']} "
          f"drop={summary['drop']} ({summary['drop_ratio']*100:.1f}%)", flush=True)
    return {"summary": summary, "tracks": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    segs = [a.seg] if a.seg else (S.TRAIN_SEGS + S.TEST_SEGS)
    if not a.seg and not a.all:
        raise SystemExit("需要 --seg 或 --all")

    res = [run_seg(s, a.force) for s in segs]
    res = [r for r in res if r]
    if len(res) > 1:
        tot = sum(r["summary"]["tracks"] for r in res)
        drop = sum(r["summary"]["drop"] for r in res)
        rev = sum(r["summary"]["review"] for r in res)
        print(f"\n[合计] 轨迹 {tot}  drop={drop} ({drop/tot*100:.1f}%)  review={rev} "
              f"keep={tot-drop-rev}")
        with open(OUT / "gate_summary.json", "w", encoding="utf-8") as f:
            json.dump([r["summary"] for r in res], f, ensure_ascii=False, indent=1)
        print(f"[out] {OUT/'gate_summary.json'}")


if __name__ == "__main__":
    main()
