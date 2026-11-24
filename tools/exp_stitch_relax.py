#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验：放宽同机位拼接的「位置外推」门槛，能否把"两人重合导致的换号"接回来。

背景（2026-11-09 排查）：
  用户报「两人一重合就丢身份 id」。逐层查到主因不在检测，而在**拼接层的判据组合**：
    * 两人靠近时 DINO 会吐出「把两人一起框住」的大框（305 t40：h 451 → 623 → 862）；
    * ByteTrack 的 minimum_matching_threshold=0.8 因此匹配失败 → track 断；
    * 拼接层外观 cos 很高（0.91~0.98），但 `pe_adj`（位置外推误差）0.093~0.160 超过 0.08 → 拒。
  → 三层各自"合理"，合起来就是"同一个人被拆成两个实体"。

本脚本只读 stitch CSV（已含所有时间门候选的 cos/pe_adj/gap），离线重做合并决策，
对比不同门槛下的链接数/实体数，并给出新增链接的画像。**不改生产、不写任何产物**（除 --json-out）。

用法：
  python tools/exp_stitch_relax.py --root output_crosscam/long35 --cams 305,323 --json-out /tmp/exp.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_tracks(root: Path, cam: str, suffix: str) -> dict[int, list[tuple[int, list[float]]]]:
    tr: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    p = root / f"{cam}{suffix}"
    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for t, b in zip(d.get("track_ids") or [], d.get("boxes") or []):
                tr[int(t)].append((int(d["frame"]), b))
    for t in tr:
        tr[t].sort(key=lambda x: x[0])
    return tr


def cooccur(tr, ta: int, tb: int) -> int:
    """两个 track 在同一帧同时出现的次数（>0 则必为不同的人）。"""
    fa = {f for f, _ in tr.get(ta, [])}
    fb = {f for f, _ in tr.get(tb, [])}
    return len(fa & fb)


def union_find_entities(n: int, links: list[tuple[int, int]]) -> list[int]:
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    return [find(i) for i in range(n)]


def mutual_best(rows: list[dict], score_fn) -> list[dict]:
    """与 track_stitching.py 同构：按 score 降序，取 a 的唯一后继 / b 的唯一前驱。"""
    best_succ, best_pred = {}, {}
    for r in sorted(rows, key=score_fn, reverse=True):
        best_succ.setdefault(r["tid_a"], r)
        best_pred.setdefault(r["tid_b"], r)
    return [r for r in rows if best_succ.get(r["tid_a"]) is r and best_pred.get(r["tid_b"]) is r]


def evaluate(name: str, rows: list[dict], tids: list[int], tr, use_covis: bool) -> dict:
    # 共现硬约束：同一帧同时出现过 → 必为两人，禁止连边
    kept, dropped_covis = [], 0
    for r in rows:
        if use_covis and cooccur(tr, r["tid_a"], r["tid_b"]) > 0:
            dropped_covis += 1
            continue
        kept.append(r)
    links = mutual_best(kept, lambda r: r["cos"] - 2.0 * r["pe_adj"])
    pairs = [(r["tid_a"], r["tid_b"]) for r in links]
    idx = {t: i for i, t in enumerate(tids)}
    comp = union_find_entities(len(tids), [(idx[a], idx[b]) for a, b in pairs])
    n_ent = len(set(comp))
    cosv = np.array([r["cos"] for r in links]) if links else np.zeros(0)
    return {
        "name": name, "cands": len(rows), "after_covis": len(kept),
        "dropped_by_covis": dropped_covis,
        "links": len(links), "entities": n_ent,
        "cos_p10": round(float(np.percentile(cosv, 10)), 3) if len(cosv) else None,
        "cos_p50": round(float(np.percentile(cosv, 50)), 3) if len(cosv) else None,
        "pairs": pairs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output_crosscam/long35")
    ap.add_argument("--cams", default="305,323,318")
    ap.add_argument("--stitch-dir", default="stitch_long35")
    ap.add_argument("--det-suffix", default="_long35_detections.jsonl")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    root = Path(a.root)
    cams = [c.strip() for c in a.cams.split(",") if c.strip()]

    # 现状 / 两组候选方案
    def base_ok(r):
        return r["cos"] >= 0.80 and r["pe_adj"] <= 0.08

    def short_gap_ok(r, pe_max=0.25, gap_max=20, cos_min=0.90, sc_lo=0.80, sc_hi=1.25):
        """短 gap（人不可能走远）时，pe 大只可能是框的问题（合并框/遮挡）→ 放宽 pe，
        同时用 scale 兜住「一大一小两个不同的人」。"""
        if r["gap"] > gap_max or r["cos"] < cos_min or r["pe_adj"] > pe_max:
            return False
        sc = r.get("scale")
        if sc is None:
            return True
        return sc_lo <= sc <= sc_hi

    schemes = [
        ("S0 现状（无共现检查）", lambda r: base_ok(r), False),
        ("S1 现状+共现检查", lambda r: base_ok(r), True),
        ("S2 +短gap放宽 pe.25/gap20/scale.8-1.25",
         lambda r: base_ok(r) or short_gap_ok(r), True),
        ("S3 +短gap放宽 pe.30/gap30/scale.75-1.3",
         lambda r: base_ok(r) or short_gap_ok(r, 0.30, 30, 0.88, 0.75, 1.30), True),
        ("S4 +短gap放宽(仅外观.93)",
         lambda r: base_ok(r) or short_gap_ok(r, 0.25, 20, 0.93), True),
    ]

    allout = {}
    for cam in cams:
        csvp = root / a.stitch_dir / f"stitch_{cam}.csv"
        if not csvp.exists():
            print(f"{cam}: 缺 {csvp}")
            continue
        rows = []
        with csvp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("cos"):
                    continue
                rows.append({"tid_a": int(r["tid_a"]), "tid_b": int(r["tid_b"]),
                             "gap": int(r["gap_frames"]),
                             "cos": float(r["cos"]), "pe_adj": float(r["pe_adj"]),
                             "dist_raw": float(r["dist_raw"]),
                             "scale": float(r["scale"]) if r.get("scale") else None})
        tr = load_tracks(root, cam, a.det_suffix)
        tids = sorted({r["tid_a"] for r in rows} | {r["tid_b"] for r in rows})
        print(f"\n== {cam}: {len(rows)} 时间门候选（有 cos 的）, {len(tids)} 轨")
        base_pairs = None
        res = []
        for name, pred, use_covis in schemes:
            sel = [r for r in rows if pred(r)]
            e = evaluate(name, sel, tids, tr, use_covis)
            if base_pairs is None:
                base_pairs = set(e["pairs"])
                e["new_vs_base"] = 0
            else:
                e["new_vs_base"] = len(set(e["pairs"]) - base_pairs)
            res.append(e)
            print(f"  {name:28s} 候选{len(sel):5d} 去共现{e['dropped_by_covis']:4d} "
                  f"链接{e['links']:4d} 实体{e['entities']:4d} "
                  f"cos_p10 {e['cos_p10']} 新增{e['new_vs_base']:4d}")
        allout[cam] = res

        # 新增链接画像
        base = {tuple(p) for p in res[0]["pairs"]}
        for e in res[1:]:
            new = [p for p in e["pairs"] if tuple(p) not in base]
            if not new:
                continue
            det = [r for r in rows if (r["tid_a"], r["tid_b"]) in set(new)]
            if det:
                cos = np.array([d["cos"] for d in det])
                pe = np.array([d["pe_adj"] for d in det])
                print(f"    ↳ {e['name']} 新增 {len(new)} 条：cos p10={np.percentile(cos,10):.3f} "
                      f"p50={np.percentile(cos,50):.3f} | pe p50={np.percentile(pe,50):.3f} "
                      f"max={pe.max():.3f}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(allout, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        print(f"\n-> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
