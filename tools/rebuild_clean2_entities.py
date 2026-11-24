#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 clean2 口径上重建实体图（net_tracks），闭环版本链。

背景（外部评审缺陷 / 口径分裂）：
  原 net_tracks 的 track 成员取自 analysis/track_cleaning/{cam}_cleaned.jsonl。
  但 316/323 是 "dualclean" 机位——生产/评测口径 clean(v1)/clean2 = cleanset(_v2)
  = **掩膜 + flags** 后的结果（316:68 / 323:138 条轨迹），
  而 *_cleaned.jsonl 是**无掩膜**版（316:127 / 323:205 条）。
  两者不一致 → 现有 net_tracks 中
    316: 46/106 个实体、323: 42/139 个实体含 clean 中不存在的轨迹（合计 88 悬空实体）。

本脚本（不改动原 stream/stitch 目录，产出到新目录，可回退）：
  1) 读 clean2 各路 track 全集与帧跨度；
  2) 读 stitch_{cam}.csv 候选对，**只在 clean2 轨迹集合上**重算 mutual-best 链接
     （门限与原 track_stitching 一致：cos>=0.80, pe_adj<=0.08；禁传递闭包）；
  3) 318/320 应用原人工修正（黑名单撤销 / 白衣人合并 / 非人剔除 / 18-23 合并）；
  4) 以 clean2 轨迹全集建 Union-Find → 实体（被恢复的 FP 轨自动成为孤立实体）；
  5) 输出 net_tracks_{cam}.json 到新目录，并写 manifest 记录输入哈希（版本链）。

用法:
  python tools/rebuild_clean2_entities.py [--view clean2] [--out <dir>]
默认: view=clean2, out=output_crosscam/analysis/stitch_clean2/
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output_crosscam"
STITCH = SRC / "analysis" / "stitch"
DEFAULT_OUT = SRC / "analysis" / "stitch_clean2"
CAMS = ["305", "316", "318", "320", "323", "343"]
RES = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}

# 与 track_stitching.py 一致的链接门限
LINK_COS = 0.80
LINK_PE = 0.08

# 原人工修正（apply_p2_corrections.py）
BLACKLIST_318 = [(36, 39), (46, 59), (57, 62)]
WHITE_318 = [3, 42, 49, 50, 58, 70, 72, 74, 77, 81, 83, 85, 86, 91,
             102, 111, 123, 134, 144, 148, 149]
NONPERSON_320 = [8, 9, 13, 16, 17]
FORCE_320 = [(18, 23)]


def md5(p, bs=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def view_path(cam, view):
    if view == "clean2":
        return SRC / "cleanset_v2" / f"{cam}_detections.jsonl"
    if view == "clean":
        return SRC / "cleanset" / f"{cam}_detections.jsonl"
    return SRC / f"{cam}_detections.jsonl"


def load_spans(p, want_frames=False):
    """track_id -> [f0, f1, n]（逐框契约口径）；want_frames 时追加第 4 项 = 帧集合。

    帧集合用于**同帧共现硬约束**：同一机位同一帧不可能出现同一个人两次，
    所以两个 track 只要有任意一帧同时在场，就绝不允许连成同一实体。
    """
    spans = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            fr = r["frame"]
            for i in range(len(r.get("boxes", []) or [])):
                if i < len(tids) and tids[i] is not None and int(tids[i]) >= 0:
                    tid = int(tids[i])
                    s = spans.get(tid)
                    if s is None:
                        spans[tid] = [fr, fr, 1, {fr} if want_frames else None]
                    else:
                        s[0] = min(s[0], fr)
                        s[1] = max(s[1], fr)
                        s[2] += 1
                        if want_frames:
                            s[3].add(fr)
    return spans


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def clean2_links(cam, valid_tids, frames=None, relax=None):
    """在 clean2 轨迹集合上重算 mutual-best 链接（读 stitch CSV 缓存分数）。

    两条判据（2026-11-09 排查「两人重合丢 id」后加）：

    * **同帧共现硬约束**（frames 非空时生效）：两个 track 只要有一帧同时在场，
      就必是两个人 → 直接排除。这是物理约束，不是阈值，纯收益。
      （实测 323 有 11 条现存边违反它，属真误连。）
    * **短 gap 放宽**（relax 非空时生效）：人不可能在 0.8s 内走远，此时
      `pe_adj` 大只可能是"合并框/遮挡让末位置偏了"，不是同一人判据失效；
      故对角 `cos` 高、`gap` 短、`scale`（框高比）稳的对，放宽 pe 上限。
      解决 305 t40→t46 这类：cos 0.912 很强但 pe 0.160 被 0.08 拒。
    """
    cp = STITCH / f"stitch_{cam}.csv"
    cands = []
    n_covis = 0
    n_relax = 0
    with open(cp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b = int(r["tid_a"]), int(r["tid_b"])
            if a not in valid_tids or b not in valid_tids:
                continue
            try:
                cos, pe = float(r["cos"]), float(r["pe_adj"])
            except (TypeError, ValueError):
                continue
            if frames is not None:
                fa, fb = frames.get(a), frames.get(b)
                if fa and fb and (fa & fb):
                    n_covis += 1
                    continue
            ok = cos >= LINK_COS and pe <= LINK_PE
            if not ok and relax:
                gap_max, cos_min, pe_max, sc_lo, sc_hi = relax
                try:
                    gap = int(r["gap_frames"])
                    sc = float(r["scale"]) if r.get("scale") else None
                except (TypeError, ValueError, KeyError):
                    gap, sc = 10 ** 9, None
                if (gap <= gap_max and cos >= cos_min and pe <= pe_max
                        and (sc is None or sc_lo <= sc <= sc_hi)):
                    ok = True
                    n_relax += 1
            if not ok:
                continue
            cands.append((a, b, cos, pe, cos - 2 * pe))
    cands.sort(key=lambda x: x[4], reverse=True)
    best_succ, best_pred = {}, {}
    for a, b, cos, pe, sc in cands:
        best_succ.setdefault(a, (a, b, cos, pe))
        best_pred.setdefault(b, (a, b, cos, pe))
    links = []
    for a, r in best_succ.items():
        b = r[1]
        if b in best_pred and best_pred[b][0] == a:
            links.append(r)
    return links, len(cands), {"covis_dropped": n_covis, "relax_added": n_relax}


def build_entities(links, forced_groups, all_tracks):
    uf = UnionFind()
    for t in all_tracks:
        uf.find(t)
    for a, b, *_ in links:
        uf.union(a, b)
    for grp in forced_groups:
        for x in grp[1:]:
            uf.union(grp[0], x)
    groups = defaultdict(list)
    for t in all_tracks:
        groups[uf.find(t)].append(t)
    return sorted(groups.values(), key=lambda g: min(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="clean2", choices=["raw", "clean", "clean2"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--src", default="", help="数据根（默认 output_crosscam）；换片段时指向新根")
    ap.add_argument("--stitch", default="", help="stitch CSV 目录（默认 analysis/stitch）")
    ap.add_argument("--cams", default="", help="逗号分隔机位；默认 6 路全集")
    ap.add_argument("--no-covis-guard", dest="covis_guard", action="store_false",
                    default=True,
                    help="关闭「同帧共现」硬约束（默认开：同一帧同时在场必是两人，禁止连边）")
    ap.add_argument("--relax-gap", type=int, default=0,
                    help="短 gap 放宽：gap ≤ 该帧数的候选可走放宽判据（0=关闭，推荐 20）")
    ap.add_argument("--relax-cos", type=float, default=0.90)
    ap.add_argument("--relax-pe", type=float, default=0.25)
    ap.add_argument("--relax-scale", nargs=2, type=float, default=[0.80, 1.25],
                    help="放宽判据下的框高比允许区间（同一人不应突变）")
    args = ap.parse_args()
    # 重定向数据根（long35 等其它片段复用同一算法，不改口径）
    global SRC, STITCH, CAMS
    if args.src:
        SRC = Path(args.src).resolve()
    if args.stitch:
        STITCH = Path(args.stitch).resolve()
    if args.cams:
        CAMS = [c.strip() for c in args.cams.split(",") if c.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    relax = ((args.relax_gap, args.relax_cos, args.relax_pe,
              args.relax_scale[0], args.relax_scale[1]) if args.relax_gap > 0 else None)

    manifest = {
        "version": "net_tracks_clean2_v1",
        "created": "2026-10-14",
        "view": args.view,
        "basis": "在 clean2 口径轨迹集合上重建（修口径分裂）；stitch CSV 分数复用，"
                 "仅重算 mutual-best，未重跑嵌入",
        "link_criteria": {"cos": LINK_COS, "pe_adj": LINK_PE,
                          "rule": "mutual_best, no transitive closure, 两端必须 ∈ clean2",
                          "covis_guard": bool(args.covis_guard),
                          "relax": (None if not relax else
                                    {"gap_max": relax[0], "cos_min": relax[1],
                                     "pe_max": relax[2],
                                     "scale": list(args.relax_scale)})},
        "cameras": {},
    }
    total_ent = 0
    print(f"{'cam':>4} {'tracks':>7} {'cand':>6} {'links':>6} {'entities':>9} "
          f"{'covis_drop':>10} {'relax_add':>9}")
    for cam in CAMS:
        vp = view_path(cam, args.view)
        spans = load_spans(vp, want_frames=args.covis_guard)
        valid = set(spans)
        frames = ({t: spans[t][3] for t in spans} if args.covis_guard else None)
        links, n_cand, diag = clean2_links(cam, valid, frames, relax)
        forced = []
        blacklist = []
        if cam == "318":
            forced = [WHITE_318]
            blacklist = BLACKLIST_318
        elif cam == "320":
            # 剔除非人轨迹
            valid = {t for t in valid if t not in NONPERSON_320}
            spans = {t: spans[t] for t in valid}
            forced = [list(x) for x in FORCE_320]
        links = [l for l in links if (l[0], l[1]) not in blacklist]

        ents = build_entities(links, forced, sorted(valid))
        ents_out = []
        for i, members in enumerate(ents, 1):
            f0 = min(spans[t][0] for t in members)
            f1 = max(spans[t][1] for t in members)
            note = ""
            if cam == "318" and sorted(members) == sorted(WHITE_318):
                note = "白衣人(工位) 21 段合并——用户确认 q1"
            if cam == "320" and sorted(members) == [18, 23]:
                note = "皮领女士：跨机位确认(318#120:18 与 120:23)——合并"
            ents_out.append({
                "eid": f"{cam}-E{i:03d}", "tracks": sorted(members),
                "f0": f0, "f1": f1, "n_frag": len(members), "note": note,
            })
        total_ent += len(ents_out)
        net = {
            "cam": cam, "source": f"rebuild_clean2_entities.py (view={args.view})",
            "n_tracks": len(valid), "n_entities": len(ents_out),
            "non_person": NONPERSON_320 if cam == "320" else [],
            "entities": ents_out,
        }
        (out_dir / f"net_tracks_{cam}.json").write_text(
            json.dumps(net, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["cameras"][cam] = {
            "n_tracks": len(valid), "n_candidates": n_cand,
            "n_links": len(links), "n_entities": len(ents_out),
            "covis_dropped": diag["covis_dropped"],
            "relax_added": diag["relax_added"],
            "md5_source": md5(vp),
            "md5_out": md5(out_dir / f"net_tracks_{cam}.json"),
        }
        print(f"{cam:>4} {len(valid):>7} {n_cand:>6} {len(links):>6} {len(ents_out):>9} "
              f"{diag['covis_dropped']:>10} {diag['relax_added']:>9}")
    manifest["totals"] = {"n_entities": total_ent}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总实体: {total_ent}")
    print(f"输出: {out_dir}/net_tracks_*.json + manifest.json")


if __name__ == "__main__":
    main()
