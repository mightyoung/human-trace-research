# -*- coding: utf-8 -*-
"""实验：把「被包含框交替」的碎轨并入主轨（detections 层后处理，不重跑检测）。

问题
----
人坐着时，检测器常同时给出「全身框」和「上半身框」候选。NMS 的包含度
（inter / min(area)）对二者是 1.0 ≥ 0.85，于是**每帧只留全身框**，上半身框被删。
但某些帧全身框漏检/低分，上半身框就露出来，ByteTrack 因 IoU 只有 ~0.24 接不上，
于是生成一条新 track。表现为同一个人标签在 #129 / #133 / #136 之间跳。

判据（四条同时满足，宁可漏合并不误合并）
----
1. 两轨**从不共现**（交集帧数 = 0）—— 共现即两人，直接排除；这也是 NMS 删框的必然结果
2. 时间区间重叠或间隔 ≤ `--max-gap` 帧
3. 短轨每个框被长轨**最近邻帧**的框包含，包含度中位 ≥ `--cont`（默认 0.85）
4. 短轨观测数 < 长轨（并入方向明确）

用法
----
    python tools/exp_contained_merge.py <detections.jsonl> [--apply-out <out.jsonl>]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    """inner 落在 outer 内的面积比（与 containment_nms 的对称口径不同：
    这里只问「短轨的框是否被长轨的框包住」，方向固定）。"""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / a if a > 0 else 0.0


def load(path: Path):
    """返回 frames[frame] = [(tid, box)] 与 tracks[tid] = (frames array, boxes array)"""
    per_frame: dict[int, list] = defaultdict(list)
    seq: dict[int, list] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            fr = int(d["frame"])
            for t, b in zip(d.get("track_ids") or [], d.get("boxes") or []):
                per_frame[fr].append((int(t), b))
                seq[int(t)].append((fr, b))
    tracks = {}
    for t, s in seq.items():
        s.sort()
        tracks[t] = (np.array([x[0] for x in s]), np.array([x[1] for x in s], dtype=np.float64))
    return per_frame, tracks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("det")
    ap.add_argument("--cont", type=float, default=0.85, help="包含度中位阈值")
    ap.add_argument("--ratio", type=float, default=0.85, help="逐帧包含度达标的比例下限")
    ap.add_argument("--max-gap", type=int, default=25, help="时间间隔上限（帧，25=1s@25fps/stride1）")
    ap.add_argument("--min-obs", type=int, default=3, help="短轨最少观测数（更短的视为噪声不并）")
    ap.add_argument("--apply-out", default="", help="写出合并后的 detections")
    ap.add_argument("--emb", default="", help="ReID 向量 npz（v{cam}_{tid}）；给了才做外观校验")
    ap.add_argument("--cos", type=float, default=0.65, help="合并所需外观 cos 下限")
    a = ap.parse_args()

    det = Path(a.det)
    per_frame, tracks = load(det)
    tids = sorted(tracks)
    print(f"[load] {det.name}: {len(per_frame)} 帧 / {len(tids)} 轨")

    # 每条轨的共现伙伴（同帧出现过的 tid 集合），用于快速排除
    covis: dict[int, set] = defaultdict(set)
    for fr, lst in per_frame.items():
        if len(lst) < 2:
            continue
        ids = [t for t, _ in lst]
        for i in ids:
            covis[i].update(x for x in ids if x != i)

    cands = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            ta, tb = tids[i], tids[j]
            fa, ba = tracks[ta]
            fb, bb = tracks[tb]
            # 1) 从不共现
            if tb in covis[ta]:
                continue
            # 2) 时间重叠或邻近
            if fa[-1] < fb[0] - a.max_gap or fb[-1] < fa[0] - a.max_gap:
                continue
            # 3) 判断包含方向：不是「短轨并入长轨」，而是「被包含者并入包含者」。
            #    #133(n=20) 是全身框、#129(n=56) 是上半身框 —— 短的那条反而框大，
            #    所以方向必须由几何决定，观测数只用来挑合并后保留谁。
            def med_contain(inner_f, inner_b, outer_f, outer_b):
                idx = np.searchsorted(outer_f, inner_f)
                idx = np.clip(idx, 0, len(outer_f) - 1)
                prev = np.clip(idx - 1, 0, len(outer_f) - 1)
                pick = np.where(np.abs(outer_f[idx] - inner_f) <= np.abs(outer_f[prev] - inner_f), idx, prev)
                cs = np.array([containment(inner_b[k], outer_b[pick[k]]) for k in range(len(inner_f))])
                return float(np.median(cs)), float((cs >= a.cont).mean())

            if len(fa) < a.min_obs or len(fb) < a.min_obs:
                continue
            m_ab, h_ab = med_contain(fb, bb, fa, ba)   # b 在 a 内？
            m_ba, h_ba = med_contain(fa, ba, fb, bb)   # a 在 b 内？
            if m_ab >= a.cont and h_ab >= a.ratio:
                inner, outer, med, hit = tb, ta, m_ab, h_ab
            elif m_ba >= a.cont and h_ba >= a.ratio:
                inner, outer, med, hit = ta, tb, m_ba, h_ba
            else:
                continue
            # 合并后保留观测更多的那条 tid（编号稳定；几何方向只决定谁并谁）
            keep, drop = (outer, inner) if len(tracks[outer][0]) >= len(tracks[inner][0]) else (inner, outer)
            cands.append((drop, keep, len(tracks[drop][0]), len(tracks[keep][0]), med, hit))

    print(f"[cand] 满足「不共现 + 被包含」的短轨->长轨: {len(cands)} 对")

    # 外观校验：纯几何无法区分「同一人的上半身框」与「被前面的人完全挡住的路人」，
    # 两者都是「从不共现 + 完全包含」。用 ReID 向量做第二道门。
    emb = {}
    if a.emb:
        z = np.load(a.emb, allow_pickle=True)
        cam = det.name.split("_")[0]
        for k in z.files:
            if k.startswith(f"v{cam}_"):
                emb[int(k.split("_")[1])] = z[k]
        print(f"[emb] {len(emb)} 条轨有向量（{a.emb}）")

    merged = {}
    passed = 0
    rejected = []
    for st, lt, ns, nl, med, hit in sorted(cands, key=lambda x: -x[4]):
        cs = None
        if emb:
            ea, eb = emb.get(st), emb.get(lt)
            if ea is None or eb is None:
                rejected.append((st, lt, None))
                continue
            cs = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))
            if cs < a.cos:
                rejected.append((st, lt, cs))
                continue
        passed += 1
        print(f"  #{st}(n={ns}) -> #{lt}(n={nl})  包含度中位={med:.3f} 达标率={hit:.2f}"
              + (f"  cos={cs:.3f}" if cs is not None else ""))
        merged.setdefault(st, lt)

    if rejected:
        print(f"[reject] 外观不过关 {len(rejected)} 对：")
        for st, lt, cs in rejected[:15]:
            print(f"  #{st} -> #{lt}  cos={'无向量' if cs is None else f'{cs:.3f}'}")

    # 链式收敛（A->B, B->C 时应归到 C；并检查环）
    def root(t):
        seen = set()
        while t in merged and t not in seen:
            seen.add(t)
            t = merged[t]
        return t

    final = {s: root(s) for s in merged}
    final = {s: r for s, r in final.items() if r != s}
    print(f"[result] 实际合并 {len(final)} 条轨 -> 由 {len(tids)} 轨变为 {len(tids) - len(final)} 轨")
    print(f"[result] 涉及观测数 {sum(len(tracks[s][0]) for s in final)}")

    if a.apply_out:
        out = Path(a.apply_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with det.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
            for ln in f:
                d = json.loads(ln)
                d["track_ids"] = [final.get(int(t), int(t)) for t in (d.get("track_ids") or [])]
                g.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[write] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
