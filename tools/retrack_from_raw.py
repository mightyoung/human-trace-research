#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C 方案后处理：从 **NMS 前原始框** 重跑 Containment-NMS + ByteTrack，验证治本策略。

为什么需要它
------------
`{cam}_detections.jsonl` 只留 NMS 后的框，救不回被删的框。`dino_raw_dump.py` 先把
原始框（阈值 0.10）落盘，本脚本再离线重算——**一次全量 dump，任意次策略实验**，
不需要反复占 GPU。

305 实测（probe，f4435 换号点）暴露的真因：
    #2  x[1101,1478] h=696  sc=0.414   ← 被人形尺度的框
    #7  x[1095,1477] h=862  sc=0.330   ← 更大的超尺框（含地面/倒影）
    包含度 1.00 → keep=larger 删 #2 留 #7 → 跟踪器看到 h=862（真人 460~520）
    → 中心/尺寸跳变 → 匹配失败 → 换号。
即：**不是"两人合并成一个框"，而是"低分超尺框吃掉了高分正常框"**。

策略（可组合，全部可配）
------------------------
  base    ：完全复刻生产（thr .30 / containment keep=larger / ByteTrack mm=.8）
  hybrid  ：包含判重时若大框分数 < 小框 × --hy-score-k，保留小框
  score   ：包含判重时一律保留高分框
  size-gate：帧间尺寸连续性门（人 0.2s 内框高不会翻倍，物理约束）——
             h > gate × 参考高 时 drop（丢）或 shrink（收缩到 gate×参考高）
  mm-thr  ：ByteTrack minimum_matching_threshold（stride=5 位移大，.8 偏严）

用法
----
  # 1) 对账：必须逐帧比上生产
  python tools/retrack_from_raw.py --raw raw/305_raw10.jsonl --policy base \
      --compare output_crosscam/long35/305_long35_detections.jsonl --out /tmp/305_base.jsonl
  # 2) 实验
  python tools/retrack_from_raw.py --raw raw/305_raw10.jsonl --policy hybrid \
      --hy-score-k 0.9 --size-gate 1.35 --size-act shrink --mm-thr 0.6 --out /tmp/305_c1.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- 几何
def containment_of(inner, outer) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih / a) if a > 0 else 0.0


def containment_sym(a, b) -> float:
    """「小框被大框包含」的比例（对称）——与生产 `containment_nms` 的一致口径。

    ★ 不能用 containment_of(当前框, 已保留框)：hybrid 会把大框顶掉、把小框放进
    kept，此后再来的框可能比 kept 里的大，非对称版就会去算「大框被小框包含」
    （恒为小值），判定与生产实现分叉（305 f23880 实测差 1 个框）。
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (inter / small) if small > 0 else 0.0


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b) -> tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


# ---------------------------------------------------------------- NMS 变体
def nms_select(boxes: np.ndarray, scores: np.ndarray, thr: float,
               policy: str, hy_k: float, hy_min_h: float = 0.0) -> tuple[list[int], list[list]]:
    """返回 (保留下标按面积降序, 被删记录[[idx, 吞它的 idx, 包含度]])。

    policy:
      larger — 生产口径：大框优先占位，被包含者丢弃
      score  — 高分优先占位
      hybrid — 大框先占位，但若大框分数 < 小框×hy_k，则改由小框占位。
               **hy_min_h > 0 时只对「小框高度 ≥ hy_min_h」的组生效**：
               远景小人（25~58px）的小框是头/上半身，顶掉全身框会让跟踪目标抖动；
               近景大人（300~800px）的大框常吃到桌椅/地面，让位给小框反而更稳。
    """
    n = len(boxes)
    if n <= 1:
        return list(range(n)), []

    # ★★ 单一实现原则：选择逻辑**直接调用生产用的 containment_nms**，本文件不再
    # 维护第二份。原因：两套实现即使数学等价，浮点运算顺序不同也会在包含度刚好
    # 0.850 的边界上分叉（305 f23880 实测：一套算 0.8499999 保留、一套算 0.8500001 丢弃）。
    # 只有共用同一份代码，才能保证「实验验证过的」就是「上线跑的」。
    from containment_nms import containment_nms  # 延迟导入，避免脚本被当库用时循环

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if policy in ("base", "larger"):
        kept_arr = containment_nms(boxes, scores, thr, "larger")
    elif policy == "score":
        kept_arr = containment_nms(boxes, scores, thr, "higher")
    elif policy == "hybrid":
        kept_arr = containment_nms(boxes, scores, thr, "larger",
                                   hy_score_k=hy_k, hy_min_h=hy_min_h)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    kept = [int(i) for i in kept_arr]

    # 诊断信息（被删的框 + 吞掉它的框 + 包含度），仅用于 audit，不参与选择
    ks = set(kept)
    supp: list[list] = []
    for i in range(n):
        if i in ks:
            continue
        best, bestc = -1, 0.0
        for j in kept:
            c = containment_sym(boxes[i], boxes[j])
            if c > bestc:
                best, bestc = j, c
        supp.append([int(i), int(best), round(float(bestc), 4)])
    return kept, supp


# ---------------------------------------------------------------- 尺寸门
def size_gate(boxes: list[list[float]], prev: list[list[float]], gate: float,
              act: str) -> tuple[list[list[float]], int]:
    """帧间尺寸连续性：按中心最近的上帧框估参考高，超 gate 倍则 drop / shrink。

    物理依据：stride=5 → 0.2s。人正常/快速行走框高变化 <20%，跑步也 <35%。
    框高突增 >gate× 基本只有两种来源：多尺度超尺框（含倒影/地面）或两人合并。
    """
    if gate <= 0 or not prev:
        return boxes, 0
    n_fix = 0
    out: list[list[float]] = []
    for b in boxes:
        cx, cy = center(b)
        h = b[3] - b[1]
        bw = b[2] - b[0]
        ref, bestd = None, 1e18
        for p in prev:
            px, py = center(p)
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < bestd and d < max(1.5 * bw, 120.0):
                ref, bestd = p, d
        if ref is None:
            out.append(b)
            continue
        rh = ref[3] - ref[1]
        if rh <= 1 or h <= gate * rh:
            out.append(b)
            continue
        n_fix += 1
        if act == "drop":
            continue
        # shrink：顶边跟随参考框顶边（人脚下位置通常一致或更低），高度压到 gate*rh
        nh = min(h, gate * rh)
        top = b[3] - nh
        out.append([b[0], top, b[2], b[3]])
    return out, n_fix


# ---------------------------------------------------------------- 统计
def track_stats(frames: list[tuple[int, list[int], list[list[float]]]], stride: int = 5) -> dict:
    tracks: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for fr, tids, boxes in frames:
        for t, b in zip(tids, boxes):
            if t is not None and t >= 0:
                tracks[t].append((fr, b))
    if not tracks:
        return {"tracks": 0, "breaks": 0, "switches": 0}
    f1 = max(f for seq in tracks.values() for f, _ in seq)
    n_break = 0
    for t, seq in tracks.items():
        seq.sort(key=lambda x: x[0])
        for i in range(1, len(seq)):
            if seq[i][0] - seq[i - 1][0] > stride and seq[i - 1][0] < f1 - 2 * stride:
                n_break += 1
                break
    per: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for t, seq in tracks.items():
        for f, b in seq:
            per[f].append((t, b))
    starts = {t: min(f for f, _ in seq) for t, seq in tracks.items()}

    def iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = area(a) + area(b) - inter
        return inter / ua if ua > 0 else 0.0

    n_sw = 0
    for t, seq in tracks.items():
        last_f, last_b = seq[-1]
        if last_f >= f1 - 2 * stride:
            continue
        for g in range(stride, stride * 41, stride):
            fr = last_f + g
            if fr not in per:
                continue
            hit = False
            for t2, b2 in per[fr]:
                if t2 == t or starts.get(t2, 0) <= last_f:
                    continue
                if iou(last_b, b2) >= 0.5:
                    n_sw += 1
                    hit = True
                    break
            if hit:
                break
    lens = sorted(len(s) for s in tracks.values())
    return {"tracks": len(tracks), "breaks": n_break, "switches": n_sw,
            "short_lt5": sum(1 for x in lens if x < 5),
            "len_p50": lens[len(lens) // 2] if lens else 0}


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dino_raw_dump 输出的原始框 jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="base",
                    choices=["base", "larger", "score", "hybrid"])
    ap.add_argument("--thr", type=float, default=0.30, help="检测分数阈值（生产 0.30）")
    ap.add_argument("--hy-score-k", type=float, default=0.9,
                    help="hybrid：大框分数 < 小框×k 时保留小框")
    ap.add_argument("--hy-min-h", type=float, default=0.0,
                    help="hybrid 只对「小框高度≥此值」生效（0=不限制）。"
                         "远景观测目标只有 25~58px，让头/上半身顶掉全身框会更抖；"
                         "近景 300~800px 时大框常吃到桌椅，让位反而更稳。建议 120~200")
    ap.add_argument("--containment", type=float, default=0.85)
    ap.add_argument("--size-gate", type=float, default=0.0, help="0=关；建议 1.3~1.5")
    ap.add_argument("--size-act", default="shrink", choices=["drop", "shrink"])
    ap.add_argument("--mm-thr", type=float, default=0.8)
    ap.add_argument("--tracker", default="std", choices=["std", "cont"],
                    help="std=上游 ByteTrack；cont=加「包含匹配」第二级")
    ap.add_argument("--cont-thresh", type=float, default=0.85, help="包含度下限")
    ap.add_argument("--cont-center", type=float, default=0.5, help="x 向中心距/框宽 上限")
    ap.add_argument("--cont-y1", type=float, default=0.15, help="头顶对齐容差/框高")
    ap.add_argument("--fuse-contained", action="store_true", default=True,
                    help="cont 跟踪后做轨迹级包含融合（默认开）")
    ap.add_argument("--no-fuse-contained", dest="fuse_contained", action="store_false")
    ap.add_argument("--fuse-min-obs", type=int, default=5, help="参与融合的最少观测数")
    ap.add_argument("--lost-buffer", type=int, default=90)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个处理帧（调试）")
    ap.add_argument("--compare", default="", help="与基准 detections.jsonl 对账")
    ap.add_argument("--dump-switch", default="", help="输出换号明细 json")
    a = ap.parse_args()

    import supervision as sv

    raw = []
    with Path(a.raw).open(encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            raw.append((int(d["frame"]), d["boxes"], d["scores"]))
    raw.sort(key=lambda x: x[0])
    if a.limit:
        raw = raw[:a.limit]
    print(f"[raw] {len(raw)} 帧  thr={a.thr} policy={a.policy}", flush=True)

    _tracker_kw = dict(
        track_activation_threshold=0.10,
        lost_track_buffer=a.lost_buffer,
        minimum_matching_threshold=a.mm_thr,
        frame_rate=a.fps,
        minimum_consecutive_frames=1,
    )
    if a.tracker == "cont":
        from containment_bytetrack import ContainmentByteTrack
        tracker = ContainmentByteTrack(cont_thresh=a.cont_thresh,
                                      center_ratio=a.cont_center, **_tracker_kw)
        print(f"[tracker] ContainmentByteTrack cont>={a.cont_thresh} "
              f"center<={a.cont_center}xW", flush=True)
    else:
        tracker = sv.ByteTrack(**_tracker_kw)

    out_frames: list[tuple[int, list[int], list[list[float]]]] = []
    n_raw = n_cand = n_kept = n_gate = 0
    prev: list[list[float]] = []

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fo:
        for fr, boxes, scores in raw:
            n_raw += len(boxes)
            b = [bb for bb, s in zip(boxes, scores) if s >= a.thr]
            sc = [s for s in scores if s >= a.thr]
            n_cand += len(b)
            if len(b) == 0:
                tracker.update_with_detections(sv.Detections.empty())
                out_frames.append((fr, [], []))
                fo.write(json.dumps({"frame": fr, "count": 0, "raw_count": len(boxes),
                                     "deduped_count": 0, "track_ids": [], "boxes": [],
                                     "scores": []}, ensure_ascii=False) + "\n")
                prev = []
                continue

            arr = np.asarray(b, dtype=np.float32)
            sca = np.asarray(sc, dtype=np.float32)
            if a.size_gate > 0:
                gated, nfix = size_gate([list(map(float, x)) for x in arr], prev,
                                        a.size_gate, a.size_act)
                if nfix:
                    n_gate += nfix
                    gated = np.asarray(gated, dtype=np.float32) if gated else np.zeros((0, 4), np.float32)
                    # 同步分数：按坐标匹配回原候选
                    keepmask = []
                    for g in gated:
                        hit = -1
                        for i, x in enumerate(arr):
                            if abs(g[0] - x[0]) < 0.6 and abs(g[2] - x[2]) < 0.6 \
                                    and abs(g[3] - x[3]) < 0.6:
                                hit = i
                                break
                        keepmask.append(hit)
                    arr = gated
                    sca = np.asarray([sc[i] for i in keepmask if i >= 0], dtype=np.float32)
                    if len(arr) == 0:
                        tracker.update_with_detections(sv.Detections.empty())
                        out_frames.append((fr, [], []))
                        fo.write(json.dumps({"frame": fr, "count": 0, "raw_count": len(boxes),
                                             "deduped_count": 0, "track_ids": [], "boxes": [],
                                             "scores": []}, ensure_ascii=False) + "\n")
                        prev = []
                        continue

            if a.policy == "base":
                kept, supp = nms_select(arr, sca, a.containment, "larger", 1.0)
            else:
                kept, supp = nms_select(arr, sca, a.containment, a.policy,
                                        a.hy_score_k, a.hy_min_h)
            kb = arr[kept]
            ks = sca[kept]
            n_kept += len(kept)

            det = sv.Detections(xyxy=kb.astype(np.float32),
                                confidence=ks.astype(np.float32),
                                class_id=np.zeros(len(kb), dtype=int))
            det = tracker.update_with_detections(det)
            tids = det.tracker_id
            tb = det.xyxy
            ts = det.confidence
            tlist = [int(t) for t in tids] if tids is not None else []
            blist = [[round(float(v), 1) for v in x] for x in tb]
            slist = [round(float(v), 4) for v in ts]
            out_frames.append((fr, tlist, blist))
            fo.write(json.dumps({
                "frame": fr, "count": len(tlist), "raw_count": len(boxes),
                "deduped_count": len(kept), "removed": len(supp),
                "track_ids": tlist, "boxes": blist, "scores": slist,
            }, ensure_ascii=False) + "\n")
            prev = [[float(v) for v in x] for x in kb]

    # ★ 轨迹级融合：把「被包含 + 头顶对齐 + 从不共现」的短轨并进主轨。
    # 为什么必须在跟踪后做：这类两轨各自都能匹配上自己的检测（#129 的轨迹在
    # lost_track_buffer 内没被移除），匹配阶段根本没机会交叉，见 containment_bytetrack。
    if a.tracker == "cont" and a.fuse_contained:
        from containment_bytetrack import fuse_contained_tracks
        mp = fuse_contained_tracks(
            out_frames, cont_thresh=a.cont_thresh, center_ratio=a.cont_center,
            y1_tol=a.cont_y1, min_obs_short=a.fuse_min_obs,
        )
        if mp:
            recs = [json.loads(ln) for ln in out_path.open(encoding="utf-8")]
            for d in recs:
                d["track_ids"] = [mp.get(int(t), int(t)) for t in d.get("track_ids") or []]
                d["count"] = len(d["track_ids"])
            with out_path.open("w", encoding="utf-8") as g:
                for d in recs:
                    g.write(json.dumps(d, ensure_ascii=False) + "\n")
            out_frames = [(r["frame"], r["track_ids"], r["boxes"]) for r in recs]
            print(f"[fuse] 轨迹融合 {len(mp)} 条 → 已并入主轨")

    st = track_stats(out_frames, a.stride)
    print(f"[out] {out_path}")
    print(f"[stats] {st}")
    if hasattr(tracker, "n_cont_match"):
        print(f"[cont] 包含匹配救回={tracker.n_cont_match} 次")
    print(f"[box] raw@{a.thr}候选={n_cand} NMS后={n_kept} "
          f"(删 {n_cand - n_kept}) 尺寸门修正={n_gate}")

    if a.compare:
        # ★ 不能比 track_id 编号：第一帧差一个 ID，后面全部错位但分组其实相同。
        # 正确口径 = 比「框→轨迹」的**划分**（忽略编号），覆盖跨帧延续性。
        def load_map(path: str) -> dict[tuple, int]:
            m: dict[tuple, int] = {}
            with Path(path).open(encoding="utf-8") as f:
                for ln in f:
                    d = json.loads(ln)
                    fr = int(d["frame"])
                    for t, b in zip(d.get("track_ids") or [], d.get("boxes") or []):
                        m[(fr, tuple(round(float(v), 1) for v in b))] = int(t)
            return m

        def partition(m: dict[tuple, int]) -> frozenset:
            g: dict[int, set] = defaultdict(set)
            for k, t in m.items():
                g[t].add(k)
            return frozenset(frozenset(v) for v in g.values())

        mine = load_map(a.out)
        theirs = load_map(a.compare)
        common = set(mine) & set(theirs)
        p_m = partition({k: mine[k] for k in common})
        p_t = partition({k: theirs[k] for k in common})
        # 帧级框集一致性（回归保护：策略不该改检测框）
        ref_by_fr: dict[int, list] = {}
        with Path(a.compare).open(encoding="utf-8") as f:
            for ln in f:
                d = json.loads(ln)
                ref_by_fr[int(d["frame"])] = d.get("boxes") or []
        same_fr = same_box = 0
        for fr, tids, blist in out_frames:
            if fr not in ref_by_fr:
                continue
            same_fr += 1
            rb = ref_by_fr[fr]
            if len(rb) == len(blist) and all(
                    all(abs(p - q) <= 0.15 for p, q in zip(x, y))
                    for x, y in zip(rb, blist)):
                same_box += 1
        print(f"[compare] 同帧 {same_fr}  框集一致 {same_box} ({same_box/max(1,same_fr)*100:.2f}%)")
        print(f"[compare] 轨迹划分一致={'是' if p_m == p_t else '否'}  "
              f"(组数 我={len(p_m)} 基准={len(p_t)} 公共框={len(common)})")

    if a.dump_switch:
        # 列出「A 结束 → 8s 内同位置出现新 B」的全部案例（人工复核入口）
        tracks: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
        for fr, tids, blist in out_frames:
            for t, b in zip(tids, blist):
                tracks[t].append((fr, b))
        starts = {t: s[0][0] for t, s in tracks.items()}
        per: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
        for t, seq in tracks.items():
            for f, b in seq:
                per[f].append((t, b))

        def iou(a_, b_):
            ix1, iy1 = max(a_[0], b_[0]), max(a_[1], b_[1])
            ix2, iy2 = min(a_[2], b_[2]), min(a_[3], b_[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = area(a_) + area(b_) - inter
            return inter / ua if ua > 0 else 0.0

        cases = []
        for t, seq in tracks.items():
            lf, lb = seq[-1]
            for g in range(a.stride, a.stride * 41, a.stride):
                fr = lf + g
                if fr not in per:
                    continue
                for t2, b2 in per[fr]:
                    if t2 == t or starts.get(t2, 0) <= lf:
                        continue
                    if iou(lb, b2) >= 0.5:
                        cases.append({"from": t, "to": t2, "f_last": lf, "f_new": fr,
                                      "gap_s": round((fr - lf) / a.fps, 2),
                                      "h_last": round(lb[3] - lb[1], 1),
                                      "h_new": round(b2[3] - b2[1], 1)})
                        break
                else:
                    continue
                break
        Path(a.dump_switch).write_text(json.dumps(cases, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        print(f"[switch] {len(cases)} 例 -> {a.dump_switch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
