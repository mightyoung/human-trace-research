#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把一段检测按**时间窗**切成「图库/查询」两段，用于 CHIRLA 的日内（同场）切分实验。

背景：VALIDATION §3 预注册的是 seq_000 作图库、seq_001/002 作查询，两段录制相隔约 50 分钟，
同人余弦从 0.758 掉到 0.662、覆盖塌到 3.4% → 无法判定。现场的协议其实是"同一录制内按时间切"，
所以这里做一个同构的第二分析（偏离预注册，须与主结果并列报告）。

做法：帧号是原始视频的绝对帧号，按 --split-frac 划一条时间线；
      边界之前的框归图库段 `<src>g`，之后的归查询段 `<src>q`，各自重编 track id。
      跨边界的轨迹会被切成两条（分别落在两段里）——这正是分段应有的行为。

用法：
  PY tools/split_seg_by_time.py --in-dir validation/chirla/det/default \
     --out-dir validation/chirla/det_time --segs c1_000,...,c7_000 --split-frac 0.6
"""
import argparse
import collections
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="validation/chirla/det/default")
    ap.add_argument("--out-dir", default="validation/chirla/det_time")
    ap.add_argument("--segs", required=True)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--min-frames", type=int, default=1,
                    help="段内少于该帧数的轨迹直接丢（默认 1，交给 make_ab_jsonl 的 6 帧门槛）")
    a = ap.parse_args()

    for src in [x.strip() for x in a.segs.split(",") if x.strip()]:
        jp = os.path.join(a.in_dir, src, f"{src}_detections.jsonl")
        rows = [json.loads(l) for l in open(jp, encoding="utf-8") if l.strip()]
        if not rows:
            print(f"[{src}] 空，跳过")
            continue
        fmax = max(r["frame"] for r in rows)
        f0 = int(round(fmax * a.split_frac))
        # 先把每条轨迹的帧窗口归属算出来（按该轨迹自己的帧）
        span = collections.defaultdict(list)
        for r in rows:
            for t in (r.get("track_ids") or []):
                if t is not None:
                    span[int(t)].append(int(r["frame"]))
        # 轨迹归哪段：多数帧（跨越边界的会被切开，这里只决定"整条算谁的"）
        side = {}
        for t, frs in span.items():
            g = sum(1 for f in frs if f < f0)
            side[t] = "g" if g * 2 >= len(frs) else "q"
        maps = {"g": {}, "q": {}}
        buf = {"g": [], "q": []}
        for r in rows:
            fr = int(r["frame"])
            win = "g" if fr < f0 else "q"
            tids = r.get("track_ids") or []
            boxes = r.get("boxes") or []
            sc = r.get("scores") or [1.0] * len(boxes)
            keep_t, keep_b, keep_s = [], [], []
            for i, t in enumerate(tids):
                if t is None:
                    continue
                t = int(t)
                # 只有"归属本窗"的轨迹在本窗的帧才留下 → 跨边界轨迹自然被切成两条
                if side[t] != win:
                    continue
                if t not in maps[win]:
                    maps[win][t] = len(maps[win])
                keep_t.append(maps[win][t])
                keep_b.append(boxes[i])
                if i < len(sc):
                    keep_s.append(sc[i])
            if keep_t:
                buf[win].append(dict(frame=fr, track_ids=keep_t, boxes=keep_b, scores=keep_s))

        for win in ("g", "q"):
            seg = f"{src}{win}"
            d = os.path.join(a.out_dir, seg)
            os.makedirs(d, exist_ok=True)
            n_tr = collections.Counter()
            with open(os.path.join(d, f"{seg}_detections.jsonl"), "w", encoding="utf-8") as fo:
                for r in buf[win]:
                    n_tr.update(r["track_ids"])
                    fo.write(json.dumps(r) + "\n")
            ok = sum(1 for v in n_tr.values() if v >= a.min_frames)
            print(f"[{seg}] 帧 {len(buf[win])}｜轨迹 {len(n_tr)}（≥{a.min_frames} 帧 {ok}）"
                  f"｜时间窗 {('0..%d' % (f0 - 1)) if win == 'g' else ('%d..%d' % (f0, fmax))}")
        print(f"     分割帧 {f0}（{a.split_frac:.0%} of {fmax}）")


if __name__ == "__main__":
    main()
