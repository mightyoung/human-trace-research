# -*- coding: utf-8 -*-
"""E21 人脸可抽性统计：只读 det jsonl，不解码视频。

对每个段（day_cam_tr/te）统计：
  - 轨数、框高中位、以及框高 >= H 的轨数
  - 每条轨的最佳采样帧（框高最大）与该帧的框
目的：决定人脸抽取要覆盖哪些机位、以及规模。
"""
import json
import os
import sys
import collections
import numpy as np

E = "/share/data4/yhh/data/human-count/e21"
H = float(sys.argv[1]) if len(sys.argv) > 1 else 130.0


def scan(seg):
    p = os.path.join(E, "dd", seg, f"{seg}_detections.jsonl")
    if not os.path.exists(p):
        return None
    per = collections.defaultdict(list)
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for t, b in zip(d["track_ids"], d["boxes"]):
                per[t].append((fr, b))
    rows = []
    for t, items in per.items():
        hs = np.array([b[3] - b[1] for _, b in items], dtype=np.float32)
        frs = np.array([fr for fr, _ in items])
        rows.append(dict(t=int(t), n=len(items), h_med=float(np.median(hs)),
                         h_max=float(hs.max()), f0=int(frs.min()), f1=int(frs.max())))
    return rows


def main():
    segs = []
    for day in ("2026-10-22", "2026-10-23"):
        d = os.path.join(E, "dd")
        for s in sorted(os.listdir(d)):
            if s.startswith(day):
                segs.append(s)
    print(f"{'segment':<34} {'trk':>5} {'h_med':>6} {'≥H':>5} {'≥H/trk':>7} {'nfr':>7}")
    tot = collections.Counter()
    for s in segs:
        rows = scan(s)
        if not rows:
            print(f"{s:<34}   MISS")
            continue
        hm = float(np.median([r["h_med"] for r in rows]))
        ok = [r for r in rows if r["h_med"] >= H]
        # 该段需要的采样帧数（每轨最多 8 帧，按框高排序取）
        nfr = sum(min(8, r["n"]) for r in ok)
        print(f"{s:<34} {len(rows):>5} {hm:>6.0f} {len(ok):>5} "
              f"{100*len(ok)/max(1,len(rows)):>6.1f}% {nfr:>7}")
        tot["trk"] += len(rows)
        tot["ok"] += len(ok)
        tot["nfr"] += nfr
    print(f"\n合计 轨 {tot['trk']}，合格轨 {tot['ok']} "
          f"({100*tot['ok']/max(1,tot['trk']):.1f}%)，需解帧 {tot['nfr']}")


if __name__ == "__main__":
    main()
