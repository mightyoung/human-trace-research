"""跨机位出入对 · 第 1 步：数据探查。

回答三件事：
  1. 每段的绝对时段、轨数、轨时长、贴边比例（"真离开画面"的可观测代理）
  2. 每天哪几个机位**同时覆盖**（只有重叠时段内的出入对才有意义）
  3. 同一天相邻机位对的规模（A 的离场事件 × B 的入场事件）

跑法：python e21_door_probe.py [--summary tmp/r5-e1/track_summary.json]
"""
import argparse
import json
from datetime import datetime, timedelta
from itertools import combinations

import numpy as np

# 机位归类：同一物理机位的不同时段合并（10-23 的办公区1/1b 是同一机位上午/下午）
CAM_GROUP = {
    "办公区1": "办公区1", "办公区1b": "办公区1",
    "办公区2": "办公区2", "办公区球机": "球机",
    "茶水间": "茶水间", "电梯间": "电梯间", "行政走廊通道": "走廊",
}


def hhmm(t):
    return (datetime(2026, 1, 1) + timedelta(seconds=float(t))).strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="tmp/r5-e1/track_summary.json")
    ap.add_argument("--min-dur", type=float, default=0.8, help="轨时长下限（秒），过滤瞬时假框")
    ap.add_argument("--min-h", type=float, default=30.0, help="框高下限（960×540 坐标）")
    args = ap.parse_args()

    d = json.loads(open(args.summary, encoding="utf-8").read())

    print("=" * 108)
    print(f"{'段':<24}{'日期':<12}{'机位':<8}{'轨':>6}{'有效':>6}{'时段':>21}{'时长h':>7}"
          f"{'轨长中位':>9}{'贴边首':>8}{'贴边末':>8}")
    print("-" * 108)
    seg_info = {}
    for name, s in d.items():
        date, cam_raw = name.split("_", 1)
        rows = np.array(s["rows"], dtype=float)
        t0, t1, n, h, u0, v0, u1, v1, e0, e1 = [rows[:, i] for i in range(10)]
        dur = t1 - t0
        ok = (dur >= args.min_dur) & (h >= args.min_h)
        info = dict(date=date, cam=CAM_GROUP.get(cam_raw, cam_raw), cam_raw=cam_raw,
                    t0=float(t0[ok].min()), t1=float(t1[ok].max()),
                    n=int(ok.sum()), n_all=len(rows),
                    edge0=float(e0[ok].mean()), edge1=float(e1[ok].mean()),
                    dur_med=float(np.median(dur[ok])),
                    rows=rows[ok])
        seg_info[name] = info
        print(f"{name:<24}{date:<12}{info['cam']:<8}{info['n_all']:>6}{info['n']:>6}"
              f"{hhmm(info['t0'])+'→'+hhmm(info['t1']):>21}"
              f"{(info['t1']-info['t0'])/3600:>7.2f}{info['dur_med']:>9.1f}"
              f"{info['edge0']*100:>7.0f}%{info['edge1']*100:>7.0f}%")

    print("\n" + "=" * 108)
    print("按天 × 机位汇总（同机位多段合并时段）")
    print("-" * 108)
    for date in sorted({v["date"] for v in seg_info.values()}):
        print(f"\n[{date}]")
        cams = {}
        for name, v in seg_info.items():
            if v["date"] != date:
                continue
            c = cams.setdefault(v["cam"], dict(segs=[], t0=1e18, t1=-1e18, n=0))
            c["segs"].append(name)
            c["t0"] = min(c["t0"], v["t0"])
            c["t1"] = max(c["t1"], v["t1"])
            c["n"] += v["n"]
        for cam in sorted(cams, key=lambda x: cams[x]["t0"]):
            c = cams[cam]
            print(f"  {cam:<8} 段{len(c['segs'])}  有效轨{c['n']:>5}  "
                  f"{hhmm(c['t0'])}→{hhmm(c['t1'])}  {(c['t1']-c['t0'])/3600:.2f} h")
        # 两两重叠
        print("  同时覆盖（重叠 ≥ 30 min）：")
        for a, b in combinations(sorted(cams, key=lambda x: cams[x]["t0"]), 2):
            A, B = cams[a], cams[b]
            lo, hi = max(A["t0"], B["t0"]), min(A["t1"], B["t1"])
            if hi - lo >= 1800:
                print(f"    {a:<8} ↔ {b:<8} 重叠 {(hi-lo)/3600:.2f} h"
                      f"（{hhmm(lo)}→{hhmm(hi)}）  轨 {A['n']} × {B['n']}")


if __name__ == "__main__":
    main()
