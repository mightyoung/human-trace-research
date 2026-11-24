"""跨机位出入对 · 第 2 步：空间分布探查。

为什么需要：第 1 步发现走廊 / 茶水间的「贴边率 = 0%」，说明人在这些画面里
**不在边界消失**（长通道两端落在画面内部、或被门框/立柱遮挡）。于是
「贴边 = 离开」这个代理在走廊上不可用，必须先看人到底在画面的哪一块活动。

输出每段：首帧/末帧框中心 u、v 的分位（p5/p25/p50/p75/p95）+ 有效轨数。
"""
import json
from datetime import datetime, timedelta

import numpy as np


def hhmm(t):
    return (datetime(2026, 1, 1) + timedelta(seconds=float(t))).strftime("%H:%M")


def pc(a, q):
    return float(np.percentile(a, q))


def main():
    d = json.loads(open("tmp/r5-e1/track_summary.json", encoding="utf-8").read())
    print(f"{'段':<24}{'轨':>6} | {'u0 分位 p5/25/50/75/95':>32} | {'u1 分位':>32} | {'v 中位':>7}")
    print("-" * 116)
    for name, s in d.items():
        r = np.array(s["rows"], dtype=float)
        t0, t1, n, h, u0, v0, u1, v1 = [r[:, i] for i in range(8)]
        ok = ((t1 - t0) >= 0.8) & (h >= 30)
        u0, u1, v0, v1, h, t0, t1 = u0[ok], u1[ok], v0[ok], v1[ok], h[ok], t0[ok], t1[ok]
        f1 = " ".join(f"{pc(u0, q):>5.0f}" for q in (5, 25, 50, 75, 95))
        f2 = " ".join(f"{pc(u1, q):>5.0f}" for q in (5, 25, 50, 75, 95))
        # 行进方向：u1-u0 的符号分布（负=向左）
        du = u1 - u0
        left = float((du < -20).mean()) * 100
        right = float((du > 20).mean()) * 100
        print(f"{name:<24}{len(u0):>6} | {f1:>32} | {f2:>32} | "
              f"{pc(v0, 50):>4.0f}/{pc(v1, 50):>4.0f}  ←{left:>3.0f}% →{right:>3.0f}%")
    print("\n注：u 是 960 宽检测帧的横坐标；←/→ 是末帧比首帧左移/右移 >20px 的轨占比")


if __name__ == "__main__":
    main()
