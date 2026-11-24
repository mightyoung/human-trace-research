"""把 19 段的 det jsonl 压成「每轨一行」的摘要表，供跨机位出入对分析用。

输出 `e21_finish/track_summary.json`：
  { "2026-10-22|办公区1": { "tids": [...], "rows": [[t0,t1,n,h_med,u0,v0,u1,v1,edge0,edge1], ...] } }

字段说明：
  t0/t1    轨的首/末**绝对时刻**（det 的 t，来自 spec 的 OSD anchor；跨机位可直接比）
  n        采样帧数
  h_med    框高中位（960×540 检测帧坐标）
  u0,v0    首帧框中心；u1,v1 末帧框中心
  edge0/1  首/末帧是否贴边（判断「人从画面边缘进出」，是「真离开」的代理）

★ 为什么用贴边而不是轨迹外推：11-12 已证轨迹预测器不能解交替（框距仅 0.07 身位、
  46.5% 失效在轨终点 1 s 内）。贴边是弱信号但不会被预测误差放大。
"""
import json
import os
from pathlib import Path

import numpy as np

DAY = Path("e21_day")
OUT = Path("e21_finish/track_summary.json")

EDGE_U = 70.0      # 960×540 下的边界带
EDGE_V_TOP = 40.0
EDGE_V_BOT = 500.0


def edge(u, v):
    return int(u < EDGE_U or u > 960 - EDGE_U or v < EDGE_V_TOP or v > EDGE_V_BOT)


def summarize(path):
    agg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r["t"]
            b = r["box"]
            u = (b[0] + b[2]) / 2
            v = (b[1] + b[3]) / 2
            h = b[3] - b[1]
            m = agg.get(r["tid"])
            if m is None:
                agg[r["tid"]] = [t, t, 1, [h], u, v, u, v, edge(u, v), edge(u, v)]
            else:
                if t < m[0]:
                    m[0] = t
                    m[4], m[5], m[8] = u, v, edge(u, v)
                if t >= m[1]:
                    m[1] = t
                    m[6], m[7], m[9] = u, v, edge(u, v)
                m[2] += 1
                if len(m[3]) < 40:
                    m[3].append(h)
    tids, rows = [], []
    for tid in sorted(agg):
        t0, t1, n, hs, u0, v0, u1, v1, e0, e1 = agg[tid]
        tids.append(int(tid))
        rows.append([round(t0, 2), round(t1, 2), n, round(float(np.median(hs)), 1),
                     round(u0, 1), round(v0, 1), round(u1, 1), round(v1, 1), e0, e1])
    return {"tids": tids, "rows": rows}


def main():
    out = {}
    metas = sorted(DAY.glob("det_*.jsonl"))
    for p in metas:
        name = p.stem.replace("det_", "")
        s = summarize(p)
        out[name] = s
        print(f"{name:<26} tracks={len(s['tids']):>5} "
              f"t=[{min(r[0] for r in s['rows']):.0f},{max(r[1] for r in s['rows']):.0f}]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("->", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
