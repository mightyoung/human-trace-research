# -*- coding: utf-8 -*-
"""TOP-2 对照：开放词表"门"检测 vs C0 v7 动态门坐标。

口径：
  - 门框底边中点（脚点）与框中心两种落点；
  - 跨帧粗聚类（0.05 网格）得"语义门"唯一簇；
  - 与 v7 门中心（exit+entry 全含）按距离 ≤0.12 判匹配；
  - 报告：语义门簇数、与 v7 门的双向匹配率、未匹配项（供人工抽看）。
输入：analysis/top2/door_probe.json（远程产物拉回）+ c0_baseline_v7.json。
"""
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
PROBE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ANA, "top2", "door_probe.json")
C0 = os.path.join(ANA, "c0", "c0_baseline_v7.json")
OUT = PROBE.replace(".json", "_compare.json")


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def main():
    probe = json.load(open(PROBE, encoding="utf-8"))
    c0 = json.load(open(C0, encoding="utf-8"))
    gates = {}
    for cam, lst in c0["gates"].items():
        gates[cam] = [dict(kind=g.get("kind"), center=g["center"]) for g in lst]

    out = {"probe_prompt": probe.get("prompt"), "box_thr": probe.get("box_thr"), "cams": {}}
    for cam, rows in probe["cams"].items():
        # 收集鞋点/中心 + 跨帧聚类
        uniq = {}
        n_boxes = 0
        for r in rows:
            W, H = r["W"], r["H"]
            for b, s in zip(r["boxes"], r["scores"]):
                n_boxes += 1
                x1, y1, x2, y2 = b
                foot = ((x1 + x2) / 2 / W, y2 / H)
                cen = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H)
                key = (round(foot[0] / 0.05), round(foot[1] / 0.05))
                it = uniq.get(key)
                if it is None or s > it["score"]:
                    uniq[key] = dict(foot=foot, cen=cen, score=float(s))
        gg = gates.get(cam, [])
        gate_hit = [False] * len(gg)
        door_hit = []
        for u in uniq.values():
            hit = None
            for i, g in enumerate(gg):
                if dist(u["foot"], g["center"]) <= 0.12 or dist(u["cen"], g["center"]) <= 0.12:
                    gate_hit[i] = True
                    hit = i
                    break
            door_hit.append(dict(foot=[round(u["foot"][0], 3), round(u["foot"][1], 3)],
                                 score=round(u["score"], 3), gate=hit))
        out["cams"][cam] = dict(
            n_boxes_raw=n_boxes, n_unique_doors=len(uniq),
            n_gates=len(gg),
            gates_matched=sum(gate_hit),
            doors_matched=sum(1 for d in door_hit if d["gate"] is not None),
            doors_unmatched=[d for d in door_hit if d["gate"] is None],
            gates_unmatched=[dict(kind=gg[i].get("kind"), center=gg[i]["center"])
                             for i in range(len(gg)) if not gate_hit[i]],
        )
        c = out["cams"][cam]
        print(f"[{cam}] 原始框 {c['n_boxes_raw']} → 唯一语义门 {c['n_unique_doors']} | "
              f"v7 门 {c['n_gates']} 被语义命中 {c['gates_matched']} | "
              f"语义门命中 v7 {c['doors_matched']}/{c['n_unique_doors']}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
