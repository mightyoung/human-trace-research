# -*- coding: utf-8 -*-
"""TOP-1 时钟敏感性：305 偏移扫描汇总（窗 [-1,10]s，shifts=1000）。"""
import json
import os

BASE = "output_crosscam/analysis/topology/clock_scan"
KEYS = ["305->316", "316->305", "305->318", "318->305", "305->320", "320->305",
        "305->323", "323->305", "343->305", "305->343"]


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    for k, v in d.items():
        if isinstance(v, dict) and "window" in v and abs(v["window"][0] + 1) < 1e-6:
            return d["tag"], v
    return None, None


rows = []
for fn in sorted(os.listdir(BASE)):
    tag, w = load(os.path.join(BASE, fn))
    m = {f"{r['A']}->{r['B']}": r["z"] for r in w["rows"]}
    rows.append((tag, w["zmax_obs"], w["p_family"], m))

print(f"{'shift':>16} {'zmax':>6} {'p_fam':>7} | " + " ".join(f"{k:>9}" for k in KEYS))
for tag, zmax, pf, m in rows:
    print(f"{tag:>16} {zmax:>6} {pf:>7} | " + " ".join(f"{m.get(k, 0):>9}" for k in KEYS))
