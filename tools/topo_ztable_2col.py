# -*- coding: utf-8 -*-
"""TOP-1 结果汇总：v10 vs v35 窄窗 z 表对比 + 全 30 对明细导出。"""
import json

P10 = "output_crosscam/analysis/topology/topo_frag_narrow_v10.json"
P35 = "output_crosscam/analysis/topology/topo_frag_narrow_v35.json"
OUT = "output_crosscam/analysis/topology/topo_ztable_2col.json"

v10 = json.load(open(P10, encoding="utf-8"))
v35 = json.load(open(P35, encoding="utf-8"))


def get_win(d, lo, hi):
    """按窗口数值匹配（键名可能是 frag[-1,5] 或 frag[-1.0,5.0]）。"""
    for k, v in d.items():
        if not isinstance(v, dict) or "window" not in v:
            continue
        if abs(v["window"][0] - lo) < 1e-6 and abs(v["window"][1] - hi) < 1e-6:
            return v
    return None


WINSPEC = [(-1, 3), (-1, 5), (-1, 10), (-1, 15), (0.5, 60)]


def key(r):
    return f"{r['A']}->{r['B']}"


table = {}
for lo, hi in WINSPEC:
    win = f"frag[{lo},{hi}]"
    a, b = get_win(v10, lo, hi), get_win(v35, lo, hi)
    m10 = {key(r): r for r in a["rows"]}
    print("=" * 78)
    print(f"{win} | v10 p_family={a['p_family']} zmax={a['zmax_obs']} | "
          f"v35 p_family={b['p_family']} zmax={b['zmax_obs']}")
    print(f"{'pair':>10} {'z10':>7} {'z35':>7} {'n_obs35':>9} {'n_obs10':>9}")
    for r in b["rows"][:12]:
        k = key(r)
        rr = m10.get(k, {})
        print(f"{k:>10} {rr.get('z', '-'):>7} {r['z']:>7} {r['n_obs']:>9} {rr.get('n_obs', '-'):>9}")

# 全表落盘（两口径 × 全窗口 × 全 30 对）
for tag, src in (("v10", v10), ("v35", v35)):
    for lo, hi in WINSPEC:
        win = f"frag[{lo},{hi}]"
        w = get_win(src, lo, hi)
        table[f"{tag}|{win}"] = {
            "p_family": w["p_family"],
            "zmax": w["zmax_obs"],
            "n_z_ge3": w["n_z_ge3"],
            "rows": w["rows"],
        }
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(table, f, ensure_ascii=False, indent=1)
print("\nsaved", OUT)
