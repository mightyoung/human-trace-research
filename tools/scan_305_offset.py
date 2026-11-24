# -*- coding: utf-8 -*-
"""305 时钟偏移扫描（OSD 标定验证）—— S-1 修复的核心假设检验。

现象：305 的 gated-exit 实体的跨机位候选，dt 几乎**全为负**（-3s/-4s/-6s/-13s…），
      而 305 的 TIME_OFFS = -5.0（相对基准时钟的秒偏移）。
假设：305 时间戳整体**偏晚** → 对端总是"先出现" → 转移在时间上不成立 → 边被丢弃。

方法：对偏移修正 δ ∈ [-20, +10]，令 t_305' = t_305 + δ，统计
        a) 满足 cos ≥ SIM 且 0 ≤ t_b_first − t_a_last' ≤ DT_MAX 的候选对数
        b) greedy 唯一化后的"新增边"数
      峰值位置即最优 δ。

用法：DET_VIEW=clean2 python tools/scan_305_offset.py
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
import c0_baseline as C  # noqa: E402
from gil import GILStore  # noqa: E402

CAMS = ["305", "316", "318", "320", "323", "343"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="305")
    ap.add_argument("--sim", type=float, default=0.94)
    ap.add_argument("--dt-max", type=float, default=30.0)
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--deltas", default="-20,-17.5,-15,-12.5,-10,-7.5,-5,-2.5,0,2.5,5,7.5,10")
    a = ap.parse_args()

    st = GILStore(ROOT + "/" + a.store, create=False)
    vb = st.vectors(load=True)
    ent = {r["id"]: dict(cam=r["cam"], eid=r["eid"]) for r in
           st.con.execute("SELECT id,cam,eid FROM entity")}
    mem = defaultdict(list)
    for r in st.con.execute("SELECT entity_id,id FROM obs WHERE entity_id IS NOT NULL"):
        mem[r["entity_id"]].append(r["id"])
    E, keys = [], []
    for eid_i, oids in mem.items():
        b = ent[eid_i]["cam"]
        vs = [vb._cache[b][0][vb._cache[b][2][o]]
              for o in oids if b in vb._cache and o in vb._cache[b][2]]
        if not vs:
            continue
        v = np.mean(vs, axis=0)
        v /= (np.linalg.norm(v) + 1e-9)
        E.append(v)
        keys.append((ent[eid_i]["cam"], ent[eid_i]["eid"]))
    E = np.stack(E)
    rowof = {k: i for i, k in enumerate(keys)}

    # 实体端点信息
    info = {}
    gated_ex, gated_en = set(), set()
    for cam in CAMS:
        ents = C.build_entity_tracks(cam)
        _, ex, en = C.classify_endpoints(cam, ents)
        for eid, e in ents.items():
            if not e["moving"]:
                continue
            info[(cam, eid)] = e
            if ex.get(eid):
                gated_ex.add((cam, eid))
            if en.get(eid):
                gated_en.add((cam, eid))

    cam = a.cam
    src = [k for k in gated_ex if k[0] == cam]
    print(f"[scan] {cam} gated-exit {len(src)} 条；对照组 = 其他机位 gated-entry {len(gated_en)-len([k for k in gated_en if k[0]==cam])} 条")
    print(f"[scan] 阈值 cos ≥ {a.sim}，时间窗 [0, {a.dt_max}]s")

    # 预计算每对 (a,b) 的 cos
    pairs = []
    for ka in src:
        i = rowof.get(ka)
        if i is None:
            continue
        s = E @ E[i]
        for j in np.argsort(-s)[:80]:
            kb = keys[j]
            if kb[0] == cam or kb not in gated_en:
                continue
            c = float(s[j])
            if c < a.sim:
                continue
            pairs.append((ka, kb, c))
    print(f"[scan] 预筛后 (a,b) 对 {len(pairs)}（cos≥{a.sim} 且 b 入口门内）")
    print()
    print(f"{'δ(s)':>7} {'候选对':>8} {'唯一化边':>9} {'涉及源实体':>10}  dt 中位")
    best = None
    for d in [float(x) for x in a.deltas.split(",")]:
        ok = []
        for ka, kb, c in pairs:
            dt = info[kb]["t_first"] - (info[ka]["t_last"] + d)
            if 0.0 <= dt <= a.dt_max:
                ok.append((c, ka, kb, dt))
        ok.sort(reverse=True)
        ua, ub = set(), set()
        kept = []
        for c, ka, kb, dt in ok:
            if ka in ua or kb in ub:
                continue
            ua.add(ka)
            ub.add(kb)
            kept.append(dt)
        med = np.median(kept) if kept else float("nan")
        flag = ""
        if best is None or len(kept) > best[1]:
            best = (d, len(kept), len(ua))
        print(f"{d:7.1f} {len(ok):8d} {len(kept):9d} {len(ua):10d}  {med:6.1f}s")
    print()
    print(f"★ 峰值: δ = {best[0]:+.1f}s → 唯一化边 {best[1]}（涉及 {best[2]} 个源实体）")
    print(f"  当前 TIME_OFFS[{cam}] = {C.T0[cam]:+.1f}  → 建议 {C.T0[cam] + best[0]:+.1f}")
    st.close()


if __name__ == "__main__":
    main()
