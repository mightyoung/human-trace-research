# -*- coding: utf-8 -*-
"""S-1 诊断：305 出门覆盖为什么是 0.1。

判定逻辑（c0_baseline）：covered = exit 端点被 gated 且该实体有跨机位出边。
305 有 20 个 gated-exit，只有 2 个有出边 → 8% 连续性，全项目最低。

本脚本回答唯一关键问题：
  那 18 个「静默」实体，是
    (a) 在其他机位存在强外观候选（cos 高、dt 合理）→ relay 漏匹配 = **可修**
    (b) 在其他机位完全没有候选 → 人没出现在别处 = **数据限制，不该背锅**

用法：
  DET_VIEW=clean2 python tools/diag_305_egress.py [--cam 305] [--topk 5]
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
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--dt-window", type=float, default=120.0)
    a = ap.parse_args()

    st = GILStore(ROOT + "/" + a.store, create=False)
    vb = st.vectors(load=True)

    # ---- 实体代表向量（成员向量均值归一）----
    ent = {}
    for r in st.con.execute("SELECT id,cam,eid,t0,t1 FROM entity"):
        ent[r["id"]] = dict(id=r["id"], cam=r["cam"], eid=r["eid"], t0=r["t0"], t1=r["t1"])
    mem = defaultdict(list)
    for r in st.con.execute("SELECT entity_id,id FROM obs WHERE entity_id IS NOT NULL"):
        mem[r["entity_id"]].append(r["id"])
    E, ids, meta = [], [], []
    for eid_i, oids in mem.items():
        b = ent[eid_i]["cam"]
        vs = [vb._cache[b][0][vb._cache[b][2][o]]
              for o in oids if b in vb._cache and o in vb._cache[b][2]]
        if not vs:
            continue
        v = np.mean(vs, axis=0)
        v /= (np.linalg.norm(v) + 1e-9)
        E.append(v)
        ids.append(eid_i)
        meta.append(ent[eid_i])
    E = np.stack(E)
    eid2idx = {(m["cam"], m["eid"]): i for i, m in enumerate(meta)}   # (cam, eid) → 行号
    print(f"[diag] 实体 {len(ids)} 个（{E.shape[1]} 维）")

    # ---- C0 的 gated 判定（与 c0_baseline 完全一致）----
    cam = a.cam
    ents = C.build_entity_tracks(cam)
    gates, ex_mask, en_mask = C.classify_endpoints(cam, ents)
    mov = {e: v for e, v in ents.items() if v["moving"]}
    gated_ex = [e for e in mov if ex_mask.get(e)]
    print(f"[{cam}] 实体 {len(ents)} 移动 {len(mov)} gated-exit {len(gated_ex)}")
    print(f"[{cam}] 门：{[(g['kind'], g['center'], g['n_pts']) for g in gates]}")
    print(f"[{cam}] CFG: cell={C.CFG['cell']} radius={C.CFG['gate_radius']} "
          f"border={C.CFG['border_margin']} min_pts={C.CFG['gate_min_pts']}")

    # ---- 已有出边 ----
    edges = st.con.execute("SELECT a_id,b_id,dt,status FROM edge WHERE kind LIKE 'relay%'").fetchall()
    has_out = set()
    dir_of = {}
    for r in edges:
        ea, eb = ent.get(r["a_id"]), ent.get(r["b_id"])
        if ea and eb:
            has_out.add((ea["cam"], ea["eid"]))
            dir_of[(ea["cam"], ea["eid"])] = (eb["cam"], eb["eid"], r["dt"])

    # ---- 对每个 gated-exit 实体：找跨机位候选 ----
    print()
    print("=" * 100)
    print(f"{'实体':14} {'dur':>6} {'出口(u,v)':>16} {'gated因':>8} "
          f"{'有边':>5} {'最佳跨机位候选(cos/dt)':<44}")
    print("=" * 100)
    n_strong_no_edge = 0
    n_no_cand = 0
    n_edge = 0
    rows = []
    for eid in gated_ex:
        e = ents[eid]
        i = eid2idx.get((cam, eid))
        u1, v1 = e["u1"], e["v1"]
        if (u1 <= C.CFG["border_margin"] or u1 >= 1 - C.CFG["border_margin"] or
                v1 <= C.CFG["border_margin"] or v1 >= 1 - C.CFG["border_margin"]):
            why = "贴边"
        else:
            why = "门内"
        edged = (cam, eid) in has_out
        if i is None:
            rows.append((eid, e, why, edged, "无向量"))
            continue
        s = E @ E[i]
        order = np.argsort(-s)
        cands = []
        for j in order:
            if meta[j]["cam"] == cam:
                continue
            # ★ 转移时间必须用「b 进入 − a 离开」（非 t_first 差）：
            #   否则重叠机位/长实体会产生负 dt 的假候选
            dt = meta[j]["t0"] - e["t_last"]
            cands.append((float(s[j]), meta[j]["cam"], meta[j]["eid"], dt))
            if len(cands) >= a.topk:
                break
        best = cands[0] if cands else None
        if edged:
            n_edge += 1
        elif best and best[0] >= 0.90 and -5 <= best[3] <= a.dt_window:
            n_strong_no_edge += 1
        elif not best or best[0] < 0.85:
            n_no_cand += 1
        rows.append((eid, e, why, edged, cands))
        cs = ", ".join(f"{c[1]}:{c[2].split('-')[-1]}({c[0]:.3f}/{c[3]:+.0f}s)" for c in cands[:2])
        print(f"{eid:14} {e['dur']:6.1f} ({u1:5.2f},{v1:5.2f}) {why:>8} "
              f"{'✓' if edged else ' ':>5} {cs:<44}")

    print("=" * 100)
    print(f"汇总（{cam} gated-exit {len(gated_ex)} 条）：")
    print(f"  · 已有跨机位出边            : {n_edge}")
    print(f"  · ★ 有强候选(cos≥0.90,dt∈[-5,{a.dt_window:.0f}]s) 但**无边** : {n_strong_no_edge}  ← 可修")
    print(f"  · 无任何强候选(cos<0.85)     : {n_no_cand}  ← 数据限制")
    print(f"  · 其余（弱候选 0.85–0.90 或 dt 超窗）: {len(gated_ex)-n_edge-n_strong_no_edge-n_no_cand}")
    print()
    print("读法：n_strong_no_edge 大 → relay 参数/预算漏匹配，可通过放宽时间门或补边修；")
    print("      n_no_cand 大 → 这些人没出现在其它机位视野内，覆盖低是采集限制，不该当缺陷。")
    st.close()


if __name__ == "__main__":
    main()
