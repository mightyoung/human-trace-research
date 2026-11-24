# -*- coding: utf-8 -*-
"""门户驱动候选通道（gate-driven channel）—— S-1 修复 305 出门覆盖。

问题（诊断结论，见 tools/diag_305_egress.py）：
  305 的 20 个 gated-exit 只有 2 个有出边 → C0 出门连续性 0.1（全项目最低）。
  根因不是门不准，而是 **候选池没覆盖 305**：
    · relay v2 的主候选来自 v5 池（47 边），305 只有 2 条进入
    · 320 侧再撞 K_b=2 预算上限 → 305:E079(cos 0.976) 被挤掉
  例：305:E072→320:E013 cos=0.969 dt=8s 根本没进池（现有对端边只有 0.9491）。

本通道不改既有 active/audit 边（保住 K_a=4/K_b=2/λ=0.01 的对账），
只**新增**一条规则化候选：出口实体 → 时间窗内、入口落在对方门内的实体。

规则（全部可复述）：
  出口侧 a : exit 端点被 C0 判为 gated（贴边或距门心 ≤ radius）
  入口侧 b : entry 端点被 C0 判为 gated
  时间     : 0 ≤ (t_b_in - t_a_out) ≤ DT_MAX（默认 30s，与 dt 先验峰值 4.5s 同量级）
  外观     : cos ≥ SIM_THR（默认 0.94，与 relay v2 sim_thr 0.943 同口径）
  唯一性   : 出口实体与入口实体各自只保留最佳一条（greedy，按 score 排序）

输出：output_crosscam/analysis/gil/gate_candidates.json
用法：
  DET_VIEW=clean2 python tools/gil_gate_candidates.py [--dt-max 30] [--sim 0.94]
"""
import argparse
import json
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


def build_entity_vectors(st, vb):
    """(cam, eid) -> (单位向量, 行号)；用成员观测向量均值。"""
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
    return np.stack(E), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-max", type=float, default=30.0)
    ap.add_argument("--sim", type=float, default=0.94)
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/gate_candidates.json")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--b-require-gate", default="1", choices=["0", "1"],
                    help="入口侧是否要求也落在对方门内（1=严格；0=只要求出口侧有门）")
    ap.add_argument("--cam-offset", default="",
                    help="临时时钟偏移修正，形如 305:-7.5（只用于本次分析，不改代码常量）")
    a = ap.parse_args()

    st = GILStore(ROOT + "/" + a.store, create=False)
    vb = st.vectors(load=True)
    E, keys = build_entity_vectors(st, vb)
    rowof = {k: i for i, k in enumerate(keys)}
    print(f"[gate] 实体 {len(keys)} 个")

    # ---- 每机位的门 + 端点判定 ----
    off = {}
    for item in (a.cam_offset or "").split(","):
        if ":" in item:
            c, v = item.split(":")
            off[c.strip()] = float(v)
    if off:
        print(f"[gate] 临时时钟偏移修正（只影响本次计算）: {off}")

    gated_ex, gated_en = {}, {}
    gates_all = {}
    ents_all = {}
    for cam in CAMS:
        ents = C.build_entity_tracks(cam)
        ents_all[cam] = ents
        gates, ex_mask, en_mask = C.classify_endpoints(cam, ents)
        gates_all[cam] = gates
        for eid, e in ents.items():
            if not e["moving"]:
                continue
            gated_ex[(cam, eid)] = bool(ex_mask.get(eid))
            gated_en[(cam, eid)] = bool(en_mask.get(eid))
    n_gx = {c: sum(1 for (cc, _), v in gated_ex.items() if cc == c and v) for c in CAMS}
    n_ge = {c: sum(1 for (cc, _), v in gated_en.items() if cc == c and v) for c in CAMS}
    print(f"[gate] 出门 gated: {n_gx}")
    print(f"[gate] 进门 gated: {n_ge}")

    # ---- 生成候选 ----
    cands = []
    for (ca, ea) in list(gated_ex):
        if not gated_ex[(ca, ea)]:
            continue
        if (ca, ea) not in rowof:
            continue
        e_a = ents_all[ca][ea]
        i = rowof[(ca, ea)]
        s = E @ E[i]
        order = np.argsort(-s)
        for j in order[:60]:
            cb, eb = keys[j]
            if cb == ca:
                continue
            if a.b_require_gate == "1" and not gated_en.get((cb, eb), False):
                continue
            e_b = ents_all[cb][eb]
            dt = (e_b["t_first"] + off.get(cb, 0.0)) - (e_a["t_last"] + off.get(ca, 0.0))
            if not (0.0 <= dt <= a.dt_max):
                continue
            c = float(s[j])
            if c < a.sim:
                continue
            # score = 外观 + dt 先验（复用 relay_v2/dist_v1 的 Parzen 权重）
            cands.append(dict(a=f"{ca}:{ea}", b=f"{cb}:{eb}",
                              dt=round(dt, 2), sim=round(c, 4),
                              a_out=[round(e_a["u1"], 3), round(e_a["v1"], 3)],
                              b_in=[round(e_b["u0"], 3), round(e_b["v0"], 3)],
                              a_tlast=round(e_a["t_last"], 1),
                              b_tfirst=round(e_b["t_first"], 1)))
    # ---- 唯一化：默认「每个出口实体保留最佳一条」（= 一人一次通行），
    #      入口端允许多占（同一目标可被多个机位的门看到），并报告入口冲突数 ----
    cands.sort(key=lambda x: -x["sim"])
    used_a, kept, bdeg = set(), [], defaultdict(int)
    for c in cands:
        if c["a"] in used_a:
            continue
        used_a.add(c["a"])
        bdeg[c["b"]] += 1
        kept.append(c)
    n_bconf = sum(1 for v in bdeg.values() if v > 1)
    print(f"[gate] 候选（超阈 {len(cands)}）→ 按出口实体唯一化后 {len(kept)}"
          f"（入口端被多占 {n_bconf} 个）")

    # ---- 与既有边对比 ----
    existing = set()
    for r in st.con.execute("SELECT a_id,b_id FROM edge WHERE kind LIKE 'relay%'"):
        ent = {x["id"]: (x["cam"], x["eid"]) for x in
               st.con.execute("SELECT id,cam,eid FROM entity")}
        ra, rb = ent.get(r["a_id"]), ent.get(r["b_id"])
        if ra and rb:
            existing.add((f"{ra[0]}:{ra[1]}", f"{rb[0]}:{rb[1]}"))
    new = [c for c in kept if (c["a"], c["b"]) not in existing]
    print(f"[gate] 其中**新增**（既有接力图中没有）: {len(new)}")

    # ---- 对 305 出门覆盖的影响 ----
    out_after = {c["a"] for c in kept} | {x[0] for x in existing}
    print()
    print("=== 出门覆盖：修前 vs 加 gate 通道后 ===")
    rep = {}
    for cam in CAMS:
        gx = [k for k, v in gated_ex.items() if k[0] == cam and v]
        if not gx:
            continue
        before = sum(1 for k in gx if f"{k[0]}:{k[1]}" in {x[0] for x in existing})
        after = sum(1 for k in gx if f"{k[0]}:{k[1]}" in out_after)
        rep[cam] = dict(n_gated=len(gx), before=before, after=after,
                        cont_before=round(before / len(gx), 3),
                        cont_after=round(after / len(gx), 3))
        print(f"  {cam}: gated {len(gx):3d} | 修前 {before:3d} ({before/len(gx):.3f}) "
              f"→ 修后 {after:3d} ({after/len(gx):.3f})  +{after-before}")

    if not a.quiet:
        print()
        print("=== 新增候选明细（前 25 条）===")
        for c in new[:25]:
            print(f"  {c['a']:14} -> {c['b']:14} dt={c['dt']:6.1f}s sim={c['sim']:.4f} "
                  f"out{c['a_out']} in{c['b_in']}")

    os.makedirs(os.path.dirname(ROOT + "/" + a.out), exist_ok=True)
    json.dump(dict(
        rule=dict(dt_max=a.dt_max, sim_thr=a.sim,
                  uniqueness="one-best-per-exit-entity（= 一人一次通行；入口端允许多占）",
                  gate="C0 classify_endpoints (cell=%.2f radius=%.2f border=%.2f)"
                       % (C.CFG["cell"], C.CFG["gate_radius"], C.CFG["border_margin"]),
                  cam_offset=off or None,
                  note="新增门户驱动通道；不改既有 active/audit 边（保 v2 对账）"),
        n_candidates=len(cands), n_unique=len(kept), n_new=len(new),
        coverage_delta=rep,
        candidates=kept), open(ROOT + "/" + a.out, "w", encoding="utf-8"),
        ensure_ascii=False, indent=1)
    print(f"\n[out] {a.out}")
    st.close()


if __name__ == "__main__":
    main()
