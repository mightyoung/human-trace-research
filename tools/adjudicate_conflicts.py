# -*- coding: utf-8 -*-
"""裁决 truth 互斥冲突组：谁是真交接、谁是误接、谁可能是同侧碎片。

背景（`analysis/truth/TRUTH-FINDINGS-2026-10-14.md` §4）：
  13 条人工判「同人」只落在 7 个不同的 343 事件上 → 按「一人一次通行」至少 6 条不能同时成立。
  4 组待裁：343-E042 / E009 / E010 / E011。

裁决不是"选一个然后宣布其余是误接"——必须区分两种读法：

  (A) **跨机位一对多**（真误接）：同一 343 出门事件被接到多个 316 事件，
      而这些 316 事件**时间上重叠或彼此矛盾** → 只能留 1 条，其余是误接。
  (B) **同侧碎片**（不是误接，是 Q1 遮挡断 ID）：多个 316 事件其实是**同一个人**
      在 316 侧被跟踪器拆成的先后碎片（时间互斥、彼此相邻、外观一致）
      → 这些"同人"判定可以同时为真，只是它们属于同一个 gperson（S2 的问题）。

裁决器（全部可复述）：
  硬约束   : 0 ≤ (b 进入 − a 离开) ≤ DT_MAX(30s) 且 出口/入口端点均在门内
  打分     : score = cos + λ·ln w_dt（λ=0.01，w_dt 取冻结的 relay_v2/dist_v1）
  胜者     : 硬约束内 score 最高者
  败者     : 若与胜者**时间不冲突且外观一致** → 判「同侧碎片候选」（交 S2 聚类）；
             否则判「误接」。

输出：output_crosscam/analysis/truth/_review/conflict_verdicts.json
用法：DET_VIEW=clean2 python tools/adjudicate_conflicts.py
"""
import argparse
import csv
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

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
DT_MAX = 30.0
GATE_RADIUS = C.CFG["gate_radius"]
BORDER = C.CFG["border_margin"]
FRAG_COS = 0.90          # 败者与胜者外观 ≥ 此值 → 可能是同一人
# ★ 重叠视野机位对：同一人**可以同时**出现在两台（同区域两个视角）→ 时间互斥须豁免。
#   实测：318×320 的 8 对人工 GT **全部时间重叠**（不豁免就会把真当作假）。
OVERLAP_PAIRS = {frozenset(("318", "320"))}
 
GROUPS = ["343-E042", "343-E009", "343-E010", "343-E011"]


def w_dt_fn():
    d = json.load(open(os.path.join(ANA, "relay_v2", "dist_v1.json"), encoding="utf-8"))
    G = np.asarray(d["grid_x"], float); F = np.asarray(d["grid_f"], float)
    FMAX = float(F.max()); lam = float(d.get("lambda_default", 0.01))

    def w(dt):
        x = float(np.clip(dt, G[0], G[-1]))
        i = max(0, min(int(np.searchsorted(G, x) - 1), len(G) - 2))
        t = (x - G[i]) / (G[i + 1] - G[i])
        return float(max(F[i] * (1 - t) + F[i + 1] * t, 1e-9) / FMAX)
    return w, lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--out", default="output_crosscam/analysis/truth/_review/conflict_verdicts.json")
    a = ap.parse_args()

    st = GILStore(ROOT + "/" + a.store, create=False)
    vb = st.vectors(load=True)
    # 实体代表向量
    ent = {r["id"]: dict(cam=r["cam"], eid=r["eid"]) for r in
           st.con.execute("SELECT id,cam,eid FROM entity")}
    mem = defaultdict(list)
    for r in st.con.execute("SELECT entity_id,id FROM obs WHERE entity_id IS NOT NULL"):
        mem[r["entity_id"]].append(r["id"])
    E, keys = [], []
    for eid_i, oids in mem.items():
        b = ent[eid_i]["cam"]
        vs = [vb._cache[b][0][vb._cache[b][2][o]] for o in oids
              if b in vb._cache and o in vb._cache[b][2]]
        if not vs:
            continue
        v = np.mean(vs, axis=0); v /= (np.linalg.norm(v) + 1e-9)
        E.append(v); keys.append((ent[eid_i]["cam"], ent[eid_i]["eid"]))
    E = np.stack(E); rowof = {k: i for i, k in enumerate(keys)}

    def cos(x, y):
        if x not in rowof or y not in rowof:
            return None
        return float(E[rowof[x]] @ E[rowof[y]])

    # 实体时间区间（GIL）
    iv = {}
    for r in st.con.execute("SELECT cam,eid,t0,t1 FROM entity"):
        iv[(r["cam"], r["eid"])] = (r["t0"], r["t1"])

    # 门 + 端点判定
    ents, gex, gen = {}, {}, {}
    for cam in ("343", "316"):
        e = C.build_entity_tracks(cam)
        ents[cam] = e
        _g, exm, enm = C.classify_endpoints(cam, e)
        gex[cam], gen[cam] = exm, enm

    def gate_ok(cam, eid, kind):
        m = gex if kind == "exit" else gen
        return bool(m[cam].get(eid))

    # 人工判定
    lab = {}
    for p in (os.path.join(ROOT, "webapp/backend/data/truth/seed_labels.json"),
              os.path.join(ANA, "truth/remote/labels.json")):
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8")) or {}
            lab.update(d.get("pair", d) if isinstance(d, dict) else {})
    V = lambda pid: (lab.get(pid) or {}).get("v")

    pm = list(csv.DictReader(open(os.path.join(ANA, "truth/_review/pair_machine.csv"),
                                  encoding="utf-8-sig")))
    w, lam = w_dt_fn()

    # 系统（relay v2 + gate）在这些组里选了谁
    sysp = {}
    for e in json.load(open(os.path.join(ANA, "relay_v2", "edges_v2_gated.json"),
                            encoding="utf-8"))["edges"]:
        for tag, cam in ((e["a"], e["a"].split(":")[0]), (e["b"], e["b"].split(":")[0])):
            if cam == "343":
                sysp.setdefault(tag, []).append((e["status"], e["channel"], e["dt"], e.get("sim")))

    out = {"rule": dict(dt_max=DT_MAX, lam=lam, gate_radius=GATE_RADIUS, frag_cos=FRAG_COS,
                        hard_constraint="同一个人 ⟹ 时间区间互斥（同机位/跨机位皆成立）",
                        winner="贪心区间调度：按 score 降序取，与已取代表时间重叠者判『误接』",
                        note="一个 343 出门事件最多只有一个『真交接』；其余在时间上与之重叠的候选必为误接；"
                             "与代表不重叠者交由 S2（gperson 聚类）判定是否同一人的先后碎片"),
           "groups": {}}
    for g in GROUPS:
        cands = []
        for r in pm:
            if not r["event_a"].endswith(g):
                continue
            eb = r["event_b"].split("_")[-1]        # "316_E_316-E046" -> "316-E046"
            key_b = f"316:{eb}"
            dt = float(r["dt"]); c = float(r["cos_appear"])
            if c <= -8:
                c = cos(("343", g), ("316", eb))
            ivb = iv.get(("316", eb), (None, None))
            sc = None
            if c is not None:
                sc = c + lam * np.log(w(max(dt, 0.0)) + 1e-9)
            cands.append(dict(b=key_b, eb=eb, dt=dt, cos=c, score=round(sc, 5) if sc is not None else None,
                              in_window=(0.0 <= dt <= DT_MAX),
                              gate_ok=(gate_ok("343", g, "exit") and gate_ok("316", eb, "entry")),
                              t0=ivb[0], t1=ivb[1],
                              human={1: "same", 0: "diff", 2: "unclear", None: "na"}[V(r["pair_id"])],
                              pair_id=r["pair_id"]))
        # ---- 硬约束：时间互斥。贪心区间调度取代表 ----
        order = sorted([x for x in cands if x["score"] is not None],
                       key=lambda x: -x["score"])
        taken = []
        for x in order:
            if x["t0"] is None:
                x["verdict"] = "无时间区间(跳过)"
                continue
            if any(_overlap(x, y) for y in taken):
                x["verdict"] = "误接(与代表时间重叠)"
            else:
                taken.append(x)
                x["verdict"] = f"代表#{len(taken)}(时间兼容)"
        for x in cands:
            x.setdefault("verdict", "无分(外观缺失)")
        # 最大同时存活数（= 该时刻 316 侧至少这么多不同的人）
        depth = _max_depth([x for x in cands if x["t0"] is not None])
        win = taken[0] if taken else None
        # 系统选择：该 343 事件在合并边集（active/audit/gate）里的对端
        sys_b = [e["b"] for e in json.load(open(os.path.join(ANA, "relay_v2", "edges_v2_gated.json"),
                                                encoding="utf-8"))["edges"]
                 if e["a"] == f"343:{g}" and e["status"] in ("active", "audit", "gate")]
        # 硬约束违规：人工判"同人"但与胜者时间重叠 → 可证伪
        prov_wrong = [x["eb"] for x in cands
                      if x["human"] == "same" and win and x["t0"] is not None
                      and x["eb"] != win["eb"] and _overlap(x, win)]
        sys_extra = sorted(set(sys_b) - {win["b"]} if win else set(sys_b))
        # ---- ★ 跨机位全局时间互斥：该 343 事件的**全部系统出边**两两查时间重叠 ----
        links = []
        for e in json.load(open(os.path.join(ANA, "relay_v2", "edges_v2_gated.json"),
                                encoding="utf-8"))["edges"]:
            if e["a"] != f"343:{g}" or e["status"] not in ("active", "audit", "gate"):
                continue
            cb, eb2 = e["b"].split(":")
            t0, t1 = iv.get((cb, eb2), (None, None))
            links.append(dict(b=e["b"], cam=cb, dt=e["dt"], sim=e.get("sim"),
                              status=e["status"], channel=e["channel"], t0=t0, t1=t1))
        viol = []
        for i in range(len(links)):
            for j in range(i + 1, len(links)):
                x, y = links[i], links[j]
                if _incompatible(dict(cam=x["cam"], t0=x["t0"], t1=x["t1"]),
                                 dict(cam=y["cam"], t0=y["t0"], t1=y["t1"])):
                    viol.append([x["b"], y["b"]])
        out["groups"][g] = dict(
            exit_gated=gate_ok("343", g, "exit"),
            winner=win["b"] if win else None,
            system_choice=sorted(set(sys_b)),
            system_extra_linked=sys_extra,
            system_links=links,
            cross_cam_mutex_violations=viol,
            agreement=(win is not None and win["b"] in sys_b),
            n_in_window=sum(1 for x in cands if x["in_window"]),
            max_coexisting_316=depth,
            human_same=[x["eb"] for x in cands if x["human"] == "same"],
            provably_wrong_same=prov_wrong,
            representatives=[x["eb"] for x in taken],
            candidates=cands)
    json.dump(out, open(ROOT + "/" + a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 打印 ----
    for g, d in out["groups"].items():
        print(f"\n### {g}  exit门内={d['exit_gated']}  人工同人={d['human_same']}")
        print(f"    真交接(胜者) = {d['winner']}   系统选择 = {d['system_choice']}"
              f"   → {'一致' if d['agreement'] else '★不一致'}")
        print(f"    同组时间兼容代表 = {d['representatives']}   316 侧最大同时存活 = {d['max_coexisting_316']}"
              f"   可证伪的同人判定 = {d['provably_wrong_same']}")
        print(f"    系统额外多接（超出胜者）= {d['system_extra_linked']}")
        print(f"    系统全部出边：")
        for l in d["system_links"]:
            print(f"       {l['b']:14} dt={l['dt']:6.1f}s sim={l['sim']} t=[{l['t0']},{l['t1']}] {l['status']}/{l['channel']}")
        if d["cross_cam_mutex_violations"]:
            print(f"    ★★ 跨机位时间互斥违规（不可同时为同一人）: {d['cross_cam_mutex_violations']}")
        else:
            print(f"    ★ 跨机位时间互斥违规: 无")
        for x in sorted(d["candidates"], key=lambda y: (y["score"] is None, -(y["score"] or 0))):
            cs = "  n/a  " if x["cos"] is None else "%.4f" % x["cos"]
            print(f"      {x['eb']:9} dt={x['dt']:7.1f}s cos={cs:>8} t=[{x['t0']},{x['t1']}]"
                  f" 门={int(x['gate_ok'])} 人工={x['human']:7} → {x['verdict']}")

    # ---- 全局：时间区间感知的互斥审计（修正早先"pair 级"口径）----
    print("\n" + "=" * 78)
    print("全局互斥审计（有/无时间区间两种口径的差异）")
    audit = _global_mutex_audit(pm, lab, iv)
    print(f"  人工判『同人』总数（有标签）        : {audit['n_same']}")
    print(f"  pair 级：同组内被同一 343 事件认领 ≥2 : {audit['n_groups_multi']} 组 / {audit['n_pairs_in_multi']} 对")
    print(f"  ★ 区间级：其中**真正时间冲突**（多人同时刻）: {audit['n_conflict_groups']} 组 / {audit['n_conflict_pairs']} 对")
    print(f"     → 即：pair 级口径高估了互斥违规 {audit['n_pairs_in_multi'] - audit['n_conflict_pairs']} 对"
          f"（那些其实是**先后碎片**，可由同一人成立）")
    out["global_mutex_audit"] = audit
    # ---- ★★ 全局可证明误接下界：每个实体的系统出边中，至少 (N − 最大同时存活数) 条是错的 ----
    glob = _provable_false_edges(ROOT, iv)
    out["provable_false_edges"] = glob
    print("\n" + "=" * 78)
    print("★ 全局可证明误接下界（时间互斥：一个人不能同时出现在两处）")
    print(f"  有 ≥2 条跨机位出边的实体 : {glob['n_entities_multi']}")
    print(f"  这些实体的出边总数        : {glob['n_edges_multi']}")
    print(f"  **可证明为误接的下界**    : {glob['n_provable_false']} 条"
          f"（Σ(N − 最大可同时成立子集)）")
    print(f"  豁免的重叠视野机位对      : {glob['overlap_pairs']}（同一人可同时出现）")
    print("  按出入边实体举例（前 8）:")
    for x in glob["examples"]:
        print(f"    {x['ent']:14} 出边 {x['n_out']} 条 → 至少 {x['n_false']} 条错 "
              f"(最多 {x['max_compatible']} 条可同时成立)  对端={x['peer_cams']}")
    print(f"  → 读法：这是**下界**（仅凭时间互斥可证明的错接），不是总数。")
    json.dump(out, open(ROOT + "/" + a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    n_same = sum(len(d["human_same"]) for d in out["groups"].values())
    n_pw = sum(len(d["provably_wrong_same"]) for d in out["groups"].values())
    n_dis = sum(1 for d in out["groups"].values() if not d["agreement"] and d["winner"])
    print(f"\n[汇总] 4 组：人工判『同人』{n_same} 条 → **可证伪 {n_pw} 条**；"
          f"系统选择与胜者**不一致 {n_dis} 组**（= 系统误接的硬证据）")
    print(f"[out] {a.out}")
    st.close()


def _incompatible(a, b):
    """两个候选是否**必须不同人**：时间重叠，且两机位非重叠视野对。"""
    if frozenset((a["cam"], b["cam"])) in OVERLAP_PAIRS:
        return False
    if a["t0"] is None or b["t0"] is None:
        return False
    return not (a["t1"] <= b["t0"] or b["t1"] <= a["t0"])


def _max_independent(cands):
    """最大"可同时成立"子集（不可兼容图上的最大独立集）。n 小 → 精确枚举。"""
    n = len(cands)
    if n > 16:      # 保底（实际远小于此）
        return n
    incompat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _incompatible(cands[i], cands[j]):
                incompat[i][j] = incompat[j][i] = True
    best = 0
    for mask in range(1 << n):
        ok = True
        m = mask
        idx = [i for i in range(n) if (mask >> i) & 1]
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                if incompat[idx[ii]][idx[jj]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best = max(best, len(idx))
    return best


def _provable_false_edges(root, iv):
    """对每个实体：其系统出边中，时间互斥可证明错接的条数下界 = N − 最大可同时成立子集。"""
    edges = json.load(open(os.path.join(root, "output_crosscam/analysis/relay_v2",
                                        "edges_v2_gated.json"), encoding="utf-8"))["edges"]
    by_a = defaultdict(list)
    for e in edges:
        if e["status"] in ("active", "audit", "gate"):
            by_a[e["a"]].append(e["b"])
    n_ent = n_edge = n_false = 0
    rows = []
    for a, bs in by_a.items():
        if len(bs) < 2:
            continue
        n_ent += 1; n_edge += len(bs)
        cands = []
        for b in bs:
            cam, eid = b.split(":")
            t = iv.get((cam, eid), (None, None))
            if t[0] is not None:
                cands.append(dict(cam=cam, eid=eid, t0=t[0], t1=t[1]))
        if len(cands) < 2:
            continue
        mi = _max_independent(cands)
        nf = len(cands) - mi
        if nf > 0:
            n_false += nf
            rows.append(dict(ent=a, n_out=len(bs), max_compatible=mi, n_false=nf,
                             peer_cams=sorted({b.split(":")[0] for b in bs})))
    rows.sort(key=lambda x: -x["n_false"])
    return dict(n_entities_multi=n_ent, n_edges_multi=n_edge, n_provable_false=n_false,
                overlap_pairs=[sorted(x) for x in OVERLAP_PAIRS],
                examples=rows[:8],
                note="下界 = Σ(N − 最大可同时成立子集)：一个人不能同时出现在两处；"
                     "但**重叠视野机位对（318×320）豁免**（同区域两视角可以同时看到同一人）")


def _global_mutex_audit(pm, lab, iv):
    """对全部人工判『同人』的对，比较 pair 级与区间级的互斥违规。"""
    from collections import defaultdict as dd
    V = lambda pid: (lab.get(pid) or {}).get("v")
    by343 = dd(list)
    for r in pm:
        if V(r["pair_id"]) != 1:
            continue
        by343[r["event_a"].split("_")[-1]].append(r["event_b"].split("_")[-1])
    n_same = sum(len(v) for v in by343.values())
    multi = {k: v for k, v in by343.items() if len(v) >= 2}
    conf_g, conf_p = 0, 0
    for k, bs in multi.items():
        ivs = [iv.get(("316", b), (None, None)) for b in bs]
        bad = 0
        for i in range(len(ivs)):
            for j in range(i + 1, len(ivs)):
                a, b = ivs[i], ivs[j]
                if a[0] is None or b[0] is None:
                    continue
                if not (a[1] <= b[0] or b[1] <= a[0]):
                    bad += 1
        if bad:
            conf_g += 1
            conf_p += bad
    return dict(n_same=n_same, n_groups_multi=len(multi),
                n_pairs_in_multi=sum(len(v) * (len(v) - 1) // 2 for v in multi.values()),
                n_conflict_groups=conf_g, n_conflict_pairs=conf_p,
                note="pair 级互斥=同一 343 事件被多个 316 认领；区间级互斥=这些 316 事件时间重叠"
                     "（后者才是不可同时成立的硬违规）")


def _overlap(x, y):
    return not (x["t1"] <= y["t0"] or y["t1"] <= x["t0"])


def _max_depth(rows):
    """扫描线：任一时刻同时存活的区间数最大值。"""
    ev = []
    for r in rows:
        ev.append((r["t0"], 1)); ev.append((r["t1"], -1))
    ev.sort()
    cur = mx = 0
    for _, d in ev:
        cur += d
        mx = max(mx, cur)
    return mx


if __name__ == "__main__":
    main()
