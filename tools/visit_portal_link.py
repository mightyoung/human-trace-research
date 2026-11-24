#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""visit 门户语义 → relay 状态机（S9：把 entry/exit 事件接进接力证据链）。

背景（缺口 WP2「门户语义发现 × H2」）：
  现有 relay 边的 dt/门户位置全部来自 **entity 包络**（a_tlast/b_tfirst、a_end/b_start）。
  GIL 的 `visit` 表另有**独立**证据：每个 entity 在每个机位的 entry/exit 事件
  （穿越时刻 t + 穿越位置 (u,v) + 朝向 dir + 边缘余量 edge_margin）。

本实验回答：用 visit 的**真实门户穿越事件**替代 entity 包络，能否更好地判别 relay 真假边？

做法（只读，不写任何生产文件）：
  ① visit → per (cam, entity_id) 的 entry/exit 事件表
  ② 每条 relay 边 a(camA:eidA)→b(camB:eidB)：
     - exit_a = camA 上最贴近 a 端结束时刻的 exit 事件；entry_b = camB 上最贴近 b 端起始时刻的 entry 事件
     - dt_visit = entry_b.t − exit_a.t（「b 进入 − a 离开」项目口径）
     - 门户归属：最近邻命中 C0 gates（exit / entry 各自），GATE_R=0.15
  ③ 评估（人工强标签：rejected=假 / confirmed+active=真）：
     - rank AUC：真边的 |dt| 是否显著小于假边（dt_visit vs dt_entity 对照）
     - 门户命中率：两端各自命中 C0 门的比例（真假边对照）
     - 门户对常见度：该 (g_a→g_b) 在全部候选中的出现次数（真边应更常见）

输出：output_crosscam/analysis/relay_v2/visit_portal_link_v1.json
用法：python tools/visit_portal_link.py
"""
import collections
import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
GIL = os.path.join(ANA, "gil", "store_v1", "gil.sqlite")
EDGES = os.path.join(ANA, "relay_v2", "edges_v2_gated_human.json")
C0 = os.path.join(ANA, "c0", "c0_baseline_v8.json")
SCORED = os.path.join(ANA, "relay_v2", "scored_v2.json")
OUT = os.path.join(ANA, "relay_v2", "visit_portal_link_v1.json")
GATE_R = 0.15          # 门命中半径（同 eval4 L1 口径）
TRUE_ST = ("confirmed", "active")
FALSE_ST = ("rejected",)


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def nearest(pt, gates, r=GATE_R):
    best, bi = None, None
    for i, g in enumerate(gates):
        dd = dist(pt, g)
        if best is None or dd < best:
            best, bi = dd, i
    return bi if (best is not None and best <= r) else None


def rank_auc(pos, neg):
    """P(正样本值 < 负样本值) 的曼-惠特尼估计（可用 ties=0.5）。"""
    if not pos or not neg:
        return None
    w = 0.0
    for a in pos:
        for b in neg:
            w += 1.0 if a < b else 0.5 if a == b else 0.0
    return w / (len(pos) * len(neg))


def main():
    con = sqlite3.connect(GIL)
    con.row_factory = sqlite3.Row

    # ---- ① visit 事件表 ----
    visits = collections.defaultdict(list)      # (cam, entity_id) -> [row]
    for r in con.execute("SELECT cam,kind,t,u,v,dir,edge_margin,entity_id,gperson_id "
                         "FROM visit WHERE entity_id IS NOT NULL"):
        visits[(r["cam"], r["entity_id"])].append(dict(r))
    ent = {r["id"]: dict(r) for r in con.execute("SELECT id,cam,eid,t0,t1 FROM entity")}
    eid2id = {(r["cam"], r["eid"]): r["id"] for r in con.execute("SELECT id,cam,eid FROM entity")}
    print(f"[visit] 事件 {sum(len(v) for v in visits.values())} 条，覆盖 "
          f"{len(visits)} 个 (cam,entity)；entity {len(ent)}")

    # ---- C0 门 ----
    g0 = json.load(open(C0, encoding="utf-8"))["gates"]
    gates = {c: dict(exit=[g["center"] for g in lst if g.get("kind") == "exit"],
                     entry=[g["center"] for g in lst if g.get("kind") == "entry"])
             for c, lst in g0.items()}

    # ---- ② 逐 relay 边取 visit 证据 ----
    ed = json.load(open(EDGES, encoding="utf-8"))
    edges = ed["edges"]

    def pick(cam, eid, kind, target_t):
        """取该 entity 在 cam 上最贴近 target_t 的 entry/exit 事件。"""
        i = eid2id.get((cam, eid))
        if i is None:
            return None, "no_entity"
        evs = [v for v in visits.get((cam, i), []) if v["kind"] == kind]
        if not evs:
            return None, "no_event"
        best = min(evs, key=lambda v: abs(v["t"] - target_t))
        return best, None

    rows, miss = [], collections.Counter()
    for e in edges:
        ca, eida = e["a"].split(":")
        cb, eidb = e["b"].split(":")
        ia, ib = eid2id.get((ca, eida)), eid2id.get((cb, eidb))
        ta = (ent.get(ia, {}) or {}).get("t1", e.get("a_tlast"))
        tb = (ent.get(ib, {}) or {}).get("t0", e.get("b_tfirst"))
        ex, r1 = pick(ca, eida, "exit", ta if ta is not None else e.get("a_tlast", 0))
        en, r2 = pick(cb, eidb, "entry", tb if tb is not None else e.get("b_tfirst", 0))
        if r1:
            miss[("a", r1)] += 1
        if r2:
            miss[("b", r2)] += 1
        rec = dict(a=e["a"], b=e["b"], status=e["status"], dt_entity=e.get("dt"),
                   portal_exit=e.get("a_end"), portal_entry=e.get("b_start"))
        # ★ entity 口径门户命中：对全部边都算（用于公平对照）
        pa = nearest(e.get("a_end") or [9, 9], gates.get(ca, {}).get("exit", []))
        pb = nearest(e.get("b_start") or [9, 9], gates.get(cb, {}).get("entry", []))
        rec["gate_exit_entity"], rec["gate_entry_entity"] = pa, pb
        if ex and en:
            rec["dt_visit"] = round(en["t"] - ex["t"], 2)
            rec["exit_ev"] = dict(t=ex["t"], uv=[ex["u"], ex["v"]], dir=ex["dir"],
                                  margin=ex["edge_margin"])
            rec["entry_ev"] = dict(t=en["t"], uv=[en["u"], en["v"]], dir=en["dir"],
                                   margin=en["edge_margin"])
            ga = nearest([ex["u"], ex["v"]], gates.get(ca, {}).get("exit", []))
            gb = nearest([en["u"], en["v"]], gates.get(cb, {}).get("entry", []))
            rec["gate_exit"], rec["gate_entry"] = ga, gb
            rec["gate_pair"] = f"{ca}#{ga}->{cb}#{gb}" if (ga is not None and gb is not None) else None
        rows.append(rec)

    n_ok = sum(1 for r in rows if "dt_visit" in r)
    print(f"[edge] {len(rows)} 条边；拿到 visit 双端证据 {n_ok}；缺失原因 {dict(miss)}")

    # 门户对常见度（全候选池）
    sc = json.load(open(SCORED, encoding="utf-8"))["candidates"]
    pair_cnt = collections.Counter()
    for c in sc:
        ca, cb = c["a"].split(":")[0], c["b"].split(":")[0]
        ga = nearest(c.get("a_end") or [9, 9], gates.get(ca, {}).get("exit", []))
        gb = nearest(c.get("b_start") or [9, 9], gates.get(cb, {}).get("entry", []))
        if ga is not None and gb is not None:
            pair_cnt[f"{ca}#{ga}->{cb}#{gb}"] += 1
    for r in rows:
        r["gate_pair_cnt"] = pair_cnt.get(r.get("gate_pair"), 0) if r.get("gate_pair") else None

    # ---- ③ 评估 ----
    def collect(key, sts):
        return [abs(r[key]) for r in rows if r.get("status") in sts and isinstance(r.get(key), (int, float))
                and key in r]

    ev = {}
    for key in ("dt_visit", "dt_entity"):
        pos, neg = collect(key, TRUE_ST), collect(key, FALSE_ST)
        ev[key] = dict(n_true=len(pos), n_false=len(neg),
                       auc=round(rank_auc(pos, neg), 4) if rank_auc(pos, neg) is not None else None,
                       med_true=round(sorted(pos)[len(pos) // 2], 2) if pos else None,
                       med_false=round(sorted(neg)[len(neg) // 2], 2) if neg else None)
    # 门户命中率（visit 口径 vs entity 口径）
    def hit_rate(key, sts):
        sub = [r for r in rows if r.get("status") in sts]
        if not sub:
            return None
        if key.startswith("gate_exit"):
            k = key
        else:
            k = key
        n = sum(1 for r in sub if r.get(k) is not None)
        return round(n / len(sub), 4)
    portal = {}
    for k in ("gate_exit", "gate_entry", "gate_exit_entity", "gate_entry_entity"):
        portal[k] = dict(true=hit_rate(k, TRUE_ST), false=hit_rate(k, FALSE_ST),
                         n_true=len([r for r in rows if r["status"] in TRUE_ST]),
                         n_false=len([r for r in rows if r["status"] in FALSE_ST]))
    # ★ 同子集对照：只在"两端都有 visit 事件"的边上，比 visit 口径 vs entity 口径
    sub_rows = [r for r in rows if "dt_visit" in r]
    portal_sub = {}
    for k in ("gate_exit", "gate_entry", "gate_exit_entity", "gate_entry_entity"):
        st = [r for r in sub_rows if r["status"] in TRUE_ST]
        sf = [r for r in sub_rows if r["status"] in FALSE_ST]
        portal_sub[k] = dict(
            true=round(sum(1 for r in st if r.get(k) is not None) / len(st), 4) if st else None,
            false=round(sum(1 for r in sf if r.get(k) is not None) / len(sf), 4) if sf else None,
            n_true=len(st), n_false=len(sf))
    # 门户对常见度
    cnt_true = [r["gate_pair_cnt"] for r in rows if r["status"] in TRUE_ST and r.get("gate_pair_cnt") is not None]
    cnt_false = [r["gate_pair_cnt"] for r in rows if r["status"] in FALSE_ST and r.get("gate_pair_cnt") is not None]
    ev["gate_pair_cnt"] = dict(n_true=len(cnt_true), n_false=len(cnt_false),
                               auc=round(rank_auc(cnt_true, cnt_false), 4) if (cnt_true and cnt_false) else None,
                               med_true=sorted(cnt_true)[len(cnt_true) // 2] if cnt_true else None,
                               med_false=sorted(cnt_false)[len(cnt_false) // 2] if cnt_false else None)

    # ---- ④ 门户语义画像：visit 事件 → C0 几何门（覆盖 + 每门流量 + 偏差）----
    prof = {}
    for cam in sorted(gates):
        ex_pts, en_pts = [], []
        for (c, _i), vs in visits.items():
            if c != cam:
                continue
            for v in vs:
                (ex_pts if v["kind"] == "exit" else en_pts).append((v["u"], v["v"]))

        def profile(pts, gs):
            cnt, ds, off = collections.Counter(), [], 0
            for p in pts:
                gi = nearest(p, gs)
                if gi is None:
                    off += 1
                else:
                    cnt[gi] += 1
                    ds.append(dist(p, gs[gi]))
            return dict(n=len(pts), hit=int(sum(cnt.values())), off=off,
                        per_gate={str(k): v for k, v in sorted(cnt.items())},
                        med_dist=round(sorted(ds)[len(ds) // 2], 4) if ds else None)

        prof[cam] = dict(exit=profile(ex_pts, gates[cam]["exit"]),
                         entry=profile(en_pts, gates[cam]["entry"]),
                         n_gates_exit=len(gates[cam]["exit"]),
                         n_gates_entry=len(gates[cam]["entry"]))

    out = dict(version="visit_portal_link_v1", edges=EDGES, c0=C0,
               gate_r=GATE_R, n_edges=len(rows), n_with_visit=n_ok,
               eval=ev, portal_hit=portal, portal_hit_visit_subset=portal_sub,
               portal_profile=prof,
               miss={str(k): v for k, v in miss.items()},
               rows_with_visit=[r for r in rows if "dt_visit" in r],
               note="只读；visit entry/exit 事件作为独立门户证据，与 entity 包络口径对照")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n=== rank AUC（|dt|：真边应更小）===")
    for k, v in ev.items():
        if k == "gate_pair_cnt":
            print(f"  {k:12s} n_true={v['n_true']} n_false={v['n_false']} "
                  f"AUC={v['auc']} 中位 真={v['med_true']} 假={v['med_false']}")
        else:
            print(f"  {k:12s} n_true={v['n_true']} n_false={v['n_false']} "
                  f"AUC={v['auc']} 中位|dt| 真={v['med_true']} 假={v['med_false']}")
    print("\n=== 门户命中率（全 196 边）===")
    for k, v in portal.items():
        print(f"  {k:20s} 真={v['true']} (n={v['n_true']}) 假={v['false']} (n={v['n_false']})")
    print("\n=== 门户命中率（仅两端有 visit 的子集，n=?）===")
    for k, v in portal_sub.items():
        print(f"  {k:20s} 真={v['true']} (n={v['n_true']}) 假={v['false']} (n={v['n_false']})")
    print("\n=== 门户语义画像（visit 事件 → C0 几何门）===")
    for cam, p in prof.items():
        e, n = p["exit"], p["entry"]
        print(f"  {cam}: exit n={e['n']} 命中={e['hit']} 未命中={e['off']} 门分布={e['per_gate']} "
              f"中位偏差={e['med_dist']} | entry n={n['n']} 命中={n['hit']} 未命中={n['off']}")
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
