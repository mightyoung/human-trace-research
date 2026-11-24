#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F1：预算约束多对多扩展分配（Relay v2 / S1 核心）

背景（docs/relay-v2-plan.md §5 F1）：
  现行 mutual-best 一对一把"一人多次通行"误杀——179 个高 sim 候选（≥0.943）中
  132 个被淘汰（74%），已判 5 对漏接中 3 对根因于此（如 343:E042 的 7 个高分候选）。

本脚本（全部增量，不动 v5 产物）：
  Step 1 primary   —— 复用 relay_triage.mutual_best_flags（与 v5 逐位一致，内置断言）
  Step 2 secondary —— 非 mutual-best 且 sim≥thr 的候选池，按 sim/score 降序
  Step 3 预算分配  —— 贪心：K_a（exit 侧）/ K_b（entry 侧）双预算；被预算挤出的候选
                      **全部**显式记录进 conflict_groups（消除静默淘汰；S6）
  Step 3b 弱通道   —— --weak：弱事件候选（≥1 侧 moving，S6）在主候选之后填充剩余预算
                      （弱证据不争抢主预算；tier=weak 标记，audit 池）
  Step 4 输出      —— relay_v2/edges_v2.json（primary=active；secondary/weak=audit 审计池）

安全设计：primary 47 边只读复制；secondary 一律 status=audit（不进生产链）；
        冲突显式化（blocked 高分候选进 conflict_groups，供人工复核，见 webapp truth 页）。

用法:
  python tools/relay_expand.py                       # 默认参数
  python tools/relay_expand.py --k-a 3 --k-b 2       # 预算可配置
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relay_triage import mutual_best_flags  # noqa: E402  (复用，不重写)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
SCORED = os.path.join(ANA, "relay_clean2", "relay_scored_cradio_v5.json")
TRIAGE = os.path.join(ANA, "relay_clean2", "relay_triage_cradio_v5.json")
OUT = os.path.join(ANA, "relay_v2", "edges_v2.json")


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------- 核心分配（可导入，供自检） ----------------
def expand_assign(cands, sim_thr=0.943, k_a=3, k_b=2, conflict_sim=0.95, sort_key="sim",
                  weak=None):
    """预算约束多对多分配（含弱事件候选第二相）。

    cands: 主候选列表（需含 a/b/sim 字段；dt 可选）
    weak:  弱事件候选列表（可选；schema 同主候选 + score/weak_side）。弱候选在主候选
           分配**之后**按 score 填充剩余预算——弱证据不争抢主预算，仅利用余量。
    sort_key: "sim"（默认，v1 行为）| "score"（F2 联合分，候选需含 score 字段）
    返回: dict(primary, secondary, blocked, weak_accepted, used_a, used_b)
      - primary:   mutual-best 且 sim>=thr（与 v5 一致）
      - secondary: 非 mutual-best 且 sim>=thr，预算内接受
      - blocked:   **全部** sim>=thr 但被预算挤出的候选（消除静默淘汰；带 _tier 标记）
    排序键：sim → (-sim, dt)；score → (-score, dt)（确定性；负 dt 表示时间交错，合理更早）
    """
    assert sort_key in ("sim", "score")
    mb = mutual_best_flags(cands)
    primary = [r for r in cands if mb[id(r)] and r["sim"] >= sim_thr]

    pool = [r for r in cands if not mb[id(r)] and r["sim"] >= sim_thr]
    if sort_key == "score":
        assert all("score" in r for r in pool), "score 排序需要候选含 score 字段"
        pool.sort(key=lambda r: (-r["score"], r.get("dt", 0.0)))
    else:
        pool.sort(key=lambda r: (-r["sim"], r.get("dt", 0.0)))

    used_a, used_b = defaultdict(int), defaultdict(int)
    secondary, blocked = [], []
    for c in pool:
        if used_a[c["a"]] < k_a and used_b[c["b"]] < k_b:
            used_a[c["a"]] += 1
            used_b[c["b"]] += 1
            secondary.append(c)
        else:
            blocked.append(dict(c, _tier="main", _high_sim=bool(c["sim"] >= conflict_sim)))

    weak_accepted = []
    if weak:
        wp = [r for r in weak if r["sim"] >= sim_thr]
        wp.sort(key=lambda r: (-r.get("score", r["sim"]), r.get("dt", 0.0)))
        for c in wp:
            if used_a[c["a"]] < k_a and used_b[c["b"]] < k_b:
                used_a[c["a"]] += 1
                used_b[c["b"]] += 1
                weak_accepted.append(c)
            else:
                blocked.append(dict(c, _tier="weak", _high_sim=bool(c["sim"] >= conflict_sim)))
    return {"primary": primary, "secondary": secondary, "blocked": blocked,
            "weak_accepted": weak_accepted, "used_a": dict(used_a), "used_b": dict(used_b)}


def edge_record(r, edge_type, status, budget_rank=0, tier="main"):
    """规范化边记录（schema v2；w_dt/score 来自 F2，无则 null；tier: main|weak）。"""
    return {
        "a": r["a"], "b": r["b"], "dir": r["dir"], "dt": r["dt"],
        "sim": r["sim"],
        "a_h": r.get("a_h"), "b_h": r.get("b_h"),
        "a_end": r.get("a_end"), "b_start": r.get("b_start"),
        "a_tlast": r.get("a_tlast"), "b_tfirst": r.get("b_tfirst"),
        "a_frag_last": r.get("a_frag_last"), "b_frag_first": r.get("b_frag_first"),
        "edge_type": edge_type, "status": status, "budget_rank": budget_rank,
        "w_dt": r.get("w_dt"), "score": r.get("score"),
        "tier": tier,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=SCORED)
    ap.add_argument("--triage", default=TRIAGE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sim-thr", type=float, default=0.943)
    ap.add_argument("--k-a", type=int, default=3, help="exit 侧 secondary 预算")
    ap.add_argument("--k-b", type=int, default=2, help="entry 侧 secondary 预算")
    ap.add_argument("--conflict-sim", type=float, default=0.95,
                    help="被预算挤出的高分候选进入 conflict_group 的下限")
    ap.add_argument("--sort-key", choices=["sim", "score"], default="sim",
                    help="secondary 排序键：sim（v1）| score（F2 联合分，需 scored 含 score 字段）")
    ap.add_argument("--weak", default=None,
                    help="弱事件候选 scored 文件（relay_weak_candidates.py 产物；填充剩余预算）")
    ap.add_argument("--archive", default=None,
                    help="覆盖前将旧 edges_v2.json 存档到该路径（留档对照）")
    args = ap.parse_args()

    d = json.load(open(args.scored, encoding="utf-8"))
    cands = d["candidates"]

    weak_cands, weak_meta = [], None
    if args.weak:
        dw = json.load(open(args.weak, encoding="utf-8"))
        weak_cands = dw["candidates"]
        weak_meta = {"path": os.path.relpath(args.weak, ROOT).replace("\\", "/"),
                     "hash": sha16(args.weak)}

    # ---- Step 1-3：分配（主候选 + 弱通道第二相）----
    res = expand_assign(cands, args.sim_thr, args.k_a, args.k_b, args.conflict_sim,
                        sort_key=args.sort_key, weak=weak_cands)
    primary, secondary, blocked = res["primary"], res["secondary"], res["blocked"]
    weak_acc = res["weak_accepted"]

    # ---- L1 断言：primary 与 v5 triage confirmed 逐位一致 ----
    tri = json.load(open(args.triage, encoding="utf-8"))
    conf_set = set()
    for s in tri["confirmed_edges"]:
        a, b = [x.strip() for x in s.split("->")]
        conf_set.add((a, b))
    prim_set = {(r["a"], r["b"]) for r in primary}
    assert prim_set == conf_set, \
        f"[L1-FAIL] primary {len(prim_set)} != triage confirmed {len(conf_set)}; " \
        f"diff={prim_set.symmetric_difference(conf_set)}"
    print(f"[L1] primary 与 v5 triage 逐位一致（{len(primary)} 边）✓")

    # ---- Step 4：输出 ----
    primary.sort(key=lambda r: -r["sim"])   # 展示排序（不影响集合）
    secondary.sort(key=lambda r: -r["sim"])
    weak_acc.sort(key=lambda r: -r["sim"])

    # conflict groups：按 a 端聚类 blocked + 记录该 a 已接受边（供对照）
    acc_by_a = defaultdict(list)
    for r in primary:
        acc_by_a[r["a"]].append({"b": r["b"], "sim": r["sim"], "type": "primary"})
    for r in secondary:
        acc_by_a[r["a"]].append({"b": r["b"], "sim": r["sim"], "type": "secondary"})
    for r in weak_acc:
        acc_by_a[r["a"]].append({"b": r["b"], "sim": r["sim"], "type": "weak"})
    groups = defaultdict(list)
    for c in blocked:
        groups[c["a"]].append(c)

    used_a, used_b = res["used_a"], res["used_b"]
    conflict_groups = []
    for a, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (-r["sim"], r.get("dt", 0)))
        conflict_groups.append({
            "group": f"CG-{a}",
            "a": a,
            "accepted": sorted(acc_by_a.get(a, []), key=lambda x: -x["sim"]),
            "blocked": [{"b": r["b"], "sim": r["sim"], "dt": r["dt"],
                         "tier": r["_tier"], "high_sim": r["_high_sim"],
                         "reason": "budget_k_a_exceeded" if used_a.get(r["a"], 0) >= args.k_a
                         else "budget_k_b_exceeded"} for r in rows],
        })

    edges = [edge_record(r, "primary", "active", 0) for r in primary]
    for i, r in enumerate(secondary):
        edges.append(edge_record(r, "secondary", "audit", i + 1))
    for i, r in enumerate(weak_acc):
        edges.append(edge_record(r, "secondary", "audit", len(secondary) + i + 1, tier="weak"))

    out = {
        "version": "relay_v2/edges/2",
        "based_on": {"scored": os.path.relpath(args.scored, ROOT).replace("\\", "/"),
                     "hash": sha16(args.scored),
                     **({"weak": weak_meta} if weak_meta else {})},
        "params": {"sim_thr": args.sim_thr, "K_a": args.k_a, "K_b": args.k_b,
                   "conflict_sim": args.conflict_sim, "sort_key": args.sort_key,
                   "weak_channel": bool(weak_cands)},
        "edges": edges,
        "conflict_groups": conflict_groups,
        "summary": {
            "n_primary": len(primary),
            "n_secondary": len(secondary),
            "n_weak_accepted": len(weak_acc),
            "n_conflict_groups": len(conflict_groups),
            "n_blocked_in_groups": sum(len(g["blocked"]) for g in conflict_groups),
            "n_blocked_main": sum(1 for r in blocked if r["_tier"] == "main"),
            "n_blocked_weak": sum(1 for r in blocked if r["_tier"] == "weak"),
            "high_sim_total": int(sum(1 for r in cands if r["sim"] >= args.sim_thr)),
            "weak_input": len(weak_cands),
            "weak_ge_thr": int(sum(1 for r in weak_cands if r["sim"] >= args.sim_thr)),
            "sort_key": args.sort_key,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.archive and os.path.exists(args.out):
        import shutil
        shutil.copy2(args.out, args.archive)
        print(f"[archive] 旧 {os.path.basename(args.out)} -> {args.archive}")
    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---- 报告（含静默淘汰硬断言：全部 ≥thr 必须有归宿）----
    s = out["summary"]
    hs = s["high_sim_total"]
    n_silent_main = hs - len(primary) - len(secondary) - s["n_blocked_main"]
    assert n_silent_main == 0, f"[L1-FAIL] 主池 ≥thr 静默淘汰 {n_silent_main} 条"
    n_silent_weak = s["weak_ge_thr"] - len(weak_acc) - s["n_blocked_weak"]
    assert n_silent_weak == 0, f"[L1-FAIL] 弱通道 ≥thr 静默淘汰 {n_silent_weak} 条"
    print(f"[F1] 主池高 sim(≥{args.sim_thr}) {hs} = primary {len(primary)} + secondary "
          f"{len(secondary)} + blocked(conflict) {s['n_blocked_main']} + 静默 0 ✓")
    print(f"[F1] 弱通道 ≥thr {s['weak_ge_thr']}/{s['weak_input']} = accepted {len(weak_acc)} "
          f"+ blocked {s['n_blocked_weak']} + 静默 0 ✓")
    print(f"[F1] secondary 边（审计池，按 sim 降序 top20）：")
    for r in secondary[:20]:
        print(f"  {r['a']} -> {r['b']}  sim={r['sim']:.3f} dt={r['dt']:.1f}")
    if weak_acc:
        print(f"[F1] weak 通道接纳边（tier=weak，按 sim 降序 top20）：")
        for r in weak_acc[:20]:
            print(f"  {r['a']} -> {r['b']}  sim={r['sim']:.3f} dt={r['dt']:.1f} "
                  f"(weak_side={r.get('weak_side')})")
    if conflict_groups:
        print(f"[F1] conflict groups {len(conflict_groups)} 个（被预算挤出的候选，需人工复核）：")
        for g in conflict_groups[:8]:
            print(f"  {g['group']}: accepted={[(x['b'], x['sim']) for x in g['accepted'][:4]]}")
            for x in g["blocked"][:6]:
                print(f"      BLOCKED {x['b']} sim={x['sim']:.3f} dt={x['dt']:.1f} "
                      f"[{x['tier']}]{' *high*' if x['high_sim'] else ''} ({x['reason']})")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
