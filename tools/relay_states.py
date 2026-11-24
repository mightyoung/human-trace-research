#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F4：图边状态机 + 延迟提交（Relay v2 / S4）

机制（docs/relay-v2-plan.md §5 F4）：
  四态：candidate（默认）→ active（primary / 复核通过 / 自动高置信晋升）
        → suspect（冲突/反证）→ dormant（逾期休眠；TG-06：不判死，保留再激活）
  证据分 E（Sentinel 式正下界）：E0=0.5；支持 +；冲突 −；clamp [0.05, 1.0]
  证据源（本轮实施；权重经 primary 分布校准，见 params）：
    human_pass  +0.15   人工判定 CSV（真值级）
    human_fail  −0.30   人工判定 CSV（→ 强制 suspect，永不 active）
    sim_p50     +0.10   sim ≥ primary 边 sim 中位数（"外观不弱于半数生产边"）
    dt_peak     +0.10   dt ∈ [−2, primary dt p75]（"时序落在生产边常见带"）
  晋升：candidate 且 E ≥ 0.7 且无 human_fail → active
        （≥0.75 含人工证据 → "human-assisted"；纯自动 0.70 → "auto"）
  实例化输入：edges_v2（primary + secondary/weak）∪ blocked（已判 或 主池高置信）∪
              triage pending(13) ∪ 判定合并（labeled.csv ∪ labels.json；含池外负例重建）
  弱事件层（tier=weak，S6）：晋升须含人工证据（require_human）——自动证据不单独放行。

用法:
  python tools/relay_states.py              # 生成 edge_states.json
  python tools/relay_states.py --selftest   # L0 状态迁移断言（6 案例）
输出:
  output_crosscam/analysis/relay_v2/edge_states.json
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
RV2 = os.path.join(ANA, "relay_v2")
EDGES_V2 = os.path.join(RV2, "edges_v2.json")
SCORED_V2 = os.path.join(RV2, "scored_v2.json")
TRIAGE_V5 = os.path.join(ANA, "relay_clean2", "relay_triage_cradio_v5.json")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
WEAK_SCORED = os.path.join(RV2, "weak_scored.json")
OUT = os.path.join(RV2, "edge_states.json")

# ---- 证据权重与门槛（初值；依据 primary 分布与 19 判定设计，见报告 §S4） ----
E0 = 0.5
PROMOTE_THR = 0.7
CLAMP = (0.05, 1.0)
W = {"human_pass": +0.15, "human_fail": -0.30, "sim_p50": +0.10, "dt_peak": +0.10}
DORMANT_DAYS = 14      # suspect → dormant 的审计逾期（天）；本轮不调度，仅机制


def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def apply_evidence(evidences, promote_thr=PROMOTE_THR, is_primary=False, require_human=False):
    """状态迁移核心（可测）：给定证据列表 → (state, E, promotion)。

    evidences: [{"src": ..., "val": float, "ref": ...}]
    规则：E = clamp(E0 + Σval)；human_fail 硬降级 suspect；is_primary 直通 active；
          candidate 且 E ≥ thr → active。
    require_human（弱事件层，S6）：自动证据足够但无人工证据时不晋升（延迟提交）——
          非移动/短碎片事件的自动证据未经校准，须人工确认后才可 active。
    """
    E = E0 + sum(ev["val"] for ev in evidences)
    E = max(CLAMP[0], min(CLAMP[1], E))
    hard_fail = any(ev["src"] == "human_fail" for ev in evidences)
    has_human_pass = any(ev["src"] == "human_pass" for ev in evidences)
    if hard_fail:
        return "suspect", E, "demoted"
    if is_primary:
        return "active", E, "primary"
    if E >= promote_thr:
        if require_human and not has_human_pass:
            return "candidate", E, None
        return "active", E, ("human-assisted" if has_human_pass else "auto")
    return "candidate", E, None


def check_dormant(state, age_days, dormant_days=DORMANT_DAYS):
    """TG-06：suspect 逾期无复验 → dormant（不判死，保留再激活）。"""
    if state == "suspect" and age_days >= dormant_days:
        return "dormant"
    return state


def load_edges(path):
    d = json.load(open(path, encoding="utf-8"))
    return d, d["edges"]


def load_judged():
    """合并判定（labeled.csv ∪ labels.json，后者优先）。{pair_id: '1'|'0'|'?'}

    ★ 命名空间过滤（防重跑崩溃/污染）：labels.json 与用户判读共用 pair 命名空间，
      含 S6B_P_/PRT_ 前缀的标准 pair（剥离前缀后并入）与 S6B_B_/S6B_R_ 等实体级
      键（跳过——无 '__' 或格式不符会在 parse_pid 处崩溃）。"""
    judged = {}
    if os.path.exists(LABELED):
        for r in csv.DictReader(open(LABELED, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v in ("1", "0", "?"):
                judged[r["pair_id"]] = v
    if os.path.exists(LABELS_JSON):
        d = json.load(open(LABELS_JSON, encoding="utf-8"))
        for pid, rec in d.get("pair", {}).items():
            p2 = pid
            for pref in ("S6B_P_", "PRT_"):
                if p2.startswith(pref):
                    p2 = p2[len(pref):]
                    break
            if "__" not in p2:
                continue
            la, lb = p2.split("__", 1)[0].split("_"), p2.split("__", 1)[1].split("_")
            if len(la) < 3 or len(lb) < 3:
                continue
            judged[p2] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
    return judged


def build_states():
    """实例化：载入全部输入 → 证据注入 → 迁移 → 状态表。"""
    v2d, edges = load_edges(EDGES_V2)
    triage = json.load(open(TRIAGE_V5, encoding="utf-8"))
    judged = load_judged()
    judged_keys = {parse_pid(pid) for pid in judged}

    prim = [e for e in edges if e["edge_type"] == "primary"]
    sec = [e for e in edges if e["edge_type"] == "secondary"]
    # 门槛（自 primary 分布动态计算）
    sim_p50 = float(np.percentile([e["sim"] for e in prim], 50))
    dt_hi = float(np.percentile([e["dt"] for e in prim], 75))
    dt_band = (-2.0, round(dt_hi, 1))
    params = {"E0": E0, "promote_thr": PROMOTE_THR, "clamp": list(CLAMP),
              "w": W, "sim_thr": round(sim_p50, 4), "dt_band": list(dt_band),
              "dormant_days": DORMANT_DAYS}

    # 弱事件 sim 表（gap 注释用；只读）
    weak_sim = {}
    if os.path.exists(WEAK_SCORED):
        dw = json.load(open(WEAK_SCORED, encoding="utf-8"))
        weak_sim = {(r["a"], r["b"]): r["sim"] for r in dw["candidates"]}

    records = {}   # key "a->b" -> rec

    def base_rec(a, b, state, source):
        return {"key": f"{a}->{b}", "state": state, "source": source,
                "E": None, "promotion": None, "evidence": [], "note": ""}

    # ---- 集合构建 ----
    for e in prim:
        records[e["a"] + "->" + e["b"]] = base_rec(e["a"], e["b"], "active", "primary")
        records[e["a"] + "->" + e["b"]]["_edge"] = e
        records[e["a"] + "->" + e["b"]]["tier"] = "main"
    for e in sec:
        k = e["a"] + "->" + e["b"]
        records[k] = base_rec(e["a"], e["b"], "candidate", "secondary")
        records[k]["_edge"] = e
        records[k]["tier"] = e.get("tier", "main")
    pend_keys = set()
    for p in triage.get("pending", []):
        k = p["a"] + "->" + p["b"]
        if k not in records:
            records[k] = base_rec(p["a"], p["b"], "candidate", "pending")
            records[k]["_edge"] = {"a": p["a"], "b": p["b"], "sim": p["sim"], "dt": p["dt"]}
            records[k]["tier"] = "main"
            pend_keys.add(k)
        else:
            records[k]["note"] += " | 亦在 triage.pending"
            pend_keys.add(k)

    # ---- blocked（预算挤出，S6 全量显式化）：纳入 F4 = 已判 或 主池高置信 ----
    n_blocked_added = 0
    for g in v2d.get("conflict_groups", []):
        for x in g["blocked"]:
            a, b = g["a"], x["b"]
            k = f"{a}->{b}"
            if k in records:
                continue
            if not (((a, b) in judged_keys) or
                    (x.get("high_sim") and x.get("tier") == "main")):
                continue
            rec = base_rec(a, b, "candidate", "blocked")
            rec["_edge"] = {"a": a, "b": b, "sim": x.get("sim"), "dt": x.get("dt")}
            rec["tier"] = x.get("tier", "main")
            rec["note"] = "预算挤出（显式记录，待复核/容量释放）"
            records[k] = rec
            n_blocked_added += 1

    # ---- 自动证据（sim/dt）注入 ----
    for k, rec in records.items():
        e = rec["_edge"]
        if e.get("sim") is not None and e["sim"] >= sim_p50:
            rec["evidence"].append({"src": "sim_p50", "val": W["sim_p50"],
                                    "ref": f"sim={e['sim']:.4f}>=p50({sim_p50:.4f})"})
        if e.get("dt") is not None and dt_band[0] <= e["dt"] <= dt_band[1]:
            rec["evidence"].append({"src": "dt_peak", "val": W["dt_peak"],
                                    "ref": f"dt={e['dt']:.1f} in {dt_band}"})

    # ---- 人工证据注入（合并判定） ----
    gaps = []
    for pid, v in judged.items():
        a, b = parse_pid(pid)
        k = f"{a}->{b}"
        ref = f"labeled:{pid}"
        if k in records:
            if v == "1":
                records[k]["evidence"].append({"src": "human_pass", "val": W["human_pass"], "ref": ref})
            elif v == "0":
                records[k]["evidence"].append({"src": "human_fail", "val": W["human_fail"], "ref": ref})
        else:
            if v == "0":
                # 池外负例（低于阈值/未生成）→ 重建 suspect 留痕
                rec = base_rec(a, b, "candidate", "rebuilt")
                rec["_edge"] = {"a": a, "b": b, "sim": weak_sim.get((a, b)), "dt": None}
                rec["tier"] = "main"
                rec["note"] = "池外（低于阈值/未生成）；按人工否决重建为 suspect 留痕"
                rec["evidence"].append({"src": "human_fail", "val": W["human_fail"], "ref": ref})
                records[k] = rec
            elif v == "1":
                ws = weak_sim.get((a, b))
                note = (f"补池已含（weak sim={ws:.4f} < thr，外观证据不足）"
                        if ws is not None else "候选生成未覆盖（需查事件/窗口根因）")
                gaps.append({"pair": f"{a}<->{b}",
                             "kind": ("below_threshold" if ws is not None
                                      else "candidate_generation_miss"),
                             "note": note})

    # ---- 迁移 ----
    for k, rec in records.items():
        is_prim = rec["source"] == "primary"
        is_weak = rec.get("tier") == "weak"
        state, E, promotion = apply_evidence(rec["evidence"], is_primary=is_prim,
                                             require_human=is_weak)
        rec["state"] = state
        rec["E"] = round(E, 3)
        rec["promotion"] = promotion
        del rec["_edge"]

    return records, params, gaps, n_blocked_added


def selftest():
    FAILS = []

    def chk(name, got, want):
        ok = got == want
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got} (want {want})")
        if not ok:
            FAILS.append(name)

    print("=== Relay v2 / F4 状态机 L0 自检 ===")
    # T1: 双自动支持（0.5+0.1+0.1=0.7）→ active/auto
    st, E, pr = apply_evidence([{"src": "sim_p50", "val": 0.10}, {"src": "dt_peak", "val": 0.10}])
    chk("T1 candidate+2自动支持", (st, round(E, 2), pr), ("active", 0.7, "auto"))
    # T2: 人工否决 → suspect（且 E 显著降）
    st, E, pr = apply_evidence([{"src": "human_fail", "val": -0.30}])
    chk("T2 人工否决→suspect", (st, round(E, 2)), ("suspect", 0.2))
    # T3: 单支持（0.6）→ 仍 candidate（延迟提交：证据不足不冒进）
    st, E, pr = apply_evidence([{"src": "sim_p50", "val": 0.10}])
    chk("T3 单支持停留 candidate", (st, round(E, 2)), ("candidate", 0.6))
    # T4: primary 直通 active
    st, E, pr = apply_evidence([], is_primary=True)
    chk("T4 primary 直通", (st, pr), ("active", "primary"))
    # T5: 人工通过 + 时序支持 = 0.75 → active/human-assisted
    st, E, pr = apply_evidence([{"src": "human_pass", "val": 0.15}, {"src": "dt_peak", "val": 0.10}])
    chk("T5 人工+自动→human-assisted", (st, round(E, 2), pr), ("active", 0.75, "human-assisted"))
    # T6: clamp 双向 + dormant 机制
    _, Ehi, _ = apply_evidence([{"src": "sim_p50", "val": 9.0}])
    _, Elo, _ = apply_evidence([{"src": "human_fail", "val": -9.0}])
    chk("T6a clamp 上界", round(Ehi, 2), 1.0)
    chk("T6b clamp 下界", round(Elo, 2), 0.05)
    chk("T6c suspect 逾期→dormant", check_dormant("suspect", 15), "dormant")
    chk("T6d suspect 未逾期保持", check_dormant("suspect", 3), "suspect")
    # T7: 弱事件层 require_human —— 自动证据足够但不晋升；含人工证据才晋升
    st, E, pr = apply_evidence([{"src": "sim_p50", "val": 0.10}, {"src": "dt_peak", "val": 0.10}],
                               require_human=True)
    chk("T7a weak 无人工→candidate(延迟)", (st, round(E, 2), pr), ("candidate", 0.7, None))
    st, E, pr = apply_evidence([{"src": "human_pass", "val": 0.15}, {"src": "dt_peak", "val": 0.10}],
                               require_human=True)
    chk("T7b weak 人工+自动→human-assisted", (st, round(E, 2), pr), ("active", 0.75, "human-assisted"))
    print(f"\n结果: {'全部通过 ✓' if not FAILS else f'{len(FAILS)} 项失败: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.selftest:
        selftest()

    records, params, gaps, n_blocked_added = build_states()
    by_state = Counter(r["state"] for r in records.values())
    by_source_state = Counter((r["source"], r["state"]) for r in records.values())
    auto_promoted = [r for r in records.values() if r["promotion"] == "auto"]
    human_assisted = [r for r in records.values() if r["promotion"] == "human-assisted"]
    suspects = [r for r in records.values() if r["state"] == "suspect"]
    n_weak = sum(1 for r in records.values() if r.get("tier") == "weak")
    weak_awaiting = [r for r in records.values()
                     if r.get("tier") == "weak" and r["state"] == "candidate" and (r["E"] or 0) >= 0.7]

    print("=== F4 状态机实例化（edges_v2 ∪ blocked ∪ pending ∪ 判定合并）===")
    print(f"  参数: {json.dumps(params, ensure_ascii=False)}")
    print(f"  状态分布: {dict(by_state)}（共 {len(records)} 条边 | 弱层 {n_weak} | "
          f"blocked 纳入 {n_blocked_added}）")
    print(f"  来源×状态: {dict(by_source_state)}")
    print(f"\n  [晋升] auto（双自动证据, E=0.70）: {len(auto_promoted)} 条")
    for r in sorted(auto_promoted, key=lambda x: -x["E"]):
        print(f"    {r['key']}  E={r['E']}")
    print(f"  [晋升] human-assisted（人工+自动, E=0.75）: {len(human_assisted)} 条")
    for r in sorted(human_assisted, key=lambda x: -x["E"]):
        print(f"    {r['key']}  E={r['E']}")
    print(f"  [降级] suspect: {len(suspects)} 条")
    for r in suspects:
        print(f"    {r['key']}  E={r['E']}  {r['note']}")
    if weak_awaiting:
        print(f"  [延迟提交] 弱层自动证据达标（E≥0.70）但无人工证据，暂不晋升: {len(weak_awaiting)} 条")
        for r in weak_awaiting:
            print(f"    {r['key']}  E={r['E']}  {r['note']}")
    if gaps:
        print(f"  [缺口] {len(gaps)}:" + "; ".join(f"{g['pair']}({g['kind']})" for g in gaps))

    # ---- 与 L2 对账一致性交叉验证（4+1 条恢复边应 active；2 条异人 suspect） ----
    expect_active = ["343:343-E042->316:316-E046", "343:343-E042->316:316-E048",
                     "343:343-E038->316:316-E043", "343:343-E011->316:316-E013",
                     "343:343-E010->316:316-E014"]
    expect_suspect = ["343:343-E042->316:316-E050", "343:343-E042->316:316-E049"]
    ok_a = all(records.get(k, {}).get("state") == "active" for k in expect_active)
    ok_s = all(records.get(k, {}).get("state") == "suspect" for k in expect_suspect)
    print(f"\n  [交叉验证] 恢复边 5 条（4 排他竞争 + 1 弱通道）全 active: {'✓' if ok_a else '✗'} | "
          f"已判异人边 2 条 suspect: {'✓' if ok_s else '✗'}")
    for k in expect_active + expect_suspect:
        r = records.get(k)
        print(f"    {k}: state={r['state'] if r else 'MISSING'} E={r['E'] if r else '-'} "
              f"promotion={r['promotion'] if r else '-'}")

    out = {"version": "relay_v2/states/1",
           "params": params,
           "states": {k: {kk: vv for kk, vv in r.items() if kk != "key"}
                      for k, r in sorted(records.items())},
           "summary": {"n_edges": len(records), "by_state": dict(by_state),
                       "by_source_state": {f"{a}/{b}": c for (a, b), c in by_source_state.items()},
                       "n_weak": n_weak, "n_blocked_added": n_blocked_added,
                       "n_weak_awaiting": len(weak_awaiting),
                       "n_auto_promoted": len(auto_promoted),
                       "n_human_assisted": len(human_assisted),
                       "n_suspect": len(suspects),
                       "auto_promoted_keys": [r["key"] for r in auto_promoted],
                       "human_assisted_keys": [r["key"] for r in human_assisted],
                       "suspect_keys": [r["key"] for r in suspects],
                       "weak_awaiting_keys": [r["key"] for r in weak_awaiting],
                       "gaps": gaps,
                       "crosscheck_ok": bool(ok_a and ok_s)}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")

    if not (ok_a and ok_s):
        sys.exit(1)


if __name__ == "__main__":
    main()
