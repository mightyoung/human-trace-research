#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 对账：Relay v2 vs 人工判定（343×316 已判 19 行）

判据（docs/relay-v2-plan.md §7.3）：
  Go  =  排他竞争 3 对全部恢复（E042↔E046 / E042↔E048 / E038↔E043）
        且已判异人 2 对（E042↔E050 / E042↔E049）不进 active
  记录 =  其余漏接（E010↔E014 候选漏[F2 靶] / E011↔E013 毫厘差）的恢复情况
  FP  =  判异人却进 active → 硬失败

用法:
  python tools/relay_v2_reconcile.py
输出:
  output_crosscam/analysis/relay_v2/reconcile.json + 可读表
"""
import argparse
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
EDGES_V2 = os.path.join(ANA, "relay_v2", "edges_v2.json")
V5_CHAINS = os.path.join(ANA, "relay_clean2", "relay_chains_cradio_v5.json")
OUT = os.path.join(ANA, "relay_v2", "reconcile.json")

# 排他竞争 3 对（必须恢复，Go 判据核心）
RACE_3 = [("343:343-E042", "316:316-E046"),
          ("343:343-E042", "316:316-E048"),
          ("343:343-E038", "316:316-E043")]


def parse_pid(pid):
    """'343_E_343-E042__316_E_316-E053' -> ('343:343-E042','316:316-E053')"""
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=LABELED)
    ap.add_argument("--edges", default=EDGES_V2)
    ap.add_argument("--v5", default=V5_CHAINS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # ---- 载入 ----
    rows = list(csv.DictReader(open(args.labeled, encoding="utf-8-sig")))
    v2 = json.load(open(args.edges, encoding="utf-8"))
    v5 = json.load(open(args.v5, encoding="utf-8"))

    v5_set = {(e["a"], e["b"]) for e in v5["edges"]}
    active_set = {(e["a"], e["b"]) for e in v2["edges"] if e["status"] == "active"}
    audit_set = {(e["a"], e["b"]) for e in v2["edges"] if e["status"] == "audit"}
    blocked_set = set()
    for g in v2.get("conflict_groups", []):
        for x in g["blocked"]:
            blocked_set.add((g["a"], x["b"]))

    judged = [r for r in rows if r.get("human_same_person", "").strip() != ""]

    # ---- 逐行对账 ----
    table, missed, recovered, fp_active, fp_audit, fp_blocked = [], [], [], [], [], []
    for r in judged:
        a, b = parse_pid(r["pair_id"])
        judge = r["human_same_person"].strip()
        has_v5 = (a, b) in v5_set
        has_act = (a, b) in active_set
        has_aud = (a, b) in audit_set
        has_blk = (a, b) in blocked_set
        rec = {"pair": f"{a} <-> {b}", "a": a, "b": b, "judge": judge,
               "dt": r["dt"], "conf": r.get("human_confidence", ""),
               "v5": has_v5, "v2_active": has_act, "v2_audit": has_aud,
               "v2_conflict_blocked": has_blk}
        table.append(rec)
        if judge == "1" and not has_v5:
            missed.append(rec)
            if has_act or has_aud:
                rec["_recovered"] = True
                recovered.append(rec)
        if judge == "0":
            if has_act:
                fp_active.append(rec)
            if has_aud:
                fp_audit.append(rec)
            if has_blk:
                fp_blocked.append(rec)

    # ---- Go/No-Go ----
    race_recovered = [p for p in missed if (p["a"], p["b"]) in RACE_3 and p.get("_recovered")]
    race_ok = len(race_recovered) == len(RACE_3)
    fp_ok = len(fp_active) == 0
    go = race_ok and fp_ok

    # ---- 打印 ----
    print("=== L2 对账（人工判定 vs v5 / v2）===")
    print(f"{'#':>2} {'pair':42s} {'判定':4s} {'v5':4s} {'v2-active':9s} {'v2-audit':8s} {'v2-blocked':10s}")
    for i, t in enumerate(table, 1):
        jd = {"1": "同人", "0": "异人", "?": "存疑"}[t["judge"]]
        mark = ""
        if t["judge"] == "1" and not t["v5"]:
            mark = "  ←漏接" + ("(已恢复)" if t.get("_recovered") else "(未恢复)")
        print(f"{i:>2} {t['pair']:42s} {jd:4s} "
              f"{'✓' if t['v5'] else '-':4s} "
              f"{'✓' if t['v2_active'] else '-':9s} "
              f"{'✓' if t['v2_audit'] else '-':8s} "
              f"{'✓' if t['v2_conflict_blocked'] else '-':10s}{mark}")

    print(f"\n[漏接] v5 漏 {len(missed)} 对（判同人但 v5 无边）")
    print(f"[恢复] v2 恢复 {len(recovered)}/{len(missed)}：")
    for p in missed:
        st = "✓ 恢复" if p.get("_recovered") else "✗ 未恢复"
        loc = ("active" if p["v2_active"] else ("audit" if p["v2_audit"] else
               ("blocked(conflict)" if p["v2_conflict_blocked"] else "不在池/被静默")))
        print(f"  {st}  {p['pair']}  ({loc})")
    print(f"\n[排他竞争 3 对] 恢复 {len(race_recovered)}/3 → {'Go ✓' if race_ok else 'No-Go ✗'}")
    print(f"[异人防误纳] active 误纳 {len(fp_active)} 条 | audit 误纳 {len(fp_audit)} 条 | "
          f"blocked(conflict 提示) {len(fp_blocked)} 条 → {'Go ✓' if fp_ok else 'No-Go ✗'}")
    for p in fp_active:
        print(f"  ★ 误纳(active): {p['pair']}")
    for p in fp_audit:
        print(f"  ⚠ 误纳(audit): {p['pair']}（审计池，未进生产链）")
    for p in fp_blocked:
        print(f"  · 冲突提示(blocked): {p['pair']}（已显式化，未进任何边集）")

    verdict = "Go" if go else "No-Go"
    print(f"\n=== S1 判定: {verdict} ===")

    res = {"n_judged": len(judged), "missed": len(missed), "recovered": len(recovered),
           "race_recovered": len(race_recovered), "race_required": len(RACE_3),
           "fp_active": len(fp_active), "fp_audit": len(fp_audit), "fp_blocked": len(fp_blocked),
           "verdict": verdict, "table": table}
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
