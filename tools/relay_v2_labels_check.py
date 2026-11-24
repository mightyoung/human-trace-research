#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relay v2 增量对账：消费最新人工判定（remote/labels.json ∪ pair_candidates_labeled.csv）。

只读输入（不写公共文件——判定回流由真值工作流负责）：
  - output_crosscam/analysis/truth/remote/labels.json        （最新批次判定，v=1/0/2）
  - output_crosscam/analysis/truth/pair_candidates_labeled.csv（早期 19 条，可选）
  - output_crosscam/analysis/relay_v2/edges_v2.json           （v2 边与状态）
  - output_crosscam/analysis/relay_clean2/relay_chains_cradio_v5.json（v5 基线）

输出：
  output_crosscam/analysis/relay_v2/labels_incremental.json + 控制台表

口径：pair_id "343_E_343-E042__316_E_316-E053" → ("343:343-E042","316:316-E053")；
      v=1 同人 / v=0 异人 / v=2 看不清；labels.json 优先于 labeled.csv。
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
EDGES_V2 = os.path.join(ANA, "relay_v2", "edges_v2.json")
STATE_JSON = os.path.join(ANA, "relay_v2", "edge_states.json")
V5 = os.path.join(ANA, "relay_clean2", "relay_chains_cradio_v5.json")
OUT = os.path.join(ANA, "relay_v2", "labels_incremental.json")


def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def main():
    # ---- 合并判定（labels.json 优先） ----
    judged = {}   # pair_id -> ("1"|"0"|"?", src)
    if os.path.exists(LABELED):
        for r in csv.DictReader(open(LABELED, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v in ("1", "0", "?"):
                judged[r["pair_id"]] = (v, "labeled.csv")
    n_early = len(judged)
    d = json.load(open(LABELS_JSON, encoding="utf-8"))
    for pid, rec in d.get("pair", {}).items():
        # 命名空间防御（同 relay_states 约定）：S6B_P_/PRT_ 前缀剥离 + 格式校验；
        # 实体级键（S6B_B_/S6B_R_ 等无 '__'）跳过，避免 parse_pid 崩溃。
        p2 = pid
        for pref in ("S6B_P_", "PRT_"):
            if p2.startswith(pref):
                p2 = p2[len(pref):]
                break
        parts = p2.split("__")
        if len(parts) != 2:
            continue
        la, lb = parts[0].split("_"), parts[1].split("_")
        if len(la) < 3 or len(lb) < 3:
            continue
        v = rec.get("v")
        judged[p2] = ({1: "1", 0: "0", 2: "?"}.get(v, "?"), "labels.json")
    print(f"[merge] labeled.csv 早期 {n_early} 条 + labels.json {len(d.get('pair', {}))} 条 "
          f"→ 合计 {len(judged)} 条")

    # ---- 边集 ----
    v2 = json.load(open(EDGES_V2, encoding="utf-8"))
    act = {(e["a"], e["b"]) for e in v2["edges"] if e["status"] == "active"}
    aud = {(e["a"], e["b"]) for e in v2["edges"] if e["status"] == "audit"}
    blk = {(g["a"], x["b"]) for g in v2.get("conflict_groups", []) for x in g["blocked"]}
    acc = act | aud
    v5 = json.load(open(V5, encoding="utf-8"))
    v5s = {(e["a"], e["b"]) for e in v5["edges"]}
    states = {}
    if os.path.exists(STATE_JSON):
        sj = json.load(open(STATE_JSON, encoding="utf-8"))
        states = {k: v["state"] for k, v in sj.get("states", {}).items()}

    pos = neg = unc = 0
    rows = []
    for pid, (v, src) in sorted(judged.items()):
        a, b = parse_pid(pid)
        key = (a, b)
        st = ("active" if key in act else "audit" if key in aud else
              "blocked" if key in blk else "")
        rows.append({"pair": f"{a}<->{b}", "v": v, "src": src, "v5": key in v5s,
                     "v2": st or "not_in_pool",
                     "f4": states.get(f"{a}->{b}", "-")})
        if v == "1":
            pos += 1
        elif v == "0":
            neg += 1
        else:
            unc += 1

    # ---- 指标 ----
    def cnt(pred):
        return sum(1 for r in rows if pred(r))
    res = {
        "n_judged": len(rows), "pos": pos, "neg": neg, "unclear": unc,
        "sources": {s: sum(1 for r in rows if r["src"] == s) for s in ("labeled.csv", "labels.json")},
        "pos_cover": {
            "v5": cnt(lambda r: r["v"] == "1" and r["v5"]),
            "v2_active": cnt(lambda r: r["v"] == "1" and r["v2"] == "active"),
            "v2_active_audit": cnt(lambda r: r["v"] == "1" and r["v2"] in ("active", "audit")),
            "v2_blocked": cnt(lambda r: r["v"] == "1" and r["v2"] == "blocked"),
            "v2_active_audit_blocked": cnt(lambda r: r["v"] == "1" and r["v2"] in
                                           ("active", "audit", "blocked")),
            "f4_state_active": cnt(lambda r: r["v"] == "1" and r.get("f4") == "active"),
            "f4_state_active_blocked": cnt(lambda r: r["v"] == "1" and r.get("f4") in
                                           ("active", "blocked")),
            "denominator": pos},
        "neg_hits": {
            "v5": cnt(lambda r: r["v"] == "0" and r["v5"]),
            "v2_active": cnt(lambda r: r["v"] == "0" and r["v2"] == "active"),
            "v2_audit": cnt(lambda r: r["v"] == "0" and r["v2"] == "audit"),
            "v2_blocked": cnt(lambda r: r["v"] == "0" and r["v2"] == "blocked"),
            "denominator": neg},
    }
    print("\n=== 判同人（v=1）覆盖 ===")
    pc = res["pos_cover"]
    print(f"  n={pc['denominator']} | v5 覆盖 {pc['v5']} | v2-active {pc['v2_active']} | "
          f"v2-active+audit {pc['v2_active_audit']} | +blocked(显露) {pc['v2_active_audit_blocked']} | "
          f"F4-active {pc['f4_state_active']} | F4-active+blocked {pc['f4_state_active_blocked']}")
    print("=== 判异人（v=0）命中（必须在 active 为 0） ===")
    nh = res["neg_hits"]
    print(f"  n={nh['denominator']} | v5 {nh['v5']} | v2-active {nh['v2_active']} | "
          f"v2-audit {nh['v2_audit']} | blocked {nh['v2_blocked']}")

    print("\n=== 明细（v=1 / v=0 全部；v=? 仅列 v2 有边者） ===")
    for r in rows:
        if r["v"] == "1":
            print(f"  [同人] {r['pair']:38s} v5={'Y' if r['v5'] else '-'} v2={r['v2']:12s} "
                  f"f4={r['f4']:12s} ({r['src']})")
    for r in rows:
        if r["v"] == "0":
            print(f"  [异人] {r['pair']:38s} v5={'Y' if r['v5'] else '-'} v2={r['v2']:12s} "
                  f"f4={r['f4']:12s} ({r['src']})")
    for r in rows:
        if r["v"] == "?" and r["v2"] in ("active", "audit", "blocked"):
            print(f"  [看不清] {r['pair']:38s} v2={r['v2']:12s}（待复验）")

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
