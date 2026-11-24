# -*- coding: utf-8 -*-
"""EVAL-1 诊断：逐条打印被判对的六层明细（找 defer/reject 根因）。"""
import json

d = json.load(open("output_crosscam/analysis/eval_registry/registry_v0.json", encoding="utf-8"))
idx = {(r["a"], r["b"]): r for r in d["candidates"]}


def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def show(pid, human):
    a, b = parse_pid(pid)
    r = idx.get((a, b))
    if r is None:
        print(f"[MISS] {pid} human={human}")
        return
    l3, l4 = r["layers"]["L3_evidence"], r["layers"]["L4_constraints"]
    l2 = r["layers"]["L2_geometry"]
    print(f"{r['action']:>6} | human={human:>1} | {pid}")
    print(f"        sim={l3['sim']} w_dt={l3['w_dt']} dt={l3['dt']} margin={l3['margin_out']} "
          f"gate={l2['status']}({l2['pd_a']},{l2['pd_b']}) overlap={l4['overlap']} exempt={l4['exempt']}")
    print(f"        iv_a={l4['a_iv']} iv_b={l4['b_iv']} reasons={r['reasons']}")


# 组裁决明细
groups = json.load(open("output_crosscam/analysis/truth/_review/conflict_verdicts.json",
                        encoding="utf-8"))["groups"]
print("========== 组裁决（same/diff） ==========")
for gk, gv in groups.items():
    for c in gv.get("candidates", []):
        if c.get("human") in ("same", "diff"):
            show(c["pair_id"], c["human"])

print()
print("========== 55 判（pos/neg/unclear） ==========")
import csv, os
judged = {}
if os.path.exists("output_crosscam/analysis/truth/pair_candidates_labeled.csv"):
    for r in csv.DictReader(open("output_crosscam/analysis/truth/pair_candidates_labeled.csv",
                                 encoding="utf-8-sig")):
        v = (r.get("human_same_person") or "").strip()
        if v in ("1", "0", "?"):
            judged[r["pair_id"]] = v
lj = json.load(open("output_crosscam/analysis/truth/remote/labels.json", encoding="utf-8"))
for pid, rec in lj.get("pair", {}).items():
    judged[pid] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
for pid, v in sorted(judged.items(), key=lambda x: x[1]):
    if v in ("1", "0"):   # 只细看 pos/neg
        # 命名空间防御：S6B_/PRT_ 命名空间与非法格式键跳过（非本标准 pair）
        if pid.startswith(("S6B", "PRT_")) or "__" not in pid:
            continue
        show(pid, v)
