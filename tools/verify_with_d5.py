#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 D5（第二标注人）复核判决结论的稳健性。

口径处理（重要）：
  第二人的编号体系与第一人**不完全等价**：
    - male5 → male6（合并：第一人的 male5 全被叫成 male6）
    - female5 → female4（改名：第一人的 female5 全被叫成 female4）
  这 7 条不是「判断分歧」，而是「编号体系歧义」，无法判定谁对，
  必须从精度比较中**剔除**并单列，否则会虚增第二人的错误数。
"""
import csv
import collections
import os

GT = "deliverable/data-request-v2/gt"
RES = "deliverable/data-request-v2/results"

# 已确认的编号体系歧义（第一人编号 → 第二人编号）
AMBIG = {"male5": "male6", "female5": "female4"}


def load(p, key):
    with open(p, encoding="utf-8-sig") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def norm(p):
    p = (p or "").strip()
    return "UNK" if (not p or p.upper() == "UNK") else p


def verdict_of(q_pid, c_pid):
    if q_pid == "UNK" or c_pid == "UNK":
        return "U"
    return "C" if q_pid == c_pid else "W"


def main():
    a2 = load(os.path.join(GT, "gt_annotator2_FILLED.csv"), "item_id")
    a1q = load(os.path.join(GT, "gt_query_gt_FILLED.csv"), "query_id")
    d5q = {k: r for k, r in a2.items() if r["layer"] == "query"}
    vd_base = load(os.path.join(RES, "verdicts", "A_baseline.csv"), "query_id")

    acc = [k for k in vd_base if k in d5q]
    print(f"baseline strict 接受 {len(vd_base)} 条｜D5 覆盖其中 {len(acc)} 条")

    # ---------- 1. 分三类 ----------
    rows = []
    for k in acc:
        c_pid = norm(vd_base[k]["c_pid"])
        q1 = norm(a1q[k]["person_id"])
        q2raw = norm(d5q[k]["person_id"])
        amb = (q1 in AMBIG and q2raw == AMBIG[q1])
        rows.append(dict(k=k, c=c_pid, q1=q1, q2raw=q2raw,
                         v1=verdict_of(q1, c_pid), v2=verdict_of(q2raw, c_pid), amb=amb))

    amb_rows = [r for r in rows if r["amb"]]
    ok_rows = [r for r in rows if not r["amb"]]

    print(f"\n=== 编号体系歧义（{len(amb_rows)} 条，无法判定谁对，剔除）===")
    for r in amb_rows:
        print(f"  {r['k']:<20} 一人={r['q1']:<8} 二人={r['q2raw']:<8} "
              f"簇主={r['c']:<8} 一人判{r['v1']} 二人判{r['v2']}")

    print(f"\n=== 可判定 {len(ok_rows)} 条：两位标注人的判决 ===")
    print(f"{'':<7}{'C':>4}{'W':>4}{'U':>4}{'精度(不含U)':>12}")
    for tag, name in (("v1", "第一人"), ("v2", "第二人")):
        c = sum(1 for r in ok_rows if r[tag] == "C")
        w = sum(1 for r in ok_rows if r[tag] == "W")
        u = sum(1 for r in ok_rows if r[tag] == "U")
        pre = c / (c + w) if c + w else float("nan")
        print(f"{name:<7}{c:>4}{w:>4}{u:>4}{pre:>12.3f}")

    print(f"\n=== 真实分歧（{sum(1 for r in ok_rows if r['v1'] != r['v2'])} 条）===")
    for r in ok_rows:
        if r["v1"] != r["v2"]:
            print(f"  {r['k']:<20} 一人={r['q1']:<9} 二人={r['q2raw']:<9} "
                  f"簇主={r['c']:<9} {r['v1']}→{r['v2']}")

    # ---------- 2. 各配置在同一批查询上重判 ----------
    print(f"\n=== 若用第二人的 query 真值重判各配置（同一批 {len(ok_rows)} 条，已剔除编号歧义）===")
    print(f"{'配置':<26}{'C2':>4}{'W2':>4}{'精度2':>8}{'C1':>4}{'W1':>4}{'精度1':>8}{'ΔC':>5}")
    for cfg in ["A_baseline", "C_dmax0.28", "C_dmax0.30", "C_dmax0.34",
                "D_split0.20", "D_split0.20_veto0.30", "G_global0.20_veto0.30"]:
        pm = os.path.join(RES, "queries", f"{cfg}.csv")
        vm = os.path.join(RES, "verdicts", f"{cfg}.csv")
        if not os.path.exists(pm) or not os.path.exists(vm):
            continue
        vmap = load(vm, "query_id")
        sub = [r for r in ok_rows if r["k"] in vmap]
        c1 = sum(1 for r in sub if vmap[r["k"]]["verdict"] == "C")
        w1 = sum(1 for r in sub if vmap[r["k"]]["verdict"] == "W")
        c2 = w2 = 0
        for r in sub:
            c_pid = norm(vmap[r["k"]]["c_pid"])
            v = verdict_of(norm(d5q[r["k"]]["person_id"]), c_pid)
            if v == "C":
                c2 += 1
            elif v == "W":
                w2 += 1
        p1 = c1 / (c1 + w1) if c1 + w1 else float("nan")
        p2 = c2 / (c2 + w2) if c2 + w2 else float("nan")
        print(f"{cfg:<26}{c2:>4}{w2:>4}{p2:>8.3f}{c1:>4}{w1:>4}{p1:>8.3f}{c2 - c1:>+5}")

    # ---------- 3. 排序稳健性 ----------
    print("\n=== 排序稳健性（同一批查询，两位标注人的 C 排序是否一致）===")
    cfgs = ["A_baseline", "C_dmax0.28", "C_dmax0.30", "C_dmax0.34",
            "D_split0.20", "D_split0.20_veto0.30", "G_global0.20_veto0.30"]
    s1, s2 = [], []
    for cfg in cfgs:
        vm = os.path.join(RES, "verdicts", f"{cfg}.csv")
        if not os.path.exists(vm):
            continue
        vmap = load(vm, "query_id")
        sub = [r for r in ok_rows if r["k"] in vmap]
        c1 = sum(1 for r in sub if vmap[r["k"]]["verdict"] == "C")
        c2 = sum(1 for r in sub
                 if verdict_of(norm(d5q[r["k"]]["person_id"]), norm(vmap[r["k"]]["c_pid"])) == "C")
        s1.append((c1, cfg))
        s2.append((c2, cfg))
    o1 = [c for _, c in sorted(s1, reverse=True)]
    o2 = [c for _, c in sorted(s2, reverse=True)]
    print(f"  第一人 C 降序: {o1}")
    print(f"  第二人 C 降序: {o2}")
    top1 = sorted(s1, reverse=True)[0][1]
    top2 = sorted(s2, reverse=True)[0][1]
    print(f"  最优配置：第一人={top1}｜第二人={top2}  "
          f"{'一致' if top1 == top2 else '不一致'}")


if __name__ == "__main__":
    main()
