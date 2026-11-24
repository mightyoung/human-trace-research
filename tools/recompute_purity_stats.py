#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用最新 D3 真值（257 行 / 40 簇）重算簇纯度相关统计。

与 `diag_cluster_harm.py` 的区别：那个脚本读 D7 verdicts 里缓存的 `cluster_n_ids`
（旧真值），本脚本每次从 `gt/clusters_gt.csv` 现算，真值更新后不需要重跑 D7。

输出
  1) 簇纯度总账（身份簇 / 混人簇 / 非人体簇）
  2) d_max 对「混人 / 产生错判 / 混人簇内有害」的 AUC
  3) 零误删前提下的 d_max 上限与可达召回
"""
import collections
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
RUN = os.path.join(DELIV, "results")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    cl = rd(os.path.join(DELIV, "gt", "clusters_gt.csv"))
    gid = {int(r["cluster_id"]): r for r in cl}
    lab2gid = {}
    for r in rd(os.path.join(DELIV, "ledger", "clusters_geom.csv")):
        if r["config"] == "baseline":
            lab2gid[int(r["cluster_lab"])] = int(r["cluster_idx"])
    geo = {}
    for r in rd(os.path.join(DELIV, "ledger", "clusters_geom.csv")):
        if r["config"] == "baseline":
            geo[int(r["cluster_idx"])] = r

    # 每个全局簇的 C/W/U（来自 baseline 判决）
    loc2gid = {}
    for r in rd(os.path.join(RUN, "members", "A_baseline.csv")):
        loc2gid[int(r["cluster_local"])] = lab2gid[int(r["cluster_lab"])]
    per = collections.defaultdict(lambda: {"C": 0, "W": 0, "U": 0})
    for r in rd(os.path.join(RUN, "verdicts", "A_baseline.csv")):
        g = loc2gid[int(r["cluster_local"])]
        per[g][r["verdict"]] += 1

    ident = [k for k, r in gid.items() if int(r["n_ids"]) >= 1]
    mixed = [k for k in ident if int(gid[k]["n_ids"]) >= 2]
    pure = [k for k in ident if int(gid[k]["n_ids"]) == 1]
    nonhuman = [k for k, r in gid.items() if int(r["n_ids"]) == 0]

    print("=== 1 簇纯度总账 ===")
    print(f"  模板簇 {len(gid)} = 身份簇 {len(ident)} + 非人体簇 {len(nonhuman)}")
    print(f"  混人 {len(mixed)}/{len(ident)} = {len(mixed)/len(ident):.1%}"
          f"（成员 {sum(int(gid[k]['n_members']) for k in mixed)} 条）")
    print(f"  非人体簇: " + ", ".join(
        f"簇{k}({gid[k]['n_members']}条,{gid[k]['main_pid'] or '—'})" for k in nonhuman))
    q_hit = sorted(per)
    q_mixed = [k for k in q_hit if len(gid[k]) and int(gid[k]["n_ids"]) >= 2]
    print(f"  有查询命中的簇 {len(q_hit)}：混人 {len(q_mixed)}，"
          f"承载查询 {sum(sum(per[k].values()) for k in q_mixed)}")

    # ---- 2 AUC（只用身份簇；非人体簇无身份可比）
    rows = [dict(k=k, dmax=float(geo[k]["d_max"]),
                 mixed=int(gid[k]["n_ids"]) >= 2,
                 C=per[k]["C"], W=per[k]["W"], U=per[k]["U"],
                 served=sum(per[k].values()),
                 harmed=per[k]["W"] > 0) for k in ident if k in per]
    print("\n=== 2 d_max 的判别力（仅身份簇且有查询）===")
    print(f"  参与 {len(rows)} 簇 | 混人 {sum(r['mixed'] for r in rows)} "
          f"| 产生过错判 {sum(r['harmed'] for r in rows)}")
    a1 = auc([r["dmax"] for r in rows if r["mixed"]],
             [r["dmax"] for r in rows if not r["mixed"]])
    a2 = auc([r["dmax"] for r in rows if r["harmed"]],
             [r["dmax"] for r in rows if not r["harmed"]])
    mm = [r for r in rows if r["mixed"]]
    a3 = auc([r["dmax"] for r in mm if r["harmed"]],
             [r["dmax"] for r in mm if not r["harmed"]])
    print(f"  簇是否混人          AUC = {a1:.3f}")
    print(f"  簇是否产生错判      AUC = {a2:.3f}")
    print(f"  混人簇内有害/无害   AUC = {a3:.3f} "
          f"(有害 {sum(r['harmed'] for r in mm)} / 无害 {sum(not r['harmed'] for r in mm)})")
    print(f"  混人簇 d_max 中位 {statistics.median([r['dmax'] for r in mm]):.3f}"
          f"｜纯簇 {statistics.median([r['dmax'] for r in rows if not r['mixed']]):.3f}")

    # ---- 3 零误删阈值
    pure_max = max(r["dmax"] for r in rows if not r["mixed"])
    above = sorted([r["dmax"] for r in mm if r["dmax"] > pure_max])
    print("\n=== 3 零误删前提下的阈值上限 ===")
    print(f"  纯簇 d_max 最大 {pure_max:.3f} → 阈值必须 > 该值才不会误删纯簇")
    print(f"  该范围内可拦到的混人簇 {len(above)}/{len(mm)} = {len(above)/len(mm):.1%}")
    if above:
        print(f"  阈值区间 [{pure_max:.3f}, {min(above):.3f}) 等价，取中 {pure_max:.3f}")
    for t in (0.28, 0.30):
        hit = [r for r in mm if r["dmax"] >= t]
        drop = [r for r in rows if not r["mixed"] and r["dmax"] >= t]
        print(f"  阈值 {t:.2f}: 拦混人 {len(hit)}/{len(mm)}"
              f"，误删纯簇 {len(drop)}，影响查询 "
              f"{sum(r['served'] for r in hit)} 条")

    # ---- 4 全量身份簇（不只被查询命中的）阈值扫描
    allrows = [dict(k=k, dmax=float(geo[k]["d_max"]),
                    mixed=int(gid[k]["n_ids"]) >= 2) for k in ident]
    print("\n=== 4 全量身份簇阈值扫描（纯度为唯一目标）===")
    print(f"{'阈值':<8}{'删混人':>8}{'误删纯':>7}{'召回':>7}{'精度':>7}")
    n_mixed_all = sum(1 for r in allrows if r["mixed"])
    for t in (0.16, 0.20, 0.24, 0.265, 0.30):
        hit = [r for r in allrows if r["mixed"] and r["dmax"] >= t]
        drop = [r for r in allrows if not r["mixed"] and r["dmax"] >= t]
        prec = len(hit) / (len(hit) + len(drop)) if (hit or drop) else float("nan")
        print(f"{t:<8.3f}{str(len(hit)) + '/' + str(n_mixed_all):>8}"
              f"{len(drop):>7}{len(hit)/n_mixed_all:>7.2f}{prec:>7.2f}")
    pmax = max(r["dmax"] for r in allrows if not r["mixed"])
    above = sum(1 for r in allrows if r["mixed"] and r["dmax"] > pmax)
    print(f"  纯簇 d_max 最大 {pmax:.3f}；零误删上限召回 {above}/{sum(r['mixed'] for r in allrows)}")


if __name__ == "__main__":
    main()
