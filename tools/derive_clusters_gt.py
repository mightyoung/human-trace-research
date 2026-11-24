#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""由 D3 的逐成员标注派生簇级真值（清单 D3 要求的 gt/clusters_gt.csv）。

判据：
  身份数 n_ids = 该簇内不同 person_id 的个数（空 person_id、UNK、no-human 不计）。
  n_ids >= 2 记为「混人簇」。
  person_id = no-human 单独计入 n_nonhuman（非人体误检，如家具/反光），
  既不构成身份，也不算「未标注」。
  另外单列 n_idswitch：簇内发生过轨迹内换人（id_switch=1）的轨迹数——
  它和「簇合并错误」是两个不同的病因，不能混为一谈。

输入  gt/gt_gallery_members_FILLED.csv（人工填写）
      ledger/clusters_geom.csv（几何，提供 d_max / 旧判定）
输出  gt/clusters_gt.csv
"""
import csv
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GT = os.path.join(ROOT, "deliverable/data-request-v2/gt")
LEDGER = os.path.join(ROOT, "deliverable/data-request-v2/ledger")

SRC = os.path.join(GT, "gt_gallery_members_FILLED.csv")
TEMPLATE = os.path.join(GT, "gallery_members_TEMPLATE.csv")
GEOM = os.path.join(LEDGER, "clusters_geom.csv")
DST = os.path.join(GT, "clusters_gt.csv")

NON_HUMAN = {"no-human", "nohuman", "no_human", "none", "无", "非人"}


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    tpl = list(csv.DictReader(open(TEMPLATE, encoding="utf-8-sig")))
    geom = {int(r["cluster_idx"]): r for r in
            csv.DictReader(open(GEOM, encoding="utf-8-sig")) if r["config"] == "baseline"}

    got = {r["track_id"] for r in rows}
    missing = sorted({r["track_id"] for r in tpl} - got)

    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["cluster_idx"])].append(r)

    out = []
    for c in sorted(by):
        rs = by[c]
        pids = [r["person_id"].strip() for r in rs
                if r["person_id"].strip() and r["person_id"].strip().upper() != "UNK"
                and r["person_id"].strip().lower() not in NON_HUMAN]
        cnt = collections.Counter(pids)
        n_nonhuman = sum(1 for r in rs
                         if r["person_id"].strip().lower() in NON_HUMAN)
        n_ids = len(cnt)
        main_pid, main_n = (cnt.most_common(1)[0] if cnt else ("", 0))
        mixed = [f"{k}({v})" for k, v in cnt.most_common() if k != main_pid]
        unk = sum(1 for r in rs if not r["person_id"].strip()
                  or r["person_id"].strip().upper() == "UNK")
        n_lab = len(rs) - unk
        g = geom.get(c, {})
        old = g.get("purity_verdict", "")
        new_bad = n_ids >= 2
        if not n_lab:
            agree = "no_data"
        elif not old:
            agree = "new"
        else:
            agree = "agree" if (old == "bad") == new_bad else "conflict"

        out.append({
            "cluster_id": c,
            "n_members": len(rs),
            "n_labelled": n_lab,
            "n_unlabelled": unk,
            "n_ids": n_ids,
            "n_nonhuman": n_nonhuman,
            "main_pid": main_pid,
            "main_n": main_n,
            "main_share": round(main_n / n_lab, 3) if n_lab else "",
            "mixed_members": "|".join(mixed),
            "n_head": sum(1 for r in rs if r["box_type"].strip() == "head"),
            "n_body": sum(1 for r in rs if r["box_type"].strip() == "body"),
            "n_partial": sum(1 for r in rs if r["box_type"].strip() == "partial"),
            "n_idswitch": sum(1 for r in rs
                              if r["id_switch"].strip() not in ("", "0")),
            "d_max": g.get("d_max", ""),
            "cams": g.get("cams", ""),
            "old_verdict": old,
            "agree": agree,
        })

    out.sort(key=lambda x: (-x["n_ids"], x["cluster_id"]))
    with open(DST, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    # ---- 控制台摘要
    lab = [r for r in out if r["n_labelled"]]
    nonhuman = [r for r in lab if r["n_ids"] == 0 and r["n_nonhuman"] > 0]
    ident = [r for r in lab if r["n_ids"] >= 1]
    mixed = [r for r in ident if r["n_ids"] >= 2]
    pure = [r for r in ident if r["n_ids"] == 1]
    print(f"[out] {DST}  {len(out)} 簇")
    print(f"  有标注 {len(lab)} 簇（成员 {sum(r['n_labelled'] for r in lab)} 条）")
    print(f"  身份簇 {len(ident)}：混人 {len(mixed)} / {len(ident)} = {len(mixed)/max(1,len(ident)):.1%}"
          f"｜纯 {len(pure)}")
    print(f"  非人体簇 {len(nonhuman)}（n_ids=0, 成员 "
          f"{sum(r['n_members'] for r in nonhuman)} 条）"
          f"｜未标 {len(out)-len(lab)} 簇")
    print(f"  旧判定一致 {sum(1 for r in out if r['agree']=='agree')}"
          f"｜冲突 {sum(1 for r in out if r['agree']=='conflict')}"
          f"｜新增 {sum(1 for r in out if r['agree']=='new')}"
          f"｜无数据 {sum(1 for r in out if r['agree']=='no_data')}")
    print(f"  标注缺失 {len(missing)} 条: {missing}")
    print(f"  未填 person_id {sum(r['n_unlabelled'] for r in out)} 条")
    print(f"  轨迹内换人 {sum(r['n_idswitch'] for r in out)} 条"
          f"（占已标 {sum(r['n_idswitch'] for r in out)/max(1,sum(r['n_labelled'] for r in out)):.1%}）")

    if mixed:
        print("\n  混人簇清单:")
        for r in mixed:
            print(f"    簇{r['cluster_id']:>4} n={r['n_members']:>2} 身份{r['n_ids']} "
                  f"主={r['main_pid']}×{r['main_n']} 混入={r['mixed_members']} "
                  f"d_max={r['d_max']} 旧={r['old_verdict'] or '—'}")

    conf = [r for r in out if r["agree"] == "conflict"]
    if conf:
        print("\n  与旧判定冲突:")
        for r in conf:
            print(f"    簇{r['cluster_id']:>4} 旧={r['old_verdict']:<3} 新身份数={r['n_ids']} "
                  f"主={r['main_pid']}×{r['main_n']} 混入={r['mixed_members'] or '—'} d_max={r['d_max']}")


if __name__ == "__main__":
    main()
