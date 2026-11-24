#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按层拆分的 a3 κ：图库层 / 查询层各自一个数，不合并。

为什么需要它：`eval_kappa_a3.py` 的 `run()` 把图库与查询拼在一起算一个 κ。
第二标注人交回查询层 100 条后，直接重跑会把已发表的 0.925（n=257，仅图库）
静默换成一个合并数，两者不可比。本脚本不改动原脚本，只复用它的函数。

对齐（本脚本的主口径）：
  第二标注人自建编号，必须先把 A3 的名字映射到 A1 的命名空间。
  对**每个 A3 编号**取与其共现次数最多的 A1 编号（many-to-one）。
  为什么不用原脚本的 `align()`：它先贪心做 A1→A3，再 `setdefault` 反演；
  当一个 A3 编号同时对应多个 A1 编号（即真实分歧）时反演只保留先到者，
  其余 A3 编号**拿不到映射**，其观测在结构上不可能判为一致。
  实测本次有 4 个 A3 编号被挤掉、承载 10 条观测（2.8%），原口径因此系统性偏低。
  原口径作为敏感性对照一并报出。

用法：
  PY tools/eval_kappa_a3_by_layer.py
  PY tools/eval_kappa_a3_by_layer.py --a2-query <path> --out <path> --loo
输出：deliverable/data-request-v2/results/kappa_a3_by_layer.json
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_kappa_a3 import (GT, RES, align, collect, is_nonperson, kappa,
                           kappa_ci, rd)

# 物件/附属物编号：A3 在查询层用 hat1、cloth 表示"不是人，是帽子/衣物"。
# 这是**事后**从数据里看到的标签，故单独成口径，不并入主口径。
OBJECT_RE = ("hat", "cloth", "coat", "bag", "cart", "chair", "screen", "door")


def is_object(p):
    p = (p or "").strip().lower()
    return any(p.startswith(k) for k in OBJECT_RE)


def build_map(rows):
    """A3 编号 -> A1 编号，取共现最多者（many-to-one）。"""
    per = collections.defaultdict(collections.Counter)
    for _, _, a1, a2, _ in rows:
        if a2 == "UNK":
            continue
        per[a2][a1] += 1
    mp, one_to_many = {}, {}
    for a2, c in per.items():
        a1 = c.most_common(1)[0][0]
        mp[a2] = a1
        if len(c) > 1:
            one_to_many[a2] = dict(best=a1, also=[k for k, _ in c.most_common()[1:]],
                                   counts=dict(c))
    return mp, one_to_many


def relabel(rows, mp):
    return [(a1, (mp.get(a2, a2) if a2 != "UNK" else "UNK")) for _, _, a1, a2, _ in rows]


def stats(aligned, drop_obj=False):
    """三口径。drop_obj: 物件编号也当非人（事后口径）。"""
    k = kappa(aligned)
    lo, hi = kappa_ci(aligned)
    col = []
    for a, b in aligned:
        an = "NONPERSON" if (is_nonperson(a) or (drop_obj and is_object(a))) else a
        bn = "NONPERSON" if (is_nonperson(b) or (drop_obj and is_object(b))) else b
        col.append((an, bn))
    kcol = kappa(col)
    per = [(a, b) for a, b in col if a not in ("NONPERSON", "UNK") and b not in ("NONPERSON", "UNK")]
    kper = kappa(per)
    lo2, hi2 = kappa_ci(per)
    return dict(n=k["n"], po=k["po"], k=k["k"], ci95=[lo, hi],
                nonperson_collapsed=dict(n=kcol["n"], po=kcol["po"], k=kcol["k"]),
                person_only=dict(n=kper["n"], po=kper["po"], k=kper["k"], ci95=[lo2, hi2]))


def kappa_loo(rows):
    """留一法映射：每条用"去掉自己"的共现证据重新估计映射再标注。

    全量映射是in-sample估计——名字的归属与被评价的条目用同一批数据，
    单个条目独占一个编号时会"构造性正确"，κ 偏乐观。留一法消掉这一层偏差，
    代价是独条编号失去唯一的证据来源，映射变得任意，κ 偏保守。
    两个数一起报，真实值在两者之间。
    """
    def build(rs):
        per = collections.defaultdict(collections.Counter)
        for _, _, a1, a2, _ in rs:
            if a2 != "UNK":
                per[a2][a1] += 1
        return {a2: c.most_common(1)[0][0] for a2, c in per.items()}

    def label(sub, mp):
        return [(a1, (mp.get(a2, a2) if a2 != "UNK" else "UNK")) for _, _, a1, a2, _ in sub]

    out, changed = {}, 0
    full, _ = build_map(rows)
    for name in ("gallery", "query", "pooled"):
        sub = rows if name == "pooled" else [r for r in rows if r[1] == name]
        if not sub:
            continue
        relab = []
        for i, r in enumerate(rows):
            mp = build(rows[:i] + rows[i + 1:])
            if name == "pooled" or r[1] == name:
                relab.append(label([r], mp)[0])
        k = kappa(relab)
        lo, hi = kappa_ci(relab)
        out[name] = dict(n=k["n"], po=k["po"], k=k["k"], ci95=[lo, hi])
    for i, r in enumerate(rows):
        mp = build(rows[:i] + rows[i + 1:])
        if label([r], mp)[0][1] != label([r], full)[0][1]:
            changed += 1
    out["_mapping_flips"] = changed
    out["_flip_rate"] = changed / len(rows)
    return out


def show(name, st):
    ci = st["ci95"]
    print(f"{name:<10}{st['n']:>5}{st['po']:>8.3f}{st['k']:>8.3f}"
          f"{'[%.3f, %.3f]' % (ci[0], ci[1]):>18}"
          f"{st['person_only']['k']:>9.3f}{st['nonperson_collapsed']['k']:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2-gallery", default=os.path.join(GT, "a3_gallery_members_FILLED.csv"))
    ap.add_argument("--a2-query", default=os.path.join(GT, "a3_query_gt_FILLED.csv"))
    ap.add_argument("--a1-gallery", default=os.path.join(GT, "gt_gallery_members_FILLED.csv"),
                    help="A1 图库真值；传 .v2.csv 可复现第一标注人复核前的 kappa")
    ap.add_argument("--a1-query", default=os.path.join(GT, "gt_query_gt_FILLED.csv"),
                    help="A1 查询真值；传 .v2.csv 可复现第一标注人复核前的 kappa")
    ap.add_argument("--out", default=os.path.join(RES, "kappa_a3_by_layer.json"))
    ap.add_argument("--loo", action="store_true", help="额外算留一法映射口径（较慢，约 1 分钟）")
    a = ap.parse_args()

    gt_g = rd(a.a1_gallery)
    gt_q = rd(a.a1_query)

    if not os.path.exists(a.a2_gallery):
        print("缺少图库层回收文件：", a.a2_gallery)
        return 2
    raw_q = rd(a.a2_query) if os.path.exists(a.a2_query) else []
    if not raw_q:
        print("⚠ 查询层回收文件缺失 → 本次只算图库层。")

    rows = collect(rd(a.a2_gallery), raw_q, gt_g, gt_q)
    if not rows:
        print("没有两人都标过的条目。请确认列名与编号未变。")
        return 2

    # ---- 主口径：全部条目估计映射，再分层统计 ----
    mp, one_to_many = build_map(rows)
    aligned_all = relabel(rows, mp)

    out = {"method": {
        "primary": "many-to-one：每个 A3 编号取共现最多的 A1 编号；映射用全部条目估计",
        "sensitivity": "eval_kappa_a3.align()：贪心 A1->A3 后 setdefault 反演",
        "mapping_estimated_on": len(rows),
        "one_to_many": one_to_many,
        "a3_names_unmapped_under_legacy": None,
    }, "items_total": len(rows), "items": {}, "layers": {}}

    print(f"\n条目 {len(rows)} 条 | A3 编号 {len({r[3] for r in rows if r[3] != 'UNK'})} 个"
          f" | 一对多 {len(one_to_many)} 个\n")
    print(f"{'层':<10}{'n':>5}{'po':>8}{'κ':>8}{'95%CI':>18}{'仅人物κ':>9}{'非人合并κ':>10}")
    for name in ("gallery", "query"):
        sub = [r for r in rows if r[1] == name]
        if not sub:
            continue
        st = stats(relabel(sub, mp))
        out["layers"][name] = st
        out["items"][name] = len(sub)
        show(name, st)
    st_pool = stats(aligned_all)
    out["pooled"] = st_pool
    show("pooled", st_pool)

    print("\n口径变体（把 hat1/cloth 这类物件编号也视为非人，事后口径）：")
    out["variant_object_as_nonperson"] = {}
    for name, sub in (("gallery", [r for r in rows if r[1] == "gallery"]),
                      ("query", [r for r in rows if r[1] == "query"]),
                      ("pooled", rows)):
        if not sub:
            continue
        st = stats(relabel(sub, mp), drop_obj=True)
        out["variant_object_as_nonperson"][name] = st
        print(f"  {name:<9} n={st['n']:<4} 非人合并κ={st['nonperson_collapsed']['k']:.3f}"
              f"  仅人物κ={st['person_only']['k']:.3f}")

    # ---- 敏感性：原脚本的对齐 ----
    m2to1, _ = align([(r[2], r[3]) for r in rows])
    legacy_unmapped = sorted({r[3] for r in rows if r[3] != "UNK"} - set(m2to1))
    out["method"]["a3_names_unmapped_under_legacy"] = legacy_unmapped
    out["sensitivity_legacy_align"] = {"layers": {}}
    print(f"\n敏感性对照（原 align()）：被挤掉的 A3 编号 {legacy_unmapped}")
    for name, sub in (("gallery", [r for r in rows if r[1] == "gallery"]),
                      ("query", [r for r in rows if r[1] == "query"]),
                      ("pooled", rows)):
        if not sub:
            continue
        st = stats(relabel(sub, m2to1))
        out["sensitivity_legacy_align"]["layers"][name] = st
        print(f"  {name:<9} n={st['n']:<4} po={st['po']:.3f} κ={st['k']:.3f}")

    if a.loo:
        print("\n留一法映射（去掉 in-sample 过拟合）：")
        loo = kappa_loo(rows)
        out["loo_mapping"] = loo
        for name in ("gallery", "query", "pooled"):
            if name not in loo:
                continue
            st = loo[name]
            print(f"  {name:<9} n={st['n']:<4} po={st['po']:.3f} κ={st['k']:.3f}"
                  f"  CI [{st['ci95'][0]:.3f}, {st['ci95'][1]:.3f}]")
        print(f"  映射翻转 {loo['_mapping_flips']}/{len(rows)} 条"
              f"（{loo['_flip_rate']:.1%}）")

    # ---- 分歧清单 ----
    diff = []
    for (k, L, a1, a2, cam), (_, mapped) in zip(rows, aligned_all):
        if a1 != mapped:
            diff.append(dict(item=k, layer=L, a1=a1, a2=a2, mapped=mapped, cam=cam))
    out["disagreements"] = diff
    q = sum(1 for d in diff if d["layer"] == "query")
    print(f"\n真分歧 {len(diff)} 条（图库 {len(diff) - q} / 查询 {q}）")

    os.makedirs(RES, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("已写", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
