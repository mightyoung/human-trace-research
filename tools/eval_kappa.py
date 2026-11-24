#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D5 第二标注人一致性分析：与第一人逐条比对，算一致率与 Cohen's kappa。

两层的对照来源：
  gallery → gt_gallery_members_FILLED.csv（已合并 review_d3 复核修正）
  query   → gt_query_gt_FILLED.csv

同时量化「是否独立盲标」：比对 note 文本相似度与编号体系重合度。
"""
import csv
import collections
import os
import sys

GT = "deliverable/data-request-v2/gt"


def load(fn, key="item_id"):
    with open(os.path.join(GT, fn), encoding="utf-8-sig") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def norm(pid):
    p = (pid or "").strip()
    if not p:
        return "UNK"
    if p.upper() == "UNK":
        return "UNK"
    return p


def kappa(pairs):
    """pairs: list of (a, b)。返回 (n, po, pe, kappa)。"""
    n = len(pairs)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return n, po, pe, k


def main():
    a2 = load("gt_annotator2_FILLED.csv")
    a1_g = load("gt_gallery_members_FILLED.csv", key="track_id")
    a1_q = load("gt_query_gt_FILLED.csv", key="query_id")

    print(f"D5 共 {len(a2)} 条")
    g = [k for k, r in a2.items() if r["layer"] == "gallery"]
    q = [k for k, r in a2.items() if r["layer"] == "query"]
    print(f"  gallery {len(g)} 条 / query {len(q)} 条")

    # ---------- 1. 逐条配对 ----------
    pairs, conflicts, missing_ref = [], [], []
    for k in g:
        r1 = a1_g.get(k)
        if r1 is None:
            missing_ref.append(("gallery", k))
            continue
        pairs.append((norm(a1_g[k]["person_id"]), norm(a2[k]["person_id"]), "gallery", k))
    for k in q:
        r1 = a1_q.get(k)
        if r1 is None:
            missing_ref.append(("query", k))
            continue
        pairs.append((norm(a1_q[k]["person_id"]), norm(a2[k]["person_id"]), "query", k))

    print(f"  可比对 {len(pairs)} 条｜第一人缺参照 {len(missing_ref)} 条")
    if missing_ref:
        print("   ", missing_ref[:6])

    # ---------- 2. 一致率与 kappa ----------
    print("\n=== 逐条一致率与 Cohen's kappa ===")
    print(f"{'层':<9}{'n':>4}{'一致':>5}{'一致率':>8}{'kappa':>8}")
    allp = [(a, b) for a, b, _, _ in pairs]
    for layer in ["gallery", "query", "all"]:
        sub = allp if layer == "all" else [(a, b) for a, b, L, _ in pairs if L == layer]
        n, po, pe, k = kappa(sub)
        agree = sum(1 for a, b in sub if a == b)
        print(f"{layer:<9}{n:>4}{agree:>5}{po:>8.3f}{k:>8.3f}")
    print(f"  （all 的期望一致率 pe = {kappa(allp)[2]:.3f}）")

    # ---------- 3. 分歧清单 ----------
    print("\n=== 全部分歧 ===")
    diff = [(L, k, a, b) for a, b, L, k in pairs if a != b]
    print(f"共 {len(diff)} 条")
    for L, k, a, b in diff:
        print(f"  {L:<8}{k:<22} 一人 {a:<9} → 二人 {b}")

    # ---------- 4. 分歧类型 ----------
    print("\n=== 分歧类型 ===")
    cat = collections.Counter()
    female4_g = 0
    for L, k, a, b in diff:
        if "UNK" in (a, b):
            cat["一方看不清/未标"] += 1
        else:
            cat["两人都给了身份但不同"] += 1
        if "female4" in (a, b):
            female4_g += 1
    print(f"  {dict(cat)}")
    print(f"  涉及已作废编号 female4 的分歧: {female4_g} 条")

    # ---------- 5. 独立性检验：note 相似度 ----------
    print("\n=== 独立性检验 ===")
    both_note, same_note, one_note = 0, 0, 0
    for k in g:
        if k not in a1_g:
            continue
        n1 = (a1_g[k].get("note") or "").strip().lower()
        n2 = (a2[k].get("note") or "").strip().lower()
        if n1 and n2:
            both_note += 1
            if n1 == n2:
                same_note += 1
        elif n1 or n2:
            one_note += 1
    print(f"  两人都有备注的条目: {both_note} 条，其中**逐字相同** {same_note} 条"
          f"（{same_note / both_note:.0%}）")
    print(f"  仅一方有备注: {one_note} 条")

    s1 = {norm(r["person_id"]) for r in a1_g.values()} | {norm(r["person_id"]) for r in a1_q.values()}
    s2 = {norm(r["person_id"]) for r in a2.values()}
    print(f"  第一人身份集合 {len(s1)} 个；第二人 {len(s2)} 个；交集 {len(s1 & s2)} 个"
          f"（{len(s1 & s2) / len(s2):.0%}）")
    print(f"  第二人独有: {sorted(s2 - s1)}")
    print(f"  第一人有、第二人未用: {sorted(s1 - s2)}")

    # ---------- 6. box_type / unk 一致 ----------
    print("\n=== 辅助字段一致率 ===")
    bt = [(a1_g[k]["box_type"].strip(), a2[k]["box_type"].strip())
          for k in g if k in a1_g]
    bt += [(a1_q[k]["box_type"].strip(), a2[k]["box_type"].strip())
           for k in q if k in a1_q]
    n, po, pe, k_ = kappa(bt)
    print(f"  box_type: n={n} 一致率 {po:.3f} kappa {k_:.3f}")

    ur = [(a1_q[k]["unk_reason"].strip(), a2[k]["unk_reason"].strip())
          for k in q if k in a1_q]
    n, po, pe, k_ = kappa(ur)
    print(f"  unk_reason: n={n} 一致率 {po:.3f} kappa {k_:.3f}")

    # ---------- 7. 若剔除 female4 与 UNK ----------
    print("\n=== 敏感性：剔除 female4 相关与 UNK 后 ===")
    clean = [(a, b) for a, b, L, kk in pairs
             if a != "UNK" and b != "UNK" and "female4" not in (a, b)]
    n, po, pe, k_ = kappa(clean)
    print(f"  可比 {n} 条｜一致率 {po:.3f}｜kappa {k_:.3f}")


if __name__ == "__main__":
    main()
