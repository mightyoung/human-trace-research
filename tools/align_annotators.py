#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D5 编号体系对齐：用共同标注条目的共现矩阵，判断两人编号是否一一对应。

若映射近似双射（每个编号只对应一个），说明分歧是「叫法不同」；
若出现一对多/多对一，说明两人对该人群的划分本身不同，这才是真分歧。
"""
import csv
import collections
import os

GT = "deliverable/data-request-v2/gt"


def load(fn, key):
    with open(os.path.join(GT, fn), encoding="utf-8-sig") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def norm(p):
    p = (p or "").strip()
    return "UNK" if (not p or p.upper() == "UNK") else p


def main():
    a2 = load("gt_annotator2_FILLED.csv", "item_id")
    a1 = {}
    a1.update(load("gt_gallery_members_FILLED.csv", "track_id"))
    a1.update(load("gt_query_gt_FILLED.csv", "query_id"))

    co = collections.Counter()
    for k, r2 in a2.items():
        r1 = a1.get(k)
        if not r1:
            continue
        co[(norm(r1["person_id"]), norm(r2["person_id"]))] += 1

    # 一人的每个编号 → 二人在其上用了哪些编号
    by_a = collections.defaultdict(collections.Counter)
    by_b = collections.defaultdict(collections.Counter)
    for (a, b), n in co.items():
        by_a[a][b] += n
        by_b[b][a] += n

    print("=== 第一人编号 → 第二人编号（只列多对一的异常）===")
    print(f"{'一人':<9}{'条目':>4}  二人分布")
    anomaly_a = []
    for a, c in sorted(by_a.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(c.values())
        if len(c) > 1:
            anomaly_a.append(a)
            det = " ".join(f"{k}:{v}" for k, v in c.most_common())
            print(f"{a:<9}{tot:>4}  {det}")
    if not anomaly_a:
        print("  （无：一人的每个编号在二人那里都只对应一个编号）")

    print("\n=== 第二人编号 → 第一人编号（只列多对一的异常）===")
    print(f"{'二人':<9}{'条目':>4}  一人分布")
    anomaly_b = []
    for b, c in sorted(by_b.items(), key=lambda x: -sum(x[1].values())):
        tot = sum(c.values())
        if len(c) > 1:
            anomaly_b.append(b)
            det = " ".join(f"{k}:{v}" for k, v in c.most_common())
            print(f"{b:<9}{tot:>4}  {det}")
    if not anomaly_b:
        print("  （无：二人的每个编号在一人那里都只对应一个编号）")

    # 双射检验
    print("\n=== 是否构成双射 ===")
    one2one = sum(1 for a, c in by_a.items() if len(c) == 1)
    print(f"  一人的 {len(by_a)} 个编号中，{one2one} 个只映射到一个二人编号"
          f"（{one2one / len(by_a):.0%}）")
    one2one_b = sum(1 for b, c in by_b.items() if len(c) == 1)
    print(f"  二人的 {len(by_b)} 个编号中，{one2one_b} 个只映射到一个一人编号"
          f"（{one2one_b / len(by_b):.0%}）")

    print("\n=== 候选映射表（取共同条目最多的对应）===")
    mapping = {}
    for a, c in by_a.items():
        b, n = c.most_common(1)[0]
        mapping[a] = (b, n, sum(c.values()))
    rev = collections.defaultdict(list)
    for a, (b, n, tot) in mapping.items():
        rev[b].append(a)
    for b, lst in sorted(rev.items()):
        if len(lst) > 1:
            print(f"  二人 {b} ← 一人 {lst}（多对一，说明划分不同）")
    print(f"  映射后的编号对: {len(set(v[0] for v in mapping.values()))} 个二人编号"
          f" ← {len(mapping)} 个一人编号")

    # 对齐后的 kappa
    pairs = []
    for k, r2 in a2.items():
        r1 = a1.get(k)
        if not r1:
            continue
        pairs.append((mapping[norm(r1["person_id"])][0], norm(r2["person_id"])))
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    print(f"\n=== 按映射对齐后的 kappa ===")
    print(f"  n={n} 一致率 {po:.3f}  pe={pe:.3f}  kappa={(po - pe) / (1 - pe):.3f}")

    # 对齐后仍不一致的
    print("\n=== 对齐后仍不一致（真实分歧）===")
    left = [(k, norm(a1[k]["person_id"]), norm(a2[k]["person_id"]),
             mapping[norm(a1[k]["person_id"])][0])
            for k in a2 if k in a1 and mapping[norm(a1[k]["person_id"])][0] != norm(a2[k]["person_id"])]
    print(f"共 {len(left)} 条")
    for k, a, b, amap in left:
        print(f"  {k:<22} 一人 {a}（映射→{amap}）  二人 {b}")


if __name__ == "__main__":
    main()
