#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为「真正的第二标注人」生成分层抽样清单。

思路：判断「这条查询是不是图库里的某人」必须有自己的身份库，
所以图库层不能抽样（否则某些身份根本没建过档），查询层才抽。
  - 图库层：全量 257 条（约 45 分钟，可接受）
  - 查询层：分层抽 100 条（机位 × 簇纯度 × box_type），约 20 分钟

产出：
  gt/a3_gallery_TEMPLATE.csv   图库全量，空表
  gt/a3_query_TEMPLATE.csv     100 条分层抽样，空表
  gt/a3_SAMPLING.md            抽样说明与分层分布
"""
import csv
import collections
import os
import random

GT = "deliverable/data-request-v2/gt"
RES = "deliverable/data-request-v2/results"

N = 100
SEED = 20261019


def load(p, key):
    with open(p, encoding="utf-8-sig") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def main():
    random.seed(SEED)
    d4 = load(os.path.join(GT, "gt_query_gt_FILLED.csv"), "query_id")
    d3 = load(os.path.join(GT, "gt_gallery_members_FILLED.csv"), "track_id")
    vd = load(os.path.join(RES, "verdicts", "A_baseline.csv"), "query_id")
    geo = {int(r["cluster_idx"]): r
           for r in csv.DictReader(open("deliverable/data-request-v2/ledger/clusters_geom.csv",
                                        encoding="utf-8-sig"))
           if r["config"] == "baseline"}
    mem = list(csv.DictReader(open(os.path.join(RES, "members", "A_baseline.csv"),
                                   encoding="utf-8-sig")))
    lab2gid = {int(g["cluster_lab"]): int(g["cluster_idx"]) for g in geo.values()}
    loc2gid = {int(r["cluster_local"]): lab2gid[int(r["cluster_lab"])] for r in mem}

    # 每条可评估查询：机位 / box_type / 是否落混人簇
    pool = []
    for k, r in d4.items():
        if r.get("evaluable") != "1":
            continue
        mixed = ""
        if k in vd:
            loc = int(vd[k]["cluster_local"])
            g = geo.get(loc2gid.get(loc, -1))
            mixed = "mixed" if (g and g["purity_verdict"] == "bad") else "pure"
        pool.append(dict(k=k, cam=r["cam"], box=r["box_type"], mixed=mixed or "nocluster",
                         r=r))
    print(f"可评估查询 {len(pool)} 条")

    # 分层：cam × box，每层按比例分配，最小 2 条
    strata = collections.defaultdict(list)
    for x in pool:
        strata[(x["cam"], x["box"])].append(x)
    total = len(pool)
    alloc = {}
    for key, lst in strata.items():
        alloc[key] = max(2, round(N * len(lst) / total))
    # 修到恰好 N
    while sum(alloc.values()) > N:
        k = max(alloc, key=lambda z: (alloc[z], -len(strata[z])))
        if alloc[k] > 2:
            alloc[k] -= 1
        else:
            break
    while sum(alloc.values()) < N:
        k = max(strata, key=lambda z: len(strata[z]) - alloc[z])
        alloc[k] += 1

    picked = []
    for key, n in sorted(alloc.items()):
        lst = strata[key]
        n = min(n, len(lst))
        picked += random.sample(lst, n)

    # 补足到 N（用未抽中的随机补）
    if len(picked) < N:
        rest = [x for x in pool if x not in picked]
        picked += random.sample(rest, min(N - len(picked), len(rest)))
    picked = picked[:N]

    print(f"抽样 {len(picked)} 条")
    print("\n分层分布：")
    print(f"{'机位':<6}{'box':<9}{'总体':>5}{'抽中':>5}{'比例':>7}")
    for key in sorted(strata):
        n_all = len(strata[key])
        n_pick = sum(1 for x in picked if (x["cam"], x["box"]) == key)
        print(f"{key[0]:<6}{key[1]:<9}{n_all:>5}{n_pick:>5}{n_pick / n_all:>7.0%}")

    print("\n簇纯度覆盖：")
    c = collections.Counter(x["mixed"] for x in picked)
    for k, n in c.most_common():
        tot = sum(1 for x in pool if x["mixed"] == k)
        print(f"  {k:<11}{n:>4} / {tot:<4}（{n / tot:.0%}）")

    # ---------- 写模板 ----------
    # 图库层全量（沿用 D3 模板，换名）
    src = os.path.join(GT, "gallery_members_TEMPLATE.csv")
    dst = os.path.join(GT, "a3_gallery_TEMPLATE.csv")
    with open(src, encoding="utf-8-sig") as fi, open(dst, "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.writer(fo)
        rd = csv.reader(fi)
        head = next(rd)
        w.writerow(head)
        for row in rd:
            row[head.index("person_id")] = ""
            row[head.index("box_type")] = ""
            row[head.index("unk_reason")] = ""
            row[head.index("note")] = ""
            w.writerow(row)
    n_g = sum(1 for _ in open(dst, encoding="utf-8-sig")) - 1
    print(f"\n[out] {dst}: {n_g} 行")

    # 查询层抽样
    head = list(next(iter(d4.values())).keys())
    keep = ["query_id", "seg", "cam", "n_frames", "median_box_h", "evaluable",
            "exclude_reason", "prompt_jpg", "person_id", "box_type", "duplicate_of",
            "unk_reason", "evidence", "annotator", "time_spent_s"]
    dst2 = os.path.join(GT, "a3_query_TEMPLATE.csv")
    rows = []
    for x in sorted(picked, key=lambda z: (z["cam"], z["k"])):
        r = {k: x["r"].get(k, "") for k in keep}
        r["person_id"] = ""
        r["box_type"] = ""
        r["unk_reason"] = ""
        r["evidence"] = ""
        r["annotator"] = ""
        r["time_spent_s"] = ""
        rows.append(r)
    with open(dst2, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keep)
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {dst2}: {len(rows)} 行")

    # ---------- 抽样说明 ----------
    md = os.path.join(GT, "a3_SAMPLING.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("# 第二标注人抽样方案（A3）\n\n")
        fh.write(f"> 随机种子 `{SEED}`｜生成脚本 `tools/make_a3_sample.py`\n\n")
        fh.write("## 为什么图库层不抽样\n\n")
        fh.write("判断「这条查询是不是图库里的某人」，前提是自己建过图库档。\n")
        fh.write("若抽样图库，某些身份根本不会出现在档里，查询层就没法判。\n")
        fh.write("所以：**图库层全量 257 条，查询层分层抽 100 条**。\n\n")
        fh.write("## 分层依据\n\n")
        fh.write("按 `机位 × box_type` 分层，按总体比例分配，每层至少 2 条；\n")
        fh.write("同时核对「落在混人簇 / 纯簇」的覆盖，避免抽样只落在容易的样本上。\n\n")
        fh.write("| 机位 | box_type | 总体 | 抽中 | 比例 |\n|---|---|---|---|---|\n")
        for key in sorted(strata):
            n_all = len(strata[key])
            n_pick = sum(1 for x in picked if (x["cam"], x["box"]) == key)
            fh.write(f"| {key[0]} | {key[1]} | {n_all} | {n_pick} | {n_pick / n_all:.0%} |\n")
        fh.write("\n| 簇纯度 | 抽中 | 总体 | 覆盖 |\n|---|---|---|---|\n")
        for k, n in c.most_common():
            tot = sum(1 for x in pool if x["mixed"] == k)
            fh.write(f"| {k} | {n} | {tot} | {n / tot:.0%} |\n")
        fh.write("\n## 执行方式\n\n")
        fh.write("1. 让第二人用**无痕窗口**打开 `annotate_standalone.html`。\n")
        fh.write("2. 切到 **D3 标签页**，按机位顺序标完 257 条（约 45 分钟）。\n")
        fh.write("3. 切到 **D4 标签页**，只标 `a3_query_TEMPLATE.csv` 里列出的那 100 条，其余跳过。\n")
        fh.write("4. 导出两个 CSV，连同本说明一起交回。\n\n")
        fh.write("## 统计口径\n\n")
        fh.write("- 在**共有的 100 条查询**上算 Cohen's κ；图库层 257 条可另算一个 κ（衡量建档一致性）。\n")
        fh.write("- 编号体系**必然不同**（第二人会自己命名），必须先用 `tools/align_annotators.py` 做共现对齐，再算 κ。\n")
        fh.write("- 对齐后仍分歧的条目，用 `tools/build_kappa_review.py` 生成定案页。\n")
        fh.write("- 报 κ 时同时给 95% CI：n=100、po≈0.93、pe≈0.12 时标准误约 0.029。\n")
    print(f"[out] {md}")

    # 落细目，便于复核
    det = os.path.join(GT, "a3_query_sample_ids.txt")
    with open(det, "w", encoding="utf-8") as fh:
        for x in sorted(picked, key=lambda z: (z["cam"], z["k"])):
            fh.write(f"{x['k']}\t{x['cam']}\t{x['box']}\t{x['mixed']}\n")
    print(f"[out] {det}")


if __name__ == "__main__":
    main()
