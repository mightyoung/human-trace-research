#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""留出复验标注模板生成（冻结版）—— 实现预注册 §3.1 标注协议的范围规则。

范围（与主实验 D3/D4 相同）：
  - 查询层：A_baseline 的全部可评查询（queries/A_baseline.csv 每一行）。
  - 图库层：在 A_baseline 或 C_dmax0.30 中至少接到 1 条已接受查询的簇，取其全部成员（并集）。
盲标：模板只含轨迹编号、段名、机位，不含簇号、分数、是否接受；行序按固定种子打乱。

用法：
  python tools/holdout_make_templates.py --run output_crosscam_local/holdout/eval/results \
         --out output_crosscam_local/holdout/eval/gt
输出：
  gallery_members_TEMPLATE.csv   track_id, seg, cam, person_id, box_type, unk_reason, annotator, note
  query_gt_TEMPLATE.csv          query_id, seg, cam, person_id, box_type, unk_reason, annotator, note
  scope.json                     范围统计（簇数、轨迹数、查询数）
标注完成后另存为 gt_gallery_members_FILLED.csv / gt_query_gt_FILLED.csv，供 holdout_eval.py 读取。
"""
import argparse
import collections
import csv
import json
import os
import random

SEED = 20261024
SCOPE_CFGS = ("A_baseline", "C_dmax0.30")
EXTRA = ["person_id", "box_type", "unk_reason", "annotator", "note"]


def rd(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tracks, clusters = set(), 0
    for cfg in SCOPE_CFGS:
        mem = collections.defaultdict(list)
        for r in rd(os.path.join(a.run, "members", cfg + ".csv")):
            mem[int(r["cluster_local"])].append(r["track_id"])
        hit = {int(r["final_cluster_local"]) for r in rd(os.path.join(a.run, "queries", cfg + ".csv"))
               if r["final_status"] == "accepted"}
        clusters += len(hit)
        for cl in hit:
            tracks.update(mem[cl])

    queries = [r["query_id"] for r in rd(os.path.join(a.run, "queries", "A_baseline.csv"))]
    rng = random.Random(SEED)
    g = sorted(tracks)
    q = sorted(queries)
    rng.shuffle(g)
    rng.shuffle(q)

    def row(key, name):
        seg = key.split("#")[0]
        return {name: key, "seg": seg, "cam": seg.split("_")[0], **{k: "" for k in EXTRA}}

    with open(os.path.join(a.out, "gallery_members_TEMPLATE.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["track_id", "seg", "cam"] + EXTRA)
        w.writeheader()
        w.writerows(row(t, "track_id") for t in g)
    with open(os.path.join(a.out, "query_gt_TEMPLATE.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_id", "seg", "cam"] + EXTRA)
        w.writeheader()
        w.writerows(row(x, "query_id") for x in q)
    scope = dict(scope_configs=list(SCOPE_CFGS), hit_clusters_summed=clusters,
                 gallery_tracks=len(g), queries=len(q), seed=SEED)
    with open(os.path.join(a.out, "scope.json"), "w", encoding="utf-8") as fh:
        json.dump(scope, fh, ensure_ascii=False, indent=2)
    print(json.dumps(scope, ensure_ascii=False))


if __name__ == "__main__":
    main()
