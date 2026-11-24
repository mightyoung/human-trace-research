#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 D3/D4/D5 的标注模板（预填待标清单，person_id 留空）。

簇号一律用 `ledger/clusters_geom.csv` 的 `cluster_idx`（全局簇号，跨配置可比，
与 `cluster_purity.csv` 的 `c` 列、论文里的"簇 36"同一口径）。

D3 待标范围：baseline 与 veto 两个配置下被任何查询命中过的簇，取并集。

用法：python tools/make_gt_templates.py
"""
from __future__ import annotations

import argparse
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "deliverable/data-request-v2/gt")
LEDGER = os.path.join(ROOT, "deliverable/data-request-v2/ledger")
KEY_CLUSTERS = [36, 29, 101, 13, 47, 69, 55]   # 论文点名的关键簇（全局簇号）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tracks = list(csv.DictReader(open(os.path.join(LEDGER, "tracks.csv"), encoding="utf-8-sig")))
    trk = {(r["seg"], int(r["tid"])): r for r in tracks}
    geo = list(csv.DictReader(open(os.path.join(LEDGER, "clusters_geom.csv"), encoding="utf-8-sig")))
    base = {int(r["cluster_idx"]): r for r in geo if r["config"] == "baseline"}
    # 全局簇号 -> 图库成员 track_id（用 tracks.csv 的 evaluable 图库行 + geom 的成员数核对）
    gid_lab = {g: int(r["cluster_lab"]) for g, r in base.items()}
    gid_n = {g: int(r["n"]) for g, r in base.items()}
    print(f"baseline 簇 {len(base)} 个（全局簇号 0..{max(base)}）")

    # 命中簇：两配置下出现过最终输出的簇
    hit = {}
    for cfg in ("baseline", "veto0.30"):
        p = os.path.join(LEDGER, f"outputs/{cfg}.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            if r["final_cluster_idx"]:
                hit.setdefault(int(r["final_cluster_idx"]), set()).add(cfg)
    scope = sorted(set(hit) | set(KEY_CLUSTERS))
    print(f"待标簇 {len(scope)} 个（被命中 {len(hit)} ∪ 关键簇 {len(KEY_CLUSTERS)}）")

    # 图库成员：需要按 lab 反查。tracks.csv 不含簇号，故从 geom 的 cams/n 校验后由 lab 反查
    # 简化做法：用 clusters_geom 的 cluster_lab + 重新载入 npz 顺序不可靠，
    # 这里改用「导出时已确定的成员表」——若不存在则退化为只给簇清单。
    mem_path = os.path.join(LEDGER, "gallery_members_by_cluster.csv")
    rows = []
    if os.path.exists(mem_path):
        for r in csv.DictReader(open(mem_path, encoding="utf-8-sig")):
            if int(r["cluster_idx"]) in scope:
                rows.append(r)
    if not rows:
        print("[warn] 缺 ledger/gallery_members_by_cluster.csv，D3 成员表将由 export 脚本产出")

    p1 = os.path.join(a.out, "gallery_members_TEMPLATE.csv")
    with open(p1, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["cluster_idx", "cluster_n", "cluster_verdict",
                                           "track_id", "seg", "cam", "hits_by", "n_frames",
                                           "prompt_jpg", "person_id", "box_type", "id_switch",
                                           "id_switch_frame", "unk_reason", "annotator",
                                           "time_spent_s", "note"])
        w.writeheader()
        for r in rows:
            g = int(r["cluster_idx"])
            tr = trk.get((r["seg"], int(r["tid"])), {})
            w.writerow({"cluster_idx": g, "cluster_n": gid_n.get(g, ""),
                        "cluster_verdict": base.get(g, {}).get("purity_verdict", ""),
                        "track_id": r["track_id"], "seg": r["seg"], "cam": r["cam"],
                        "hits_by": "|".join(sorted(hit.get(g, []))),
                        "n_frames": tr.get("n_frames", ""),
                        "prompt_jpg": f"crops/{r['seg']}/{r['tid']}.jpg"})
    print(f"[D3] {p1}: {len(rows)} 个图库成员（覆盖 {len({int(r['cluster_idx']) for r in rows})} 簇）")

    # D4
    qrows = []
    for r in tracks:
        if r["split"] != "query":
            continue
        qrows.append({"query_id": r["track_id"], "seg": r["seg"], "cam": r["cam"],
                      "n_frames": r["n_frames"], "median_box_h": r["median_box_h"],
                      "evaluable": r["evaluable"], "exclude_reason": r["exclude_reason"],
                      "prompt_jpg": f"crops/{r['seg']}/{r['tid']}.jpg",
                      "person_id": "", "box_type": "", "duplicate_of": "",
                      "unk_reason": "", "evidence": "", "annotator": "", "time_spent_s": ""})
    p2 = os.path.join(a.out, "query_gt_TEMPLATE.csv")
    with open(p2, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qrows[0].keys()))
        w.writeheader()
        w.writerows(qrows)
    print(f"[D4] {p2}: {len(qrows)} 条（其中可评估 {sum(1 for x in qrows if x['evaluable']=='1')}）")

    # D5：关键簇成员 + 全部被接受的查询
    acc = set()
    for cfg in ("baseline", "veto0.30"):
        p = os.path.join(LEDGER, f"outputs/{cfg}.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                if r["final_status"] == "accepted":
                    acc.add(r["query_id"])
    k5 = [x for x in rows if int(x["cluster_idx"]) in KEY_CLUSTERS]
    q5 = [x for x in qrows if x["query_id"] in acc]
    p3 = os.path.join(a.out, "annotator2_TEMPLATE.csv")
    with open(p3, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["layer", "item_id", "cam", "prompt_jpg",
                                           "person_id", "box_type", "unk_reason", "note"])
        w.writeheader()
        for x in k5:
            w.writerow({"layer": "gallery", "item_id": x["track_id"], "cam": x["cam"],
                        "prompt_jpg": f"crops/{x['seg']}/{x['tid']}.jpg"})
        for x in q5:
            w.writerow({"layer": "query", "item_id": x["query_id"], "cam": x["cam"],
                        "prompt_jpg": x["prompt_jpg"]})
    print(f"[D5] {p3}: 关键簇成员 {len(k5)} + 被接受查询 {len(q5)} = {len(k5)+len(q5)}")

    # 簇清单
    p4 = os.path.join(a.out, "cluster_checklist.csv")
    with open(p4, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_idx", "cluster_lab", "n_members", "n_cams", "d_max",
                    "purity_verdict", "hits_by", "n_members_done", "verdict", "note"])
        for g in scope:
            r = base.get(g, {})
            w.writerow([g, r.get("cluster_lab", ""), r.get("n", ""), r.get("n_cam", ""),
                        r.get("d_max", ""), r.get("purity_verdict", ""),
                        "|".join(sorted(hit.get(g, []))), "", "", ""])
    print(f"[清单] {p4}: {len(scope)} 行")


if __name__ == "__main__":
    main()
