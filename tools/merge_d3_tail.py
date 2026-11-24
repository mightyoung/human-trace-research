#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并 D3 收尾 20 条（4 个此前从未标注的图库簇：61 / 32 / 34 / 16）。

输入
  gt/review_d3_tail_FILLED.csv      标注页导出（track_id, cluster_idx, person_id, box_type, unk_reason, note）
  gt/gallery_members_TEMPLATE.csv   257 行完整名单（提供簇 61 四行的其余列）
  gt/gt_gallery_members_FILLED.csv  现有人工标注（253 行）
  ledger/tracks.csv                 补 box_type

输出
  gt/gt_gallery_members_FILLED.csv  257 行，无空缺
  gt/_backup/gt_gallery_members_FILLED.<时间戳>.csv

约定
  person_id = no-human   该簇内并非人体（家具/椅子/反光等误检），不计入「身份数」
  person_id = shadow*    非名册身份（画面里存在但无法归到 R1–R29 的同一对象/同一人），
                         记为独立身份参与「簇纯度」判定，但不参与名册匹配评估
"""
import csv
import datetime as dt
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GT = os.path.join(ROOT, "deliverable/data-request-v2/gt")
LEDGER = os.path.join(ROOT, "deliverable/data-request-v2/ledger")

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(GT, "review_d3_tail_FILLED.csv")
TEMPLATE = os.path.join(GT, "gallery_members_TEMPLATE.csv")
DST = os.path.join(GT, "gt_gallery_members_FILLED.csv")
BACKUP = os.path.join(GT, "_backup")


def main():
    fill = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    tpl = list(csv.DictReader(open(TEMPLATE, encoding="utf-8-sig")))
    cur = list(csv.DictReader(open(DST, encoding="utf-8-sig")))
    tracks = {r["track_id"]: r for r in csv.DictReader(
        open(os.path.join(LEDGER, "tracks.csv"), encoding="utf-8-sig"))}
    cols = list(tpl[0].keys())

    # ---- 校验 1：行数 / 重复 / 归属
    assert len(fill) == 20, f"应为 20 条，实得 {len(fill)}"
    keys = [r["track_id"] for r in fill]
    assert len(set(keys)) == 20, "有重复 track_id"
    by_cluster = {}
    for r in fill:
        by_cluster.setdefault(r["cluster_idx"], set()).add((r["person_id"] or "").strip())
    for c, ids in sorted(by_cluster.items(), key=lambda kv: int(kv[0])):
        if len(ids) > 1:
            print(f"  [warn] 簇 {c} 出现多个身份：{sorted(ids)} —— 按「混人簇」处理")

    # ---- 校验 2：track_id 必须能在模板里找到，且簇号一致
    tpl_by = {r["track_id"]: r for r in tpl}
    for r in fill:
        t = tpl_by.get(r["track_id"])
        assert t is not None, f"模板里没有 {r['track_id']}"
        assert t["cluster_idx"] == r["cluster_idx"], \
            f"簇号不一致 {r['track_id']}: 页面 {r['cluster_idx']} vs 台账 {t['cluster_idx']}"

    # ---- 备份
    os.makedirs(BACKUP, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(DST, os.path.join(BACKUP, f"gt_gallery_members_FILLED.{ts}.csv"))

    # ---- 合并
    cur_by = {r["track_id"]: r for r in cur}
    n_fill, n_add = 0, 0
    for r in fill:
        pid = (r["person_id"] or "").strip()
        box = (r["box_type"] or "").strip() or tracks.get(r["track_id"], {}).get("box_type_auto", "body")
        note = (r["note"] or "").strip() or "d3_tail"
        unk = (r["unk_reason"] or "").strip()
        if r["track_id"] in cur_by:
            x = cur_by[r["track_id"]]
            x["person_id"] = pid
            x["box_type"] = box
            x["unk_reason"] = unk
            x["note"] = note
            x["annotator"] = "user"
            n_fill += 1
        else:
            t = dict(tpl_by[r["track_id"]])
            t["person_id"] = pid
            t["box_type"] = box
            t["unk_reason"] = unk
            t["note"] = note
            t["annotator"] = "user"
            t["time_spent_s"] = ""
            cur.append(t)
            n_add += 1

    cur.sort(key=lambda r: (int(r["cluster_idx"]), r["track_id"]))
    with open(DST, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in cols} for r in cur)

    blank = [r["track_id"] for r in cur if not (r["person_id"] or "").strip()]
    print(f"[out] {DST}")
    print(f"  回填 {n_fill} 条 / 新增 {n_add} 条 → 共 {len(cur)} 行")
    print(f"  person_id 仍为空 {len(blank)} 条: {blank}")
    for c in sorted(by_cluster, key=int):
        print(f"    簇 {c:>3}: {len([r for r in fill if r['cluster_idx'] == c])} 条 → "
              f"{sorted(by_cluster[c])}")


if __name__ == "__main__":
    main()
