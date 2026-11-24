# -*- coding: utf-8 -*-
"""把原始 json 里的**判定证据**回填到 `label.data_json`（读切换的前置一次性迁移）。

背景：镜像/摄入只把判定主值写进 `label.value`（`'0'`/`'CONF'`/`'pid:28'`），
置信度 `c`、自由备注 `n`、`pid`+`fp` 并存、人脸 `name` 全部丢失。
B1「标注页读切换到台账」后页面只认台账，这些字段必须先在台账里存在。

来源与台账的对应关系（8 个 task 全部一一对应，已实测）：

    labels.json.pair_decisions   -> t_pair_318x320 / t_s6b_pending（按 S6B_/PRT_ 前缀路由）
    labels.json.mirror_decisions -> t_mirror_6cam
    labels.json.glass_masks      -> t_mask_glass
    truth/labels.json.pair       -> t_truth_pair / t_s6b_pending（同上路由）
    truth/labels.json.event      -> t_truth_event
    truth/seed_labels.json       -> t_truth_pair（仅补台账里完全没有判定的条目）
    faces_labels.json.persons    -> t_face_person
    faces_labels.json.pairs      -> t_face_pair

用法：

    python tools/ops_backfill_labeldata.py            # 预演（默认不写）
    python tools/ops_backfill_labeldata.py --apply    # 实际写入
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops_db as db          # noqa: E402
import ops_ingest as ing     # noqa: E402

ROOT = db.ROOT


def rj(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect() -> list[tuple[str, str, object, str]]:
    """返回 [(item_id, annotator, data, 来源说明)]，顺序即优先级（后者覆盖前者）。"""
    out: list[tuple[str, str, object, str]] = []
    labels = rj(ing.LABELS, {}) or {}
    truth = rj(ing.TRUTH_LABELS, {}) or {}
    faces = rj(ing.FACE_LABELS, {}) or {}
    seed = rj(ing.TRUTH_SEED, {}) or {}

    for k, v in (labels.get("pair_decisions") or {}).items():
        out.append((f"{ing._route_pair(k)}:{k}", "console", v, "labels.pair_decisions"))
    for k, v in (labels.get("mirror_decisions") or {}).items():
        out.append((f"t_mirror_6cam:{k}", "console", v, "labels.mirror_decisions"))
    for k, v in (labels.get("glass_masks") or {}).items():
        out.append((f"t_mask_glass:{k}", "console", v, "labels.glass_masks"))

    for pid, v in (truth.get("pair") or {}).items():
        task = "t_s6b_pending" if ing.is_s6b(pid) else "t_truth_pair"
        out.append((f"{task}:{pid}", "console", v, "truth.pair"))
    for eid, v in (truth.get("event") or {}).items():
        out.append((f"t_truth_event:{eid}", "console", v, "truth.event"))

    for lab, v in (faces.get("persons") or {}).items():
        out.append((f"t_face_person:{lab}", "console", v, "faces.persons"))
    for key, v in (faces.get("pairs") or {}).items():
        out.append((f"t_face_pair:{key}", "console", v, "faces.pairs"))

    # seed 最后：只在台账完全没有判定时补充（页面逻辑是 labels 覆盖 seed）
    for pid, v in seed.items():
        out.append((f"t_truth_pair:{pid}", "seed", v, "truth.seed"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=db.DB_PATH)
    ap.add_argument("--apply", action="store_true", help="实际写入（默认只预演）")
    a = ap.parse_args()

    conn = db.connect(a.db)
    db.init_db(conn)
    stat = {"updated": 0, "inserted": 0, "same": 0, "skip_no_item": 0, "skip_has_label": 0}
    details: list[str] = []
    try:
        for item_id, who, data, src in collect():
            row = conn.execute("SELECT id, task_id FROM annotation_item WHERE id=?",
                               (item_id,)).fetchone()
            if not row:
                stat["skip_no_item"] += 1
                details.append(f"  跳过（item 不存在）{item_id} ← {src}")
                continue
            lab = db.latest_label(conn, item_id)
            if lab is None:
                if src != "truth.seed":
                    stat["skip_has_label"] += 1
                    continue
                # seed 只在完全无判定时补一条（页面合并逻辑：labels 覆盖 seed）
                stat["inserted"] += 1
                details.append(f"  新增 seed 判定 {item_id} = {json.dumps(data, ensure_ascii=False)}")
                if a.apply:
                    db.set_label(conn, item_id, str((data or {}).get("v")), who,
                                 ui_version="seed", note="backfill:truth.seed", data=data)
                continue
            have = lab["data_json"]
            want = db.jdump(data)
            if have == want:
                stat["same"] += 1
                continue
            if src == "truth.seed":
                stat["skip_has_label"] += 1
                continue
            stat["updated"] += 1
            details.append(f"  补证据 {item_id} ← {src}: {have or '(空)'} → {want}")
            if a.apply:
                conn.execute("UPDATE label SET data_json=? WHERE id=?", (want, int(lab["id"])))
                db.audit(conn, "backfill", "label.data", "annotation_item", item_id,
                         {"data": db.jload(have)}, {"data": data, "source": src})
        if a.apply:
            db.audit(conn, "backfill", "label.backfill", "label", "all",
                     None, {**stat, "mode": "apply"})
    finally:
        conn.close()

    print(("写入" if a.apply else "预演") + f"：{stat}")
    if details:
        print(f"明细（前 30 条）：")
        for line in details[:30]:
            print(line)
        if len(details) > 30:
            print(f"  …… 其余 {len(details) - 30} 条略")
    if not a.apply:
        print("未写库。确认无误后加 --apply。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
