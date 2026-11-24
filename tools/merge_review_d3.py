"""把 review_d3_FILLED.csv 的复核结果合并回 gt_gallery_members_FILLED.csv。

规则：
- 只覆盖 person_id / id_switch / id_switch_frame / unk_reason；原 note 保留并按 `review` 标记追加。
- 复核页导出的 note 是**上一轮的旧备注**（不是本次输入），若与最终 person_id 冲突会单列告警。
- 备份原文件到 gt/_backup/，保证可回溯。
"""
import argparse
import collections
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "deliverable", "data-request-v2", "gt")
MAIN = os.path.join(GT, "gt_gallery_members_FILLED.csv")
REV = os.path.join(GT, "review_d3_FILLED.csv")


def rd(p):
    with open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    rows = rd(MAIN)
    rev = rd(REV)
    idx = {r["track_id"]: r for r in rows}

    print(f"主表 {len(rows)} 行｜复核表 {len(rev)} 行（task={collections.Counter(r['task'] for r in rev)}）")
    missing = [r["track_id"] for r in rev if r["track_id"] not in idx]
    if missing:
        print(f"  ⚠ 主表找不到: {missing}")
    print()

    changed, same, conflicts = [], [], []
    for r in rev:
        tid = r["track_id"]
        if tid not in idx:
            continue
        old = idx[tid]
        new_pid = r["person_id"].strip()
        old_pid = old["person_id"].strip()
        note = (r["note"] or "").strip()
        if new_pid != old_pid:
            changed.append((tid, old["cluster_idx"], old_pid, new_pid, note))
            old["person_id"] = new_pid
            for k_src, k_dst in [("id_switch", "id_switch"), ("id_switch_frame", "id_switch_frame"),
                                 ("unk_reason", "unk_reason")]:
                if r[k_src].strip():
                    old[k_dst] = r[k_src].strip()
            old["annotator"] = (old.get("annotator") or "checker").strip() or "checker"
            old["note"] = (note + " [review]").strip() if note else "[review]"
        else:
            same.append((tid, old["cluster_idx"], old_pid))

        # 旧备注与新编号冲突（备注是上一轮写的 todo）
        if note.startswith("change to") or note.startswith("change "):
            target = note.replace("change to", "").replace("change", "").strip()
            if target.replace(" ", "") != new_pid.replace(" ", ""):
                conflicts.append((tid, old["cluster_idx"], new_pid, note))

    print(f"改动 {len(changed)} 条，保持不变 {len(same)} 条")
    print(f"\n{'簇':>4}  {'原标注':<9}→ {'新标注':<9} track_id")
    for tid, cl, o, n, note in changed:
        print(f"{cl:>4}  {o:<9}→ {n:<9} {tid}   {('[' + note + ']') if note else ''}")
    if same:
        print("\n保持不变:")
        for tid, cl, p in same:
            print(f"{cl:>4}  {p:<9} 保持      {tid}")

    if conflicts:
        print("\n⚠ 旧备注与最终编号不一致（备注是上一轮遗留的 todo，需人工确认）:")
        for tid, cl, p, note in conflicts:
            print(f"  簇{cl:>4} {tid:<22} 最终={p:<8} 旧备注=「{note}」")

    if a.dry:
        print("\n[dry] 未写入")
        return

    bk = os.path.join(GT, "_backup")
    os.makedirs(bk, exist_ok=True)
    dst = os.path.join(bk, "gt_gallery_members_FILLED.before_review.csv")
    if not os.path.exists(dst):
        shutil.copy2(MAIN, dst)
        print(f"\n[backup] {dst}")

    fields = list(rows[0].keys())
    with open(MAIN, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {MAIN}  {len(rows)} 行")


if __name__ == "__main__":
    main()
