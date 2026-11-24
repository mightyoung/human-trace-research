"""回收「未标查询」的补标结果（review_323_unk.html / 簇 61 等导出的 CSV），并给出影响面。

安全措施：
  - 只允许改**输入 CSV 里列出、且在 GT 里仍为空**的 query_id，其它行原样不动
  - 改动前把原文件备份到 gt/archive/gt_query_gt_FILLED.<时间戳>.csv
  - 打印逐条 diff，`--dry-run` 时只报不改
  - 事后重算：可答 / 图库外 / 非人 / 无真值

取值约定：
  身份名（male*/female*） → person_id = 该身份，unk_reason 清空
  UNK                     → 保持无真值（不改）
  no-human                → person_id = no-human（非人，计数时排除）
  __NEW__                 → 保持无真值，unk_reason = gallery_out（图库外新人）

用法：
  PY tools/ingest_323_unk.py --csv C:/.../rev323_unk_FILLED.csv --dry-run
  PY tools/ingest_323_unk.py --csv C:/.../rev323_unk_FILLED.csv            # 真写（自动备份）
  PY tools/ingest_323_unk.py --csv .../318_c61.csv --seg 318_test_534
"""
import argparse
import collections
import csv
import json
import os
import shutil
import time

import eval_query_fusion as F

SEG = "323_test_527"   # 仅作默认提示，不再限制改动范围


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--deliv", default="")
    ap.add_argument("--seg", default="", help="只接受该段的 query_id（留空=不限）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    gt = os.path.join(deliv, "gt")
    path = os.path.join(gt, "gt_query_gt_FILLED.csv")

    rows = F.rd(path)
    labeled = {r["query_id"]: (r["person_id"] or "").strip() for r in rows}
    got = F.rd(a.csv)
    # 只改「输入 CSV 里有、GT 里仍为空、且段号匹配」的查询
    target = set()
    for r in got:
        q = r["query_id"]
        if q not in labeled:
            continue
        if a.seg and not q.startswith(a.seg + "#"):
            continue
        if not labeled[q]:
            target.add(q)
    print("补标文件 %d 行；可改目标 %d 条" % (len(got), len(target)))

    valid_ids = {r["person_id"].strip() for r in F.rd(os.path.join(gt, "gt_gallery_members_FILLED.csv"))
                 if (r["person_id"] or "").strip()}
    valid_ids = {x for x in valid_ids if not (x.lower() == "no-human" or x.lower().startswith("shadow"))}
    bad = [r for r in got if r["query_id"] not in target]
    if bad:
        print("!! 有 %d 行不在可改范围（已标过/段号不符/不存在，会被忽略）：%s"
              % (len(bad), [r["query_id"] for r in bad][:5]))
    unknown = [r for r in got if (r["person_id"] or "").strip() not in
               (valid_ids | {"", "UNK", "no-human", "__NEW__"})]
    if unknown:
        raise SystemExit("!! 非法取值：%s" % [(r["query_id"], r["person_id"]) for r in unknown][:5])

    change = {}
    for r in got:
        q, v = r["query_id"], (r["person_id"] or "").strip()
        if q not in target or v in ("", "UNK"):
            continue
        if v == "__NEW__":
            change[q] = ("", "gallery_out")
        elif v == "no-human":
            change[q] = ("no-human", "")
        else:
            change[q] = (v, "")
    print("将改动 %d 条：%s" % (len(change),
                            dict(collections.Counter(v[0] or v[1] for v in change.values()))))
    for q, (pid, reason) in sorted(change.items()):
        print("  %-22s → %s%s" % (q, pid or "(空)", (" reason=" + reason) if reason else ""))
    if a.dry_run:
        print("（dry-run，不写盘）")
        return
    if not change:
        print("没有可写入的改动")
        return

    arch = os.path.join(gt, "archive")
    os.makedirs(arch, exist_ok=True)
    bak = os.path.join(arch, "gt_query_gt_FILLED.%s.csv" % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(path, bak)
    print("备份 → %s" % bak)

    n = 0
    for r in rows:
        if r["query_id"] in change:
            pid, reason = change[r["query_id"]]
            r["person_id"] = pid
            r["unk_reason"] = reason
            if pid:
                r["evaluable"] = r["evaluable"]  # 保持原判定（14 条不可评估仍不可评估）
            n += 1
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("已写入 %d 条" % n)

    # 影响面
    gtruth = F.load_gallery_truth(gt)
    qtruth = F.load_query_truth(gt)
    idv = F.identity_values(gtruth)
    ev = [r for r in rows if r["evaluable"] == "1"]
    np_ = [r for r in ev if F.is_nonperson(qtruth.get(r["query_id"], ""))]
    ans = [r for r in ev if qtruth.get(r["query_id"], "") in idv]
    out = [r for r in ev if qtruth.get(r["query_id"], "") and
           qtruth.get(r["query_id"], "") not in idv and
           not F.is_nonperson(qtruth.get(r["query_id"], ""))]
    noT = [r for r in ev if not qtruth.get(r["query_id"], "")]
    print("\n可评估 %d：可答 %d｜图库外 %d｜非人 %d｜无真值 %d（原为 196/11/0/22）"
          % (len(ev), len(ans), len(out), len(np_), len(noT)))
    json.dump(dict(changed=n, backup=bak, evaluable=len(ev), answerable=len(ans),
                   gallery_out=len(out), nonperson=len(np_), no_truth=len(noT)),
              open(os.path.join(deliv, "results", "unk323_ingest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
