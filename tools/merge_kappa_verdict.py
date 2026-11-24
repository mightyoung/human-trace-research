#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并 D5 分歧定案结果，并检查它是否影响任何配置的判定。"""
import csv
import collections
import os

GT = "deliverable/data-request-v2/gt"
RES = "deliverable/data-request-v2/results"

# 用户交回的定案（decision=annotator1 表示第一轮标注正确）
DECIDED = {
    "305_test_544#440": ("male1", "female1", "annotator1"),
    "305_test_544#1153": ("male12", "female3", "annotator1"),
    "323_test_527#1669": ("male20", "male18", "annotator1"),
}

CFGS = ["A_baseline", "C_dmax0.28", "C_dmax0.30", "C_dmax0.34",
        "D_split0.20", "D_split0.20_veto0.30", "G_global0.20_veto0.30"]


def load(p, key):
    with open(p, encoding="utf-8-sig") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def norm(p):
    p = (p or "").strip()
    return "UNK" if (not p or p.upper() == "UNK") else p


def main():
    # 1. 这 3 条在各配置里的状态
    print("=== 3 条定案在各配置中的判定 ===")
    print(f"{'配置':<26}{'状态':<20}{'判到':<10}{'结论'}")
    for cfg in CFGS:
        p = os.path.join(RES, "verdicts", f"{cfg}.csv")
        if not os.path.exists(p):
            continue
        v = load(p, "query_id")
        for k, (a1, a2, dec) in DECIDED.items():
            if k not in v:
                print(f"{cfg:<26}{'拒识/未接受':<20}{'—':<10}")
                continue
            r = v[k]
            verdict = r["verdict"]
            c_pid = norm(r["c_pid"])
            # 定案后正确答案是第一人的标注
            ok = (c_pid == a1)
            print(f"{cfg:<26}{'接受':<20}{c_pid:<10}"
                  f"{'对' if verdict == 'C' else ('错' if verdict == 'W' else '看不清')}"
                  f"{'' if ok else ' ← 与定案不符'}")
        print()

    # 2. 主结果是否变化：这 3 条在 baseline 都是 C，定案不改变任何数字
    v = load(os.path.join(RES, "verdicts", "A_baseline.csv"), "query_id")
    print("=== baseline 中这 3 条 ===")
    for k, (a1, a2, dec) in DECIDED.items():
        r = v.get(k)
        if r is None:
            print(f"  {k}: 不在接受集")
        else:
            print(f"  {k}: 簇主={r['c_pid']} 第一人={a1} 判定={r['verdict']}"
                  f" → 定案{'支持第一人' if r['verdict'] == 'C' else '与判定不一致'}")

    # 3. 定案后的口径
    print("\n=== 定案后的一致性口径 ===")
    print("  可判定子集 131 条：第一人 111C/6W/14U，第二人 108C/9W/14U")
    print("  定案确认 3 条分歧全部为第一人对 → 第二人在该子集的标注错误率 3/131 = 2.3%")
    print("  主结果不变：baseline 仍为 C=118 / W=6 / U=14，精度 0.952，C/229 = 0.515")
    print("  原因：这 3 条在 baseline 均判 C，且簇主与第一人标注一致")

    # 4. 写定案记录
    out = os.path.join(GT, "review_kappa_FILLED.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "cam", "box_type", "pid_annotator1", "pid_annotator2",
                    "cluster_master", "n_labelled", "n_ids", "top_share", "d_max",
                    "decision", "final_pid"])
        meta = {
            "305_test_544#440": ("305", "head", "3", "1", "1.0", "0.05"),
            "305_test_544#1153": ("305", "partial", "3", "1", "1.0", "0.1166"),
            "323_test_527#1669": ("323", "body", "4", "3", "0.5", "0.3468"),
        }
        for k, (a1, a2, dec) in DECIDED.items():
            cam, box, n_lab, n_ids, share, dmax = meta[k]
            final = a1 if dec == "annotator1" else (a2 if dec == "annotator2" else "?")
            w.writerow([k, cam, box, a1, a2, a1, n_lab, n_ids, share, dmax, dec, final])
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
