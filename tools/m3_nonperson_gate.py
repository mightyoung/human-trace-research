"""M3-A 非人簇门控（拒识）：把命中「全非人簇」的查询判为拒识。

背景：基线有 14 条「已接受但无真值」的查询（U），全部落在 4 个全非人簇
      （318 的 61 = 挂帽椅子；323 的 16/32/34 = 影子）。它们不进 C/W，却拉低了
      「已接受集合」的可核验比例（U 占 10.1%）。

做法：图库标注里已有 no-human / shadow* 标签 → 若一个簇的**全部已标成员**都是非人，
      则该簇不可作为答案；命中它的查询改判拒识。
      只门控「全非人」簇，混人簇（人 + 非人）不动，避免误伤。

硬约束（脚本内断言）：C 与 W 数量在门控前后必须完全不变。

用法：PY tools/m3_nonperson_gate.py [--run results] [--cfg A_baseline]
输出：results/m3_gate.json、results/m3_gate.csv
"""
import argparse
import collections
import csv
import json
import os

import eval_query_fusion as F


def is_nonperson(pid):
    p = (pid or "").strip().lower()
    return p == "no-human" or p.startswith("shadow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results")
    ap.add_argument("--deliv", default="")
    ap.add_argument("--cfg", default="A_baseline")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)

    gtruth = F.load_gallery_truth(gt)          # 只含人物（非人已被过滤）
    qtruth = F.load_query_truth(gt)
    ident = F.build_ident(run, a.cfg, gtruth)

    # 原始标签（含非人）
    raw = {}
    for r in F.rd(os.path.join(gt, "gt_gallery_members_FILLED.csv")):
        pid = (r["person_id"] or "").strip()
        if pid:
            raw[r["track_id"]] = pid

    mem = collections.defaultdict(list)
    for r in F.rd(os.path.join(run, "members", a.cfg + ".csv")):
        mem[int(r["cluster_local"])].append(r["track_id"])

    nonperson, mixed = {}, {}
    for cl, ts in mem.items():
        labs = [raw[t] for t in ts if raw.get(t)]
        if not labs:
            continue
        n_np = sum(1 for x in labs if is_nonperson(x))
        if n_np == len(labs):
            nonperson[cl] = dict(n=len(ts), labels=dict(collections.Counter(labs)))
        elif n_np:
            mixed[cl] = dict(n=len(ts), n_nonperson=n_np, labels=dict(collections.Counter(labs)))

    rows = F.rd(os.path.join(run, "queries", a.cfg + ".csv"))
    before = collections.Counter()
    after = collections.Counter()
    gated = []
    for r in rows:
        if r["final_status"] != "accepted":
            continue
        cl = int(r["final_cluster_local"])
        qp = qtruth.get(r["query_id"], "")
        cp = ident.get(cl, "")
        v = "U" if (not qp or not cp) else ("C" if qp == cp else "W")
        before[v] += 1
        if cl in nonperson:
            gated.append(dict(query_id=r["query_id"], cam=r["cam"], cluster=cl,
                              verdict=v, q_pid=qp, c_pid=cp))
            continue
        after[v] += 1

    assert after["C"] == before["C"], "门控不得改变 C（%d → %d）" % (before["C"], after["C"])
    assert after["W"] == before["W"], "门控不得改变 W（%d → %d）" % (before["W"], after["W"])

    def pack(c):
        acc = sum(c.values())
        return dict(accepted=acc, C=c["C"], W=c["W"], U=c["U"],
                    precision=round(c["C"] / (c["C"] + c["W"]), 4) if c["C"] + c["W"] else 0.0,
                    c_over_229=round(c["C"] / 229, 4),
                    u_share=round(c["U"] / acc, 4) if acc else 0.0)

    print("全非人簇 %d 个：%s" % (len(nonperson), {k: v["n"] for k, v in nonperson.items()}))
    print("混人簇（人+非人，**不门控**）%d 个：%s" % (len(mixed), {k: v["n"] for k, v in mixed.items()}))
    print("\n%-22s %-8s %-6s %-6s %-6s %-8s %-8s %s"
          % ("", "接受", "C", "W", "U", "精度", "C/229", "U 占比"))
    for name, c in (("门控前", before), ("门控后", after)):
        p = pack(c)
        print("%-22s %-8d %-6d %-6d %-6d %-8.4f %-8.4f %.1f%%"
              % (name, p["accepted"], p["C"], p["W"], p["U"], p["precision"],
                 p["c_over_229"], 100 * p["u_share"]))
    print("\n被门控的 %d 条：" % len(gated))
    for g in gated:
        print("  %-22s cam %-4s 簇 %-4d %s" % (g["query_id"], g["cam"], g["cluster"], g["verdict"]))

    out = dict(cfg=a.cfg, run=run,
               nonperson_clusters={str(k): v for k, v in nonperson.items()},
               mixed_clusters={str(k): v for k, v in mixed.items()},
               before=pack(before), after=pack(after), gated=gated)
    json.dump(out, open(os.path.join(run, "m3_gate.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(run, "m3_gate.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "cam", "cluster", "verdict",
                                          "q_pid", "c_pid"])
        w.writeheader()
        w.writerows(gated)
    print("\n→ results/m3_gate.{json,csv}")


if __name__ == "__main__":
    main()
