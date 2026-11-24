"""把「非人簇门控」落成一个可复用的运行目录（基线 v2 = A_baseline + 门控）。

规则：簇的全部已标成员都是 no-human / shadow* → 该簇不可作答。
      命中它的查询 final_status 改为 rejected_nonperson，并清空 final_cluster_local。
      混人簇（人 + 非人）不动。

产物：<deliv>/results_gated/
  summary.csv / members/（原样复制）
  queries/<cfg>.csv（门控后）
  gate_report.json（前后计数、被门控清单、硬断言结果）

用法：PY tools/apply_nonperson_gate.py [--src results] [--dst results_gated] [--cfg A_baseline]
"""
import argparse
import collections
import csv
import json
import os
import shutil

import eval_query_fusion as F


def is_nonperson(pid):
    p = (pid or "").strip().lower()
    return p == "no-human" or p.startswith("shadow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deliv", default="")
    ap.add_argument("--src", default="results")
    ap.add_argument("--dst", default="results_gated")
    ap.add_argument("--cfg", default="A_baseline")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    gt = os.path.join(deliv, "gt")
    src = os.path.join(deliv, a.src)
    dst = os.path.join(deliv, a.dst)

    gtruth = F.load_gallery_truth(gt)
    qtruth = F.load_query_truth(gt)
    ident = F.build_ident(src, a.cfg, gtruth)

    raw = {}
    for r in F.rd(os.path.join(gt, "gt_gallery_members_FILLED.csv")):
        pid = (r["person_id"] or "").strip()
        if pid:
            raw[r["track_id"]] = pid
    def np_of(cfg):
        """该配置下的「全非人簇」与「混人簇」——簇号是配置相关的，必须逐配置算。"""
        mem = collections.defaultdict(list)
        for r in F.rd(os.path.join(src, "members", cfg + ".csv")):
            mem[int(r["cluster_local"])].append(r["track_id"])
        np_, mix_ = {}, {}
        for cl, ts in mem.items():
            labs = [raw[t] for t in ts if raw.get(t)]
            if not labs:
                continue
            n_np = sum(1 for x in labs if is_nonperson(x))
            if n_np == len(labs):
                np_[cl] = dict(n=len(ts), labels=dict(collections.Counter(labs)))
            elif n_np:
                mix_[cl] = dict(n=len(ts), n_nonperson=n_np)
        return np_, mix_

    # ---- 复制骨架 ----
    os.makedirs(dst, exist_ok=True)
    for name in ("summary.csv",):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, name))
    for sub in ("members", "verdicts"):
        s = os.path.join(src, sub)
        if os.path.isdir(s):
            d = os.path.join(dst, sub)
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)

    # ---- 逐配置门控 ----
    qdir = os.path.join(src, "queries")
    os.makedirs(os.path.join(dst, "queries"), exist_ok=True)
    report = {}
    for fn in sorted(os.listdir(qdir)):
        if not fn.endswith(".csv"):
            continue
        cfg = fn[:-4]
        nonperson, mixed = np_of(cfg)
        ident = F.build_ident(src, cfg, gtruth)
        rows = F.rd(os.path.join(qdir, fn))
        before = collections.Counter()
        after = collections.Counter()
        gated = []
        out = []
        for r in rows:
            r = dict(r)
            if r["final_status"] == "accepted":
                cl = int(r["final_cluster_local"])
                qp = qtruth.get(r["query_id"], "")
                cp = ident.get(cl, "")
                counted = not F.is_nonperson(qp)   # 非人真值不参与 C/W/U
                v = "U" if (not qp or not cp) else ("C" if qp == cp else "W")
                if counted:
                    before[v] += 1
                if cl in nonperson:
                    gated.append(r["query_id"])
                    r["final_status"] = "rejected_nonperson"
                    r["final_cluster_local"] = ""
                elif counted:
                    after[v] += 1
            out.append(r)
        if before["C"] != after["C"] or before["W"] != after["W"]:
            raise SystemExit("门控改变了 C/W：%s %s → %s" % (cfg, dict(before), dict(after)))
        with open(os.path.join(dst, "queries", fn), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(out)
        report[cfg] = dict(before=dict(before), after=dict(after),
                           accepted_before=sum(before.values()),
                           accepted_after=sum(after.values()),
                           gated=gated)

    gate = dict(src=a.src, dst=a.dst, configs=report)
    json.dump(gate, open(os.path.join(dst, "gate_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    r = report[a.cfg]
    np_, mix_ = np_of(a.cfg)
    print("配置 %s：全非人簇 %d 个 %s｜混人簇（不门控）%d 个"
          % (a.cfg, len(np_), {k: v["n"] for k, v in np_.items()}, len(mix_)))
    print("配置 %s：接受 %d → %d｜C %d→%d｜W %d→%d｜U %d→%d"
          % (a.cfg, r["accepted_before"], r["accepted_after"],
             r["before"].get("C", 0), r["after"].get("C", 0),
             r["before"].get("W", 0), r["after"].get("W", 0),
             r["before"].get("U", 0), r["after"].get("U", 0)))
    print("硬断言：C/W 不变 ✓")
    print("→ %s" % dst)


if __name__ == "__main__":
    main()
