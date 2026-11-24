#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨运行的 A/B 报告：同一套轨迹与真值、只换表征（例如 BGR → RGB），比 A_baseline。

与 compare_cfgs.py 的区别：compare_cfgs 在**同一个 run 内部**拿各配置比 A_baseline；
本脚本比的是**两个 run 的同一个配置**（逐查询配对），回答"换表征到底多答对几条"。

用法：
  python tools/ab_emb_report.py \
    --run-a deliverable/data-request-v2/results \
    --run-b tmp/rgb_pkg/results \
    --gt deliverable/data-request-v2/gt/gt_query_gt_FILLED.csv \
    --cfg A_baseline --label-a BGR --label-b RGB \
    --out tmp/rgb_pkg/ab_report.md
"""
import argparse
import collections
import csv
import json
import math
import os
import random


def load_verdicts(run, cfg):
    p = os.path.join(run, "verdicts", f"{cfg}.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    return {r["query_id"]: r for r in csv.DictReader(open(p, encoding="utf-8-sig"))}


def load_summary(run, cfg):
    p = os.path.join(run, "summary.csv")
    if not os.path.exists(p):
        return {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if r["config"] == cfg:
            return r
    return {}


def mcnemar_exact(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def boot_ci(deltas, iters=2000, seed=0):
    rnd = random.Random(seed)
    n = len(deltas)
    out = []
    for _ in range(iters):
        s = sum(deltas[rnd.randrange(n)] for _ in range(n))
        out.append(s)
    out.sort()
    return out[int(0.025 * iters)], out[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--gt", default="deliverable/data-request-v2/gt/gt_query_gt_FILLED.csv")
    ap.add_argument("--cfg", default="A_baseline")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--c-denom", type=int, default=229)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    qids = [r["query_id"] for r in csv.DictReader(open(a.gt, encoding="utf-8-sig"))
            if r.get("evaluable") == "1"]
    va, vb = load_verdicts(a.run_a, a.cfg), load_verdicts(a.run_b, a.cfg)
    sa, sb = load_summary(a.run_a, a.cfg), load_summary(a.run_b, a.cfg)

    def counts(v):
        C = W = U = 0
        for q in qids:
            r = v.get(q)
            if r is None:
                continue
            if r["verdict"] == "C":
                C += 1
            elif r["verdict"] == "W":
                W += 1
            else:
                U += 1
        return C, W, U

    Ca, Wa, Ua = counts(va)
    Cb, Wb, Ub = counts(vb)

    da = {q: (1 if va.get(q, {}).get("verdict") == "C" else 0) for q in qids}
    db = {q: (1 if vb.get(q, {}).get("verdict") == "C" else 0) for q in qids}
    n01 = sum(1 for q in qids if da[q] == 0 and db[q] == 1)
    n10 = sum(1 for q in qids if da[q] == 1 and db[q] == 0)
    p = mcnemar_exact(n01, n10)
    deltas = [db[q] - da[q] for q in qids]
    lo, hi = boot_ci(deltas)

    def prec(C, W):
        return C / (C + W) if C + W else 0.0

    def cam_of(q):
        return q.split("_")[0]

    cams = sorted({cam_of(q) for q in qids})
    cam_rows = []
    for c in cams:
        qs = [q for q in qids if cam_of(q) == c]
        ca = sum(da[q] for q in qs)
        cb = sum(db[q] for q in qs)
        cam_rows.append((c, len(qs), ca, cb, cb - ca))

    lines = [
        f"# 表征 A/B：{a.label_a} vs {a.label_b}（{a.cfg}）",
        "",
        f"分母 {a.c_denom}；查询 {len(qids)} 条可评估；两 run 轨迹与真值完全相同，只换表征。",
        "",
        "| 口径 | C | W | U | 精度 C/(C+W) | C/denom | cov | θ | K |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {a.label_a} | {Ca} | {Wa} | {Ua} | {prec(Ca, Wa):.4f} | {Ca/a.c_denom:.4f} | "
        f"{sa.get('both_accept_rate','')} | {sa.get('theta','')} | {sa.get('K','')} |",
        f"| {a.label_b} | {Cb} | {Wb} | {Ub} | {prec(Cb, Wb):.4f} | {Cb/a.c_denom:.4f} | "
        f"{sb.get('both_accept_rate','')} | {sb.get('theta','')} | {sb.get('K','')} |",
        "",
        f"**配对变化**：{a.label_b} 救回 {n01} 条 / 弄丢 {n10} 条 / 净 {(Cb-Ca):+d}；"
        f"McNemar 精确 p = {p:.4g}；ΔC bootstrap 95% CI [{lo:+.0f}, {hi:+.0f}]",
        "",
        "| 机位 | 可评估 | C(A) | C(B) | Δ |",
        "|---|---|---|---|---|",
    ]
    for c, n, ca, cb, d in cam_rows:
        lines.append(f"| {c} | {n} | {ca} | {cb} | {d:+d} |")
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(txt)
        json.dump(dict(cfg=a.cfg, label_a=a.label_a, label_b=a.label_b,
                       C_a=Ca, W_a=Wa, U_a=Ua, C_b=Cb, W_b=Wb, U_b=Ub,
                       rescue=n01, lose=n10, delta=Cb - Ca, mcnemar_p=p,
                       ci=[lo, hi], per_cam=cam_rows),
                  open(a.out.replace(".md", ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
