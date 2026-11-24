"""对 D7 各配置做配对比较与置信区间。

- 主指标 C/229（每题判对的数量，U 记错），另报 C/(C+W) 精度
- 配对 McNemar 精确检验（baseline vs 各配置，逐查询对错）
- bootstrap 95% CI（按查询重采样，2000 次，固定种子）
用法：PY tools/compare_cfgs.py [--run results]
"""
import argparse
import collections
import csv
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
N = 229


def load_verdicts(run, cfg):
    p = os.path.join(run, "verdicts", f"{cfg}.csv")
    return {r["query_id"]: r for r in csv.DictReader(open(p, encoding="utf-8-sig"))}


GTDIR = ""


def _gtdir():
    return GTDIR or os.path.join(DELIV, "gt")


def evaluable_qids():
    p = os.path.join(_gtdir(), "gt_query_gt_FILLED.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    if rows and "evaluable" in rows[0]:
        return [r["query_id"] for r in rows if r["evaluable"] == "1"]
    # 外部验证包的真值只有 query_id/person_id（自动真值）：有身份即进入评价
    return [r["query_id"] for r in rows if (r.get("person_id") or "").strip()]


def correct_map(v, qids):
    """query_id -> 1/0（未被接受或 U 记 0，与 C/229 口径一致）"""
    return {q: (1 if v.get(q, {}).get("verdict") == "C" else 0) for q in qids}


def mcnemar_exact(b, c):
    """b, c 是两个方法逐样本对错向量；返回 (b01, b10, 双侧精确 p)"""
    n01 = sum(1 for x, y in zip(b, c) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(b, c) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    # 双侧精确二项检验 p=0.5
    from math import comb
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, p)


def boot_ci(vals, n_boot=2000, seed=20261019):
    rnd = random.Random(seed)
    n = len(vals)
    out = []
    for _ in range(n_boot):
        s = sum(vals[rnd.randrange(n)] for _ in range(n))
        out.append(s / n)
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot) - 1]


def main():
    global DELIV, N
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results")
    ap.add_argument("--deliv", default="", help="数据包根目录（默认 deliverable/data-request-v2）")
    ap.add_argument("--gt-dir", default="", help="真值目录（默认 <deliv>/gt）")
    ap.add_argument("--c-denom", type=int, default=229, help="C 的分母；默认 229")
    a = ap.parse_args()
    if a.deliv:
        DELIV = os.path.abspath(a.deliv)
        if a.gt_dir:
            globals()["GTDIR"] = os.path.abspath(a.gt_dir)
    N = a.c_denom
    run = os.path.join(DELIV, a.run)

    pcf = {r["config"]: r for r in csv.DictReader(open(os.path.join(run, "precision.csv"),
                                                      encoding="utf-8-sig"))}
    base = "A_baseline"
    bv = load_verdicts(run, base)
    qids = evaluable_qids()
    bmap = correct_map(bv, qids)

    key = [c for c in pcf if c.startswith(("A_", "C_dmax0.2", "C_dmax0.28", "C_dmax0.30",
                                            "C_dmax0.34", "D_split0.15", "D_split0.17",
                                            "D_split0.18", "D_split0.20", "D_split0.22",
                                            "D_split0.25", "G_"))]
    key = sorted(set(key))

    print(f"基线 {base}: C={sum(bmap[q] for q in qids)}/{N}")
    print(f"\n{'配置':<28}{'θ':>6}{'C/229':>7}{'95%CI':>16}{'精度':>7}{'Δ':>6}{'McNemar':>12}{'p':>8}")
    rows = []
    for c in key:
        if c == base:
            continue
        cv = load_verdicts(run, c)
        cmap = correct_map(cv, qids)
        cs = sum(cmap[q] for q in qids)
        lo, hi = boot_ci([cmap[q] for q in qids])
        n01, n10, p = mcnemar_exact([bmap[q] for q in qids], [cmap[q] for q in qids])
        d = cs - sum(bmap[q] for q in qids)
        r = pcf[c]
        prec = float(r["precision"]) if r["precision"] else float("nan")
        print(f"{c:<28}{float(r['theta']):>6.2f}{cs:>7}{f'[{lo:.3f},{hi:.3f}]':>16}"
              f"{prec:>7.3f}{d:>+6}{f'+{n01}/-{n10}':>12}{p:>8.3f}")
        rows.append(dict(config=c, theta=r["theta"], C_over_229=cs,
                         ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                         precision=r["precision"], delta=d,
                         mcnemar_gain=n01, mcnemar_loss=n10, p=round(p, 4)))

    op = os.path.join(run, "compare.csv")
    with open(op, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {op}")

    # 逐机位
    print("\n=== 逐机位（baseline 严格口径）===")
    per = collections.defaultdict(lambda: collections.Counter())
    for q, r in bv.items():
        per[r["cam"]][r["verdict"]] += 1
    print(f"{'机位':<6}{'接受':>5}{'C':>4}{'W':>4}{'U':>4}{'精度':>7}")
    for cam in sorted(per):
        c = per[cam]
        n = c["C"] + c["W"]
        print(f"{cam:<6}{sum(c.values()):>5}{c['C']:>4}{c['W']:>4}{c['U']:>4}"
              f"{(c['C']/n if n else float('nan')):>7.3f}")


if __name__ == "__main__":
    main()
