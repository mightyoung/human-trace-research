#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对级判读的 Cohen's κ：「时空矛盾 27 项」第二标注人结果 vs 第一标注人。

用法
----
    python tools/eval_kappa_spacetime.py deliverable/spacetime_review/kappa2/kappa2_verdicts.json

口径
----
- 标签空间 {same, diff}；`unk`（看不清）**单独统计**，不并入一致率，也不当作某一类。
  报告两种口径：a) 剔除 unk 后算 κ；b) 把 unk 当作一个独立类别算 κ。
- CI 用 bootstrap（1000 次，按项重采样），因为 n 只有 27，正态近似不可靠。
- 同时报 Gwet's AC1：κ 在「一类占绝对多数」时会因 pe 偏高而被压低，AC1 用来交叉验证。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A1 = ROOT / "deliverable/spacetime_review/labels_20261112.json"


def kappa(a, b, cats=("same", "diff")):
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, pe, k


def ac1(a, b, cats=("same", "diff")):
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p = np.zeros(len(cats))
    for x, y in zip(a, b):
        p[idx[x]] += 1
        p[idx[y]] += 1
    p /= 2 * n
    pe = float((p * (1 - p)).sum()) / (len(cats) - 1)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot(a, b, f, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        v = f([a[i] for i in s], [b[i] for i in s])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main():
    p = Path(sys.argv[1] if len(sys.argv) > 1 else
             ROOT / "deliverable/spacetime_review/kappa2/kappa2_verdicts.json")
    if not p.exists():
        print("找不到第二标注人结果：%s" % p)
        print("先让第二标注人打开 kappa2_blind.html 判完 27 项并导出 JSON。")
        return
    a2 = json.load(open(p, encoding="utf-8"))
    ans2 = a2["ans"] if "ans" in a2 else a2
    a1 = json.load(open(A1, encoding="utf-8"))["labels"]

    ids = [k for k in a1 if k in ans2 and ans2[k]]
    print("第一标注人 %d 项，第二标注人 %d 项，可配对 %d 项"
          % (len(a1), len([1 for v in ans2.values() if v]), len(ids)))
    unk = [k for k in ids if ans2[k] == "unk"]
    print("第二标注人标为「看不清」%d 项：%s" % (len(unk), ",".join(unk) if unk else "无"))

    core = [k for k in ids if ans2[k] in ("same", "diff")]
    y1 = [a1[k]["v"] for k in core]
    y2 = [ans2[k] for k in core]

    print("\n=== 剔除 unk 口径（n=%d）===" % len(core))
    po, pe, k = kappa(y1, y2)
    ci = boot(y1, y2, lambda x, y: kappa(x, y)[2])
    print("  po=%.3f  pe=%.3f  κ=%.3f  95%%CI [%.3f, %.3f]  AC1=%.3f"
          % (po, pe, k, ci[0], ci[1], ac1(y1, y2)))
    from collections import Counter
    print("  交叉表 A1×A2:", dict(Counter(zip(y1, y2))))
    dis = [(c, x, y) for c, x, y in zip(core, y1, y2) if x != y]
    print("  分歧 %d 项:" % len(dis))
    for c, x, y in dis:
        print("    %s  A1=%s  A2=%s  pair=%s" % (c, x, y, a1[c].get("pair")))

    if unk:
        y1u = [a1[k]["v"] for k in ids]
        y2u = [ans2[k] for k in ids]
        po, pe, kk = kappa(y1u, y2u, cats=("same", "diff", "unk"))
        ci2 = boot(y1u, y2u, lambda x, y: kappa(x, y, ("same", "diff", "unk"))[2])
        print("\n=== unk 当作独立类别（n=%d）===" % len(ids))
        print("  po=%.3f  κ=%.3f  95%%CI [%.3f, %.3f]" % (po, kk, ci2[0], ci2[1]))


if __name__ == "__main__":
    main()
