#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交叉验证：生产接入的 containment_nms(hybrid) 与实验用的 nms_select 是否逐帧等价。

为什么必须做：C 的结论是在 `retrack_from_raw.nms_select` 上验证的，真正上线的是
`containment_nms`。两者若有一处不等价（例如排序、冲突处理、阈值边界），
上线后的效果就不是验证过的那个。

用法: python tools/check_nms_equiv.py <raw.jsonl> [--hy-score-k 0.8] [--hy-min-h 200]
退出码 0 = 完全等价
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from containment_nms import containment_nms  # noqa: E402
from retrack_from_raw import nms_select  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--thr", type=float, default=0.30)
    ap.add_argument("--containment", type=float, default=0.85)
    ap.add_argument("--hy-score-k", type=float, default=0.8)
    ap.add_argument("--hy-min-h", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    n_fr = n_diff = n_skip = 0
    diffs = []
    with open(a.raw, encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            n_fr += 1
            if a.limit and n_fr > a.limit:
                break
            b = [bb for bb, s in zip(d["boxes"], d["scores"]) if s >= a.thr]
            s = [x for x in d["scores"] if x >= a.thr]
            if len(b) <= 1:
                n_skip += 1
                continue
            arr = np.asarray(b, dtype=np.float32)     # 与生产一致：模型输出是 float32
            sc = np.asarray(s, dtype=np.float32)
            k_old, _ = nms_select(arr, sc, a.containment, "hybrid", a.hy_score_k, a.hy_min_h)
            k_new = containment_nms(arr, sc, a.containment, "larger",
                                    hy_score_k=a.hy_score_k, hy_min_h=a.hy_min_h)
            if list(k_old) != [int(x) for x in k_new]:
                n_diff += 1
                if len(diffs) < 5:
                    diffs.append({"frame": d["frame"], "old": list(k_old),
                                  "new": [int(x) for x in k_new],
                                  "n": len(b)})

    print(f"帧数 {n_fr}  单框/空帧跳过 {n_skip}  不一致 {n_diff}")
    for x in diffs:
        print(f"  f{x['frame']} n={x['n']} old={x['old']} new={x['new']}")
    if n_diff:
        print("FAILED: 两个实现不等价，不能上线")
        return 1
    print("OK: 生产实现与实验实现逐帧等价")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
