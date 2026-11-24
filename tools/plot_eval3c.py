#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval3c 长流轨迹图：读 eval3c_long_stream.json → 四联图。

产物：output_crosscam_local/d3b_ab/eval3c_long_stream.png
用法：python tools/plot_eval3c.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "output_crosscam_local", "d3b_ab")
J = os.path.join(BASE, "eval3c_long_stream.json")
OUT = os.path.join(BASE, "eval3c_long_stream.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

COL = {"naive": "#d62728", "gated": "#2ca02c", "p20": "#1f77b4",
       "p15": "#9467bd", "p10": "#ff7f0e"}
NAME = {"naive": "朴素流式", "gated": "门控流式(p30)", "p20": "门控 p20",
        "p15": "门控 p15", "p10": "门控 p10"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=J)
    ap.add_argument("--out", default="")
    ap.add_argument("--subtitle", default="按协议 MIN_MEMBERS=3")
    args = ap.parse_args()
    out = args.out or args.json.replace(".json", ".png")
    d = json.load(open(args.json, encoding="utf-8"))
    r = d["rounds"]
    naive, gated, scans = r["naive"], r["gated"], r.get("scans", {})
    arms = d["arms"]
    frozen = arms["A1_frozen"]
    n_rounds = len(naive)
    xs = np.arange(1, n_rounds + 1)

    def series(traj, key, div=None):
        out = []
        for e in traj:
            v = e.get(key)
            if div:
                v = (v / e[div]) if (v is not None and e.get(div)) else np.nan
            out.append(np.nan if v is None else v)
        return np.asarray(out, float)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), dpi=140)

    # (a) ab_same
    ax = axes[0]
    ax.axhline(frozen["ab_same"], color="#888", ls="--", lw=1,
               label=f"冻结基线 {frozen['ab_same']:.4f}")
    ax.plot(xs, series(naive, "ab_same"), "o-", color=COL["naive"], label=NAME["naive"])
    ax.plot(xs, series(gated, "ab_same"), "s-", color=COL["gated"], label=NAME["gated"])
    for mk in ("p20", "p15", "p10"):
        if mk in scans:
            ax.plot(xs, series(scans[mk], "ab_same"), "--", alpha=0.7, color=COL[mk],
                    label=NAME[mk])
    ax.set_title("(a) 留出段 ab_same（a/b 分片一致率）", fontsize=10)
    ax.set_xlabel("更新轮次")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7)

    # (b) retention
    ax = axes[1]
    ax.axhline(frozen["ret_same"] / max(frozen["ret_n"], 1), color="#888", ls="--", lw=1,
               label=f"冻结 {frozen['ret_same']}/{frozen['ret_n']}")
    ax.plot(xs, series(naive, "ret_same", "ret_n"), "o-", color=COL["naive"], label=NAME["naive"])
    ax.plot(xs, series(gated, "ret_same", "ret_n"), "s-", color=COL["gated"], label=NAME["gated"])
    for mk in ("p20", "p15", "p10"):
        if mk in scans:
            ax.plot(xs, series(scans[mk], "ret_same", "ret_n"), "--", alpha=0.7,
                    color=COL[mk], label=NAME[mk])
    ax.set_title(f"(b) 跨段保持率（留出段↔参考段，n={frozen['ret_n']}）", fontsize=10)
    ax.set_xlabel("更新轮次")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7)

    # (c) viol
    ax = axes[2]
    ax.axhline(frozen["viol"], color="#888", ls="--", lw=1, label=f"冻结 {frozen['viol']}")
    ax.plot(xs, series(naive, "viol"), "o-", color=COL["naive"], label=NAME["naive"])
    ax.plot(xs, series(gated, "viol"), "s-", color=COL["gated"], label=NAME["gated"])
    for mk in ("p20", "p15", "p10"):
        if mk in scans:
            ax.plot(xs, series(scans[mk], "viol"), "--", alpha=0.7, color=COL[mk], label=NAME[mk])
    ax.set_title("(c) 留出段跨机位时间冲突 viol（越低越好）", fontsize=10)
    ax.set_xlabel("更新轮次")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7)

    # (d) drift / hijack / 原型总量
    ax = axes[3]
    ax.plot(xs, series(naive, "drift_med"), "o-", color=COL["naive"], label="朴素 漂移中位")
    ax.plot(xs, series(gated, "drift_med"), "s-", color=COL["gated"], label="门控 漂移中位")
    ax.plot(xs, series(naive, "hijack_max"), "o--", alpha=0.6, color=COL["naive"],
            label="朴素 跨簇最大 cos")
    ax.plot(xs, series(gated, "hijack_max"), "s--", alpha=0.6, color=COL["gated"],
            label="门控 跨簇最大 cos")
    ax.set_ylim(0, 1.02)
    ax.set_title("(d) 原型漂移 / 跨簇相似度（越高越危险）", fontsize=10)
    ax.set_xlabel("更新轮次")
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(xs, series(naive, "proto_total"), ":", color=COL["naive"], alpha=0.5)
    ax2.plot(xs, series(gated, "proto_total"), ":", color=COL["gated"], alpha=0.5)
    ax2.set_ylabel("原型累计数（点线）", fontsize=8)

    fig.suptitle(
        f"EVAL-3c 长数据流门控自训练：long35 35min × {n_rounds} 轮"
        f"（同 session bootstrap，{args.subtitle}）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
