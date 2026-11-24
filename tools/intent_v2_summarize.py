#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图 v2 汇总：4 配置 × 3 seeds 的 test 指标对比（acc / mF1 / per-cam / 分层）。"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "output_crosscam", "analysis", "intent", "v2")
CFGS = ["base", "social", "behavior", "both"]
SEEDS = [42, 43, 44]


def main():
    data = {}
    for cfg in CFGS:
        data[cfg] = []
        for s in SEEDS:
            p = os.path.join(D, "out_v2", f"{cfg}_s{s}", "metrics_full.json")
            if not os.path.exists(p):
                print(f"[missing] {p}")
                continue
            r = json.load(open(p))["rep"]
            data[cfg].append(r)

    print(f"{'cfg':10s} | {'acc (mean±std)':16s} | {'macroF1 (mean±std)':18s} | best_ep")
    print("-" * 70)
    summ = {}
    for cfg in CFGS:
        accs = [r["overall"]["acc"] for r in data[cfg]]
        f1s = [r["overall"]["macro_f1"] for r in data[cfg]]
        eps = [r["config"]["best_ep"] for r in data[cfg]]
        summ[cfg] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                     "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                     "accs": accs, "f1s": f1s, "best_eps": eps}
        print(f"{cfg:10s} | {np.mean(accs):.4f} ± {np.std(accs):.4f}  | "
              f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}   | {eps}")

    print(f"\n=== per-camera acc（mean over seeds） ===")
    print(f"{'cfg':10s} | {'305':>8s} {'318':>8s} {'320':>8s} {'323':>8s}")
    for cfg in CFGS:
        row = []
        for c in ("305", "318", "320", "323"):
            vals = [r["per_cam"][c]["acc"] for r in data[cfg] if c in r["per_cam"]]
            row.append(np.mean(vals) if vals else float("nan"))
        print(f"{cfg:10s} | " + " ".join(f"{v:8.4f}" for v in row))
        summ[cfg]["per_cam_acc"] = {c: float(np.mean([r["per_cam"][c]["acc"] for r in data[cfg]
                                                      if c in r["per_cam"]]))
                                    for c in ("305", "318", "320", "323")}

    print(f"\n=== per-neighbor 分层 acc（mean over seeds） ===")
    for cfg in CFGS:
        for tag in ("has_neighbor", "no_neighbor"):
            vals = [r["per_nb"][tag]["acc"] for r in data[cfg] if tag in r.get("per_nb", {})]
            if vals:
                print(f"  {cfg:10s} {tag:13s}: {np.mean(vals):.4f} (n_seeds={len(vals)})")

    print(f"\n=== per-behavior-cluster acc（mean over seeds, cfg=behavior/both） ===")
    for cfg in ("behavior", "both", "base"):
        bcs = {}
        for r in data[cfg]:
            for k, d in r.get("per_bc", {}).items():
                bcs.setdefault(k, []).append(d["acc"])
        txt = " ".join(f"{k}:{np.mean(v):.3f}" for k, v in sorted(bcs.items()))
        print(f"  {cfg:10s} {txt}")

    json.dump(summ, open(os.path.join(D, "summary_v2.json"), "w"),
              ensure_ascii=False, indent=1)

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        x = np.arange(len(CFGS))
        for ax, key, name in ((axes[0], "acc", "test acc"),
                              (axes[1], "f1", "test macro-F1")):
            means = [summ[c][f"{key}_mean"] for c in CFGS]
            stds = [summ[c][f"{key}_std"] for c in CFGS]
            ax.bar(x, means, yerr=stds, capsize=4,
                   color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
            ax.set_xticks(x)
            ax.set_xticklabels(CFGS)
            ax.set_ylabel(name)
            ax.grid(alpha=0.3, axis="y")
            for xi, m in zip(x, means):
                ax.text(xi, m + 0.005, f"{m:.3f}", ha="center", fontsize=9)
        axes[0].set_title("Intent v2: config comparison (3 seeds)")
        fig.tight_layout()
        fig.savefig(os.path.join(D, "v2_compare.png"), dpi=140)
        print(f"[fig] {os.path.join(D, 'v2_compare.png')}")

        # per-cam 细图
        fig, ax = plt.subplots(figsize=(9, 4.4))
        w = 0.8 / len(CFGS)
        cams = ("305", "318", "320", "323")
        xx = np.arange(len(cams))
        for i, cfg in enumerate(CFGS):
            vals = [summ[cfg]["per_cam_acc"][c] for c in cams]
            ax.bar(xx + i * w - 0.4 + w / 2, vals, w, label=cfg)
        ax.set_xticks(xx)
        ax.set_xticklabels([f"cam {c}" for c in cams])
        ax.set_ylabel("test acc")
        ax.set_title("Intent v2: per-camera acc (3-seed mean)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(D, "v2_percam.png"), dpi=140)
        print(f"[fig] {os.path.join(D, 'v2_percam.png')}")
    except Exception as e:
        print("[fig] skip:", e)


if __name__ == "__main__":
    main()
