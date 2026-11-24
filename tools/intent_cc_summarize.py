#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总意图 cc 实验（v1 10 模型 + v2 12 模型）：对角线对比 / seed 方差 / proto 配对。"""
import glob
import json
import os

import numpy as np

BASE = os.path.join("output_crosscam", "analysis", "intent", "cc")
CAMS = ["305", "318", "320", "323"]


def load_all(sub):
    out = {}
    for d in sorted(glob.glob(os.path.join(BASE, sub, "*", ""))):
        name = os.path.basename(os.path.normpath(d))
        p = os.path.join(d, "metrics_full.json")
        if not os.path.exists(p):
            continue
        out[name] = json.load(open(p, encoding="utf-8"))["rep"]
    return out


def main():
    allm = {**load_all("out"), **load_all("out2")}
    print(f"loaded {len(allm)} models\n")

    # ---- 表 1：全部模型概览 ----
    print(f"{'model':18s} {'cams':6s} {'mode':5s} | {'all acc':7s} {'all f1':6s} | "
          f"per-cam acc 305/318/320/323 | kn-acc unk-acc")
    print("-" * 108)
    for name in sorted(allm):
        r = allm[name]
        pc = r["per_cam"]
        pk = r["per_known"]
        percam = "/".join(f"{pc[c]['acc']:.3f}" for c in CAMS if c in pc)
        kn = pk.get("known", {}).get("acc", float("nan"))
        unk = pk.get("unknown", {}).get("acc", float("nan"))
        print(f"{name:18s} {str(r['config']['cams']):6s} {r['config']['id_mode']:5s} | "
              f"{r['overall']['acc']:.4f}  {r['overall']['macro_f1']:.4f} | {percam:25s} | {kn:.3f} {unk:.3f}")
    print()

    # ---- 表 2：对角线对比（分组模型 vs 全局模型，在分组自家子集上） ----
    print("== 对角线对比：per-camera test 子集 accuracy（同子集口径） ==")
    print(f"{'cam':5s} {'global':8s} {'grp-none':9s} {'grp-proto':10s} {'grp-none-v2':11s} "
          f"{'global-f1':9s} {'grp-v2-f1':9s}")
    diag = {}
    for c in CAMS:
        g = allm.get("global_none", {}).get("per_cam", {}).get(c, {})
        gn = allm.get(f"cam{c}_none", {}).get("per_cam", {}).get(c, {})
        gp = allm.get(f"cam{c}_proto", {}).get("per_cam", {}).get(c, {})
        gv2 = allm.get(f"cam{c}_none_v2", {}).get("per_cam", {}).get(c, {})
        diag[c] = {"global": g, "group": gn, "group_proto": gp, "group_v2": gv2}
        f = lambda d, k: (f"{d.get(k, float('nan')):.3f}" if d.get(k) is not None else "  -  ")
        print(f"{c:5s} {f(g,'acc'):8s} {f(gn,'acc'):9s} {f(gp,'acc'):10s} {f(gv2,'acc'):11s} "
              f"{f(g,'macro_f1'):9s} {f(gv2,'macro_f1'):9s}")
    print()

    # ---- 表 3：seed 重复（总体 + 对角线） ----
    print("== seed 重复汇总（seed 0/42/123） ==")
    seedsets = {
        "global_none": ["global_none_s0", "global_none", "global_none_s123"],
        "cam318_none": ["cam318_none_s0", "cam318_none", "cam318_none_s123"],
        "cam323_none": ["cam323_none_s0", "cam323_none", "cam323_none_s123"],
        "cam305_proto": ["cam305_proto_s0", "cam305_proto", "cam305_proto_s123"],
    }
    seed_rep = {}
    for k, names in seedsets.items():
        accs = [allm[n]["overall"]["acc"] for n in names if n in allm]
        f1s = [allm[n]["overall"]["macro_f1"] for n in names if n in allm]
        seed_rep[k] = {"accs": accs, "f1s": f1s,
                       "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                       "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
        print(f"  {k:14s} acc={[round(x,3) for x in accs]} mean={np.mean(accs):.4f} "
              f"std={np.std(accs):.4f} | f1 mean={np.mean(f1s):.4f}")
    print()

    # ---- 表 4：proto 配对（同 camera 同 seed42，none vs proto） ----
    print("== proto 条件化配对（全局 + 各组，seed42） ==")
    pairs = [("global", "global_none", "global_proto")] + \
            [(c, f"cam{c}_none", f"cam{c}_proto") for c in CAMS]
    for tag, n_name, p_name in pairs:
        if n_name in allm and p_name in allm:
            na, pa_ = allm[n_name]["overall"]["acc"], allm[p_name]["overall"]["acc"]
            print(f"  {tag:8s} none={na:.4f} proto={pa_:.4f} diff={pa_-na:+.4f}")
    print()

    json.dump({"diag": diag, "seed_rep": seed_rep,
               "all": {k: {"config": v["config"], "overall": {kk: v["overall"][kk] for kk in
                                                              ("acc", "macro_f1")},
                           "per_cam": {c: v["per_cam"].get(c, {}) for c in CAMS},
                           "per_known": {kk: {k2: v["per_known"].get(kk, {}).get(k2)
                                              for k2 in ("acc", "macro_f1", "n")}
                                         for kk in ("known", "unknown")}}
                      for k, v in allm.items()}},
              open(os.path.join(BASE, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(CAMS))
        w = 0.28
        gvals = [diag[c]["global"].get("acc", np.nan) for c in CAMS]
        nvals = [diag[c]["group"].get("acc", np.nan) for c in CAMS]
        v2vals = [diag[c]["group_v2"].get("acc", np.nan) for c in CAMS]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.bar(x - w, gvals, w, label="global (6710 smp)", color="#4C78A8")
        ax.bar(x, nvals, w, label="per-camera (44 ep)", color="#F58518")
        ax.bar(x + w, v2vals, w, label="per-camera (160 ep)", color="#54A24B")
        ax.set_xticks(x)
        ax.set_xticklabels([f"cam {c}" for c in CAMS])
        ax.set_ylabel("test acc on own subset")
        ax.set_ylim(0, 1.0)
        ax.set_title("Intent prediction: global vs per-camera models (own camera test subset)")
        for xi, v in zip(x - w, gvals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        for xi, v in zip(x, nvals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        for xi, v in zip(x + w, v2vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(BASE, "diag_compare.png"), dpi=140)
        print("[fig] diag_compare.png")
    except Exception as e:
        print("[fig] skip:", e)
    print("[done] summary ->", os.path.join(BASE, "summary.json"))


if __name__ == "__main__":
    main()
