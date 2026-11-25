"""像素扫描的 AUC 重算（离线，不重编码）。

用 wt_sweep_gpu.py 落盘的 sweep_{p}.npz（每档每人裁剪的 OSNet 嵌入）重算等量身份
重采样的 AUC，只把 AUC 换成本次修正后的实现（1 起秩 + 并列平均秩）。
采样顺序与 wt_sweep_gpu.py 完全一致（rng seed 7、K=15、REP=200、stem 取 10 帧）。
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

WT = Path("tmp/wt")
NV, K, REP, N_STEM = 7, 15, 200, 10
SCALES = [320, 160, 112, 80, 56, 40, 24, 16]


def auc_fixed(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def auc_old(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort().astype(float)
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    meta = json.loads((WT / "crops_wt/meta.json").read_text(encoding="utf-8"))
    stems = sorted({m["stem"] for m in meta})
    idx = np.linspace(0, len(stems) - 1, N_STEM).round().astype(int)
    keep = {stems[i] for i in sorted(set(idx))}
    meta = [m for m in meta if m["stem"] in keep]

    byv = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(meta):
        byv[m["k"]][m["pid"]].append((i, m["stem"]))

    old = json.loads(Path("deliverable/wildtrack-p2-2026-11-24/pixel_sweep.json")
                     .read_text(encoding="utf-8"))
    out = {"scales": SCALES, "n_stem": N_STEM, "k": K, "rep": REP,
           "note": "AUC 已修正（1 起秩 + 并列平均秩）；采样与旧版一致", "pairs": {}}
    summary = []
    for p in SCALES:
        E = np.load(WT / f"sweep_{p}.npz")["E"].astype(np.float32)
        assert E.shape[0] == len(meta), (p, E.shape, len(meta))
        rng = np.random.default_rng(7)
        rows, rows_old = [], []
        for i, j in combinations(range(NV), 2):
            shared = sorted(set(byv[i]) & set(byv[j]))
            if len(shared) < K:
                continue
            pos_by, neg_pool = {}, {}
            for pid in shared:
                Ai, Bj = byv[i][pid], byv[j][pid]
                pos_by[pid] = np.array([float(E[a] @ E[b]) for a, sa in Ai
                                        for b, sb in Bj if sa != sb])
                neg_pool[pid] = [q for q in shared if q != pid]
            vals, vals_old = [], []
            for _ in range(REP):
                pick = rng.choice(shared, K, replace=False)
                pos = np.concatenate([pos_by[x] for x in pick])
                neg = np.array([float(E[byv[i][x][rng.integers(len(byv[i][x]))][0]]
                                      @ E[byv[j][y][rng.integers(len(byv[j][y]))][0]])
                                for x in pick for y in neg_pool[x]])
                vals.append(auc_fixed(pos, neg))
                vals_old.append(auc_old(pos, neg))
            vals = [v for v in vals if v is not None]
            rows.append({"a": i, "b": j, "auc": round(float(np.mean(vals)), 4),
                         "auc_old": round(float(np.mean([v for v in vals_old
                                                         if v is not None])), 4),
                         "n": len(vals)})
        out["pairs"][str(p)] = rows
        med = float(np.median([r["auc"] for r in rows]))
        med_old = float(np.median([r["auc_old"] for r in rows]))
        summary.append((p, med_old, med, len(rows)))
        print("  %3d px：旧中位 %.4f → 新中位 %.4f（%d 对）" % (p, med_old, med, len(rows)),
              flush=True)

    (WT / "pixel_sweep_fixed.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n旧文件对照（deliverable 版）：")
    for p in SCALES:
        rs = old["pairs"].get(str(p), [])
        if rs:
            print("  %3d px 旧记录中位 %.4f" % (p, float(np.median([r["auc"] for r in rs]))))
    print("-> tmp/wt/pixel_sweep_fixed.json")


if __name__ == "__main__":
    main()
