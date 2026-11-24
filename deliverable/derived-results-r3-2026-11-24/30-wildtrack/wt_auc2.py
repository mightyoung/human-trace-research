"""WILDTRACK · 阶段 C2：等量身份重采样，检验「共视 → AUC」的零相关是否稳健。

阶段 C 发现两个问题：
  1) 共视率与 AUC 的相关随样本过滤而消失（全 21 对 r=+0.37，n_id≥30 的 17 对 r=-0.05）
  2) 共享身份数 n_id 与 AUC 强负相关（228 个 → 0.71；43 个 → 0.90）——这是**选择效应**：
     能在两台相距很远的相机里都出现的，往往是显眼的人；共享集越大，越会纳入普通衣着的人。

做法：每对**固定抽 15 个身份**（该对共享集内随机），重复 200 次取均值与 2.5/97.5 分位，
使各对的「身份构成难度」在估计层面可比；再与共视率、几何重叠求相关。

产物：tmp/wt/auc_equal_k.json
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path("tmp/wt")
NV = 7
K = 15
REP = 200


def auc_of(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort().astype(float)
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def spear(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    meta = json.loads((OUT / "crops_wt/meta.json").read_text(encoding="utf-8"))
    E = np.load(OUT / "wt_emb.npy").astype(np.float32)
    res = json.loads((OUT / "wildtrack_p2.json").read_text(encoding="utf-8"))
    geo = json.loads((OUT / "geo_overlap.json").read_text(encoding="utf-8"))
    ov = geo["overlap"]

    byv = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(meta):
        byv[m["k"]][m["pid"]].append((i, m["stem"]))

    rng = np.random.default_rng(7)
    rows = []
    for i, j in combinations(range(NV), 2):
        shared = sorted(set(byv[i]) & set(byv[j]))
        rec = dict(a=i, b=j,
                   cov=next(r["cov"] for r in res["pairs"] if r["a"] == i and r["b"] == j),
                   overlap=ov[f"{i}-{j}"], n_id=len(shared))
        if len(shared) < K:
            rec["auc"] = None
            rows.append(rec)
            continue
        # 预算每对每身份的得分向量，避免重复计算
        pos_by, neg_pool = {}, {}
        for pid in shared:
            Ai, Bj = byv[i][pid], byv[j][pid]
            pos_by[pid] = np.array([float(E[a] @ E[b]) for a, sa in Ai
                                    for b, sb in Bj if sa != sb])
            neg_pool[pid] = [p for p in shared if p != pid]

        vals = []
        for _ in range(REP):
            pick = rng.choice(shared, K, replace=False)
            pos = np.concatenate([pos_by[p] for p in pick])
            neg = np.array([float(E[byv[i][p][rng.integers(len(byv[i][p]))][0]]
                                  @ E[byv[j][q][rng.integers(len(byv[j][q]))][0]])
                            for p in pick for q in neg_pool[p]])
            a = auc_of(pos, neg)
            if a is not None:
                vals.append(a)
        if len(vals) < 30:
            rec["auc"] = None
        else:
            rec["auc"] = round(float(np.mean(vals)), 4)
            rec["ci"] = [round(float(np.percentile(vals, 2.5)), 3),
                         round(float(np.percentile(vals, 97.5)), 3)]
        rows.append(rec)

    print(f"等量身份重采样：每对 {K} 个身份 × {REP} 次")
    print("=" * 78)
    print(f"{'对':<9}{'共视':>7}{'几何重叠':>9}{'n_id':>6}{'AUC(等量)':>11}{'95%CI':>16}")
    print("-" * 78)
    for r in sorted(rows, key=lambda z: -z["cov"]):
        ci = r.get("ci") or [None, None]
        c = f"[{ci[0]},{ci[1]}]" if ci[0] is not None else "—"
        a = f"{r['auc']:.3f}" if r["auc"] is not None else "—"
        print(f"C{r['a']+1}-C{r['b']+1:<6}{r['cov']:>7.2f}{r['overlap']:>9.2f}"
              f"{r['n_id']:>6}{a:>11}{c:>16}")

    v = [r for r in rows if r["auc"] is not None]
    print(f"\n可用对 {len(v)} / {len(rows)}（要求共享身份 ≥{K}）")
    if len(v) >= 6:
        for key in ("cov", "overlap"):
            x = [r[key] for r in v]
            y = [r["auc"] for r in v]
            print(f"  {key:<8} vs AUC(等量)：r={np.corrcoef(x, y)[0,1]:+.3f}  ρ={spear(x, y):+.3f}")
    # n_id 与 AUC 的关系（未等量时显著，等量后应消失）
    x = [r["n_id"] for r in v]
    y = [r["auc"] for r in v]
    print(f"  n_id     vs AUC(等量)：ρ={spear(x, y):+.3f}  （若仍强负相关，说明构成效应未消）")

    (OUT / "auc_equal_k.json").write_text(
        json.dumps(dict(k=K, rep=REP, pairs=rows), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("-> tmp/wt/auc_equal_k.json")


if __name__ == "__main__":
    main()
