"""WILDTRACK · 阶段 D2：用内参+外参算**几何重叠**，反推相机编号映射。

为什么必须做：C1 与 C4 的相机中心只差 15 cm，但实测共视率仅 0.15——物理上矛盾，
说明「C1–C4=CVLab1–4」这个假设的设备名对应是错的。若不先定映射，
任何「几何 → AUC」的结论都建立在一个可能错位的表上。

做法：
  1. 在地面 z=0 铺 80×80 网格（世界单位 cm，范围 ±20 m）
  2. 每个网格点按内外参投影到每个相机，落在图像内即「该相机看得到」
  3. 预测重叠 = |两台都看到| / |至少一台看到|（与实测共视率的定义同构）
  4. 枚举 7! 种设备↔视角映射，取「预测重叠 vs 实测共视率」Spearman 最高者
  5. 用正确映射重算几何量，并与 AUC 相关

产物：tmp/wt/geo_overlap.json
"""
import json
import re
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

ROOT = Path("tmp/wt/data/Wildtrack_dataset")
CAL = ROOT / "calibrations"
DEV = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
NV = 7
W, H = 1920, 1080


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def load_cam(name):
    s = (CAL / f"extrinsic/extr_{name}.xml").read_text(encoding="utf-8")
    rv = np.array([float(x) for x in re.search(r"<rvec>(.*?)</rvec>", s, re.S).group(1).split()])
    tv = np.array([float(x) for x in re.search(r"<tvec>(.*?)</tvec>", s, re.S).group(1).split()])
    R = rodrigues(rv)
    C = -R.T @ tv
    d = R.T @ np.array([0.0, 0.0, 1.0])

    t = (CAL / f"intrinsic_original/intr_{name}.xml").read_text(encoding="utf-8")
    data = re.search(r"<camera_matrix.*?<data>(.*?)</data>", t, re.S).group(1).split()
    K = np.array([float(x) for x in data]).reshape(3, 3)
    return dict(R=R, t=tv, C=C, d=d / np.linalg.norm(d), K=K)


def main():
    cams = {n: load_cam(n) for n in DEV}
    for n, c in cams.items():
        print(f"{n:<7} C=({c['C'][0]:>7.0f},{c['C'][1]:>7.0f},{c['C'][2]:>6.0f}) cm  "
              f"f={c['K'][0,0]:.0f}")
    print("\n设备间基线（m）")
    print("        " + "".join(f"{n[:6]:>8}" for n in DEV))
    for a in DEV:
        cells = []
        for b in DEV:
            if a == b:
                cells.append(f"{'—':>8}")
            else:
                d = np.linalg.norm(cams[a]["C"] - cams[b]["C"]) / 100.0
                cells.append(f"{d:>8.1f}")
        print(f"{a:<8}" + "".join(cells))

    # 地面网格（cm）
    xs = np.arange(-2000, 2001, 60.0)
    ys = np.arange(-2000, 2001, 60.0)
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)   # (N,3)

    vis = {}
    for n, c in cams.items():
        Xc = (c["R"] @ P.T).T + c["t"]                 # 世界 -> 相机
        Z = Xc[:, 2]
        ok = Z > 1.0
        u = np.full(len(P), -1.0)
        v = np.full(len(P), -1.0)
        K = c["K"]
        u[ok] = K[0, 0] * Xc[ok, 0] / Z[ok] + K[0, 2]
        v[ok] = K[1, 1] * Xc[ok, 1] / Z[ok] + K[1, 2]
        vis[n] = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        print(f"{n} 地面可见点 {vis[n].sum()} / {len(P)}"
              f"（占网格 {100*vis[n].mean():.0f}%）")

    meas = {}
    res = json.loads(Path("tmp/wt/wildtrack_p2.json").read_text(encoding="utf-8"))
    for r in res["pairs"]:
        meas[(r["a"], r["b"])] = r["cov"]
        meas[(r["b"], r["a"])] = r["cov"]

    def overlap(a, b):
        both = np.count_nonzero(vis[a] & vis[b])
        either = np.count_nonzero(vis[a] | vis[b])
        return both / max(either, 1)

    def spear(x, y):
        rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
        ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    # 枚举映射
    best = []
    for perm in permutations(DEV):
        xs_, ys_ = [], []
        for i, j in combinations(range(NV), 2):
            xs_.append(overlap(perm[i], perm[j]))
            ys_.append(meas[(i, j)])
        rho = spear(xs_, ys_)
        best.append((rho, perm, xs_))
    best.sort(key=lambda z: -z[0])
    print("\n映射搜索（按 预测重叠 vs 实测共视 ρ 排序，前 4）")
    for rho, perm, _ in best[:4]:
        print(f"  ρ={rho:+.3f}  " + " ".join(f"C{i+1}={p}" for i, p in enumerate(perm)))

    rho, perm, pred = best[0]
    print(f"\n采用映射：ρ={rho:+.3f}")
    print(f"{'对':<8}{'实测共视':>9}{'几何预测':>9}")
    for k, (i, j) in enumerate(combinations(range(NV), 2)):
        print(f"C{i+1}-C{j+1:<5}{meas[(i,j)]:>9.2f}{pred[k]:>9.2f}")

    # 正确映射下的几何量 + AUC
    pairs = []
    for k, (i, j) in enumerate(combinations(range(NV), 2)):
        a, b = perm[i], perm[j]
        ca, cb = cams[a], cams[b]
        ang = float(np.degrees(np.arccos(np.clip(float(ca["d"] @ cb["d"]), -1, 1))))
        base = float(np.linalg.norm(ca["C"] - cb["C"]) / 100.0)
        rec = dict(a=i, b=j, cov=meas[(i, j)], overlap=round(pred[k], 3),
                   axis_deg=round(ang, 1), base_m=round(base, 1))
        pr = next(r for r in res["pairs"] if r["a"] == i and r["b"] == j)
        rec["auc_cross"] = pr.get("auc_cross")
        rec["auc_same"] = pr.get("auc_same")
        rec["n_id"] = pr.get("n_id")
        pairs.append(rec)

    print("\n" + "=" * 86)
    print(f"{'对':<8}{'共视':>7}{'几何重叠':>9}{'光轴°':>8}{'基线m':>7}{'共享ID':>8}"
          f"{'AUC跨':>8}{'AUC同':>8}")
    print("-" * 86)
    for p in sorted(pairs, key=lambda z: -z["overlap"]):
        print(f"C{p['a']+1}-C{p['b']+1:<5}{p['cov']:>7.2f}{p['overlap']:>9.2f}"
              f"{p['axis_deg']:>8.1f}{p['base_m']:>7.1f}{p['n_id']:>8}"
              f"{(p['auc_cross'] if p['auc_cross'] else float('nan')):>8.3f}"
              f"{(p['auc_same'] if p['auc_same'] else float('nan')):>8.3f}")

    def stat(key, tgt, filt=None):
        v = [p for p in pairs if p.get(tgt) is not None and p.get(key) is not None
             and (filt is None or filt(p))]
        if len(v) < 5:
            return None
        x = [p[key] for p in v]
        y = [p[tgt] for p in v]
        rr = float(np.corrcoef(x, y)[0, 1])
        return (len(v), rr, spear(x, y))

    print("\n相关（r / ρ）")
    for tgt in ("auc_cross", "auc_same"):
        for key in ("cov", "overlap", "axis_deg", "base_m"):
            for tag, filt in (("全部", None), ("n_id≥30", lambda p: (p["n_id"] or 0) >= 30)):
                s = stat(key, tgt, filt)
                if s:
                    print(f"  {tgt:<10}{key:<10}{tag:<9}n={s[0]:<3} r={s[1]:+.3f} ρ={s[2]:+.3f}")

    # 两变量 OLS：几何重叠 + 共视
    v = [p for p in pairs if p["auc_cross"] is not None]
    X = np.array([[1.0, p["overlap"], p["cov"]] for p in v])
    y = np.array([p["auc_cross"] for p in v])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred_y = X @ beta
    r2 = 1 - ((y - pred_y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    print(f"\nOLS AUC ~ 1 + 几何重叠 + 实测共视  β={np.round(beta,4).tolist()}  R²={r2:.3f}")

    Path("tmp/wt/geo_overlap.json").write_text(
        json.dumps(dict(mapping=list(perm), rho_mapping=rho,
                        overlap={f"{i}-{j}": float(pred[k])
                                 for k, (i, j) in enumerate(combinations(range(NV), 2))},
                        pairs=pairs), ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> tmp/wt/geo_overlap.json")


if __name__ == "__main__":
    main()
