"""WILDTRACK · 阶段 E：区域级检验（论文 P2 原始命题的直接实验）。

论文 §P2 的原话是「尚未在共享区域内测过外观 AUC；P2 预测该处 L1 可用」。
对级检验（阶段 C/C2）已经否掉「对级共视 → AUC」，但对级 ≠ 区域级，
所以这里做真正的区域级 A/B：

  1. 用标定把每个检测框的**落脚点**（框底中心）反投影到地面 z=0 → 世界坐标
  2. 两台相机的共同可见区域由地面网格给出（阶段 D2 的 vis 掩膜）
  3. 每条配对（同人跨时刻 / 异人）分两组：
       in  —— 两个框的落脚点都落在共同可见区域内
       out —— 至少一个不在
  4. 同一对相机上比较 AUC_in 与 AUC_out

若 P2 成立，AUC_in 应显著高于 AUC_out；若两者接近，则「共享区域」不构成
外观可用的条件，P2 只能证明「同时可见」，不能证明「可匹配」。

产物：tmp/wt/region_auc.json
"""
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("tmp/wt/data/Wildtrack_dataset")
CAL = ROOT / "calibrations"
OUT = Path("tmp/wt")
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
    t = (CAL / f"intrinsic_original/intr_{name}.xml").read_text(encoding="utf-8")
    data = re.search(r"<camera_matrix.*?<data>(.*?)</data>", t, re.S).group(1).split()
    K = np.array([float(x) for x in data]).reshape(3, 3)
    return dict(R=R, t=tv, C=-R.T @ tv, K=K)


def auc_of(pos, neg):
    if len(pos) < 8 or len(neg) < 8:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort().astype(float)
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    meta = json.loads((OUT / "crops_wt/meta.json").read_text(encoding="utf-8"))
    E = np.load(OUT / "wt_emb.npy").astype(np.float32)
    geo = json.loads((OUT / "geo_overlap.json").read_text(encoding="utf-8"))
    mapping = geo["mapping"]
    print("相机映射:", {f"C{i+1}": mapping[i] for i in range(NV)})
    cams = {n: load_cam(n) for n in set(mapping)}

    # 1) 地面网格与共同可见掩膜
    xs = np.arange(-2000, 2001, 60.0)
    ys = np.arange(-2000, 2001, 60.0)
    gx, gy = np.meshgrid(xs, ys)
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    vis = {}
    for i in range(NV):
        c = cams[mapping[i]]
        Xc = (c["R"] @ P.T).T + c["t"]
        Z = Xc[:, 2]
        ok = Z > 1.0
        u = np.full(len(P), -1.0)
        v = np.full(len(P), -1.0)
        u[ok] = c["K"][0, 0] * Xc[ok, 0] / Z[ok] + c["K"][0, 2]
        v[ok] = c["K"][1, 1] * Xc[ok, 1] / Z[ok] + c["K"][1, 2]
        vis[i] = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # 2) 落脚点反投影
    def ground(k, box):
        c = cams[mapping[k]]
        u = (box[0] + box[2]) / 2.0
        v = box[3]
        d = np.linalg.inv(c["K"]) @ np.array([u, v, 1.0])
        dw = c["R"].T @ d
        if abs(dw[2]) < 1e-9:
            return None
        s = -c["C"][2] / dw[2]
        if s <= 0 or s > 100000:
            return None
        return c["C"] + s * dw

    # 读 bbox
    box_of = {}
    for f in sorted((ROOT / "annotations_positions").glob("*.json")):
        for rec in json.loads(f.read_text(encoding="utf-8")):
            pid = rec.get("personID")
            for v in rec.get("views") or []:
                k = v.get("viewNum")
                if not isinstance(k, int) or not (0 <= k < NV):
                    continue
                x1, y1, x2, y2 = (v.get("xmin", -1), v.get("ymin", -1),
                                  v.get("xmax", -1), v.get("ymax", -1))
                if min(x1, y1, x2, y2) < 0 or x2 - x1 < 10 or y2 - y1 < 20:
                    continue
                box_of[(f.stem, pid, k)] = (x1, y1, x2, y2)

    def in_region(pt, i, j):
        """点是否落在 i、j 的共同可见区域内（容差 1 格）。"""
        if pt is None:
            return False
        gi = int(round((pt[0] - xs[0]) / 60.0))
        gj = int(round((pt[1] - ys[0]) / 60.0))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                a, b = gi + dx, gj + dy
                if 0 <= a < len(xs) and 0 <= b < len(ys):
                    idx = b * len(xs) + a
                    if vis[i][idx] and vis[j][idx]:
                        return True
        return False

    idx_by = defaultdict(lambda: defaultdict(list))
    for n, m in enumerate(meta):
        idx_by[m["k"]][m["pid"]].append((n, m["stem"]))

    rng = np.random.default_rng(11)
    rows = []
    for i, j in combinations(range(NV), 2):
        shared = sorted(set(idx_by[i]) & set(idx_by[j]))
        if len(shared) < 6:
            continue
        pos_in, pos_out, neg_in, neg_out = [], [], [], []
        for pid in shared:
            Ai, Bj = idx_by[i][pid], idx_by[j][pid]
            for ia, sa in Ai:
                for ib, sb in Bj:
                    if sa == sb:
                        continue
                    v = float(E[ia] @ E[ib])
                    pi = ground(i, box_of[(sa, pid, i)])
                    pj = ground(j, box_of[(sb, pid, j)])
                    ok = in_region(pi, i, j) and in_region(pj, i, j)
                    (pos_in if ok else pos_out).append(v)
            for p2 in shared:
                if p2 == pid:
                    continue
                ia, sa = Ai[rng.integers(len(Ai))]
                jb = idx_by[j][p2]
                ib, sb = jb[rng.integers(len(jb))]
                v = float(E[ia] @ E[ib])
                pi = ground(i, box_of[(sa, pid, i)])
                pj = ground(j, box_of[(sb, p2, j)])
                ok = in_region(pi, i, j) and in_region(pj, i, j)
                (neg_in if ok else neg_out).append(v)

        a_in = auc_of(np.array(pos_in), np.array(neg_in))
        a_out = auc_of(np.array(pos_out), np.array(neg_out))
        rows.append(dict(a=i, b=j, n_pos_in=len(pos_in), n_neg_in=len(neg_in),
                         n_pos_out=len(pos_out), n_neg_out=len(neg_out),
                         auc_in=None if a_in is None else round(a_in, 4),
                         auc_out=None if a_out is None else round(a_out, 4),
                         med_in=None if not pos_in else round(float(np.median(pos_in)), 4),
                         med_out=None if not pos_out else round(float(np.median(pos_out)), 4)))
        print(f"C{i+1}-C{j+1}: in {len(pos_in)}/人 · {len(neg_in)}/异 → "
              f"AUC {a_in if a_in is None else round(a_in,3)} | "
              f"out {len(pos_out)}/{len(neg_out)} → {a_out if a_out is None else round(a_out,3)}",
              flush=True)

    both = [r for r in rows if r["auc_in"] is not None and r["auc_out"] is not None]
    print(f"\n可同时比较的对：{len(both)}")
    if both:
        d = [r["auc_in"] - r["auc_out"] for r in both]
        print("  AUC_in − AUC_out：中位 %+.3f，范围 %+.3f…%+.3f，in>out 的占 %d/%d"
              % (float(np.median(d)), min(d), max(d),
                 sum(1 for x in d if x > 0), len(d)))
        print("  按对：")
        for r in both:
            print(f"    C{r['a']+1}-C{r['b']+1}  in {r['auc_in']:.3f} (n={r['n_pos_in']})"
                  f"  out {r['auc_out']:.3f} (n={r['n_pos_out']})  Δ{r['auc_in']-r['auc_out']:+.3f}")

    (OUT / "region_auc.json").write_text(
        json.dumps(dict(mapping=mapping, pairs=rows), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("-> tmp/wt/region_auc.json")


if __name__ == "__main__":
    main()
