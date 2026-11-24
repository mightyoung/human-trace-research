#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""簇级坏簇判别探针：在这 39 个已人工判定真值的图库簇上，找能标出坏簇的信号。

真值来源：`baseline_audit/index.csv` + `baseline_audit2/index.csv`（同簇多次判定 0 冲突）。
簇     ：`camera_aware_labels(Vtr, cams, 0.15, 0.25, MIN_CLUSTER)` 的 kept 下标（0..K-1）。

特征（只看图库侧，不用测试集，避免"用答案做题"）：
  n            簇成员数
  n_cam        横跨机位数
  cam_share    最大机位占比
  pw_min/max   簇内成对余弦的最小/最大值
  d_max        簇内最大成对距离 = 1 - pw_min（平均链式串人的直接指纹）
  d_p95        簇内成对距离的 95 分位
  cent_mean    到质心的平均距离
  mst_gap      MST 边排序后的最大间隙（链式合并会在两群之间留一条长边）
  sep_ratio    去掉最大 MST 边后，组内平均余弦 / 组间平均余弦（越大越像"两拨人"）
  sub_share    按 0.12 再切一次，最大子簇占比（越小越碎）

报告：
  1) 每个特征的 AUC（好坏簇区分度，0.5 = 无信息）
  2) 单特征规则在留一/分机位留出下的表现（能拦多少坏簇、误伤多少好簇）
  3) 不做自动决策：n=39、13 个坏簇，任何规则都只是候选，需第二次数据确认

用法：python tools/cluster_purity_probe.py
"""
import collections
import csv
import itertools
import os
import sys

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
import cluster_camera_aware as CA  # noqa: E402
from selector import _norm  # noqa: E402

EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
AUD = [os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/baseline_audit/index.csv"),
       os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/baseline_audit2/index.csv")]

B1 = dict(p.split("=") for p in """01=ok 02=ok 03=ok 04=ok 05=ok 06=ok 07=ok 08=bad 09=ok 10=bad
11=ok 12=ok 13=ok 14=bad 15=bad 16=ok 17=ok 18=bad 19=bad 20=bad 21=ok 22=bad 23=bad 24=ok 25=bad""".split())
B2 = dict(p.split("=") for p in """01=ok 02=ok 03=bad 04=ok 05=ok 06=ok 07=ok 08=bad 09=bad 10=ok
11=ok 12=ok 13=ok 14=ok 15=ok 16=bad 17=bad 18=ok 19=unk 20=bad 21=ok 22=unk 23=ok 24=unk
25=bad 26=bad""".split())


def load_truth():
    cl = collections.defaultdict(list)
    for p, v in zip(AUD, [B1, B2]):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            cl[int(r["cluster"])].append(v[r["image"][:2]])
    return {c: (([x for x in v if x != "unk"] or ["unk"])[0]) for c, v in cl.items()}


def features(V, cams, lab, kept, c):
    m = lab == kept[c]
    X = V[m]
    cc = collections.Counter(cams[m].tolist())
    n = int(m.sum())
    f = {"n": n, "n_cam": len(cc), "cam_share": max(cc.values()) / n}
    if n >= 2:
        S_ = X @ X.T
        iu = np.triu_indices(n, 1)
        pw = S_[iu]
        D = 1.0 - pw
        f["pw_min"] = float(pw.min())
        f["pw_max"] = float(pw.max())
        f["d_max"] = float(D.max())
        f["d_p95"] = float(np.percentile(D, 95))
        f["d_mean"] = float(D.mean())
        cent = X.mean(0)
        cent /= np.linalg.norm(cent) + 1e-12
        dc = 1.0 - X @ cent
        f["cent_mean"] = float(dc.mean())
        # MST：链式合并的指纹
        Dm = D.copy()
        Mst = minimum_spanning_tree(
            np.array([[0.0 if i == j else 1.0 - float(X[i] @ X[j]) for j in range(n)]
                      for i in range(n)], np.float64)).toarray()
        e = np.sort(Mst[Mst > 0])
        f["mst_max"] = float(e[-1]) if e.size else 0.0
        f["mst_gap"] = float(e[-1] - e[-2]) if e.size >= 2 else 0.0
        # 去掉最大 MST 边 → 两组，看组内/组间余弦比
        if e.size >= 1:
            big = e[-1]
            i, j = np.argwhere(np.isclose(Mst, big))[0]
            seen, stack = {int(i)}, [int(i)]
            while stack:
                u = stack.pop()
                for v in np.where(Mst[u] > 0)[0]:
                    if int(v) not in seen:
                        seen.add(int(v)); stack.append(int(v))
                for v in np.where(Mst[:, u] > 0)[0]:
                    if int(v) not in seen:
                        seen.add(int(v)); stack.append(int(v))
            A = np.array(sorted(seen))
            B = np.array([x for x in range(n) if x not in seen])
            if A.size >= 2 and B.size >= 2:
                ia = A[:, None], A[None, :]
                intra_A = (X[A] @ X[A].T)[np.triu_indices(A.size, 1)].mean()
                intra_B = (X[B] @ X[B].T)[np.triu_indices(B.size, 1)].mean()
                cross = (X[A] @ X[B].T).mean()
                f["sep_ratio"] = float(((intra_A + intra_B) / 2) / (cross + 1e-9))
            else:
                f["sep_ratio"] = float("nan")
        # 再切一刀看碎不碎
        from cluster_camera_aware import _labels_by_thr
        sub = _labels_by_thr(X, 0.12)
        cnt = collections.Counter(sub.tolist())
        f["sub_share"] = max(cnt.values()) / n
        f["sub_k"] = len(cnt)
    else:
        for k in ["pw_min", "pw_max", "d_max", "d_p95", "d_mean", "cent_mean",
                  "mst_max", "mst_gap", "sep_ratio", "sub_share", "sub_k"]:
            f[k] = float("nan")
    return f


def auc(pos, neg):
    pos = [x for x in pos if np.isfinite(x)]
    neg = [x for x in neg if np.isfinite(x)]
    if not pos or not neg:
        return float("nan")
    w = sum((1 if a > b else .5 if a == b else 0) for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def main():
    ktr, Va, Vb = M.load_split(EMB, S.TRAIN_SEGS)
    Vtr = _norm((Va + Vb) / 2.0)
    cams = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams, 0.15, 0.25, S.MIN_CLUSTER)
    truth = load_truth()

    rows = []
    for c in sorted(truth):
        f = features(Vtr, cams, lab, kept, c)
        f["c"] = c
        f["v"] = truth[c]
        rows.append(f)
    ok = [r for r in rows if r["v"] == "ok"]
    bad = [r for r in rows if r["v"] == "bad"]
    print(f"簇级真值：ok {len(ok)} / bad {len(bad)} / unk {len(rows)-len(ok)-len(bad)}")

    keys = ["n", "n_cam", "cam_share", "pw_min", "pw_max", "d_max", "d_p95", "d_mean",
            "cent_mean", "mst_max", "mst_gap", "sep_ratio", "sub_share", "sub_k"]
    print("\n== 单特征 AUC（>0.5 表示越大越坏；<0.5 表示越大越好）==")
    res = []
    for k in keys:
        a = auc([r[k] for r in bad], [r[k] for r in ok])
        # 统一成"越大越坏"的方向
        a_bad = a if a >= .5 else 1 - a
        res.append((a_bad, k, a))
        print(f"  {k:10s} AUC={a:.3f}   （按'越大越坏'折算 {a_bad:.3f}）")
    res.sort(reverse=True)

    print("\n== 中位数对比 ==  ok / bad")
    for k in ["n", "n_cam", "d_max", "d_p95", "mst_gap", "sep_ratio", "sub_share", "cent_mean"]:
        def med(s):
            v = [r[k] for r in s if np.isfinite(r[k])]
            return f"{np.median(v):.3f}" if v else "n/a"
        print(f"  {k:10s} {med(ok)} / {med(bad)}")

    print("\n== 简单阈值规则（在全部 39 簇上找最佳切点，仅作上限参考）==")
    for a_bad, k, _ in res[:5]:
        vals = sorted({r[k] for r in rows if np.isfinite(r[k])})
        best = None
        for t in vals:
            tp = sum(1 for r in bad if np.isfinite(r[k]) and r[k] >= t)
            fp = sum(1 for r in ok if np.isfinite(r[k]) and r[k] >= t)
            if best is None or (tp - fp) > (best[0] - best[1]):
                best = (tp, fp, t)
        tp, fp, t = best
        print(f"  {k:10s} 切点 >= {t:<8.3f} 拦到坏簇 {tp}/{len(bad)}，误伤好簇 {fp}/{len(ok)}")

    print("\n== 分机位留出（在 305+318 上选切点，用到 323/320 的簇上）==")
    cam_of = {}
    for c in truth:
        m = lab == kept[c]
        cam_of[c] = collections.Counter(cams[m].tolist()).most_common(1)[0][0]
    tr = [r for r in rows if cam_of[r["c"]] in ("305", "318")]
    te = [r for r in rows if cam_of[r["c"]] in ("323", "320")]
    print(f"  训练 {len(tr)} 簇（{sum(1 for r in tr if r['v']=='bad')} 坏）｜"
          f"留出 {len(te)} 簇（{sum(1 for r in te if r['v']=='bad')} 坏）")
    for a_bad, k, _ in res[:5]:
        vals = sorted({r[k] for r in tr if np.isfinite(r[k])})
        best = None
        for t in vals:
            tp = sum(1 for r in tr if r["v"] == "bad" and np.isfinite(r[k]) and r[k] >= t)
            fp = sum(1 for r in tr if r["v"] == "ok" and np.isfinite(r[k]) and r[k] >= t)
            if best is None or (tp - fp) > (best[0] - best[1]):
                best = (tp, fp, t)
        tp, fp, t = best
        te_tp = sum(1 for r in te if r["v"] == "bad" and np.isfinite(r[k]) and r[k] >= t)
        te_fp = sum(1 for r in te if r["v"] == "ok" and np.isfinite(r[k]) and r[k] >= t)
        nb = sum(1 for r in te if r["v"] == "bad")
        no = sum(1 for r in te if r["v"] == "ok")
        print(f"  {k:10s} 切点 >= {t:<8.3f}｜训练 拦{tp} 误{fp}｜留出 拦{te_tp}/{nb} 误{te_fp}/{no}")

    out = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/cluster_purity.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["c", "v"] + keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["v"], -r["n"])):
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k])
                        for k in w.fieldnames})
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
