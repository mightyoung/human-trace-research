"""两套人员向量在人工真值上的可分性对比（跨机位到底是不是「特征不行」）。

背景：126 条人工真值（deliverable/gt_long35/gt_long35.json，7 个机位对）里，
OSNet-x1.0 的跨机位余弦 AUC 只有 0.490——等于随机。这直接决定了全局 ReID 的天花板。
本脚本回答一个具体问题：**换一套预训练向量，跨机位可分性会不会起来？**

对比对象：
  osnet  —— 生产链路在用的 OSNet-x1.0-MSMT17（512d，轨级均值）
  eg2    —— 属性线在用的 Gemma 视觉图向量（768d，轨级池化，eg2_emb_all.npz）

口径（沿用主线的分层报告，不混池）：
  - 只报「跨机位」对；同机位对单列，仅作参照
  - AUC = 同人对 cos > 异人对 cos 的概率（Mann-Whitney），带 bootstrap 95% CI
  - 另报「阈值下能接通多少 GT 正例」，因为最终要的是合并决策而不是排序
"""
import collections
import json

import numpy as np

ROOT = "."
CAMS = ["305", "316", "318", "320", "323", "343"]
GT = "deliverable/gt_long35/gt_long35.json"
rng = np.random.default_rng(20261112)


def load_osnet():
    E = {}
    for cam in CAMS:
        z = np.load(f"output_crosscam/long35/embs_osnet/{cam}_emb_long35_osnet.npz",
                    allow_pickle=True)
        for k in z.files:
            E[(k.split("_")[0].lstrip("v"), int(k.split("_")[1]))] = z[k].astype(np.float32)
    return E


def load_eg2():
    z = np.load("tmp/full/eg2_emb_all.npz", allow_pickle=True)
    keys = [str(k) for k in z["keys"]]
    E = {}
    for i, k in enumerate(keys):
        cam, tid = k.split("#")
        E[(cam, int(tid))] = z["E"][i].astype(np.float32)
    return E


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def cam_center(U):
    """每机位去均值：跨机位的主要干扰是机位级系统偏移（视角/光照/尺度），
    全局 ReID 阶段只需要「不同机位之间」可比，不需要机位内可比（同机位已由聚类解决）。"""
    by = collections.defaultdict(list)
    for k, v in U.items():
        by[k[0]].append(v)
    mu = {c: np.mean(np.stack(vs), axis=0) for c, vs in by.items()}
    return {k: unit(v - mu[k[0]]) for k, v in U.items()}, mu


def cam_whiten(U, ratio=1 / 3.0):
    """每机位白化（PCA 白化）。★ 维度必须各机位统一，否则跨机位向量无法点积
    （第一版按 n/3 各机位各自取维，跑出 139 维 vs 60 维的 ValueError）。"""
    ks_by_cam = {c: [k for k in U if k[0] == c] for c in CAMS}
    d = min(max(8, int(len(v) * ratio)) for v in ks_by_cam.values() if v)
    out, info = {}, {"d": d}
    for cam, ks in ks_by_cam.items():
        if len(ks) < 12:
            for k in ks:
                out[k] = U[k]
            continue
        X = np.stack([U[k] for k in ks])
        mu = X.mean(0)
        Xc = X - mu
        Uu, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        W = Vt[:d].T / (S[:d] / np.sqrt(len(ks)) + 1e-6)
        Y = Xc @ W
        info[cam] = {"n": len(ks), "d": d}
        for i, k in enumerate(ks):
            out[k] = unit(Y[i])
    return out, info


def auc(pos, neg):
    """P(pos > neg)，并列记 0.5。"""
    if not pos or not neg:
        return float("nan")
    a = np.asarray(pos)[:, None]
    b = np.asarray(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (len(pos) * len(neg)))


def boot_ci(pos, neg, n=2000):
    if len(pos) < 2 or len(neg) < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n):
        p = [pos[i] for i in rng.integers(0, len(pos), len(pos))]
        q = [neg[i] for i in rng.integers(0, len(neg), len(neg))]
        vals.append(auc(p, q))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    gt = json.load(open(GT, encoding="utf-8"))["items"]
    sets = {"osnet": load_osnet(), "eg2": load_eg2()}
    print("向量覆盖：", {k: len(v) for k, v in sets.items()})
    # 三种变换：raw / 机位去均值 / 机位白化
    variants = {}
    for name, E in sets.items():
        U = {k: unit(v) for k, v in E.items()}
        variants[name] = U
        variants[name + "+cc"] = cam_center(U)[0]
        variants[name + "+cw"], info = cam_whiten(U)
        print(f"{name} 白化维度：", info)

    for name, U in variants.items():
        rows = collections.defaultdict(lambda: {"pos": [], "neg": [], "pairs": []})
        miss = collections.Counter()
        for it in gt:
            ca, ta = it["a"].split(":")
            cb, tb = it["b"].split(":")
            A, B = (ca, int(ta)), (cb, int(tb))
            if A not in U or B not in U:
                miss["missing"] += 1
                continue
            c = float(np.dot(U[A], U[B]))
            cross = ca != cb
            key = "cross" if cross else "same_cam"
            (rows[key]["pos"] if it["label"] == 1 else rows[key]["neg"]).append(c)
            rows[key]["pairs"].append((it["pair"], it["label"], c))
            rows["pair:" + it["pair"]][("pos" if it["label"] == 1 else "neg")].append(c)

        raw = name in ("osnet", "eg2")
        p = rows["cross"]["pos"]
        q = rows["cross"]["neg"]
        lo, hi = boot_ci(p, q)
        print(f"\n{name:14s} 跨机位 同人 n={len(p):3d} 中位 {np.median(p):.3f}"
              f" | 异人 n={len(q):3d} 中位 {np.median(q):.3f}"
              f" | AUC {auc(p, q):.3f} [{lo:.3f},{hi:.3f}]")
        # 阈值扫描按分位数给：跨机位合并真正关心「在低误接率下能接通多少」
        for fp_rate in (0.02, 0.05, 0.10):
            thr = float(np.quantile(q, 1 - fp_rate))
            hit = sum(1 for x in p if x >= thr)
            print(f"    误接率 {fp_rate:.0%} (阈值 {thr:.3f}): 接通 {hit}/{len(p)} 正例")
        if raw:
            sk = rows["same_cam"]
            print(f"    [参照] 同机位 AUC {auc(sk['pos'], sk['neg']):.3f}")
            print(f"  {'机位对':10s}{'同':>4s}{'异':>4s}{'同人中位':>9s}{'异人中位':>9s}{'AUC':>8s}")
            for pr in sorted({it["pair"] for it in gt}):
                r = rows.get("pair:" + pr) or {}
                pp, qq = r.get("pos") or [], r.get("neg") or []
                med = lambda xs: float(np.median(xs)) if xs else float("nan")
                print(f"  {pr:10s}{len(pp):4d}{len(qq):4d}{med(pp):9.3f}{med(qq):9.3f}{auc(pp, qq):8.3f}")


if __name__ == "__main__":
    main()
