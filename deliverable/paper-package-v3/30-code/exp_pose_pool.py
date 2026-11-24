#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""姿态/形态整合：池到底整合了什么？建库侧能不能零成本白拿？

用户提出的机制（本轮要检验的）
--------------------------
人脸触发写入的「最新全身特征」，不只是换衣兜底，更是**把同一人的不同姿势/动作
的特征整合进该身份模板**。若成立 → 运行时（实时池）与训练/建库时（多原型库）
都该有价值。

四个实验
--------
E1  复现基线：S0 仅历史库（单均值）/ S1 库 + 实时池（top-3）。确认分母没漂。
E2  **按形态差分档**：δh=|log(h_q/h_lib)|、δar=|log(ar_q/ar_lib)|。
    若 S1−S0 的增量集中在 δh 大的一档 → 「整合的是形态差异」被证实。
E3  **形态 oracle**：对每条查询，在上午同身份同机位轨里取**形态最接近**的那条
    作模板（S4）。这一条不依赖人脸、不依赖池、不因果，但它回答一个关键问题：
    「姿态对齐」本身值多少 pt —— 是池的魔法，还是只要找到形近模板就行。
E4  **多原型库**（建库侧整合，零训练）：上午每「机位×身份」出 K 个原型
    · P_h：按框高分位切 K 段，段内均值（尺度/姿态分层）
    · P_k：特征 k-means K 簇，簇心
    查询取 max over 原型。K=1 即单均值（退化）。若 K>1 有提升 → 建库侧成立。
E5  **写入质量门**：绝对形态门（框高下限 / 宽高比下限），回答 male10 那种
    「池只有 1 个烂样本反而变差」怎么防。

★ 口径声明（必须随结果引用）
  - 人脸「识别到谁」用 GT 身份 → 这是**上限**；「这条轨能否写入池」用真实可用脸
    （face_vecs 有 |full）→ 这部分现实（141/445 = 31.7%）。
  - 因果：所有池写入发生在该查询**打分之后**，查询看不到自己。
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_pos import load_emb            # noqa: E402

FPS = 25.0


def cam_of(k):
    return k.rsplit("#", 1)[0].split("_")[0]


def load_face(face_dir, proto="full"):
    out = {}
    for p in sorted(glob.glob(os.path.join(face_dir, "face_vecs_*.npz"))):
        z = np.load(p, allow_pickle=True)
        for k, v in zip(z["keys"], z["vecs"]):
            k = str(k)
            if "|" not in k:
                continue
            key, tag = k.rsplit("|", 1)
            if tag != proto:
                continue
            v = np.asarray(v, dtype=np.float64)
            n = float(np.linalg.norm(v))
            if n > 0:
                out[key] = v / n
    return out


def load_det_meta(det_dir, segs, keyset):
    """一次读完：起始帧 + 框高中位 + 宽高比中位。"""
    start, hh, aa = {}, collections.defaultdict(list), collections.defaultdict(list)
    for v in segs:
        p = os.path.join(det_dir, v, v + "_detections.jsonl")
        if not os.path.exists(p):
            print("  [warn] 缺检测 %s" % p)
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            f = int(r.get("frame", 0))
            tids = r.get("track_ids") or []
            bxs = r.get("boxes") or []
            for t, b in zip(tids, bxs):
                if t is None or int(t) < 0:
                    continue
                k = "%s#%d" % (v, int(t))
                if k not in keyset:
                    continue
                if k not in start or f < start[k]:
                    start[k] = f
                h = float(b[3]) - float(b[1])
                w = float(b[2]) - float(b[0])
                if h > 1.0 and w > 1.0:
                    hh[k].append(h)
                    aa[k].append(w / h)
    shape = {k: (float(np.median(v)), float(np.median(aa[k]))) for k, v in hh.items() if v}
    return start, shape


def topk_mean(E, qi, idx_list, k=3):
    """返回 (top-k 均值, 池内最像那条的索引)。索引用于查它的框形。"""
    if not idx_list:
        return -1.0, -1
    v = E[idx_list] @ E[qi]
    j = int(np.argmax(v))
    if len(v) > k:
        v = np.sort(v)[-k:]
    return float(v.mean()), idx_list[j]


def kmeans_np(X, k, iters=30, seed=0):
    """极简 k-means（k-means++ 初始化）。X: (n,d) 已归一化。n 很小时直接退化。"""
    n = len(X)
    if n <= k:
        return [X[i] for i in range(n)]
    rng = np.random.RandomState(seed)
    c = [X[rng.randint(n)]]
    for _ in range(k - 1):
        d = 1.0 - np.max(np.stack([X @ cc for cc in c], 1), 1)
        d = np.maximum(d, 1e-9)
        p = d / d.sum()
        c.append(X[rng.choice(n, p=p)])
    C = np.stack(c)
    for _ in range(iters):
        lab = np.argmax(X @ C.T, 1)
        newC = C.copy()
        for j in range(k):
            m = lab == j
            if m.sum() > 0:
                v = X[m].mean(0)
                newC[j] = v / max(np.linalg.norm(v), 1e-9)
        if np.allclose(newC, C, atol=1e-6):
            break
        C = newC
    return [C[j] for j in range(len(C))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--segs", required=True, help="空格分隔")
    ap.add_argument("--proto", default="full")
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--pool-max", type=int, default=8)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.75, 0.80, 0.85])
    ap.add_argument("--main-tau", type=float, default=0.80)
    ap.add_argument("--low-thr", type=float, default=0.60)
    ap.add_argument("--k-list", default="1,2,3,4")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    segs = args.segs.split()
    E64, keys = load_emb(args.emb_dir, segs)
    E = E64 / np.maximum(np.linalg.norm(E64, axis=1, keepdims=True), 1e-9)
    eidx = {k: i for i, k in enumerate(keys)}
    gt = json.load(open(args.gt, encoding="utf-8"))
    face = load_face(args.face, args.proto)
    start, shape = load_det_meta(args.det, segs, set(eidx))

    def is_train(k):
        return "_train_" in k

    rows = []
    for k in keys:
        if k not in gt:
            continue
        h, ar = shape.get(k, (-1.0, -1.0))
        rows.append(dict(k=k, i=eidx[k], pid=gt[k], cam=cam_of(k), tr=is_train(k),
                         t=start.get(k, 0), face=(k in face), h=h, ar=ar))
    n_noshape = sum(1 for r in rows if r["h"] < 0)
    print("评估轨 %d ｜ 有人脸 %d（%.1f%%）｜ 缺框形 %d ｜ 机位 %d"
          % (len(rows), sum(r["face"] for r in rows),
             100 * sum(r["face"] for r in rows) / max(len(rows), 1),
             n_noshape, len({r["cam"] for r in rows})))

    # 历史库（上午，按机位）
    lib = collections.defaultdict(list)
    for r in rows:
        if r["tr"]:
            lib[(r["cam"], r["pid"])].append(r)
    libv = {}
    for key, rs in lib.items():
        v = E[[x["i"] for x in rs]].mean(0)
        libv[key] = v / max(np.linalg.norm(v), 1e-9)
    hl = {key: float(np.median([x["h"] for x in rs if x["h"] > 0])) if any(
        x["h"] > 0 for x in rs) else -1.0 for key, rs in lib.items()}
    arl = {key: float(np.median([x["ar"] for x in rs if x["ar"] > 0])) if any(
        x["ar"] > 0 for x in rs) else -1.0 for key, rs in lib.items()}
    print("历史库 %d 个「机位×身份」组 ｜ 每组上午轨数 中位 %.1f"
          % (len(libv), np.median([len(v) for v in lib.values()])))

    queries = sorted([r for r in rows if not r["tr"]], key=lambda r: (r["cam"], r["t"], r["k"]))
    print("查询轨（下午）%d" % len(queries))
    ikey = {i: k for k, i in eidx.items()}

    # ---------- E4 多原型库（建库侧，与池无关） ----------
    Ks = [int(x) for x in args.k_list.split(",")]
    proto_h, proto_k = {}, {}
    for key, rs in lib.items():
        idx = [x["i"] for x in rs]
        order = np.argsort([x["h"] if x["h"] > 0 else 1e9 for x in rs])
        for K in Ks:
            # P_h：按框高分位切段，段内均值。每组样本不足时自动退化为 ≤K 段
            n = len(idx)
            kk = max(1, min(K, n))
            bounds = np.linspace(0, n, kk + 1).astype(int)
            vs = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b <= a:
                    continue
                sel = [idx[order[j]] for j in range(a, b)]
                v = E[sel].mean(0)
                vs.append(v / max(np.linalg.norm(v), 1e-9))
            proto_h[(key, K)] = np.stack(vs) if vs else None
            # P_k：特征 k-means
            if K == 1:
                proto_k[(key, K)] = np.stack([libv[key]])
            else:
                c = kmeans_np(E[idx], max(1, min(K, n)))
                proto_k[(key, K)] = np.stack(c)

    def proto_score(q, table, K, p):
        M = table[((q["cam"], p), K)]
        if M is None:
            return -1.0
        return float(np.max(M @ E[q["i"]]))

    # ---------- 主循环（因果池） ----------
    pool = collections.defaultdict(list)
    rec = []
    for q in queries:
        cands = [p for (sc, p) in libv if sc == q["cam"]]
        if not cands:
            continue
        key = (q["cam"], q["pid"])
        libsc = {p: float(E[q["i"]] @ libv[(q["cam"], p)]) for p in cands}
        poolsc, pbest = {}, {}
        for p in cands:
            v, bi = topk_mean(E, q["i"], pool[(q["cam"], p)], args.pool_k)
            poolsc[p], pbest[p] = v, bi
        s1 = {p: max(libsc[p], poolsc[p]) for p in cands}
        # 池里被用上的那条（本人）的框形：验「池是不是在做形态匹配」
        own_b = pbest.get(q["pid"], -1)
        bh = shape[ikey[own_b]][0] if (own_b >= 0 and ikey.get(own_b) in shape) else -1.0
        # 形态 oracle：上午同身份同机位里 h 最接近的一条（仅本人，作参考）
        s4 = None
        if key in lib and q["h"] > 0:
            rs = [x for x in lib[key] if x["h"] > 0]
            if rs:
                j = int(np.argmin([abs(np.log(x["h"] / q["h"])) for x in rs]))
                s4 = float(E[q["i"]] @ E[rs[j]["i"]])
        d_h = np.log(q["h"] / hl[key]) if (q["h"] > 0 and hl.get(key, -1) > 0) else np.nan
        d_ar = np.log(q["ar"] / arl[key]) if (q["ar"] > 0 and arl.get(key, -1) > 0) else np.nan
        d = dict(q=q, s0=libsc, s1=s1, s4=s4,
                 n_pool=len(pool[(q["cam"], q["pid"])]),
                 d_h=d_h, d_ar=d_ar, h=q["h"], ar=q["ar"],
                 h_lib=hl.get(key, -1.0), ar_lib=arl.get(key, -1.0),
                 pool_best_h=bh,
                 pool_best_dh=(np.log(bh / q["h"]) if (bh > 0 and q["h"] > 0) else np.nan))
        rec.append(d)
        # 因果写入
        if q["face"] and q["pid"] in cands:
            lst = pool[(q["cam"], q["pid"])]
            lst.append(q["i"])
            if len(lst) > args.pool_max:
                del lst[0]

    def top1(v):
        return max(v, key=v.get)

    def cov_same(key, tau, sub=None):
        sub = rec if sub is None else sub
        ans = [r for r in sub if max(r[key].values()) >= tau]
        if not ans:
            return 0.0, 0.0, 0
        same = np.mean([top1(r[key]) == r["q"]["pid"] for r in ans])
        return 100.0 * len(ans) / len(sub), 100.0 * same, len(ans)

    print("\n===== E1 基线复现（S0 仅库 / S1 库+池）=====")
    print("  %-5s %-19s %-19s %-19s" % ("τ", "S0 cov/same", "S1 cov/same", "Δsame"))
    for tau in args.tau:
        c0, s0, _ = cov_same("s0", tau)
        c1, s1, _ = cov_same("s1", tau)
        print("  %-5.2f %6.0f%% / %5.1f%%     %6.0f%% / %5.1f%%     %+6.1f"
              % (tau, c0, s0, c1, s1, s1 - s0))

    mt = args.main_tau
    print("\n===== E2 按形态差分档（δh=|log(查询框高/库框高中位)|，τ=%.2f）=====" % mt)
    dhs = [r["d_h"] for r in rec if not np.isnan(r["d_h"])]
    print("  δh 分布：中位 %.3f ｜ 75%% %.3f ｜ 90%% %.3f" % (
        np.median(dhs), np.percentile(dhs, 75), np.percentile(dhs, 90)))
    bands = [("形近 |δh|<0.15", lambda v: v < 0.15),
             ("中 0.15–0.35", lambda v: 0.15 <= v < 0.35),
             ("形差 0.35–0.8", lambda v: 0.35 <= v < 0.8),
             ("极端 ≥0.8", lambda v: v >= 0.8)]
    print("  %-16s %4s %-15s %-15s %-9s %-9s"
          % ("档", "n", "S0 判对@τ", "S1 判对@τ", "池覆盖", "Δ判对"))
    for name, fn in bands:
        sub = [r for r in rec if not np.isnan(r["d_h"]) and fn(abs(r["d_h"]))]
        if not sub:
            continue
        n = len(sub)
        a0 = sum(1 for r in sub if top1(r["s0"]) == r["q"]["pid"] and max(r["s0"].values()) >= mt)
        a1 = sum(1 for r in sub if top1(r["s1"]) == r["q"]["pid"] and max(r["s1"].values()) >= mt)
        print("  %-16s %4d %5.1f%%(%3d/%3d) %5.1f%%(%3d/%3d) %7.0f%% %+9d"
              % (name, n, 100.0 * a0 / n, a0, n, 100.0 * a1 / n, a1, n,
                 100 * np.mean([r["n_pool"] > 0 for r in sub]), a1 - a0))

    # 交叉：形态差 × 库失效
    print("\n===== E2b 交叉表（|δh| 档 × 历史库本人余弦档；recall@τ 的增量）=====")
    lo = args.low_thr
    print("  %-16s %-24s %-24s %-24s" % ("|δh|\\库cos", "<%.2f" % lo,
                                          "%.2f–%.2f" % (lo, mt), "≥%.2f" % mt))
    for name, fn in bands:
        cells = []
        for bname, bfn in [("低", lambda v: v < lo), ("中", lambda v: lo <= v < mt),
                           ("高", lambda v: v >= mt)]:
            sub = [r for r in rec if not np.isnan(r["d_h"]) and fn(abs(r["d_h"]))
                   and 0 <= r["s0"].get(r["q"]["pid"], -1.0) and bfn(r["s0"][r["q"]["pid"]])]
            if not sub:
                cells.append("%-24s" % "n=0")
                continue
            n = len(sub)
            a0 = sum(1 for r in sub if top1(r["s0"]) == r["q"]["pid"] and max(r["s0"].values()) >= mt)
            a1 = sum(1 for r in sub if top1(r["s1"]) == r["q"]["pid"] and max(r["s1"].values()) >= mt)
            cells.append("%-24s" % ("n=%d  %d→%d (%+d)" % (n, a0, a1, a1 - a0)))
        print("  %-16s %s" % (name, "".join(cells)))

    print("\n===== E2c 按历史库本人余弦分档（recall@τ，分母=档内全部，剔除无上午库的轨）=====")
    for name, fn in [("低 <%.2f" % lo, lambda v: v < lo),
                     ("中 %.2f–%.2f" % (lo, mt), lambda v: lo <= v < mt),
                     ("高 ≥%.2f" % mt, lambda v: v >= mt)]:
        sub = [r for r in rec if 0 <= r["s0"].get(r["q"]["pid"], -1.0)
               and fn(r["s0"][r["q"]["pid"]])]
        if not sub:
            continue
        n = len(sub)
        a0 = sum(1 for r in sub if top1(r["s0"]) == r["q"]["pid"] and max(r["s0"].values()) >= mt)
        a1 = sum(1 for r in sub if top1(r["s1"]) == r["q"]["pid"] and max(r["s1"].values()) >= mt)
        print("  %-14s n=%3d  S0 %3d (%.1f%%)  →  S1 %3d (%.1f%%)   池覆盖 %.0f%%"
              % (name, n, a0, 100.0 * a0 / n, a1, 100.0 * a1 / n,
                 100 * np.mean([r["n_pool"] > 0 for r in sub])))
    noLib = [r for r in rec if r["s0"].get(r["q"]["pid"], -1.0) < 0]
    if noLib:
        print("  另有 %d 条查询的本人上午没在该机位出现（无库，不可评）" % len(noLib))

    print("\n===== E6 池里被用上的那条，形态比库更接近查询吗？（池命中且库失效的查询）=====")
    sub6 = [r for r in rec if r["n_pool"] > 0 and not np.isnan(r["pool_best_dh"])
            and 0 <= r["s0"].get(r["q"]["pid"], -1.0) and r["s0"][r["q"]["pid"]] < mt]
    if sub6:
        ok = [r for r in sub6 if not np.isnan(r["d_h"]) and not np.isnan(r["pool_best_dh"])]
        closer = sum(1 for r in ok if abs(r["pool_best_dh"]) < abs(r["d_h"]))
        print("  n=%d（有效 %d）｜ 库 |δh| 中位 %.2f ｜ 池内最像样本 |δh| 中位 %.2f ｜ 池更接近 %d/%d = %.0f%%"
              % (len(sub6), len(ok), np.median([abs(r["d_h"]) for r in ok]) if ok else -1,
                 np.median([abs(r["pool_best_dh"]) for r in ok]) if ok else -1,
                 closer, len(ok), 100.0 * closer / max(len(ok), 1)))
        print("  （如果是 100%，说明池基本只在「把和查询同尺度的样本捡回来」，不是在比外观）")

    # ---------- E3 形态条件化模板（公平版：每个候选身份都取形态最像自己的一条） ----------
    print("\n===== E3 形态条件化模板（不依赖人脸，纯建库侧）=====")
    print("  S4 = 每个候选身份都取「框高最接近查询」的那条轨当模板（同场公平竞争）")
    print("  S5 = 按 |log(框高比)| 做核加权均值（σ 越小越只看形近样本）")

    def s_shape_pick(q, cands):
        out = {}
        for p in cands:
            rs = [x for x in lib.get((q["cam"], p), []) if x["h"] > 0]
            if not rs or q["h"] <= 0:
                out[p] = -1.0
                continue
            j = int(np.argmin([abs(np.log(x["h"] / q["h"])) for x in rs]))
            out[p] = float(E[q["i"]] @ E[rs[j]["i"]])
        return out

    def s_shape_w(q, cands, sig):
        out = {}
        for p in cands:
            rs = [x for x in lib.get((q["cam"], p), []) if x["h"] > 0]
            if not rs or q["h"] <= 0:
                out[p] = -1.0
                continue
            dd = np.array([abs(np.log(x["h"] / q["h"])) for x in rs])
            w = np.exp(-dd / sig)
            w = w / max(w.sum(), 1e-9)
            v = (E[[x["i"] for x in rs]] * w[:, None]).sum(0)
            v = v / max(np.linalg.norm(v), 1e-9)
            out[p] = float(v @ E[q["i"]])
        return out

    sigs = [0.3, 0.6, 1.2, 2.5]
    sub3 = [r for r in rec if 0 <= r["s0"].get(r["q"]["pid"], -1.0)]
    c4, c5 = {}, {}
    for r in sub3:
        q = r["q"]
        cands = [p for (c, p) in libv if c == q["cam"]]
        c4[q["k"]] = s_shape_pick(q, cands)
        for s in sigs:
            c5[(q["k"], s)] = s_shape_w(q, cands, s)
    print("  可评 %d 条（本人上午在该机位有库）" % len(sub3))
    print("  %-5s %-9s %-9s %-30s %-8s" % ("τ", "S0 库均值", "S4 挑一条",
                                            "S5 形态加权 σ=" + "/".join("%.1f" % s for s in sigs),
                                            "S1 池"))
    for tau in (0.70, mt, 0.85):
        def rec_of(get):
            a = 0
            for r in sub3:
                sc = get(r)
                if sc and max(sc.values()) >= tau and top1(sc) == r["q"]["pid"]:
                    a += 1
            return 100.0 * a / len(sub3)
        print("  %-5.2f %5.1f%%    %5.1f%%    %-30s %5.1f%%" % (
            tau, rec_of(lambda r: r["s0"]), rec_of(lambda r: c4[r["q"]["k"]]),
            "  ".join("%5.1f%%" % rec_of(lambda r, s=s: c5[(r["q"]["k"], s)]) for s in sigs),
            rec_of(lambda r: r["s1"])))

    print("\n===== E11 上限参考：同身份「当天全部其他轨」按形态加权（★非因果，仅作上限）=====")
    print("  含义：如果该身份的「多姿态素材」足够多（这正是训练素材侧要攒的东西），能到多少")
    for sig in (0.3, 0.6, 1.2):
        a = 0
        for r in sub3:
            q = r["q"]
            same = [x for x in rows if x["pid"] == q["pid"] and x["cam"] == q["cam"]
                    and x["k"] != q["k"] and x["h"] > 0]
            if not same or q["h"] <= 0:
                continue
            dd = np.array([abs(np.log(x["h"] / q["h"])) for x in same])
            w = np.exp(-dd / sig)
            w = w / max(w.sum(), 1e-9)
            v = (E[[x["i"] for x in same]] * w[:, None]).sum(0)
            v = v / max(np.linalg.norm(v), 1e-9)
            cands = [p for (c, p) in libv if c == q["cam"]]
            sc = {p: float(E[q["i"]] @ libv[(q["cam"], p)]) for p in cands}
            sc[q["pid"]] = max(sc.get(q["pid"], -1.0), float(v @ E[q["i"]]))
            if max(sc.values()) >= mt and top1(sc) == q["pid"]:
                a += 1
        print("  σ=%.1f：recall@%.2f = %.1f%%（分母 %d，样本数中位 %d）"
              % (sig, mt, 100.0 * a / len(sub3), len(sub3),
                 np.median([sum(1 for x in rows if x["pid"] == r["q"]["pid"]
                                and x["cam"] == r["q"]["cam"]) for r in sub3])))

    # ---------- E4 多原型库 ----------    print("\n===== E4 多原型库（建库侧，不用池；上午每身份 K 个原型，查询取 max）=====")
    print("  %-4s %-24s %-24s" % ("K", "P_h 尺度分位原型 cov/same",
                                  "P_k 特征k-means cov/same"))
    for K in Ks:
        line = []
        for table, nm in ((proto_h, "h"), (proto_k, "k")):
            ans = []
            for r in rec:
                q = r["q"]
                sc = {p: proto_score(q, table, K, p)
                      for (sc0, p) in libv if sc0 == q["cam"]}
                if not sc:
                    continue
                if max(sc.values()) >= mt:
                    ans.append(top1(sc) == q["pid"])
            line.append("%5.1f%% /%5.1f%% (n=%d)" % (
                100 * len(ans) / len(rec), 100 * np.mean(ans) if ans else 0.0, len(ans)))
        print("  %-4d %-24s %-24s" % (K, line[0], line[1]))

    # 多原型能否替代池：S1' = max(库多原型, 池)
    print("\n  -- 组合：库多原型 ∪ 池（K=2/3）--")
    for K in [k for k in Ks if k > 1]:
        for table, nm in ((proto_h, "P_h"), (proto_k, "P_k")):
            a0 = a1 = 0
            for r in rec:
                q = r["q"]
                sc = {p: proto_score(q, table, K, p) for (c0, p) in libv if c0 == q["cam"]}
                if not sc:
                    continue
                if max(r["s1"].values()) >= mt and top1(r["s1"]) == q["pid"]:
                    a0 += 1
                sc2 = {p: max(sc[p], r["s1"][p]) for p in sc}
                if max(sc2.values()) >= mt and top1(sc2) == q["pid"]:
                    a1 += 1
            print("    %s K=%d：S1 %d → S1∪原型 %d（%+.1f pt）"
                  % (nm, K, a0, a1, 100.0 * (a1 - a0) / len(rec)))

    # ---------- E5 写入质量门 ----------
    print("\n===== E5 写入质量门（绝对形态门；τ=%.2f）=====" % mt)
    print("  %-26s %-12s %-12s" % ("门", "S1 cov", "S1 same"))
    for gname, gfn in [("无门（现状）", lambda r: True),
                       ("框高 ≥120px", lambda r: r["h"] >= 120),
                       ("框高 ≥160px", lambda r: r["h"] >= 160),
                       ("宽高比 ≥0.35", lambda r: r["ar"] >= 0.35),
                       ("高≥120 且 宽高比≥0.35", lambda r: r["h"] >= 120 and r["ar"] >= 0.35)]:
        pool2 = collections.defaultdict(list)
        ans = []
        for q in queries:
            cands = [p for (sc, p) in libv if sc == q["cam"]]
            if not cands:
                continue
            libsc = {p: float(E[q["i"]] @ libv[(q["cam"], p)]) for p in cands}
            poolsc = {p: topk_mean(E, q["i"], pool2[(q["cam"], p)], args.pool_k)[0] for p in cands}
            s1 = {p: max(libsc[p], poolsc[p]) for p in cands}
            if max(s1.values()) >= mt:
                ans.append(top1(s1) == q["pid"])
            if q["face"] and q["pid"] in cands and gfn(q):
                lst = pool2[(q["cam"], q["pid"])]
                lst.append(q["i"])
                if len(lst) > args.pool_max:
                    del lst[0]
        print("  %-26s %5.0f%%        %5.1f%%"
              % (gname, 100.0 * len(ans) / len(rec), 100 * np.mean(ans) if ans else 0.0))

    print("\n===== E8 池内多模板取 top-k（k=1 只认最像的 1 条）=====")
    for pk in (1, 2, 3, 5):
        pool3 = collections.defaultdict(list)
        ans = []
        for q in queries:
            cands = [p for (sc, p) in libv if sc == q["cam"]]
            if not cands:
                continue
            libsc = {p: float(E[q["i"]] @ libv[(q["cam"], p)]) for p in cands}
            poolsc = {p: topk_mean(E, q["i"], pool3[(q["cam"], p)], pk)[0] for p in cands}
            s1 = {p: max(libsc[p], poolsc[p]) for p in cands}
            if max(s1.values()) >= mt:
                ans.append(top1(s1) == q["pid"])
            if q["face"] and q["pid"] in cands:
                lst = pool3[(q["cam"], q["pid"])]
                lst.append(q["i"])
                if len(lst) > args.pool_max:
                    del lst[0]
        print("  top-%d：覆盖 %2.0f%% ｜ 判对 %.1f%%"
              % (pk, 100.0 * len(ans) / len(rec), 100 * np.mean(ans) if ans else 0.0))

    print("\n===== E10 池内按形态门检索（写入仍不加门；打分时只取与查询形态可比的池样本）=====")
    for gate in (0.3, 0.6, 1.0, 9.9):
        pool4 = collections.defaultdict(list)
        ans = []
        for q in queries:
            cands = [p for (sc, p) in libv if sc == q["cam"]]
            if not cands:
                continue
            libsc = {p: float(E[q["i"]] @ libv[(q["cam"], p)]) for p in cands}
            poolsc = {}
            for p in cands:
                if gate > 9:
                    lst = list(pool4[(q["cam"], p)])
                else:
                    lst = []
                    for ix in pool4[(q["cam"], p)]:
                        hk = shape.get(ikey.get(ix, ""), (-1.0, -1.0))[0]
                        if hk > 0 and q["h"] > 0 and abs(np.log(hk / q["h"])) <= gate:
                            lst.append(ix)
                poolsc[p] = topk_mean(E, q["i"], lst, args.pool_k)[0]
            s1 = {p: max(libsc[p], poolsc[p]) for p in cands}
            if max(s1.values()) >= mt:
                ans.append(top1(s1) == q["pid"])
            if q["face"] and q["pid"] in cands:
                lst = pool4[(q["cam"], q["pid"])]
                lst.append(q["i"])
                if len(lst) > args.pool_max:
                    del lst[0]
        print("  |δh|≤%-4s：覆盖 %2.0f%% ｜ 判对 %.1f%%"
              % ("∞" if gate > 9 else "%.1f" % gate,
                 100.0 * len(ans) / len(rec), 100 * np.mean(ans) if ans else 0.0))

    # ---------- 明细 dump ----------
    if args.dump:
        out = []
        for r in rec:
            q = r["q"]
            out.append(dict(k=q["k"], pid=q["pid"], cam=q["cam"], t=q["t"], h=r["h"],
                            ar=r["ar"], h_lib=r["h_lib"], ar_lib=r["ar_lib"],
                            d_h=None if np.isnan(r["d_h"]) else float(r["d_h"]),
                            d_ar=None if np.isnan(r["d_ar"]) else float(r["d_ar"]),
                            own0=float(r["s0"].get(q["pid"], -1.0)),
                            s0_top=top1(r["s0"]), s1_top=top1(r["s1"]),
                            s0_ok=int(top1(r["s0"]) == q["pid"]),
                            s1_ok=int(top1(r["s1"]) == q["pid"]),
                            s4=None if r["s4"] is None else float(r["s4"]),
                            n_pool=r["n_pool"], face=int(q["face"])))
        json.dump(out, open(args.dump, "w"), ensure_ascii=False, indent=1)
        print("\n明细已写 %s（%d 条）" % (args.dump, len(out)))


if __name__ == "__main__":
    main()
