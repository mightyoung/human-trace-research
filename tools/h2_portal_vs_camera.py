#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2 对照实验：门户级转移建模 vs 纯相机级图（冻结集独立评测）。

研究假设（research-charter H2，限定版）：
  在无人工身份标签、开放集、强反光/不重叠条件下，**门户级转移建模**相比纯相机级图
  能**减少达到固定误接约束所需的有效通行事件数**（Charter 措辞）/ 在固定误接预算下
  保有更多有效接续事件（本实现的度量口径，见下）。

评测集（只读，禁止回流调参）：
  output_crosscam/analysis/relay_v2/frozen_eval_v1.json（sha1_16 f02de2063ffda3a0）
  n=109（same 32 / diff 40 / unclear 37）；unclear 不参与判别指标，仅作分布诊断。

三个转移模型（除转移粒度外完全同构，拟合集、eps、λ 全部相同）：
  A  cam  纯相机级：w = cnt(ca->cb)/tot(ca)（相机对粒度的转移频次）
  B1 c0  门户级（C0 几何门）：w = cnt((ca,ge)->(cb,gn))/tot(ca,ge)，ge/gn=最近出口/入口门（R=0.25）
  B2 v2  门户级（WP2 数据驱动门户 40 簇）：同上，zone 换为 v2 簇（命中半径 radius+eps/2）
  注：none 桶（未命中任何门/簇）为合法 zone 取值。

评测模式：
  E1 纯转移（主）：s = log(w + eps)          —— 隔离转移模型本身的判别力；全 109 对可用
  E2 系统口径：   s = base + λ_T·log(w+eps)  —— base = 候选池 score(=sim+0.01·log w_dt)，
                 同一 base 只换转移项；池外 22 对无 base → E2 子集 n=87

预注册（实验前声明，先于看数）：
  - 主指标：E1 的 TP@FP<=2（误接预算 2/40=5%；fp<=b 下可接受的最大真接续数）
  - 次指标：TP@FP ∈ {0,1,3,5}、AUC、E2 同表、冷启动曲线（TP@FP<=2 vs 拟合样本量 N）
  - 冷启动效能：N@50%（达到各模型自身终值 50% 所需的事件数，越小越好，横比用同一终值水平）
  - 固定超参（不选参，敏感性仅展示）：R=0.25（eval4 L2 前例）；v2 命中半径=radius+0.03
    （WP2 作图口径 eps/2）；eps=1/(N_fit+2)（relay_portal_prior 同款）；λ_T=0.01（家法）
  - 拟合纪律：T 计数表剔除**全部 109 判定行**（dist_v1 家法"fit 集剔除全部已判行"）；
    评估对不参与拟合 → 无自证
  - 判定（三档）：B2/B1 主指标点估计 > cam 且 AUC 不低于 cam 且冷启动不劣 → "方向支持"；
    仅部分方向一致 → "证据不足/弱"；主指标低于 cam → "不支持"。bootstrap CI 若覆盖负向
    则注明"样本量不足以排除负向"。

运行：
  python tools/h2_portal_vs_camera.py            # 全量 + bootstrap 2000
  python tools/h2_portal_vs_camera.py --boot 400 # 快速
产物：
  output_crosscam/analysis/relay_v2/h2_portal_vs_camera_v1.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

os.environ.setdefault("DET_VIEW", "clean2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from crosscam_relay import load_entities  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
FROZEN = os.path.join(ANA, "relay_v2", "frozen_eval_v1.json")
POOL = os.path.join(ANA, "relay_v2", "scored_v2.json")
C0 = os.path.join(ANA, "c0", "c0_baseline_v9.json")
V2 = os.path.join(ANA, "relay_v2", "portal_discovery_v1.json")
OUT = os.path.join(ANA, "relay_v2", "h2_portal_vs_camera_v1.json")

FROZEN_SHA = "f02de2063ffda3a0"
CAMS = ["305", "316", "318", "320", "323", "343"]
GATE_R = 0.25          # C0 门 zone 命中半径（eval4 L2 前例）
V2_MARGIN = 0.03       # v2 簇命中半径加宽（WP2 作图 eps/2）
LAM_T = 0.01           # 转移项权重（与 λ_dt 同尺度，家法）
BUDGETS = [0, 1, 2, 3, 5]
COLD_GRID = [64, 128, 256, 512, 768, 1024, 1536, 2048, 2531]


# ---------------- 工具 ----------------
def nearest_r(pt, centers, r):
    """列表 centers 中距离 <= r 的最近下标；否则 None。"""
    best, bi = None, None
    for i, g in enumerate(centers):
        d = ((pt[0] - g[0]) ** 2 + (pt[1] - g[1]) ** 2) ** 0.5
        if best is None or d < best:
            best, bi = d, i
    return bi if (best is not None and best <= r) else None


def nearest_v2(pt, clusters):
    """clusters: [(center, radius)]；逐簇命中判定（距离<=自身半径）取最近。"""
    best, bi = None, None
    for i, (cc, rr) in enumerate(clusters):
        d = ((pt[0] - cc[0]) ** 2 + (pt[1] - cc[1]) ** 2) ** 0.5
        if d <= rr and (best is None or d < best):
            best, bi = d, i
    return bi


def tp_at_fp(pos_scores, neg_scores, budgets):
    """离散阈值扫描：accept = s >= t；返回 {b: maxTP}, curve=[(fp,tp,t)]。"""
    allv = sorted(set(pos_scores) | set(neg_scores), reverse=True)
    curve = [(0, 0, None)]                      # accept-none 点
    for t in allv:
        tp = sum(1 for s in pos_scores if s >= t)
        fp = sum(1 for s in neg_scores if s >= t)
        curve.append((fp, tp, t))
    best = {}
    for b in budgets:
        cands = [tp for fp, tp, _ in curve if fp <= b]
        best[b] = max(cands) if cands else 0
    return best, curve


def auc_score(pos_scores, neg_scores):
    """Mann-Whitney AUC（含并列平均秩）。"""
    pos = np.asarray(pos_scores, float)
    neg = np.asarray(neg_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 并列平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i in np.where(cnt > 1)[0]:
        m = inv == i
        ranks[m] = ranks[m].mean()
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logw(w, eps):
    return float(np.log(w + eps))


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap 次数（0=跳过）")
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # ---- 载入 ----
    fz = json.load(open(FROZEN, encoding="utf-8"))
    assert fz["sha1_16"] == FROZEN_SHA, f"冻结集 sha 不符: {fz['sha1_16']}"
    pairs = fz["pairs"]
    judged_keys = {(p["a"], p["b"]) for p in pairs}

    pool = json.load(open(POOL, encoding="utf-8"))
    cands = pool["candidates"]
    pool_by_key = {(c["a"], c["b"]): c for c in cands}
    fit = [c for c in cands if (c["a"], c["b"]) not in judged_keys]

    c0 = json.load(open(C0, encoding="utf-8"))
    gates = {cam: dict(exit=[g["center"] for g in c0["gates"].get(cam, []) if g["kind"] == "exit"],
                       entry=[g["center"] for g in c0["gates"].get(cam, []) if g["kind"] == "entry"])
             for cam in CAMS}
    pd = json.load(open(V2, encoding="utf-8"))
    v2cl = {cam: dict(exit=[(c["center"], c["radius"] + V2_MARGIN) for c in pd["per_cam"][cam]["exit"]["clusters"]],
                      entry=[(c["center"], c["radius"] + V2_MARGIN) for c in pd["per_cam"][cam]["entry"]["clusters"]])
            for cam in CAMS}

    ents = {cam: load_entities(cam) for cam in CAMS}

    def endpoints(p):
        """(a_end, b_start)：池内用候选记录；池外实体回退（审计 87/87 一致，maxΔ=1e-4）。"""
        key = (p["a"], p["b"])
        if key in pool_by_key:
            c = pool_by_key[key]
            return (float(c["a_end"][0]), float(c["a_end"][1])), (float(c["b_start"][0]), float(c["b_start"][1]))
        ca, ea = p["a"].split(":")
        cb, eb = p["b"].split(":")
        e1 = ents[ca].get(ea)
        e2 = ents[cb].get(eb)
        if e1 is None or e2 is None:
            return None, None
        return (float(e1[1][-1]), float(e1[2][-1])), (float(e2[1][0]), float(e2[2][0]))

    def zones(ca, cb, ae, bs):
        ge1 = nearest_r(ae, gates[ca]["exit"], GATE_R)
        gn1 = nearest_r(bs, gates[cb]["entry"], GATE_R)
        ge2 = nearest_v2(ae, v2cl[ca]["exit"])
        gn2 = nearest_v2(bs, v2cl[cb]["entry"])
        return (ge1, gn1), (ge2, gn2)

    # 预计算评估对
    ev = []
    for p in pairs:
        ca, _ = p["a"].split(":")
        cb, _ = p["b"].split(":")
        ae, bs = endpoints(p)
        (ge1, gn1), (ge2, gn2) = zones(ca, cb, ae, bs)
        base = pool_by_key.get((p["a"], p["b"]), {}).get("score")
        ev.append(dict(a=p["a"], b=p["b"], label=p["label"], dir=f"{ca}->{cb}",
                       ca=ca, cb=cb, ge1=ge1, gn1=gn1, ge2=ge2, gn2=gn2,
                       a_end=ae, b_start=bs, base=base))

    # ---- 拟合（全量）----
    def fit_counts(rows):
        Tcam, Tcam_tot = Counter(), Counter()
        T1, T1t = Counter(), Counter()
        T2, T2t = Counter(), Counter()
        for c in rows:
            ca, cb = c["a"].split(":")[0], c["b"].split(":")[0]
            Tcam[(ca, cb)] += 1
            Tcam_tot[ca] += 1
            (ge1, gn1), (ge2, gn2) = zones(ca, cb, tuple(c["a_end"]), tuple(c["b_start"]))
            T1[(ca, ge1, cb, gn1)] += 1
            T1t[(ca, ge1)] += 1
            T2[(ca, ge2, cb, gn2)] += 1
            T2t[(ca, ge2)] += 1
        return dict(Tcam=Tcam, Tcam_tot=Tcam_tot, T1=T1, T1t=T1t, T2=T2, T2t=T2t)

    def score_ev(cnt, evs, mode):
        """返回逐对 {w_cam,w_b1,w_b2, s_cam,s_b1,s_b2}。"""
        N = sum(cnt["Tcam_tot"].values())
        eps = 1.0 / (N + 2.0)
        out = []
        for e in evs:
            w_cam = cnt["Tcam"].get((e["ca"], e["cb"]), 0) / max(cnt["Tcam_tot"].get(e["ca"], 0), 1)
            w1 = cnt["T1"].get((e["ca"], e["ge1"], e["cb"], e["gn1"]), 0) / max(cnt["T1t"].get((e["ca"], e["ge1"]), 0), 1)
            w2 = cnt["T2"].get((e["ca"], e["ge2"], e["cb"], e["gn2"]), 0) / max(cnt["T2t"].get((e["ca"], e["ge2"]), 0), 1)
            r = dict(e)
            r.update(w_cam=w_cam, w_b1=w1, w_b2=w2, eps=eps)
            if mode == "E1":
                r["s_cam"], r["s_b1"], r["s_b2"] = logw(w_cam, eps), logw(w1, eps), logw(w2, eps)
            else:  # E2
                b = e["base"] if e["base"] is not None else float("nan")
                r["s_cam"] = b + LAM_T * logw(w_cam, eps)
                r["s_b1"] = b + LAM_T * logw(w1, eps)
                r["s_b2"] = b + LAM_T * logw(w2, eps)
            out.append(r)
        return out, eps

    cnt_full = fit_counts(fit)
    scored, eps_full = score_ev(cnt_full, ev, "E1")
    scored_e2, _ = score_ev(cnt_full, ev, "E2")

    # ---- 指标 ----
    def metrics(rows, mode, tag):
        rows = [r for r in rows if r["label"] in ("same", "diff")]
        if mode == "E2":
            rows = [r for r in rows if r["base"] is not None]
        pos = [r for r in rows if r["label"] == "same"]
        neg = [r for r in rows if r["label"] == "diff"]
        res = {}
        for m in ("cam", "b1", "b2"):
            ps = [r[f"s_{m}"] for r in pos]
            ns = [r[f"s_{m}"] for r in neg]
            b, curve = tp_at_fp(ps, ns, BUDGETS)
            res[m] = dict(auc=round(auc_score(ps, ns), 4),
                          tp_at_fp={str(k): v for k, v in b.items()},
                          curve=[(int(fp), int(tp)) for fp, tp, _ in curve],
                          n_pos=len(pos), n_neg=len(neg))
        return res

    M_E1 = metrics(scored, "E1", "E1")
    M_E2 = metrics(scored_e2, "E2", "E2")

    # ---- bootstrap（按 a 端实体成组，pos/neg 分层）----
    def boot_ci(rows, mode, stat, B, seed):
        """成组 bootstrap（按 a 端实体分层）。stat: 单模型量（tp2_*/auc_*）或差值（d*）。"""
        rows = [r for r in rows if r["label"] in ("same", "diff")]
        if mode == "E2":
            rows = [r for r in rows if r["base"] is not None]
        pos_g, neg_g = defaultdict(list), defaultdict(list)
        for r in rows:
            (pos_g if r["label"] == "same" else neg_g)[r["a"]].append(r)
        pk, nk = list(pos_g), list(neg_g)
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(B):
            ps = [r for k in rng.choice(pk, len(pk), replace=True) for r in pos_g[k]]
            ns = [r for k in rng.choice(nk, len(nk), replace=True) for r in neg_g[k]]

            def tp2(m):
                p = [r[f"s_{m}"] for r in ps]
                n = [r[f"s_{m}"] for r in ns]
                b, _ = tp_at_fp(p, n, [2])
                return b[2]

            def auc(m):
                return auc_score([r[f"s_{m}"] for r in ps], [r[f"s_{m}"] for r in ns])

            if stat == "d2_tp2":
                vals.append(tp2("b2") - tp2("cam"))
            elif stat == "d1_tp2":
                vals.append(tp2("b1") - tp2("cam"))
            elif stat == "dAUC_B2_minus_cam":
                vals.append(auc("b2") - auc("cam"))
            elif stat == "dAUC_B1_minus_cam":
                vals.append(auc("b1") - auc("cam"))
            else:  # tp2_cam / tp2_b1 / tp2_b2 / auc_cam / auc_b1 / auc_b2
                kind, m = stat.split("_", 1)
                vals.append(tp2(m) if kind == "tp2" else auc(m))
        vals = np.array([v for v in vals if np.isfinite(v)])
        return [round(float(np.percentile(vals, 2.5)), 4),
                round(float(np.percentile(vals, 97.5)), 4),
                round(float(np.mean(vals)), 4)]

    boot = {}
    if args.boot > 0:
        stats = [("d2_tp2", "dTP2_B2_minus_cam"), ("d1_tp2", "dTP2_B1_minus_cam"),
                 ("tp2_cam", "TP2_cam"), ("tp2_b1", "TP2_b1"), ("tp2_b2", "TP2_b2"),
                 ("auc_cam", "AUC_cam"), ("auc_b1", "AUC_b1"), ("auc_b2", "AUC_b2"),
                 ("dAUC_B2_minus_cam", "dAUC_B2_minus_cam"), ("dAUC_B1_minus_cam", "dAUC_B1_minus_cam")]
        for mode, rows in (("E1", scored), ("E2", scored_e2)):
            for stat, name in stats:
                boot[f"{mode}_{name}"] = boot_ci(rows, mode, stat, args.boot, args.seed)
        print("[boot] done")

    # ---- 冷启动曲线（按 a_tlast 时序累积拟合）----
    fit_sorted = sorted(fit, key=lambda c: c["a_tlast"])
    cold = {m: [] for m in ("cam", "b1", "b2")}
    cold_e2 = {m: [] for m in ("cam", "b1", "b2")}
    for N in COLD_GRID:
        sub = fit_sorted[:N]
        cnt = fit_counts(sub)
        sc, _ = score_ev(cnt, ev, "E1")
        sc2, _ = score_ev(cnt, ev, "E2")
        rows1 = [r for r in sc if r["label"] in ("same", "diff")]
        rows2 = [r for r in sc2 if r["label"] in ("same", "diff") and r["base"] is not None]
        for m in ("cam", "b1", "b2"):
            p = [r[f"s_{m}"] for r in rows1 if r["label"] == "same"]
            n = [r[f"s_{m}"] for r in rows1 if r["label"] == "diff"]
            b, _ = tp_at_fp(p, n, [2, 5])
            cold[m].append(dict(N=N, t_last=round(sub[-1]["a_tlast"], 1),
                                tp2=b[2], tp5=b[5], auc=round(auc_score(p, n), 4)))
            p2 = [r[f"s_{m}"] for r in rows2 if r["label"] == "same"]
            n2 = [r[f"s_{m}"] for r in rows2 if r["label"] == "diff"]
            b2, _ = tp_at_fp(p2, n2, [2])
            cold_e2[m].append(dict(N=N, tp2=b2[2]))

    # 冷启动效能：达到各模型自身终值 50% / 25% / 首次 TP>=1 所需最小 N
    def n_at_frac(curve, frac=0.5):
        final = curve[-1]["tp2"]
        target = int(np.ceil(final * frac))
        for pt in curve:
            if pt["tp2"] >= target and target > 0:
                return pt["N"], target
        return None, target

    def n_first_tp1(curve):
        for pt in curve:
            if pt["tp2"] >= 1:
                return pt["N"]
        return None

    eff = {m: dict(N_at_50=n_at_frac(cold[m], 0.5), N_at_25=n_at_frac(cold[m], 0.25),
                   N_first_tp1=n_first_tp1(cold[m]))
           for m in ("cam", "b1", "b2")}

    # ---- 辅助诊断 ----
    diag = {}
    # cell 统计
    diag["cells"] = {
        "cam": len(cnt_full["Tcam"]),
        "b1": len(cnt_full["T1"]),
        "b2": len(cnt_full["T2"]),
    }
    # 逐 dir 明细（E1）
    bydir = defaultdict(lambda: dict(pos=[], neg=[]))
    for r in scored:
        if r["label"] in ("same", "diff"):
            bydir[r["dir"]]["pos" if r["label"] == "same" else "neg"].append(r)
    diag["per_dir"] = {d: dict(n_pos=len(v["pos"]), n_neg=len(v["neg"]),
                             w_cam_pos=round(float(np.mean([r["w_cam"] for r in v["pos"]])) if v["pos"] else 0, 4),
                             w_cam_neg=round(float(np.mean([r["w_cam"] for r in v["neg"]])) if v["neg"] else 0, 4),
                             w_b1_pos=round(float(np.mean([r["w_b1"] for r in v["pos"]])) if v["pos"] else 0, 4),
                             w_b1_neg=round(float(np.mean([r["w_b1"] for r in v["neg"]])) if v["neg"] else 0, 4),
                             w_b2_pos=round(float(np.mean([r["w_b2"] for r in v["pos"]])) if v["pos"] else 0, 4),
                             w_b2_neg=round(float(np.mean([r["w_b2"] for r in v["neg"]])) if v["neg"] else 0, 4))
                       for d, v in sorted(bydir.items())}
    # unclear 分布
    und = [r for r in scored if r["label"] == "unclear"]
    diag["unclear"] = dict(n=len(und),
                           w_cam_mean=round(float(np.mean([r["w_cam"] for r in und])), 4) if und else None,
                           w_b1_mean=round(float(np.mean([r["w_b1"] for r in und])), 4) if und else None,
                           w_b2_mean=round(float(np.mean([r["w_b2"] for r in und])), 4) if und else None)

    # 敏感性（展示用不选参）：门半径 / v2 边距 / λ_T
    sens = {}
    for r_gate in (0.15, 0.25, 0.35):
        ev2 = []
        for p in pairs:
            ca, _ = p["a"].split(":")
            cb, _ = p["b"].split(":")
            ae, bs = endpoints(p)
            ge1 = nearest_r(ae, gates[ca]["exit"], r_gate)
            gn1 = nearest_r(bs, gates[cb]["entry"], r_gate)
            ev2.append(dict(a=p["a"], b=p["b"], label=p["label"], ca=ca, cb=cb,
                            ge1=ge1, gn1=gn1))
        # 重拟合 b1（用同一 fit，按新半径）
        T1, T1t = Counter(), Counter()
        for c in fit:
            ca, cb = c["a"].split(":")[0], c["b"].split(":")[0]
            g1 = nearest_r(tuple(c["a_end"]), gates[ca]["exit"], r_gate)
            g2 = nearest_r(tuple(c["b_start"]), gates[cb]["entry"], r_gate)
            T1[(ca, g1, cb, g2)] += 1
            T1t[(ca, g1)] += 1
        sc = []
        for e in ev2:
            w = T1.get((e["ca"], e["ge1"], e["cb"], e["gn1"]), 0) / max(T1t.get((e["ca"], e["ge1"]), 0), 1)
            sc.append(dict(e, s=logw(w, eps_full)))
        p_ = [r["s"] for r in sc if r["label"] == "same"]
        n_ = [r["s"] for r in sc if r["label"] == "diff"]
        b, _ = tp_at_fp(p_, n_, BUDGETS)
        sens[f"gate_R_{r_gate}"] = dict(tp_at_fp={str(k): v for k, v in b.items()},
                                        auc=round(auc_score(p_, n_), 4))
    for mg in (0.0, 0.03, 0.06):
        T2, T2t = Counter(), Counter()
        for c in fit:
            ca, cb = c["a"].split(":")[0], c["b"].split(":")[0]
            g1 = nearest_v2(tuple(c["a_end"]), [(cc, rr - V2_MARGIN + mg) for cc, rr in v2cl[ca]["exit"]])
            g2 = nearest_v2(tuple(c["b_start"]), [(cc, rr - V2_MARGIN + mg) for cc, rr in v2cl[cb]["entry"]])
            T2[(ca, g1, cb, g2)] += 1
            T2t[(ca, g1)] += 1
        sc = []
        for p in pairs:
            ca, _ = p["a"].split(":")
            cb, _ = p["b"].split(":")
            ae, bs = endpoints(p)
            ge2 = nearest_v2(ae, [(cc, rr - V2_MARGIN + mg) for cc, rr in v2cl[ca]["exit"]])
            gn2 = nearest_v2(bs, [(cc, rr - V2_MARGIN + mg) for cc, rr in v2cl[cb]["entry"]])
            w = T2.get((ca, ge2, cb, gn2), 0) / max(T2t.get((ca, ge2), 0), 1)
            sc.append(dict(label=p["label"], s=logw(w, eps_full)))
        p_ = [r["s"] for r in sc if r["label"] == "same"]
        n_ = [r["s"] for r in sc if r["label"] == "diff"]
        b, _ = tp_at_fp(p_, n_, BUDGETS)
        sens[f"v2_margin_{mg}"] = dict(tp_at_fp={str(k): v for k, v in b.items()},
                                       auc=round(auc_score(p_, n_), 4))
    for lam in (0.0, 0.005, 0.01, 0.02):
        rows = [r for r in scored if r["base"] is not None and r["label"] in ("same", "diff")]
        for m in ("cam", "b1", "b2"):
            for r in rows:
                r[f"_s"] = r["base"] + lam * logw(r[f"w_{m}"], eps_full)
            p_ = [r["_s"] for r in rows if r["label"] == "same"]
            n_ = [r["_s"] for r in rows if r["label"] == "diff"]
            b, _ = tp_at_fp(p_, n_, [2])
            sens.setdefault(f"lam_{lam}", {})[m] = dict(tp2=b[2], auc=round(auc_score(p_, n_), 4))

    # 补充（探索性，不作判定）：同源(a端)组内 pos>neg 排序正确率——系统实际消费口径
    def _score_of(r, m, use_base):
        if not use_base:
            return r[f"w_{m}"]
        if r["base"] is None:
            return None
        return r["base"] + LAM_T * logw(r[f"w_{m}"], r.get("eps", eps_full))

    def within_a(rows, use_base, a_prefix=None):
        g = defaultdict(list)
        for r in rows:
            if r["label"] in ("same", "diff") and (a_prefix is None or r["a"].startswith(a_prefix)):
                g[r["a"]].append(r)
        out = {}
        for m in ("cam", "b1", "b2"):
            tot, corr, ng = 0, 0.0, 0
            for a, rs in g.items():
                pos = [r for r in rs if r["label"] == "same"]
                neg = [r for r in rs if r["label"] == "diff"]
                if not pos or not neg:
                    continue
                ng += 1
                for p in pos:
                    for n in neg:
                        sp, sn = _score_of(p, m, use_base), _score_of(n, m, use_base)
                        if sp is None or sn is None:
                            continue
                        tot += 1
                        corr += 0.5 if abs(sp - sn) < 1e-12 else float(sp > sn)
            out[m] = dict(groups=ng, comparisons=tot, acc=round(corr / max(tot, 1), 4))
        return out

    supp = dict(E1=within_a(scored, False), E2=within_a(scored_e2, True),
                E1_343=within_a(scored, False, "343:"), E2_343=within_a(scored_e2, True, "343:"),
                note="探索性补充（未预注册，不作判定）：同源组内 pos 应排在 neg 之前；"
                     "E1 用纯转移 w，E2 用系统分。n 小（比较数 ~20），仅作机制参考。")

    # ---- 输出 ----
    res = dict(
        version="h2/portal_vs_camera/1",
        frozen=dict(path=os.path.relpath(FROZEN, ROOT).replace("\\", "/"), sha1_16=fz["sha1_16"],
                    n=fz["n"], counts=fz["counts"]),
        prereg=dict(primary="E1 TP@FP<=2", secondary=["TP@FP 0/1/3/5", "AUC", "E2", "cold-start"],
                    fixed=dict(GATE_R=GATE_R, V2_MARGIN=V2_MARGIN, LAM_T=LAM_T,
                               eps_rule="1/(N_fit+2)"),
                    discipline="fit 剔除全部 109 判定行；评估对零回流"),
        fit=dict(n=len(fit), excluded_judged=len(cands)-len(fit), N_total=len(cands)),
        models=dict(
            cam="相机级 w=cnt(ca->cb)/tot(ca)",
            b1="门户级(C0门 R=0.25) w=cnt((ca,ge)->(cb,gn))/tot(ca,ge)",
            b2="门户级(WP2 v2 簇 radius+0.03) 同上"),
        metrics=dict(E1=M_E1, E2=M_E2),
        bootstrap=boot,
        cold_start=dict(E1=cold, E2=cold_e2, efficiency=eff),
        supplement_within_a=supp,
        diag=diag,
        sensitivity=sens,
        pairs=[dict(a=r["a"], b=r["b"], label=r["label"], dir=r["dir"],
                    w_cam=round(r["w_cam"], 6), w_b1=round(r["w_b1"], 6), w_b2=round(r["w_b2"], 6),
                    s_cam=round(r["s_cam"], 5), s_b1=round(r["s_b1"], 5), s_b2=round(r["s_b2"], 5),
                    base=(None if r["base"] is None else round(r["base"], 5)),
                    zone_b1=[r["ge1"], r["gn1"]], zone_b2=[r["ge2"], r["gn2"]]) for r in scored],
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---- 控制台速览 ----
    print("=== H2 主表 · E1 纯转移（pos=32, neg=40）===")
    print(f"{'model':<5} {'AUC':>7} " + " ".join(f"TP@FP<={b}" for b in BUDGETS))
    for m in ("cam", "b1", "b2"):
        r = M_E1[m]
        print(f"{m:<5} {r['auc']:>7} " + " ".join(f"{r['tp_at_fp'][str(b)]:>8}" for b in BUDGETS))
    print("=== H2 主表 · E2 系统（pos/neg=池内子集）===")
    for m in ("cam", "b1", "b2"):
        r = M_E2[m]
        print(f"{m:<5} {r['auc']:>7} " + " ".join(f"{r['tp_at_fp'][str(b)]:>8}" for b in BUDGETS)
              + f"   (n_pos={r['n_pos']}, n_neg={r['n_neg']})")
    if boot:
        print("=== bootstrap 95% CI（成组，a 端实体；[lo, hi, mean]）===")
        for k in ("E1_dTP2_B2_minus_cam", "E1_dTP2_B1_minus_cam",
                  "E1_AUC_cam", "E1_AUC_b1", "E1_AUC_b2",
                  "E1_dAUC_B2_minus_cam", "E1_dAUC_B1_minus_cam",
                  "E2_dAUC_B2_minus_cam", "E2_dAUC_B1_minus_cam",
                  "E2_dTP2_B2_minus_cam", "E2_dTP2_B1_minus_cam"):
            if k in boot:
                print(f"  {k}: {boot[k]}")
    print("=== 敏感性（展示用）===")
    for k in ("gate_R_0.15", "gate_R_0.25", "gate_R_0.35", "v2_margin_0.0", "v2_margin_0.03", "v2_margin_0.06"):
        if k in sens:
            print(f"  {k}: auc={sens[k]['auc']} tp@fp={sens[k]['tp_at_fp']}")
    print("=== 冷启动（TP@FP<=2 vs N，E1）===")
    print(f"{'N':>5} " + " ".join(f"{m:>4}" for m in ("cam", "b1", "b2")))
    for i, N in enumerate(COLD_GRID):
        print(f"{N:>5} " + " ".join(f"{cold[m][i]['tp2']:>4}" for m in ("cam", "b1", "b2"))
              + f"   (t={cold['cam'][i]['t_last']}s)")
    print("效率 N@50%:", {m: eff[m]["N_at_50"] for m in ("cam", "b1", "b2")})
    print("补充（探索性）同源组内排序:", json.dumps(supp, ensure_ascii=False))
    print("cell 数:", diag["cells"])
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
