#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S7 倒影门接入选择器的同序列冻结 A/B（H1 门控族的第 4 重门）。

背景：S7 训练了倒影判别器（收缩 LDA 方向，留一机位 AUC≈0.758）→ obs.flags.reflection_score。
本实验回答：把该分数接进 IdentitySelector 本体（predict(refl=…)）后，对**同一序列**的身份
分配产生什么影响。

设计（同序列冻结：除倒影门以外，选择器/数据/度量全部一致）：
  臂 A  off    现状（selector_v2 原样，回归自检须与旧版逐位一致）
  臂 B  soft   只作用于"接受判定"、不改簇归属：eff = best − α·max(0, refl − base)
  臂 C  hard   refl ≥ τ 直接 unknown
  操作点（α / τ）一律在 **val 段**按「覆盖损失 ≤ 5pp 预算」校准（禁 test-oracle，项目铁律）。

度量（沿用 EVAL-3 口径，标签无关、跨臂可比）：
  ① 覆盖率 = 1 − unknown；② 分片一致 ab_same；③ 跨段保持 ret_same；④ 误接代理 viol
  另报"公共子集"（各臂皆已分配的轨）上的 viol，实现同覆盖约束比较。

诚实边界：1012 无倒影标签 → 本实验只给**权衡曲线**（覆盖 vs 误接），不给"净收益"；
收益下界由判别器在 GIL 上的留一机位 AUC 佐证。只读，不写生产文件。
输出 output_crosscam/analysis/eval3/reflection_gate_ab.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector import IdentitySelector  # noqa: E402
from eval3_gated_selftrain import (load_emb, load_intervals, overlap_pairs,  # noqa: E402
                                   SEGS, SEL, OUT)
from build_selector_v2 import load_reflection_probe, reflection_score  # noqa: E402

TAUS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
ALPHAS = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
BUDGET = 0.05          # val 段允许的覆盖损失预算（5pp）


def build_data():
    data = {}
    for split, segs in SEGS.items():
        keys, cams, vas, vbs, vs, t = [], [], [], [], [], []
        for seg in segs:
            ks, va, vb, v = load_emb(seg)
            iv = load_intervals(seg)
            cam = seg[:3]
            for k, a, b, vm in zip(ks, va, vb, v):
                tid = int(k.split("#")[-1])
                if tid not in iv:
                    continue
                keys.append(k); cams.append(cam)
                vas.append(a); vbs.append(b); vs.append(vm); t.append(iv[tid])
        data[split] = dict(keys=keys, cams=cams, V=np.asarray(vs, np.float32),
                           Va=np.asarray(vas, np.float32), Vb=np.asarray(vbs, np.float32), t=t)
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = IdentitySelector.load(SEL)
    probe = load_reflection_probe()
    if probe is None:
        print("[abort] 找不到倒影判别器 reflection_cluster_v1.json")
        return
    print(f"[selector] protos={sel.protos.shape[0]} clusters={sel.n_clusters}")
    print(f"[probe] {probe['source']} loo_auc_mean={probe['loo_auc_mean']:.4f}")

    data = build_data()
    for sp, d in data.items():
        d["refl"] = reflection_score(d["V"], probe)
        d["refl_a"] = reflection_score(d["Va"], probe)
        d["refl_b"] = reflection_score(d["Vb"], probe)
        print(f"[{sp}] tracks={len(d['keys'])} | refl mean={d['refl'].mean():.3f} "
              f"p50={np.percentile(d['refl'], 50):.3f} p95={np.percentile(d['refl'], 95):.3f}")

    base_cal = float(np.median(data["val"]["refl"]))   # 软门基线：val 中位
    print(f"[calib] 软门 base=val中位 {base_cal:.4f}")

    # ---- 回归自检：refl=None 与 refl=全零(soft) 必须逐位一致（默认行为不变）----
    z = np.zeros(len(data["test"]["V"]), np.float32)
    ids_none, _ = sel.predict(data["test"]["V"], cams=data["test"]["cams"])
    ids_zero, _ = sel.predict(data["test"]["V"], cams=data["test"]["cams"], refl=z,
                              refl_mode="soft", refl_base=base_cal, refl_alpha=1.0)
    same = bool((np.asarray(ids_none) == np.asarray(ids_zero)).all())
    print(f"[regress] refl=None vs refl=0(soft) 逐位一致: {'OK' if same else 'FAIL'}")

    # ---- 操作点校准：val 段，覆盖损失 ≤ BUDGET；取预算内最激进者 ----
    def val_unknown(alpha=None, tau=None):
        if tau is not None:
            ids, _ = sel.predict(data["val"]["V"], cams=data["val"]["cams"],
                                 refl=data["val"]["refl"], refl_tau=tau, refl_mode="hard")
        else:
            ids, _ = sel.predict(data["val"]["V"], cams=data["val"]["cams"],
                                 refl=data["val"]["refl"], refl_mode="soft",
                                 refl_base=base_cal, refl_alpha=alpha)
        return float((np.asarray(ids) == -1).mean())

    unk_v_off = val_unknown(alpha=0.0)
    alpha_used = ALPHAS[0]
    for a in ALPHAS:
        if val_unknown(alpha=a) - unk_v_off <= BUDGET:
            alpha_used = a
        else:
            break
    tau_used = None
    for t in TAUS:
        if val_unknown(tau=t) - unk_v_off <= BUDGET:
            tau_used = t
            break
    if tau_used is None:
        tau_used = TAUS[-1]
    print(f"[calib] val off unknown={unk_v_off:.4f}; 预算{BUDGET} → "
          f"alpha*={alpha_used} tau*={tau_used}")

    # ---- 固定对集：test↔train 互近邻（sim≥0.90）----
    Vt, Vr = data["test"]["V"], data["train"]["V"]
    S = Vt @ Vr.T
    bt, br = S.argmax(1), S.argmax(0)
    pairs = [(i, int(bt[i])) for i in range(len(Vt))
             if int(br[int(bt[i])]) == i and S[i, int(bt[i])] >= 0.90]
    print(f"[retention-pairs] n={len(pairs)}")

    def predict(arm, V, cams, refl):
        if arm == "off":
            return sel.predict(V, cams=cams)
        if arm == "soft":
            return sel.predict(V, cams=cams, refl=refl, refl_mode="soft",
                               refl_base=base_cal, refl_alpha=alpha_used)
        return sel.predict(V, cams=cams, refl=refl, refl_tau=tau_used, refl_mode="hard")

    def evaluate(tag, arm):
        ids_t, _ = predict(arm, data["test"]["V"], data["test"]["cams"], data["test"]["refl"])
        ids_a, _ = predict(arm, data["test"]["Va"], data["test"]["cams"], data["test"]["refl_a"])
        ids_b, _ = predict(arm, data["test"]["Vb"], data["test"]["cams"], data["test"]["refl_b"])
        ids_r, _ = predict(arm, data["train"]["V"], data["train"]["cams"], data["train"]["refl"])
        ids_t, ids_a, ids_b, ids_r = map(np.asarray, (ids_t, ids_a, ids_b, ids_r))
        n = len(ids_t)
        unknown = float((ids_t == -1).mean())
        ab_both = (ids_a != -1) & (ids_b != -1)
        ab_same = float(((ids_a == ids_b) & ab_both).sum() / max(ab_both.sum(), 1))
        ret_same = sum(1 for i, j in pairs if ids_t[i] != -1 and ids_t[i] == ids_r[j])
        meta = [(data["test"]["keys"][i], data["test"]["cams"][i],
                 data["test"]["t"][i][0], data["test"]["t"][i][1], int(ids_t[i])) for i in range(n)]
        return dict(tag=tag, unknown=round(unknown, 4), covered=round(1 - unknown, 4),
                    ab_same=round(ab_same, 4), ret_same=ret_same, ret_n=len(pairs),
                    viol=len(overlap_pairs(meta)), n_test=n, ids=ids_t.tolist())

    arms = {"A_off": evaluate("A_off", "off"),
            "B_soft": evaluate("B_soft", "soft"),
            "C_hard": evaluate("C_hard", "hard")}

    # ---- 门控强度扫描（曲线）----
    def sweep_viol(ids):
        ids = np.asarray(ids)
        meta = [(data["test"]["keys"][i], data["test"]["cams"][i],
                 data["test"]["t"][i][0], data["test"]["t"][i][1], int(ids[i]))
                for i in range(len(ids))]
        return round(float((ids == -1).mean()), 4), len(overlap_pairs(meta))

    hard_sweep, soft_sweep = [], []
    for tau in TAUS:
        ids, _ = sel.predict(data["test"]["V"], cams=data["test"]["cams"],
                             refl=data["test"]["refl"], refl_tau=tau, refl_mode="hard")
        unk, v = sweep_viol(ids)
        hard_sweep.append(dict(tau=round(tau, 2), unknown=unk, viol=v))
    for al in ALPHAS:
        ids, _ = sel.predict(data["test"]["V"], cams=data["test"]["cams"],
                             refl=data["test"]["refl"], refl_mode="soft",
                             refl_base=base_cal, refl_alpha=al)
        unk, v = sweep_viol(ids)
        soft_sweep.append(dict(alpha=al, unknown=unk, viol=v))

    # ---- 公共子集（同覆盖）viol ----
    common = np.ones(len(data["test"]["keys"]), bool)
    for a in arms.values():
        common &= np.asarray(a["ids"]) != -1
    common_viol = {}
    for tag, a in arms.items():
        ids = np.asarray(a["ids"])
        meta = [(data["test"]["keys"][i], data["test"]["cams"][i],
                 data["test"]["t"][i][0], data["test"]["t"][i][1],
                 int(ids[i]) if common[i] else -1) for i in range(len(ids))]
        common_viol[tag] = len(overlap_pairs(meta))

    out = dict(version="reflection_gate_ab/1", selector=SEL, probe=probe["source"],
               probe_loo_auc_mean=round(probe["loo_auc_mean"], 4),
               refl_dist={sp: dict(mean=round(float(d["refl"].mean()), 4),
                                   p50=round(float(np.percentile(d["refl"], 50)), 4),
                                   p95=round(float(np.percentile(d["refl"], 95)), 4),
                                   n=int(len(d["refl"]))) for sp, d in data.items()},
               soft_base_cal=round(base_cal, 4),
               calib=dict(rule="val 段覆盖损失 ≤ %.2f 预算，取预算内最激进操作点" % BUDGET,
                          val_unknown_off=round(unk_v_off, 4),
                          alpha_star=alpha_used, tau_star=round(tau_used, 4)),
               arms={k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in arms.items()},
               hard_sweep=hard_sweep, soft_sweep=soft_sweep,
               regress_refl_none_equals_zero=same,
               common_subset=dict(n=int(common.sum()), viol=common_viol),
               caveat=("1012 无倒影标签 → 只给权衡曲线（覆盖 vs 误接代理），不给净收益；"
                       "操作点 α/τ 在 val 段按 5pp 覆盖预算校准（禁 test-oracle）；"
                       "判别器在 GIL 上留一机位 AUC 见 probe_loo_auc_mean；"
                       "注：1012 test 的跨机位时间重叠误接恒为 0 → viol 在本数据集无区分度；"
                       "selector 阈值很紧（卡在 score 分布上）→ 门控对偏移极敏感，α 必须很小"),
               note="只读实验；不写生产 selector_v2.npz")
    p = os.path.join(OUT, "reflection_gate_ab.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n{'arm':>8} {'covered':>8} {'ab_same':>8} {'ret_same':>10} {'viol':>5}")
    for tag, a in arms.items():
        print(f"{tag:>8} {a['covered']:>8} {a['ab_same']:>8} {a['ret_same']:>4}/{a['ret_n']:<5} "
              f"{a['viol']:>5}")
    print("hard sweep:", json.dumps(hard_sweep, ensure_ascii=False))
    print("soft sweep:", json.dumps(soft_sweep, ensure_ascii=False))
    print("common subset viol:", common_viol, "| n:", int(common.sum()))
    print("saved", p)


if __name__ == "__main__":
    main()
