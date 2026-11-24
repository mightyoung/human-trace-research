# -*- coding: utf-8 -*-
"""EVAL-2（对照实验，紧凑版）：学习式小评审器 vs EVAL-1 规则基线。

纪律（按计划）：
  - 训练样本 = **受控扰动 + 规则自动生成**（不读身份真值）；身份真值仅用于隔离评测；
  - 预注册失败条件：若学习器只复现规则且**增加误拒** → 不引入。

实现：
  1) 从 registry_v0.json 的候选六层快照取 12 维数值特征；
  2) 生成 20× 扰动样本（连续特征高斯噪声、类别位随机翻转），规则重算得标签（3 类）；
  3) softmax 回归（numpy）训练；在留出扰动样本上测"规则复现率"；
  4) 隔离评测：真实判读集上比较 规则 vs 学习器（误拒/误收/AUC(P(keep))）。

输出：analysis/eval2/learned_reviewer.json
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
REG = os.path.join(ANA, "eval_registry", "registry_v0.json")
OUT = os.path.join(ANA, "eval2")

P = dict(sim_hi=0.95, sim_rej=0.943, w_dt_min=0.30)   # 与 EVAL-1 v0.1 一致


def rule_action(sim, w_dt, dt, overlap, l1_fail, l4_fail, gate_warn):
    if l4_fail or l1_fail:
        return 2   # reject
    if sim is not None and sim < P["sim_rej"]:
        return 2
    if (sim is not None and sim >= P["sim_hi"] and w_dt is not None and w_dt >= P["w_dt_min"]
            and dt is not None and dt >= -1 and not gate_warn):
        return 0   # keep
    return 1       # defer


def feats(layers):
    e3 = layers["L3_evidence"]
    l2 = layers["L2_geometry"]
    l4 = layers["L4_constraints"]
    l1 = layers["L1_quality"]
    sim = e3["sim"] if e3["sim"] is not None else 0.0
    wdt = e3["w_dt"] if e3["w_dt"] is not None else 0.0
    dt = e3["dt"] if e3["dt"] is not None else 99.0
    mg = e3["margin_out"] if e3["margin_out"] is not None else 0.0
    pda = l2["pd_a"] if l2["pd_a"] is not None else 0.5
    pdb = l2["pd_b"] if l2["pd_b"] is not None else 0.5
    ov = 1.0 if l4["overlap"] else 0.0
    l4f = 1.0 if l4["status"] == "fail" else 0.0
    l1f = 1.0 if l1["status"] == "fail" else 0.0
    gw = 1.0 if l2["status"] == "warn" else 0.0
    ha = (l1["a_h"] or 0) / 1000.0
    hb = (l1["b_h"] or 0) / 1000.0
    return np.array([sim, wdt, dt, mg, pda, pdb, ov, l4f, l1f, gw, ha, hb], float)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train(X, y, K=3, iters=600, lr=0.5, l2=1e-3):
    N, D = X.shape
    W, b = np.zeros((D, K)), np.zeros(K)
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1
    for _ in range(iters):
        Z = X @ W + b
        Pm = softmax(Z)
        G = (Pm - Y) / N
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def main():
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(REG, encoding="utf-8"))
    rows = [(r["a"], r["b"], r["layers"]) for r in d["candidates"]]
    F = np.asarray([feats(l) for _, _, l in rows])
    y = np.asarray([rule_action(*_rule_args(l)) for _, _, l in rows])

    rng = np.random.default_rng(0)
    NF = 20
    Xa, ya = [], []
    for _ in range(NF):
        Fp = F.copy()
        Fp[:, [0, 1, 3, 4, 5, 10, 11]] += rng.normal(0, 0.02, size=(len(F), 7))
        Fp[:, 2] += rng.normal(0, 2.0, size=len(F))
        flip = rng.random(len(F)) < 0.05
        Fp[flip, 6] = 1 - Fp[flip, 6]
        Fp = np.clip(Fp, -5, 120)
        # 扰动后标签：规则重算（近似：直接对扰动特征重跑规则核心）
        yp = np.where(
            (Fp[:, 7] > 0.5) | (Fp[:, 8] > 0.5) | (Fp[:, 0] < P["sim_rej"]), 2,
            np.where((Fp[:, 0] >= P["sim_hi"]) & (Fp[:, 1] >= P["w_dt_min"])
                     & (Fp[:, 2] >= -1) & (Fp[:, 9] < 0.5), 0, 1))
        Xa.append(Fp)
        ya.append(yp)
    Xa = np.concatenate(Xa)
    ya = np.concatenate(ya)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    ntr = int(len(Xa) * 0.9)
    W, b = train((Xa[:ntr] - mu) / sd, ya[:ntr])
    pred = softmax(((Xa[ntr:] - mu) / sd) @ W + b).argmax(1)
    agree_perturb = float((pred == ya[ntr:]).mean())

    # 隔离评测（真实判读）
    def parse_pid(pid):
        left, right = pid.split("__")
        la, lb = left.split("_"), right.split("_")
        return (f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}")

    judged = {}
    for r in csv.DictReader(open(os.path.join(ANA, "truth", "pair_candidates_labeled.csv"),
                                 encoding="utf-8-sig")):
        v = (r.get("human_same_person") or "").strip()
        if v in ("1", "0", "?"):
            judged[r["pair_id"]] = v
    lj = json.load(open(os.path.join(ANA, "truth", "remote", "labels.json"), encoding="utf-8"))
    for pid, rec in lj.get("pair", {}).items():
        judged[pid] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
    idx = {(a, b): (l, int(yy)) for (a, b, l), yy in zip(rows, y)}
    pos_scores, neg_scores = [], []
    res = dict(rule=dict(pos_keep=0, pos_rej=0, neg_rej=0, neg_keep=0, n_pos=0, n_neg=0),
               lrn=dict(pos_keep=0, pos_rej=0, neg_rej=0, neg_keep=0, n_pos=0, n_neg=0))
    for pid, v in judged.items():
        # 命名空间防御：S6B_/PRT_ 命名空间与非法格式键跳过（非本标准 pair）
        if pid.startswith(("S6B", "PRT_")) or "__" not in pid:
            continue
        k = parse_pid(pid)
        it = idx.get(k)
        if it is None:
            continue
        layers, y_rule = it
        f = feats(layers)[None, :]
        prob = softmax(((f - mu) / sd) @ W + b)[0]
        a_lrn = int(prob.argmax())
        if v == "1":
            res["rule"]["n_pos"] += 1
            res["lrn"]["n_pos"] += 1
            if y_rule == 0:
                res["rule"]["pos_keep"] += 1
            if y_rule == 2:
                res["rule"]["pos_rej"] += 1
            if a_lrn == 0:
                res["lrn"]["pos_keep"] += 1
            if a_lrn == 2:
                res["lrn"]["pos_rej"] += 1
            pos_scores.append(float(prob[0]))
        elif v == "0":
            res["rule"]["n_neg"] += 1
            res["lrn"]["n_neg"] += 1
            if y_rule == 2:
                res["rule"]["neg_rej"] += 1
            if y_rule == 0:
                res["rule"]["neg_keep"] += 1
            if a_lrn == 2:
                res["lrn"]["neg_rej"] += 1
            if a_lrn == 0:
                res["lrn"]["neg_keep"] += 1
            neg_scores.append(float(prob[0]))

    def auc(pos, neg):
        if not pos or not neg:
            return None
        p, n = np.asarray(pos), np.asarray(neg)
        return round(float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean()), 4)

    out = dict(version="eval2/1",
               note="训练=19×扰动+规则标签（无真值）；真值仅隔离评测",
               agree_perturb=round(agree_perturb, 4),
               eval_judged=res,
               auc_keep=auc(pos_scores, neg_scores),
               fail_condition="若学习器只复现规则且增加误拒 → 不引入")
    with open(os.path.join(OUT, "learned_reviewer.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("扰动留出集上规则复现率:", round(agree_perturb, 4))
    print("规则:", res["rule"], "| 学习器:", res["lrn"])
    print("AUC(P_keep | pos vs neg):", out["auc_keep"])
    print("saved", os.path.join(OUT, "learned_reviewer.json"))


def _rule_args(l):
    e3 = l["L3_evidence"]
    return (e3["sim"], e3["w_dt"], e3["dt"],
            1.0 if l["L4_constraints"]["overlap"] else 0.0,
            1.0 if l["L1_quality"]["status"] == "fail" else 0.0,
            1.0 if l["L4_constraints"]["status"] == "fail" else 0.0,
            1.0 if l["L2_geometry"]["status"] == "warn" else 0.0)


if __name__ == "__main__":
    main()
