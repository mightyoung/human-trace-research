#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测先验接入：下一门户头 → relay 候选排序（EVAL-4 建议落地）

背景（docs/eval4-next-portal-report.md §3）：
  "学习头（组件级）成立并可接入——建议作为 relay 的候选排序先验
   （对多出口机位的源实体：预测门 → 调 T 条件概率 → 候选重排/预选；
    对单出口机位无增益、无害）。"

本脚本（并列增量，不覆盖任何现有产物）：
  1) 训练 per-cam 门预测 softmax（口径 = eval4：10min 实体轨迹 / 9 维特征 / 500it），
     ★ 训练集剔除**全部评估目标边的 a 端实体**（active/confirmed/rejected + 判读对，防自证增强）
  2) T 表：scored_v2 全量 2618 候选聚合 zone-pair 计数（含 none 桶）；
     ★ 对每个"评估对"做 LOO 剔除（计算其 w_portal 时排除自身计数）
  3) 每候选：p(g|traj_a) → w_portal = Σ_g p_g[g]·cnt[(ca,g)→(cb,gn)]/tot[(ca,g)]
     - 仅对"多出口机位（exit>1）且头存在"的源应用（EVAL-4：单门无增益无害 → 不加）
  4) 三个并列量：
     - rank_old  按 score（= sim + λ·log w_dt，沿用 relay_transition_dist λ=0.01）
     - w_portal  纯门先验质量
     - score_prior = score + λ2·log(w_portal + eps)（λ2=0.01 原则性：与 λ 同量级同对数尺度；
       eps = 1/(N+2) 池规模归一；附 λ2 敏感性表，展示用不选参）
     - pre 预选标记：per a 端 w_portal ≥ 0.5×max（相对阈值，无真值依赖）→ "固定预算"形态
  5) 验收（人工强标签：rejected 7 假边 / confirmed 5 真边 / active 44 生产代理 / 判读对）
     - rejected 的预选剔除率与 rank 压制；confirmed+active 的预选保留率与 rank 保持
     - 多门机位源专项（305/316/323）；单门（318/343）"无害"核验
     - 对峙案例：323-E003(confirmed) vs 323-E035(rejected)

输出（并列）：
  analysis/relay_v2/next_portal_head_v1.json   训练头（含训练口径指纹）
  analysis/relay_v2/scored_v2_prior.json       全量候选 + w_portal/score_prior/pre 预选标记
用法：
  python tools/relay_portal_prior.py [--lam2 0.01] [--out-dir DIR]
"""
import argparse
import csv
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
C0 = os.path.join(ANA, "c0", "c0_baseline_v8.json")           # gates（v7==v8，用最新）
CAND = os.path.join(ANA, "relay_v2", "scored_v2.json")
GATED = os.path.join(ANA, "relay_v2", "edges_v2_gated_human.json")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
CAMS = ["305", "316", "318", "320", "323", "343"]
LONG35 = os.path.join(ANA, "topology", "long35_data")
RES = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
GATE_R = 0.15      # 门命中半径（同 eval4 L1 口径）
ZONE_R = 0.25      # zone 定位半径（同 eval4 L2 口径）
PRESELECT_FRAC = 0.5   # 预选相对阈值：w ≥ 0.5×max（per a 端）


# ---------------- 基础几何 / 特征（与 eval4 同口径，原样复用） ----------------
def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def nearest(pt, gates, r):
    best, bi = None, None
    for i, g in enumerate(gates):
        dd = dist(pt, g)
        if best is None or dd < best:
            best, bi = dd, i
    return bi if (best is not None and best <= r) else None


def feats(u, v, h, npts):
    k = min(6, len(u))
    du = (u[-1] - u[-k]) / max(k - 1, 1)
    dv = (v[-1] - v[-k]) / max(k - 1, 1)
    du_tot, dv_tot = u[-1] - u[0], v[-1] - v[0]
    sp = (du ** 2 + dv ** 2) ** 0.5
    return [u[-1], v[-1], du, dv, du_tot, dv_tot, sp, h[-1] / 1000.0, np.log(max(npts, 1))]


def split_traj(u, v, h):
    n = len(u)
    m = max(4, int(n * 0.5))
    return u[:m], v[:m], h[:m]


def train_softmax(X, y, K, iters=500, lr=0.5, l2=1e-3):
    N, D = X.shape
    W, b = np.zeros((D, K)), np.zeros(K)
    Y = np.zeros((N, K))
    Y[np.arange(N), y] = 1
    for _ in range(iters):
        Z = X @ W + b
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) / N
        W -= lr * (X.T @ G + l2 * W)
        b -= lr * G.sum(0)
    return W, b


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------- 数据加载 ----------------
def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def load_gates():
    d = json.load(open(C0, encoding="utf-8"))
    out = {}
    for cam, lst in d["gates"].items():
        out[cam] = dict(exit=[g["center"] for g in lst if g.get("kind") == "exit"],
                        entry=[g["center"] for g in lst if g.get("kind") == "entry"])
    return out


def load_judged():
    """合并判读（labeled.csv ∪ labels.json；后者优先）。{pair_id: '1'|'0'|'?'}

    ★ 命名空间过滤（同 relay_states）：S6B_P_/PRT_ 前缀剥离后并入（它们是标准 pair）；
      S6B_B_/S6B_R_ 实体级键与格式不符者跳过。"""
    judged = {}
    if os.path.exists(LABELED):
        for r in csv.DictReader(open(LABELED, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v in ("1", "0", "?"):
                judged[r["pair_id"]] = v
    if os.path.exists(LABELS_JSON):
        d = json.load(open(LABELS_JSON, encoding="utf-8"))
        for pid, rec in d.get("pair", {}).items():
            p2 = pid
            for pref in ("S6B_P_", "PRT_"):
                if p2.startswith(pref):
                    p2 = p2[len(pref):]
                    break
            if "__" not in p2:
                continue
            la, lb = p2.split("__", 1)[0].split("_"), p2.split("__", 1)[1].split("_")
            if len(la) < 3 or len(lb) < 3:
                continue
            judged[p2] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
    return judged


def load_long35_train(gates):
    """long35 train split 轨迹 → 训练样本（与 10min 使用域零交集，跨 session 零自证）。
    读取口径同 eval4 L1（detections.jsonl + splits_v4 keep 集）。"""
    keep = json.load(open(os.path.join(LONG35, "splits_v4.json"), encoding="utf-8"))
    keep = {cam: set(keep["train"]["tracks"][cam]) for cam in CAMS}
    samples = defaultdict(lambda: dict(X=[], y=[]))
    for cam in CAMS:
        W, Hh = RES[cam]
        gates_e = gates[cam]["exit"]
        acc = defaultdict(list)
        with open(os.path.join(LONG35, f"{cam}_detections.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for tid, box in zip(d["track_ids"], d["boxes"]):
                    if tid is None or int(tid) < 0 or int(tid) not in keep[cam]:
                        continue
                    x1, y1, x2, y2 = box
                    acc[int(tid)].append((d["frame"], (x1 + x2) / 2 / W, y2 / Hh, y2 - y1))
        for tid, lst in acc.items():
            lst.sort()
            fr = np.array([x[0] for x in lst], float)
            u = np.array([x[1] for x in lst])
            v = np.array([x[2] for x in lst])
            h = np.array([x[3] for x in lst])
            if len(u) < 8:
                continue
            hu, hv, hh = split_traj(u, v, h)
            lab = nearest((u[-1], v[-1]), gates_e, GATE_R)
            lab = int(lab) if lab is not None else -1
            samples[cam]["X"].append(feats(hu, hv, hh, len(fr)))
            samples[cam]["y"].append(lab)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam2", type=float, default=0.01,
                    help="门先验对数项权重（原则性取值：与 relay_transition_dist λ 同量级同尺度）")
    ap.add_argument("--out-dir", default=os.path.join(ANA, "relay_v2"))
    args = ap.parse_args()

    gates = load_gates()
    cand = json.load(open(CAND, encoding="utf-8"))["candidates"]
    N = len(cand)
    gd = json.load(open(GATED, encoding="utf-8"))["edges"]
    judged = load_judged()

    # 机位门数概览
    print("=== 机位门结构 ===")
    for cam in CAMS:
        print(f"  {cam}: exit {len(gates[cam]['exit'])} / entry {len(gates[cam]['entry'])}")

    # ---------------- 评估目标集（防自证剔除 + LOO） ----------------
    eval_status = ("active", "confirmed", "rejected")
    eval_edges = [e for e in gd if e["status"] in eval_status]
    eval_keys = {(e["a"], e["b"]) for e in eval_edges}
    # 状态映射（含 audit 背景，用于 rank 语境展示）
    status_of = {(e["a"], e["b"]): e["status"] for e in gd}
    judged_keys = {parse_pid(p) for p in judged}
    eval_keys |= judged_keys

    # 评估 a 端（记录用；训练源 = long35，与本域零交集）
    multi_cams = {c for c in CAMS if len(gates[c]["exit"]) > 1}
    eval_a_any = {a for a, b in eval_keys}
    print(f"多门机位（先验应用）: {sorted(multi_cams)}；评估涉及 a 端 {len(eval_a_any)}"
          f"（训练源 long35，与其零交集）")

    # ---------------- 1) 训练 per-cam 头（long35 train split，跨 session 零自证） ----------------
    ents = {}
    for cam in CAMS:
        ents[cam] = load_entities(cam)
    samples = load_long35_train(gates)
    train_n = {c: len(samples[c]["y"]) for c in CAMS}
    print(f"[train] long35 train split；per-cam n: {train_n}")

    models = {}
    head_out = {}
    for cam in CAMS:
        Xs = np.asarray(samples[cam]["X"], np.float64)
        ys = np.asarray(samples[cam]["y"], np.int64)
        K = len(gates[cam]["exit"]) + 1
        if len(Xs) < 20:
            models[cam] = None
            head_out[cam] = None
            continue
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
        W, b = train_softmax((Xs - mu) / sd, ys, K)
        freq = np.array([(ys == k).sum() for k in range(K)], float)
        freq /= freq.sum()
        models[cam] = dict(W=W, b=b, mu=mu, sd=sd, K=K, freq=freq, n=len(ys))
        head_out[cam] = dict(K=K, n=len(ys),
                             W=W.round(6).tolist(), b=b.round(6).tolist(),
                             mu=mu.round(6).tolist(), sd=sd.round(6).tolist(),
                             freq=freq.round(6).tolist())
    print("[train] models:", {c: (models[c]["n"] if models[c] else 0) for c in CAMS})

    def forward(cam, hu, hv, hh, npts):
        mm = models.get(cam)
        if mm is None:
            return None
        f = np.asarray(feats(hu, hv, hh, npts))[None, :]
        P = softmax(((f - mm["mu"]) / mm["sd"]) @ mm["W"] + mm["b"])[0]
        return P[:len(gates[cam]["exit"])]   # 门概率（不含 none 桶）

    # ---------------- 2) zone 键 + T 表 ----------------
    zkey = {}
    for c in cand:
        ca, cb = c["a"].split(":")[0], c["b"].split(":")[0]
        ge = nearest(c["a_end"], gates[ca]["exit"], ZONE_R)
        gn = nearest(c["b_start"], gates[cb]["entry"], ZONE_R)
        zkey[(c["a"], c["b"])] = (ca, ge, cb, gn)

    T = defaultdict(Counter)
    for k, (ca, ge, cb, gn) in zkey.items():
        T[(ca, ge)][(cb, gn)] += 1
    T_tot = {k2: sum(v.values()) for k2, v in T.items()}

    def w_portal_of(key, p_g):
        """Σ_g p_g[g]·cnt[(ca,g)→(cb,gn)] / tot[(ca,g)]；★ LOO：评估对剔除自身计数。"""
        ca, ge, cb, gn = zkey[key]
        loo = key in eval_keys
        w = 0.0
        any_tot = False
        for gi, pg in enumerate(p_g):
            if pg <= 0:
                continue
            cnts = T.get((ca, gi))
            if not cnts:
                continue
            cnt = cnts.get((cb, gn), 0)
            tot = T_tot[(ca, gi)]
            if loo and ge == gi and zkey[key] == (ca, gi, cb, gn):
                cnt -= 1
                tot -= 1
            if tot <= 0:
                continue
            any_tot = True
            w += float(pg) * max(cnt, 0) / tot
        return (w if any_tot else None)

    # ---------------- 3) 全量打分 ----------------
    eps_p = 1.0 / (N + 2.0)
    lam2 = args.lam2
    recs = []
    n_no_head = n_short = n_applied = 0
    for c in cand:
        key = (c["a"], c["b"])
        ca, ea = c["a"].split(":", 1)
        ent = ents.get(ca, {}).get(ea)
        rec = dict(a=c["a"], b=c["b"], sim=c["sim"], dt=c["dt"], score=c["score"],
                   status=status_of.get(key))
        p_g = None
        if ent is not None and len(ent[1]) >= 8:
            fr, u, v, h = ent
            hu, hv, hh = split_traj(u, v, h)
            p_g = forward(ca, hu, hv, hh, len(fr))
        if p_g is None:
            n_no_head += 1
            rec.update(w_portal=None, portal_applied=False, score_prior=c["score"],
                       note=("no_head" if models.get(ca) is None else "short_traj"))
            recs.append(rec)
            continue
        w = w_portal_of(key, p_g)
        multi = len(gates[ca]["exit"]) > 1
        applied = bool(multi and w is not None)
        sp = c["score"]
        if applied:
            sp = c["score"] + lam2 * float(np.log(w + eps_p))
            n_applied += 1
        rec.update(w_portal=(None if w is None else round(w, 6)),
                   portal_applied=applied, score_prior=round(sp, 6),
                   predicted_gate=int(np.argmax(p_g)))
        recs.append(rec)

    print(f"[score] n={len(recs)}  应用先验 {n_applied}  无头/短轨 {n_no_head}")

    # ---------------- 4) rank 计算（per a 端） ----------------
    by_a = defaultdict(list)
    for r in recs:
        by_a[r["a"]].append(r)
    for a, lst in by_a.items():
        old = sorted(lst, key=lambda r: -r["score"])
        new = sorted(lst, key=lambda r: -r["score_prior"])
        for i, r in enumerate(old, 1):
            r["rank_old"] = i
        for i, r in enumerate(new, 1):
            r["rank_new"] = i
        # 纯门先验 rank（仅 applied 者参与）
        ap = [r for r in lst if r["portal_applied"]]
        for i, r in enumerate(sorted(ap, key=lambda r: -(r["w_portal"] or 0)), 1):
            r["rank_portal"] = i
        # 预选（相对阈值 0.5×max）
        wmax = max([r["w_portal"] for r in ap], default=None)
        for r in lst:
            if r["portal_applied"] and wmax and wmax > 0:
                r["preselect"] = bool(r["w_portal"] >= PRESELECT_FRAC * wmax)
            else:
                r["preselect"] = None

    # ---------------- 5) 验收 ----------------
    def sel(pred):
        return [r for r in recs if pred(r)]

    def rank_stats(rs, label):
        ok1o = sum(r["rank_old"] == 1 for r in rs)
        ok1n = sum(r["rank_new"] == 1 for r in rs)
        keep2 = sum(r["rank_new"] <= 2 for r in rs)
        print(f"  [{label}] n={len(rs)} | rank_old=1: {ok1o}  rank_new=1: {ok1n}  "
              f"rank_new≤2: {keep2}")
        return dict(n=len(rs), top1_old=ok1o, top1_new=ok1n, top2_new=keep2)

    print("\n=== 验收 A：人工强标签 ===")
    rej = sel(lambda r: r["status"] == "rejected")
    conf = sel(lambda r: r["status"] == "confirmed")
    act = sel(lambda r: r["status"] == "active")
    print("-- rejected 7（人工假边）:")
    acc_rej = rank_stats(rej, "rejected")
    for r in sorted(rej, key=lambda r: r["a"]):
        print(f"    {r['a']:<18}->{r['b']:<18} rank {r['rank_old']}->{r['rank_new']} "
              f"w={r['w_portal']} pre={r['preselect']}")
    print("-- confirmed 5（人工真边）:")
    acc_conf = rank_stats(conf, "confirmed")
    for r in sorted(conf, key=lambda r: r["a"]):
        print(f"    {r['a']:<18}->{r['b']:<18} rank {r['rank_old']}->{r['rank_new']} "
              f"w={r['w_portal']} pre={r['preselect']}")
    print("-- active 44（生产代理）:")
    acc_act = rank_stats(act, "active")

    print("\n=== 验收 B：多门机位源专项（305/316/323）===")
    m_all = [r for r in recs if r["a"].split(":")[0] in multi_cams and r["portal_applied"]]
    m_rej = [r for r in rej if r["a"].split(":")[0] in multi_cams]
    m_conf = [r for r in conf if r["a"].split(":")[0] in multi_cams]
    m_act = [r for r in act if r["a"].split(":")[0] in multi_cams]
    print(f"  多门源候选（applied）n={len(m_all)}")
    rank_stats(m_rej, "rejected@多门")
    rank_stats(m_conf, "confirmed@多门")
    rank_stats(m_act, "active@多门")
    # 预选形态：正/负边保留率（分母 = 该组内 applied 的条数）
    for grp, lbl in ((m_rej, "rejected"), (m_conf, "confirmed"), (m_act, "active")):
        n_ap = sum(1 for r in grp if r["portal_applied"])
        kept = sum(1 for r in grp if r["preselect"])
        print(f"  预选保留 {lbl}@多门: {kept}/{n_ap}（applied）")
    n_pre = sum(1 for r in m_all if r["preselect"])
    print(f"  预选面（多门 applied 候选）: {n_pre}/{len(m_all)} = {n_pre/max(len(m_all),1):.1%}")

    print("\n=== 验收 C：单门机位无害核验（318/343）===")
    sing = [r for r in recs if r["a"].split(":")[0] in ("318", "343")]
    n_app = sum(1 for r in sing if r["portal_applied"])
    print(f"  单门源候选 n={len(sing)}  应用先验 {n_app}（应为 0）")

    print("\n=== 验收 C2：10min 使用域健全性（非评估实体；学习 vs 运动学 vs 边际）===")
    sanity = {}
    for cam in sorted(multi_cams):
        mm = models.get(cam)
        gates_e = gates[cam]["exit"]
        if mm is None:
            print(f"  {cam}: 无头（样本不足）")
            continue
        n_hit = n_kin = n_marg = n_tot = 0
        for eid, (fr, u, v, h) in ents[cam].items():
            if f"{cam}:{eid}" in eval_a_any or len(u) < 8:
                continue
            ytrue = nearest((u[-1], v[-1]), gates_e, GATE_R)
            if ytrue is None:
                continue
            hu, hv, hh = split_traj(u, v, h)
            p = forward(cam, hu, hv, hh, len(fr))
            k = min(6, len(hu))
            du = (hu[-1] - hu[-k]) / max(k - 1, 1)
            dv = (hv[-1] - hv[-k]) / max(k - 1, 1)
            ks = []
            for gt in gates_e:
                vec = np.array([gt[0] - hu[-1], gt[1] - hv[-1]])
                nv = np.linalg.norm(vec) + 1e-9
                dvv = np.array([du, dv])
                nd = np.linalg.norm(dvv) + 1e-9
                ks.append(float(dvv @ vec / (nd * nv)) - 0.3 * float(nv))
            n_tot += 1
            n_hit += int(int(np.argmax(p)) == ytrue)
            n_kin += int(int(np.argmax(ks)) == ytrue)
            n_marg += int(int(np.argmax(mm["freq"][:len(gates_e)])) == ytrue)
        sanity[cam] = dict(n=n_tot, lr=n_hit, kin=n_kin, marg=n_marg,
                           lr_top1=round(n_hit / max(n_tot, 1), 4),
                           kin_top1=round(n_kin / max(n_tot, 1), 4),
                           marg_top1=round(n_marg / max(n_tot, 1), 4))
        print(f"  {cam}: n={n_tot} 学习 {n_hit/max(n_tot,1):.3f} | "
              f"运动学 {n_kin/max(n_tot,1):.3f} | 边际 {n_marg/max(n_tot,1):.3f}")

    print("\n=== 验收 D：判读对回放 ===")
    jmap = {}
    for pid, v in judged.items():
        jmap[parse_pid(pid)] = v
    jpos = [r for r in recs if jmap.get((r["a"], r["b"])) == "1"]
    jneg = [r for r in recs if jmap.get((r["a"], r["b"])) == "0"]
    print(f"  判读正对（池内）n={len(jpos)}；判读负对（池内）n={len(jneg)}")
    if jpos:
        rank_stats(jpos, "judged_pos")
    if jneg:
        rank_stats(jneg, "judged_neg")

    print("\n=== 验收 E：对峙案例 323-E003(confirmed) vs 323-E035(rejected) ===")
    for r in recs:
        if r["a"] in ("323:323-E003", "323:323-E035"):
            print(f"    {r['a']:<18}->{r['b']:<18} status={r['status']} "
                  f"score {r['score']:.4f}->{r['score_prior']:.4f} "
                  f"w={r['w_portal']} gate={r.get('predicted_gate')} "
                  f"rank {r['rank_old']}->{r['rank_new']} pre={r['preselect']}")

    # λ2 敏感性（展示用）
    print("\n=== 敏感性：λ2 扫描（展示用，不选参）===")
    sens = {}
    for l2 in (0.0, 0.005, 0.01, 0.02, 0.05):
        for a, lst in by_a.items():
            for r in lst:
                if r["portal_applied"] and r["w_portal"] is not None:
                    r["_sp"] = r["score"] + l2 * float(np.log(r["w_portal"] + eps_p))
                else:
                    r["_sp"] = r["score"]
            new = sorted(lst, key=lambda r: -r["_sp"])
            for i, r in enumerate(new, 1):
                r["_rank"] = i
        rj1 = sum(r["_rank"] == 1 for r in rej)
        rj2 = sum(r["_rank"] <= 2 for r in rej)
        cf1 = sum(r["_rank"] == 1 for r in conf)
        ac1 = sum(r["_rank"] == 1 for r in act)
        sens[l2] = dict(rejected_top1=rj1, rejected_top2=rj2, confirmed_top1=cf1,
                        active_top1=ac1)
        print(f"  λ2={l2:<6} rejected top1={rj1} top2={rj2} | confirmed top1={cf1} | "
              f"active top1={ac1}")
    for r in recs:
        r.pop("_sp", None)
        r.pop("_rank", None)

    # ---------------- 6) 输出 ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    head_path = os.path.join(args.out_dir, "next_portal_head_v1.json")
    json.dump({"version": "relay_v2/portal_head/1",
               "note": "per-cam 门预测 softmax；训练=long35 train split（跨 session，"
                       "10min 使用域零自证）；口径同 eval4（9 维特征/500it/lr0.5/l2=1e-3）",
               "train": {"n": train_n, "source": "long35_train_split",
                         "eval_a_any": len(eval_a_any),
                         "gates_file": os.path.relpath(C0, ROOT).replace("\\", "/")},
               "models": head_out},
              open(head_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    out_path = os.path.join(args.out_dir, "scored_v2_prior.json")
    json.dump({"version": "relay_v2/scored_prior/1",
               "based_on": ["relay_v2/scored_v2.json", "relay_v2/edges_v2_gated_human.json"],
               "lam2": lam2, "eps_p": eps_p, "preselect_frac": PRESELECT_FRAC,
               "n": len(recs), "n_applied": n_applied,
               "acceptance": {"rejected": acc_rej, "confirmed": acc_conf, "active": acc_act,
                              "sanity_10min": sanity},
               "sensitivity": {str(k): v for k, v in sens.items()},
               "candidates": recs},
              open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {head_path}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
