#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""留出复验判定脚本（冻结版）—— 实现预注册 §3 冻结口径与 §4 五条假设。

输入（--pkg 目录）：
  results/summary.csv          d7_sweep 输出（取 coverage = both_accept_rate）
  results/members/<cfg>.csv    d7_sweep 输出（config, cluster_local, cluster_lab, track_id）
  results/queries/<cfg>.csv    d7_sweep 输出（query_id, cam, final_status, final_cluster_local …）
  gt/gt_gallery_members_FILLED.csv   列 track_id, person_id
  gt/gt_query_gt_FILLED.csv          列 query_id, person_id

口径（预注册 §3，冻结）：
  - 簇主体身份 = 簇内已标成员投票多数派；并列第一 → 无主体（U）。
  - 查询真值为空/UNK → 不计（no_truth）；no-human 或 shadow* → 非人，剔出；
    真值不在图库已标身份中 → 图库外，剔出（不计入 C/W/U）。其余为"可答"。
  - 精度 = C/(C+W)；C/N 的 N = 可评查询数（queries 文件行数）。
  - 继承精度（严格）= 已接受可答查询中，所落簇恰含 1 个已标身份的比例。
  - 置信区间：按查询真值身份聚类 bootstrap，4000 次，种子 20261024。
  - 配对检验：在全部可答查询上做精确 McNemar（双侧），未接受即未答对。

假设（预注册 §4，冻结）：
  H1a  42 配置上 Spearman(继承精度, C) ≤ −0.30
  H1b  继承精度最高的配置（并列取配置名字典序最小者）ΔC < 0
  H1c  A_baseline 继承精度 < 逐条精度
  H2c  C_dmax0.30 相对 A_baseline：ΔC < 0 且 McNemar p < 0.05
  H3   A_baseline：未答 > W
前置条件：A_baseline 可答 ≥ 100 且 coverage ≥ 0.10；不满足则五条全部"无法判定"，
照常输出描述性数值。若 |ρ + 0.30| < 0.05，另报"成员行序决胜"口径的 ρ（§6，敏感性，不参与判定）。

用法：
  python tools/holdout_eval.py --pkg output_crosscam_local/holdout/eval
输出：<pkg>/eval_out/（verdicts/、precision.csv、h1.csv、loss_decomposition.csv、
      camera.csv、multicomp.csv、verdicts.json、VERDICTS.md）
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import sys

BASE = "A_baseline"
VETO = "C_dmax0.30"
SEED = 20261024
N_BOOT = 4000
MIN_ANSWERABLE = 100
MIN_COVERAGE = 0.10
RHO_THR = -0.30


def rd(p):
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def wr(p, rows, fields=None):
    if not rows:
        open(p, "w", encoding="utf-8-sig").close()
        return
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields or list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_gt(path, key):
    d = {}
    for r in rd(path):
        v = (r.get("person_id") or "").strip()
        if v and v.upper() != "UNK":
            d[r[key]] = v
    return d


def is_np(p):
    p = (p or "").lower()
    return p == "no-human" or p.startswith("shadow")


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def cluster_identity(members, gtruth, tie):
    """cluster_local → (主体身份或 '', 已标数, 身份数)。tie='strict'：并列→''；'order'：按成员行序决胜。"""
    out = {}
    for cl, trks in members.items():
        v = collections.Counter()
        for t in trks:
            p = gtruth.get(t)
            if p:
                v[p] += 1
        if not v:
            out[cl] = ("", 0, 0)
            continue
        top = max(v.values())
        led = [p for p, n in v.items() if n == top]
        if tie == "strict" and len(led) >= 2:
            out[cl] = ("", sum(v.values()), len(v))
        else:
            out[cl] = (v.most_common(1)[0][0], sum(v.values()), len(v))
    return out


def evaluate(run, gtruth, qtruth, tie):
    gallery_ids = set(gtruth.values())
    cfgs = sorted(f[:-4] for f in os.listdir(os.path.join(run, "members")) if f.endswith(".csv"))
    per = {}
    for cfg in cfgs:
        qpath = os.path.join(run, "queries", cfg + ".csv")
        if not os.path.exists(qpath):
            continue
        mem = collections.defaultdict(list)
        for r in rd(os.path.join(run, "members", cfg + ".csv")):
            mem[int(r["cluster_local"])].append(r["track_id"])
        ident = cluster_identity(mem, gtruth, tie)
        rows = rd(qpath)
        verdict, detail = {}, []
        loss = collections.defaultdict(collections.Counter)
        C = W = U = acc = inh = 0
        for r in rows:
            cam, qid = r["cam"], r["query_id"]
            q = qtruth.get(qid, "")
            loss[cam]["evaluable"] += 1
            if not q:
                loss[cam]["no_truth"] += 1
                continue
            if is_np(q):
                loss[cam]["nonperson"] += 1
                continue
            if q not in gallery_ids:
                loss[cam]["gallery_out"] += 1
                continue
            loss[cam]["answerable"] += 1
            if r["final_status"] != "accepted":
                loss[cam]["unanswered"] += 1
                verdict[qid] = "-"
                continue
            acc += 1
            loss[cam]["accepted"] += 1
            cl = int(r["final_cluster_local"])
            cp, n_lab, n_ids = ident.get(cl, ("", 0, 0))
            inh += 1 if n_ids == 1 else 0
            if not cp:
                v = "U"
                U += 1
            elif cp == q:
                v = "C"
                C += 1
            else:
                v = "W"
                W += 1
            loss[cam][{"C": "correct", "W": "wrong", "U": "unresolved"}[v]] += 1
            verdict[qid] = v
            detail.append(dict(query_id=qid, cam=cam, cluster_local=cl, q_pid=q, c_pid=cp,
                               verdict=v, cluster_n_lab=n_lab, cluster_n_ids=n_ids))
        per[cfg] = dict(C=C, W=W, U=U, acc=acc, N=len(rows),
                        precision=(C / (C + W)) if C + W else float("nan"),
                        inh=(inh / acc) if acc else float("nan"),
                        verdict=verdict, detail=detail, loss=loss, rows=rows)
    return per


def clustered_ci(rows, qtruth, verdict, seed=SEED, n=N_BOOT):
    """C/N 的身份聚类 bootstrap：按查询真值身份分组（无真值者各自成组）。"""
    groups = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        qid = r["query_id"]
        g = qtruth.get(qid) or ("__none__" + qid)
        groups[g][1] += 1
        groups[g][0] += 1 if verdict.get(qid) == "C" else 0
    keys = list(groups)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        c = t = 0
        for _k in range(len(keys)):
            a, b = groups[keys[rng.randrange(len(keys))]]
            c += a
            t += b
        vals.append(c / t if t else float("nan"))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1], len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", default="", help="默认 <pkg>/eval_out")
    a = ap.parse_args()

    run = os.path.join(a.pkg, "results")
    gtg = os.path.join(a.pkg, "gt", "gt_gallery_members_FILLED.csv")
    gtq = os.path.join(a.pkg, "gt", "gt_query_gt_FILLED.csv")
    for p in (gtg, gtq, os.path.join(run, "summary.csv")):
        if not os.path.exists(p):
            print(f"[ABORT] 缺少 {p}", file=sys.stderr)
            sys.exit(3)
    out = a.out or os.path.join(a.pkg, "eval_out")
    os.makedirs(os.path.join(out, "verdicts"), exist_ok=True)

    gtruth = load_gt(gtg, "track_id")
    qtruth = load_gt(gtq, "query_id")
    summ = {r["config"]: r for r in rd(os.path.join(run, "summary.csv"))}
    per = evaluate(run, gtruth, qtruth, "strict")
    if BASE not in per:
        print("[ABORT] 缺少 A_baseline 结果", file=sys.stderr)
        sys.exit(3)
    b = per[BASE]

    # ---- 逐配置表与逐条判决
    prec, h1 = [], []
    for cfg, p in per.items():
        wr(os.path.join(out, "verdicts", cfg + ".csv"), p["detail"],
           ["query_id", "cam", "cluster_local", "q_pid", "c_pid", "verdict",
            "cluster_n_lab", "cluster_n_ids"])
        s = summ.get(cfg, {})
        prec.append(dict(config=cfg, K=s.get("K", ""), theta=s.get("theta", ""),
                         coverage=s.get("both_accept_rate", ""), accepted=p["acc"],
                         C=p["C"], W=p["W"], U=p["U"],
                         precision=round(p["precision"], 4) if p["C"] + p["W"] else "",
                         C_over_N=round(p["C"] / p["N"], 4) if p["N"] else "", N=p["N"]))
        h1.append(dict(config=cfg, accepted=p["acc"], C=p["C"],
                       inh_strict=round(p["inh"], 4) if p["acc"] else ""))
    wr(os.path.join(out, "precision.csv"), prec)
    wr(os.path.join(out, "h1.csv"), h1)

    # ---- 损失分解与逐机位
    cams = sorted(b["loss"])
    allc = collections.Counter()
    for c in cams:
        allc.update(b["loss"][c])
    fields = ["evaluable", "no_truth", "nonperson", "gallery_out", "answerable",
              "accepted", "correct", "wrong", "unresolved", "unanswered"]
    loss_rows = [dict(group="ALL", **{k: allc[k] for k in fields})]
    loss_rows += [dict(group=c, **{k: b["loss"][c][k] for k in fields}) for c in cams]
    wr(os.path.join(out, "loss_decomposition.csv"), loss_rows)
    cam_rows = []
    for c in cams:
        L = b["loss"][c]
        cam_rows.append(dict(cam=c, answerable=L["answerable"], accepted=L["accepted"],
                             C=L["correct"], W=L["wrong"], U=L["unresolved"],
                             precision=round(L["correct"] / (L["correct"] + L["wrong"]), 4)
                             if L["correct"] + L["wrong"] else ""))
    wr(os.path.join(out, "camera.csv"), cam_rows)

    # ---- 配对比较（全部可答查询）与多重校正
    answerable = [q for q, v in b["verdict"].items()]
    def paired(cfg):
        v = per[cfg]["verdict"]
        lost = sum(1 for q in answerable if b["verdict"][q] == "C" and v.get(q) != "C")
        gain = sum(1 for q in answerable if v.get(q) == "C" and b["verdict"][q] != "C")
        return gain - lost, lost, gain, mcnemar(lost, gain)
    mc = []
    for cfg in per:
        if cfg == BASE:
            continue
        d, lo, ga, p = paired(cfg)
        mc.append(dict(config=cfg, C=per[cfg]["C"], delta=d, lost=lo, gained=ga, p=p))
    mc.sort(key=lambda r: r["p"])
    m = len(mc)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, mc[i]["p"] * m / (i + 1))
        mc[i]["bh_q"] = prev
    for r in mc:
        r["bonf_sig"] = r["p"] < 0.05 / m if m else False
        r["bh_sig"] = r["bh_q"] < 0.05
    wr(os.path.join(out, "multicomp.csv"), mc)

    # ---- 假设
    ok = [(cfg, p) for cfg, p in per.items() if p["acc"] > 0]
    rho = spearman([p["inh"] for _, p in ok], [p["C"] for _, p in ok])
    best_inh = max(p["inh"] for _, p in ok)
    best_cfg = sorted(cfg for cfg, p in ok if p["inh"] == best_inh)[0]
    tied_best = sorted(cfg for cfg, p in ok if p["inh"] == best_inh)
    h1b_d, _, _, h1b_p = paired(best_cfg)
    veto_d, veto_lost, veto_gain, veto_p = paired(VETO) if VETO in per else (None, None, None, None)
    cov = float(summ.get(BASE, {}).get("both_accept_rate") or 0)
    n_ans = allc["answerable"]
    gate = n_ans >= MIN_ANSWERABLE and cov >= MIN_COVERAGE

    def judge(cond):
        if not gate:
            return "无法判定"
        return "成立" if cond else "不成立"

    H = {
        "H1a": dict(value=round(rho, 3), threshold="rho <= -0.30", n_configs=len(ok),
                    verdict=judge(rho <= RHO_THR)),
        "H1b": dict(config=best_cfg, tied=tied_best, inherited=round(best_inh, 4),
                    delta_C=h1b_d, p=h1b_p, verdict=judge(h1b_d < 0)),
        "H1c": dict(inherited=round(b["inh"], 4), precision=round(b["precision"], 4),
                    verdict=judge(b["inh"] < b["precision"])),
        "H2c": dict(delta_C=veto_d, lost=veto_lost, gained=veto_gain, p=veto_p,
                    verdict=judge(veto_d is not None and veto_d < 0 and veto_p < 0.05)),
        "H3": dict(unanswered=allc["unanswered"], wrong=allc["wrong"],
                   verdict=judge(allc["unanswered"] > allc["wrong"])),
    }
    sens = None
    if abs(rho - RHO_THR) < 0.05:
        po = evaluate(run, gtruth, qtruth, "order")
        oko = [(c, p) for c, p in po.items() if p["acc"] > 0]
        sens = round(spearman([p["inh"] for _, p in oko], [p["C"] for _, p in oko]), 3)
        H["H1a"]["sensitivity_order_tiebreak_rho"] = sens

    lo, hi, ng = clustered_ci(b["rows"], qtruth, b["verdict"])
    summary = dict(
        gate=dict(answerable=n_ans, min_answerable=MIN_ANSWERABLE, coverage=cov,
                  min_coverage=MIN_COVERAGE, passed=gate),
        baseline=dict(K=summ.get(BASE, {}).get("K"), theta=summ.get(BASE, {}).get("theta"),
                      coverage=cov, N=b["N"], accepted=b["acc"], C=b["C"], W=b["W"], U=b["U"],
                      precision=round(b["precision"], 4) if b["C"] + b["W"] else None,
                      C_over_N=round(b["C"] / b["N"], 4) if b["N"] else None,
                      C_over_N_CI95=[round(lo, 4), round(hi, 4)], identity_groups=ng),
        hypotheses=H,
        configs_evaluated=len(per),
        multiple_comparisons=dict(m=m, bonferroni_alpha=0.05 / m if m else None,
                                  bonferroni_sig=[r["config"] for r in mc if r["bonf_sig"]],
                                  bonferroni_sig_positive=[r["config"] for r in mc
                                                           if r["bonf_sig"] and r["delta"] > 0]),
    )
    with open(os.path.join(out, "verdicts.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    L = ["# 留出复验判定（holdout_eval.py 自动生成）", "",
         f"- 前置条件：可答 {n_ans}（门槛 {MIN_ANSWERABLE}）｜覆盖 {cov:.3f}（门槛 {MIN_COVERAGE}）"
         f" → **{'通过' if gate else '未通过：五条一律无法判定'}**",
         f"- A_baseline：接受 {b['acc']}｜C {b['C']} / W {b['W']} / U {b['U']}｜精度 "
         f"{summary['baseline']['precision']}｜C/N {summary['baseline']['C_over_N']} "
         f"（身份聚类 95% CI [{lo:.3f}, {hi:.3f}]，{ng} 组）｜N {b['N']}",
         f"- 评估配置数 {len(per)}（d7_sweep 无可行点的配置不出现）", "",
         "| 假设 | 实测 | 判定 |", "|---|---|---|",
         f"| H1a | ρ = {H['H1a']['value']}（{len(ok)} 配置）"
         + (f"；敏感性（行序决胜）ρ = {sens}" if sens is not None else "") + f" | {H['H1a']['verdict']} |",
         f"| H1b | {best_cfg}（继承 {best_inh:.3f}）ΔC = {h1b_d:+d}，p = {h1b_p:.3g} | {H['H1b']['verdict']} |",
         f"| H1c | 继承 {b['inh']:.3f} vs 逐条 {b['precision']:.3f} | {H['H1c']['verdict']} |",
         (f"| H2c | ΔC = {veto_d:+d}（{veto_gain} 增 / {veto_lost} 减），p = {veto_p:.3g} | {H['H2c']['verdict']} |"
          if veto_d is not None else f"| H2c | {VETO} 无结果 | {H['H2c']['verdict']} |"),
         f"| H3 | 未答 {allc['unanswered']} vs 答错 {allc['wrong']} | {H['H3']['verdict']} |", "",
         f"多重比较：m = {m}，Bonferroni 显著 {len(summary['multiple_comparisons']['bonferroni_sig'])} 个，"
         f"其中正向 {len(summary['multiple_comparisons']['bonferroni_sig_positive'])} 个。"]
    with open(os.path.join(out, "VERDICTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
