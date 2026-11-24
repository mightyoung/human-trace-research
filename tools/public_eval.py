#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公开数据集验证的判定脚本：把 d7_sweep + eval_precision 的产物算成 H1/H2/H3。

判定口径逐条对齐 datasets/VALIDATION.md §2（预注册，不得事后改）：

  H1a  Spearman(逐条 C , 继承精度·严格) ≤ −0.3
  H1b  继承精度最高的配置：ΔC(相对 A_baseline) < 0
  H1c  A_baseline 的 继承精度 < 逐条精度
  H2a  AUC(d_max → cluster_mixed) ≥ 0.80
  H2b  AUC(d_max → 该簇产生 ≥1 条错判) 的 bootstrap 95% CI 含 0.5，或点估计 ≤ 0.65
  H2c  ΔC(C_dmax0.30 − A_baseline) < 0 且 McNemar p < 0.05
  H3   未答 > 答错（描述性）

"继承精度·严格"：查询落入的簇
  - 只含 1 个名册身份 → 记对
  - 混人簇（≥2 身份）→ 记错
  - 无任何已标成员（非名册/非人）→ 记错
分母 = 已接受的查询（与现场 NUMBERS §11 一致）。

用法：
  PY tools/public_eval.py --pkg validation/mars
输出：
  <pkg>/results/h1_rank.csv, harm_auc.json, loss_decomposition.csv
  <pkg>/VERDICTS.md
"""
import argparse
import collections
import csv
import json
import os

import numpy as np


# ------------------------------------------------------------------ 读入

def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def load_gt(pkg):
    g = {}
    for r in rd(os.path.join(pkg, "gt", "gt_gallery_members_FILLED.csv")):
        pid = (r.get("person_id") or "").strip()
        if pid and pid.upper() != "UNK":
            g[r["track_id"]] = pid
    q = {}
    for r in rd(os.path.join(pkg, "gt", "gt_query_gt_FILLED.csv")):
        pid = (r.get("person_id") or "").strip()
        q[r["query_id"]] = "" if pid.upper() == "UNK" else pid
    return g, q


def load_emb(emb_dir, segs):
    """→ keys, V（图库特征，(a+b)/2 归一化后的均值向量）"""
    keys, va, vb = [], [], []
    for s in segs:
        za = np.load(os.path.join(emb_dir, f"{s}_emb_a.npz"))
        zb = np.load(os.path.join(emb_dir, f"{s}_emb_b.npz"))
        for k in za.files:
            if k not in zb.files:
                continue
            keys.append(k)
            va.append(za[k])
            vb.append(zb[k])
    A = np.asarray(va, np.float32)
    B = np.asarray(vb, np.float32)
    V = (A + B) / 2.0
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    return keys, V


# ------------------------------------------------------------------ 统计

def auc(score, label):
    """Mann-Whitney AUC；label 为 bool，True=正类"""
    score = np.asarray(score, float)
    y = np.asarray(label, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    pos, neg = score[y], score[~y]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def boot_auc(score, label, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    score = np.asarray(score, float)
    y = np.asarray(label, bool)
    idx = np.arange(len(y))
    out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0 or (~y[b]).sum() == 0:
            continue
        out.append(auc(score[b], y[b]))
    out = np.asarray([x for x in out if np.isfinite(x)])
    if out.size < 50:
        return [float("nan"), float("nan")]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def spearman(x, y):
    """秩相关（不依赖 scipy）"""
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # 并列取平均秩
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def mcnemar(b, c):
    """双侧精确 McNemar：b=基对变错，c=基错变对"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n) * 2.0
    return float(min(1.0, p))


# ------------------------------------------------------------------ 主流程

def cluster_info(members, gtruth):
    """cluster_local -> (主体身份, 已标数, 身份数, 主体占比)"""
    out = {}
    for cl, mem in members.items():
        votes = collections.Counter()
        for t in mem:
            p = gtruth.get(t)
            if p:
                votes[p] += 1
        n_lab = sum(votes.values())
        if votes:
            pid, n_top = votes.most_common(1)[0]
            out[cl] = (pid, n_lab, len(votes), n_top / n_lab)
        else:
            out[cl] = ("", 0, 0, 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--run", default="results")
    ap.add_argument("--base", default="A_baseline")
    ap.add_argument("--veto-cfg", default="C_dmax0.30")
    ap.add_argument("--emb-dir", default="", help="表征目录（默认 <pkg>/emb）")
    ap.add_argument("--dmax", type=float, default=0.30, help="H2 的 d_max 用哪一档做有害性判据（默认 0.30）")
    a = ap.parse_args()

    pkg = os.path.abspath(a.pkg)
    run = os.path.join(pkg, a.run)
    gtruth, qtruth = load_gt(pkg)
    meta = json.load(open(os.path.join(pkg, "run_info.json"), encoding="utf-8"))
    segs_tr = meta["subset"]["gallery"]
    keys, V = load_emb(a.emb_dir or os.path.join(pkg, "emb"), segs_tr)
    kidx = {k: i for i, k in enumerate(keys)}

    cfgs = sorted(f[:-4] for f in os.listdir(os.path.join(run, "members")) if f.endswith(".csv"))
    summ = {r["config"]: r for r in rd(os.path.join(run, "summary.csv"))}
    print(f"[in] 配置 {len(cfgs)}｜图库轨迹 {len(keys)}（有特征）｜"
          f"有身份 {len(gtruth)}｜查询有真值 {sum(1 for v in qtruth.values() if v)}")

    # ---------------- 逐配置：逐条 C/W/U + 继承精度（严格）
    per_cfg = {}
    for cfg in cfgs:
        mem = collections.defaultdict(list)
        for r in rd(os.path.join(run, "members", f"{cfg}.csv")):
            mem[int(r["cluster_local"])].append(r["track_id"])
        ci = cluster_info(mem, gtruth)
        rows = rd(os.path.join(run, "queries", f"{cfg}.csv"))
        C = W = U = acc = inh_ok = 0
        det = {}
        for r in rows:
            if r["final_status"] != "accepted":
                continue
            acc += 1
            qp = qtruth.get(r["query_id"], "")
            cl = int(r["final_cluster_local"])
            cp, n_lab, n_ids, share = ci.get(cl, ("", 0, 0, 0.0))
            if not qp or not cp:
                U += 1
                verdict = "U"
            elif qp == cp:
                C += 1
                verdict = "C"
            else:
                W += 1
                verdict = "W"
            # 继承精度：只看簇是否"唯一身份"，不看 q 的真值
            inh_ok += 1 if (n_ids == 1) else 0
            det[r["query_id"]] = dict(verdict=verdict, q_pid=qp, c_pid=cp,
                                      n_lab=n_lab, n_ids=n_ids, cluster=cl)
        per_cfg[cfg] = dict(C=C, W=W, U=U, acc=acc,
                            precision=C / (C + W) if C + W else float("nan"),
                            inherited=inh_ok / acc if acc else float("nan"),
                            detail=det, ci=ci, members=mem)
        s = summ.get(cfg, {})
        per_cfg[cfg].update(theta=s.get("theta", ""), K=s.get("K", ""),
                            cov=s.get("both_accept_rate", ""),
                            out_rate=s.get("final_output_rate", ""))

    # ---------------- H1
    h1 = [dict(config=c, C=per_cfg[c]["C"], W=per_cfg[c]["W"], U=per_cfg[c]["U"],
               accepted=per_cfg[c]["acc"],
               inherited=round(per_cfg[c]["inherited"], 4),
               precision=round(per_cfg[c]["precision"], 4),
               theta=per_cfg[c]["theta"], K=per_cfg[c]["K"])
          for c in cfgs]
    with open(os.path.join(run, "h1_rank.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h1[0].keys()))
        w.writeheader()
        w.writerows(h1)
    ok = [r for r in h1 if np.isfinite(r["inherited"]) and r["accepted"] > 0]
    rho = spearman([r["inherited"] for r in ok], [r["C"] for r in ok]) if len(ok) > 2 else float("nan")
    best_inh = max(ok, key=lambda r: r["inherited"]) if ok else None
    base = per_cfg.get(a.base)
    h1b_dc = (best_inh["C"] - base["C"]) if (best_inh and base) else None
    h1c = (base["inherited"], base["precision"]) if base else (float("nan"),) * 2

    # ---------------- H2：d_max 对混人 / 有害性
    dmax_of = {}
    for cl, mem in per_cfg[a.base]["members"].items():
        idx = [kidx[t] for t in mem if t in kidx]
        if len(idx) < 2:
            dmax_of[cl] = 0.0
            continue
        X = V[idx]
        S = X @ X.T
        iu = np.triu_indices(len(idx), 1)
        dmax_of[cl] = float(1.0 - S[iu].min())

    clusters = []
    for cl, (cp, n_lab, n_ids, share) in per_cfg[a.base]["ci"].items():
        cl_verdicts = [d["verdict"] for d in per_cfg[a.base]["detail"].values() if d["cluster"] == cl]
        clusters.append(dict(cluster=cl, d_max=round(dmax_of.get(cl, 0.0), 4),
                             n_lab=n_lab, n_ids=n_ids, mixed=int(n_ids >= 2),
                             n_query=len(cl_verdicts),
                             harmful=int(any(v == "W" for v in cl_verdicts)),
                             n_wrong=sum(1 for v in cl_verdicts if v == "W")))
    withq = [c for c in clusters if c["n_query"] > 0]
    auc_mixed = auc([c["d_max"] for c in clusters], [c["mixed"] for c in clusters])
    auc_harm_all = auc([c["d_max"] for c in clusters], [c["harmful"] for c in clusters])
    auc_harm_q = (auc([c["d_max"] for c in withq], [c["harmful"] for c in withq])
                  if len(withq) > 2 else float("nan"))
    ci_harm = boot_auc([c["d_max"] for c in withq], [c["harmful"] for c in withq]) if len(withq) > 2 else [float("nan")] * 2
    harm = dict(n_clusters=len(clusters), n_clusters_with_query=len(withq),
                n_mixed=sum(c["mixed"] for c in clusters),
                n_harmful=sum(c["harmful"] for c in withq),
                auc_mixed=round(auc_mixed, 4),
                auc_harmful_all=round(auc_harm_all, 4),
                auc_harmful_with_query=round(auc_harm_q, 4),
                auc_harmful_ci=([round(x, 4) for x in ci_harm]
                                if np.isfinite(ci_harm[0]) else None),
                clusters=clusters)
    json.dump(harm, open(os.path.join(run, "harm_auc.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---------------- H2c：veto 相对 baseline 的 ΔC + McNemar
    vet = per_cfg.get(a.veto_cfg)
    h2c = None
    if vet and base:
        # 配对必须在**全部查询**上做：某配置"没接受"就是"没答对"，
        # 不能用两个配置 detail 的交集（那会把 veto 丢掉的查询整批排除，p 假性=1）
        b = c = 0
        for q in qtruth:
            ok_b = base["detail"].get(q, {}).get("verdict") == "C"
            ok_v = vet["detail"].get(q, {}).get("verdict") == "C"
            if ok_b and not ok_v:
                b += 1
            elif ok_v and not ok_b:
                c += 1
        h2c = dict(delta_C=vet["C"] - base["C"], n_base=base["C"], n_veto=vet["C"],
                   base_only=b, veto_only=c, n_pairs=len(qtruth), p=mcnemar(b, c),
                   theta_base=base["theta"], theta_veto=vet["theta"])

    # ---------------- H3：损失结构
    rows = rd(os.path.join(run, "queries", f"{a.base}.csv"))
    both_fail = one_pass = ab_split = 0
    for r in rows:
        aa, bb = int(r["accept_a"]), int(r["accept_b"])
        if r["final_status"] == "accepted":
            continue
        if aa == 0 and bb == 0:
            both_fail += 1
        elif aa == 1 and bb == 1:
            ab_split += 1
        else:
            one_pass += 1
    loss = dict(n_query=len(rows), accepted=base["acc"], C=base["C"], W=base["W"], U=base["U"],
                unanswerable=len(rows) - base["acc"],
                both_fail=both_fail, one_pass=one_pass, ab_split=ab_split)
    with open(os.path.join(run, "loss_decomposition.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(loss.keys()))
        w.writeheader()
        w.writerow(loss)

    # ---------------- 判定表
    # 预注册门槛之外的前置条件：接受量太小 / 覆盖太低时，任何"成立/不成立"都是噪声。
    # 这一条在 CHIRLA 上先验触发（覆盖 3%、接受 2 条），必须显式判"无法判定"而不是硬判。
    base_cov = float(base["cov"]) if base and str(base["cov"]) not in ("", "None") else float("nan")
    feasible = bool(base and base["acc"] >= 30 and np.isfinite(base_cov) and base_cov >= 0.10)
    if feasible:
        infeasible_reason = ""
    else:
        why = []
        if not base or base["acc"] < 30:
            why.append(f"接受 {base['acc'] if base else 0} 条 < 30")
        if not (np.isfinite(base_cov) and base_cov >= 0.10):
            why.append(f"覆盖 {base_cov:.3f} < 0.10")
        infeasible_reason = ("前置条件不满足：" + "、".join(why) +
                             "，任何成立/不成立都是个位数样本的噪声 → 除 H3 外一律判无法判定")

    def verdict(cond):
        if not feasible:
            return "无法判定"
        return "成立" if cond else "不成立"

    V = []
    V.append(("H1a", "Spearman(逐条C, 继承精度·严格)", f"{rho:.3f}", "ρ ≤ −0.3",
              verdict(np.isfinite(rho) and rho <= -0.3)))
    V.append(("H1b", "继承精度最高配置的 ΔC", f"{h1b_dc:+d}（{best_inh['config']}）" if h1b_dc is not None else "n/a",
              "ΔC < 0", verdict(h1b_dc is not None and h1b_dc < 0)))
    V.append(("H1c", "A_baseline 继承精度 vs 逐条精度",
              f"{h1c[0]:.3f} vs {h1c[1]:.3f}" if base else "n/a",
              "继承 < 逐条", verdict(base is not None and h1c[0] < h1c[1])))
    V.append(("H2a", "AUC(d_max → 混人簇)", f"{auc_mixed:.3f}",
              "AUC ≥ 0.80", verdict(np.isfinite(auc_mixed) and auc_mixed >= 0.80)))
    h2b_ok = (np.isfinite(auc_harm_q) and auc_harm_q <= 0.65) or (
        ci_harm and np.isfinite(ci_harm[0]) and ci_harm[0] <= 0.5 <= ci_harm[1])
    V.append(("H2b", "AUC(d_max → 有害簇) [95%CI]",
              f"{auc_harm_q:.3f} [{ci_harm[0]:.3f},{ci_harm[1]:.3f}]" if np.isfinite(ci_harm[0]) else f"{auc_harm_q:.3f}",
              "CI 含 0.5 或 AUC ≤ 0.65", verdict(bool(h2b_ok))))
    V.append(("H2c", f"ΔC({a.veto_cfg} − A_baseline) 与 McNemar p",
              f"{h2c['delta_C']:+d}, p={h2c['p']:.4g}" if h2c else "n/a",
              "ΔC < 0 且 p < 0.05", verdict(bool(h2c and h2c["delta_C"] < 0 and h2c["p"] < 0.05))))
    V.append(("H3", "未答 vs 答错", f"{loss['unanswerable']} vs {base['W']}",
              "未答 > 答错", verdict(loss["unanswerable"] > base["W"])))

    lines = [f"# 公开数据验证判定（{meta['dataset']}）", "",
             f"- 图库 {meta['counts']['gallery_tracks']} 条（有身份 "
             f"{meta['counts'].get('gallery_labeled', '?')}）｜查询 "
             f"{meta['counts']['query_tracks']} 条｜配置 {len(cfgs)} 个",
             f"- A_baseline：K={base['K']} θ={base['theta']} 接受 {base['acc']} "
             f"C={base['C']} W={base['W']} U={base['U']}", "",
             "| 假设 | 指标 | 实测 | 判定门槛 | 结论 |", "|---|---|---|---|---|"]
    for v in V:
        lines.append("| " + " | ".join(v) + " |")
    if not feasible:
        lines += ["", f"> ⚠ {infeasible_reason}", ""]
    lines += ["", "## H2 簇级明细", "",
              f"- 簇 {harm['n_clusters']} 个（有查询 {harm['n_clusters_with_query']}）｜混人 {harm['n_mixed']}｜有害 {harm['n_harmful']}",
              f"- AUC 混人 {harm['auc_mixed']:.3f}｜有害（全部簇）{harm['auc_harmful_all']:.3f}"
              f"｜有害（有查询簇）{harm['auc_harmful_with_query']:.3f}", "",
              "## H3 损失分解", "",
              f"- 可回答 {base['acc']}：答对 {base['C']} / 答错 {base['W']} / 无法归属 {base['U']}",
              f"- 未答 {loss['unanswerable']}：双片未过 {loss['both_fail']} / 单过 {loss['one_pass']} / 双过异簇 {loss['ab_split']}",
              "", "## 口径与偏离", ""]
    for d in meta.get("deviations", []):
        lines.append(f"- {d}")
    open(os.path.join(pkg, "VERDICTS.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("\n".join(lines[:14]))
    print(f"\n[out] {os.path.join(pkg, 'VERDICTS.md')}")


if __name__ == "__main__":
    main()
