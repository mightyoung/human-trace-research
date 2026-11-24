"""M2 决策层融合评测：在不改聚类的前提下，只改「两片怎么合并成一条答案」。

输入：某次运行的 queries/<cfg>.csv（含 score_a/score_b/accept_a/accept_b/cluster_a/cluster_b）
      members/<cfg>.csv + gt（逐成员真值）→ 簇主体身份
输出：results/query_fusion.csv、results/query_fusion.json

口径与 tools/eval_precision.py 完全一致：correct = 簇主体身份 == query 真值；
簇无已标成员或 query 无真值 → U，不进 C/W。C 分母固定 229。

用法：
  PY tools/eval_query_fusion.py [--run results] [--deliv ...] [--cfg A_baseline]
"""
import argparse
import collections
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
C_DENOM = 229


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def load_gallery_truth(gt_dir):
    out = {}
    for r in rd(os.path.join(gt_dir, "gt_gallery_members_FILLED.csv")):
        pid = (r["person_id"] or "").strip()
        if pid and pid.upper() != "UNK":
            out[r["track_id"]] = pid
    return out


def load_query_truth(gt_dir):
    out = {}
    for r in rd(os.path.join(gt_dir, "gt_query_gt_FILLED.csv")):
        pid = (r["person_id"] or "").strip()
        out[r["query_id"]] = "" if pid.upper() == "UNK" else pid
    return out


def is_nonperson(pid):
    """no-human / shadow* = 影子·镜像·物件。它们不是人，不进 C/W/U，也不算可答。"""
    p = (pid or "").strip().lower()
    return p == "no-human" or p.startswith("shadow")


def identity_values(gtruth):
    """图库里可作答的身份集合（剔除 no-human / shadow*）。"""
    return {p for p in gtruth.values() if p and not is_nonperson(p)}


def cluster_ident(members, gtruth):
    votes = collections.Counter()
    for t in members:
        p = gtruth.get(t)
        if p:
            votes[p] += 1
    if not votes:
        return ""
    return votes.most_common(1)[0][0]


def build_ident(run, cfg, gtruth, with_share=False):
    m = collections.defaultdict(list)
    p = os.path.join(run, "members", cfg + ".csv")
    for r in rd(p):
        m[int(r["cluster_local"])].append(r["track_id"])
    ident = {cl: cluster_ident(mem, gtruth) for cl, mem in m.items()}
    if not with_share:
        return ident
    share = {}
    for cl, mem in m.items():
        votes = collections.Counter(p for p in (gtruth.get(t) for t in mem) if p)
        n = sum(votes.values())
        share[cl] = (votes.most_common(1)[0][1] / n) if n else 0.0
    return ident, share


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def decide(r, rule, p):
    """返回 (accepted: bool, cluster: int|None, tag: str)"""
    ca, cb = r["cluster_a_local"], r["cluster_b_local"]
    sa, sb = fnum(r["score_a"]), fnum(r["score_b"])
    aa, ab = r["accept_a"] == "1", r["accept_b"] == "1"
    same = ca == cb and ca != ""
    if rule == "base":
        return (aa and ab and same, int(ca) if same else None, "base")
    if rule == "single_same":
        # 基线 + 只有一片过、但两片同簇
        if aa and ab and same:
            return True, int(ca), "both"
        if same and (aa or ab):
            return True, int(ca), "single"
        return False, None, "reject"
    if rule == "single_any":
        if aa and ab and same:
            return True, int(ca), "both"
        if aa and not ab:
            return True, int(ca), "a"
        if ab and not aa:
            return True, int(cb), "b"
        return False, None, "reject"
    if rule == "single_any_m":
        # 单侧接受，但要求过的那侧分数 ≥ θ + 余量
        m = p.get("margin", 0.25)
        if aa and ab and same:
            return True, int(ca), "both"
        if aa and not ab and sa >= p["theta"] + m:
            return True, int(ca), "a"
        if ab and not aa and sb >= p["theta"] + m:
            return True, int(cb), "b"
        return False, None, "reject"
    if rule == "both_diff_hi":
        if aa and ab and same:
            return True, int(ca), "both"
        if aa and ab and not same:
            return True, (int(ca) if sa >= sb else int(cb)), "pick"
        return False, None, "reject"
    if rule == "both_diff_hi_m":
        m = p.get("margin", 0.25)
        if aa and ab and same:
            return True, int(ca), "both"
        if aa and ab and not same:
            if abs(sa - sb) < m:
                return False, None, "tie"
            return True, (int(ca) if sa > sb else int(cb)), "pick"
        return False, None, "reject"
    if rule == "max_score":
        # 不管 a/b 是否过，取两侧最高分那片，只要 ≥ θ
        if same and max(sa, sb) >= p["theta"]:
            return True, int(ca), "both"
        if max(sa, sb) < p["theta"]:
            return False, None, "low"
        return True, (int(ca) if sa >= sb else int(cb)), "max"
    if rule == "max_score_m":
        m = p.get("margin", 0.25)
        if max(sa, sb) < p["theta"]:
            return False, None, "low"
        if same:
            return True, int(ca), "both"
        if abs(sa - sb) < m:
            return False, None, "tie"
        return True, (int(ca) if sa > sb else int(cb)), "max"
    if rule == "a_first":
        # 信任前半片（query 的定义就取自轨迹起点，a 片更接近被查的人）
        if aa and ab and same:
            return True, int(ca), "both"
        if aa:
            return True, int(ca), "a"
        return False, None, "reject"
    if rule == "a_first_m":
        m = p.get("margin", 0.0)
        if aa and ab and same:
            return True, int(ca), "both"
        if aa and (ab or sa >= p["theta"] + m):
            return True, int(ca), "a"
        return False, None, "reject"
    if rule == "share_pick":
        # 单侧过 → 接；两片都过且异簇 → 取主体占比更高的簇
        sh = p.get("share", {})
        if aa and ab and same:
            return True, int(ca), "both"
        if aa and ab:
            c = ca if sh.get(int(ca), 0) >= sh.get(int(cb), 0) else cb
            return True, int(c), "share"
        if aa:
            return True, int(ca), "a"
        if ab:
            return True, int(cb), "b"
        return False, None, "reject"
    if rule == "a_first_share":
        # 单侧过 → 接；两片都过异簇 → a 的纯度不低于 b 才取 a，否则取 b
        sh = p.get("share", {})
        if aa and ab and same:
            return True, int(ca), "both"
        if aa and ab:
            c = ca if sh.get(int(ca), 0) >= sh.get(int(cb), 0) * 1.0 else cb
            return True, int(c), "share"
        if aa:
            return True, int(ca), "a"
        return False, None, "reject"
    if rule == "oracle":
        # 上限：非基线接受里，任一片簇身份==真值就接受（离线标定，仅作天花板）
        qp = p["qtruth"].get(r["query_id"], "")
        if aa and ab and same:
            return True, int(ca), "both"
        for c in ({ca, cb} - {""}):
            if p["ident"].get(int(c), "") == qp and qp:
                return True, int(c), "oracle"
        return False, None, "miss"
    raise ValueError(rule)


def evaluate(rows, ident, qtruth, rule, p):
    C = W = U = 0
    detail = []
    for r in rows:
        if r.get("final_status") == "rejected_nonperson":
            continue          # 已被非人门控拦掉，任何决策规则都不再考虑
        ok, cl, tag = decide(r, rule, p)
        if not ok:
            continue
        qp = qtruth.get(r["query_id"], "")
        if is_nonperson(qp):
            continue          # 非人轨迹：既不判对也不判错，也不计未答
        cp = ident.get(cl, "") if cl is not None else ""
        if not qp or not cp:
            U += 1
            v = "U"
        elif qp == cp:
            C += 1
            v = "C"
        else:
            W += 1
            v = "W"
        detail.append(dict(query_id=r["query_id"], cam=r["cam"], rule=rule, tag=tag,
                           cluster=cl, q_pid=qp, c_pid=cp, verdict=v,
                           score_a=r["score_a"], score_b=r["score_b"],
                           accept_a=r["accept_a"], accept_b=r["accept_b"]))
    acc = C + W + U
    return dict(rule=rule, accepted=acc, C=C, W=W, U=U,
                precision=round(C / (C + W), 4) if C + W else 0.0,
                c_over_229=round(C / C_DENOM, 4)), detail


def mcnemar(a, b):
    """逐条 C/W 翻转的精确二项检验（只比较两边都给出 C/W 的查询）。"""
    from math import comb
    b10 = b01 = 0
    for q in set(a) & set(b):
        x, y = a[q], b[q]
        if x == "C" and y == "W":
            b10 += 1
        elif x == "W" and y == "C":
            b01 += 1
    n = b10 + b01
    if not n:
        return dict(c2w=b10, w2c=b01, n=0, p=1.0)
    k = min(b10, b01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return dict(c2w=b10, w2c=b01, n=n, p=round(p, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results")
    ap.add_argument("--deliv", default="")
    ap.add_argument("--gt-dir", default="")
    ap.add_argument("--cfg", default="A_baseline")
    ap.add_argument("--theta", type=float, default=1.85)
    ap.add_argument("--out-prefix", default="query_fusion")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else DELIV
    gt = os.path.abspath(a.gt_dir) if a.gt_dir else os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)
    gtruth = load_gallery_truth(gt)
    qtruth = load_query_truth(gt)
    ident, share = build_ident(run, a.cfg, gtruth, with_share=True)
    rows = rd(os.path.join(run, "queries", a.cfg + ".csv"))
    print("查询 %d｜簇 %d｜图库真值 %d" % (len(rows), len(ident), len(gtruth)))
    print("θ = %.2f（基线 a/b 双侧接受阈）" % a.theta)

    base_p = dict(theta=a.theta, qtruth=qtruth, ident=ident, share=share)
    rules = [("base", {}), ("single_same", {}), ("single_any", {}),
             ("a_first", {}), ("a_first_m", dict(margin=0.0)),
             ("a_first_m", dict(margin=0.25)),
             ("share_pick", {}), ("a_first_share", {}),
             ("single_any_m", dict(margin=0.25)), ("single_any_m", dict(margin=0.50)),
             ("both_diff_hi", {}), ("both_diff_hi_m", dict(margin=0.25)),
             ("both_diff_hi_m", dict(margin=0.50)),
             ("max_score", {}), ("max_score_m", dict(margin=0.25)),
             ("max_score_m", dict(margin=0.50)), ("oracle", {})]
    out, details = [], {}
    print("\n%-16s %-6s %-4s %-4s %-4s %-8s %-8s %s"
          % ("规则", "接受", "C", "W", "U", "精度", "C/229", "增量"))
    base_c = None
    for rule, extra in rules:
        p = dict(base_p, **extra)
        s, det = evaluate(rows, ident, qtruth, rule, p)
        name = rule + ("(%s)" % extra["margin"] if "margin" in extra else "")
        s["rule"] = name
        if rule == "base":
            base_c = s["C"]
        s["delta_C"] = s["C"] - (base_c or 0)
        print("%-16s %-6d %-4d %-4d %-4d %-8.4f %-8.4f %+d"
              % (name, s["accepted"], s["C"], s["W"], s["U"], s["precision"],
                 s["c_over_229"], s["delta_C"]))
        out.append(s)
        details[name] = det

    # McNemar：最好的可行规则 vs 基线
    base_det = {d["query_id"]: d["verdict"] for d in details["base"]}
    print("\nMcNemar（vs 基线，仅 C/W 翻转）：")
    for s in out:
        if s["rule"] == "base":
            continue
        d = {x["query_id"]: x["verdict"] for x in details[s["rule"]]}
        t = mcnemar(base_det, d)
        s.update(mcnemar=t)
        print("  %-16s C→W %d｜W→C %d｜p = %.3f" % (s["rule"], t["c2w"], t["w2c"], t["p"]))

    # 决策层天花板：非基线接受的查询里，任一片簇身份命中的比例
    cei = collections.Counter()
    for r in rows:
        ca, cb = r["cluster_a_local"], r["cluster_b_local"]
        aa, ab = r["accept_a"] == "1", r["accept_b"] == "1"
        if aa and ab and ca == cb:
            continue
        qp = qtruth.get(r["query_id"], "")
        cei["n"] += 1
        if not qp:
            cei["no_truth"] += 1
        hit = any(ident.get(int(c), "") == qp for c in ({ca, cb} - {""})) if qp else False
        cei["hit" if hit else "miss"] += 1
    print("\n决策层天花板（%d 条未接受中）：任一片簇身份==真值 %d 条 → 上限 C/229 = %.3f"
          % (cei["n"], cei["hit"], (base_c + cei["hit"]) / C_DENOM))

    json.dump(dict(run=run, cfg=a.cfg, theta=a.theta, rows=out,
                   ceiling=dict(base_c=base_c, **cei)),
              open(os.path.join(run, a.out_prefix + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(run, a.out_prefix + "_detail.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = None
        for name, det in details.items():
            for d in det:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(d.keys()))
                    w.writeheader()
                w.writerow(d)
    print("→ %s" % os.path.join(run, a.out_prefix + ".json"))


if __name__ == "__main__":
    main()
