"""M2 追加诊断：两片不一致时，究竟有没有「选哪片」的信号。

回答三个问题：
  Q1 未接受的 91 条里，有多少条「某一片的簇身份就是真值」（决策层天花板），按组拆开。
  Q2 那批天花板样本上，各种「可选信号」能挑对多少：
     score / 簇规模 / 已标数 / 主体占比 / 分数×规模。
  Q3 把规则写成「取最高分且领先 margin ≥ M，且分数 ≥ T」，在 精度≥0.95 约束下
     能拿到多少 C —— 即「有没有可用工作点」。

用法：PY tools/diag_fusion_signal.py [--run results] [--cfg A_baseline]
输出：results/fusion_signal.csv、results/fusion_signal.json
"""
import argparse
import collections
import csv
import itertools
import json
import os

import eval_query_fusion as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results")
    ap.add_argument("--deliv", default="")
    ap.add_argument("--cfg", default="A_baseline")
    ap.add_argument("--theta", type=float, default=1.85)
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)
    gtruth = F.load_gallery_truth(gt)
    qtruth = F.load_query_truth(gt)
    ident = F.build_ident(run, a.cfg, gtruth)
    rows = F.rd(os.path.join(run, "queries", a.cfg + ".csv"))

    # 簇规模与纯度
    mem = collections.defaultdict(list)
    for r in F.rd(os.path.join(run, "members", a.cfg + ".csv")):
        mem[int(r["cluster_local"])].append(r["track_id"])
    size = {c: len(v) for c, v in mem.items()}
    nlab, share = {}, {}
    for c, v in mem.items():
        votes = collections.Counter(p for p in (gtruth.get(t) for t in v) if p)
        nlab[c] = sum(votes.values())
        share[c] = (votes.most_common(1)[0][1] / nlab[c]) if votes else 0.0

    # 分组
    groups = collections.Counter()
    cand = []          # 未接受且有机会的查询
    for r in rows:
        ca, cb = r["cluster_a_local"], r["cluster_b_local"]
        sa, sb = F.fnum(r["score_a"]), F.fnum(r["score_b"])
        aa, ab = r["accept_a"] == "1", r["accept_b"] == "1"
        if aa and ab and ca == cb:
            continue
        qp = qtruth.get(r["query_id"], "")
        if aa and ab:
            g = "两片都过·异簇"
        elif aa or ab:
            g = "仅一片过"
        else:
            g = "两片都不过"
        groups[g] += 1
        cset = sorted({int(x) for x in (ca, cb) if x != ""})
        ids = {c: ident.get(c, "") for c in cset}
        hit_c = [c for c in cset if qp and ids[c] == qp]
        cand.append(dict(query_id=r["query_id"], cam=r["cam"], group=g, q_pid=qp,
                         cset=cset, ids=ids, hit_c=hit_c, sa=sa, sb=sb,
                         aa=aa, ab=ab,
                         side_of_hit=(("a" if int(ca) in hit_c else "") +
                                      ("b" if int(cb) in hit_c else "")) if hit_c else ""))

    print("=== Q1 未接受 91 条的构成与天花板 ===")
    print("%-14s %-5s %-8s %-6s" % ("组", "条数", "有真值", "命中片"))
    tot = collections.Counter()
    for g in ("两片都过·异簇", "仅一片过", "两片都不过"):
        sub = [x for x in cand if x["group"] == g]
        nt = sum(1 for x in sub if x["q_pid"])
        nh = sum(1 for x in sub if x["hit_c"])
        print("%-14s %-5d %-8d %-6d" % (g, len(sub), nt, nh))
        tot["n"] += len(sub)
        tot["truth"] += nt
        tot["hit"] += nh
    print("%-14s %-5d %-8d %-6d" % ("合计", tot["n"], tot["truth"], tot["hit"]))

    # Q2 选片信号（只在有命中片的样本上评估：能不能挑中那一片）
    hs = [x for x in cand if x["hit_c"] and len(x["hit_c"]) == 1]
    print("\n=== Q2 在「有唯一正确片」的 %d 条上，各信号的选片准确率 ===" % len(hs))

    def chosen(fn):
        """fn(x, c) -> 可比数值；返回被选中的簇。"""
        ok = 0
        for x in hs:
            best, bv = None, None
            for c in x["cset"]:
                v = fn(x, c)
                if v is None:
                    continue
                if bv is None or v > bv:
                    best, bv = c, v
            if best is not None and best in x["hit_c"]:
                ok += 1
        return ok

    def by_score(x, c):
        return x["sa"] if x["cset"].index(c) == 0 else x["sb"]

    sigs = {
        "固定取 a 片": lambda x, c: 1.0 if x["cset"].index(c) == 0 else 0.0,
        "分数": by_score,
        "簇规模": lambda x, c: size.get(c, 0),
        "已标数": lambda x, c: nlab.get(c, 0),
        "主体占比": lambda x, c: share.get(c, 0),
        "分数×规模": lambda x, c: by_score(x, c) * size.get(c, 1),
        "分数×已标数": lambda x, c: by_score(x, c) * nlab.get(c, 1),
    }
    res = {}
    for name, fn in sigs.items():
        ok = chosen(fn)
        res[name] = ok
        print("  %-12s %2d/%2d = %.0f%%" % (name, ok, len(hs), 100 * ok / max(1, len(hs))))
    print("  随机基线 ~50%")

    # Q3 工作点搜索：取最高分，要求 ≥T 且领先 ≥M；看精度≥0.95 下最大 C
    print("\n=== Q3 「取最高分 + 阈值 T + 领先 M」在 精度≥0.95 下的最佳点 ===")
    best = []
    for T in [round(1.2 + 0.1 * i, 2) for i in range(31)]:
        for M in [round(0.05 * i, 2) for i in range(31)]:
            rule = "max_score_m"
            p = dict(theta=T, margin=M, qtruth=qtruth, ident=ident)
            s, _ = F.evaluate(rows, ident, qtruth, rule, p)
            if s["precision"] >= 0.95:
                best.append((s["C"], s["W"], s["precision"], T, M))
    best.sort(reverse=True)
    if best:
        seen = set()
        shown = 0
        for C, W, pr, T, M in best:
            if (C, W) in seen:
                continue
            seen.add((C, W))
            print("  C %d / W %d｜精度 %.4f｜T=%.2f M=%.2f｜C/229 %.3f"
                  % (C, W, pr, T, M, C / 229))
            shown += 1
            if shown >= 6:
                break
    else:
        print("  无可行工作点（任何 T/M 组合都达不到精度 ≥0.95）")

    # 分机位：未接受查询的分数分布，用于判断「按机位校准」有没有空间
    print("\n=== 分机位：未接受查询的分数与天花板 ===")
    print("%-4s %-6s %-8s %-8s %-8s" % ("cam", "未接受", "有真值", "天花板", "最高分中位"))
    for cam in sorted({x["cam"] for x in cand}):
        sub = [x for x in cand if x["cam"] == cam]
        nt = sum(1 for x in sub if x["q_pid"])
        nh = sum(1 for x in sub if x["hit_c"])
        mx = sorted(max(x["sa"], x["sb"]) for x in sub)
        med = mx[len(mx) // 2] if mx else 0
        print("%-4s %-6d %-8d %-8d %-8.2f" % (cam, len(sub), nt, nh, med))

    json.dump(dict(groups=dict(groups), ceiling=dict(tot), signals=res,
                   n_signal=len(hs), operating_points=best[:20]),
              open(os.path.join(run, "fusion_signal.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(run, "fusion_signal.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "cam", "group", "q_pid",
                                          "cset", "ids", "hit_c", "sa", "sb",
                                          "aa", "ab", "side_of_hit"])
        w.writeheader()
        for x in cand:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in x.items()})
    print("\n→ results/fusion_signal.{json,csv}")


if __name__ == "__main__":
    main()
