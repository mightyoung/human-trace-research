# -*- coding: utf-8 -*-
"""本地复算 /api/truth/metrics 的数学（不依赖 fastapi/远程环境）。"""
import json, csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
st = json.load(open(os.path.join(ROOT, "webapp/backend/data/truth/state.json"), encoding="utf-8"))
pairs, N = st["pairs"], len(st["pairs"])
labeled = os.path.join(ROOT, "output_crosscam/analysis/truth/pair_candidates_labeled.csv")
labels = {}
for r in csv.DictReader(open(labeled, encoding="utf-8-sig")):
    v = (r.get("human_same_person") or "").strip()
    if v in ("1", "0", "?"):
        labels[r["pair_id"]] = {"v": 1 if v == "1" else (0 if v == "0" else 2)}

sel = st["meta"]["select"]
K_t, K_a = sel["k_topk"], sel["k_audit"]
budget = K_t + K_a
L = lambda p: (labels.get(p["pair_id"]) or {}).get("v")  # noqa: E731

pos = [p for p in pairs if L(p) == 1]
neg = [p for p in pairs if L(p) == 0]
hit = tot = 0.0
for a in pos:
    for b in neg:
        tot += 1
        hit += 1.0 if a["pred"] > b["pred"] else (0.5 if a["pred"] == b["pred"] else 0.0)
print("排序准确率(AUC)=", round(hit / tot, 4) if tot else None, " (%d pos x %d neg)" % (len(pos), len(neg)))

se = n = 0
for p in pairs:
    v = L(p)
    if v in (0, 1):
        pp = 1.0 - (p["rank"] - 0.5) / N
        se += (pp - v) ** 2
        n += 1
print("Brier=", round(se / n, 4), " n=", n)

strata = {}
for name in ("topk", "audit", "rest"):
    rows = [p for p in pairs if p["role"] == name]
    a = [p for p in rows if L(p) in (0, 1)]
    s = sum(1 for p in a if L(p) == 1)
    rate = (s / len(a)) if a else None
    strata[name] = {"size": len(rows), "annotated": len(a), "same": s, "rate": rate}
    print("  %-6s size=%3d ann=%2d same=%d rate=%s" % (name, len(rows), len(a), s, rate))

ok = all(strata[k]["rate"] is not None for k in strata)
est = sum(strata[k]["rate"] * strata[k]["size"] for k in strata) if ok else None
prev = (est / N) if est else None
print("估计同人总数=", round(est, 1) if est else None, " prevalence=", round(prev, 4) if prev else None)
arm = strata["topk"]["same"] + strata["audit"]["same"]
rnd = budget * prev if prev else None
print("同预算(%d)命中=%d  随机期望=%s  效率=%s"
      % (budget, arm, round(rnd, 2) if rnd else None, round(arm / rnd, 2) if rnd else None))
r_ann = [p for p in pairs if p["relay"] == 1 and L(p) in (0, 1)]
print("relay v5 精度=", sum(1 for p in r_ann if L(p) == 1), "/", len(r_ann))
print("NOTE: 基于当前 %d 条人工判定的部分结果（rest 层无判定 → 估计暂不可用）" % len(labels))
