"""在新真值下重测：分数能否区分对/错（旧结论 AUC 0.571 基于簇级继承标签）。"""
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
RUN = os.path.join(DELIV, "results")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


verd = {r["query_id"]: r for r in rd(os.path.join(RUN, "verdicts", "A_baseline.csv"))}
q = {r["query_id"]: r for r in rd(os.path.join(RUN, "queries", "A_baseline.csv"))}


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


C, W = [], []
for qid, v in verd.items():
    if v["verdict"] not in ("C", "W"):
        continue
    s = min(float(q[qid]["score_a"]), float(q[qid]["score_b"]))
    (C if v["verdict"] == "C" else W).append(s)

print(f"对 {len(C)} 条 / 错 {len(W)} 条")
import statistics
print(f"分数（两片取小）中位：对 {statistics.median(C):.3f}｜错 {statistics.median(W):.3f}")
print(f"AUC（分数区分对/错）= {auc(C, W):.3f}")
print()

# 按分数分档看错误率
print("=== 按 minimum score 分档 ===")
print(f"{'档':<14}{'条数':>5}{'错':>4}{'错误率':>7}")
pairs = sorted([(min(float(q[i]['score_a']), float(q[i]['score_b'])),
                 verd[i]['verdict']) for i in verd], key=lambda x: x[0])
for lo, hi in [(0, 1.5), (1.5, 1.8), (1.8, 2.0), (2.0, 2.2), (2.2, 2.4), (2.4, 99)]:
    sub = [(s, v) for s, v in pairs if lo <= s < hi]
    if not sub:
        continue
    w = sum(1 for _, v in sub if v == "W")
    print(f"{f'[{lo:.1f},{hi:.1f})':<14}{len(sub):>5}{w:>4}{w/len(sub):>7.3f}")
print()
# 若按分数拒识能到多少
print("=== 若在 baseline 上再加一道分数门（只保留 min score ≥ T）===")
print(f"{'T':>5}{'保留':>5}{'对':>4}{'错':>4}{'精度':>7}{'C/229':>7}")
for T in [1.5, 1.7, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]:
    keep = [(s, v) for s, v in pairs if s >= T]
    c = sum(1 for _, v in keep if v == "C")
    w = sum(1 for _, v in keep if v == "W")
    u = sum(1 for _, v in keep if v == "U")
    print(f"{T:>5.1f}{len(keep):>5}{c:>4}{w:>4}{(c/(c+w) if c+w else float('nan')):>7.3f}{c/229:>7.3f}")
