"""簇级诊断：d_max 预测「混人」与预测「实际错判」的能力是否一致。"""
import collections
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
RUN = os.path.join(DELIV, "results")


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


geo = {}
for r in rd(os.path.join(DELIV, "ledger", "clusters_geom.csv")):
    if r["config"] == "baseline":
        geo[int(r["cluster_idx"])] = r

lab2gid = {}
for gid, r in geo.items():
    lab2gid[int(r["cluster_lab"])] = gid

loc2gid = {}
for r in rd(os.path.join(RUN, "members", "A_baseline.csv")):
    loc2gid[int(r["cluster_local"])] = lab2gid[int(r["cluster_lab"])]

v = rd(os.path.join(RUN, "verdicts", "A_baseline.csv"))
per = collections.defaultdict(lambda: {"C": 0, "W": 0, "U": 0})
info = {}
for r in v:
    gid = loc2gid[int(r["cluster_local"])]
    per[gid][r["verdict"]] += 1
    if gid not in info:
        info[gid] = r

rows = []
for gid, s in per.items():
    i = info[gid]
    rows.append(dict(gid=gid, dmax=float(geo[gid]["d_max"]),
                     n_ids=int(i["cluster_n_ids"]), n_lab=int(i["cluster_n_lab"]),
                     mixed=int(i["cluster_n_ids"]) >= 2,
                     served=s["C"] + s["W"] + s["U"],
                     C=s["C"], W=s["W"], harmed=s["W"] > 0))


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


print(f"有查询的簇 {len(rows)} 个 | 混人 {sum(1 for r in rows if r['mixed'])} "
      f"| 产生过错判 {sum(1 for r in rows if r['harmed'])}")
print()
print("=== 同一个 d_max，预测两个不同问题 ===")
a1 = auc([r["dmax"] for r in rows if r["mixed"]],
         [r["dmax"] for r in rows if not r["mixed"]])
a2 = auc([r["dmax"] for r in rows if r["harmed"]],
         [r["dmax"] for r in rows if not r["harmed"]])
mm = [r for r in rows if r["mixed"]]
a3 = auc([r["dmax"] for r in mm if r["harmed"]],
         [r["dmax"] for r in mm if not r["harmed"]])
print(f"  簇是否混人            AUC = {a1:.3f}")
print(f"  簇是否产生错判        AUC = {a2:.3f}")
print(f"  混人簇内部有害/无害   AUC = {a3:.3f} "
      f"(有害 {sum(1 for r in mm if r['harmed'])} / 无害 {sum(1 for r in mm if not r['harmed'])})")
print()
print("=== 混人簇的有害性 ===")
print(f"{'簇':>4}{'d_max':>7}{'身份':>5}{'已标':>5}{'服务':>5}{'C':>4}{'W':>4}  有害")
for r in sorted(mm, key=lambda x: -x["dmax"]):
    print(f"{r['gid']:>4}{r['dmax']:>7.3f}{r['n_ids']:>5}{r['n_lab']:>5}{r['served']:>5}"
          f"{r['C']:>4}{r['W']:>4}  {'是' if r['harmed'] else '否'}")
pure = [r for r in rows if not r["mixed"]]
print()
print("混人簇：%d 个，平均服务 %.1f 条，共错 %d 条" % (
    len(mm), statistics.mean([r["served"] for r in mm]), sum(r["W"] for r in mm)))
print("纯簇：  %d 个，平均服务 %.1f 条，共错 %d 条" % (
    len(pure), statistics.mean([r["served"] for r in pure]), sum(r["W"] for r in pure)))

# d_max 与「错判条数」的关系
print()
print("=== 按 d_max 分档看实际错判（错误率分母＝已判定 C+W，不含 U）===")
print(f"{'档':<12}{'簇数':>4}{'已判定':>6}{'错判':>5}{'错误率':>7}")
for lo, hi in [(-1, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 9)]:
    sub = [r for r in rows if lo <= r["dmax"] < hi]
    if not sub:
        continue
    s = sum(r["C"] + r["W"] for r in sub)
    w = sum(r["W"] for r in sub)
    print(f"{f'[{lo:.2f},{hi:.2f})':<12}{len(sub):>4}{s:>6}{w:>5}{(w/s if s else 0):>7.3f}")
