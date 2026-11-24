# -*- coding: utf-8 -*-
"""把人工判定写入真值表，并做一致性/冲突检查 + relay v5 首版 precision/recall。

人工判定来源：用户 2026-10-14 对 pair_sheet_1/2/3 的逐对判读。
一次性辅助脚本，可复跑（幂等）。
"""
import csv, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = os.path.join(ROOT, "output_crosscam", "analysis", "truth")
REV = os.path.join(TRUTH, "_review")


def pid(a, b):
    return "343_E_343-%s__316_E_316-%s" % (a, b)


# 1=同人 0=不是 ?=无法判断
LABELS = {
    # ---- sheet 1 ----
    pid("E042", "E053"): ("1", "high", "sheet1"),
    pid("E046", "E061"): ("1", "high", "sheet1"),
    pid("E045", "E054"): ("1", "high", "sheet1"),
    pid("E010", "E014"): ("1", "high", "sheet1"),
    pid("E037", "E040"): ("?", "", "sheet1"),
    pid("E011", "E013"): ("1", "high", "sheet1"),
    # sheet1 第 5 行 E011↔E012 用户未提 → 记为未判
    # ---- sheet 2 ----
    pid("E042", "E046"): ("1", "high", "sheet2"),
    pid("E042", "E048"): ("1", "high", "sheet2"),
    pid("E037", "E044"): ("?", "", "sheet2"),
    pid("E038", "E043"): ("1", "high", "sheet2"),
    pid("E042", "E050"): ("0", "high", "sheet2"),
    pid("E009", "E025"): ("?", "", "sheet2"),
    pid("E042", "E049"): ("0", "high", "sheet2"),
    # ---- sheet 3 ----
    pid("E003", "E009"): ("?", "", "sheet3"),
    pid("E015", "E027"): ("?", "", "sheet3"),
    pid("E030", "E040"): ("?", "", "sheet3"),
    pid("E031", "E040"): ("?", "", "sheet3"),
    pid("E024", "E029"): ("?", "", "sheet3"),
    pid("E010", "E015"): ("1", "high", "sheet3"),
}

rows = list(csv.DictReader(open(os.path.join(REV, "pair_machine.csv"), encoding="utf-8-sig")))
by_pid = {r["pair_id"]: r for r in rows}
miss = [k for k in LABELS if k not in by_pid]
print("标签数", len(LABELS), "未在候选表中的:", miss)

for r in rows:
    lab = LABELS.get(r["pair_id"])
    if lab:
        r["human_same_person"], r["human_confidence"] = lab[0], lab[1]
        r["human_note"] = ""   # 来源标记不入界面
    else:
        r.setdefault("human_same_person", "")
        r.setdefault("human_confidence", "")
        r.setdefault("human_note", "")
out_cols = list(rows[0].keys())
with open(os.path.join(TRUTH, "pair_candidates_labeled.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=out_cols)
    w.writeheader()
    w.writerows(rows)

c = collections.Counter(r["human_same_person"] for r in rows)
print("判定分布（1同人/0异人/?无法判断/空未判）:", dict(c))

# ---- 冲突检测：同一事件被判"同人"到多个对端 ----
def comp(edges):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    g = collections.defaultdict(list)
    for x in parent:
        g[find(x)].append(x)
    return list(g.values())


same = [(r["event_a"], r["event_b"]) for r in rows if r["human_same_person"] == "1"]
print("\n=== 判为'同人'的边 (%d) ===" % len(same))
for a, b in same:
    print("  ", a.split("_E_")[1], "|", b.split("_E_")[1])

print("\n=== 连通分量检查（>1 的任一侧 = 冲突/碎片）===")
conflict = []
for grp in comp(same):
    A = sorted(x.split("_E_")[1] for x in grp if x.startswith("343"))
    B = sorted(x.split("_E_")[1] for x in grp if x.startswith("316"))
    flag = "  <-- 冲突" if (len(A) > 1 and len(B) > 1) else ""
    if len(A) > 1 or len(B) > 1:
        conflict.append((A, B))
    print("  343%s x 316%s%s" % (A, B, flag))

# ---- relay v5 首版 precision/recall（在已判样本上）----
relay = [r for r in rows if r["in_relay_v5"] == "1"]
tp = [r for r in relay if r["human_same_person"] == "1"]
fp = [r for r in relay if r["human_same_person"] == "0"]
un = [r for r in relay if r["human_same_person"] == "?"]
fn = [r for r in rows if r["in_relay_v5"] == "0" and r["human_same_person"] == "1"]
print("\n=== relay v5 首版（仅已判样本）===")
print("  relay 命中 %d：同人 %d / 异人 %d / 无法判断 %d" % (len(relay), len(tp), len(fp), len(un)))
print("  relay 未命中但确为同人（漏接）%d：%s" % (len(fn), [r["event_a"].split("_E_")[1] + "|" + r["event_b"].split("_E_")[1] for r in fn]))
print("  → 命中率(recall) = %d/%d = %.2f ；已判精确率(precision) = %d/%d = %.2f"
      % (len(tp), len(tp) + len(fn), len(tp) / max(1, len(tp) + len(fn)), len(tp), len(tp) + len(fp), len(tp) / max(1, len(tp) + len(fp))))

# ---- 外观 cos 能否分离（已判样本）----
print("\n=== 外观 cos 的分离能力（已判样本）===")
for lab, name in (("1", "同人"), ("0", "异人")):
    v = sorted(float(r["cos_appear"]) for r in rows if r["human_same_person"] == lab)
    print("  %s: %s" % (name, v))
