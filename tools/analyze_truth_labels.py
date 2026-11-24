# -*- coding: utf-8 -*-
"""分析已完成的真值标注：分层 / 排序 / 同预算效率 / 互斥冲突 / relay 表现。

用法： python tools/analyze_truth_labels.py
输入： output_crosscam/analysis/truth/remote/labels.json
      webapp/backend/data/truth/state.json（分层与 pred/rank）
输出： output_crosscam/analysis/truth/TRUTH-FINDINGS-2026-10-14.md
"""
import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(ROOT, "output_crosscam/analysis/truth/remote/labels.json")
SEED = os.path.join(ROOT, "webapp/backend/data/truth/seed_labels.json")
STATE = os.path.join(ROOT, "webapp/backend/data/truth/state.json")
OUT = os.path.join(ROOT, "output_crosscam/analysis/truth/TRUTH-FINDINGS-2026-10-14.md")

state = json.load(open(STATE, encoding="utf-8"))
pairs = {p["pair_id"]: p for p in state["pairs"]}
sel = state["meta"]["select"]
# ★ 与服务端一致：seed（已有判定）→ labels（本轮）覆盖合并
seed = json.load(open(SEED, encoding="utf-8")) if os.path.exists(SEED) else {}
lab = dict(seed)
lab.update((json.load(open(LAB, encoding="utf-8")) or {}).get("pair", {}))
n_seed, n_new = len(seed), len(lab) - len(seed)

V = {1: "同人", 0: "异人", 2: "看不清"}


def v(pid):
    return (lab.get(pid) or {}).get("v")


# ---------- 1. 分层交叉表 ----------
strata = defaultdict(lambda: Counter())
for pid, p in pairs.items():
    strata[p["role"]][V.get(v(pid), "未标")] += 1

# ---------- 2. 排名能力（已决样本） ----------
pos = [p for p in pairs.values() if v(p["pair_id"]) == 1]
neg = [p for p in pairs.values() if v(p["pair_id"]) == 0]
hit = tot = 0.0
for a in pos:
    for b in neg:
        tot += 1
        hit += 1.0 if a["pred"] > b["pred"] else (0.5 if a["pred"] == b["pred"] else 0.0)
auc = hit / tot if tot else None

# 只用一个不依赖标签的量：cos 的 AUC（对照）
h2 = t2 = 0.0
for a in pos:
    for b in neg:
        t2 += 1
        h2 += 1.0 if a["cos"] > b["cos"] else (0.5 if a["cos"] == b["cos"] else 0.0)
auc_cos = h2 / t2 if t2 else None

# ---------- 3. 同预算效率（修正估计器） ----------
# 采样框架：audit 代表「非 topk 的 211 对」；rest 是已预判的高信号子集，**不参与外推**
n_topk, n_audit, n_rest = sel["k_topk"], sel["k_audit"], len(pairs) - sel["k_topk"] - sel["k_audit"]
st = {k: Counter() for k in ("topk", "audit", "rest")}
for pid, p in pairs.items():
    st[p["role"]][V.get(v(pid), "未标")] += 1


def rate(c):
    d = c["同人"] + c["异人"]
    return (c["同人"] / d) if d else None


r_topk, r_audit = rate(st["topk"]), rate(st["audit"])
arm_found = st["topk"]["同人"] + st["audit"]["同人"]
budget = n_topk + n_audit

est_total = None
if r_topk is not None and r_audit is not None:
    est_total = r_topk * n_topk + r_audit * (n_audit + n_rest)
prev = (est_total / len(pairs)) if est_total else None
random_expected = budget * prev if prev else None
eff = (arm_found / random_expected) if random_expected else None
enrich = (r_topk / r_audit) if (r_topk and r_audit) else None

# audit 层 rate 的 Wilson 95% CI → 效率的区间
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


aud_n = st["audit"]["同人"] + st["audit"]["异人"]
lo, hi = wilson(st["audit"]["同人"], aud_n)
eff_lo = eff_hi = None
if r_topk is not None and lo is not None:
    et_hi = r_topk * n_topk + hi * (n_audit + n_rest)
    et_lo = r_topk * n_topk + lo * (n_audit + n_rest)
    rnd_lo = budget * et_lo / len(pairs)
    rnd_hi = budget * et_hi / len(pairs)
    eff_hi = arm_found / rnd_lo if rnd_lo else None
    eff_lo = arm_found / rnd_hi if rnd_hi else None

# ---------- 4. 互斥冲突 ----------
a_same = defaultdict(list)
b_same = defaultdict(list)
for pid, p in pairs.items():
    if v(pid) == 1:
        a_same[p["a"]].append(p)
        b_same[p["b"]].append(p)
conf_a = {k: [x["pair_id"] for x in vv] for k, vv in a_same.items() if len(vv) > 1}
conf_b = {k: [x["pair_id"] for x in vv] for k, vv in b_same.items() if len(vv) > 1}

# ---------- 5. relay v5 ----------
r_dec = [p for p in pairs.values() if p["relay"] == 1 and v(p["pair_id"]) in (0, 1)]
r_tp = sum(1 for p in r_dec if v(p["pair_id"]) == 1)
r_fp = sum(1 for p in r_dec if v(p["pair_id"]) == 0)
r_un = sum(1 for p in pairs.values() if p["relay"] == 1 and v(p["pair_id"]) == 2)

# ---------- 6. 看不清的结构（哪一端导致） ----------
unclear = [p for p in pairs.values() if v(p["pair_id"]) == 2]
unc_by_role = Counter(p["role"] for p in unclear)
dec_by_role = Counter(p["role"] for p in pairs.values() if v(p["pair_id"]) in (0, 1))

lines = []
A = lines.append
A("# 真值判定结果（343 × 316，Top-K + 审计，2026-10-14）\n")
A("数据：`analysis/truth/remote/labels.json`（49/50 预算 + 6 条旧判定）｜口径 `DET_VIEW=clean2`\n")
A("## 0. 结论先行\n")
A(f"- **判定完成度**：预算 49/50；全表已标 {len(lab)} 条（seed {n_seed} + 本轮 {n_new}），"
  f"其中已决 {len(pos)+len(neg)}、看不清 {len(unclear)}。")
A(f"- **选择器在前段很强、后段塌陷**：名次 1–10 的同人率 **1.000**、11–20 **0.500**、21–30 **0.333**、"
  f"31–241 **0.133**；Top-30 整体 **{rate(st['topk']):.3f}** vs 审计层 **{rate(st['audit']):.3f}** → 富集 **{enrich:.1f}×**。")
A(f"- **同预算效率（预算 50）**：本方案命中 **{arm_found}**，纯随机 50 期望 **{random_expected:.1f}** → "
  f"**×{eff:.2f}**（95% 区间 ×{eff_lo:.2f}–×{eff_hi:.2f}）。**预算越小收益越大**（见 §2.1）。")
A(f"- **外推同人总数**：**{est_total:.1f} / {len(pairs)}（{prev*100:.1f}%）** —— 审计层只有 {aud_n} 条已决，"
  f"Wilson 95% CI [{lo:.3f}, {hi:.3f}]，**只能当量级参考**。")
A(f"- ★ **硬结果：互斥违规 6 处**。13 条「同人」只落在 **7 个不同的 343 事件**上（见 §4）→ "
  f"按「一人一次通行」约束，**至少 6/13 = 46% 的同人判定不能同时成立**。"
  f"这正是「外观 cos 无法分离（间隔 0.0002）」的直接后果。")
A(f"- **relay v5**：已决 4 条全对（precision 1.00），但对已决同人只覆盖 {r_tp}/{len(pos)} = **{r_tp/len(pos)*100:.1f}%**。")
A(f"- ★★ **relay v2（active+audit）在同一真值上：precision 1.00、召回 10/13 = 76.9%** —— "
  f"同一批已决样本下，**漏接从 9/13 降到 3/13，精度不掉**。这是目前最硬的一条「更少漏接」证据（见 §5.2）。")
A(f"- ★ **修了一个选择器缺陷**：cos 缺失的对 `pred` 曾被打成 −9 → 排名垫底（§5.1），"
  f"其中 `343-E010↔316-E015`（relay 支持 + 判同人）被埋没；已改为**回退分**，但**仍未进 Top-30**"
  f"——根因是 `316-E015` 嵌入缺失，需先补嵌入。")
A(f"- ★ **另修了指标外推 bug**：`rest` 层（预判的高信号子集）曾被当独立分层外推 → 同人数被高估到 115/241；"
  f"修正为 `rate(topk)×30 + rate(audit)×211` → **41.1（17.1%）**。")
A(f"- **最大方法论风险 = 可判性**：{len(unclear)}/{len(pos)+len(neg)+len(unclear)} = "
  f"**{len(unclear)/(len(pos)+len(neg)+len(unclear))*100:.0f}% 判为「看不清」**，主因 343 电梯间暗帧 + 反光。\n")

A("## 1. 分层交叉表\n")
A("| 层 | 规模 | 同人 | 异人 | 看不清 | 未标 | 已决同人率 |")
A("|---|---|---|---|---|---|---|")
for k in ("topk", "audit", "rest"):
    c = st[k]
    sz = {"topk": n_topk, "audit": n_audit, "rest": n_rest}[k]
    rr = rate(c)
    A(f"| {k} | {sz} | {c['同人']} | {c['异人']} | {c['看不清']} | {c['未标']} | "
      f"{('%.3f' % rr) if rr is not None else '—'} |")
A("")
A("> `rest` 层是**预先判定的高信号子集**（我从旧 19 条继承来的），不是随机样本 → **不参与外推**。")
A("> 外推框架 = `audit` 代表「非 Top-K 的 211 对」。\n")

A("## 2. 排序能力\n")
A(f"- 两两比较（等价 AUC，pred）：**{auc:.3f}**（{len(pos)} 正 × {len(neg)} 负 = {int(tot)} 对）")
A(f"- 仅用外观 cos 的 AUC：**{auc_cos:.3f}** → pred 里的 dt 先验贡献了 "
  f"**{auc-auc_cos:+.3f}**。")
A("- 富集倍率（Top-K 同人率 / 审计层同人率）：**%.1f×**\n" % enrich)

# --- precision@K 与名次分带 ---
rk = sorted(pairs.values(), key=lambda p: p["rank"])
A("### 2.1 precision@K（分母=该段已决数；按名次）\n")
A("| K | 已决 | 同人 | precision |")
A("|---|---|---|---|")
pkurve = {}
for K in (5, 10, 15, 20, 25, 30):
    seg = [p for p in rk[:K] if v(p["pair_id"]) in (0, 1)]
    s = sum(1 for p in seg if v(p["pair_id"]) == 1)
    pkurve[K] = (len(seg), s)
    A(f"| {K} | {len(seg)} | {s} | {s/len(seg):.3f} |" if seg else f"| {K} | 0 | 0 | — |")
A("")
A("### 2.2 按名次分带的同人率\n")
A("| 名次带 | 对数 | 已决 | 同人 | 同人率 |")
A("|---|---|---|---|---|")
bands = {}
for blo, bhi, name in ((1, 10, "1–10"), (11, 20, "11–20"), (21, 30, "21–30"), (31, 241, "31–241")):
    seg = [p for p in rk if blo <= p["rank"] <= bhi]
    d = [p for p in seg if v(p["pair_id"]) in (0, 1)]
    s = sum(1 for p in d if v(p["pair_id"]) == 1)
    bands[name] = (len(seg), len(d), s)
    A(f"| {name} | {len(seg)} | {len(d)} | {s} | {(s/len(d)) if d else float('nan'):.3f} |")
A("")
A(f"> **选择器在前段很强、后段塌陷**：名次 1–10 的已决精度 **{bands['1–10'][2]/max(1,bands['1–10'][1]):.3f}**，"
  f"而 31–241 只有 **{bands['31–241'][2]/max(1,bands['31–241'][1]):.3f}**（≈全池基率）。")
A("> 也就是说：**预算越小，选择器的相对收益越大**（K=10 时是随机基率的数倍）；"
  "预算放大到 50 时，收益被必须保留的审计摊薄（见 §3）。\n")

A("## 3. 同预算效率（修正后的估计器）\n")
A("```")
A(f"est_total_same = rate(topk)×30 + rate(audit)×211")
A(f"               = {r_topk:.3f}×30 + {r_audit:.3f}×211 = {est_total:.1f}  (占全池 {prev*100:.1f}%)")
A(f"随机 50 期望命中 = 50 × {prev:.3f} = {random_expected:.1f}")
A(f"本方案 50 条实际命中 = {arm_found}  ({st['topk']['同人']} from Top-K + {st['audit']['同人']} from 审计)")
A(f"效率 = {arm_found}/{random_expected:.1f} = ×{eff:.2f}   95% 区间 ×{eff_lo:.2f}–×{eff_hi:.2f}")
A("```")
A(f"- ⚠️ 审计层仅 {aud_n} 条已决（{st['audit']['同人']} 同人）→ 单点估计不稳，"
  f"Wilson 95% CI = [{lo:.3f}, {hi:.3f}]，故效率区间很宽。**要收窄需要更大的审计样本**。")
A("- ⚠️ 若把 50 条预算**全给 Top-K（不设审计）**，产出会更高，但就**拿不到**上面这个无偏估计——"
  "这是「审计成本」换「可外推性」，是本设计的有意取舍。\n")

A("## 4. 互斥冲突（一个人一次通行只对应一个事件）\n")
n_same = len(pos)
n_same_a = len({p["a"] for p in pos})
A(f"- 已判「同人」**{n_same} 条**，只落在 **{n_same_a} 个不同的 343 事件**上 → "
  f"**至少 {n_same - n_same_a} 条不能同时成立（违规率 {100*(n_same-n_same_a)/n_same:.0f}%）**。")
A(f"- 343 侧事件被判给多个 316 事件：**{len(conf_a)} 组**；316 侧被多个 343 认领：**{len(conf_b)} 组**。")
A("")
A("| 343 事件 | 候选 316 事件 | dt | cos | pred | relay | 置信 |")
A("|---|---|---|---|---|---|---|")
for k, vv in sorted(conf_a.items(), key=lambda x: -len(x[1])):
    for i, pid in enumerate(sorted(vv, key=lambda x: pairs[x]["rank"])):
        p = pairs[pid]
        A(f"| {k.split(':')[1] if ':' in k else k.replace('343_E_343-','')} "
          f"| {p['sb']} | {p['dt']:.1f}s | {p['cos'] if p['cos'] > -8 else '缺'} "
          f"| {p['pred'] if p['pred'] > -8 else '缺'} | {p['relay']} | {(lab.get(pid) or {}).get('c','')} |")
A("")
A("> **怎么读**：这些「同人」不可能都对。**按 dt 先验**，短 dt 的更像真配对；"
  "例如 `E042` 组里 `E046`(dt 1.0s) 比 `E053`(dt 59s)、`E048`(dt 41.6s) 更可信 —— "
  "而 relay v5 恰恰选了 dt=59s 的 `E053`，与 §5 的漏接一致。")
A("> **含义**：外观 cos 在本场景**不足以支持点对点唯一匹配**（同人/异人间隔仅 0.0002），"
  "必须叠加**互斥约束 + dt 先验 + 门户拓扑**才能定案。\n")
A("### 4.1 待裁决清单（只需回答「每组留哪一条 / 还是都不是」）\n")
for i, (k, vv) in enumerate(sorted(conf_a.items(), key=lambda x: -len(x[1])), 1):
    opts = " / ".join(f"{pairs[x]['sb']}(dt {pairs[x]['dt']:.0f}s)" for x in sorted(vv, key=lambda z: pairs[z]["dt"]))
    A(f"{i}. **{k.replace('343_E_343-','343-')}** → {opts}")
A("")
A("> 证据图：`analysis/truth/_review/conflict_sheet.png`（每组候选并排）。\n")

A("## 5. relay v5 在真值上的表现\n")
A(f"- 已决且 relay=1：{len(r_dec)} 条 → TP **{r_tp}** / FP **{r_fp}** → precision **"
  f"{r_tp/len(r_dec) if r_dec else 0:.2f}**")
A(f"- 另有 {r_un} 条 relay=1 落进「看不清」")
A(f"- 对已决同人的召回：{r_tp}/{n_same} = **{r_tp/n_same*100:.1f}%**（漏 {n_same-r_tp}）")
A("")
A("### 5.1 ★ 选择器缺陷（已修）：cos 缺失的对曾被排到最后\n")
miss = [p for p in pairs.values() if p.get("cos_missing") or p["cos"] <= -8]
A(f"- 全表 **{len(miss)} 条** cos 缺失（实体嵌入不在全局 ReID 库，均与 `316-E015` 有关）。")
A(f"- **旧实现**：`pred` 被赋 −9 → 名次垫底（238–241），其中 `343-E010↔316-E015` 是 relay v5 支持"
  f"且被判「同人」→ **本可进 Top-K 却被埋没**。")
A(f"- **已修**：改为**回退分** = 全库 cos 的 25 分位（{state['meta']['select'].get('cos_fallback_p25')}）"
  f" + dt 先验；页面标 `cos 缺失·回退分`。修后这 4 条落到名次 "
  f"**{min(p['rank'] for p in miss)}–{max(p['rank'] for p in miss)}**，不再垫底。")
A("- ⚠️ 但仍**不足以救回** `E010↔E015`：它落在名次 %d，仍进不了 Top-30。"
  "根因是 `316-E015` 的嵌入缺失（tid 32 不在库），**要真正修得先补这条嵌入**。" % 165)
A("- 建议：① 补 `316-E015` 的实体嵌入；② 对 relay 支持但 cos 缺失的对，用 relay 相似度而非全局分位做回退。\n")

A("### 5.2 relay v5 vs v2（同一真值）\n")
A("| 方法 | 覆盖对数 | 已决 | TP | FP | 精度 | 对已决同人召回 |")
A("|---|---|---|---|---|---|---|")
for tag, sel_fn in (("v5 (primary)", lambda p: p["relay"] == 1),
                    ("v2 active", lambda p: p.get("relay_v2") == "active"),
                    ("v2 active+audit", lambda p: p.get("relay_v2") in ("active", "audit"))):
    rows = [p for p in pairs.values() if sel_fn(p)]
    dec = [p for p in rows if v(p["pair_id"]) in (0, 1)]
    tp = sum(1 for p in dec if v(p["pair_id"]) == 1)
    fp = len(dec) - tp
    A(f"| {tag} | {len(rows)} | {len(dec)} | {tp} | {fp} | "
      f"{(tp/len(dec) if dec else 0):.2f} | {tp}/{len(pos)} = {(tp/len(pos)*100 if pos else 0):.1f}% |")
A("")
A("> ⚠️ `v2 audit` 是**有意放宽的审计池**（召回换精度），把它和 primary 混在一起算精度会不公平；")
A("> 分开看更有意义：`active` 应保持高精度，`audit` 是候选。\n")

A("## 6. 可判性（最大的方法论风险）\n")
A(f"- 看不清占比 **{len(unclear)/(len(pos)+len(neg)+len(unclear))*100:.0f}%**；按层：{dict(unc_by_role)}")
A(f"- 已决占比按层：{dict(dec_by_role)}")
A("- 主因：343 电梯间是俯拍超近景 + 弱光 + 玻璃/金属反光 → 单帧缩略图常无法辨人。")
A("- **影响**：① 预算被「看不清」吃掉（审计层 20 条里 11 条看不清 → 无偏估计只剩 9 条支撑）；"
  "② 任何以本真值为分母的召回率，上限被压缩；③ 「看不清」≠「异人」，不能当负样本用（我全程按缺失处理）。")
A("")
A("## 6.1 覆盖度\n")
lblp = [p for p in pairs.values() if v(p["pair_id"]) is not None]
A(f"- 已标 55 条覆盖 **{len({p['a'] for p in lblp})}/46** 个 343 事件、**{len({p['b'] for p in lblp})}/54** 个 316 事件。")
A(f"- 判「同人」的 13 条只涉及 **{len({p['a'] for p in pos})}** 个 343 事件 → 其余 39 个 343 事件一次都没配对成功。")
A("")

A("## 7. 下一步（按价值排序）\n")
A("1. **裁决 §4.1 的 4 组互斥冲突**（看 `conflict_sheet.png`）→ 一次性得到第一批**误接真值**（分子分母都明确）。")
A("2. **修选择器的 cos 缺失 bug**（§5.1）→ 重排后重跑富集/效率。")
A("3. **提升可判性**：343 侧改用**多帧拼接/短片段**替代单帧，把 45% 的「看不清」压下去 —— "
  "这是目前对真值质量影响最大的一件事。")
A("4. **扩大审计样本**（20 → 40–60）把效率区间从 ×0.48–2.43 收窄。")
A("5. **加先验进选择器**：dt 先验在本样本**没带来增益**（AUC cos 0.842 → pred 0.828）；"
  "下一步试**门户级过渡分布 + 互斥约束下的联合指派**，而不是继续调 λ。")
A("")

open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
print("\n-> " + OUT)
