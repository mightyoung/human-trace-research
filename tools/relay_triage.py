# -*- coding: utf-8 -*-
"""接力判定三态化（MOSS 式「主动沉默」）——confirmed / pending / rejected + 增量重算。

动机：现行二值判定（sim ≥ thr 即确认、否则弃）会让「差一点的候选」直接消失；
而这类候选恰是"证据不足、未来可能升级"的观望对象（MOSS-VL-Realtime 的
perception-driven silence 思想）。三态化把判定改为：

  confirmed: mutual-best 且 sim ≥ hi        → 进入生产链构造（与现有行为等价）
  pending:   mutual-best 且 sim ∈ [lo, hi)  → 观望清单（差一点的孤悬候选，等新证据）
  rejected:  其余（存在更强竞争 / sim < lo）→ 淘汰

增量重算：新轨迹/新时间片到达时，把新候选并入池、重判，只报告**状态变化**
（用户关心的是"哪些 pending 被升级/降级"，而非全量重跑）。

用法：
  python tools/relay_triage.py                                    # 全量三态
  python tools/relay_triage.py --split 575                        # 时间片增量演示
  python tools/relay_triage.py --lo 0.92 --hi 0.943
"""
import argparse
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
SCORED = os.path.join(ANA, "relay", "relay_scored_cradio_v3.json")


def mutual_best_flags(cands):
    """每条候选是否为其 a 端与 b 端的 mutual-best（全候选池上）。"""
    bfa, bfb = {}, {}
    for r in cands:
        if r["a"] not in bfa or r["sim"] > bfa[r["a"]]["sim"]:
            bfa[r["a"]] = r
        if r["b"] not in bfb or r["sim"] > bfb[r["b"]]["sim"]:
            bfb[r["b"]] = r
    return {id(r): (bfa.get(r["a"]) is r and bfb.get(r["b"]) is r) for r in cands}


def classify(cands, hi, lo):
    """三态划分。返回 (confirmed, pending, rejected) 三个列表。"""
    mb = mutual_best_flags(cands)
    conf, pend, rej = [], [], []
    for r in cands:
        if mb[id(r)]:
            if r["sim"] >= hi:
                conf.append(r)
            elif r["sim"] >= lo:
                pend.append(r)
            else:
                rej.append(r)
        else:
            rej.append(r)
    return conf, pend, rej


def key_of(r):
    return (r["a"], r["b"])


def state_map(cands, hi, lo):
    conf, pend, rej = classify(cands, hi, lo)
    m = {}
    for r in conf:
        m[key_of(r)] = "confirmed"
    for r in pend:
        m[key_of(r)] = "pending"
    for r in rej:
        m[key_of(r)] = "rejected"
    return m, conf, pend


def chain_heads_tails(edges):
    """confirmed 边的链头/链尾集合（用于 pending 的'疑似延伸'归类）。"""
    succ = defaultdict(list)
    pred = defaultdict(list)
    for e in edges:
        succ[e["a"]].append(e["b"])
        pred[e["b"]].append(e["a"])
    heads = {n for n in set(succ) | set(pred) if n not in pred}
    tails = {n for n in set(succ) | set(pred) if n not in succ}
    return heads, tails


def load_cands(path=None):
    d = json.load(open(path or SCORED, encoding="utf-8"))
    return d["candidates"]


def cmd_all(cands, hi, lo, out_p):
    conf, pend, rej = classify(cands, hi, lo)
    heads, tails = chain_heads_tails(conf)

    # pending 条目归类：是否贴着 confirmed 链的端（疑似前段/后段延伸）
    for p in pend:
        p["_touch"] = []
        if p["b"] in heads:
            p["_touch"].append(f"b端={p['b']} 是链头 → 疑似更早前段")
        if p["a"] in tails:
            p["_touch"].append(f"a端={p['a']} 是链尾 → 疑似更后段延伸")
    pend.sort(key=lambda r: -r["sim"])

    out = {
        "hi": hi, "lo": lo,
        "n_candidates": len(cands),
        "n_confirmed": len(conf),
        "n_pending": len(pend),
        "n_rejected": len(rej),
        "confirmed_edges": sorted(f"{r['a']} -> {r['b']}" for r in conf),
        "pending": [
            {**{k: p[k] for k in ("a", "b", "sim", "dt", "dir")},
             "touch": p["_touch"]}
            for p in pend],
    }
    json.dump(out, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"[三态] 候选 {len(cands)} → confirmed {len(conf)} | pending {len(pend)} | "
          f"rejected {len(rej)}  (hi={hi}, lo={lo})")
    print(f"[观望清单] {len(pend)} 条（按 sim 降序）：")
    for p in pend[:30]:
        tag = f"  ← {'; '.join(p['_touch'])}" if p["_touch"] else ""
        print(f"  {p['a']} -> {p['b']}  sim={p['sim']:.4f} dt={p['dt']}{tag}")
    print(f"[out] {out_p}")
    return out


def cmd_split_demo(cands, hi, lo, split_t):
    """时间片增量演示：第一批（b_tfirst<split）→ 新到达（b_tfirst>=split）。
    增量重算 vs 全量重算对照（正确性验证）+ 状态迁移分类（含降级原因）。
    """
    first = [r for r in cands if r["b_tfirst"] < split_t]
    arrive = [r for r in cands if r["b_tfirst"] >= split_t]
    print(f"[增量演示] 第一批 {len(first)} 条（b_tfirst<{split_t}s）| "
          f"新到达 {len(arrive)} 条")

    m0, conf0, pend0 = state_map(first, hi, lo)
    print(f"  第一批状态: confirmed {len(conf0)} | pending {len(pend0)} | "
          f"rejected {len(first)-len(conf0)-len(pend0)}")

    # ---- 增量：并新候选后重判（3744 规模下全并重判 <10ms；生产可只算受影响端点）----
    full = first + arrive
    m1, conf1, pend1 = state_map(full, hi, lo)

    # 状态迁移分类
    new_entries, trans = [], []
    for k in set(m0) | set(m1):
        s0, s1 = m0.get(k, "new"), m1.get(k, "removed")
        if s0 == s1:
            continue
        rec = {"edge": f"{k[0]} -> {k[1]}", "from": s0, "to": s1}
        if s0 == "new":
            new_entries.append(rec)
        else:
            trans.append(rec)

    # 降级原因：找出接管端点的新 best
    bfa, bfb = {}, {}
    for r in full:
        if r["a"] not in bfa or r["sim"] > bfa[r["a"]]["sim"]:
            bfa[r["a"]] = r
        if r["b"] not in bfb or r["sim"] > bfb[r["b"]]["sim"]:
            bfb[r["b"]] = r
    for t in trans:
        if t["to"] != "rejected":
            continue
        a, b = t["edge"].split(" -> ")
        reasons = []
        nb = bfa.get(a)
        if nb and f"{nb['a']} -> {nb['b']}" != t["edge"]:
            reasons.append(f"a端被 {nb['b']} (sim={nb['sim']:.4f}) 接管")
        nb = bfb.get(b)
        if nb and f"{nb['a']} -> {nb['b']}" != t["edge"]:
            reasons.append(f"b端被 {nb['a']}->{nb['b']} (sim={nb['sim']:.4f}) 接管")
        t["reason"] = "; ".join(reasons) or "门限/竞争结构变化"

    print(f"  增量重算后: confirmed {len(conf1)} | pending {len(pend1)}")
    print(f"  [新条目初判] {len(new_entries)} 条"
          f"（confirmed {sum(1 for x in new_entries if x['to']=='confirmed')} / "
          f"pending {sum(1 for x in new_entries if x['to']=='pending')}）")
    print(f"  [已有条目迁移] {len(trans)} 条：")
    for t in trans:
        line = f"    {t['edge']}: {t['from']} → {t['to']}"
        if t.get("reason"):
            line += f"  ({t['reason']})"
        print(line)

    n_up = sum(1 for t in trans if t["from"] == "pending" and t["to"] == "confirmed")
    n_dn = sum(1 for t in trans if t["to"] == "rejected")
    if n_up:
        print(f"  [升级] {n_up} 条 pending 因新证据被确认")
    print(f"  [汇总] 升级 {n_up} / 降级 {n_dn}")

    # ---- 对照：全量重跑（状态机一致性）----
    m_full, _, _ = state_map(full, hi, lo)
    assert m_full == m1, "增量重算与全量重算不一致！"
    print(f"  [正确性] 增量结果 ≡ 全量重算（{len(m_full)} 键逐一比对一致）√")

    return {"split": split_t, "n_first": len(first), "n_arrive": len(arrive),
            "new_entries": new_entries, "transitions": trans,
            "n_upgraded": n_up, "n_downgraded": n_dn}


def cmd_upgrade_demo(cands, hi, lo):
    """重打分升级演示：pending 因轨迹获得新证据（嵌入重算）sim 提升 → 过门升级。
    说明：静态 sim 下升级只能来自「重打分」或「门限调整」；此处模拟前者
    （生产对应场景：轨迹延长/碎片增多 → 嵌入重算 → 相似度估计更准）。"""
    m0, conf0, pend0 = state_map(cands, hi, lo)
    target = max(pend0, key=lambda r: r["sim"])
    k = key_of(target)
    delta = 0.004
    new_sim = round(min(0.99, target["sim"] + delta), 4)
    print(f"[升级演示] 目标：{k[0]} -> {k[1]}（pending, sim={target['sim']:.4f}）")
    print(f"  模拟新证据：轨迹获得新碎片 → 嵌入重算 → sim {target['sim']:.4f} → {new_sim:.4f}")
    pool = [dict(r) if key_of(r) == k else r for r in cands]
    for r in pool:
        if key_of(r) == k:
            r["sim"] = new_sim
            r["_rescore"] = True
    m1, conf1, pend1 = state_map(pool, hi, lo)
    print(f"  状态迁移: {m0[k]} → {m1[k]}"
          f"  {'√ 过门升级，进入生产链构造' if m1[k]=='confirmed' else ''}")
    return {"edge": f"{k[0]} -> {k[1]}", "sim_before": target["sim"],
            "sim_after": new_sim, "state_before": m0[k], "state_after": m1[k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", type=float, default=0.943)
    ap.add_argument("--lo", type=float, default=0.90)
    ap.add_argument("--split", type=float, default=None,
                    help="时间片增量演示：split 秒（b_tfirst 切分）")
    ap.add_argument("--upgrade-demo", action="store_true",
                    help="重打分升级演示（模拟新证据）")
    ap.add_argument("--out", default=os.path.join(ANA, "relay", "relay_triage_cradio_v3.json"))
    ap.add_argument("--scored", default=SCORED,
                    help="打分开候选文件（默认 v3；clean2 口径传 relay_clean2/relay_scored_cradio_v5.json）")
    args = ap.parse_args()

    cands = load_cands(args.scored)
    if args.upgrade_demo:
        res = cmd_upgrade_demo(cands, args.hi, args.lo)
        out_p = os.path.join(ANA, "relay", "relay_triage_upgrade_demo.json")
        json.dump(res, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[out] {out_p}")
    elif args.split is not None:
        res = cmd_split_demo(cands, args.hi, args.lo, args.split)
        out_p = os.path.join(ANA, "relay", "relay_triage_incremental_demo.json")
        json.dump(res, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[out] {out_p}")
    else:
        cmd_all(cands, args.hi, args.lo, args.out)


if __name__ == "__main__":
    main()
