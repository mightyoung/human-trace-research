#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F3：handover 评估协议（Relay v2 / S3）

内容（docs/relay-v2-plan.md §5 F3）：
  ① 边级 FN/FP（active / active+audit 两口径，对接人工判定 CSV；只读）
  ② crossing / returning 近似分类（Ristani ECCV'16 交接缺陷的边级近似）
  ③ 覆盖率重算（c0 口径：v5 复现自证 + v2 active+audit 试算 = c0 v6 预览）
  ④ EM−ES：列出公式与口径差距（需帧级真值，暂缓硬算）
  ⑤ top-10 待复核清单 + 存疑对裁决就绪表（供人工审核页逐条过）

用法:
  python tools/handover_metrics.py
输出:
  output_crosscam/analysis/relay_v2/metrics_v1.json + 控制台可读表
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
# 口径纪律：v5/v2 边集均为 clean2 口径 → 脚点/实体同用 clean2（对齐 c0_baseline_v5）
os.environ.setdefault("DET_VIEW", "clean2")
import numpy as np  # noqa: E402
import c0_baseline as c0  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
RV2 = os.path.join(ANA, "relay_v2")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
PAIR_CSV = os.path.join(ANA, "truth", "pair_candidates.csv")
EDGES_V2 = os.path.join(RV2, "edges_v2.json")
V5_CHAINS = os.path.join(ANA, "relay_clean2", "relay_chains_cradio_v5.json")
TRIAGE_V5 = os.path.join(ANA, "relay_clean2", "relay_triage_cradio_v5.json")
OUT = os.path.join(RV2, "metrics_v1.json")

# dt 主峰带（依据：确认边 p25=1.2s / p75=15.5s；min=-0.6s 允许微负交错；
# 上界取 primary dt p75，在 F4 中动态计算复验）
DT_BAND = (-2.0, 15.5)


def parse_pid(pid):
    """'343_E_343-E042__316_E_316-E053' -> ('343:343-E042','316:316-E053')"""
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def load_edges(path):
    d = json.load(open(path, encoding="utf-8"))
    edges = []
    for e in d["edges"]:
        edges.append({"a": e["a"], "b": e["b"], "dt": e.get("dt"), "sim": e.get("sim"),
                      "status": e.get("status", "active"),
                      "edge_type": e.get("edge_type", "primary"),
                      "tier": e.get("tier", "main")})
    return d, edges


def coverage_for(edges, all_ents):
    """与 c0_baseline.main 覆盖率段同构（同口径重算）。"""
    has_out = {e["a"] for e in edges}
    has_in = {e["b"] for e in edges}
    coverage = {}
    for cam in c0.CAMS:
        _, ex_mask, en_mask = c0.classify_endpoints(cam, all_ents[cam])
        mov_eids = list(ex_mask.keys())
        n_ex_gated = sum(ex_mask.values())
        n_en_gated = sum(en_mask.values())
        ex_cov = sum(1 for eid in mov_eids if ex_mask[eid] and f"{cam}:{eid}" in has_out)
        en_cov = sum(1 for eid in mov_eids if en_mask[eid] and f"{cam}:{eid}" in has_in)
        coverage[cam] = dict(
            n_exit_gated=n_ex_gated, n_exit_covered=ex_cov,
            exit_continuity=round(ex_cov / n_ex_gated, 3) if n_ex_gated else None,
            n_entry_gated=n_en_gated, n_entry_covered=en_cov,
            entry_attribution=round(en_cov / max(n_en_gated, 1), 3) if n_en_gated else None)
    tot_ex = sum(c["n_exit_gated"] for c in coverage.values())
    tot_ex_cov = sum(c["n_exit_covered"] for c in coverage.values())
    tot_en = sum(c["n_entry_gated"] for c in coverage.values())
    tot_en_cov = sum(c["n_entry_covered"] for c in coverage.values())
    return {"per_cam": coverage,
            "global": dict(n_exit_gated=tot_ex, n_exit_covered=tot_ex_cov,
                           exit_continuity=round(tot_ex_cov / max(tot_ex, 1), 3),
                           n_entry_gated=tot_en, n_entry_covered=tot_en_cov,
                           entry_attribution=round(tot_en_cov / max(tot_en, 1), 3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=EDGES_V2)
    ap.add_argument("--labeled", default=LABELED)
    ap.add_argument("--v5", default=V5_CHAINS)
    ap.add_argument("--triage", default=TRIAGE_V5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-coverage", action="store_true",
                    help="跳过覆盖率重算（跳过 stitch_clean2 实体加载，秒级完成）")
    args = ap.parse_args()

    # 判定合并：labeled.csv（早期）∪ labels.json（最新，优先）；dt 由 pair_candidates.csv 补全
    judged = {}
    if os.path.exists(args.labeled):
        for r in csv.DictReader(open(args.labeled, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v in ("1", "0", "?"):
                judged[r["pair_id"]] = v
    if os.path.exists(LABELS_JSON):
        d = json.load(open(LABELS_JSON, encoding="utf-8"))
        for pid, rec in d.get("pair", {}).items():
            # 命名空间防御：S6B_P_/PRT_ 前缀剥离并入；非 pair 格式键（S6B_B_ 等实体级）跳过
            p2 = pid
            for pref in ("S6B_P_", "PRT_"):
                if p2.startswith(pref):
                    p2 = p2[len(pref):]
                    break
            parts = p2.split("__")
            if len(parts) != 2:
                continue
            la, lb = parts[0].split("_"), parts[1].split("_")
            if len(la) < 3 or len(lb) < 3:
                continue
            judged[p2] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
    dt_map = {}
    if os.path.exists(PAIR_CSV):
        dt_map = {r["pair_id"]: r.get("dt") for r in
                  csv.DictReader(open(PAIR_CSV, encoding="utf-8-sig"))}

    def pitem(pid):
        return (parse_pid(pid), {"pair_id": pid, "dt": dt_map.get(pid, ""),
                                 "human_same_person": judged[pid]})
    pos = [pitem(pid) for pid, v in judged.items() if v == "1"]
    neg = [pitem(pid) for pid, v in judged.items() if v == "0"]
    qry = [pitem(pid) for pid, v in judged.items() if v == "?"]

    _, v5_edges = load_edges(args.v5)
    v2d, v2_edges = load_edges(args.edges)
    v5_set = {(e["a"], e["b"]) for e in v5_edges}
    act_set = {(e["a"], e["b"]) for e in v2_edges if e["status"] == "active"}
    aud_set = {(e["a"], e["b"]) for e in v2_edges if e["status"] == "audit"}
    acc_set = act_set | aud_set
    blk_set = {(g["a"], x["b"]) for g in v2d.get("conflict_groups", []) for x in g["blocked"]}

    triage = json.load(open(args.triage, encoding="utf-8"))
    pending = [(p["a"], p["b"], p) for p in triage.get("pending", [])]

    report = {"version": "relay_v2/metrics/1",
              "based_on": {"edges": os.path.relpath(args.edges, ROOT),
                           "labeled": os.path.relpath(args.labeled, ROOT)},
              "n_judged": {"pos": len(pos), "neg": len(neg), "query": len(qry)}}

    # ================= ① 边级 FN / FP =================
    print("=== ① 边级 FN / FP（判定口径）===")
    def fn_of(edge_set):
        return [f"{a}<->{b}" for (a, b), _ in pos if (a, b) not in edge_set]
    fn_v5 = fn_of(v5_set)
    fn_v2a = fn_of(act_set)
    fn_v2aa = fn_of(acc_set)
    fn_v2aab = fn_of(acc_set | blk_set)
    fp_act = [f"{a}<->{b}" for (a, b), _ in neg if (a, b) in act_set]
    fp_aud = [f"{a}<->{b}" for (a, b), _ in neg if (a, b) in aud_set]
    fp_blk = [f"{a}<->{b}" for (a, b), _ in neg if (a, b) in blk_set]
    n_pos_decided_v2 = sum(1 for (a, b), _ in pos if (a, b) in act_set)
    print(f"  判定同人 {len(pos)} 对 → FN: v5={len(fn_v5)}/{len(pos)} | "
          f"v2-active={len(fn_v2a)}/{len(pos)} | v2-active+audit={len(fn_v2aa)}/{len(pos)} | "
          f"+blocked(显露)={len(fn_v2aab)}/{len(pos)}")
    print(f"  判定异人 {len(neg)} 对 → FP: active={len(fp_act)} | audit={len(fp_aud)} | "
          f"blocked={len(fp_blk)}")
    print(f"  （FP 分母提示：有 active 边且已判样本 {n_pos_decided_v2} 条，全部判同人；"
          f"已判负样本 {len(neg)} 条）")
    report["edge_level"] = {
        "fn": {"denominator_pos": len(pos), "v5": fn_v5, "v2_active": fn_v2a,
               "v2_active_audit": fn_v2aa, "v2_surfaced_blocked": fn_v2aab},
        "fp": {"denominator_neg": len(neg), "active": fp_act, "audit": fp_aud,
               "blocked": fp_blk,
               "n_positive_edges_judged": n_pos_decided_v2},
        "note": "FN=v5/v2 缺边的同人对；FP=有边但判异人（audit 不构成生产误接）；"
                "blocked=预算挤出显式层（亦非误接，需复核）"}

    # ================= ② crossing / returning 近似 =================
    print("\n=== ② crossing / returning 近似分类（accepted={} 边）===".format(len(acc_set)))
    pid2judge = {p: r for p, r in pos + neg + qry}
    out_by_a, in_by_b = defaultdict(list), defaultdict(list)
    for e in v2_edges:
        key = (e["a"], e["b"])
        if key in acc_set:
            out_by_a[e["a"]].append(e)
            in_by_b[e["b"]].append(e)

    def tag(key):
        r = pid2judge.get(key)
        if not r:
            return "-"
        return {"1": "judged:same", "0": "judged:DIFF", "?": "judged:?"}[r["human_same_person"].strip()]

    crossing = []   # 同 a 多边（一人多次通行分支；FP 近似 = 其中含判异人）
    for a, es in out_by_a.items():
        if len(es) >= 2:
            crossing.append({"a": a, "n": len(es),
                             "bs": [{"b": e["b"], "sim": round(e["sim"], 4),
                                     "dt": e["dt"], "tag": tag((e["a"], e["b"]))} for e in
                                    sorted(es, key=lambda x: -x["sim"])]})
    crossing.sort(key=lambda x: -x["n"])
    returning = []  # 同 b 多边回联（Ristani returning 缺陷近似）
    for b, es in in_by_b.items():
        if len(es) >= 2:
            returning.append({"b": b, "n": len(es),
                              "as": [{"a": e["a"], "sim": round(e["sim"], 4),
                                      "dt": e["dt"], "tag": tag((e["a"], e["b"]))} for e in es]})
    returning.sort(key=lambda x: -x["n"])
    fp_crossing = [c for c in crossing if any(x["tag"] == "judged:DIFF" for x in c["bs"])]
    print(f"  crossing 候选（同一 exit 多条 accepted 边）: {len(crossing)} 个 exit"
          f"（含判异人边: {len(fp_crossing)} 个）")
    for c in crossing[:6]:
        print(f"    {c['a']}: {c['n']} 条 | " +
              "; ".join(f"{x['b'].split(':')[1]}({x['sim']:.3f},{x['tag']})" for x in c["bs"][:5]))
    print(f"  returning 候选（同一 entry 被多条边回联）: {len(returning)} 个 entry")
    for r in returning[:6]:
        print(f"    {r['b']}: {r['n']} 条 | " +
              "; ".join(f"{x['a'].split(':')[1]}({x['sim']:.3f},{x['tag']})" for x in r["as"][:5]))
    report["crossing_returning"] = {"n_accepted": len(acc_set), "crossing": crossing[:30],
                                    "returning": returning[:30],
                                    "fp_crossing_n": len(fp_crossing),
                                    "note": "crossing=同 a 多边（多对多分支，非错误本身）；"
                                            "returning=同 b 多回联（K_b=2 上限）；"
                                            "判异人且被接受者才会升级为缺陷（当前 0）"}

    # ================= ③ 覆盖率重算 =================
    cov = {}
    if not args.skip_coverage:
        print("\n=== ③ 覆盖率重算（c0 口径）===")
        all_ents = {}
        for cam in c0.CAMS:
            all_ents[cam] = c0.build_entity_tracks(cam)
        cov["v5_47"] = coverage_for(v5_edges, all_ents)
        cov["v2_active"] = coverage_for([e for e in v2_edges if e["status"] == "active"], all_ents)
        cov["v2_active_audit"] = coverage_for([e for e in v2_edges
                                               if (e["a"], e["b"]) in acc_set], all_ents)
        g5, g2 = cov["v5_47"]["global"], cov["v2_active_audit"]["global"]
        print(f"  v5(47):     exit {g5['n_exit_covered']}/{g5['n_exit_gated']} = {g5['exit_continuity']} | "
              f"entry {g5['n_entry_covered']}/{g5['n_entry_gated']} = {g5['entry_attribution']}"
              f"   ← 自证（期望 0.301/0.286）")
        print(f"  v2({len(acc_set)}):  exit {g2['n_exit_covered']}/{g2['n_exit_gated']} = {g2['exit_continuity']} | "
              f"entry {g2['n_entry_covered']}/{g2['n_entry_gated']} = {g2['entry_attribution']}"
              f"   ← c0 v6 试算（audit 含未复核边，上界性质）")
        for cam in c0.CAMS:
            a, b = cov["v5_47"]["per_cam"][cam], cov["v2_active_audit"]["per_cam"][cam]
            print(f"    [{cam}] exit {a['exit_continuity']} → {b['exit_continuity']} | "
                  f"entry {a['entry_attribution']} → {b['entry_attribution']}")
        report["coverage"] = cov
    else:
        report["coverage"] = {"skipped": True}

    # ================= ④ EM−ES（暂缓） =================
    report["em_es"] = {
        "status": "deferred",
        "formula": "EM = mean_errors(单机位内), ES = mean_errors(跨机位); EM−ES 衡量跨机难度差"
                   "（Ristani ECCV'16 §4，IDF1 分解）",
        "gap": "需帧级时序真值（完整轨迹 GT）才能计算；当前仅有边级判定（19/241），"
               "不具备段落级匹配评估条件。待独立短窗真值（P0 差距项）完成后启用。"}

    # ================= ⑤ top-10 复核清单 + 存疑裁决表 =================
    print("\n=== ⑤ top-10 待复核（audit/pending 未判边）===")
    q_keys = {p for p, _ in qry}
    cand = []
    for e in v2_edges:
        if e["status"] == "audit":
            cand.append({"a": e["a"], "b": e["b"], "sim": e["sim"], "dt": e["dt"],
                         "tier": e.get("tier", "main"),
                         "src": ("secondary_weak" if e.get("tier") == "weak" else "secondary"),
                         "in_band": DT_BAND[0] <= e["dt"] <= DT_BAND[1]})
    for a, b, p in pending:
        cand.append({"a": a, "b": b, "sim": p["sim"], "dt": p["dt"],
                     "src": "pending", "in_band": DT_BAND[0] <= p["dt"] <= DT_BAND[1]})
    cand = [c for c in cand if (c["a"], c["b"]) not in pid2judge]
    for c in cand:
        c["priority"] = round(c["sim"] + (0.030 if (c["a"], c["b"]) in q_keys else 0)
                              + (0.010 if c["in_band"] else 0), 4)
    top10 = sorted(cand, key=lambda x: -x["priority"])[:10]
    for c in top10:
        print(f"  {c['a']} -> {c['b']} sim={c['sim']:.4f} dt={c['dt']:5.1f} [{c['src']}]"
              + (" ⚑存疑相关" if (c["a"], c["b"]) in q_keys else ""))
    # 存疑（v=?）在池中的状态（裁决就绪表）
    q_table = []
    for (a, b), r in qry:
        st = ("active" if (a, b) in act_set else
              "audit" if (a, b) in aud_set else
              "blocked" if (a, b) in blk_set else
              "pending" if any((a, b) == (x[0], x[1]) for x in pending) else "not_in_pool")
        q_table.append({"pair": f"{a}<->{b}", "state": st, "dt": r["dt"],
                        "in_v5": (a, b) in v5_set})
    print(f"\n  —— 存疑 {len(qry)} 对裁决就绪表 ——")
    for t in q_table:
        print(f"    {t['pair']:34s} 池内状态={t['state']:12s} dt={t['dt']} in_v5={t['in_v5']}")
    report["top10_review"] = top10
    report["query_pairs"] = q_table

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
