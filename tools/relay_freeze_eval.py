#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A2 冻结评测（批次 A）：人工判定冻结 + 回流泄漏审计 + 公平覆盖口径。

子命令：
  freeze [--dry-run]   合并判定源 → output_crosscam/analysis/relay_v2/frozen_eval_v1.json
  check                泄漏审计 + 双口径覆盖 → .../frozen_fair_v1.json
  all [--dry-run]      freeze + check

设计纪律（docs/commit-015/PHASE_A_PLAN.md §3）：
  - 只读输入：truth/remote/labels.json ∪ truth/pair_candidates_labeled.csv；
    labels.json 优先；不修改任何现有工具与数据（公平口径为独立并列报告）。
  - 命名空间铁律：前缀剥离（S6B_P_/PRT_/S6B_M_/S6B_B_/S6B_R_）+ 格式校验，
    绝不裸 parse；实体级键（无 '__'）跳过并显式记录清单。
  - freeze 幂等：sha1_16（对 [a,b,v] 排序三元组取 sha1 前 16）相同 → 不动；
    不同 → 备份 .prev + ⚠ 提示"受控修订需人工确认"。
  - 泄漏审计两条回流路径：
      ① 软回流 soft：edge_states.json 中带 human_pass/human_fail evidence 的边；
      ② 硬裁决 hard：gil/human_adjudication_v2.json 中 action∈{confirm,reject} 的边；
      另记 defer（人工接触、未给信息，不并入 assisted）。
  - 覆盖双口径：active（现状生产直读口径 36 条）与 production_pool（c0 v9 口径 =
    gated_human 剔除 rejected，186 条）；各报"现状 / 公平（剥离 human-assisted）"两列。

产物（原子写，避免半写；A1 同款纪律）：
  frozen_eval_v1.json   {version,frozen_ts,sha1_16,n,counts,composition,pairs[]}
  frozen_fair_v1.json   {version,checked_ts,freeze,leak_audit,human_assisted,coverage,adjudication_impact}
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
RV2 = os.path.join(ANA, "relay_v2")

LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
LABELED_CSV = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
EDGES_GATED_HUMAN = os.path.join(RV2, "edges_v2_gated_human.json")
EDGE_STATES = os.path.join(RV2, "edge_states.json")
ADJUDICATION = os.path.join(ANA, "gil", "human_adjudication_v2.json")

OUT_FREEZE = os.path.join(RV2, "frozen_eval_v1.json")
OUT_FAIR = os.path.join(RV2, "frozen_fair_v1.json")

PREFIXES = ("S6B_P_", "PRT_", "S6B_M_", "S6B_B_", "S6B_R_")
V2LABEL = {1: "same", 0: "diff", 2: "unclear"}
CSV2LABEL = {"1": "same", "0": "diff", "?": "unclear"}


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def strip_norm(pid):
    """前缀剥离 + 格式校验（铁律：绝不裸 parse）。

    'S6B_P_305_E_305-E056__323_E_323-E051' → ('305:305-E056','323:323-E051')
    实体级键（无 '__'）或格式不符 → (None, 原因)。
    """
    p2 = pid
    for pref in PREFIXES:
        if p2.startswith(pref):
            p2 = p2[len(pref):]
            break
    parts = p2.split("__")
    if len(parts) != 2:
        return None, "no__"
    la, lb = parts[0].split("_"), parts[1].split("_")
    if len(la) < 3 or len(lb) < 3:
        return None, "bad_format"
    return (f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"), p2


def load_judgments():
    """合并判定源 → (pairs dict, composition dict)。

    pairs: 规范键 p2（无前缀）→ {a,b,label,v,src,key}；labels.json 优先于 csv。
    """
    pairs = {}
    comp = {"labels_json": {}, "csv": {}, "entity_level_skipped": [],
            "dups_collapsed": [], "bad_v": []}

    # ---- labels.json（权威） ----
    d = json.load(open(LABELS_JSON, encoding="utf-8"))
    raw = d.get("pair", {})
    comp["labels_json"]["raw_pairs"] = len(raw)
    n_parsable = 0
    for pid, rec in raw.items():
        nm, p2 = strip_norm(pid)
        if nm is None:
            comp["entity_level_skipped"].append({"key": pid, "reason": p2})
            continue
        n_parsable += 1
        v = rec.get("v")
        if v not in V2LABEL:
            comp["bad_v"].append({"key": pid, "v": v})
            v = 2
        if p2 in pairs:
            prev = pairs[p2]
            comp["dups_collapsed"].append({
                "p2": p2, "kept": prev["key"], "dropped": pid,
                "v_consistent": prev["v"] == v})
            if prev["v"] != v:
                # 同键冲突判定：保留先到者并显式记录（受控修订信号）
                print(f"[warn] 同键判定冲突 {p2}: {prev['key']}(v={prev['v']}) vs {pid}(v={v})，保留先到者")
            continue
        pairs[p2] = {"a": nm[0], "b": nm[1], "label": V2LABEL[v], "v": v,
                     "src": "labels.json", "key": pid}
    comp["labels_json"]["parsable"] = n_parsable
    comp["labels_json"]["unique"] = sum(1 for p in pairs.values() if p["src"] == "labels.json")

    # ---- pair_candidates_labeled.csv（早期，labels.json 优先） ----
    n_csv_judged = n_csv_new = n_csv_overlap = 0
    if os.path.exists(LABELED_CSV):
        for r in csv.DictReader(open(LABELED_CSV, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v not in CSV2LABEL:
                continue
            n_csv_judged += 1
            nm, p2 = strip_norm(r["pair_id"])
            if nm is None:
                comp["entity_level_skipped"].append({"key": r["pair_id"], "reason": p2})
                continue
            if p2 in pairs:
                n_csv_overlap += 1
                continue
            pairs[p2] = {"a": nm[0], "b": nm[1], "label": CSV2LABEL[v],
                         "v": {"1": 1, "0": 0, "?": 2}[v],
                         "src": "labeled.csv", "key": r["pair_id"]}
            n_csv_new += 1
    comp["csv"] = {"judged": n_csv_judged, "new": n_csv_new, "overlap": n_csv_overlap}
    comp["final_n"] = len(pairs)
    return pairs, comp


def semantic_sha1(pairs):
    """语义指纹：对排序后的 [a,b,v] 三元组取 sha1 前 16 位（不含 src/key）。"""
    rows = sorted([[p["a"], p["b"], p["v"]] for p in pairs.values()])
    canon = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]


def cmd_freeze(dry_run=False):
    pairs, comp = load_judgments()
    sha = semantic_sha1(pairs)
    counts = {"same": 0, "diff": 0, "unclear": 0}
    for p in pairs.values():
        counts[p["label"]] += 1

    print("=== A2 freeze · 判定源合并 ===")
    lj, cv = comp["labels_json"], comp["csv"]
    print(f"  labels.json: 原始 {lj['raw_pairs']} → 可解析 {lj['parsable']} → 唯一 {lj['unique']}"
          f"（实体级跳过 {len(comp['entity_level_skipped'])}、重复折叠 {len(comp['dups_collapsed'])}）")
    for d in comp["dups_collapsed"]:
        flag = "v 一致" if d["v_consistent"] else "★v 冲突"
        print(f"     dup: {d['p2']}  [{d['kept']}] vs [{d['dropped']}] {flag}")
    if comp["bad_v"]:
        print(f"  [warn] 非法 v 值 {len(comp['bad_v'])} 条（按 unclear 处理）: {comp['bad_v']}")
    print(f"  labeled.csv: 已判 {cv['judged']} → 新增 {cv['new']}（重叠 {cv['overlap']} 跳过）")
    print(f"  合计冻结 n={len(pairs)} | same {counts['same']} / diff {counts['diff']} / unclear {counts['unclear']}")
    print(f"  sha1_16 = {sha}")

    # ---- 幂等 / 受控修订 ----
    old = None
    if os.path.exists(OUT_FREEZE):
        try:
            old = json.load(open(OUT_FREEZE, encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] 旧 frozen_eval_v1.json 读取失败（{e}），将覆盖")
    if old is not None and old.get("sha1_16") == sha:
        print(f"  无变化（与现存冻结集 sha1_16 相同）→ 不重写")
        return {"sha1_16": sha, "n": len(pairs), "changed": False}

    if old is not None:
        delta = len(pairs) - int(old.get("n", 0))
        print(f"  ⚠ 冻结集相对现存版本有变化（n {old.get('n')} → {len(pairs)}，Δ{delta:+d}）"
              f"——受控修订需人工确认；旧版备份 .prev")

    if dry_run:
        print("  [dry-run] 不写文件")
        return {"sha1_16": sha, "n": len(pairs), "changed": True, "dry_run": True}

    if old is not None:
        shutil.copy(OUT_FREEZE, OUT_FREEZE + ".prev")
    obj = {
        "version": "frozen_eval/1",
        "frozen_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sha1_16": sha,
        "n": len(pairs),
        "counts": counts,
        "composition": comp,
        "note": "冻结集 = labels.json ∪ labeled.csv（前缀剥离+格式校验）；"
                "sha1_16 对排序 [a,b,v] 三元组取 sha1 前 16 位；"
                "消费纪律：本集用于独立评测，禁止回流参与任何阈值/参数选择。",
        "pairs": [pairs[k] for k in sorted(pairs.keys())],
    }
    _atomic_write_json(OUT_FREEZE, obj)
    print(f"  [out] {OUT_FREEZE}")
    return {"sha1_16": sha, "n": len(pairs), "changed": True}


def _load_edges():
    g = json.load(open(EDGES_GATED_HUMAN, encoding="utf-8"))
    edges = g["edges"]
    act = {(e["a"], e["b"]) for e in edges if e.get("status") == "active"}
    pool = {(e["a"], e["b"]) for e in edges
            if e.get("status") in ("active", "audit", "gate", "confirmed")}
    n_rej = sum(1 for e in edges if e.get("status") == "rejected")
    # 人工回流前快照（gated.json，13:43 版）：解读"泄漏性质"的参照
    pre_st = {}
    pre_path = os.path.join(RV2, "edges_v2_gated.json")
    if os.path.exists(pre_path):
        pre = json.load(open(pre_path, encoding="utf-8"))
        pre_st = {(e["a"], e["b"]): e.get("status") for e in pre["edges"]}
    return act, pool, n_rej, pre_st


def _load_human_assisted():
    """两条回流路径 + defer（接触）→ (soft, hard, defer_touch, detail)。"""
    # 软回流：edge_states 的 human_pass/human_fail evidence
    soft, refs = set(), []
    if os.path.exists(EDGE_STATES):
        s = json.load(open(EDGE_STATES, encoding="utf-8"))
        for k, st in s.get("states", {}).items():
            for ev in st.get("evidence", []):
                if ev.get("src") in ("human_pass", "human_fail"):
                    a, b = k.split("->")
                    soft.add((a, b))
                    refs.append({"edge": k, "src": ev["src"], "ref": ev.get("ref", "")})
    # 硬裁决：human_adjudication_v2 的 confirm/reject / defer
    hard, defer_touch = set(), set()
    if os.path.exists(ADJUDICATION):
        adj = json.load(open(ADJUDICATION, encoding="utf-8"))
        for it in adj.get("items", []):
            e = it.get("edge")
            if not e or "->" not in e:
                continue
            a, b = e.split("->")
            if it.get("action") in ("confirm", "reject"):
                hard.add((a, b))
            elif it.get("action") == "defer":
                defer_touch.add((a, b))
    return soft, hard, defer_touch, refs


def _coverage(pool, pos, assisted):
    cur = sorted(pool & pos)
    excluded = sorted((pool & pos) & assisted)
    fair = sorted((pool & pos) - assisted)
    return {"pool_n": len(pool), "pos_n": len(pos),
            "current": {"n": len(cur), "list": [f"{a}->{b}" for a, b in cur]},
            "fair": {"n": len(fair), "list": [f"{a}->{b}" for a, b in fair],
                     "excluded": [f"{a}->{b}" for a, b in excluded]}}


def cmd_check():
    if not os.path.exists(OUT_FREEZE):
        print(f"[error] 冻结集不存在：{OUT_FREEZE}\n  先跑：python tools/relay_freeze_eval.py freeze")
        sys.exit(1)
    fz = json.load(open(OUT_FREEZE, encoding="utf-8"))
    pos = {(p["a"], p["b"]) for p in fz["pairs"] if p["label"] == "same"}
    neg = {(p["a"], p["b"]) for p in fz["pairs"] if p["label"] == "diff"}

    act, pool, n_rej, pre_st = _load_edges()
    soft, hard, defer_touch, refs = _load_human_assisted()
    assisted = soft | hard

    print("=== A2 check · 泄漏审计 ===")
    print(f"  冻结集: n={fz['n']} sha1_16={fz['sha1_16']}"
          f"（same {fz['counts']['same']} / diff {fz['counts']['diff']} / unclear {fz['counts']['unclear']}）")
    # ref → 冻结集对账（铁律：前缀剥离后按规范键比对）
    frozen_p2 = set()
    for p in fz["pairs"]:
        nm, p2 = strip_norm(p["key"])
        if nm is not None:
            frozen_p2.add(p2)
    n_ref_in = n_ref_out = 0
    refs_out = []
    for r in refs:
        pid = r["ref"][len("labeled:"):] if r["ref"].startswith("labeled:") else r["ref"]
        nm, p2 = strip_norm(pid)
        if nm is not None and p2 in frozen_p2:
            n_ref_in += 1
        else:
            n_ref_out += 1
            refs_out.append(r)
    print(f"  软回流 evidence refs: {len(refs)} → 在冻结集内 {n_ref_in} / 集外 {n_ref_out}")
    if refs_out:
        for r in refs_out[:8]:
            print(f"     [集外] {r['edge']}  ref={r['ref']}")

    print("\n=== 覆盖口径（人工判定 pos = 同人） ===")
    cov_act = _coverage(act, pos, assisted)
    cov_pool = _coverage(pool, pos, assisted)
    print(f"  [active 口径] 池 n={cov_act['pool_n']} | 现状 pos∩active = {cov_act['current']['n']}/{cov_act['pos_n']}"
          f" | 公平（剥离 human-assisted）= {cov_act['fair']['n']}/{cov_act['pos_n']}")
    for x in cov_act["current"]["list"]:
        tag = " [human-assisted→剔除]" if x in cov_act["fair"]["excluded"] else ""
        print(f"     {x}{tag}")
    print(f"  [生产池口径] 池 n={cov_pool['pool_n']}（gated_human 剔除 {n_rej} rejected）"
          f" | 现状 pos∩pool = {cov_pool['current']['n']}/{cov_pool['pos_n']}"
          f" | 公平 = {cov_pool['fair']['n']}/{cov_pool['pos_n']}")
    neg_in_act = sorted(neg & act)
    print(f"  一致性: 判异人 ∩ active = {len(neg_in_act)}（必须 0）"
          + (f"  ★异常: {neg_in_act}" if neg_in_act else " ✔"))

    # adjudication 影响面
    adj_impact = {"applied_n": len(hard), "defer_n": len(defer_touch)}
    if os.path.exists(ADJUDICATION):
        adj = json.load(open(ADJUDICATION, encoding="utf-8"))
        rc = [it for it in adj.get("items", []) if it.get("group") == "relay_conflict"]
        rej_from_active = [it["edge"] for it in rc
                           if it.get("action") == "reject" and it.get("edge_status") == "active"]
        conf_from_active = [it["edge"] for it in rc
                            if it.get("action") == "confirm" and it.get("edge_status") == "active"]
        adj_impact.update({
            "confirm_n": sum(1 for it in rc if it.get("action") == "confirm"),
            "reject_n": sum(1 for it in rc if it.get("action") == "reject"),
            "reject_from_active": rej_from_active,
            "confirm_from_active": conf_from_active,
            "defer_from_active": sorted(f"{a}->{b}" for a, b in (defer_touch & act)),
            "note": "active 47 → 36 = −8 confirm（转 confirmed）、−3 reject（转 rejected）；"
                    "另 confirm 2 条自 audit、reject 7 条自 audit；defer 4 条未改动。",
        })
        # 人工回流前状态分解（泄漏性质解读：机器先行 vs 人工抬升）
        pos_pool = pos & pool
        pre_cnt = {}
        for a, b in pos_pool:
            s = pre_st.get((a, b))
            key = s if s is not None else "not_in_pre"
            pre_cnt[key] = pre_cnt.get(key, 0) + 1
        seven = sorted(soft & act)
        seven_pre_all_active = bool(seven) and all(pre_st.get(k) == "active" for k in seven)
        adj_impact.update({
            "pos_pool_pre_status": pre_cnt,
            "soft_active_7_pre_all_active": seven_pre_all_active,
            "interpretation": "pos∩pool 26 条中机器先行（人工前已 active）15 条；"
                              "11 条人工前为 audit。fair=0 的含义 = 全部覆盖边均被人工回流直接作用，"
                              "而非'系统无人工则无输出'（后者需看 gated 快照 47 边口径）。",
        })
        print("\n=== 裁决影响面（硬裁决）===")
        print(f"  applied {len(hard)}（confirm {adj_impact['confirm_n']} / reject {adj_impact['reject_n']}）"
              f" | 由 active 剔除 {len(rej_from_active)}: {rej_from_active}")
        print(f"  defer {len(defer_touch)}（人工接触未改动）"
              f" | 其中在 active: {adj_impact['defer_from_active']}")
        print(f"  [解读] pos∩pool 人工前状态: {pre_cnt}"
              f" | 7 条 human_pass active 边人工前均为 active: {seven_pre_all_active}")

    obj = {
        "version": "frozen_fair/1",
        "checked_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "freeze": {"path": os.path.relpath(OUT_FREEZE, ROOT), "sha1_16": fz["sha1_16"],
                   "n": fz["n"], "counts": fz["counts"]},
        "leak_audit": {
            "n_soft_evidence_refs": len(refs),
            "n_refs_in_frozen": n_ref_in, "n_refs_outside_frozen": n_ref_out,
            "refs_outside": refs_out,
            "note": "软回流=edge_states human_pass/fail evidence；硬裁决=adjudication confirm/reject；"
                    "both 均视为人工判定曾直接作用于该边（评测时须分组/剥离）。"},
        "human_assisted": {
            "soft_edges": len(soft), "hard_edges": len(hard),
            "defer_touch_edges": len(defer_touch), "union_edges": len(assisted),
            "soft_in_active": sorted(f"{a}->{b}" for a, b in (soft & act)),
            "hard_in_active": sorted(f"{a}->{b}" for a, b in (hard & act)),
        },
        "coverage": {"active": cov_act, "production_pool": cov_pool,
                     "neg_in_active_n": len(neg_in_act),
                     "note": "公平口径 = 池内 pos 剥离 human-assisted（soft∪hard）后的命中。"
                             "fair=0 表示全部 pos 覆盖边均被人工作用过（解读见 "
                             "adjudication_impact.interpretation），不代表系统无人工时无输出。"},
        "adjudication_impact": adj_impact,
    }
    _atomic_write_json(OUT_FAIR, obj)
    print(f"\n[out] {OUT_FAIR}")


def main():
    ap = argparse.ArgumentParser(description="A2 冻结评测：freeze / check / all")
    ap.add_argument("cmd", choices=["freeze", "check", "all"])
    ap.add_argument("--dry-run", action="store_true", help="freeze 不写文件（check 仍写报告）")
    args = ap.parse_args()

    if args.cmd in ("freeze", "all"):
        r = cmd_freeze(dry_run=args.dry_run)
        if args.cmd == "all" and r.get("dry_run"):
            print("[info] dry-run 模式，check 跳过")
            return
    if args.cmd in ("check", "all"):
        cmd_check()


if __name__ == "__main__":
    main()
