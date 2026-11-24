# -*- coding: utf-8 -*-
"""回放评估器（最小版）——Dream-RSI 式「先回放打分、再上线」的判定策略离线评估。

任何判定参数的调整（镜像规则阈值 / 接力门限 / 匹配门限）先在**历史标注数据**上
回放重算，量化对精度与结构的影响，避免"调一个参数破坏另一处"的回归。

三个回放面板（全部复用历史确认的标注，零新标注成本）：
  mirror  镜像运动学规则   29 真对 + 408 对照（mirror_kinematics.json）
  match30 318×320 匹配门限 8 正 + 22 负（30 判读对；嵌入现算）
  relay   接力门限 + 链结构 3744 候选池 + 白帽链回归锚点（2 条视觉确认边）

用法：
  python tools/replay_eval.py                                   # baseline 复现
  python tools/replay_eval.py --set relay.sim_thr=0.95          # 单参数覆盖
  python tools/replay_eval.py --sweep match30.sim_thr --values 0.90,0.92,0.94,0.95,0.96
  python tools/replay_eval.py --config my.json --out result.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")

BASELINE = {
    "mirror": {"du_max": 0.06, "pred_err_max": 0.10, "offset_lo": 0.35, "offset_hi": 0.60},
    "match30": {"sim_thr": 0.9433},
    "relay": {"sim_thr": 0.943},
    "registry": {"sim_hi": 0.95, "sim_reject": 0.943, "w_dt_min": 0.30},
    "zone": {"k": 7},
}

# relay 回归锚点（视觉确认的白帽链两跳，REPORT 样例 A + 本次会话复现）
RELAY_ANCHORS = [
    ("343:343-E046", "316:316-E105"),
    ("316:316-E105", "318:318-E099"),
]

POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# ================= 面板 1：镜像运动学规则 =================
def load_mirror_rows():
    p = os.path.join(ANA, "mirror_kinematics", "mirror_kinematics.json")
    d = json.load(open(p, encoding="utf-8"))
    return d["true_rows"], d["control_rows"]


def eval_mirror(cfg, rows=None):
    if rows is None:
        rows = load_mirror_rows()
    tr, cr = rows
    def rule(m):
        if m is None or m.get("pred_err") is None:
            return False
        return (abs(m["du_med"]) < cfg["du_max"]
                and m["pred_err"] <= cfg["pred_err_max"]
                and cfg["offset_lo"] <= m["offset_ratio"] <= cfg["offset_hi"])
    tp = sum(1 for r in tr if rule(r))
    fp = sum(1 for r in cr if rule(r))
    fn = len(tr) - tp
    out = prf(tp, fp, fn)
    out["passed_control"] = [r["pair"] for r in cr if rule(r)]
    return out


# ================= 面板 2：318×320 匹配门限 =================
def load_match30_scores():
    """从 cradio_scale_v3 嵌入现算 30 对 sim（正/负），缓存到 json。"""
    cache = os.path.join(ANA, "replay", "match30_scores.json")
    if os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        return d["pos"], d["neg"]
    embs = {}
    for cam in ("318", "320"):
        f = os.path.join(ANA, "cradio_scale_v3", f"{cam}_emb_scale.npz")
        d = np.load(f)
        embs[cam] = {key_to_tid(k): d[k] for k in d.files}
    def sim(a, b):
        x, y = embs["318"][a], embs["320"][b]
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    pos = [round(sim(a, b), 5) for a, b in POS30]
    neg = [round(sim(a, b), 5) for a, b in NEG30]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({"pos": pos, "neg": neg}, open(cache, "w"), indent=1)
    return pos, neg


def eval_match30(cfg, scores=None):
    pos, neg = scores or load_match30_scores()
    thr = cfg["sim_thr"]
    tp = sum(1 for v in pos if v >= thr)
    fp = sum(1 for v in neg if v >= thr)
    fn = len(pos) - tp
    return prf(tp, fp, fn)


# ================= 面板 3：接力门限 + 链结构 =================
def load_relay_candidates():
    p = os.path.join(ANA, "relay", "relay_scored_cradio_v3.json")
    d = json.load(open(p, encoding="utf-8"))
    return d["candidates"]


def relay_mutual_chains(cands, thr):
    """复刻 relay_rescore.py 的 mutual-best + 链构造（保持与生产逻辑一致）。"""
    rows = [c for c in cands if c["sim"] >= thr]
    bfa, bfb = {}, {}
    for r in rows:
        if r["a"] not in bfa or r["sim"] > bfa[r["a"]]["sim"]:
            bfa[r["a"]] = r
        if r["b"] not in bfb or r["sim"] > bfb[r["b"]]["sim"]:
            bfb[r["b"]] = r
    mut = [r for r in rows if bfa.get(r["a"]) is r and bfb.get(r["b"]) is r]
    succ, pred = defaultdict(list), defaultdict(list)
    for i, e in enumerate(mut):
        e["_i"] = i
        succ[e["a"]].append(e)
        pred[e["b"]].append(e)
    nodes = set()
    for e in mut:
        nodes.add(e["a"]); nodes.add(e["b"])
    heads = sorted(n for n in nodes if not pred[n])
    chains, used = [], set()
    for h in heads:
        chain = [h]; cur = h
        while True:
            if len(succ[cur]) != 1:
                break
            e = succ[cur][0]
            if e["_i"] in used:
                break
            used.add(e["_i"])
            chain.append(e["b"]); cur = e["b"]
        if len(chain) >= 2:
            chains.append(chain)
    for e in mut:
        if e["_i"] in used:
            continue
        chain = [e["a"], e["b"]]; used.add(e["_i"]); cur = e["b"]
        while True:
            if len(succ[cur]) != 1:
                break
            e2 = succ[cur][0]
            if e2["_i"] in used:
                break
            used.add(e2["_i"])
            chain.append(e2["b"]); cur = e2["b"]
        if len(chain) >= 2:
            chains.append(chain)
    chains.sort(key=lambda c: -len(c))
    return rows, mut, chains


def eval_relay(cfg, cands=None, baseline_edges=None):
    if cands is None:
        cands = load_relay_candidates()
    rows, mut, chains = relay_mutual_chains(cands, cfg["sim_thr"])
    edge_set = {(r["a"], r["b"]) for r in mut}
    anchors = {f"{a} -> {b}": (a, b) in edge_set for a, b in RELAY_ANCHORS}
    out = {
        "sim_thr": cfg["sim_thr"],
        "n_candidates": len(cands),
        "n_pass_thr": len(rows),
        "n_mutual_edges": len(mut),
        "n_chains": len(chains),
        "n_multihop": len([c for c in chains if len(c) >= 3]),
        "anchors_kept": anchors,
        "anchors_all_kept": all(anchors.values()),
    }
    if baseline_edges is not None:
        out["removed_vs_baseline"] = sorted(f"{a} -> {b}" for a, b in baseline_edges - edge_set)
        out["added_vs_baseline"] = sorted(f"{a} -> {b}" for a, b in edge_set - baseline_edges)
    return out, edge_set


# ================= 主流程 =================
REG_FILE = os.path.join(ANA, "eval_registry", "registry_v0.json")
ZONE_FILE = os.path.join(ANA, "topology", "portal_graph_v2.json")


def _pid_to_pair(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return (f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}")


def load_registry_ctx():
    """读 EVAL-1 产物（含逐候选六层快照）+ 判读集（隔离）。"""
    d = json.load(open(REG_FILE, encoding="utf-8"))
    rows = {(r["a"], r["b"]): r["layers"] for r in d["candidates"]}
    judged = {}
    p_lab = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
    if os.path.exists(p_lab):
        import csv as _csv
        for r in _csv.DictReader(open(p_lab, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v in ("1", "0", "?"):
                judged[r["pair_id"]] = v
    lj = json.load(open(os.path.join(ANA, "truth", "remote", "labels.json"), encoding="utf-8"))
    for pid, rec in lj.get("pair", {}).items():
        # ★ 命名空间过滤：S6B_P_/PRT_ 前缀剥离并入；实体级键跳过（防 _pid_to_pair 崩溃）
        p2 = pid
        for pref in ("S6B_P_", "PRT_"):
            if p2.startswith(pref):
                p2 = p2[len(pref):]
                break
        if "__" not in p2:
            continue
        la, lb = p2.split("__", 1)[0].split("_"), p2.split("__", 1)[1].split("_")
        if len(la) < 3 or len(lb) < 3:
            continue
        judged[p2] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
    gsets = []
    groups = json.load(open(os.path.join(ANA, "truth", "_review", "conflict_verdicts.json"),
                            encoding="utf-8"))["groups"]
    for gk, gv in groups.items():
        for c in gv.get("candidates", []):
            if c.get("human") in ("same", "diff"):
                gsets.append((c["pair_id"], c["human"]))
    return rows, judged, gsets


def eval_registry_panel(cfg):
    """面板 4：EVAL-1 规则库阈值回放（在冻结的六层快照上重算动作，零重跑）。"""
    rows, judged, gsets = load_registry_ctx()
    hi, rj, wm = cfg["sim_hi"], cfg["sim_reject"], cfg["w_dt_min"]

    def action(layers):
        if layers is None:
            return None
        if layers["L4_constraints"]["status"] == "fail" or layers["L1_quality"]["status"] == "fail":
            return "reject"
        e3 = layers["L3_evidence"]
        sim, wdt, dt = e3["sim"], e3["w_dt"], e3["dt"]
        if sim is not None and sim < rj:
            return "reject"
        l2w = layers["L2_geometry"]["status"] == "warn"
        if (sim is not None and sim >= hi and wdt is not None and wdt >= wm
                and dt is not None and dt >= -1 and not l2w):
            return "keep"
        return "defer"

    out = dict(n_pos=0, pos_keep=0, pos_defer=0, pos_reject=0,
               n_neg=0, neg_reject=0, neg_defer=0, neg_keep=0,
               n_unclear=0, unclear_defer=0,
               n_gsame=0, gsame_keep=0, gsame_reject=0,
               n_gdiff=0, gdiff_reject=0, gdiff_keep=0)
    for pid, v in judged.items():
        a = action(rows.get(_pid_to_pair(pid)))
        if a is None:
            continue
        if v == "1":
            out["n_pos"] += 1
            out["pos_keep" if a == "keep" else ("pos_defer" if a == "defer" else "pos_reject")] += 1
        elif v == "0":
            out["n_neg"] += 1
            out["neg_reject" if a == "reject" else ("neg_defer" if a == "defer" else "neg_keep")] += 1
        else:
            out["n_unclear"] += 1
            if a == "defer":
                out["unclear_defer"] += 1
    for pid, human in gsets:
        a = action(rows.get(_pid_to_pair(pid)))
        if a is None:
            continue
        if human == "same":
            out["n_gsame"] += 1
            out["gsame_keep" if a == "keep" else ("gsame_reject" if a == "reject" else "gsame_d")] = \
                out.get("gsame_keep" if a == "keep" else ("gsame_reject" if a == "reject" else "gsame_d"), 0) + 1
        else:
            out["n_gdiff"] += 1
            out["gdiff_reject" if a == "reject" else ("gdiff_keep" if a == "keep" else "gdiff_d")] = \
                out.get("gdiff_reject" if a == "reject" else ("gdiff_keep" if a == "keep" else "gdiff_d"), 0) + 1
    out["false_reject"] = out["pos_reject"] + out.get("gsame_reject", 0)
    out["false_keep"] = out["neg_keep"] + out.get("gdiff_keep", 0)
    out["same_total"] = out["n_pos"] + out["n_gsame"]
    out["diff_total"] = out["n_neg"] + out["n_gdiff"]
    out["pos_keep_rate"] = round(out["pos_keep"] / max(out["n_pos"], 1), 4)
    out["neg_reject_rate"] = round(out["neg_reject"] / max(out["n_neg"], 1), 4)
    return out


def eval_zone_panel(cfg):
    """面板 5：TOP-3 zone-pair 排序的 top-k 覆盖率回放（读冻结排序表）。"""
    d = json.load(open(ZONE_FILE, encoding="utf-8"))
    k = int(cfg["k"])
    ranks = d["exit_rank_conf"]
    cand = json.load(open(os.path.join(ANA, "relay_v2", "scored_v2.json"), encoding="utf-8"))["candidates"]
    c0 = json.load(open(os.path.join(ANA, "c0", "c0_baseline_v7.json"), encoding="utf-8"))
    gates = {}
    for cam, lst in c0["gates"].items():
        gates[cam] = dict(exit=[g["center"] for g in lst if g.get("kind") == "exit"],
                          entry=[g["center"] for g in lst if g.get("kind") == "entry"])

    def dist(p, q):
        return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5

    def nearest(pt, gs, r=0.25):
        best, bi = None, None
        for i, g in enumerate(gs):
            dd = dist(pt, g)
            if best is None or dd < best:
                best, bi = dd, i
        return bi if (best is not None and best <= r) else None

    c2zone = {}
    for c in cand:
        ca, cb = c["a"].split(":")[0], c["b"].split(":")[0]
        ge = nearest(c["a_end"], gates.get(ca, {}).get("exit", []))
        gn = nearest(c["b_start"], gates.get(cb, {}).get("entry", []))
        c2zone[(c["a"], c["b"])] = f"{ca}|g{ge if ge is not None else '?'}|{cb}|g{gn if gn is not None else '?'}"

    def rank_of(zkey):
        ca, ge = zkey.split("|")[0], zkey.split("|")[1]
        lst = ranks.get(f"{ca}{ge}", [])
        return (lst.index(zkey) + 1) if zkey in lst else None

    _, judged, _ = load_registry_ctx()
    pos = [p for p, v in judged.items() if v == "1"]
    edges = json.load(open(os.path.join(ANA, "relay_v2", "edges_v2_gated.json"),
                           encoding="utf-8"))["edges"]
    active = [(e["a"], e["b"]) for e in edges if e["status"] == "active"]
    audit = [(e["a"], e["b"]) for e in edges if e["status"] == "audit"]

    def cov(pairs):
        rr = []
        for a, b in pairs:
            z = c2zone.get((a, b))
            if z is None:
                continue
            r = rank_of(z)
            if r is not None:
                rr.append(r)
        return sum(1 for r in rr if r <= k), len(rr)

    pk, pn = cov([_pid_to_pair(p) for p in pos])
    ak, an = cov(active)
    uk, un = cov(audit)
    return dict(k=k, pos=f"{pk}/{pn}", active=f"{ak}/{an}", audit=f"{uk}/{un}",
                pos_rate=round(pk / max(pn, 1), 4), active_rate=round(ak / max(an, 1), 4))


def run(cfg, verbose=True):
    res = {}
    mt, mc = load_mirror_rows()
    res["mirror"] = eval_mirror(cfg["mirror"], (mt, mc))
    res["match30"] = eval_match30(cfg["match30"])
    res["relay"], relay_edges = eval_relay(cfg["relay"])
    res["registry"] = eval_registry_panel(cfg["registry"])
    res["zone"] = eval_zone_panel(cfg["zone"])
    if verbose:
        m = res["mirror"]
        print(f"[mirror ] 规则 |du|<{cfg['mirror']['du_max']} & perr<={cfg['mirror']['pred_err_max']} "
              f"& {cfg['mirror']['offset_lo']}<=off<={cfg['mirror']['offset_hi']}")
        print(f"          TP={m['tp']} FP={m['fp']} FN={m['fn']} | P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f}")
        if m["passed_control"]:
            print(f"          控制池通过: {m['passed_control']}")
        mm = res["match30"]
        print(f"[match30] thr={cfg['match30']['sim_thr']}: TP={mm['tp']} FP={mm['fp']} FN={mm['fn']} "
              f"| P={mm['precision']:.3f} R={mm['recall']:.3f} F1={mm['f1']:.3f}")
        r = res["relay"]
        print(f"[relay  ] thr={r['sim_thr']}: 候选 {r['n_candidates']} → 过门 {r['n_pass_thr']} → "
              f"mutual 边 {r['n_mutual_edges']} → 链 {r['n_chains']}（多跳 {r['n_multihop']}）")
        print(f"          锚点保留: {r['anchors_kept']}")
        if "removed_vs_baseline" in r and r["removed_vs_baseline"]:
            print(f"          相对 baseline 丢失 {len(r['removed_vs_baseline'])} 边:")
            for e in r["removed_vs_baseline"][:12]:
                print(f"            - {e}")
        if "added_vs_baseline" in r and r["added_vs_baseline"]:
            print(f"          相对 baseline 新增 {len(r['added_vs_baseline'])} 边:")
            for e in r["added_vs_baseline"][:12]:
                print(f"            + {e}")
        rg = res["registry"]
        print(f"[registry] sim_hi={cfg['registry']['sim_hi']} sim_rej={cfg['registry']['sim_reject']} "
              f"w_dt_min={cfg['registry']['w_dt_min']}: 同人 {rg['n_pos']}→keep {rg['pos_keep']}/"
              f"defer {rg['pos_defer']}/误拒 {rg['pos_reject']}；异人 {rg['n_neg']}→拒 {rg['neg_reject']}/"
              f"误收 {rg['neg_keep']}；组同 {rg['n_gsame']} 组异 {rg['n_gdiff']}")
        zn = res["zone"]
        print(f"[zone    ] top-k={zn['k']}: pos {zn['pos']} ({zn['pos_rate']}) | "
              f"active {zn['active']} ({zn['active_rate']}) | audit {zn['audit']}")
    return res, relay_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="config JSON（缺省 = 生产 baseline）")
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖参数，如 relay.sim_thr=0.95（可多次）")
    ap.add_argument("--sweep", default=None, help="扫描参数名，如 match30.sim_thr")
    ap.add_argument("--values", default=None, help="扫描值（逗号分隔）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = json.loads(json.dumps(BASELINE))
    if args.config:
        cfg.update(json.load(open(args.config, encoding="utf-8")))
    for kv in args.set:
        k, v = kv.split("=", 1)
        panel, key = k.split(".")
        cfg.setdefault(panel, {})[key] = float(v)

    os.makedirs(os.path.join(ANA, "replay"), exist_ok=True)

    if args.sweep:
        panel, key = args.sweep.split(".")
        vals = [float(x) for x in args.values.split(",")]
        base_edges = None
        if panel == "relay":
            _, base_edges = eval_relay({"sim_thr": BASELINE["relay"]["sim_thr"]})
        sweep_rows = []
        print(f"===== 扫描 {args.sweep} ∈ {vals} =====")
        for v in vals:
            c = json.loads(json.dumps(cfg))
            c[panel][key] = v
            if panel == "mirror":
                r = eval_mirror(c["mirror"])
                print(f"  {key}={v}: TP={r['tp']} FP={r['fp']} FN={r['fn']} "
                      f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
                sweep_rows.append({"value": v, **r})
            elif panel == "match30":
                r = eval_match30(c["match30"])
                print(f"  thr={v}: TP={r['tp']} FP={r['fp']} FN={r['fn']} "
                      f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
                sweep_rows.append({"value": v, **r})
            elif panel == "relay":
                if base_edges is None:
                    _, base_edges = eval_relay({"sim_thr": BASELINE["relay"]["sim_thr"]})
                r, edges = eval_relay(c["relay"], baseline_edges=base_edges)
                print(f"  thr={v}: 过门 {r['n_pass_thr']} → 边 {r['n_mutual_edges']} → "
                      f"链 {r['n_chains']}（多跳 {r['n_multihop']}）| 锚点 "
                      f"{'√' if r['anchors_all_kept'] else '×'} | 丢失 {len(r.get('removed_vs_baseline', []))} "
                      f"新增 {len(r.get('added_vs_baseline', []))}")
                sweep_rows.append({"value": v, **r})
            elif panel == "registry":
                r = eval_registry_panel(c["registry"])
                print(f"  {key}={v}: pos_keep={r['pos_keep']}/{r['n_pos']} 误拒={r['pos_reject']} | "
                      f"neg_reject={r['neg_reject']}/{r['n_neg']} 误收={r['neg_keep']} | "
                      f"组同 keep={r.get('gsame_keep', 0)}/{r['n_gsame']} 组异拒={r.get('gdiff_reject', 0)}/{r['n_gdiff']}")
                sweep_rows.append({"value": v, **r})
            elif panel == "zone":
                r = eval_zone_panel(c["zone"])
                print(f"  k={int(v)}: pos {r['pos']} | active {r['active']} | audit {r['audit']}")
                sweep_rows.append({"value": v, **r})
        out = {"sweep": args.sweep, "baseline_thr": cfg[panel][key], "rows": sweep_rows}
        out_p = args.out or os.path.join(ANA, "replay",
                                         f"sweep_{panel}_{key}_{datetime.now():%H%M%S}.json")
        json.dump(out, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[out] {out_p}")
        return

    res, edges = run(cfg)
    res["_config"] = cfg
    out_p = args.out or os.path.join(ANA, "replay", f"replay_{datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump(res, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {out_p}")


if __name__ == "__main__":
    main()
