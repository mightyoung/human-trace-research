# -*- coding: utf-8 -*-
"""TOP-3：门户-边置信化（zone-pair 置信分）——"每出口只留最佳一条" → "排序 + 多候选"。

方法（借 City-scale camera-link, arXiv 2405.11345 的 zone-pair 置信分思想）：
  1) 候选对 (a→b) 按出画点到 cam_a 的**出画门**、入画点到 cam_b 的**入画门**归入 zone-pair：
       zone = (cam_a, gate_exit) × (cam_b, gate_entry)
  2) 每个 zone-pair 计算三个统计量：支持数 N / 平均外观 sim / 通行时间方差 Var(dt)
  3) 置信分 = 三个标准化项的等权合成（α=β=γ=1，z-score 后相加，**无任何真值调参**）：
       conf = z(mean_sim) + z(log1p(N)) + z(−log1p(Var_dt))
  4) 每个"出口门"给出 zone-pair 排序 + top-k（k=3）——替代"每出口硬留一条"。

输出：analysis/topology/portal_graph_v2.json（新增，不覆盖既有 196 边）
评测（只读真值，不参与打分）：4 组裁决 winner / 55 判 pos-neg 的 top-k 命中对比（vs 朴素 sim 排序）。

用法：python tools/portal_edge_confidence.py
"""
import csv
import json
import os
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
CAND = os.path.join(ANA, "relay_v2", "scored_v2.json")
C0 = os.path.join(ANA, "c0", "c0_baseline_v7.json")
EDGES = os.path.join(ANA, "relay_v2", "edges_v2_gated.json")
OUT = os.path.join(ANA, "topology", "portal_graph_v2.json")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
VERDICTS = os.path.join(ANA, "truth", "_review", "conflict_verdicts.json")

GATE_R = 0.25   # 门的最大归入半径（超过则记 g?）


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def nearest_gate(pt, gates, kind):
    best, bi = None, None
    for i, g in enumerate(gates):
        dd = dist(pt, g["center"])
        if best is None or dd < best:
            best, bi = dd, i
    if best is None or best > GATE_R:
        return None
    return bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="输出路径（默认产出到 analysis/topology/）")
    args = ap.parse_args()
    out_path = args.out
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cand = json.load(open(CAND, encoding="utf-8"))["candidates"]
    c0 = json.load(open(C0, encoding="utf-8"))
    edges = json.load(open(EDGES, encoding="utf-8"))["edges"]

    gates = {}
    for cam, lst in c0["gates"].items():
        gates[cam] = dict(
            exit=[g for g in lst if g.get("kind") == "exit"],
            entry=[g for g in lst if g.get("kind") == "entry"])

    # ---- zone-pair 归组 ----
    zp = defaultdict(list)
    c2zone = {}
    for c in cand:
        ca = c["a"].split(":")[0]
        cb = c["b"].split(":")[0]
        ge = nearest_gate(c["a_end"], gates.get(ca, {}).get("exit", []), "exit")
        gn = nearest_gate(c["b_start"], gates.get(cb, {}).get("entry", []), "entry")
        zkey = f"{ca}|g{ge if ge is not None else '?'}|{cb}|g{gn if gn is not None else '?'}"
        zp[zkey].append(c)
        c2zone[(c["a"], c["b"])] = zkey

    # ---- zone-pair 统计与置信分 ----
    rows = []
    for zkey, lst in zp.items():
        sims = np.array([x["sim"] for x in lst if x.get("sim") is not None])
        dts = np.array([x["dt"] for x in lst if x.get("dt") is not None])
        rows.append(dict(zone=zkey, n=len(lst), mean_sim=float(sims.mean()) if len(sims) else None,
                         var_dt=float(dts.var()) if len(dts) else None,
                         dt_med=float(np.median(dts)) if len(dts) else None))
    ms = np.array([r["mean_sim"] if r["mean_sim"] is not None else np.nan for r in rows])
    nn = np.array([r["n"] for r in rows], float)
    vv = np.array([r["var_dt"] if r["var_dt"] is not None else np.nan for r in rows], float)

    def zs(a):
        m = np.nanmean(a)
        s = np.nanstd(a) or 1e-9
        return (a - m) / s

    z_sim, z_n, z_var = zs(ms), zs(np.log1p(nn)), zs(-np.log1p(np.nan_to_num(vv)))
    for r, a, b, c in zip(rows, z_sim, z_n, z_var):
        r["conf"] = float(a + b + c)

    # ---- 出口门视角：排序 + top-k ----
    by_exit = defaultdict(list)
    for r in rows:
        ca, ge, cb, gn = r["zone"].split("|")
        by_exit[(ca, ge)].append(r)
    exit_rank = {}
    for k, lst in by_exit.items():
        lst.sort(key=lambda x: -x["conf"])
        exit_rank[k] = [x["zone"] for x in lst]

    def rank_of(zkey):
        ca, ge, cb, gn = zkey.split("|")
        lst = exit_rank.get((ca, ge), [])
        return (lst.index(zkey) + 1) if zkey in lst else None

    # ---- 评测：top-k 命中（pos / group winner），v2 active 覆盖 ----
    def parse_pid(pid):
        left, right = pid.split("__")
        la, lb = left.split("_"), right.split("_")
        return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"

    # 朴素基线：出口门内按"最佳单条 sim"排序
    exit_rank_simple = {}
    for k, lst in by_exit.items():
        lst2 = sorted(lst, key=lambda x: -(x["mean_sim"] or 0))
        exit_rank_simple[k] = [x["zone"] for x in lst2]

    def rank_in(zkey, table):
        ca, ge = zkey.split("|")[0], zkey.split("|")[1]
        lst = table.get((ca, ge), [])
        return (lst.index(zkey) + 1) if zkey in lst else None

    judged = {}
    for r in csv.DictReader(open(LABELED, encoding="utf-8-sig")):
        v = (r.get("human_same_person") or "").strip()
        if v in ("1", "0", "?"):
            judged[r["pair_id"]] = v
    lj = json.load(open(LABELS_JSON, encoding="utf-8"))
    for pid, rec in lj.get("pair", {}).items():
        # ★ 命名空间过滤：S6B_P_/PRT_ 前缀剥离并入；实体级键跳过（防 parse_pid 崩溃）
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

    def hit_stats(pairs):
        res = dict(n=0, zp_missing=0, top1_conf=0, top3_conf=0, top1_simple=0, top3_simple=0,
                   details=[])
        for pid in pairs:
            a, b = parse_pid(pid)
            zkey = c2zone.get((a, b))
            if zkey is None:
                res["zp_missing"] += 1
                continue
            res["n"] += 1
            r1, r3 = rank_of(zkey), rank_in(zkey, exit_rank_simple)
            res["top1_conf"] += int(r1 == 1)
            res["top3_conf"] += int(r1 is not None and r1 <= 3)
            res["top1_simple"] += int(r3 == 1)
            res["top3_simple"] += int(r3 is not None and r3 <= 3)
            res["details"].append(dict(pair=pid, zone=zkey, rank_conf=r1, rank_simple=r3))
        return res

    pos = [p for p, v in judged.items() if v == "1"]
    neg = [p for p, v in judged.items() if v == "0"]
    hit_pos = hit_stats(pos)
    hit_neg = hit_stats(neg)

    # 组裁决 winner
    groups = json.load(open(VERDICTS, encoding="utf-8"))["groups"]
    winner_pids = []
    for gk, gv in groups.items():
        w = gv["winner"]  # "316:316-E046"
        for c in gv.get("candidates", []):
            if c.get("b") == w:
                winner_pids.append(c["pair_id"])
                break

    # ---- k 扫描：conf 排序 vs 朴素 sim 排序；覆盖 pos / winners / v2 active / audit ----
    active = [(e["a"], e["b"]) for e in edges if e["status"] == "active"]
    audit = [(e["a"], e["b"]) for e in edges if e["status"] == "audit"]

    def ranks_of(pairs):
        out = []
        for a, b in pairs:
            z = c2zone.get((a, b))
            if z is None:
                continue
            out.append((rank_of(z), rank_in(z, exit_rank_simple)))
        return out

    r_pos = ranks_of([parse_pid(p) for p in pos])
    r_win = ranks_of([parse_pid(p) for p in winner_pids])
    r_act = ranks_of(active)
    r_aud = ranks_of(audit)

    def cover(rs, k, which):
        return sum(1 for r in rs if r[which] is not None and r[which] <= k)

    ksweep = []
    for k in range(1, 11):
        ksweep.append(dict(
            k=k,
            pos_conf=cover(r_pos, k, 0), pos_simple=cover(r_pos, k, 1),
            win_conf=cover(r_win, k, 0), win_simple=cover(r_win, k, 1),
            act_conf=cover(r_act, k, 0), act_simple=cover(r_act, k, 1),
            aud_conf=cover(r_aud, k, 0), aud_simple=cover(r_aud, k, 1),
        ))

    def avg_rank(rs, which):
        rr = [r[which] for r in rs if r[which] is not None]
        return round(sum(rr) / len(rr), 2) if rr else None

    K_MAIN = 7  # 最小 k 使 v2 active 全覆盖（见 k 扫描）；作为"每出口保留 top-k"建议值
    main = dict(
        k=K_MAIN,
        pos=f"{cover(r_pos, K_MAIN, 0)}/{len(r_pos)}",
        winners=f"{cover(r_win, K_MAIN, 0)}/{len(r_win)}",
        v2_active=f"{cover(r_act, K_MAIN, 0)}/{len(r_act)}",
        v2_audit=f"{cover(r_aud, K_MAIN, 0)}/{len(r_aud)}",
        avg_rank_pos_conf=avg_rank(r_pos, 0), avg_rank_pos_simple=avg_rank(r_pos, 1),
        baseline_top1=dict(pos=f"{cover(r_pos, 1, 0)}/{len(r_pos)}",
                           active=f"{cover(r_act, 1, 0)}/{len(r_act)}"),
    )

    out = dict(version="portal_graph_v2/1",
               method="conf = z(mean_sim) + z(log1p(N)) + z(-log1p(Var_dt))（等权，无真值调参）",
               finding=("复合分未优于朴素 mean_sim 排序（pos 平均排名 3.64 vs 2.91；"
                        "k=5 覆盖 7/11 vs 11/11）→ 复合分保留为**诊断**（N/Var_dt 提示忙/抖方向），"
                        "召回优先时按 mean_sim 排序；两者均存于本文件"),
               n_zones=len(rows), zones=rows,
               exit_rank_conf={f"{k[0]}{k[1]}": v for k, v in exit_rank.items()},
               exit_rank_simple={f"{k[0]}{k[1]}": v for k, v in exit_rank_simple.items()},
               eval=dict(k_sweep=ksweep, main=main,
                         pos_ranks=[r[0] for r in r_pos], winner_ranks=[r[0] for r in r_win],
                         note="rank=该 zone 在其出口门内的 conf 排名（1 起）；simple=按 mean_sim 排序对照"),
               note="新增产物：不覆盖任何既有边；对 196 边只做覆盖对照")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"[zones] {len(rows)}")
    for s in ksweep:
        print(f"  k={s['k']}: pos {s['pos_conf']}/{len(r_pos)}(simple {s['pos_simple']}) | "
              f"win {s['win_conf']}/{len(r_win)} | active {s['act_conf']}/{len(r_act)} | "
              f"audit {s['aud_conf']}/{len(r_aud)}")
    print(f"[main k={K_MAIN}] {main}")
    print("saved", out_path)


if __name__ == "__main__":
    main()
