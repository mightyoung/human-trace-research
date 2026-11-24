# -*- coding: utf-8 -*-
"""EVAL-1：六层评估规则库 v0（"自动评估器·规则版"，离线 CPU）。

定位：给"候选接续/合并决策"（relay 候选对、gperson 更新、选择器新版本）一份**独立、可复述的
六层评审**，输出动作建议（保留/隔离/拆分/合并/暂缓）。本版评估对象 = relay 候选对（边域），
并产出 gperson 建议清单。**规则先于学习模型**（EVAL-2 的确定性基线）。

六层（边域）：
  ① 数据质量   : 框高合理 / sim 存在 / 实体在 GIL 可查
  ② 几何结构   : 出画点近出画门 + 入画点近入画门（C0 v7 门坐标）
  ③ 证据强度   : sim / w_dt / dt + 同源最佳竞争者 margin
  ④ 约束冲突   : dt≥-1 且 **实体时间区间不相交**（318×320 重叠豁免）
  ⑤ 跨机位支持 : 该方向候选支撑数 + 时段分箱覆盖
  ⑥ 身份/下游  : gperson 归属一致性 + relay 状态 + 四态 E

动作（边域）: keep(保留) / reject(隔离) / defer(暂缓)（拆分/合并见于 gperson 建议域）。

只读输入：gil.sqlite、scored_v2.json、edges_v2_gated.json、edge_states.json、c0_baseline_v7.json；
评测输入（隔离记录，不参与规则）：truth/remote/labels.json、pair_candidates_labeled.csv、
truth/_review/conflict_verdicts.json。

用法：python tools/cluster_eval_registry.py [--cand <scored_v2.json>] [--out <dir>]
"""
import argparse
import csv
import json
import os
import sqlite3
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
GIL = os.path.join(ANA, "gil", "store_v1", "gil.sqlite")
CAND = os.path.join(ANA, "relay_v2", "scored_v2.json")
EDGES = os.path.join(ANA, "relay_v2", "edges_v2_gated.json")
STATES = os.path.join(ANA, "relay_v2", "edge_states.json")
C0 = os.path.join(ANA, "c0", "c0_baseline_v7.json")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
VERDICTS = os.path.join(ANA, "truth", "_review", "conflict_verdicts.json")

# ---- 规则参数（原则性取值，与既有冻结工作点对齐；不对真值做网格搜索） ----
P = dict(
    sim_hi=0.95,       # 强证据带（= relay conflict_sim 冻结值）
    sim_lo=0.943,      # 可判下界（= relay sim_thr 冻结值）
    sim_reject=0.943,  # 低于冻结声称阈值 → 隔离（系统自身不应声称此类边）
    w_dt_min=0.30,     # Parzen 时间合理性
    gate_r=0.15,       # 门邻近半径（门聚类 radius 0.10 + cell 0.06 的 1.5 倍）
    dt_min=-1.0,       # dt 硬下界
    overlap_tol=1.0,   # 区间相交判定容差（秒）
    dir_support_min=5, # 方向支撑最小值
    dir_bins_min=2,    # 方向至少覆盖的时段分箱数
    h_range=(40, 1500),# 框高合理范围（px）
    bins=(0, 205.6, 411.2, 616.8),  # 10min 窗三段
    overlap_pairs={("318", "320")},  # 视野重叠对豁免
)


def parse_pair(s):
    """'343:343-E011' -> ('343','343-E011')"""
    cam, eid = s.split(":", 1)
    return cam, eid


def load_gil():
    con = sqlite3.connect(GIL)
    cur = con.cursor()
    ent = {}
    for r in cur.execute("SELECT id, cam, eid, t0, t1 FROM entity").fetchall():
        ent[(r[1], r[2])] = dict(id=r[0], t0=r[3], t1=r[4])
    obs_gp = {}
    for r in cur.execute("SELECT node_id, gperson_id FROM membership WHERE node_kind='obs'"):
        obs_gp[r[0]] = r[1]
    obs_ent = {}
    for r in cur.execute("SELECT id, cam, entity_id FROM obs"):
        obs_ent[r[0]] = (r[1], r[2])
    # entity -> gperson 集合（经 obs 成员）
    ent_gp = defaultdict(set)
    for obs_id, (cam, ent_id) in obs_ent.items():
        if ent_id is None:
            continue
        for (c, e), eo in ent.items():
            if eo["id"] == ent_id:
                gp = obs_gp.get(obs_id)
                if gp is not None:
                    ent_gp[(c, e)].add(gp)
                break
    con.close()
    return ent, ent_gp


def load_gates():
    d = json.load(open(C0, encoding="utf-8"))
    gates = {}
    for cam, lst in d.get("gates", {}).items():
        gates[cam] = dict(
            exit=[g["center"] for g in lst if g.get("kind") == "exit"],
            entry=[g["center"] for g in lst if g.get("kind") == "entry"],
        )
    return gates


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=CAND)
    ap.add_argument("--out", default=os.path.join(ANA, "eval_registry"))
    ap.add_argument("--name", default="registry_v0",
                    help="输出文件名（不含 .json）；S3 传导后建议 registry_v1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ent, ent_gp = load_gil()
    gates = load_gates()
    cand = json.load(open(args.cand, encoding="utf-8"))["candidates"]
    edges = json.load(open(EDGES, encoding="utf-8"))["edges"]
    edge_status = {(e["a"], e["b"]): e for e in edges}
    states = json.load(open(STATES, encoding="utf-8"))["states"]
    state_map = {}
    for k, v in states.items():
        a, b = k.split("->")
        state_map[(a, b)] = v

    # 竞争者 margin：同 a 的最佳其它 sim
    by_a = defaultdict(list)
    for c in cand:
        by_a[c["a"]].append(c)
    best_other = {}
    for a, lst in by_a.items():
        lst2 = sorted(lst, key=lambda x: -x["score"])
        for i, c in enumerate(lst2):
            best_other[(c["a"], c["b"])] = lst2[0]["sim"] if i > 0 else (
                lst2[1]["sim"] if len(lst2) > 1 else None)
    # 方向支撑
    dir_cnt = Counter((parse_pair(c["a"])[0], parse_pair(c["b"])[0]) for c in cand)
    bin_set = defaultdict(set)
    for c in cand:
        d = (parse_pair(c["a"])[0], parse_pair(c["b"])[0])
        t = c.get("a_tlast")
        if t is None:
            continue
        for i in range(3):
            if P["bins"][i] <= t < P["bins"][i + 1]:
                bin_set[d].add(i)

    def cam_pair_exempt(ca, cb):
        return tuple(sorted((ca, cb))) in P["overlap_pairs"]

    results = []
    for c in cand:
        ca, ea = parse_pair(c["a"])
        cb, eb = parse_pair(c["b"])
        layers = {}
        reasons = []

        # ---- L1 数据质量 ----
        a_h, b_h = c.get("a_h"), c.get("b_h")
        q_ok = (a_h is not None and b_h is not None
                and P["h_range"][0] <= min(a_h, b_h) and max(a_h, b_h) <= P["h_range"][1])
        ent_ok = (ca, ea) in ent and (cb, eb) in ent
        sim = c.get("sim")
        l1 = "ok" if (q_ok and ent_ok and sim is not None) else "fail"
        if not q_ok:
            reasons.append(f"框高越界 a_h={a_h} b_h={b_h}")
        layers["L1_quality"] = dict(status=l1, a_h=a_h, b_h=b_h, ent=ent_ok)

        # ---- L2 几何 ----
        a_end, b_start = c.get("a_end"), c.get("b_start")
        ga, gb = gates.get(ca, {}), gates.get(cb, {})
        pd_a = min([dist(a_end, g) for g in ga.get("exit", [])], default=None)
        pd_b = min([dist(b_start, g) for g in gb.get("entry", [])], default=None)
        gate_ok = True
        if pd_a is not None and pd_b is not None:
            gate_ok = pd_a <= P["gate_r"] and pd_b <= P["gate_r"]
        elif pd_a is None or pd_b is None:
            gate_ok = None  # 门缺失 → 中性
        inb = (0 <= a_end[0] <= 1 and 0 <= a_end[1] <= 1 and
               0 <= b_start[0] <= 1 and 0 <= b_start[1] <= 1)
        l2 = "ok" if (gate_ok is not False and inb) else ("warn" if gate_ok is False else "ok")
        if gate_ok is False:
            reasons.append(f"门邻近超限 pd_a={pd_a and round(pd_a,3)} pd_b={pd_b and round(pd_b,3)}")
        layers["L2_geometry"] = dict(status=l2, pd_a=None if pd_a is None else round(pd_a, 3),
                                     pd_b=None if pd_b is None else round(pd_b, 3))

        # ---- L3 证据强度 ----
        w_dt, dt, score = c.get("w_dt"), c.get("dt"), c.get("score")
        margin = None
        bo = best_other.get((c["a"], c["b"]))
        if bo is not None and sim is not None:
            margin = round(sim - bo, 4)
        strong = (sim is not None and sim >= P["sim_hi"] and w_dt is not None
                  and w_dt >= P["w_dt_min"] and dt is not None and P["dt_min"] <= dt <= 60)
        mid = sim is not None and sim >= P["sim_lo"]
        l3 = "strong" if strong else ("mid" if mid else "weak")
        layers["L3_evidence"] = dict(status=l3, sim=sim, w_dt=w_dt, dt=dt,
                                     margin_out=margin, score=score)

        # ---- L4 约束 ----
        exempt = cam_pair_exempt(ca, cb)
        ea_t = ent.get((ca, ea), {})
        eb_t = ent.get((cb, eb), {})
        overlap = None
        if ea_t and eb_t:
            ov_len = min(ea_t["t1"], eb_t["t1"]) - max(ea_t["t0"], eb_t["t0"])
            overlap = ov_len > P["overlap_tol"]   # 重叠长度超过容差才判相交
        dt_ok = (dt is None) or dt >= P["dt_min"] or exempt
        hard_fail = False
        if not dt_ok:
            hard_fail = True
            reasons.append(f"dt={dt} 越界")
        if overlap is True and not exempt:
            hard_fail = True
            reasons.append(f"区间相交 [{ea_t['t0']},{ea_t['t1']}]×[{eb_t['t0']},{eb_t['t1']}]")
        l4 = "fail" if hard_fail else "ok"
        layers["L4_constraints"] = dict(status=l4, dt_ok=dt_ok, overlap=overlap, exempt=exempt,
                                        a_iv=[ea_t.get("t0"), ea_t.get("t1")],
                                        b_iv=[eb_t.get("t0"), eb_t.get("t1")])

        # ---- L5 跨机位支持 ----
        dc = dir_cnt.get((ca, cb), 0)
        nb = len(bin_set.get((ca, cb), set()))
        l5 = "ok" if (dc >= P["dir_support_min"] and nb >= P["dir_bins_min"]) else (
            "warn" if dc >= 3 else "fail")
        layers["L5_cross_support"] = dict(status=l5, dir=f"{ca}->{cb}", dir_cnt=dc, bins=nb)

        # ---- L6 身份/下游 ----
        gp_a, gp_b = ent_gp.get((ca, ea), set()), ent_gp.get((cb, eb), set())
        same_gp = bool(gp_a & gp_b) if (gp_a and gp_b) else None
        st = edge_status.get((c["a"], c["b"]))
        stt = state_map.get((c["a"], c["b"]))
        l6 = "ok"
        layers["L6_identity"] = dict(status=l6, same_gperson=same_gp,
                                     relay_status=st["status"] if st else None,
                                     state=stt["state"] if stt else None,
                                     E=stt["E"] if stt else None)

        # ---- 动作 ----
        if l4 == "fail":
            action = "reject"
        elif l1 == "fail":
            action = "reject"
        elif sim is not None and sim < P["sim_reject"]:
            action = "reject"
        elif strong and l2 != "warn":
            action = "keep"
        elif strong and l2 == "warn":
            action = "defer"
            reasons.append("强证据但门几何超限")
        else:
            action = "defer"
        results.append(dict(a=c["a"], b=c["b"], dir=c["dir"], dt=dt, sim=sim, action=action,
                            reasons=reasons, layers=layers))
    idx = {(r["a"], r["b"]): r for r in results}

    # ================= 评测（隔离记录） =================
    def parse_pid(pid):
        if "__" not in pid:      # 非双实体格式（如 S6B_* 裁决项、"316_E_xxx" 单实体）→ 不适用本库
            return None, None
        left, right = pid.split("__")
        la, lb = left.split("_"), right.split("_")
        return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"

    judged = {}
    n_skip_fmt = 0
    if os.path.exists(LABELED):
        for r in csv.DictReader(open(LABELED, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v in ("1", "0", "?"):
                judged[r["pair_id"]] = v
    if os.path.exists(LABELS_JSON):
        d = json.load(open(LABELS_JSON, encoding="utf-8"))
        for pid, rec in d.get("pair", {}).items():
            if "__" not in pid:
                n_skip_fmt += 1
                continue
            judged[pid] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")

    def action_of(pid):
        a, b = parse_pid(pid)
        if a is None:
            return None
        r = idx.get((a, b))
        if r is None:
            return None
        return r["action"]

    acc = dict(n_pos=0, n_neg=0, n_unclear=0, pos_keep=0, pos_defer=0, pos_reject=0,
               neg_reject=0, neg_defer=0, neg_keep=0, un_def=0, un_keep=0, un_rej=0,
               missing=0, pos_missing=0)
    pos_list, neg_list, un_list = [], [], []
    for pid, v in judged.items():
        act = action_of(pid)
        if act is None:
            acc["missing"] += 1
            if v == "1":
                acc["pos_missing"] += 1
            continue
        if v == "1":
            acc["n_pos"] += 1
            acc[f"pos_{'keep' if act == 'keep' else ('defer' if act == 'defer' else 'reject')}"] += 1
            pos_list.append((pid, act))
        elif v == "0":
            acc["n_neg"] += 1
            acc[f"neg_{'reject' if act == 'reject' else ('defer' if act == 'defer' else 'keep')}"] += 1
            neg_list.append((pid, act))
        else:
            acc["n_unclear"] += 1
            acc[f"un_{'def' if act == 'defer' else ('keep' if act == 'keep' else 'rej')}"] += 1
            un_list.append((pid, act))

    # 组裁决回放
    groups = json.load(open(VERDICTS, encoding="utf-8"))["groups"]
    g_acc = dict(n_same=0, same_keep=0, same_defer=0, same_reject=0,
                 n_diff=0, diff_reject=0, diff_defer=0, diff_keep=0, details=[])
    for gk, gv in groups.items():
        for c in gv.get("candidates", []):
            human = c.get("human")
            if human not in ("same", "diff"):
                continue
            pid = c["pair_id"]
            act = action_of(pid)
            row = dict(group=gk, pid=pid, human=human, act=act, verdict=c.get("verdict"))
            g_acc["details"].append(row)
            if human == "same":
                g_acc["n_same"] += 1
                g_acc[f"same_{act if act else 'missing'}"] = g_acc.get(
                    f"same_{act if act else 'missing'}", 0) + 1
            else:
                g_acc["n_diff"] += 1
                g_acc[f"diff_{act if act else 'missing'}"] = g_acc.get(
                    f"diff_{act if act else 'missing'}", 0) + 1

    # ================= 与 relay_states 四态对账 =================
    rec = dict(n=0, agree=0, conflict=0, hard_conflict=0, soft_conflict=0, detail=[])
    for (a, b), stt in state_map.items():
        r = idx.get((a, b))
        if r is None:
            continue
        rec["n"] += 1
        state = stt["state"]
        act = r["action"]
        if act == "defer":
            rec["agree"] += 1
            continue
        ok = (act == "keep" and state in ("active", "candidate")) or (
            act == "reject" and state in ("suspect", "dormant"))
        if ok:
            rec["agree"] += 1
            continue
        # 硬冲突：reject vs active / keep vs dormant；软冲突：reject vs candidate / keep vs suspect
        hard = (act == "reject" and state == "active") or (act == "keep" and state == "dormant")
        rec["conflict"] += 1
        rec["hard_conflict" if hard else "soft_conflict"] += 1
        rec["detail"].append(dict(a=a, b=b, action=act, state=state, E=stt["E"], hard=hard))

    # ================= gperson 建议清单（≥3 成员簇；五动作） =================
    # 数据源：默认 = GIL 库当前 membership（跟随最新落库）；--gp-json 可指定快照对照
    con = sqlite3.connect(GIL)
    cur = con.cursor()
    obs_meta, id2label = {}, {}
    for r in cur.execute("SELECT id, cam, tid, entity_id, t_first, t_last FROM obs"):
        obs_meta[r[0]] = dict(cam=r[1], tid=r[2], ent=r[3], t0=r[4], t1=r[5])
    for r in cur.execute("SELECT id, label FROM gperson"):
        id2label[r[0]] = r[1]
    # obs 级倒影/墙镜注释（S5/S6 落库；target_id = "cam|tid"）
    obs_key = {(v["cam"], v["tid"]): k for k, v in obs_meta.items()}
    contam = set()
    for r in cur.execute("SELECT target_kind, target_id FROM annotation "
                         "WHERE value IN ('reflection','wall_mirror')"):
        if r[0] == "obs":
            c, t = r[1].split("|")
            k = obs_key.get((c, int(t)))
            if k is not None:
                contam.add(k)
    obs_gp = {}
    for r in cur.execute("SELECT node_id, gperson_id FROM membership WHERE node_kind='obs'"):
        obs_gp[r[0]] = id2label.get(r[1])
    con.close()
    # S3 碎片链接：路径覆盖已采纳、但当前身份层未生效的跨簇链 → 合并复核建议
    # （剔除 long_gap：S3 结论"长间隔重认不可解 P 0.071"，不入建议；口径见 docs S3 段）
    fl = json.load(open(os.path.join(ANA, "gil", "frag_link_v1.json"), encoding="utf-8"))["links"]
    merge_pairs, n_longgap_skipped = [], 0
    for lk in fl:
        if lk.get("same_entity"):
            continue
        if (lk.get("cos") or 0) < 0.8:
            continue
        if lk.get("pe_adj") is None or lk["pe_adj"] > 0.08:
            continue
        if lk.get("long_gap"):
            n_longgap_skipped += 1
            continue
        ca, ta = lk["a"].split("|")
        cb, tb = lk["b"].split("|")
        oa, ob = obs_key.get((ca, int(ta))), obs_key.get((cb, int(tb)))
        ga, gb = obs_gp.get(oa), obs_gp.get(ob)
        if ga and gb and ga != gb:
            merge_pairs.append(dict(g1=ga, g2=gb, cos=lk["cos"], pe=lk.get("pe_adj"), dt=lk["dt"]))

    suggestions = []
    # 重建当前 clusters（≥1 成员；跟随库最新 membership）
    byg = {}
    for oid, lab in obs_gp.items():
        if lab is None:
            continue
        byg.setdefault(lab, []).append(oid)
    clusters_src = []
    for lab, ms in byg.items():
        t0s = [obs_meta[m]["t0"] for m in ms if m in obs_meta and obs_meta[m]["t0"] is not None]
        t1s = [obs_meta[m]["t1"] for m in ms if m in obs_meta and obs_meta[m]["t1"] is not None]
        clusters_src.append(dict(
            gid=lab, n=len(ms), obs=sorted(ms),
            cams=sorted({obs_meta[m]["cam"] for m in ms if m in obs_meta}),
            t0=(min(t0s) if t0s else 0), t1=(max(t1s) if t1s else 0)))
    for cl in clusters_src:
        members = cl.get("obs", [])
        n = cl.get("n", len(members))
        span = round(cl.get("t1", 0) - cl.get("t0", 0), 1)
        cams = cl.get("cams", [])
        gid = cl.get("gid")
        n_ref = len([m for m in members if m in contam])
        frac = n_ref / max(n, 1)
        mg = [m for m in merge_pairs if gid in (m["g1"], m["g2"])]
        act, why = "keep", "无违规"
        if frac >= 0.5:
            act, why = "quarantine", f"{n_ref}/{n} 成员为倒影/墙镜（疑幽灵人）"
        elif n_ref > 0:
            act, why = "split", f"{n_ref}/{n} 成员为倒影/墙镜（建议剔除后复审）"
        elif mg:
            act, why = "merge", f"{len(mg)} 条 S3 已采纳链未传导（建议复核合并）"
        elif n >= 7:
            act, why = "defer", "大簇（n≥7）建议人工抽查"
        elif len(cams) >= 4:
            act, why = "defer", f"跨 {len(cams)} 机位建议人工抽查"
        elif span > 2000:
            act, why = "defer", f"长跨度 {span}s 建议人工抽查"
        suggestions.append(dict(gid=gid, n=n, cams=cams, span=span, n_reflection=n_ref,
                                n_merge_links=len(mg), action=act, reason=why))
    gp_counts = dict(Counter(s["action"] for s in suggestions))

    out = dict(
        version="eval_registry/0.1",
        params={**{k: v for k, v in P.items() if k != "overlap_pairs"},
                "overlap_pairs": [list(x) for x in P["overlap_pairs"]]},
        n_candidates=len(results),
        action_counts=dict(Counter(r["action"] for r in results)),
        claimable_pool=dict(
            definition=f"sim>= {P['sim_lo']}（可声称池）",
            n=sum(1 for r in results if r.get("sim") is not None and r["sim"] >= P["sim_lo"]),
            actions=dict(Counter(r["action"] for r in results
                                 if r.get("sim") is not None and r["sim"] >= P["sim_lo"])),
        ),
        acceptance_55=acc, acceptance_groups=g_acc,
        reconcile_states=rec, gperson=dict(n_clusters=len(suggestions),
                                           action_counts=gp_counts,
                                           n_merge_pairs=len(merge_pairs),
                                           n_longgap_skipped=n_longgap_skipped,
                                           suggestions=suggestions),
        candidates=results,
    )
    # 双框架指标：① 精确一致（keep↔same / reject↔diff / defer↔unclear）；② 无自信错误
    denom_same = acc["n_pos"] + g_acc["n_same"]
    denom_diff = acc["n_neg"] + g_acc["n_diff"]
    false_rej = acc["pos_reject"] + g_acc.get("same_reject", 0)
    false_keep = acc["neg_keep"] + g_acc.get("diff_keep", 0)
    out["agreement_summary"] = dict(
        same_total=denom_same, same_keep=acc["pos_keep"] + g_acc.get("same_keep", 0),
        same_defer=acc["pos_defer"] + g_acc.get("same_defer", 0), false_reject=false_rej,
        diff_total=denom_diff, diff_reject=acc["neg_reject"] + g_acc.get("diff_reject", 0),
        diff_defer=acc["neg_defer"] + g_acc.get("diff_defer", 0), false_keep=false_keep,
        unclear_total=acc["n_unclear"], unclear_defer=acc["un_def"],
        missing=acc["missing"] + g_acc.get("same_missing", 0) + g_acc.get("diff_missing", 0),
    )
    outp = os.path.join(args.out, f"{args.name}.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 控制台报告 ----
    print(f"[registry] n_candidates={len(results)} actions={out['action_counts']}")
    if n_skip_fmt:
        print(f"[registry] 跳过非双实体格式判读 {n_skip_fmt} 条（S6B_* 裁决项等，不适用边域库）")
    print(f"[55 判] pos keep={acc['pos_keep']}/{acc['n_pos']} (defer {acc['pos_defer']}, "
          f"reject {acc['pos_reject']}, missing {acc['pos_missing']}) | "
          f"neg reject={acc['neg_reject']}/{acc['n_neg']} (defer {acc['neg_defer']}, keep {acc['neg_keep']}) | "
          f"unclear defer={acc['un_def']}/{acc['n_unclear']} (keep {acc['un_keep']}, rej {acc['un_rej']})")
    print(f"[组裁决] same keep={g_acc.get('same_keep',0)}/{g_acc['n_same']} "
          f"(defer {g_acc.get('same_defer',0)}, reject {g_acc.get('same_reject',0)}, missing {g_acc.get('same_missing',0)}) | "
          f"diff reject={g_acc.get('diff_reject',0)}/{g_acc['n_diff']} "
          f"(defer {g_acc.get('diff_defer',0)}, keep {g_acc.get('diff_keep',0)}, missing {g_acc.get('diff_missing',0)})")
    print(f"[四态对账] n={rec['n']} agree={rec['agree']} conflict={rec['conflict']} "
          f"(hard={rec['hard_conflict']}, soft={rec['soft_conflict']})")
    print(f"[gperson] clusters={len(suggestions)} actions={gp_counts} "
          f"merge_pairs={len(merge_pairs)}")
    print("saved", outp)


if __name__ == "__main__":
    main()
