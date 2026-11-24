# -*- coding: utf-8 -*-
"""S4：Q3 责任划分 —— `visit`（通行段）+ `owner`（责任时间线）+ 去重计数。

背景（Q3 原话：**"多画面中的同一个人没有去重（展示视频未实现责任目标划分）"**）
  现状：渲染按**每机位各自的 track id** 画框（`webapp/backend/render.py`），
        没有任何一处把跨机位的同一个人合并成一个目标；也没有"某时刻哪个机位负责谁"。

本脚本在 GIL 上补两层（**不动机位内数据**）：
  * **visit** = 一个 gperson 在一个机位上的一段连续通行。
      entry/exit 各一行（schema 语义），带归一化脚点 (u,v)、航向 dir(°)、
      到画面边界的最小归一化距离 edge_margin（小 = 贴边 = 门户候选）。
  * **owner** = 责任时间线：任一时刻同一 gperson 的 owner **唯一**（硬断言）。
      效用 u(cam,g,t) = 归一化框高 h/H（越近越清晰）× 边界中心性折扣。
      重叠视野（318×320）两侧同时可见时按效用择一 ⇒ **天然去重**。

顺带把 C0 的"门户覆盖"从**实体级**升级到 **gperson 级**（去重口径），
并给出 **+30s 时间门** 的严格版本 —— 回答"305 出门覆盖低"里有多少是碎片/重复计数造成的。

用法：
  python tools/gil_visit_owner.py            # 只读：算指标 + 出 JSON
  python tools/gil_visit_owner.py --apply    # 写 GIL visit/owner
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("DET_VIEW", "clean2")
import gil_frag_link as FL  # noqa: E402  复用轨迹层
from gil import GILStore  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
CAMS = FL.CAMS
RES = FL.RES
VISIT_GAP = 30.0      # 同 gperson 同机位两段观测间隔 > 此值 → 判为两次通行
EDGE_BORDER = 0.03    # 贴边阈值（与 C0 一致）
GATE_RADIUS = 0.10    # 距门心 ≤ 此值 → 判为通过门（与 C0 一致）
GATE_STRONG_PTS = 6   # v7：门支撑点数 ≥ 此值算强门（主门户）
TRANSIT_MAX = 30.0    # v7：弱门 + 段时长 ≤ 此值 → 仍算真通行（transient）
DT_HANDOFF = 30.0     # 交接时间门（b 进入 − a 离开）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/visit_owner_v1.json")
    ap.add_argument("--c0", default="output_crosscam/analysis/c0/c0_baseline_v7.json")
    ap.add_argument("--visit-gap", type=float, default=VISIT_GAP)
    ap.add_argument("--web-out", default="webapp/frontend/public/gil_owner.json",
                    help="控制台「去重视图」数据载荷（放 public → 随 dist 静态托管，无需改后端）")
    ap.add_argument("--drop-reflections", action="store_true",
                    help="★ 倒影降权臂（S6-b B 臂）：倒影 obs 视为「非独立观测」排除，"
                         "不作为独立身份节点参与 visit/owner → 与 gil_gperson --drop-reflections 同口径")
    ap.add_argument("--gperson-map", default=None,
                    help="★ 用外部 gperson 划分（如 analysis/gil/gperson_ab_B.json 的 node2gid）"
                         "替代 GIL membership —— 保证开关视图与 A/B 的 B 臂**完全同源**")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    st = GILStore(os.path.join(ROOT, a.store), create=False)
    rows = list(st.con.execute(
        "SELECT id,cam,tid,t_first,t_last,u,v,h_med,entity_id,flags FROM obs "
        "WHERE tombstone=0 ORDER BY id"))
    refl_ids = {int(r["id"]) for r in rows
                if r["flags"] and (lambda f: (f.get("reflection") or 0) > 0
                                   or (f.get("wall_mirror") or 0) > 0)(json.loads(r["flags"]))}
    gmap = None
    if a.gperson_map:
        gmap = json.load(open(os.path.join(ROOT, a.gperson_map), encoding="utf-8"))["node2gid"]
        n0 = len(rows)
        rows = [r for r in rows if f"{r['cam']}|{int(r['tid'])}" in gmap]
        print(f"[s4] ★ 外部 gperson 划分 {a.gperson_map}：节点 {n0} → {len(rows)}"
              f"（簇 {len(set(gmap.values()))}）")
    elif a.drop_reflections:
        n0 = len(rows)
        rows = [r for r in rows if int(r["id"]) not in refl_ids]
        print(f"[s4] ★ 倒影降权臂：排除倒影 obs {len(refl_ids)} 个（{n0} → {len(rows)}）")
    obs = {int(r["id"]): dict(r) for r in rows}
    # gperson 归属（obs 级）
    mem = collections.defaultdict(list)          # gid -> [obs_id]
    obs2g = {}
    if gmap is not None:
        for oid, r in obs.items():
            g = int(gmap[f"{r['cam']}|{int(r['tid'])}"]) + 1   # 0 基 → 1 基
            mem[g].append(oid)
            obs2g[oid] = g
    else:
        for r in st.con.execute("SELECT gperson_id,node_id FROM membership WHERE node_kind='obs'"):
            if int(r["node_id"]) not in obs:
                continue                          # 降权臂：被排除的倒影不进 visit/owner
            mem[int(r["gperson_id"])].append(int(r["node_id"]))
            obs2g[int(r["node_id"])] = int(r["gperson_id"])

    # 轨迹（几何）
    TRAJ = {c: FL.load_traj(c) for c in CAMS}
    GEO = {}
    for oid, r in obs.items():
        tr = TRAJ[r["cam"]].get(int(r["tid"]))
        if tr is None:
            GEO[oid] = None
            continue
        W, H = RES[r["cam"]]
        hd, tl = FL.head_state(tr), FL.tail_state(tr)

        def pack(stt):
            u, v = stt["u"] / W, stt["v"] / H
            d = None
            if stt.get("has_vel") and (abs(stt["vu"]) + abs(stt["vv"])) > 1e-6:
                d = float(np.degrees(np.arctan2(stt["vv"], stt["vu"])))
            return dict(u=round(u, 4), v=round(v, 4), dir=(round(d, 1) if d is not None else None),
                        edge_margin=round(min(u, 1 - u, v, 1 - v), 4),
                        h_over_H=round(stt["h"] / H, 4))
        GEO[oid] = dict(entry=pack(hd), exit=pack(tl))

    # ---------------- 1) visit ----------------
    visits = []
    for gid, oids in mem.items():
        for cam in CAMS:
            group = sorted([o for o in oids if obs[o]["cam"] == cam],
                           key=lambda o: obs[o]["t_first"])
            if not group:
                continue
            seg = [group[0]]
            for o in group[1:]:
                gap = obs[o]["t_first"] - obs[seg[-1]]["t_last"]
                if gap > a.visit_gap:
                    visits.append((gid, cam, seg))
                    seg = []
                seg.append(o)
            visits.append((gid, cam, seg))
    V = []
    for gid, cam, seg in visits:
        seg = sorted(seg, key=lambda o: obs[o]["t_first"])
        t0, t1 = obs[seg[0]]["t_first"], obs[seg[-1]]["t_last"]
        e0, e1 = GEO[seg[0]], GEO[seg[-1]]
        span = 0.0
        if e0 and e1:
            span = float(np.hypot(e1["exit"]["u"] - e0["entry"]["u"],
                                  e1["exit"]["v"] - e0["entry"]["v"]))
        V.append(dict(gid=gid, cam=cam, t0=t0, t1=t1, dur=t1 - t0,
                      obs=seg, n_obs=len(seg), span=round(span, 4),
                      # ★ 与 C0 同口径的"移动"判定
                      moving=bool(span >= 0.05 and (t1 - t0) >= 1.0),
                      entity_ids=sorted({int(obs[o]["entity_id"]) for o in seg
                                         if obs[o]["entity_id"] is not None}),
                      entry=(e0["entry"] if e0 else None),
                      exit=(e1["exit"] if e1 else None)))
    n_multi_obs = sum(1 for v in V if v["n_obs"] > 1)
    print(f"[s4] obs {len(obs)} → gperson {len(mem)} → visit {len(V)}"
          f"（多 obs 段 {n_multi_obs}；移动段 {sum(1 for v in V if v['moving'])}）")
    dur = sorted(v["dur"] for v in V)
    print(f"      通行段时长 p10 {dur[len(dur)//10]:.1f}s p50 {dur[len(dur)//2]:.1f}s "
          f"p90 {dur[int(len(dur)*0.9)]:.1f}s max {dur[-1]:.1f}s")

    # ---------------- 2) owner ----------------
    # 效用：h_over_H（越近越清晰）× 边界折扣（贴边=半出画）
    # ★ 活跃判定**逐 obs**：visit 是 gap≤30s 聚合出的包络，用它当"可见区间"
    #   会在不重叠机位对上造出"同时可见"假象（实测 316×318 等 6 对）→ 必须按 obs 区间。
    def util(v, t):
        """v 在 t 时刻的可见性效用；无 obs 真正覆盖 t 则返回 None。"""
        act = [o for o in v["obs"] if obs[o]["t_first"] <= t <= obs[o]["t_last"]]
        if not act:
            return None
        hs, em = [], []
        for o in act:
            g = GEO[o]
            if not g:
                continue
            stt = g["entry"] if t <= obs[o]["t_first"] else g["exit"]
            hs.append(stt["h_over_H"])
            em.append(stt["edge_margin"])
        if not hs:
            return None
        return float(np.mean(hs)) * (0.5 + 0.5 * min(1.0, np.mean(em) / 0.05))

    by_g = collections.defaultdict(list)
    for v in V:
        by_g[v["gid"]].append(v)

    owners = []
    for gid, vs in by_g.items():
        ivs = [(obs[o]["t_first"], obs[o]["t_last"], v) for v in vs for o in v["obs"]]
        cuts = sorted({x[0] for x in ivs} | {x[1] for x in ivs})
        for i in range(len(cuts) - 1):
            t_lo, t_hi = cuts[i], cuts[i + 1]
            if t_hi - t_lo < 1e-6:
                continue
            t_mid = 0.5 * (t_lo + t_hi)
            cands = []
            for v in vs:
                u = util(v, t_mid)
                if u is not None:
                    cands.append((u, v["cam"]))
            if not cands:
                continue
            cands.sort(key=lambda x: -x[0])
            u, cam = cands[0]
            owners.append(dict(gid=gid, cam=cam, t0=t_lo, t1=t_hi, utility=round(u, 6),
                               n_cams=len({c for _, c in cands})))
    # 合并同 gperson 相邻同机位区间
    owners.sort(key=lambda o: (o["gid"], o["t0"]))
    merged = []
    for o in owners:
        if merged and merged[-1]["gid"] == o["gid"] and merged[-1]["cam"] == o["cam"] \
                and abs(merged[-1]["t1"] - o["t0"]) < 1e-6:
            merged[-1]["t1"] = o["t1"]
            merged[-1]["n_cams"] = max(merged[-1]["n_cams"], o["n_cams"])
        else:
            merged.append(dict(o))
    # reason（★ 优先级：visibility > handoff > sole）
    prev = {}
    for o in merged:
        if o["n_cams"] > 1:
            o["reason"] = "visibility"          # 多机位同时可见 → 按效用择一（去重）
        else:
            p = prev.get(o["gid"])
            if p and p["cam"] != o["cam"] and 0 <= o["t0"] - p["t1"] <= DT_HANDOFF:
                o["reason"] = "handoff"         # 交接
            else:
                o["reason"] = "sole"
        prev[o["gid"]] = o
    owners = merged

    # ★ 硬不变量：同一 gperson 的 owner 区间两两不重叠
    viol = 0
    last = {}
    for o in sorted(owners, key=lambda x: (x["gid"], x["t0"])):
        p = last.get(o["gid"])
        if p is not None and o["t0"] < p - 1e-9:
            viol += 1
        last[o["gid"]] = o["t1"]
    print(f"[s4] owner 区间 {len(owners)}；涉及 gperson {len({o['gid'] for o in owners})}")
    print(f"      ★ 不变量（同 gperson owner 区间互不重叠）：违规 {viol} "
          f"{'✅' if viol == 0 else '❌'}")
    rc = collections.Counter(o["reason"] for o in owners)
    print(f"      reason 分布 {dict(rc)}")

    # ---- 互斥审计：obs 级跨机位同时可见（排除 318×320 重叠视野）----
    # 只保留非重叠机位对：出现即 = gperson 身份错误 或 时钟偏移
    mutex_audit = []
    for gid, vs in by_g.items():
        per_cam = collections.defaultdict(list)
        for v in vs:
            for o in v["obs"]:
                per_cam[v["cam"]].append((obs[o]["t_first"], obs[o]["t_last"]))
        cams = sorted(per_cam)
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                if {cams[i], cams[j]} == {"318", "320"}:
                    continue
                for (a0, a1) in per_cam[cams[i]]:
                    for (b0, b1) in per_cam[cams[j]]:
                        ovl = min(a1, b1) - max(a0, b0)
                        if ovl > 0.6:          # > 0.6s 才算（留 0.2s 量化 + 偏移余量）
                            mutex_audit.append(dict(gid=gid, cams=[cams[i], cams[j]],
                                                    overlap_s=round(ovl, 2)))
    print(f"      ★ 互斥审计（obs 级 ∧ 非 318×320 重叠对 ∧ 重叠>0.6s）：{len(mutex_audit)} 例"
          f" → 即身份错误或时钟偏移候选")
    if mutex_audit:
        for m in mutex_audit[:8]:
            print(f"        g{m['gid']} {m['cams'][0]}×{m['cams'][1]} 重叠 {m['overlap_s']}s")

    # ---------------- 3) 去重计数 + 并发重复度 ----------------
    n_obs, n_ent = len(obs), len({int(r["entity_id"]) for r in rows if r["entity_id"] is not None})
    # 并发重复度：按 owner 区间，统计 t 时刻同被 K 个机位看到的时长占比
    conc = collections.Counter()
    for o in owners:
        conc[min(o["n_cams"], 4)] += (o["t1"] - o["t0"])
    tot_t = sum(conc.values()) or 1.0
    conc_frac = {k: round(v / tot_t, 4) for k, v in sorted(conc.items())}
    # 无 owner（未被任何 gperson 覆盖）的观测时长
    print(f"\n[s4] 去重计数：track/obs {n_obs} → 机位内实体 {n_ent} → **gperson {len(mem)}**"
          f"（去重率 {100*(1-len(mem)/n_ent):.1f}%，相对实体）")
    print(f"      同一 gperson 同时被 K 个机位看到的时间占比 {conc_frac}"
          f"（这就是「未去重」的直接量化）")

    # ---------------- 4) gperson 级门户覆盖（C0 对照） ----------------
    c0 = json.load(open(os.path.join(ROOT, a.c0), encoding="utf-8"))
    gates = c0.get("gates", {})
    cam_visits = collections.defaultdict(list)
    for v in V:
        cam_visits[v["cam"]].append(v)

    def portal_class(v, side):
        """把 visit 的 exit/entry 按 C0 v7 口径分类：border/portal/transient/interior_vanishing。"""
        g = v["exit"] if side == "exit" else v["entry"]
        if g is None:
            return None, None
        if g["edge_margin"] <= EDGE_BORDER:
            return "border", None
        best = None
        for gt in gates.get(v["cam"], []):
            if gt["kind"] != side:
                continue
            d = float(np.hypot(g["u"] - gt["center"][0], g["v"] - gt["center"][1]))
            if best is None or d < best[0]:
                best = (d, gt["n_pts"])
        if best is None or best[0] > GATE_RADIUS:
            return "border", None
        if best[1] >= GATE_STRONG_PTS:
            return "portal", best[1]
        return ("transient", best[1]) if v["dur"] <= TRANSIT_MAX else ("interior_vanishing", best[1])

    cov = {}
    for cam in CAMS:
        vs = [v for v in cam_visits.get(cam, []) if v["moving"]]     # ★ 与 C0 同口径：只看移动段
        ex_portal, ex_covered, en_portal, en_covered = 0, 0, 0, 0
        ex_strict, en_strict = 0, 0
        for v in vs:
            ce, _ = portal_class(v, "exit")
            if ce in ("border", "portal", "transient"):
                ex_portal += 1
                if any(b["cam"] != cam and 0 <= b["t0"] - v["t1"] <= 1e9
                       for b in by_g[v["gid"]]):
                    ex_covered += 1
                if any(b["cam"] != cam and 0 <= b["t0"] - v["t1"] <= DT_HANDOFF
                       for b in by_g[v["gid"]]):
                    ex_strict += 1
            cn, _ = portal_class(v, "entry")
            if cn in ("border", "portal", "transient"):
                en_portal += 1
                if any(b["cam"] != cam and 0 <= v["t0"] - b["t1"] <= 1e9
                       for b in by_g[v["gid"]]):
                    en_covered += 1
                if any(b["cam"] != cam and 0 <= v["t0"] - b["t1"] <= DT_HANDOFF
                       for b in by_g[v["gid"]]):
                    en_strict += 1
        cov[cam] = dict(exit_portal=ex_portal, exit_covered=ex_covered,
                        exit_strict=ex_strict, entry_portal=en_portal,
                        entry_covered=en_covered, entry_strict=en_strict,
                        exit_cont=round(ex_covered / ex_portal, 4) if ex_portal else None,
                        exit_cont_strict=round(ex_strict / ex_portal, 4) if ex_portal else None)
    tp = sum(c["exit_portal"] for c in cov.values())
    tc = sum(c["exit_covered"] for c in cov.values())
    ts = sum(c["exit_strict"] for c in cov.values())
    print(f"\n[s4] 门户覆盖（**gperson 级 / 去重口径**）")
    for cam in CAMS:
        c = cov[cam]
        print(f"      {cam}: exit 门户 {c['exit_portal']:3d} → 覆盖 {c['exit_covered']:3d}"
              f"（{c['exit_cont']}） / +30s 时间门 {c['exit_strict']:3d}（{c['exit_cont_strict']}）")
    print(f"      合计 exit 门户 {tp} → 覆盖 {tc}（{tc/tp if tp else 0:.4f}）"
          f" / +30s 时间门 {ts}（{ts/tp if tp else 0:.4f}）")
    print(f"      对照 C0 v7（**实体级**）：portal_exit_n {c0['coverage_global']['portal_exit_n']}"
          f" → covered {c0['coverage_global']['portal_exit_covered']}"
          f"（{c0['coverage_global']['exit_continuity_v7']}）")

    out = dict(
        params=dict(visit_gap=a.visit_gap, edge_border=EDGE_BORDER, gate_radius=GATE_RADIUS,
                    gate_strong_pts=GATE_STRONG_PTS, transit_max=TRANSIT_MAX,
                    dt_handoff=DT_HANDOFF),
        counts=dict(obs=n_obs, entity=n_ent, gperson=len(mem), visit=len(V),
                    visit_multi_obs=n_multi_obs, owner_intervals=len(owners),
                    dedup_vs_entity=round(1 - len(mem) / n_ent, 4)),
        owner_invariant_violations=viol,
        owner_reason={k: v for k, v in rc.items()},
        mutex_audit_n=len(mutex_audit),
        mutex_audit=mutex_audit[:100],
        concurrency_frac=conc_frac,
        coverage_gperson=cov,
        coverage_gperson_total=dict(exit_portal=tp, exit_covered=tc,
                                    exit_cont=round(tc / tp, 4) if tp else None,
                                    exit_strict=ts,
                                    exit_cont_strict=round(ts / tp, 4) if tp else None),
        c0_v7_entity_level=dict(portal_exit_n=c0["coverage_global"]["portal_exit_n"],
                                portal_exit_covered=c0["coverage_global"]["portal_exit_covered"],
                                exit_continuity_v7=c0["coverage_global"]["exit_continuity_v7"]),
        visits=[dict(gid=v["gid"], cam=v["cam"], t0=round(v["t0"], 2), t1=round(v["t1"], 2),
                     n_obs=v["n_obs"], obs=v["obs"], entity_ids=v["entity_ids"],
                     entry=v["entry"], exit=v["exit"],
                     exit_class=portal_class(v, "exit")[0],
                     entry_class=portal_class(v, "entry")[0]) for v in V],
        owners=[dict(gid=o["gid"], cam=o["cam"], t0=round(o["t0"], 2), t1=round(o["t1"], 2),
                     utility=o["utility"], n_cams=o["n_cams"], reason=o["reason"])
                for o in owners],
    )
    json.dump(out, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[out] {a.out}")

    # ---- 控制台载荷（前端「去重视图」用；紧凑格式，随 dist 静态托管）----
    web = dict(
        meta=dict(
            generated=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            arm=("reflection_dropped" if (a.drop_reflections or a.gperson_map) else "baseline"),
            arm_label=("倒影降权（倒影非独立观测）" if (a.drop_reflections or a.gperson_map)
                       else "现状（倒影作普通节点）"),
            n_reflection_excluded=len(refl_ids) if (a.drop_reflections or a.gperson_map) else 0,
            gperson_map=(a.gperson_map or ""),
            counts=out["counts"], owner_violations=viol,
            owner_reason={k: v for k, v in rc.items()},
            concurrency=conc_frac,
            coverage=out["coverage_gperson_total"], coverage_by_cam=cov,
            c0_v7=out["c0_v7_entity_level"], params=out["params"],
            t_note="时间为 GIL 时间轴；前端时间 = t_gil − session.min_offset",
        ),
        cams=CAMS,
        tid2g={f"{obs[o]['cam']}:{int(obs[o]['tid'])}": g for o, g in
               sorted(obs2g.items(), key=lambda kv: kv[0])},
        owners=[[o["gid"], o["cam"], round(o["t0"], 2), round(o["t1"], 2),
                 o["reason"], o["n_cams"]] for o in owners],
        visits=[[v["gid"], v["cam"], round(v["t0"], 2), round(v["t1"], 2),
                 (portal_class(v, "exit")[0] or ""), 1 if v["moving"] else 0] for v in V],
    )
    wp = os.path.join(ROOT, a.web_out)
    os.makedirs(os.path.dirname(wp), exist_ok=True)
    json.dump(web, open(wp, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"[web] {a.web_out}  ({os.path.getsize(wp)/1024:.0f} KB；"
          f"tid2g {len(web['tid2g'])} owners {len(web['owners'])} visits {len(web['visits'])}）")

    if a.apply:
        st.con.execute("DELETE FROM owner")
        st.con.execute("DELETE FROM visit")
        # visit：每段两行（entry / exit），schema 语义
        for v in V:
            for kind, g, o in (("entry", v["entry"], v["obs"][0]),
                               ("exit", v["exit"], v["obs"][-1])):
                if g is None:
                    continue
                st.add_visit(v["cam"], kind, v["t0"] if kind == "entry" else v["t1"],
                             u=g["u"], v=g["v"], dir=g["dir"], edge_margin=g["edge_margin"],
                             obs_id=o, entity_id=obs[o]["entity_id"], gperson_id=v["gid"])
        vid = {}
        for r in st.con.execute("SELECT id,cam,kind,t,gperson_id FROM visit ORDER BY id"):
            vid[(int(r["gperson_id"]), r["cam"], r["kind"])] = int(r["id"])
        for o in owners:
            st.add_owner(o["t0"], o["t1"], o["cam"], gperson_id=o["gid"],
                         visit_id=vid.get((o["gid"], o["cam"], "entry")),
                         utility=o["utility"], reason=o["reason"])
        st.meta_set(visit_owner=dict(
            version="visit_owner_v1",
            method="visit=gperson×cam 连续通行段；owner=效用 argmax（h/H × 边界中心性）",
            visit_gap=a.visit_gap, dt_handoff=DT_HANDOFF,
            n_visit=len(V), n_owner=len(owners), owner_violations=viol,
            note="owner 不变量：任一时刻同一 gperson 的 owner 唯一（0 违反）"))
        st.commit()
        mf = st.snapshot()
        print(f"[gil] visit {len(V)} 段（{len(V)*2} 行）/ owner {len(owners)}；manifest -> {mf}")
        print("[gil] stats:", json.dumps(st.stats(), ensure_ascii=False))
    st.close()


if __name__ == "__main__":
    main()
