#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WP2：数据驱动门户发现（visit 事件聚类 → 门户集 v2）+ 与 C0 几何门对照。

背景（research-agenda / visit-portal-semantics-report §5.3 缺口）：
  C0 的门来自 relay 边端点的稀疏聚类（n_pts 4~5），覆盖率有限——343 无 entry 门、
  318/305 覆盖约一半。本实验**不依赖边端点**，直接对 GIL `visit` 表的全部
  entry/exit 穿越事件（穿越位置 u,v + 朝向 dir + 边缘余量 edge_margin）做
  数据驱动的密度聚类，产出「门户集 v2」，并回答：
    ① 哪些 C0 门被 visit 数据支持/证伪；
    ② 哪些区域存在 visit 密集簇但 C0 没有门（缺失门户，重点=343 entry）；
    ③ relay 人工强标签边（confirmed/active vs rejected）的端点事件落在哪些簇
       → 用人类已核验的边给门户簇做"真值级"支撑度排序。

方法（只读；不写任何生产文件）：
  - DBSCAN（手写，确定性）：eps=0.06（归一化坐标），min_pts=3；附 eps 敏感性 0.05/0.08
  - 簇统计：n、中心、半径、edge_margin 中位数、朝向圆均值与 resultant R
  - C0 对照：v9 门，命中半径 0.10（gate_radius）/0.15（visit 旧口径）
  - 边交叉验证：复刻 visit_portal_link 的 pick()（时间最近邻事件）→ 事件→簇映射
输出：analysis/relay_v2/portal_discovery_v1.json + .png（6 机位散点图）
用法：python tools/wp2_portal_discovery.py
"""
import collections
import json
import os
import sqlite3

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
GIL = os.path.join(ANA, "gil", "store_v1", "gil.sqlite")
EDGES = os.path.join(ANA, "relay_v2", "edges_v2_gated_human.json")
C0 = os.path.join(ANA, "c0", "c0_baseline_v9.json")
OUT_JSON = os.path.join(ANA, "relay_v2", "portal_discovery_v1.json")
OUT_PNG = os.path.join(ANA, "relay_v2", "portal_discovery_v1.png")
CAMS = ["305", "316", "318", "320", "323", "343"]
EPS = 0.06
MIN_PTS = 3
GATE_R = 0.10
GATE_R_WIDE = 0.15
TRUE_ST = ("confirmed", "active")
FALSE_ST = ("rejected",)


# ---------------- 手写 DBSCAN（确定性，n≤100） ----------------
def dbscan(points, eps, min_pts):
    pts = np.asarray(points, np.float64)
    n = len(pts)
    if n == 0:
        return np.full(0, -1, int)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    labels = np.full(n, -1, int)
    visited = np.zeros(n, bool)
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = np.where(d[i] <= eps)[0]
        if len(nb) < min_pts:
            continue  # 暂标 noise（可能后续被边界归入其他簇，简化为不回收）
        labels[i] = cid
        seeds = collections.deque(int(x) for x in nb if x != i)
        while seeds:
            j = seeds.popleft()
            if not visited[j]:
                visited[j] = True
                nb2 = np.where(d[j] <= eps)[0]
                if len(nb2) >= min_pts:
                    for x in nb2:
                        if labels[x] == -1 and x not in seeds:
                            seeds.append(int(x))
            if labels[j] == -1:
                labels[j] = cid
        cid += 1
    return labels


def circ_stats(dirs):
    """朝向圆均值（度）与 resultant 长度 R∈[0,1]。"""
    ds = np.asarray([d for d in dirs if d is not None], np.float64)
    if len(ds) == 0:
        return None, None
    rad = np.deg2rad(ds)
    c, s = np.cos(rad).mean(), np.sin(rad).mean()
    R = float(np.hypot(c, s))
    mean = float(np.rad2deg(np.arctan2(s, c)))
    return round(mean, 1), round(R, 3)


def dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def build_clusters(events):
    """events: list[dict(u,v,dir,edge_margin,t,kind)] → 簇列表（含统计）。"""
    pts = [(e["u"], e["v"]) for e in events]
    labels = dbscan(pts, EPS, MIN_PTS)
    clusters = []
    for cid in sorted(set(labels) - {-1}):
        idx = np.where(labels == cid)[0]
        uv = np.asarray([pts[i] for i in idx])
        center = uv.mean(0).round(4).tolist()
        radius = round(float(np.sqrt(((uv - center) ** 2).sum(1)).max()), 4)
        margins = [events[i]["edge_margin"] for i in idx if events[i]["edge_margin"] is not None]
        dir_mean, dir_R = circ_stats([events[i]["dir"] for i in idx])
        clusters.append(dict(id=int(cid), n=int(len(idx)),
                             center=center, radius=radius,
                             med_edge_margin=round(float(np.median(margins)), 4) if margins else None,
                             dir_mean=dir_mean, dir_R=dir_R,
                             members=[int(i) for i in idx]))
    noise = [i for i, l in enumerate(labels) if l == -1]
    return clusters, labels, noise


def main():
    con = sqlite3.connect(GIL)
    con.row_factory = sqlite3.Row
    visits = collections.defaultdict(list)   # (cam, kind) -> [event]
    visits_by_entity = collections.defaultdict(list)  # (cam, entity_id) -> [event]
    for r in con.execute("SELECT cam,kind,t,u,v,dir,edge_margin,entity_id FROM visit "
                         "WHERE entity_id IS NOT NULL"):
        ev = dict(cam=r["cam"], kind=r["kind"], t=r["t"], u=r["u"], v=r["v"],
                  dir=r["dir"], edge_margin=r["edge_margin"])
        visits[(r["cam"], r["kind"])].append(ev)
        visits_by_entity[(r["cam"], r["entity_id"])].append(ev)
    print(f"[visit] {(sum(len(v) for v in visits.values()))} 事件，"
          f"{len(visits)} 个 (cam,kind) 组", flush=True)

    c0 = json.load(open(C0, encoding="utf-8"))["gates"]
    gates = {c: dict(exit=[g["center"] for g in lst if g.get("kind") == "exit"],
                     entry=[g["center"] for g in lst if g.get("kind") == "entry"])
             for c, lst in c0.items()}

    # ---- 逐 (cam,kind)：聚类 + C0 对照 ----
    result = {}
    all_clusters = {}   # (cam,kind) -> clusters
    for cam in CAMS:
        result[cam] = {}
        for kind in ("exit", "entry"):
            evs = visits.get((cam, kind), [])
            clusters, labels, noise = build_clusters(evs)
            # C0 对照
            gs = gates.get(cam, {}).get(kind, [])
            for cl in clusters:
                if gs:
                    ds = [dist(cl["center"], g) for g in gs]
                    j = int(np.argmin(ds))
                    cl["nearest_gate"] = j if ds[j] <= GATE_R else None
                    cl["gate_dist"] = round(ds[j], 4)
                    cl["gate_dist_wide_hit"] = bool(ds[j] <= GATE_R_WIDE)
                else:
                    cl["nearest_gate"] = None
                    cl["gate_dist"] = None
                    cl["gate_dist_wide_hit"] = False
                # 该簇内事件被 C0 门命中比例
                hit = sum(1 for i in cl["members"]
                          if gs and min(dist((evs[i]["u"], evs[i]["v"]), g) for g in gs) <= GATE_R)
                cl["c0_hit_rate"] = round(hit / max(cl["n"], 1), 3)
            # 覆盖率
            n = len(evs)
            c0_hit = sum(1 for e in evs
                         if gs and min(dist((e["u"], e["v"]), g) for g in gs) <= GATE_R) if gs else 0
            c0_hit_wide = sum(1 for e in evs
                              if gs and min(dist((e["u"], e["v"]), g) for g in gs) <= GATE_R_WIDE) if gs else 0
            v2_hit = n - len(noise)
            result[cam][kind] = dict(
                n_events=n, n_clusters=len(clusters), n_noise=len(noise),
                coverage_c0=round(c0_hit / max(n, 1), 3),
                coverage_c0_wide=round(c0_hit_wide / max(n, 1), 3),
                coverage_v2=round(v2_hit / max(n, 1), 3),
                clusters=[{k: v for k, v in cl.items() if k != "members"} for cl in clusters],
            )
            all_clusters[(cam, kind)] = (clusters, labels, evs)

    # ---- eps 敏感性 ----
    sens = {}
    for eps in (0.05, 0.06, 0.08):
        cnt = {}
        for cam in CAMS:
            for kind in ("exit", "entry"):
                evs = visits.get((cam, kind), [])
                labels = dbscan([(e["u"], e["v"]) for e in evs], eps, MIN_PTS)
                cnt[f"{cam}_{kind}"] = int(len(set(labels) - {-1}))
        sens[f"eps_{eps}"] = cnt
    print("[sens]", json.dumps(sens, ensure_ascii=False)[:200], flush=True)

    # ---- C0 门审计（每个门被 visit 支持度） ----
    gate_audit = {}
    for cam in CAMS:
        gate_audit[cam] = {}
        for kind in ("exit", "entry"):
            evs = visits.get((cam, kind), [])
            rows = []
            for gi, g in enumerate(gates.get(cam, {}).get(kind, [])):
                sup = [e for e in evs if dist((e["u"], e["v"]), g) <= GATE_R]
                rows.append(dict(gate_idx=gi, center=g, support_n=len(sup)))
            gate_audit[cam][kind] = rows

    # ---- 新门户（簇无 C0 门覆盖 且 簇支撑较强） ----
    new_portals = []
    for (cam, kind), (clusters, labels, evs) in all_clusters.items():
        for cl in clusters:
            if cl["nearest_gate"] is None and cl["n"] >= 3:
                new_portals.append(dict(cam=cam, kind=kind, n=cl["n"],
                                        center=cl["center"], radius=cl["radius"],
                                        med_edge_margin=cl["med_edge_margin"],
                                        gate_dist=cl["gate_dist"],
                                        wide_hit=cl["gate_dist_wide_hit"]))
    new_portals.sort(key=lambda d: -d["n"])

    # ---- relay 边交叉验证：真/假边事件 → 簇 ----
    ed = json.load(open(EDGES, encoding="utf-8"))
    edges = ed["edges"]
    eid2id = {(r["cam"], r["eid"]): r["id"] for r in
              con.execute("SELECT id,cam,eid FROM entity")}
    ent = {r["id"]: dict(r) for r in con.execute("SELECT id,cam,eid,t0,t1 FROM entity")}

    def pick(cam, eid, kind, target_t):
        i = eid2id.get((cam, eid))
        if i is None:
            return None
        evs = [v for v in visits_by_entity.get((cam, i), []) if v["kind"] == kind]
        if not evs:
            return None
        return min(evs, key=lambda v: abs(v["t"] - (target_t if target_t is not None else 0)))

    def ev_to_cluster(cam, kind, ev):
        clusters, labels, evs = all_clusters[(cam, kind)]
        if ev is None:
            return None
        # 优先：事件在簇成员里（同一条记录）；否则最近簇中心 ≤ EPS
        for cl in clusters:
            for i in cl["members"]:
                if evs[i]["t"] == ev["t"] and evs[i]["u"] == ev["u"]:
                    return cl["id"]
        best, bd = None, 1e9
        for cl in clusters:
            dd = dist(cl["center"], (ev["u"], ev["v"]))
            if dd < bd:
                bd, best = dd, cl["id"]
        return best if bd <= EPS else None

    sup = collections.defaultdict(lambda: dict(true_edges=[], false_edges=[]))
    edge_events = []
    for e in edges:
        ca, eida = e["a"].split(":")
        cb, eidb = e["b"].split(":")
        ia, ib = eid2id.get((ca, eida)), eid2id.get((cb, eidb))
        ta = (ent.get(ia, {}) or {}).get("t1", e.get("a_tlast"))
        tb = (ent.get(ib, {}) or {}).get("t0", e.get("b_tfirst"))
        ex = pick(ca, eida, "exit", ta)
        en = pick(cb, eidb, "entry", tb)
        ce = ev_to_cluster(ca, "exit", ex) if ex else None
        cn = ev_to_cluster(cb, "entry", en) if en else None
        st = e["status"]
        rec = dict(edge=e["a"] + "->" + e["b"], status=st,
                   exit_cluster=ce, entry_cluster=cn)
        edge_events.append(dict(edge=rec["edge"], status=st,
                                a_cam=ca, b_cam=cb,
                                exit_uv=[round(ex["u"], 3), round(ex["v"], 3)] if ex else None,
                                exit_cluster=ce,
                                entry_uv=[round(en["u"], 3), round(en["v"], 3)] if en else None,
                                entry_cluster=cn))
        for (cam, kind, cid) in ((ca, "exit", ce), (cb, "entry", cn)):
            if cid is None:
                continue
            key = (cam, kind, cid)
            if st in TRUE_ST:
                sup[key]["true_edges"].append(rec["edge"])
            elif st in FALSE_ST:
                sup[key]["false_edges"].append(rec["edge"])

    edge_support = [dict(cam=c, kind=k, cluster=cid,
                         n_true=len(v["true_edges"]), n_false=len(v["false_edges"]),
                         true_edges=v["true_edges"], false_edges=v["false_edges"])
                    for (c, k, cid), v in sorted(sup.items())]

    # ---- 真边事件的未映射审计（暴露聚类漏掉的门户区域） ----
    unmapped = collections.defaultdict(list)
    for ee in edge_events:
        if ee["status"] not in TRUE_ST:
            continue
        if ee["exit_uv"] is not None and ee["exit_cluster"] is None:
            unmapped[f"{ee['a_cam']}_exit"].append(ee["edge"] + f" @{ee['exit_uv']}")
        if ee["entry_uv"] is not None and ee["entry_cluster"] is None:
            unmapped[f"{ee['b_cam']}_entry"].append(ee["edge"] + f" @{ee['entry_uv']}")
    print("\n=== 真边事件未被任何簇覆盖（按 cam_kind 分组）===")
    for k in sorted(unmapped):
        print(f"  {k}: {len(unmapped[k])} 条")
        for s in unmapped[k][:6]:
            print(f"      {s}")

    out = dict(version="portal_discovery_v1",
               params=dict(eps=EPS, min_pts=MIN_PTS, gate_r=GATE_R, gate_r_wide=GATE_R_WIDE),
               c0=C0, edges=EDGES,
               per_cam=result, eps_sensitivity=sens,
               gate_audit=gate_audit, new_portals=new_portals,
               edge_support=edge_support,
               edge_events=edge_events,
               unmapped_true_events={k: v for k, v in unmapped.items()},
               note="只读；visit 事件数据驱动门户发现，与 C0 几何门对照；边交叉验证用人工状态")
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 打印摘要 ----
    print("\n=== 逐机位门户发现（eps=0.06, min_pts=3）===")
    for cam in CAMS:
        for kind in ("exit", "entry"):
            r = result[cam][kind]
            cl_desc = " ".join(f"[{cl['center']} n={cl['n']}"
                               + (f" gate#{cl['nearest_gate']}" if cl["nearest_gate"] is not None
                                  else f" gate_d={cl['gate_dist']}")
                               + "]" for cl in r["clusters"])
            print(f"  {cam} {kind:5s}: 事件={r['n_events']} 簇={r['n_clusters']} 噪声={r['n_noise']} "
                  f"C0覆盖={r['coverage_c0']}(w:{r['coverage_c0_wide']}) v2覆盖={r['coverage_v2']}")
            if cl_desc:
                print(f"      {cl_desc}")
    print("\n=== C0 门审计（support_n<2 视为弱支持）===")
    for cam in CAMS:
        for kind in ("exit", "entry"):
            rows = gate_audit[cam][kind]
            if rows or gates.get(cam, {}).get(kind):
                print(f"  {cam} {kind:5s}: " + " ".join(f"门{i}@{g['center']} sup={g['support_n']}"
                                                        for i, g in enumerate(rows))
                      if rows else f"  {cam} {kind:5s}: (C0 无门)")
    print("\n=== 疑似新门户（visit 簇>=3 且无 C0 门覆盖）===")
    for d in new_portals[:20]:
        print(f"  {d['cam']} {d['kind']:5s} n={d['n']:>2} center={d['center']} "
              f"半径={d['radius']} 边缘余量={d['med_edge_margin']} 最近门距离={d['gate_dist']}")
    print("\n=== 簇的边级支撑（人工标签）Top ===")
    es = sorted(edge_support, key=lambda d: (-d["n_true"], d["n_false"]))
    for d in es[:15]:
        print(f"  {d['cam']} {d['kind']:5s} 簇{d['cluster']}: 真边支撑={d['n_true']} 假边={d['n_false']}")
    print("\nsaved", OUT_JSON)

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=130)
        for ai, cam in enumerate(CAMS):
            ax = axes[ai // 3][ai % 3]
            for kind, color in (("entry", "#1f77b4"), ("exit", "#d62728")):
                evs = visits.get((cam, kind), [])
                if evs:
                    ax.scatter([e["u"] for e in evs], [e["v"] for e in evs],
                               s=14, alpha=0.55, c=color, label=f"{kind}(n={len(evs)})")
                clusters, _, _ = all_clusters[(cam, kind)]
                for cl in clusters:
                    ax.add_patch(plt.Circle(cl["center"], cl["radius"] + EPS / 2, fill=False,
                                            color=color, lw=1.0, ls="--"))
            for kind, marker in (("entry", "^"), ("exit", "s")):
                for g in gates.get(cam, {}).get(kind, []):
                    ax.scatter([g[0]], [g[1]], marker=marker, s=90, facecolors="none",
                               edgecolors="#2ca02c", linewidths=1.8,
                               label=("C0 门" if (g[0], g[1]) == gates[cam][kind][0] else None))
            ax.set_title(f"{cam}")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(1.03, -0.03)   # 图像坐标：v 向下
            ax.grid(alpha=0.25, lw=0.5)
            if ai == 0:
                ax.legend(fontsize=7, loc="lower right")
        fig.suptitle("WP2 数据驱动门户发现：visit entry/exit 事件簇（虚线圆） vs C0 几何门（绿）", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT_PNG)
        print("saved", OUT_PNG)
    except Exception as e:  # noqa: BLE001
        print("[warn] figure skipped:", e)


if __name__ == "__main__":
    main()
