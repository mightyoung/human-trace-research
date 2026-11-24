# -*- coding: utf-8 -*-
"""C0 无深度跨机位接力基线（对照组）。

定位：C0 = 不使用任何深度信号 的跨机位 handoff 系统，即当前生产链
      时间门(统一 OSD 时间轴) → 外观门(末碎片⇄首碎片 cos) → 全局 mutual-best 唯一化。
      它是 E3/E4（引入 C-RADIO 深度头后）的**对照组**，必须先把基线数字钉死。

本脚本做四件事：
  1) 实体盘点      —— 六路 net_tracks 实体/碎片规模、时长分布
  2) handoff 图体检 —— 边数、方向矩阵、出/入度（唯一化是否真的 1 对 1）
  3) 门（gate）发现 + 覆盖率 —— 对移动实体的首/末脚点做密度聚类得"门"，
                                统计"出门静默率"(exit continuity) 与"进门归属率"(entry attribution)
                                —— 这是 C0 口径下"目标是否丢失"的可测代理
  4) 318×320 GT 校验 —— 8 对人工确认对（7 人）能否被 C0 边连通 → recall

用法:
  python tools/c0_baseline.py                 # 用最新 v4 边集
  python tools/c0_baseline.py --edges v3      # 用 triage v3 confirmed 49 边
输出:
  output_crosscam/analysis/c0/c0_baseline_{tag}.json / .png
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from det_view import local_det, describe  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
OUT = os.path.join(ANA, "c0")
CAMS = ["305", "316", "318", "320", "323", "343"]
RES = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0

# 318×320 人工确认对（fragment 级 tid 对）；120 出现两次 → 7 人 / 8 对
# 注意：这 8 对是 **重叠对**（两机位同时可见），属 C0-A 支路（同步映射），
#       不是 C0-B（时间门接力）的 GT。
GT_PAIRS = [(1, 1), (14, 3), (37, 11), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28)]

# ---- C0 口径阈值 ----
CFG = dict(
    move_span=0.05,      # 脚点总位移 ≥ 0.05（归一化）视为移动目标
    move_dur=1.0,        # 持续 ≥ 1.0s
    cell=0.06,           # 门发现网格分辨率
    gate_min_pts=3,      # 门簇最少点数
    gate_radius=0.10,    # 实体端点距门心 ≤ 此值 → 判为"通过门"
    border_margin=0.03,  # 贴边也算门（走廊/电梯间出口）
)


# ---- C0 分母修正（v7，2026-10-14）----
# 问题：原 exit_continuity 的分母把所有 gated-exit 都当"该被接续的出门"，
#       但其中一部分是**室内消失点**（人在房间内停留/被遮挡后消失，被一个支撑极弱的
#       内部门收编），根本不存在跨机位对端 → 计为"静默"是冤枉（见 docs/s1-305-egress-findings.md）。
# 规则：gated-exit 实体按 4 类划分，分母只算前 3 类：
#   border            端点贴边，或就近 exit 门**门心**贴边（底边/侧边出口）
#   portal            就近 exit 门支撑点数 ≥ gate_strong_pts（走廊/电梯主门）
#   transient         就弱门，但实体时长 ≤ transit_max（短时穿过，仍算真通行）
#   interior_vanishing 以上都不满足 = **室内弱门 + 长实体** → 单列，不进分母
# 依据：305 两个内部门（(0.45,0.39) n=4 / (0.51,0.81) n=3）成员中位时长 67–70s 且 0 条出边；
#       而 316 的弱门 (0.51,0.33) n=3 成员中位 7s、4/5 有边（真门户）→ 故**仅凭门支撑数不够**，
#       必须叠加"时长"信号。entry 侧对称处理。
V7CFG = dict(gate_strong_pts=6, transit_max=30.0, border_margin=None)


def _near_border(u, v, m):
    return (u <= m or u >= 1 - m or v <= m or v >= 1 - m)


def classify_portals(cam, ents, gates, ex_mask, en_mask):
    """对 gated 端点做 4 分类（border/portal/transient/interior_vanishing）。

    返回 {eid: {'exit_class':..., 'entry_class':..., 'exit_gate_npts':..., 'entry_gate_npts':...}}
    仅含至少一侧 gated 的移动实体。
    """
    m = V7CFG["border_margin"] if V7CFG["border_margin"] is not None else CFG["border_margin"]
    ex_gates = [g for g in gates if g["kind"] == "exit"]
    en_gates = [g for g in gates if g["kind"] == "entry"]

    def nearest(u, v, side):
        best = None
        for g in side:
            c = g["center"]
            dd = float(np.hypot(u - c[0], v - c[1]))
            if best is None or dd < best[0]:
                best = (dd, g)
        return best

    def one(e, kind, side):
        if kind == "exit":
            u, v, gated = e["u1"], e["v1"], ex_mask.get(e["eid"])
        else:
            u, v, gated = e["u0"], e["v0"], en_mask.get(e["eid"])
        if not gated:
            return None, None
        if _near_border(u, v, m):
            return "border", None
        best = nearest(u, v, side)
        if best is None:
            return "border", None      # 贴边已排除却无门可达 → 稳妥归 border
        dd, g = best
        if _near_border(g["center"][0], g["center"][1], m) and dd <= CFG["gate_radius"]:
            return "border", g["n_pts"]
        if g["n_pts"] >= V7CFG["gate_strong_pts"] and dd <= CFG["gate_radius"]:
            return "portal", g["n_pts"]
        if e["dur"] <= V7CFG["transit_max"]:
            return "transient", g["n_pts"]
        return "interior_vanishing", g["n_pts"]

    out = {}
    for eid, e in ents.items():
        if not e["moving"]:
            continue
        ce, ne = one(e, "exit", ex_gates)
        cn, nn = one(e, "entry", en_gates)
        if ce is None and cn is None:
            continue
        out[eid] = dict(exit_class=ce, entry_class=cn,
                        exit_gate_npts=ne, entry_gate_npts=nn)
    return out


# 实体图目录（默认 = 规范目录 analysis/stitch_clean2；可用 --nets / C0_NET_DIR 覆盖）
NET_DIR = os.environ.get("C0_NET_DIR", os.path.join(ANA, "stitch_clean2"))


def load_entities(cam):
    p = os.path.join(NET_DIR, f"net_tracks_{cam}.json")
    return json.load(open(p, encoding="utf-8"))


def load_foot(cam):
    """tid -> (frame[], u[], v[], h[])；DET_VIEW 决定 raw/clean。"""
    W, H = RES[cam]
    acc = defaultdict(list)
    with open(local_det(cam), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for tid, box in zip(d.get("track_ids", []), d.get("boxes", [])):
                if tid is None or int(tid) < 0:
                    continue
                x1, y1, x2, y2 = box
                acc[int(tid)].append((fr, (x1 + x2) / 2 / W, y2 / H, y2 - y1))
    out = {}
    for tid, rows in acc.items():
        rows.sort()
        a = np.array(rows, dtype=float)
        out[tid] = (a[:, 0], a[:, 1], a[:, 2], a[:, 3])
    return out


def build_entity_tracks(cam):
    """eid -> {'tids':[...], 'fr':..., 'u':..., 'v':..., 'h':..., f0,f1}"""
    ent = load_entities(cam)
    foot = load_foot(cam)
    out = {}
    for e in ent["entities"]:
        frs, us, vs, hs = [], [], [], []
        for tid in e["tracks"]:
            if tid not in foot:
                continue
            fr, u, v, h = foot[tid]
            frs.append(fr); us.append(u); vs.append(v); hs.append(h)
        if not frs:
            continue
        fr = np.concatenate(frs); u = np.concatenate(us)
        v = np.concatenate(vs); h = np.concatenate(hs)
        idx = np.argsort(fr)
        fr, u, v, h = fr[idx], u[idx], v[idx], h[idx]
        span = float(np.hypot(u.max() - u.min(), v.max() - v.min()))
        out[e["eid"]] = dict(eid=e["eid"], tids=list(e["tracks"]), fr=fr, u=u, v=v, h=h,
                             f0=int(fr[0]), f1=int(fr[-1]), n_frag=e.get("n_frag", 1),
                             t_first=T0[cam] + float(fr[0]) / FPS,
                             t_last=T0[cam] + float(fr[-1]) / FPS,
                             dur=(float(fr[-1]) - float(fr[0])) / FPS,
                             span=span,
                             moving=bool(span >= CFG["move_span"] and
                                         (float(fr[-1]) - float(fr[0])) / FPS >= CFG["move_dur"]),
                             u0=float(u[0]), v0=float(v[0]),
                             u1=float(u[-1]), v1=float(v[-1]))
    return out


# ---------------- 门发现（网格密度 + 连通域） ----------------
def _label(occ):
    """4/8 邻域连通域标注（等价 scipy.ndimage.label，避免依赖）。"""
    h, w = occ.shape
    lab = np.zeros((h, w), dtype=int)
    cur = 0
    for i in range(h):
        for j in range(w):
            if not occ[i, j] or lab[i, j]:
                continue
            cur += 1
            stack = [(i, j)]
            lab[i, j] = cur
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and occ[ny, nx] and not lab[ny, nx]:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
    return lab, cur


def discover_gates(pts, kind):
    """pts: (n,2) 归一化端点。返回门列表 [{center, n, cells}]。"""
    if len(pts) < CFG["gate_min_pts"]:
        return []
    cell = CFG["cell"]
    gi = np.clip((pts[:, 0] / cell).astype(int), 0, int(1 / cell))
    gj = np.clip((pts[:, 1] / cell).astype(int), 0, int(1 / cell))
    h = w = int(1 / cell) + 1
    occ = np.zeros((h, w), dtype=bool)
    cnt = np.zeros((h, w), dtype=int)
    for a, b in zip(gi, gj):
        cnt[a, b] += 1
    occ = cnt >= CFG["gate_min_pts"]
    lab, n = _label(occ)
    gates = []
    for k in range(1, n + 1):
        ys, xs = np.where(lab == k)
        m = len(ys)
        # 该簇内实际点数
        pts_in = []
        for a, b in zip(gi, gj):
            if lab[a, b] == k:
                pts_in.append((a, b))
        if not pts_in:
            continue
        arr = np.array(pts_in, dtype=float)
        center = arr.mean(axis=0) * cell + cell / 2
        gates.append(dict(kind=kind, center=[round(float(center[0]), 4),
                                             round(float(center[1]), 4)],
                          n_cells=int(m), n_pts=int(len(pts_in))))
    gates.sort(key=lambda g: -g["n_pts"])
    return gates


def classify_endpoints(cam, ents):
    """对移动实体切出 exit/entry 端点，发现门，返回 (gates, exit gated mask ...)."""
    mov = [e for e in ents.values() if e["moving"]]
    if not mov:
        return [], {}, {}
    exits = np.array([[e["u1"], e["v1"]] for e in mov])
    entries = np.array([[e["u0"], e["v0"]] for e in mov])
    g_exit = discover_gates(exits, "exit")
    g_entry = discover_gates(entries, "entry")
    gates = g_exit + g_entry

    def gated(e, p, kind):
        # 贴边 或 距门心 ≤ radius
        u, v = p
        if (u <= CFG["border_margin"] or u >= 1 - CFG["border_margin"] or
                v <= CFG["border_margin"] or v >= 1 - CFG["border_margin"]):
            return True
        for g in (g_exit if kind == "exit" else g_entry):
            gc = np.array(g["center"])
            if np.hypot(u - gc[0], v - gc[1]) <= CFG["gate_radius"]:
                return True
        return False

    ex_mask = {e["eid"]: gated(e, (e["u1"], e["v1"]), "exit") for e in mov}
    en_mask = {e["eid"]: gated(e, (e["u0"], e["v0"]), "entry") for e in mov}
    return gates, ex_mask, en_mask


# ---------------- 边集加载 ----------------
def _load_gated_human(p, note=""):
    """gated_human 边集通用加载（active/audit/gate/confirmed；rejected 剔除）。"""
    d = json.load(open(p, encoding="utf-8"))
    edges = [{"a": e["a"], "b": e["b"], "dir": e["dir"], "dt": e["dt"],
              "sim": e.get("sim"), "channel": e.get("channel")} for e in d["edges"]
             if e.get("status") in ("active", "audit", "gate", "confirmed")]
    ch = Counter(e["channel"] for e in edges)
    n_rej = sum(1 for e in d["edges"] if e.get("status") == "rejected")
    st = Counter(e.get("status") for e in d["edges"] if e.get("status") in
                 ("active", "audit", "gate", "confirmed", "rejected"))
    return edges, (f"{os.path.relpath(p, ANA).replace(os.sep, '/')} ({len(edges)} 边 = "
                   f"{ch.get('main', 0)} main + {ch.get('gate', 0)} gate；"
                   f"status: {dict(st)}；剔除 {n_rej} rejected{note})")


def load_edges(tag):
    if tag == "v9":
        # v9 = v8 口径 + PRT 补口批人工裁决（2026-10-14 晚）：
        #      3 条 active 人工否证 → rejected（累计 10 rejected 剔除）；
        #      5 条 active 人工确认 → confirmed（另有 S6-b 批 10 条 confirmed）
        p = os.path.join(ANA, "relay_v2", "edges_v2_gated_human.json")
        return _load_gated_human(p, "；含 PRT 批 3 否证 + 5 确认")
    if tag == "v8":
        # v8 = v7 口径 + S6-b 人工裁决（历史口径；绑定 .v8snap 快照保证复现：
        #      7 条假链 rejected（全在 audit 池）；2 audit→confirmed；3 active→confirmed）
        p = os.path.join(ANA, "relay_v2", "edges_v2_gated_human.v8snap.json")
        if not os.path.exists(p):
            p = os.path.join(ANA, "relay_v2", "edges_v2_gated_human.json")
        return _load_gated_human(p, "（v8 快照口径）")
    if tag == "v7":
        # v7 = Relay v2（main）+ 门户驱动通道（gate）；见 tools/merge_relay_channels.py
        p = os.path.join(ANA, "relay_v2", "edges_v2_gated.json")
        d = json.load(open(p, encoding="utf-8"))
        edges = [{"a": e["a"], "b": e["b"], "dir": e["dir"], "dt": e["dt"],
                  "sim": e.get("sim"), "channel": e.get("channel")} for e in d["edges"]
                 if e.get("status") in ("active", "audit", "gate")]
        ch = Counter(e["channel"] for e in edges)
        return edges, (f"relay_v2/edges_v2_gated.json ({len(edges)} 边 = "
                       f"{ch.get('main', 0)} main + {ch.get('gate', 0)} gate)")
    if tag == "v6":
        # v6 = Relay v2 试算口径（active + audit，含未复核 secondary；上界性质）
        p = os.path.join(ANA, "relay_v2", "edges_v2.json")
        d = json.load(open(p, encoding="utf-8"))
        edges = [{"a": e["a"], "b": e["b"], "dir": e["dir"], "dt": e["dt"],
                  "sim": e.get("sim")} for e in d["edges"]
                 if e.get("status") in ("active", "audit")]
        n_act = sum(1 for e in d["edges"] if e.get("status") == "active")
        return edges, (f"relay_v2/edges_v2.json ({len(edges)} 边 = {n_act} active + "
                       f"{len(edges)-n_act} audit, 试算口径含未复核)")
    if tag == "v5":
        p = os.path.join(ANA, "relay_clean2", "relay_chains_cradio_v5.json")
        d = json.load(open(p, encoding="utf-8"))
        edges = [{"a": e["a"], "b": e["b"], "dir": e["dir"], "dt": e["dt"],
                  "sim": e.get("sim")} for e in d["edges"]]
        return edges, (f"relay_chains_cradio_v5.json ({len(edges)} 边/"
                       f"{d.get('n_chains', len(d.get('chains', [])))} 链, clean2 口径)")
    if tag == "v4":
        d = json.load(open(os.path.join(ANA, "relay", "relay_chains_cradio_v4.json"),
                           encoding="utf-8"))
        edges = [{"a": e["a"], "b": e["b"], "dir": e["dir"], "dt": e["dt"],
                  "sim": e.get("sim")} for e in d["edges"]]
        return edges, "relay_chains_cradio_v4.json (40 边/30 链)"
    d = json.load(open(os.path.join(ANA, "relay", "relay_triage_cradio_v3.json"),
                       encoding="utf-8"))
    edges = []
    for s in d["confirmed_edges"]:
        a, b = [x.strip() for x in s.split("->")]
        edges.append({"a": a, "b": b, "dir": f"{a.split(':')[0]}->{b.split(':')[0]}",
                      "dt": None, "sim": None})
    return edges, "relay_triage_cradio_v3.json (49 confirmed)"


def eval_overlap_branch():
    """C0-A：318×320 重叠对（同时可见）——映射+外观自动分级 vs 人工标签。

    修订（外部评审缺陷 #4）：
      (a) 按 candidates_review.csv 的 **label 列** 评价候选（CONF/MODERATE/REJ/空），
          不再把"不属于已知 GT 的 AUTO 候选"直接判为 FP；
      (b) 身份键保留数据集 + 相机命名空间：((318, tid_a), (320, tid_b))，
          避免 sorted(tid) 跨命名空间碰撞。
    结果文件 = analysis/candidates_review.csv（tier ∈ AUTO/REVIEW/DROP）。"""
    import csv
    p = os.path.join(ANA, "candidates_review.csv")
    rows = []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    def key(r, cam_a="318", cam_b="320"):
        return ((cam_a, int(r["tid_a"])), (cam_b, int(r["tid_b"])))

    gt = {(("318", a), ("320", b)) for a, b in GT_PAIRS}
    tiers = defaultdict(list)
    for r in rows:
        tiers[r["tier"]].append(r)

    def lab(r):
        return (r.get("label") or "").strip() or "UNLABELED"

    ev = {}
    for t, rs in tiers.items():
        ks = [key(r) for r in rs]
        ev[t] = dict(
            n=len(rs),
            tp_gt=sum(1 for k in ks if k in gt),      # 与人工 GT 交集（旧口径参考）
            conf=sum(1 for r in rs if lab(r) == "CONF"),
            moderate=sum(1 for r in rs if lab(r) == "MODERATE"),
            rejected=sum(1 for r in rs if lab(r) == "REJ"),
            unlabeled=sum(1 for r in rs if lab(r) == "UNLABELED"),
        )
    auto = tiers.get("AUTO", [])
    a_conf = sum(1 for r in auto if lab(r) == "CONF")
    a_mod = sum(1 for r in auto if lab(r) == "MODERATE")
    a_rej = sum(1 for r in auto if lab(r) == "REJ")
    a_gt = sum(1 for r in auto if key(r) in gt)
    # 有明确标签的 AUTO 中，已确认正确 / 已确认错误
    labeled = a_conf + a_rej
    print(f"\n[C0-A 重叠对 318×320] 人工 GT {len(gt)} 对（7 人）")
    print(f"  AUTO 档 {len(auto)} 条 → 标签：CONF {a_conf} / MODERATE {a_mod} / REJ {a_rej}")
    print(f"  · AUTO 召回(与 GT 交集)：{a_gt}/{len(gt)} = {a_gt/len(gt):.3f}")
    print(f"  · AUTO 精度(仅在有标签集上)：CONF/(CONF+REJ) = {a_conf}/{labeled} "
          f"= {a_conf/labeled:.3f}" if labeled else "  · AUTO 精度: n/a")
    if a_mod:
        mods = [f"318#{r['tid_a']}↔320#{r['tid_b']}" for r in auto if lab(r) == "MODERATE"]
        print(f"  · 未决（MODERATE，不计为误报）：{a_mod} 条 {mods}")
    return dict(
        n_gt=len(gt),
        by_tier={k: v for k, v in ev.items()},
        auto=dict(n=len(auto), conf=a_conf, moderate=a_mod, rejected=a_rej,
                  gt_intersect=a_gt,
                  recall=round(a_gt / len(gt), 3),
                  precision_labeled=round(a_conf / labeled, 3) if labeled else None),
        note="precision 仅在已标注(CONF/REJ)集上计算；MODERATE 记为未决，非误报")


def main():
    global NET_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="v5", choices=["v3", "v4", "v5", "v6", "v7", "v8", "v9"])
    ap.add_argument("--nets", default="",
                    help="实体图目录（默认 analysis/stitch_clean2 规范口径；"
                         "复现旧链传 analysis/stitch）")
    ap.add_argument("--tag", default="", help="输出文件名 tag（默认=edges）")
    args = ap.parse_args()
    if args.nets:
        NET_DIR = args.nets if os.path.isabs(args.nets) else os.path.join(ROOT, args.nets)
    elif args.edges in ("v5", "v6", "v7", "v8", "v9"):
        # v5/v6/v7/v8 边集绑定 clean2 口径实体（stitch_clean2），保证 eid 命名空间一致
        NET_DIR = os.path.join(ANA, "stitch_clean2")
    tag = args.tag or args.edges
    # 口径防御警告（不改行为）：v5/v6/v7/v8 数字依赖 DET_VIEW=clean2（脚点数据源）
    if args.edges in ("v5", "v6", "v7", "v8", "v9"):
        _v = os.environ.get("DET_VIEW", "raw") or "raw"
        if _v.strip().lower() != "clean2":
            print(f"[warn] DET_VIEW={_v} ≠ clean2 —— v5/v6 边集绑定 clean2 实体/脚点口径，"
                  f"数字将漂移！请用 DET_VIEW=clean2 重跑。")
    os.makedirs(OUT, exist_ok=True)
    print(describe())
    print(f"[nets] 实体图目录 = {NET_DIR}")

    # ---- 1) 实体盘点 ----
    all_ents = {}
    inventory = {}
    for cam in CAMS:
        ents = build_entity_tracks(cam)
        all_ents[cam] = ents
        durs = np.array([e["dur"] for e in ents.values()])
        mov = [e for e in ents.values() if e["moving"]]
        inventory[cam] = dict(n_entities=len(ents), n_moving=len(mov),
                              n_frag_total=int(sum(e["n_frag"] for e in ents.values())),
                              dur_p50=round(float(np.percentile(durs, 50)), 1),
                              dur_p90=round(float(np.percentile(durs, 90)), 1))
        print(f"[{cam}] 实体 {len(ents)}（移动 {len(mov)}） 碎片 {inventory[cam]['n_frag_total']} "
              f"时长 p50={inventory[cam]['dur_p50']}s p90={inventory[cam]['dur_p90']}s")

    # ---- 2) handoff 图体检 ----
    edges, src = load_edges(args.edges)
    out_deg = Counter(e["a"] for e in edges)
    in_deg = Counter(e["b"] for e in edges)
    dir_mat = Counter(e["dir"] for e in edges)
    dts = [e["dt"] for e in edges if e["dt"] is not None]
    print(f"\n[handoff] 边 {len(edges)}（来源 {src}）")
    print(f"  最大出度={max(out_deg.values()) if out_deg else 0} "
          f"最大入度={max(in_deg.values()) if in_deg else 0} （=1 表示严格 1 对 1）")
    if dts:
        print(f"  dt: p50={np.percentile(dts,50):.1f}s max={max(dts):.1f}s")
    print("  方向矩阵:")
    for d, n in sorted(dir_mat.items(), key=lambda x: -x[1]):
        print(f"    {d}: {n}")

    # ---- 3) 门发现 + 覆盖率 ----
    coverage = {}
    gates_all = {}
    for cam in CAMS:
        gates, ex_mask, en_mask = classify_endpoints(cam, all_ents[cam])
        gates_all[cam] = gates
        mov_eids = list(ex_mask.keys())
        n_ex = len(mov_eids)
        n_ex_gated = sum(ex_mask.values())
        n_en_gated = sum(en_mask.values())
        # 覆盖：该实体是否有出边 / 入边（跨机位）
        has_out = {e["a"] for e in edges}
        has_in = {e["b"] for e in edges}
        ex_cov = sum(1 for eid in mov_eids if ex_mask[eid] and f"{cam}:{eid}" in has_out)
        en_cov = sum(1 for eid in mov_eids if en_mask[eid] and f"{cam}:{eid}" in has_in)
        # ---- v7：分母修正（区分真门户 / 室内消失点）----
        pc = classify_portals(cam, all_ents[cam], gates, ex_mask, en_mask)
        ex_cls = Counter(v["exit_class"] for v in pc.values() if v["exit_class"])
        en_cls = Counter(v["entry_class"] for v in pc.values() if v["entry_class"])
        portal_ex = sum(ex_cls[c] for c in ("border", "portal", "transient"))
        portal_ex_cov = sum(1 for eid, v in pc.items()
                            if v["exit_class"] in ("border", "portal", "transient")
                            and f"{cam}:{eid}" in has_out)
        vanish = [eid for eid, v in pc.items() if v["exit_class"] == "interior_vanishing"]
        vanish_cov = sum(1 for eid in vanish if f"{cam}:{eid}" in has_out)
        portal_en = sum(en_cls[c] for c in ("border", "portal", "transient"))
        portal_en_cov = sum(1 for eid, v in pc.items()
                            if v["entry_class"] in ("border", "portal", "transient")
                            and f"{cam}:{eid}" in has_in)
        coverage[cam] = dict(
            n_moving=n_ex, n_exit_gated=n_ex_gated, n_exit_covered=ex_cov,
            exit_continuity=round(ex_cov / n_ex_gated, 3) if n_ex_gated else None,
            n_entry_gated=n_en_gated, n_entry_covered=en_cov,
            entry_attribution=round(en_cov / n_en_gated, 3) if n_en_gated else None,
            n_gates=len(gates),
            exit_classes=dict(ex_cls), entry_classes=dict(en_cls),
            portal_exit_n=portal_ex, portal_exit_covered=portal_ex_cov,
            exit_continuity_v7=round(portal_ex_cov / portal_ex, 3) if portal_ex else None,
            portal_entry_n=portal_en, portal_entry_covered=portal_en_cov,
            entry_attribution_v7=round(portal_en_cov / portal_en, 3) if portal_en else None,
            n_interior_vanishing=len(vanish), n_interior_vanishing_with_edge=vanish_cov,
            interior_vanishing=sorted(vanish))
        print(f"[{cam}] 移动 {n_ex} | 出门 {n_ex_gated} 被接 {ex_cov} "
              f"(连续性 {coverage[cam]['exit_continuity']}) | "
              f"进门 {n_en_gated} 有源 {en_cov} (归属 {coverage[cam]['entry_attribution']}) | "
              f"门 {len(gates)}")
        print(f"      └ v7 分母修正：门户出门 {portal_ex}（{dict(ex_cls)}）被接 {portal_ex_cov} "
              f"→ 连续性 {coverage[cam]['exit_continuity_v7']}｜"
              f"室内消失点 {len(vanish)}（其中已接 {vanish_cov}）")

    tot_ex = sum(c["n_exit_gated"] for c in coverage.values())
    tot_ex_cov = sum(c["n_exit_covered"] for c in coverage.values())
    tot_en = sum(c["n_entry_gated"] for c in coverage.values())
    tot_en_cov = sum(c["n_entry_covered"] for c in coverage.values())
    tot_px = sum(c["portal_exit_n"] for c in coverage.values())
    tot_px_cov = sum(c["portal_exit_covered"] for c in coverage.values())
    tot_pn = sum(c["portal_entry_n"] for c in coverage.values())
    tot_pn_cov = sum(c["portal_entry_covered"] for c in coverage.values())
    tot_vanish = sum(c["n_interior_vanishing"] for c in coverage.values())
    tot_vanish_cov = sum(c["n_interior_vanishing_with_edge"] for c in coverage.values())
    print(f"\n[全局] 出门 gated {tot_ex} → 被接 {tot_ex_cov} "
          f"(连续性 {tot_ex_cov/max(tot_ex,1):.3f})")
    print(f"[全局] 进门 gated {tot_en} → 有源 {tot_en_cov} "
          f"(归属 {tot_en_cov/max(tot_en,1):.3f})")
    print(f"[全局·v7] 门户出门 {tot_px} → 被接 {tot_px_cov} "
          f"(连续性 {tot_px_cov/max(tot_px,1):.3f})  |  "
          f"室内消失点剔除 {tot_vanish}（其中已接 {tot_vanish_cov}）")
    print(f"[全局·v7] 门户进门 {tot_pn} → 有源 {tot_pn_cov} "
          f"(归属 {tot_pn_cov/max(tot_pn,1):.3f})")

    # ---- 4) 链装配 + 实体覆盖 ----
    succ = defaultdict(list); pred = defaultdict(list)
    for e in edges:
        succ[e["a"]].append(e["b"]); pred[e["b"]].append(e["a"])
    nodes = set(succ) | set(pred)
    heads = sorted(n for n in nodes if n not in pred)
    chains = []
    for h in heads:
        ch = [h]; cur = h
        while succ.get(cur):
            cur = succ[cur][0]
            if cur in ch:
                break
            ch.append(cur)
        chains.append(ch)
    len_dist = Counter(len(c) for c in chains)
    multi = [c for c in chains if len(c) >= 3]
    n_entities_total = sum(v["n_entities"] for v in inventory.values())
    print(f"\n[链] {len(chains)} 条（多跳≥3节点 {len(multi)}） 长度分布 {dict(sorted(len_dist.items()))}")
    print(f"  实体覆盖：{len(nodes)}/{n_entities_total} = {len(nodes)/n_entities_total:.1%} 进入跨机位链")

    # ---- GT 校验：C0 分两支 ----
    #  C0-A 重叠对(318×320)：同步映射+外观 → 结果文件 candidates_review.csv 的 tier 档
    #  C0-B 非重叠对：时间门接力（上面 handoff/coverage 已测）
    c0a = eval_overlap_branch()

    result = dict(
        meta=dict(edges_source=src, n_edges=len(edges), cfg=CFG, cfg_v7=V7CFG,
                  view=describe(), cams=CAMS, nets_dir=NET_DIR,
                  version=("c0_v7_portal_denominator" if args.edges == "v7"
                           else "c0_v2_labeled")),
        inventory=inventory,
        handoff=dict(n_edges=len(edges), max_out_degree=int(max(out_deg.values()) or 0),
                     max_in_degree=int(max(in_deg.values()) or 0),
                     direction_matrix=dict(dir_mat),
                     dt_p50=round(float(np.percentile(dts, 50)), 2) if dts else None,
                     dt_max=round(float(max(dts)), 2) if dts else None),
        gates=gates_all,
        coverage=coverage,
        coverage_global=dict(n_exit_gated=tot_ex, n_exit_covered=tot_ex_cov,
                             exit_continuity=round(tot_ex_cov / max(tot_ex, 1), 3),
                             n_entry_gated=tot_en, n_entry_covered=tot_en_cov,
                             entry_attribution=round(tot_en_cov / max(tot_en, 1), 3),
                             # ---- v7 分母修正 ----
                             portal_exit_n=tot_px, portal_exit_covered=tot_px_cov,
                             exit_continuity_v7=round(tot_px_cov / max(tot_px, 1), 3),
                             portal_entry_n=tot_pn, portal_entry_covered=tot_pn_cov,
                             entry_attribution_v7=round(tot_pn_cov / max(tot_pn, 1), 3),
                             n_interior_vanishing=tot_vanish,
                             n_interior_vanishing_with_edge=tot_vanish_cov,
                             # 敏感性：若把"室内消失点但确有边"的实体回填分母（非主口径）
                             exit_continuity_v7_alt=round(
                                 (tot_px_cov + tot_vanish_cov) / max(tot_px + tot_vanish, 1), 3)),
        chains=dict(n_chains=len(chains), n_multi_hop=len(multi),
                    len_dist={str(k): v for k, v in sorted(len_dist.items())},
                    n_entities_in_chains=len(nodes), n_entities_total=n_entities_total,
                    entity_coverage=round(len(nodes) / n_entities_total, 3),
                    chains=[c for c in chains if len(c) >= 2]),
        gt_318_320=c0a,
    )
    out_p = os.path.join(OUT, f"c0_baseline_{tag}.json")
    json.dump(result, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {out_p}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
            if os.path.exists(fp):
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
                break
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
        # (a) 覆盖率条形
        ax = axes[0]
        cams = CAMS
        exr = [coverage[c]["exit_continuity"] or 0 for c in cams]
        exr7 = [coverage[c]["exit_continuity_v7"] or 0 for c in cams]
        enr = [coverage[c]["entry_attribution"] or 0 for c in cams]
        x = np.arange(len(cams)); wd = 0.27
        ax.bar(x - wd, exr, wd, label="出门连续性(原分母)", color="#c9d6ea")
        ax.bar(x, exr7, wd, label="出门连续性(v7 门户)", color="#1f77b4")
        ax.bar(x + wd, enr, wd, label="进门归属率", color="#ff7f0e")
        ax.set_xticks(x); ax.set_xticklabels(cams)
        ax.set_ylim(0, 1); ax.set_title("C0 覆盖率（无深度基线）")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        # (b) 方向矩阵热力
        ax = axes[1]
        M = np.zeros((6, 6))
        idx = {c: i for i, c in enumerate(cams)}
        for e in edges:
            ca, cb = e["dir"].split("->")
            M[idx[ca], idx[cb]] += 1
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(6)); ax.set_xticklabels(cams)
        ax.set_yticks(range(6)); ax.set_yticklabels(cams)
        ax.set_title(f"C0 接力方向矩阵（{len(edges)} 边）")
        for i in range(6):
            for j in range(6):
                if M[i, j]:
                    ax.text(j, i, int(M[i, j]), ha="center", va="center",
                            color="white" if M[i, j] > M.max() * 0.6 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        # (c) 端点覆盖桑基替代——用堆叠条形
        ax = axes[2]
        labels, gated, covered = [], [], []
        for c in cams:
            labels.append(c); gated.append(coverage[c]["n_exit_gated"])
            covered.append(coverage[c]["n_exit_covered"])
        x = np.arange(len(cams))
        ax.bar(x, gated, color="#d9d9d9", label="出门 gated")
        ax.bar(x, covered, color="#2ca02c", label="被接续")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title("出门目标：被接续 vs 静默")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"C0 无深度跨机位接力基线（{src}）", fontsize=13)
        fig.tight_layout()
        fig_p = os.path.join(OUT, f"c0_baseline_{tag}.png")
        plt.savefig(fig_p, dpi=120)
        print(f"[fig] {fig_p}")
    except Exception as e:
        print(f"[fig] 失败: {e}")


if __name__ == "__main__":
    main()
