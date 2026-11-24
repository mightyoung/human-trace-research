#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 human-count 的分析产物归一化成 Web 前端可直接消费的数据包。

输入（默认相对项目根）:
  output_crosscam/{cam}_detections.jsonl        逐帧检测（frame/boxes/track_ids/scores/count）
  output_crosscam/{cam}_summary.json            每路汇总
  output_crosscam/analysis/stitch_clean2/net_tracks_{cam}.json   实体图（clean2 口径）
  output_crosscam/analysis/stitch_clean2/manifest.json
  output_crosscam/analysis/relay_clean2/relay_chains_cradio_v5.json        接力链
  output_crosscam/analysis/relay_clean2/relay_mutual_global_cradio_v5.json 接力边
  output_crosscam/analysis/c0/c0_baseline_v5.json               C0 基线
  output_crosscam/analysis/multicam_overview.json               六路总览
  output_crosscam/analysis/candidates_review.csv                318x320 候选对（人工标签）
  output_crosscam/glass_mask/glass_masks.json                   玻璃掩膜
  output_crosscam/analysis/spread/depth_filter/floorref_labels.json  倒影/源 标注

输出: webapp/backend/data/
  cams.json  tracks/{cam}.json  entities.json  chains.json
  metrics.json  masks.json  mirror.json  pairs.json  labels.json
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "backend" / "data"
CAMS = ["305", "316", "318", "320", "323", "343"]

CAM_META = {
    "305": {"name": "办公区 1", "role": "近景为主；人流最强", "w": 2560, "h": 1440, "fps": 25.000, "offset": -5.0,
            "note": "站位/坐姿混杂；时钟 ±2s 推定"},
    "316": {"name": "行政走廊", "role": "拓扑脊柱（通行必经）", "w": 2560, "h": 1440, "fps": 24.999, "offset": 4.0,
            "note": "玻璃墙镜 → 镜像伪影最重"},
    "318": {"name": "办公区 2", "role": "远景机位（1/3 小目标）", "w": 1920, "h": 1080, "fps": 24.991, "offset": 20.0,
            "note": "与 320 真实重叠"},
    "320": {"name": "办公区球机", "role": "鱼眼俯拍，双峰尺度", "w": 2560, "h": 1440, "fps": 25.003, "offset": 19.0,
            "note": "与 318 真实重叠；针孔几何模型退化"},
    "323": {"name": "茶水间", "role": "中景", "w": 1920, "h": 1080, "fps": 25.000, "offset": 18.0,
            "note": "有玻璃反射"},
    "343": {"name": "电梯间", "role": "超近景；进出楼层必经", "w": 2560, "h": 1440, "fps": 25.000, "offset": 19.0,
            "note": "出入口最稳定"},
}
STRIDE = 5
SRC = ROOT / "output_crosscam"
ENT_DIR = SRC / "analysis" / "stitch_clean2"
RELAY_DIR = SRC / "analysis" / "relay_clean2"


def r(x, nd=1):
    return round(float(x), nd)


def load_json(p: Path, default=None):
    if not p.exists():
        print(f"  [warn] missing: {p.relative_to(ROOT)}")
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _chains_from_edges(edge_list):
    """从边集装配链（--relay v2 数据源；与 c0_baseline.py 链装配同构）。

    贪心路径分解：从无前驱节点出发，沿（首条）后继走到尾。v5（1对1）下等价连通路径；
    v2（多对多）为试算性质的近似分解（与 c0 v6 显示口径一致，便于对照）。
    """
    from collections import defaultdict
    succ = defaultdict(list)
    pred = defaultdict(list)
    for e in edge_list:
        succ[e["a"]].append(e["b"])
        pred[e["b"]].append(e["a"])
    nodes = set(succ) | set(pred)
    heads = sorted(n for n in nodes if n not in pred)
    chains = []
    for h in heads:
        ch = [h]
        cur = h
        while succ.get(cur):
            cur = succ[cur][0]
            if cur in ch:
                break
            ch.append(cur)
        chains.append(ch)
    chains.sort(key=lambda c: (-len(c), c[0]))
    return chains


# ---- 疑似同人（换号）候选的判据 ----
# 目的：把「同一个人被拆成两段」和「本来就是两个人」在页面上区分开。
# 这是**解释性标注**，不改实体图、不进任何下游指标，只影响渲染。
SUSPECT_CRIT = {"cos_min": 0.85, "gap_max_s": 4.0, "dist_max_h": 1.5, "min_obs": 5}
SUSPECT_STRONG = {"cos_min": 0.88, "gap_max_s": 2.0, "dist_max_h": 0.8}


def build_suspects(tracks_by_cam: dict, entities: dict, stitch_dir: Path) -> dict:
    """从 stitch 候选里挑「外观很像、时间紧邻、位置接得上，但当前没被合并」的实体对。

    必要性：拼接层只连 mutual-best 且要过 cos/pe 双门；两人重合导致的换号
    （框被合并 → ByteTrack 断开 → 新 track）大多落在这道门外，页面上表现为
    「身份丢了」。本函数把这些"差一点就连上"的相邻段找出来，供前端标注。
    """
    min_off = min(CAM_META[c]["offset"] for c in CAMS)
    pairs: list[dict] = []
    fset_by_cam: dict[str, dict[int, frozenset]] = {}
    for cam in CAMS:
        tr = tracks_by_cam.get(cam) or {}
        ents = (entities.get(cam) or {}).get("entities") or {}
        cp = stitch_dir / f"stitch_{cam}.csv"
        if not tr or not ents or not cp.exists():
            continue
        t2e = {}
        for eid, e in ents.items():
            for t in e["tracks"]:
                t2e[int(t)] = eid
        span = {}
        fset: dict[int, frozenset] = {}
        for k, pts in tr.items():
            if pts:
                span[int(k)] = (pts[0], pts[-1])
                fset[int(k)] = frozenset(p[0] for p in pts)
        fset_by_cam[cam] = fset
        fps = CAM_META[cam]["fps"]
        delay = CAM_META[cam]["offset"] - min_off
        dropped = 0
        with cp.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    a, b = int(row["tid_a"]), int(row["tid_b"])
                    cos = float(row["cos"])
                    pe = float(row["pe_adj"])
                except (KeyError, TypeError, ValueError):
                    continue
                pa, pb = span.get(a), span.get(b)
                ea, eb = t2e.get(a), t2e.get(b)
                if not pa or not pb or not ea or not eb or ea == eb:
                    continue
                a_first, a_last = pa
                b_first, b_last = pb
                # 只收「a 结束后 b 才开始」的方向；帧区间不重叠 ⇒ 必然非同帧共现
                if b_first[0] <= a_last[0]:
                    dropped += 1
                    continue
                gap_s = (b_first[0] - a_last[0]) / fps
                if gap_s > SUSPECT_CRIT["gap_max_s"] or cos < SUSPECT_CRIT["cos_min"]:
                    continue
                # 极短段多半是误检碎片（只框到半个身子/一次遮挡），标出来只会干扰
                if len(fset[a]) < SUSPECT_CRIT["min_obs"] or len(fset[b]) < SUSPECT_CRIT["min_obs"]:
                    continue
                ba, bb = a_last[1:5], b_first[1:5]
                ha = max(1.0, ba[3] - ba[1])
                hb = max(1.0, bb[3] - bb[1])
                dx = (ba[0] + ba[2]) / 2 - (bb[0] + bb[2]) / 2
                dy = ba[3] - bb[3]                      # 脚点纵向差
                dist_h = math.hypot(dx, dy) / max(ha, hb)
                if dist_h > SUSPECT_CRIT["dist_max_h"]:
                    continue
                strong = (cos >= SUSPECT_STRONG["cos_min"]
                          and gap_s <= SUSPECT_STRONG["gap_max_s"]
                          and dist_h <= SUSPECT_STRONG["dist_max_h"])
                pairs.append({
                    "cam": cam, "a": ea, "b": eb, "a_tid": a, "b_tid": b,
                    "a_t0": round(delay + a_first[0] / fps, 1),
                    "t_a_end": round(delay + a_last[0] / fps, 1),
                    "b_t0": round(delay + b_first[0] / fps, 1),
                    "t_b_start": round(delay + b_first[0] / fps, 1),
                    "gap_s": round(gap_s, 2), "cos": round(cos, 3),
                    "pe": round(pe, 3), "dist_h": round(dist_h, 2),
                    "grade": "strong" if strong else "weak",
                })
        if dropped:
            print(f"  [{cam}] 跳过非后接方向候选 {dropped}")

    # 并查集 + 跨组共现校验
    # 单条 link 已保证「a 结束帧 < b 起始帧」，但 A→B→C 传递之后 A 与 C 可能共现，
    # 那就说明 A、C 是两个人（B 只是夹在中间的过渡段）。合并前检查两组的帧集合交集，
    # 相交则拒绝合并——宁可多一组，不可把两个人标成一个人。
    parent: dict = {}
    members: dict = {}   # root -> {tid: frozenset(frames)}
    rejected = 0

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return True
        ax = members.get(rx) or {x[1]: fset_by_cam[x[0]][x[1]]}
        ay = members.get(ry) or {y[1]: fset_by_cam[y[0]][y[1]]}
        for fa in ax.values():
            for fb in ay.values():
                if fa & fb:          # 同帧同时在场 → 必是两个人
                    return False
        parent[rx] = ry
        members.pop(rx, None)
        members[ry] = {**ax, **ay}
        return True

    kept: list[dict] = []
    for p in pairs:
        if union((p["cam"], p["a_tid"]), (p["cam"], p["b_tid"])):
            kept.append(p)
        else:
            rejected += 1
    if rejected:
        print(f"  [suspects] 链式传递拒绝 {rejected} 条（合并后组内会共现 → 实为两人）")

    buckets: dict = {}
    for p in kept:
        buckets.setdefault(find((p["cam"], p["a_tid"])), []).append(p)

    groups = []
    for ps in buckets.values():
        ps.sort(key=lambda x: x["a_t0"])
        eids, tids = [], []
        for p in ps:
            for e, t in ((p["a"], p["a_tid"]), (p["b"], p["b_tid"])):
                if e not in eids:
                    eids.append(e)
                    tids.append(t)
        groups.append({
            "cam": ps[0]["cam"],
            "grade": "strong" if any(p["grade"] == "strong" for p in ps) else "weak",
            "eids": eids, "tids": tids,
            "t0": min(p["a_t0"] for p in ps),
            "t1": max(p["t_b_start"] for p in ps),
            "links": ps,
        })
    groups.sort(key=lambda g: (g["cam"], g["t0"]))
    for i, g in enumerate(groups):
        g["gid"] = f"G{g['cam']}-{i + 1:02d}"

    return {
        "criteria": {
            **SUSPECT_CRIT, "strong": SUSPECT_STRONG,
            "note": "外观相似度(cos)+时间间隙+脚点距离(按框高归一化)三重门；"
                    "组内两两校验同帧共现，共现即判为两人并拒绝合并；"
                    "只标注不合并，实体图与指标均不受影响",
        },
        "n_pairs": len(kept), "n_groups": len(groups),
        "n_rejected_by_overlap": rejected,
        "groups": groups,
    }


def main():
    global OUT
    ap = argparse.ArgumentParser(description="human-count Web 数据包构建")
    ap.add_argument("--relay", choices=["v5", "v6", "v2", "osnet"], default="v5",
                    help="接力边源：v5=relay_clean2（历史口径，默认）；"
                         "v6=relay_clean2 v6（v5 − 人工否证 3 边，见 relay_chains_update.py）；"
                         "v2=relay_v2/edges_v2.json（active+audit，链由连通分量重算）")
    ap.add_argument("--c0", dest="c0_tag", choices=["v5", "v6", "none"], default="v5",
                    help="c0 指标源：v5=现行（default）；v6=Relay v2 试算口径")
    ap.add_argument("--out-dir", default="",
                    help="输出目录（默认 webapp/backend/data；预览/回归建议指向临时目录）")
    ap.add_argument("--root", default="",
                    help="输入数据根（默认 output_crosscam）；换片段时指向新根，如 output_crosscam/long35")
    ap.add_argument("--det-dir", default="", help="检测根目录（默认 <root>；long35 用 <root>/cleanset_v2）")
    ap.add_argument("--det-suffix", default="_detections.jsonl",
                    help="检测文件名后缀（long35 为 _long35_detections.jsonl）")
    ap.add_argument("--window-sec", type=float, default=600,
                    help="会话窗口秒数（10min=600；long35=2100）")
    ap.add_argument("--base-wall", default="08:55:00", help="会话起点墙钟")
    ap.add_argument("--entities", default="", help="实体图目录（默认 <root>/analysis/stitch_clean2）")
    ap.add_argument("--stitch-dir", default="",
                    help="stitch CSV 目录（默认同 --entities；用于生成「疑似同人」候选）")
    ap.add_argument("--no-suspects", action="store_true",
                    help="不生成 suspects.json（疑似同人/换号候选，仅用于页面标注）")
    ap.add_argument("--relay-dir", default="", help="接力链目录（默认 <root>/analysis/relay_clean2）")
    args = ap.parse_args()
    global SRC, ENT_DIR, RELAY_DIR
    SRC = Path(args.root).resolve() if args.root else ROOT / "output_crosscam"
    ENT_DIR = Path(args.entities).resolve() if args.entities else SRC / "analysis" / "stitch_clean2"
    RELAY_DIR = Path(args.relay_dir).resolve() if args.relay_dir else SRC / "analysis" / "relay_clean2"
    DET_DIR = Path(args.det_dir).resolve() if args.det_dir else SRC
    if args.out_dir:
        OUT = Path(args.out_dir).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tracks").mkdir(exist_ok=True)
    print(f"[out] {OUT}")
    print(f"[src] root={SRC} suffix={args.det_suffix} window={args.window_sec}s "
          f"entities={ENT_DIR} relay={RELAY_DIR}")
    print(f"[src] relay={args.relay} c0={args.c0_tag}")

    # ---- 1. 逐路：detections -> tracks 折线 + 每帧概览 ----
    cams_out = []
    track_status = {}   # cam -> {tid: 'keep'|'reflection'|'masked'}
    frames_index = {}   # cam -> {frame: [tid,...]}  用于帧级查询
    tracks_by_cam = {}  # cam -> {tid: [点]}  供实体/同人候选复用（避免二次读盘）
    for cam in CAMS:
        meta = CAM_META[cam]
        det = DET_DIR / f"{cam}{args.det_suffix}"
        summ = load_json(SRC / f"{cam}_summary.json", {}) or {}
        if not det.exists():
            print(f"  [warn] no detections for {cam}")
            continue
        tracks: dict[str, list] = {}
        nframes = 0
        stride = summ.get("stride", STRIDE)
        with det.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fk = int(rec["frame"])
                nframes = max(nframes, fk)
                tids = rec.get("track_ids") or []
                boxes = rec.get("boxes") or []
                scores = rec.get("scores") or []
                fr_list = []
                for i, tid in enumerate(tids):
                    if i >= len(boxes):
                        break
                    b = boxes[i]
                    sc = scores[i] if i < len(scores) else None
                    key = str(int(tid))
                    tracks.setdefault(key, []).append(
                        [fk, r(b[0]), r(b[1]), r(b[2]), r(b[3]), r(sc, 3) if sc is not None else None]
                    )
                    fr_list.append(int(tid))
                frames_index.setdefault(cam, {})[fk] = fr_list
        nframes_total = summ.get("frames_total", nframes)
        tjson = {
            "cam": cam, "w": meta["w"], "h": meta["h"], "fps": meta["fps"],
            "offset": meta["offset"], "stride": stride, "nframes": nframes_total,
            "n_tracks": len(tracks), "tracks": tracks,
        }
        (OUT / "tracks" / f"{cam}.json").write_text(
            json.dumps(tjson, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tracks_by_cam[cam] = tracks
        size_mb = (OUT / "tracks" / f"{cam}.json").stat().st_size / 1e6
        cams_out.append({
            "id": cam, **{k: meta[k] for k in ("name", "role", "w", "h", "fps", "offset", "note")},
            "nframes": nframes_total, "stride": stride, "n_tracks": len(tracks),
            "summary": {k: summ.get(k) for k in (
                "count_mean", "count_max", "count_min", "count_std",
                "deduped_count_mean", "raw_count_mean", "boxes_removed_pct",
                "unique_track_ids") if k in summ},
        })
        print(f"  [{cam}] {len(tracks)} tracks, {nframes_total} frames, {size_mb:.1f}MB")

    # ---- 2. 实体图 ----
    entities = {}
    for cam in CAMS:
        d = load_json(ENT_DIR / f"net_tracks_{cam}.json")
        if not d:
            continue
        ents = {}
        for e in d.get("entities", []):
            ents[e["eid"]] = {
                "tracks": [int(t) for t in e.get("tracks", [])],
                "f0": e.get("f0"), "f1": e.get("f1"), "n_frag": e.get("n_frag", len(e.get("tracks", []))),
                "note": e.get("note", ""),
            }
        entities[cam] = {
            "source": d.get("source", ""), "n_tracks": d.get("n_tracks"),
            "n_entities": d.get("n_entities"), "entities": ents,
        }
    (OUT / "entities.json").write_text(json.dumps(entities, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- 2.5 疑似同人（换号）候选：只标注，不改口径 ----
    if args.no_suspects:
        print("[suspects] 已跳过（--no-suspects）")
    else:
        sus_dir = Path(args.stitch_dir).resolve() if args.stitch_dir else ENT_DIR
        sus = build_suspects(tracks_by_cam, entities, sus_dir)
        (OUT / "suspects.json").write_text(
            json.dumps(sus, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"[suspects] {sus['n_pairs']} 段 / {sus['n_groups']} 组 "
              f"（strong {sum(1 for g in sus['groups'] if g['grade'] == 'strong')} 组）"
              f" -> {(OUT / 'suspects.json').stat().st_size / 1024:.1f} KB")

    # ---- 3. 接力链 + 边（数据源：--relay v5|v2） ----
    if args.relay == "v2":
        v2d = load_json(ROOT / "output_crosscam/analysis/relay_v2/edges_v2.json", {}) or {}
        mutual_edges = [e for e in v2d.get("edges", [])
                        if e.get("status") in ("active", "audit")]
        chains_raw = None
        thr = None
        print(f"  [relay] v2: {len(mutual_edges)} 条 accepted 边（active+audit），链由连通分量重算")
    else:
        if args.relay == "osnet":
            chains_raw = load_json(RELAY_DIR / "relay_chains_osnet.json", {}) or {}
            mutual = load_json(RELAY_DIR / "relay_mutual_global_osnet.json", {}) or {}
            mutual_edges = chains_raw.get("edges", []) or mutual.get("mutual", [])
            thr = mutual.get("thr")
            print(f"  [relay] osnet: {len(chains_raw.get('chains', []))} 链 / {len(mutual_edges)} 边")
        elif args.relay == "v6":
            chains_raw = load_json(RELAY_DIR / "relay_chains_cradio_v6.json", {}) or {}
            mutual = load_json(RELAY_DIR / "relay_mutual_global_cradio_v5.json", {}) or {}
            mutual_edges = chains_raw.get("edges", []) or mutual.get("mutual", [])
            thr = mutual.get("thr")
            n_excl = len(chains_raw.get("excluded_edges") or [])
            print(f"  [relay] v6: {len(chains_raw.get('chains', []))} 链 / {len(mutual_edges)} 边"
                  f"（v5 − 人工否证 {n_excl} 边）")
        else:
            chains_raw = load_json(RELAY_DIR / "relay_chains_cradio_v5.json", {}) or {}
            mutual = load_json(RELAY_DIR / "relay_mutual_global_cradio_v5.json", {}) or {}
            mutual_edges = mutual.get("mutual", [])
            thr = mutual.get("thr")

    def ent_time(cam, eid):
        e = entities.get(cam, {}).get("entities", {}).get(eid)
        if not e:
            return None, None
        fps = CAM_META[cam]["fps"]
        off = CAM_META[cam]["offset"]
        return (off + (e["f0"] or 0) / fps), (off + (e["f1"] or 0) / fps)

    min_off = min(CAM_META[c]["offset"] for c in CAMS)
    chain_lists = (_chains_from_edges(mutual_edges) if args.relay == "v2"
                   else chains_raw.get("chains", []))
    chains = []
    for idx, ch in enumerate(chain_lists):
        nodes = []
        for ref in ch:
            cam, eid = ref.split(":", 1)
            t0, t1 = ent_time(cam, eid)
            nodes.append({"cam": cam, "eid": eid, "t0": t0, "t1": t1,
                          "t0s": (t0 - min_off) if t0 is not None else None,
                          "t1s": (t1 - min_off) if t1 is not None else None})
        chains.append({"id": f"C{idx+1:02d}", "nodes": nodes, "len": len(nodes)})
    edges = []
    for m in mutual_edges:
        rec = {"a": m.get("a"), "b": m.get("b"), "dir": m.get("dir"), "dt": m.get("dt"),
               "sim": m.get("sim"), "a_h": m.get("a_h"), "b_h": m.get("b_h"),
               "a_end": m.get("a_end"), "b_start": m.get("b_start")}
        if args.relay == "v2":
            rec["edge_type"] = m.get("edge_type")
            rec["status"] = m.get("status")
        edges.append(rec)
    (OUT / "chains.json").write_text(json.dumps(
        {"n_chains": len(chains), "thr": thr, "chains": chains, "edges": edges},
        ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- 4. 指标 ----
    c0 = ({} if args.c0_tag == "none"
          else load_json(SRC / "analysis" / "c0" / f"c0_baseline_{args.c0_tag}.json", {}) or {})
    overview = load_json(ROOT / "output_crosscam/analysis/multicam_overview.json", {}) or {}
    manifest = load_json(ENT_DIR / "manifest.json", {}) or {}
    metrics = {"c0": c0, "c0_tag": args.c0_tag, "relay_tag": args.relay,
               "overview": overview, "manifest": manifest}
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- 5. 掩膜 ----
    masks = load_json(SRC / "glass_mask" / "glass_masks.json", {}) or {}
    (OUT / "masks.json").write_text(json.dumps(masks, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- 6. 倒影/源 标注 ----
    refl = load_json(SRC / "analysis" / "spread" / "depth_filter" / "floorref_labels.json", {}) or {}
    (OUT / "mirror.json").write_text(json.dumps(refl, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- 7. 候选对（标注用） ----
    pairs = []
    csvp = ROOT / "output_crosscam/analysis/candidates_review.csv"
    if csvp.exists():
        with csvp.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                pairs.append({
                    "cam_a": "318", "cam_b": "320",
                    "tid_a": int(row["tid_a"]), "tid_b": int(row["tid_b"]),
                    "tier": row.get("tier", ""), "note": row.get("note", ""),
                    "sync_r": _f(row.get("sync_r")), "reid_cos": _f(row.get("reid_cos")),
                    "sp_median": _f(row.get("sp_median")), "sp_p90": _f(row.get("sp_p90")),
                    "sp_n": _f(row.get("sp_n")), "covis_sec": _f(row.get("covis_sec")),
                    "label": row.get("label", "") or "UNLABELED",
                })
    (OUT / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- 8. track_status（keep / reflection / masked） ----
    refl_set = {(str(c), int(t)) for c, t in refl.get("reflection", [])}
    for cam in CAMS:
        st = {}
        tjson = load_json(OUT / "tracks" / f"{cam}.json", {}) or {}
        regs = (masks.get("cameras", {}).get(cam, {}) or {})
        enabled = regs.get("enabled", False)
        poly = regs.get("regions", [])
        W, H = CAM_META[cam]["w"], CAM_META[cam]["h"]
        for tid, pts in tjson.get("tracks", {}).items():
            if (cam, int(tid)) in refl_set:
                st[tid] = "reflection"
                continue
            if enabled and poly:
                inside = 0
                for p in pts:
                    u = (p[1] + p[3]) / 2.0 / W
                    v = p[4] / H
                    if _in_polys(u, v, poly):
                        inside += 1
                if pts and inside / len(pts) >= 0.5:
                    st[tid] = "masked"
                    continue
            st[tid] = "keep"
        track_status[cam] = st

    # ---- 9. cams.json ----
    (OUT / "cams.json").write_text(json.dumps({
        "session": {"base_wall": args.base_wall, "min_offset": min_off, "window_sec": args.window_sec},
        "stride": STRIDE,
        "cameras": cams_out,
        "track_status": track_status,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # ---- 10. labels.json（可变标注库，首次从项目已有标签播种） ----
    labels_path = OUT / "labels.json"
    if not labels_path.exists():
        seed = {
            "version": 1,
            "pair_decisions": {},          # "318:14|320:3" -> "CONF"|"REJ"|"MODERATE"
            "mirror_decisions": {},        # "316:10" -> "reflection"|"source"|"person"
            "glass_masks": {},             # cam -> [[u,v],...]
            "notes": {},
        }
        for p in pairs:
            if p["label"] in ("CONF", "REJ", "MODERATE"):
                seed["pair_decisions"][f'{p["cam_a"]}:{p["tid_a"]}|{p["cam_b"]}:{p["tid_b"]}'] = p["label"]
        for c, t in refl.get("reflection", []):
            seed["mirror_decisions"][f"{c}:{t}"] = "reflection"
        for c, t in refl.get("source", []):
            seed["mirror_decisions"].setdefault(f"{c}:{t}", "source")
        labels_path.write_text(json.dumps(seed, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 11. GIL 存储概况（存在则并入数据包，供前端展示数据代际） ----
    gil_manifest = load_json(ROOT / "output_crosscam/analysis/gil/store_v1/manifest.json", {}) or {}
    gil_out = {"available": bool(gil_manifest)}
    if gil_manifest:
        import sqlite3
        db = ROOT / "output_crosscam/analysis/gil/store_v1/gil.sqlite"
        edges_by, tables = {}, {}
        if db.exists():
            con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
            try:
                edges_by = {f"{k}:{s}": n for k, s, n in
                            con.execute("select kind,status,count(*) from edge group by kind,status")}
                tables = {t: con.execute(f"select count(*) from {t}").fetchone()[0]
                          for t in ("obs", "frag", "entity", "gperson", "membership",
                                    "visit", "owner", "annotation", "mutex")}
            finally:
                con.close()
        skipped = load_json(ROOT / "output_crosscam/analysis/gil/store_v1/skipped_edges.json", []) or []
        gil_out.update({
            "created": gil_manifest.get("created"),
            "stats": gil_manifest.get("stats", {}),
            "tables": tables,
            "edges_by": edges_by,
            "meta": gil_manifest.get("meta", {}),
            "files": {k: (v or {}).get("sha1_16") for k, v in (gil_manifest.get("files") or {}).items()},
            "skipped_edges": len(skipped),
            "note": "GIL 统一存储 v1（obs/frag/entity + relay 边 + 统一标注）",
        })
        print(f"  [gil] obs={tables.get('obs')} entity={tables.get('entity')} "
              f"edges={edges_by} annotation={tables.get('annotation')}")
    (OUT / "gil.json").write_text(json.dumps(gil_out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"[done] entities={sum(len(v['entities']) for v in entities.values())} "
          f"chains={len(chains)} edges={len(edges)} pairs={len(pairs)}")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _in_polys(u, v, polys):
    for poly in polys:
        if _in_poly(u, v, poly):
            return True
    return False


def _in_poly(u, v, poly):
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > v) != (yj > v)) and (u < (xj - xi) * (v - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


if __name__ == "__main__":
    main()
