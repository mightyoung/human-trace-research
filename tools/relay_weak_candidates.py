#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6：弱事件候选补充通道（Relay v2 增量）

根因（55 判增量对账发现，见 docs/relay-v2-s6-report.md）：
  - E010↔E014（判同人，cradio sim=0.9584，**可恢复**）因 316 侧 E014 事件 moving=False
    （span 0.045 < 0.05）被 `crosscam_relay links --only-moving` 过滤——2618 候选池从未包含该对；
  - E009↔E024（判同人）同理被过滤，但补池后 sim=0.8560 < 0.943 → 外观证据不足（独立限制，
    非候选生成问题）。

本脚本（全部增量，不动 v5 产物、不改 v5 池）：
  1. 事件表（6 路，events_{cam}.json）→ 生成"≥1 侧 moving"的补充候选
     （排除 v5 池已有对；dt 窗 [-1, 60]s 与 v5 一致；两侧均 non-moving 的噪声对不生成）
  2. cradio v5 嵌入打分（cradio_scale/{cam}_emb_scale.npz；口径由 3 个已知对复现锚点自证）
  3. w_dt / score 用**冻结**的 dist_v1.json 分布插值（不重拟合，fit provenance 不变）
  4. 输出 relay_v2/weak_scored.json（schema 对齐 scored_v2.json + weak_side 字段）

用法:
  python tools/relay_weak_candidates.py            # 生成 + 打分 + 存盘
输出:
  output_crosscam/analysis/relay_v2/weak_scored.json
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
# 口径纪律：与 relay_clean2 管线一致（脚点/实体 clean2；实体目录 stitch_clean2）
os.environ.setdefault("DET_VIEW", "clean2")
from crosscam_relay import load_foot_trajs  # noqa: E402
from relay_rescore import key_to_tid  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
RLC = os.path.join(ANA, "relay_clean2")
RV2 = os.path.join(ANA, "relay_v2")
EMB_DIR = os.path.join(ANA, "cradio_scale")
EMB_PAT = "{cam}_emb_scale.npz"
CAMS = ["316", "305", "318", "320", "323", "343"]
DT_MIN, DT_MAX = -1.0, 60.0
SIM_THR = 0.943
LABELED = os.path.join(ANA, "truth", "pair_candidates_labeled.csv")
LABELS_JSON = os.path.join(ANA, "truth", "remote", "labels.json")
OUT = os.path.join(RV2, "weak_scored.json")

# 口径自证锚点（(a,b) -> 期望 sim；来自 relay_scored_cradio_v5.json 记录值）
ANCHORS = [
    (("343:343-E010", "316:316-E015"), 0.9741),
    (("343:343-E009", "316:316-E025"), 0.9446),
    (("343:343-E011", "316:316-E013"), 0.9545),
]


def parse_pid(pid):
    left, right = pid.split("__")
    la, lb = left.split("_"), right.split("_")
    return f"{la[0]}:{la[2]}", f"{lb[0]}:{lb[2]}"


def load_judged():
    """合并判定：labeled.csv ∪ labels.json（labels.json 优先）。{pair_id: '1'|'0'|'?'}"""
    judged = {}
    if os.path.exists(LABELED):
        for r in csv.DictReader(open(LABELED, encoding="utf-8-sig")):
            v = (r.get("human_same_person") or "").strip()
            if v in ("1", "0", "?"):
                judged[r["pair_id"]] = v
    if os.path.exists(LABELS_JSON):
        d = json.load(open(LABELS_JSON, encoding="utf-8"))
        for pid, rec in d.get("pair", {}).items():
            judged[pid] = {1: "1", 0: "0", 2: "?"}.get(rec.get("v"), "?")
    return judged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # ---- 1. 事件表 + v5 池 ----
    events, ev_index = {}, {}
    for cam in CAMS:
        evs = json.load(open(os.path.join(RLC, f"events_{cam}.json"), encoding="utf-8"))["events"]
        events[cam] = evs
        for e in evs:
            ev_index[f"{cam}:{e['eid']}"] = e
    pool = {(c["a"], c["b"]) for c in
            json.load(open(os.path.join(RLC, "relay_candidates.json"), encoding="utf-8"))["candidates"]}
    print(f"[load] 事件 { {c: len(v) for c, v in events.items()} } | v5 池 {len(pool)} 对")

    # ---- 2. 实体首/末碎片（与 relay_rescore.do_score 同构） ----
    spans, ent_frag = {}, {}
    for cam in CAMS:
        foot = load_foot_trajs(cam)
        spans[cam] = {tid: (float(fr[0]), float(fr[-1]))
                      for tid, (fr, u, v, h) in foot.items()}
    for cam in CAMS:
        d = json.load(open(os.path.join(ANA, "stitch_clean2", f"net_tracks_{cam}.json"),
                           encoding="utf-8"))
        for e in d["entities"]:
            tids = [t for t in e["tracks"] if t in spans[cam]]
            if not tids:
                continue
            tids.sort(key=lambda t: spans[cam][t][0])
            ent_frag[f"{cam}:{e['eid']}"] = (tids[0], tids[-1])

    # ---- 3. cradio 嵌入 ----
    embs = {}
    for cam in CAMS:
        p = os.path.join(EMB_DIR, EMB_PAT.format(cam=cam))
        z = np.load(p)
        embs[cam] = {key_to_tid(k): z[k] for k in z.files}
    print(f"[emb] cradio_scale 载入: { {c: len(v) for c, v in embs.items()} }")

    def sim_of(a, b):
        fa = ent_frag.get(a, (None, None))[1]
        fb = ent_frag.get(b, (None, None))[0]
        ca, cb = a.split(":")[0], b.split(":")[0]
        va = embs[ca].get(fa) if fa is not None else None
        vb = embs[cb].get(fb) if fb is not None else None
        if va is None or vb is None:
            return None, fa, fb
        s = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        return s, fa, fb

    # ---- 4. 口径自证（3 锚点必须逐位复现） ----
    fails = []
    for (a, b), want in ANCHORS:
        s, _, _ = sim_of(a, b)
        ok = s is not None and abs(s - want) < 2e-4
        print(f"[anchor] {a} -> {b}: sim={None if s is None else round(s, 4)} want={want} "
              f"{'✓' if ok else '✗'}")
        if not ok:
            fails.append((a, b))
    if fails:
        print(f"[FATAL] 口径锚点复现失败: {fails}")
        sys.exit(1)

    # ---- 5. 弱候选生成（≥1 侧 moving；排除 v5 池；dt 窗同 v5） ----
    seen = {}
    for i, ca in enumerate(CAMS):
        for cb in CAMS[i + 1:]:
            for ra in events[ca]:
                for rb in events[cb]:
                    if not (ra["moving"] or rb["moving"]):
                        continue
                    dt1 = rb["t_first"] - ra["t_last"]
                    dt2 = ra["t_first"] - rb["t_last"]
                    if DT_MIN <= dt1 <= DT_MAX:
                        k = (f"{ca}:{ra['eid']}", f"{cb}:{rb['eid']}")
                        if k not in pool:
                            seen.setdefault(k, (round(dt1, 1), "a" if not ra["moving"] else "b"))
                    if DT_MIN <= dt2 <= DT_MAX:
                        k = (f"{cb}:{rb['eid']}", f"{ca}:{ra['eid']}")
                        if k not in pool:
                            seen.setdefault(k, (round(dt2, 1), "a" if not rb["moving"] else "b"))
    print(f"[gen] 弱候选 {len(seen)} 对（≥1 侧 moving，排除 v5 池）")

    # ---- 6. 打分 ----
    scored, n_missing = [], 0
    for (a, b), (dt, ws) in seen.items():
        s, fa, fb = sim_of(a, b)
        if s is None:
            n_missing += 1
            continue
        ca, cb = a.split(":")[0], b.split(":")[0]
        ra, rb = ev_index[a], ev_index[b]
        scored.append({
            "a": a, "b": b, "dir": f"{ca}->{cb}", "dt": dt,
            "a_h": ra["h_med"], "b_h": rb["h_med"],
            "a_end": [ra["u1"], ra["v1"]], "b_start": [rb["u0"], rb["v0"]],
            "a_tlast": ra["t_last"], "b_tfirst": rb["t_first"],
            "a_frag_last": fa, "b_frag_first": fb,
            "sim": round(s, 4), "weak_side": ws,
        })
    scored.sort(key=lambda r: -r["sim"])

    # ---- 7. w_dt / score（冻结 dist_v1 分布插值） ----
    dist = json.load(open(os.path.join(RV2, "dist_v1.json"), encoding="utf-8"))
    GRID = np.asarray(dist["grid_x"], dtype=float)
    FGRID = np.asarray(dist["grid_f"], dtype=float)
    FMAX = float(FGRID.max())
    lam = float(dist.get("lambda_default", 0.01))

    def wdt_of(dt):
        x = float(np.clip(dt, GRID[0], GRID[-1]))
        i = int(np.searchsorted(GRID, x) - 1)
        i = max(0, min(i, len(GRID) - 2))
        t = (x - GRID[i]) / (GRID[i + 1] - GRID[i])
        f = FGRID[i] * (1 - t) + FGRID[i + 1] * t
        return float(max(f, 1e-9) / FMAX)

    for r in scored:
        w = wdt_of(r["dt"])
        r["w_dt"] = round(w, 5)
        r["score"] = round(r["sim"] + lam * float(np.log(w + 1e-9)), 5)

    # ---- 8. 报告 ----
    sims = np.array([r["sim"] for r in scored])
    n_ge = int((sims >= SIM_THR).sum())
    print(f"[score] {len(scored)} 对已打分（缺嵌入 {n_missing}）| "
          f"sim p50={np.percentile(sims, 50):.3f} p90={np.percentile(sims, 90):.3f} "
          f"p99={np.percentile(sims, 99):.3f} max={sims.max():.3f}")
    print(f"[score] sim >= {SIM_THR}: {n_ge} 对")
    for r in scored[:20]:
        flag = " ***≥阈值***" if r["sim"] >= SIM_THR else ""
        print(f"  {r['sim']:.4f}  {r['a']} -{r['dt']:>5.1f}s-> {r['b']}  "
              f"(weak_side={r['weak_side']}){flag}")
    if n_ge > 20:
        print("  ......（≥阈值完整清单见 weak_scored.json）")

    # 目标对专项
    print("\n[目标对]")
    for a, b, want in (("343:343-E010", "316:316-E014", "期望 0.9584（可恢复）"),
                       ("343:343-E009", "316:316-E024", "期望 0.8560（低于阈值）")):
        hits = [r for r in scored if r["a"] == a and r["b"] == b]
        if hits:
            print(f"  {a} -> {b}: sim={hits[0]['sim']} dt={hits[0]['dt']} "
                  f"score={hits[0]['score']} | {want}")
        else:
            print(f"  {a} -> {b}: **未生成** | {want}")

    # 已判异人碰撞检查（弱池 ≥阈值 中不得出现已判 v=0）
    judged = load_judged()
    neg_keys = {parse_pid(p) for p, v in judged.items() if v == "0"}
    pos_keys = {parse_pid(p) for p, v in judged.items() if v == "1"}
    neg_hits = [(r["a"], r["b"]) for r in scored
                if r["sim"] >= SIM_THR and (r["a"], r["b"]) in neg_keys]
    pos_hits = [(r["a"], r["b"]) for r in scored
                if r["sim"] >= SIM_THR and (r["a"], r["b"]) in pos_keys]
    print(f"\n[对账预检] 弱池 ≥阈值 中：已判同人命中 {len(pos_hits)} 条 {pos_hits} | "
          f"已判异人命中 {len(neg_hits)} 条 {neg_hits}")

    out = {
        "version": "relay_v2/weak/1",
        "based_on": {
            "events_dir": "output_crosscam/analysis/relay_clean2/events_*.json",
            "emb_dir": "output_crosscam/analysis/cradio_scale",
            "emb_pat": EMB_PAT,
            "frozen_dist": "output_crosscam/analysis/relay_v2/dist_v1.json",
            "anchors_reproduced": [{"pair": list(k), "sim": v} for k, v in ANCHORS],
        },
        "params": {"dt_window": [DT_MIN, DT_MAX], "sim_thr": SIM_THR,
                   "lam": lam, "rule": "at_least_one_side_moving; exclude_v5_pool"},
        "n_generated": len(seen), "n_scored": len(scored), "n_missing": n_missing,
        "n_ge_thr": n_ge,
        "candidates": scored,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")
    print(f"[dist] weak_side 分布: {dict(Counter(r['weak_side'] for r in scored))}")


if __name__ == "__main__":
    main()
