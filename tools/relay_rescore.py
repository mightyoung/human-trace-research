#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接力候选重打分（可切换嵌入模型）→ global mutual-best → 链构造 → 与旧结果对比。

用法（d3b 复现 / cradio 新版）:
  python tools/relay_rescore.py score --emb-dir output_crosscam/analysis/relay/remote_emb \
      --emb-pat "reid_emb_{cam}_d3b.npz" --tag d3b --sim-thr 0.65
  python tools/relay_rescore.py all --emb-dir output_crosscam/analysis/cradio_scale \
      --emb-pat "{cam}_emb_scale.npz" --tag cradio --sim-thr 0.90
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosscam_relay import ROOT, OUT_DIR, NET_DIR, load_foot_trajs  # noqa: E402

CAMS = ["316", "305", "318", "320", "323", "343"]
PREFIX = {"318": "a", "320": "b", "305": "v305_", "316": "v316_",
          "323": "v323_", "343": "v343_"}


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def do_score(args):
    spans, ent_frag = {}, {}
    for cam in CAMS:
        foot = load_foot_trajs(cam)
        spans[cam] = {tid: (float(fr[0]), float(fr[-1]))
                      for tid, (fr, u, v, h) in foot.items()}
    for cam in CAMS:
        with open(os.path.join(NET_DIR, f"net_tracks_{cam}.json"),
                  encoding="utf-8") as f:
            d = json.load(f)
        for e in d["entities"]:
            tids = [t for t in e["tracks"] if t in spans[cam]]
            if not tids:
                continue
            tids.sort(key=lambda t: spans[cam][t][0])
            ent_frag[f"{cam}:{e['eid']}"] = (tids[0], tids[-1])

    embs = {}
    for cam in CAMS:
        p = os.path.join(args.emb_dir, args.emb_pat.format(cam=cam))
        d = np.load(p)
        embs[cam] = {key_to_tid(k): d[k] for k in d.files}
    print(f"[score-{args.tag}] 嵌入载入: "
          f"{ {c: len(v) for c, v in embs.items()} }")

    cand_p = os.path.join(OUT_DIR, "relay_candidates.json")
    with open(cand_p, encoding="utf-8") as f:
        cand = json.load(f)["candidates"]

    n_missing = 0
    scored = []
    for c in cand:
        ca, ea = c["a"].split(":")
        cb, eb = c["b"].split(":")
        if c["a"] not in ent_frag or c["b"] not in ent_frag:
            n_missing += 1
            continue
        fa = ent_frag[c["a"]][1]
        fb = ent_frag[c["b"]][0]
        va = embs[ca].get(fa)
        vb = embs[cb].get(fb)
        if va is None or vb is None:
            n_missing += 1
            continue
        sim = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        rec = dict(c)
        rec["sim"] = round(sim, 4)
        rec["a_frag_last"] = fa
        rec["b_frag_first"] = fb
        scored.append(rec)
    scored.sort(key=lambda r: -r["sim"])
    out_p = os.path.join(OUT_DIR, f"relay_scored_{args.tag}.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"sim_thr": args.sim_thr, "n_scored": len(scored),
                   "n_missing": n_missing, "candidates": scored},
                  f, ensure_ascii=False, indent=1)
    sims = np.array([r["sim"] for r in scored])
    print(f"[score-{args.tag}] {len(scored)} 候选已打分（缺嵌入 {n_missing}）")
    print(f"  sim: p50={np.percentile(sims,50):.3f} p90={np.percentile(sims,90):.3f} "
          f"p99={np.percentile(sims,99):.3f} max={sims.max():.3f}")
    n_ge = int((sims >= args.sim_thr).sum())
    print(f"  sim >= {args.sim_thr}: {n_ge} 对")
    print(f"[score-{args.tag}] -> {out_p}")
    return out_p


def do_mutual(args):
    scored_p = os.path.join(OUT_DIR, f"relay_scored_{args.tag}.json")
    with open(scored_p, encoding="utf-8") as f:
        rows = [c for c in json.load(f)["candidates"] if c["sim"] >= args.sim_thr]
    bfa, bfb = {}, {}
    for r in rows:
        if r["a"] not in bfa or r["sim"] > bfa[r["a"]]["sim"]:
            bfa[r["a"]] = r
        if r["b"] not in bfb or r["sim"] > bfb[r["b"]]["sim"]:
            bfb[r["b"]] = r
    mut = [r for r in rows if bfa.get(r["a"]) is r and bfb.get(r["b"]) is r]
    out_p = os.path.join(OUT_DIR, f"relay_mutual_global_{args.tag}.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"thr": args.sim_thr, "n": len(mut), "mutual": mut},
                  f, ensure_ascii=False, indent=1)
    print(f"[mutual-{args.tag}] sim>=thr {len(rows)} → global mutual {len(mut)}")

    # 链构造
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
    ch_p = os.path.join(OUT_DIR, f"relay_chains_{args.tag}.json")
    with open(ch_p, "w", encoding="utf-8") as f:
        json.dump({"n_chains": len(chains), "chains": chains, "edges": mut},
                  f, ensure_ascii=False, indent=1)
    n_multi = len([c for c in chains if len(c) >= 3])
    print(f"[chains-{args.tag}] {len(chains)} 链（多跳 {n_multi}）-> {ch_p}")

    # 与 d3b 旧结果对比
    old_p = os.path.join(OUT_DIR, "relay_mutual_global.json")
    if args.tag != "d3b" and os.path.exists(old_p):
        with open(old_p, encoding="utf-8") as f:
            old = json.load(f)["mutual"]
        old_set = {(r["a"], r["b"]) for r in old}
        new_set = {(r["a"], r["b"]) for r in mut}
        both = old_set & new_set
        print()
        print(f"[对比] d3b {len(old_set)} 边 vs {args.tag} {len(new_set)} 边 | "
              f"共同 {len(both)} | 新增 {len(new_set-old_set)} | 丢失 {len(old_set-new_set)}")
        lost = sorted(old_set - new_set)
        new = sorted(new_set - old_set)
        if lost:
            print(f"  丢失（d3b 有、{args.tag} 无）:")
            for a, b in lost[:15]:
                print(f"    {a} -> {b}")
        if new:
            print(f"  新增（{args.tag} 独有）top10 by sim:")
            sim_of = {(r["a"], r["b"]): r["sim"] for r in mut}
            for a, b in sorted(new, key=lambda x: -sim_of.get(x, 0))[:10]:
                print(f"    {a} -> {b}  (sim {sim_of.get((a,b))})")
    return mut, chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "mutual", "all"])
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--emb-pat", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sim-thr", type=float, default=0.65)
    args = ap.parse_args()
    if args.cmd in ("score", "all"):
        do_score(args)
    if args.cmd in ("mutual", "all"):
        do_mutual(args)


if __name__ == "__main__":
    main()
