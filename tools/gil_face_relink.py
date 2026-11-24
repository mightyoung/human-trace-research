#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人脸管线 × GIL gperson 重链接（只读 GIL）

背景：GIL 的 gperson 层于 2026-10-14 17:43-17:49 被重建（s3v2 重建 + 人工裁决落地，
id 空间从 1..264 → 265..423，158 候选 + 1 倒影桶）。人脸管线（p0b）绑定的是旧空间；
本工具按稳定键 obs_id 重链接到当前 gperson 空间，产出 p0c 全套（shot/best/embed）。

用法：python tools/gil_face_relink.py --src-tag p0b --out-tag p0c
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gil_face_capture import build_best_pools, FACE_DIR, GIL_DB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-tag", default="p0b")
    ap.add_argument("--out-tag", default="p0c")
    args = ap.parse_args()

    with open(os.path.join(FACE_DIR, f"face_shot_{args.src_tag}.json"), encoding="utf-8") as f:
        src = json.load(f)
    with open(os.path.join(FACE_DIR, f"face_best_{args.src_tag}.json"), encoding="utf-8") as f:
        best_old = json.load(f)
    shots = src["shots"]
    cams = src["meta"]["cams"]

    con = sqlite3.connect(GIL_DB)
    obs_map = {}
    dups = 0
    for node_id, gp in con.execute("select node_id, gperson_id from membership where node_kind='obs'"):
        if node_id in obs_map and obs_map[node_id] != gp:
            dups += 1
        obs_map[node_id] = gp
    gp_meta = {r[0]: (r[1], r[2]) for r in con.execute("select id, label, state from gperson")}
    gp_label = {i: v[0] for i, v in gp_meta.items()}
    n_cand = sum(1 for v in gp_meta.values() if v[1] == "candidate")
    con.close()

    changed = unmapped = 0
    old_cover, new_cover = set(), set()
    for s in shots:
        old = s.get("gperson_id")
        oid = s.get("obs_id")
        new = obs_map.get(oid) if oid is not None else None
        if oid is not None and new is None:
            unmapped += 1
        if new != old:
            changed += 1
        s["gperson_id"] = new
        m = gp_meta.get(new) if new is not None else None
        s["gperson_label"] = m[0] if m else None
        s["gperson_state"] = m[1] if m else None
        if s.get("usable") and s.get("tid") is not None:
            if old:
                old_cover.add(old)
            if new and m and m[1] == "candidate":
                new_cover.add(new)

    best_tid, best_ent, best_gp, _ = build_best_pools(shots, cams)
    # gperson 池仅保留 candidate 态（倒影桶等非人态不进证据页）
    best_gp_people = {k: v for k, v in best_gp.items()
                      if gp_meta.get(k, ("", ""))[1] == "candidate"}

    meta = dict(src["meta"])
    meta["relink"] = dict(from_tag=args.src_tag, to_tag=args.out_tag,
                          created=time.strftime("%Y-%m-%d %H:%M:%S"),
                          gperson_space="store_v1 @ s3v2+human_adj(2026-10-14 17:49); 158 候选 + 1 倒影桶",
                          changed_shots=changed, unmapped_shots=unmapped, obs_dup_rows=dups,
                          n_candidates=n_cand)
    out = dict(meta=meta, stats=src["stats"], shots=shots)
    with open(os.path.join(FACE_DIR, f"face_shot_{args.out_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    best = dict(meta=meta, per_tid=best_tid, per_entity=best_ent,
                per_gperson=best_gp_people, gp_label=gp_label,
                crop_map=best_old.get("crop_map", {}))
    with open(os.path.join(FACE_DIR, f"face_best_{args.out_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=1, default=float)

    # ---- embed 重链接（cam|tid 为稳定键）----
    src_emb_p = os.path.join(FACE_DIR, f"face_embed_{args.src_tag}.json")
    if os.path.exists(src_emb_p):
        with open(src_emb_p, encoding="utf-8") as f:
            emb = json.load(f)
        tid2gp = {}
        for s in shots:
            if s.get("tid") is not None:
                tid2gp[f"{s['cam']}|{s['tid']}"] = s.get("gperson_id")
        for k, v in emb.get("per_tid", {}).items():
            gp = tid2gp.get(k)
            v["gperson_id"] = gp
            v["gp_label"] = gp_label.get(gp) if gp else None
        for arr in (emb.get("cross_top", []), emb.get("same_pairs", [])):
            for r in arr:
                ga = tid2gp.get(r["a"])
                gb = tid2gp.get(r["b"])
                r["ga"], r["gb"] = ga, gb
                r["la"], r["lb"] = gp_label.get(ga), gp_label.get(gb)
        # 仅保留两侧均为 candidate 的对（跨人候选需在"人"层面）
        cand_ids = {i for i, v in gp_meta.items() if v[1] == "candidate"}
        emb["cross_top"] = [r for r in emb["cross_top"] if r["ga"] in cand_ids and r["gb"] in cand_ids and r["ga"] != r["gb"]]
        emb["same_pairs"] = [r for r in emb["same_pairs"] if r["ga"] in cand_ids and r["gb"] in cand_ids]
        emb.setdefault("meta", {})["relink"] = meta["relink"]
        with open(os.path.join(FACE_DIR, f"face_embed_{args.out_tag}.json"), "w", encoding="utf-8") as f:
            json.dump(emb, f, ensure_ascii=False, indent=1)

    print(f"relink: shots={len(shots)} changed={changed} unmapped={unmapped} obs_dup={dups}")
    print(f"covered persons: old(264-space)={len(old_cover)} -> new(158-space)={len(new_cover)} "
          f"/ candidates={n_cand}")
    print(f"pools: tids={len(best_tid)} entities={len(best_ent)} gpersons(candidate)={len(best_gp_people)}")
    print(f"-> face_shot_{args.out_tag}.json / face_best_{args.out_tag}.json / face_embed_{args.out_tag}.json")


if __name__ == "__main__":
    main()
