#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人脸页数据包导出（纯前端，零后端改动；随 dist 静态托管）

读取 analysis/face/ 各产物 → 生成 webapp/frontend/public/faces/face_data.json + 图片资产。

用法：python tools/gil_face_web.py --tag p0b
"""
import argparse
import json
import os
import shutil
import sqlite3
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE = os.path.join(ROOT, "output_crosscam", "analysis", "face")
PUB = os.path.join(ROOT, "webapp", "frontend", "public", "faces")
GIL_DB = os.path.join(ROOT, "output_crosscam", "analysis", "gil", "store_v1", "gil.sqlite")


def jload(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0b")
    ap.add_argument("--fusion-tag", default=None, help="fusion/enh/embed 产物标签，默认同 --tag")
    ap.add_argument("--enh-tag", default=None)
    ap.add_argument("--embed-tag", default=None)
    ap.add_argument("--live-tag", default="v0", help="L0 liveness 执行产物标签（face_live_flags_{v}.json）")
    args = ap.parse_args()
    tag = args.tag
    ftag = args.fusion_tag or tag
    etag = args.enh_tag or tag
    mtag = args.embed_tag or tag

    os.makedirs(PUB, exist_ok=True)
    for fn in os.listdir(PUB):          # 清旧（防跨标签残留）
        os.remove(os.path.join(PUB, fn))
    best = jload(os.path.join(FACE, f"face_best_{tag}.json"))
    shots = jload(os.path.join(FACE, f"face_shot_{tag}.json"))
    enh = jload(os.path.join(FACE, f"face_enhance_{etag}.json"), {})
    fus = jload(os.path.join(FACE, f"face_fusion_{ftag}.json"), {})
    emb = jload(os.path.join(FACE, f"face_embed_{mtag}.json"), {})
    liveness = jload(os.path.join(FACE, f"face_live_flags_{args.live_tag}.json"), {}) or {}
    if not best or not shots:
        raise SystemExit(f"missing face_best/face_shot for tag={tag}")
    lf_tid = liveness.get("tid_flags", {})

    crop_map = best["crop_map"]
    gp_label = best.get("gp_label", {})
    con = sqlite3.connect(GIL_DB)
    gp_total = con.execute("select count(*) from gperson where state='candidate'").fetchone()[0]
    con.close()

    # ---- 统计 ----
    all_shots = shots["shots"]
    n_total = len(all_shots)
    usable = [s for s in all_shots if s.get("usable")]
    per_cam = {}
    for cam in shots["meta"]["cams"]:
        ss = [s for s in all_shots if s["cam"] == cam]
        uu = [s for s in ss if s.get("usable")]
        per_cam[cam] = dict(shots=len(ss), usable=len(uu),
                            usable_assoc=sum(1 for s in uu if s.get("tid") is not None),
                            tids=len(set(s["tid"] for s in uu if s.get("tid") is not None)),
                            gps=len(set(s["gperson_id"] for s in uu if s.get("gperson_id"))))
    fus_ok = {f"{r['cam']}|{r['tid']}": r for r in fus.get("per_tid", []) if r.get("verdict") == "accepted"}
    enh_acc = [r for r in enh.get("records", []) if r.get("verdict") == "accepted"]
    n_enh = len(enh_acc)
    n_cand = len(emb.get("cross_top", []))

    # ---- 复制图片 ----
    copied = 0

    def cp(src, dst_name):
        nonlocal copied
        if src and os.path.exists(src):
            shutil.copy(src, os.path.join(PUB, dst_name))
            copied += 1
            return f"faces/{dst_name}"
        return None

    img_abf = {}       # "cam|tid" -> ab 文件
    for fn in (fus.get("ab_files") or []):
        parts = fn.replace(".jpg", "").split("_")
        if len(parts) >= 4:
            img_abf[f"{parts[2]}|{parts[3]}"] = cp(os.path.join(FACE, "ab_fused", fn),
                                                  f"abf_{parts[2]}_{parts[3]}.jpg")
    img_ft = {}
    for key, r in fus_ok.items():
        cam, tid = key.split("|")
        img_ft[key] = cp(os.path.join(FACE, "fused", r.get("fused_file") or ""), f"ft_{cam}_{tid}.jpg")

    # 增强：tid -> enh 文件（从 records）
    enh_by_tid = {}
    for r in enh.get("records", []):
        if r.get("verdict") == "accepted" and r.get("enh_file"):
            key = f"{r['cam']}|{r['tid']}"
            enh_by_tid[key] = cp(os.path.join(FACE, "crops_enhanced", r["enh_file"]), f"enh_{r['cam']}_{r['tid']}.jpg")

    # ---- 候选合并逐行对比缩略图（a/b 端点 → cand_{cam}_{tid}.jpg，高≤220px）----
    def cand_thumb(key):
        nonlocal copied
        cam, tid = key.split("|")
        dst_name = f"cand_{cam}_{tid}.jpg"
        dst = os.path.join(PUB, dst_name)
        if os.path.exists(dst):
            return f"faces/{dst_name}"
        fn = crop_map.get(f"{cam}_{tid}")
        src = os.path.join(FACE, "crops", fn) if fn else None
        if not (src and os.path.exists(src)):
            return None
        img = cv2.imread(src)
        if img is None:
            return None
        if img.shape[0] > 220:
            sc = 220 / img.shape[0]
            img = cv2.resize(img, (max(1, int(img.shape[1] * sc)), 220), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        copied += 1
        return f"faces/{dst_name}"

    # 复核对账产物（face_embed_review_v1.json）：簇扩展候选 / 320 谨慎标记
    review = jload(os.path.join(FACE, "face_embed_review_v1.json"), {}) or {}
    ext_map, caution_map = {}, {}
    for e in review.get("cluster_extensions", []):
        a, b = e["cand"]
        ext_map[f"{a}@{b}"] = (e["cluster"], e.get("note", ""))
        ext_map[f"{b}@{a}"] = (e["cluster"], e.get("note", ""))
    for e in review.get("caution_320", []):
        a, b = e["cand"]
        caution_map[f"{a}@{b}"] = e.get("note", "")
        caution_map[f"{b}@{a}"] = e.get("note", "")

    cand_rows = []
    for c in (emb.get("cross_top") or []):
        rec = dict(c, a_img=cand_thumb(c["a"]), b_img=cand_thumb(c["b"]))
        k = f"{c['a']}@{c['b']}"
        if k in ext_map:
            rec["ext_cluster"], rec["ext_note"] = ext_map[k]
        if k in caution_map:
            rec["caution"] = True
            rec["caution_note"] = caution_map[k]
        cand_rows.append(rec)

    persons = []
    for gpid, lst in sorted(best["per_gperson"].items(), key=lambda x: -int(x[0])):
        if not (lst and lst[0].get("usable")):
            continue
        s0 = lst[0]
        label = gp_label.get(str(gpid), f"g{gpid}")
        key = f"{s0['cam']}|{s0['tid']}"
        single = cp(os.path.join(FACE, "crops", crop_map.get(f"{s0['cam']}_{s0['tid']}", "")),
                    f"gp_{label}.jpg")
        fused = img_ft.get(key)
        enhanced = enh_by_tid.get(key)
        src = "fused" if fused else "single"
        best_img = fused if fused else single
        lf = lf_tid.get(key)
        lf_out = None
        if lf and lf.get("tier") in ("warn", "hint"):
            lf_out = dict(tier=lf["tier"], flags=lf.get("flags"), note=lf.get("note"),
                          top1_flagged=bool((lf.get("top1") or {}).get("flags")))
        persons.append(dict(
            gp=int(gpid), label=label, q=s0["q"], cam=s0["cam"], tid=s0["tid"],
            usable=bool(s0.get("usable")), w=s0.get("w"), pd=s0.get("pd"), lap=s0.get("lap"),
            n_usable=sum(1 for s in lst if s.get("usable")),
            img_single=single, img_fused=fused, img_enh=enhanced, img_best=best_img,
            ab=img_abf.get(key), src=src if fused else "single", lf=lf_out))

    # ---- 融合明细 ----
    fusion_rows = []
    for key, r in sorted(fus_ok.items()):
        cam, tid = key.split("|")
        fusion_rows.append(dict(
            cam=cam, tid=int(tid), k=r.get("k_kept"), ratio=r.get("lap_ratio"),
            single=r.get("single"), fused=r.get("fused"),
            ecc_min=r.get("ecc_min"), ncc_min=r.get("ncc_min"),
            window=r.get("window"), gp=r.get("gperson_id"),
            ab=img_abf.get(key)))

    # ---- 证据图 ----
    ev = {}
    ev["candidates"] = cp(os.path.join(FACE, "debug_assoc", "verify", "embed_candidates.jpg"), "embed_candidates.jpg")
    ev["montage_align"] = cp(os.path.join(FACE, "debug_assoc", "verify", "montage_316_88.jpg"), "montage_316_88.jpg")
    ev["wall"] = cp(os.path.join(FACE, f"face_wall_gperson_{tag}.jpg"), "wall.jpg")
    ev["merge_clusters"] = cp(os.path.join(FACE, "face_merge_clusters_v2.jpg"), "merge_clusters_v2.jpg")
    ab_enh = []
    for fn in (enh.get("ab_files") or [])[:18]:
        p = cp(os.path.join(FACE, f"ab_{tag}", fn), f"abenh_{fn}")
        if p:
            ab_enh.append(p)

    # ---- 复核完成摘要（face_embed_review_v2.json）----
    review2 = jload(os.path.join(FACE, "face_embed_review_v2.json"), {}) or {}
    review_block = None
    if review2.get("review"):
        rs = review2["review"]
        ms = review2.get("merge_suggestions", {}) or {}
        review_block = dict(total=rs.get("total"), conf=rs.get("conf"), rej=rs.get("rej"),
                            completed=bool(rs.get("completed")),
                            n_clusters=ms.get("n_clusters"), gp_before=ms.get("gperson_before"),
                            gp_after=ms.get("gperson_after"),
                            clusters=[dict(cid=c["cid"], cams=c["cams"],
                                           labels=list(dict.fromkeys(g["label"] for g in c["gpersons"])),
                                           n_tids=len(c["tids"]), cross=len(c["cams"]) > 1)
                                      for c in review2.get("clusters", [])])


    stats = dict(gperson_total=gp_total, gp_with_face=len(persons),
                 gp_cover=round(len(persons) / max(gp_total, 1), 4),
                 shots=n_total, usable=len(usable),
                 usable_rate=round(len(usable) / max(n_total, 1), 4),
                 tids=len({(s["cam"], s["tid"]) for s in usable if s.get("tid") is not None}),
                 fused_accepted=len(fus_ok), gp_fused=fus.get("n_gp_fused", 0),
                 enh_accepted=n_enh, n_cand=n_cand, per_cam=per_cam)
    if liveness:
        ls = liveness.get("summary", {}) or {}
        stats["liveness"] = dict(flagged_shots=ls.get("n_flagged"), shots=ls.get("n_shots"),
                                 tids_flagged=ls.get("n_tids_flagged"), tiers=ls.get("tiers"),
                                 n_persons_lf=sum(1 for p in persons if p.get("lf")))
    data = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), tag=tag,
                  gperson_space=(shots.get("meta", {}) or {}).get("relink", {}).get("gperson_space", "store_v1"),
                  note="人脸证据页：P0抓取 + P0b重关联 + P1增强 + P1b多帧融合 + E嵌入可行性 + p0c重链接（store_v1 s3v2+人工裁决空间）+ L0质检flag；只读产物，未改 GIL"),
        stats=stats,
        persons=persons, fusion=fusion_rows,
        enhance=dict(accepted=enh_acc, ab=ab_enh, summary=enh.get("summary", {})),
        embed=dict(stats=emb.get("stats", {}), top=cand_rows,
                   same_bottom=(emb.get("same_pairs") or [])[:8]),
        liveness=dict(summary=liveness.get("summary"), review=liveness.get("review"),
                      tids=lf_tid) if liveness else None,
        review=review_block,
        evidence=ev)
    with open(os.path.join(PUB, "face_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=float)
    n_files = len(os.listdir(PUB))
    print(f"faces data -> {PUB}: persons={len(persons)} fusion={len(fusion_rows)} "
          f"enh={n_enh} copied_img={copied} total_files={n_files}", flush=True)


if __name__ == "__main__":
    main()
