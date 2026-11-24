#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 `data-request-v2-2026-10-19.md`（D1/D2/D6）导出可自动产出的数据账本。

只做**导出**，不改任何判定逻辑、不改 profile、不重跑实验：
  D1  tracks.csv            图库 579 + 查询检测轨迹 295 = 874 行
  D2  outputs/<config>.csv  每配置 229 行（逐查询 a/b 簇、分数、接受、最终输出）
  D6  clusters_geom.csv     每配置全部 kept 簇的几何统计
      cooccur_pairs.csv     同帧共现对（同时在场 = 铁定不同人）
  --  summary.csv           配置级汇总（D7 的表头格式）
      veto_transitions.csv  baseline -> veto 的逐条转移（回答 90->87、152->121）

判定管线与 `eval_det_switch.py::run`（proto / camera-aware）逐位一致：
会话均值对齐 -> 簇原型余弦 -> 每身份 z 校准 -> 同一 θ 扫描 + self-cal 选点。

簇编号有两套，文件里同时给：
  cluster_idx = kept 列表下标（`cluster_purity.csv` 的 `c` 列、论文里说的“簇 36”）
  cluster_lab = 聚类输出标签（数据侧 id，不连续）

用法：
  python tools/export_paper_ledger.py
  python tools/export_paper_ledger.py --out deliverable/data-request-v2
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
import cluster_camera_aware as CA  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402
from render_head_track_audit import find_head_tracks  # noqa: E402

PROFILE = os.path.join(ROOT, "configs/selector_deploy.json")
SEG_META = os.path.join(ROOT, "media-resource/1012/segments/segments_meta.json")
AUDIT = [
    (os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/baseline_audit/index.csv"),
     """01=ok 02=ok 03=ok 04=ok 05=ok 06=ok 07=ok 08=bad 09=ok 10=bad
        11=ok 12=ok 13=ok 14=bad 15=bad 16=ok 17=ok 18=bad 19=bad 20=bad 21=ok 22=bad
        23=bad 24=ok 25=bad"""),
    (os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/baseline_audit2/index.csv"),
     """01=ok 02=ok 03=bad 04=ok 05=ok 06=ok 07=ok 08=bad 09=bad 10=ok
        11=ok 12=ok 13=ok 14=ok 15=ok 16=bad 17=bad 18=ok 19=unk 20=bad 21=ok 22=unk
        23=ok 24=unk 25=bad 26=bad"""),
]
FPS = 25.0          # 由 dur=3600s / 90000 帧、dur=900s / 22500 帧推得


# ---------------------------------------------------------------- 基础装载
def load_meta():
    meta = json.load(open(SEG_META, encoding="utf-8"))
    return {s["seg"]: s for s in meta["segments"]}


def read_dets(seg):
    """{frame: (tids, boxes)} + {tid: [frames]} + {tid: [heights]}"""
    p = os.path.join(ROOT, f"output_crosscam_local/1012_yolo8_dd/{seg}/{seg}_detections.jsonl")
    frames, by_tid, h_by_tid = {}, collections.defaultdict(list), collections.defaultdict(list)
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            tids, boxes = o["track_ids"], o["boxes"]
            frames[o["frame"]] = (tids, boxes)
            for t, b in zip(tids, boxes):
                if t is None or t < 0:
                    continue
                by_tid[int(t)].append(o["frame"])
                h_by_tid[int(t)].append(float(b[3]) - float(b[1]))
    return frames, by_tid, h_by_tid


def halves(fr_list):
    """与 make_ab_jsonl.py 完全一致：按轨迹自身帧序列中点切。"""
    frs = sorted(fr_list)
    k = len(frs) // 2
    return set(frs[:k]), set(frs[k:])


def load_truth():
    """39 个已人工判定簇：{cluster_idx: ok|bad|unk}（同簇多次判定取非 unk）。"""
    cl = collections.defaultdict(list)
    for p, spec in AUDIT:
        mp = dict(x.split("=") for x in spec.split())
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            cl[int(r["cluster"])].append(mp.get(r["image"][:2], "unk"))
    return {c: (([x for x in v if x != "unk"] or ["unk"])[0]) for c, v in cl.items()}


def npz_keys(emb_dir, seg):
    za = np.load(os.path.join(emb_dir, f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(emb_dir, f"{seg}_emb_b.npz"))
    a = set(int(str(k).rsplit("#", 1)[1]) for k in za.files)
    b = set(int(str(k).rsplit("#", 1)[1]) for k in zb.files)
    return a, b


# ---------------------------------------------------------------- 管线
def build(emb_dir, det_dir, veto=None, thr_in=0.15, thr_out=0.25, gid_of_lab=None):
    ktr, Atr, Btr = M.load_split(emb_dir, S.TRAIN_SEGS)
    kte, Ate, Bte = M.load_split(emb_dir, S.TEST_SEGS)
    Vtr = _norm((Atr + Btr) / 2.0)
    cams = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams, thr_in, thr_out, S.MIN_CLUSTER)
    lab_all, kept_all = lab.copy(), list(kept)          # 否决前，供诊断
    if gid_of_lab is None:                              # 首次调用：以本次 kept 建立全局簇号
        gid_of_lab = {c: i for i, c in enumerate(kept)}
    vetoed = set()
    if veto is not None:
        for c in kept:
            idx = np.where(lab == c)[0]
            if idx.size >= 2:
                Xc = Vtr[idx]
                Sx = Xc @ Xc.T
                dmax = float(1.0 - Sx[np.triu_indices(idx.size, 1)].min())
            else:
                dmax = 0.0
            if dmax >= veto:
                vetoed.add(c)
        kept = [c for c in kept if c not in vetoed]

    K = len(kept)
    mu_g = Vtr.mean(0)
    protos = _norm(np.asarray([_norm(Vtr[lab == c].mean(0, keepdims=True))[0] for c in kept],
                              np.float32) - mu_g)
    pcl = np.arange(K)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Ate + Bte) / 2.0).mean(0)
    Sa = S.score_rows(Ate, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Bte, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))

    cooc = []
    for s in S.TEST_SEGS:
        p = os.path.join(det_dir, s, f"{s}_detections.jsonl")
        if os.path.exists(p):
            cooc += [((s, u), (s, v)) for u, v in cooc_pairs(p, set(kte), s, 5)]

    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    rows = S.sweep(Sa, Sb, kte, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None

    # 原始（未聚类、未校准）全库 top-1 余弦，用于 §5.3
    raw_a = (Ate @ Vtr.T)
    raw_b = (Bte @ Vtr.T)
    return dict(ktr=ktr, kte=kte, Ate=Ate, Bte=Bte, Vtr=Vtr, cams=cams, lab=lab,
                lab_all=lab_all, kept=kept, kept_all=kept_all, vetoed=vetoed, K=K,
                gid_of_lab=gid_of_lab,
                gid=[gid_of_lab[c] for c in kept],
                Sa=Sa, Sb=Sb, cooc=cooc, rows=rows, best=best,
                theta=float(best["thr"]) if best else float("nan"),
                raw_a=raw_a, raw_b=raw_b)


def cluster_geom(V, cams, lab, kept, c, track_ids):
    m = lab == c
    idx = np.where(m)[0]
    n = int(idx.size)
    cc = collections.Counter(cams[m].tolist())
    g = {"n": n, "n_cam": len(cc), "cam_share": round(max(cc.values()) / n, 4),
         "cams": "|".join(sorted(cc)), }
    if n >= 2:
        X = V[idx]
        Sx = X @ X.T
        iu = np.triu_indices(n, 1)
        pw = Sx[iu]
        D = 1.0 - pw
        g.update(pw_min=round(float(pw.min()), 4), pw_max=round(float(pw.max()), 4),
                 d_max=round(float(D.max()), 4), d_p95=round(float(np.percentile(D, 95)), 4),
                 d_mean=round(float(D.mean()), 4))
        cent = X.mean(0)
        cent /= np.linalg.norm(cent) + 1e-12
        g["cent_mean"] = round(float((1.0 - X @ cent).mean()), 4)
        # 最小余弦 = 最大距离
        i, j = np.unravel_index(np.argmin(np.where(np.eye(n, dtype=bool), 9.0, Sx)), Sx.shape)
        g["dmax_i"] = track_ids[idx[i]]
        g["dmax_j"] = track_ids[idx[j]]
        g["dmax_pair_same_cam"] = int(cams[idx[i]] == cams[idx[j]])
        Dm = np.array([[0.0 if a == b else 1.0 - float(X[a] @ X[b]) for b in range(n)]
                       for a in range(n)], np.float64)
        Mst = minimum_spanning_tree(Dm).toarray()
        e = np.sort(Mst[Mst > 0])
        g["mst_max"] = round(float(e[-1]), 4) if e.size else 0.0
        g["mst_gap"] = round(float(e[-1] - e[-2]), 4) if e.size >= 2 else 0.0
        if e.size:
            big = e[-1]
            a, b = np.argwhere(np.isclose(Mst, big))[0]
            g["mst_max_i"] = track_ids[idx[a]]
            g["mst_max_j"] = track_ids[idx[b]]
            g["mst_max_cross_cam"] = int(cams[idx[a]] != cams[idx[b]])
        from cluster_camera_aware import _labels_by_thr
        sub = _labels_by_thr(X, 0.12)
        cnt = collections.Counter(sub.tolist())
        g["sub_share"] = round(max(cnt.values()) / n, 4)
        g["sub_k"] = len(cnt)
        # 各机位子簇（机位内 0.15 再切）与跨机位质心距离，D6 要求
        parts, cents = [], {}
        for cam in sorted(set(cams[m].tolist())):
            sel = np.where(cams[m] == cam)[0]
            loc = _labels_by_thr(X[sel], 0.15) if sel.size > 1 else np.zeros(sel.size, int)
            for li in sorted(set(loc.tolist())):
                member = sel[loc == li]
                parts.append(f"{cam}:{li}({member.size})")
            v = X[sel].mean(0)
            cents[cam] = v / (np.linalg.norm(v) + 1e-12)
        g["sub_clusters"] = "|".join(parts)
        pairs = []
        cams_sorted = sorted(cents)
        for a_i in range(len(cams_sorted)):
            for b_i in range(a_i + 1, len(cams_sorted)):
                ca_, cb_ = cams_sorted[a_i], cams_sorted[b_i]
                pairs.append(f"{ca_}-{cb_}:{1.0 - float(cents[ca_] @ cents[cb_]):.3f}")
        g["centroid_dists"] = "|".join(pairs)
    else:
        for k in ["pw_min", "pw_max", "d_max", "d_p95", "d_mean", "cent_mean",
                  "mst_max", "mst_gap", "sub_share", "sub_k",
                  "sub_clusters", "centroid_dists"]:
            g[k] = ""
        g["dmax_i"] = g["dmax_j"] = g["mst_max_i"] = g["mst_max_j"] = ""
        g["dmax_pair_same_cam"] = g["mst_max_cross_cam"] = ""
    return g


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--out", default=os.path.join(ROOT, "deliverable/data-request-v2"))
    a = ap.parse_args()

    prof = json.load(open(a.profile, encoding="utf-8"))
    primary = prof.get("primary", "yolo8dd")
    emb = os.path.join(ROOT, prof["variants"][primary])
    det = os.path.join(ROOT, prof["cooc_map"][primary])
    clu = prof.get("cluster", {})
    thr_in, thr_out = float(clu.get("thr_in", 0.15)), float(clu.get("thr_out", 0.25))
    veto_thr = clu.get("veto_dmax")

    print(f"[cfg] detector={prof.get('detector')} emb={emb}")
    print(f"[cfg] cluster camera-aware thr_in={thr_in} thr_out={thr_out} veto={veto_thr}")

    meta = load_meta()
    out_dir = a.out
    for sub in ("ledger", "ledger/outputs", "gt", "gt/crops"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # ---------- D1 轨迹总账（检测层面，295 查询 + 579 图库） ----------
    print("\n[D1] 轨迹总账")
    tracks, det_cache = [], {}
    for split, segs in (("gallery", S.TRAIN_SEGS), ("query", S.TEST_SEGS)):
        for seg in segs:
            frames, by_tid, h_by_tid = read_dets(seg)
            det_cache[seg] = (frames, by_tid, h_by_tid)
            ka, kb = npz_keys(emb, seg)
            m = meta.get(seg, {})
            cam = seg.split("_")[0]
            heads = find_head_tracks(seg, os.path.join(det, seg, f"{seg}_detections.jsonl"))
            for tid in sorted(by_tid):
                fr = by_tid[tid]
                ha, hb = halves(fr)
                in_a, in_b = tid in ka, tid in kb
                n = len(fr)
                if n < 6:
                    ev, why = 0, "n_frames_lt_6"
                elif not in_a and not in_b:
                    ev, why = 0, "no_half_crop"
                elif not in_a:
                    ev, why = 0, "missing_in_half_a"
                elif not in_b:
                    ev, why = 0, "missing_in_half_b"
                else:
                    ev, why = 1, ""
                hmin = float(np.min(h_by_tid[tid]))
                if not ev and hmin < 48:
                    why += "+crop_below_48"
                t0, t1 = min(fr), max(fr)
                sec0 = t0 / FPS
                osd = m.get("osd_start", "")
                clock0 = ""
                if osd:
                    hh, mm = [int(x) for x in osd.split(":")]
                    tot = hh * 3600 + mm * 60 + sec0
                    clock0 = f"{int(tot // 3600) % 24:02d}:{int(tot % 3600 // 60):02d}:{tot % 60:05.2f}"
                tracks.append({
                    "track_id": f"{seg}#{tid}", "split": split, "seg": seg, "cam": cam,
                    "tid": tid, "n_frames": n,
                    "t_start_frame": t0, "t_end_frame": t1,
                    "t_start_s": round(sec0, 2), "t_end_s": round(t1 / FPS, 2),
                    "t_start_clock": clock0,
                    "median_box_h": round(float(np.median(h_by_tid[tid])), 1),
                    "min_box_h": round(hmin, 1),
                    "n_crops_a": len(ha), "n_crops_b": len(hb),
                    "half_a_frames": len(ha), "half_b_frames": len(hb),
                    "box_type_auto": "head_like" if tid in heads else "body",
                    "head_host_tid": heads[tid][0] if tid in heads else "",
                    "head_nested_ratio": round(heads[tid][1], 3) if tid in heads else "",
                    "evaluable": ev, "exclude_reason": why,
                    "duplicate_of": "", "crop_dir": f"gt/crops/{seg}",
                })
    # tracks.csv 严格按清单 874 行：图库只列可评估 579 条，查询列全部 295 条
    sel = [t for t in tracks if t["split"] == "query" or t["evaluable"]]
    tp = os.path.join(out_dir, "ledger/tracks.csv")
    with open(tp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tracks[0].keys()))
        w.writeheader()
        w.writerows(sel)
    tpa = os.path.join(out_dir, "ledger/tracks_all.csv")
    with open(tpa, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tracks[0].keys()))
        w.writeheader()
        w.writerows(tracks)
    n_g = sum(1 for t in tracks if t["split"] == "gallery")
    n_q = sum(1 for t in tracks if t["split"] == "query")
    n_qe = sum(1 for t in tracks if t["split"] == "query" and t["evaluable"])
    n_ge = sum(1 for t in tracks if t["split"] == "gallery" and t["evaluable"])
    print(f"  {tp}: {len(sel)} 行（图库可评估 {n_ge} / 查询 {n_q}）（应 874）")
    print(f"  {tpa}: {len(tracks)} 行（图库检测层全部 {n_g} / 查询 {n_q}）")
    print(f"  evaluable: 图库 {n_ge} / 查询 {n_qe}（应 579 / 229）")
    why = collections.Counter(t["exclude_reason"] for t in tracks
                              if t["split"] == "query" and not t["evaluable"])
    print(f"  查询未入可评估集 {sum(why.values())} 条: {dict(why)}")
    whg = collections.Counter(t["exclude_reason"] for t in tracks
                              if t["split"] == "gallery" and not t["evaluable"])
    print(f"  图库未入可评估集 {sum(whg.values())} 条: {dict(whg)}")

    # ---------- 两个配置 ----------
    cfgs = [("baseline", None)]
    if veto_thr is not None:
        cfgs.append((f"veto{veto_thr:.2f}", float(veto_thr)))
    R = {}
    gid_of_lab = None
    for name, vt in cfgs:
        print(f"\n[{name}] 建簇 + 打分 + θ 扫描 …")
        R[name] = build(emb, det, veto=vt, thr_in=thr_in, thr_out=thr_out, gid_of_lab=gid_of_lab)
        if gid_of_lab is None:
            gid_of_lab = R[name]["gid_of_lab"]
        b = R[name]["best"]
        print(f"  K={R[name]['K']} θ={b['thr']:.2f} cov={b['coverage']:.3f} "
              f"same={b['same']:.3f} fm={b['false_merge']*100:.2f}% eval={b['evaluable']}")

    # ---------- D2 逐查询输出 ----------
    qmeta = {(t["seg"], t["tid"]): t for t in tracks if t["split"] == "query"}
    base = R["baseline"]
    base_idx = {}
    for i, k in enumerate(base["kte"]):
        base_idx[k] = int(base["Sa"][i].argmax())

    for name, r in R.items():
        kept, lab, theta = r["kept"], r["lab"], r["theta"]
        kept_pos = {c: i for i, c in enumerate(kept)}
        lab_kept = {c: i for i, c in enumerate(r["kept_all"])}
        rows = []
        for i, k in enumerate(r["kte"]):
            sa = r["Sa"][i]
            sb = r["Sb"][i]
            ia, ib = int(sa.argmax()), int(sb.argmax())
            va, vb = float(sa.max()), float(sb.max())
            ca, cb = kept[ia], kept[ib]
            acca, accb = va >= theta, vb >= theta
            same = (ca == cb)
            if acca and accb and same:
                fin, st = r["gid"][ia], "accepted"
            elif not (acca and accb):
                fin, st = "", "rejected_low_score"
            else:
                fin, st = "", "rejected_ab_split"
            # baseline 的 top-1 簇在本配置是否被弃用
            bca = base["kept"][base_idx[k]]
            vet_hit = int(bca in r["vetoed"]) if r["vetoed"] else 0
            q = qmeta[k]
            # 原始全库 top-1（未聚类、未校准）
            row_raw = r["raw_a"][i]
            j_raw = int(row_raw.argmax())
            rows.append({
                "query_id": f"{k[0]}#{k[1]}", "seg": k[0], "tid": k[1],
                "cam": k[0].split("_")[0], "config": name, "theta": round(theta, 2),
                "cluster_a_idx": r["gid"][ia], "cluster_a_lab": int(ca),
                "cluster_a_size": int((lab == ca).sum()),
                "cluster_a_n_cam": len(set(r["cams"][lab == ca].tolist())),
                "score_a": round(va, 3), "raw_cos_a": round(float(r["Ate"][i] @ r["Vtr"][j_raw]), 4),
                "cluster_b_idx": r["gid"][ib], "cluster_b_lab": int(cb),
                "score_b": round(vb, 3),
                "accept_a": int(acca), "accept_b": int(accb),
                "ab_same_cluster": int(same),
                "final_cluster_idx": fin, "final_status": st,
                "vetoed_by_baseline_argmax": vet_hit,
                "top1_raw_cos": round(float(row_raw.max()), 4),
                "top1_raw_track": f"{r['ktr'][j_raw][0]}#{r['ktr'][j_raw][1]}",
                "n_frames": q["n_frames"],
                "median_box_h": q["median_box_h"],
                "box_type_auto": q["box_type_auto"],
            })
        op = os.path.join(out_dir, f"ledger/outputs/{name}.csv")
        with open(op, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        both = sum(1 for x in rows if x["accept_a"] and x["accept_b"])
        same_n = sum(1 for x in rows if x["ab_same_cluster"] and x["accept_a"] and x["accept_b"])
        acc = sum(1 for x in rows if x["final_status"] == "accepted")
        print(f"  {op}: {len(rows)} 行 | 两片均接受 {both} | 两片同簇 {same_n} | 最终输出 {acc}")

    # ---------- D6 簇几何 ----------
    gp = os.path.join(out_dir, "ledger/clusters_geom.csv")
    truth = load_truth()
    grow = []
    for name, r in R.items():
        tid_of = [f"{k[0]}#{k[1]}" for k in r["ktr"]]
        for ci, c in enumerate(r["kept"]):
            g = cluster_geom(r["Vtr"], r["cams"], r["lab"], r["kept"], c, tid_of)
            gid = r["gid"][ci]
            grow.append({"config": name, "cluster_idx": gid, "cluster_lab": int(c),
                         **g, "vetoed": 0,
                         "purity_verdict": truth.get(gid, "") if name == "baseline" else "",
                         "is_key_cluster": int(gid in truth)})
        for c in sorted(r["vetoed"]):
            gid = gid_of_lab[c]
            g = cluster_geom(r["Vtr"], r["cams"], r["lab_all"], r["kept_all"], c, tid_of)
            grow.append({"config": name, "cluster_idx": gid, "cluster_lab": int(c),
                         **g, "vetoed": 1,
                         "purity_verdict": truth.get(gid, ""),
                         "is_key_cluster": int(gid in truth)})
    with open(gp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(grow[0].keys()))
        w.writeheader()
        w.writerows(grow)
    print(f"\n[D6] {gp}: {len(grow)} 行（含被否决簇，vetoed=1）")

    # ---------- D3 基础：图库簇成员表（按全局簇号） ----------
    mp = os.path.join(out_dir, "ledger/gallery_members_by_cluster.csv")
    mrow = []
    for ci, c in enumerate(base["kept"]):
        gid = base["gid"][ci]
        for i in np.where(base["lab"] == c)[0]:
            seg, tid = base["ktr"][i]
            mrow.append({"cluster_idx": gid, "cluster_lab": int(c),
                         "track_id": f"{seg}#{tid}", "seg": seg, "tid": int(tid),
                         "cam": seg.split("_")[0]})
    with open(mp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mrow[0].keys()))
        w.writeheader()
        w.writerows(mrow)
    print(f"[D3] {mp}: {len(mrow)} 行（{len(base['kept'])} 个簇的全部成员）")

    # ---------- D6 共现对 ----------
    cp = os.path.join(out_dir, "ledger/cooccur_pairs.csv")
    kidx = {k: i for i, k in enumerate(base["kte"])}
    frames_cache = {s: det_cache[s][0] for s in S.TEST_SEGS}
    crow = []
    for (su, tu), (sv, tv) in base["cooc"]:
        n_cf = 0
        for fr, (tids, _b) in frames_cache[su].items():
            if tu in tids and tv in tids:
                n_cf += 1
        merged = {}
        for name, r in R.items():
            if (su, tu) not in r["kte"] or (su, tv) not in r["kte"]:
                merged[name] = ""
                continue
            iu, iv = r["kte"].index((su, tu)), r["kte"].index((su, tv))
            ia, ib = int(r["Sa"][iu].argmax()), int(r["Sb"][iv].argmax())
            acc = (float(r["Sa"][iu].max()) >= r["theta"] and float(r["Sb"][iv].max()) >= r["theta"])
            merged[name] = int(acc and r["kept"][ia] == r["kept"][ib])
        crow.append({"seg": su, "track_i": f"{su}#{tu}", "track_j": f"{sv}#{tv}",
                     "cam": su.split("_")[0], "n_cooccur_frames": n_cf,
                     **{f"merged_{k}": v for k, v in merged.items()},
                     "same_person_dup": ""})
    with open(cp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crow[0].keys()))
        w.writeheader()
        w.writerows(crow)
    print(f"[D6] {cp}: {len(crow)} 对（应 1120）")

    # ---------- summary ----------
    sp = os.path.join(out_dir, "ledger/summary.csv")
    srow = []
    for name, r in R.items():
        b = r["best"]
        rows_q = [x for x in crow]
        ev = sum(1 for u, v in r["cooc"]
                 if ((float(r["Sa"][r["kte"].index(u)].max()) >= r["theta"] and
                      float(r["Sb"][r["kte"].index(v)].max()) >= r["theta"]) or
                     (float(r["Sa"][r["kte"].index(v)].max()) >= r["theta"] and
                      float(r["Sb"][r["kte"].index(u)].max()) >= r["theta"])))
        srow.append({"config": name, "theta": round(r["theta"], 2), "K": r["K"],
                     "n_gallery": len(r["ktr"]), "n_query": len(r["kte"]),
                     "n_cooc_total": len(r["cooc"]), "n_cooc_evaluable": ev,
                     "both_accept_rate": round(b["coverage"], 4),
                     "ab_same_rate": round(b["same"], 4),
                     "false_merge": round(b["false_merge"], 5),
                     "final_output_rate": "",
                     "C": "", "W": "", "U": "", "precision": "", "C_over_229": "",
                     "note": "C/W/U 需 D4 查询真值，本表留空；final_output_rate 见 outputs/*.csv 计数"})
    with open(sp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srow[0].keys()))
        w.writeheader()
        w.writerows(srow)
    print(f"[sum] {sp}")

    # ---------- veto 逐条转移 ----------
    if len(R) > 1:
        vname = [n for n in R if n != "baseline"][0]
        b, v = R["baseline"], R[vname]
        tb = {}
        for i, k in enumerate(b["kte"]):
            sa, sb = float(b["Sa"][i].max()), float(b["Sb"][i].max())
            tb[k] = (int(sa >= b["theta"] and sb >= b["theta"]), int(b["Sa"][i].argmax() == b["Sb"][i].argmax()))
        tv = {}
        for i, k in enumerate(v["kte"]):
            sa, sb = float(v["Sa"][i].max()), float(v["Sb"][i].max())
            tv[k] = (int(sa >= v["theta"] and sb >= v["theta"]), int(v["Sa"][i].argmax() == v["Sb"][i].argmax()))
        trow = []
        for k in b["kte"]:
            ab, asm = tb[k]
            vb, vsm = tv[k]
            if ab and vb:
                st = "kept"
            elif ab and not vb:
                st = "dropped"
            elif not ab and vb:
                st = "added"
            else:
                st = "rejected_both"
            trow.append({"query_id": f"{k[0]}#{k[1]}", "cam": k[0].split("_")[0],
                         "baseline_accept": ab, "baseline_same": asm,
                         "veto_accept": vb, "veto_same": vsm, "transition": st})
        tp2 = os.path.join(out_dir, "ledger/veto_transitions.csv")
        with open(tp2, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(trow[0].keys()))
            w.writeheader()
            w.writerows(trow)
        c = collections.Counter(x["transition"] for x in trow)
        print(f"[veto] {tp2}: {dict(c)}")

    # ---------- 覆盖校验 ----------
    print("\n== 校验 ==")
    for name, r in R.items():
        b = r["best"]
        print(f"{name:10s} K={r['K']:3d} cov={b['coverage']:.3f} same={b['same']:.3f} "
              f"fm={b['false_merge']*100:.2f}% ev={b['evaluable']} θ={r['theta']:.2f}")
    print(f"D1 期望 874 行（{len(tracks)}）; 图库 evaluable 579（{n_ge}）; 查询 229（{n_qe}）")


if __name__ == "__main__":
    main()
