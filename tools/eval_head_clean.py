#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""头框轨迹的两个修正动作 + 指标负对修正，一次跑完对照矩阵。

三个变量维度：
  变体（图库侧）
    base             现有生产
    drop_head_clusters  把「头框成员 ≥50% 且 size≥3」的簇标为非身份簇，不参与匹配
    drop_head_tracks    从 train 图库里直接删掉头框轨迹，重聚类
    inherit             测试侧：头框不独立查询，继承其宿主（同一个人）的簇与分数
  负对（指标侧）
    dirty            现在的口径：全部共现对
    clean            剔除「头框 ↔ 其宿主」这类同人重复检测对（假负对）
  报告阈值
    self-cal：same≥0.90 且 evaluable≥100 中取覆盖最大
    另附 fm≤2% 约束下的结果（生产目前未启用该约束）

用法：
    python tools/eval_head_clean.py
输出：
    selector_osnet_1012/head_clean_matrix.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402
from render_head_track_audit import find_head_tracks  # noqa: E402

EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/head_clean_matrix.json")
GRID = np.round(np.arange(-2.0, 8.01, 0.05), 3)


def head_maps():
    tr, te = {}, {}
    for seg in S.TRAIN_SEGS:
        for t, v in find_head_tracks(seg).items():
            tr[(seg, t)] = v[0]
    for seg in S.TEST_SEGS:
        for t, v in find_head_tracks(seg).items():
            te[(seg, t)] = v[0]
    return tr, te


def build(ktr, Vtr, drop_clusters=False, head_tr=frozenset(), mode="camera",
          thr_in=0.15, thr_out=0.25):
    """聚类 → 原型 → 每身份尾部统计。mode=camera 为生产口径（机位感知）。"""
    if mode == "camera":
        import cluster_camera_aware as CA
        cams = np.array([k[0].split("_")[0] for k in ktr])
        lab, kept = CA.camera_aware_labels(Vtr, cams, thr_in, thr_out, S.MIN_CLUSTER)
    else:
        lab, kept = S.hier_labels(Vtr)
    sizes = {c: int((lab == c).sum()) for c in kept}
    head_of_cl = {c: sum(1 for i, k in enumerate(ktr) if lab[i] == c and k in head_tr) for c in kept}
    if drop_clusters:
        keep = [c for c in kept
                if not (sizes[c] >= 3 and head_of_cl[c] / max(sizes[c], 1) >= 0.50)]
    else:
        keep = list(kept)
    protos, pcl = [], []
    for ci, c in enumerate(keep):
        m = lab == c
        if m.sum() == 0:
            continue
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - Vtr.mean(0))
    pcl = np.asarray(pcl)
    K = len(keep)
    mu_g = Vtr.mean(0)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, keep)
    return dict(lab=lab, kept=keep, protos=protos, pcl=pcl, K=K, calib=(m_hat, s_hat), mu_g=mu_g,
                n_head_clusters=len(kept) - len(keep),
                head_cluster_tracks=sum(sizes[c] for c in kept
                                       if c not in keep))


def scores_for(d, Qa, Qb):
    mu_q = _norm((Qa + Qb) / 2.0).mean(0)
    Sa = S.score_rows(Qa, d["protos"], d["pcl"], d["K"], mu=mu_q, calib=d["calib"])
    Sb = S.score_rows(Qb, d["protos"], d["pcl"], d["K"], mu=mu_q, calib=d["calib"])
    return Sa, Sb


def apply_inherit(Sa, Sb, kte, kids, te_head, thr):
    """头框的 a/b 各自继承宿主对应半段的结果（宿主该半段被接受时才继承）。"""
    Sa, Sb = Sa.copy(), Sb.copy()
    n_inh = 0
    for k, host in te_head.items():
        if k not in kids or (k[0], host) not in kids:
            continue
        i, j = kids[k], kids[(k[0], host)]
        for Sm in (Sa, Sb):
            if Sm[j].max() >= thr:
                c = int(Sm[j].argmax())
                Sm[i, :] = -1e9
                Sm[i, c] = Sm[j, c]
                n_inh += 1
    return Sa, Sb, n_inh


def evaluate(Sa, Sb, kte, cooc, same_min=0.90, min_eval=100):
    rows = S.sweep(Sa, Sb, kte, cooc, GRID)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= same_min and r["evaluable"] >= min_eval]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    ok2 = [r for r in ok if np.isfinite(r["false_merge"]) and r["false_merge"] <= 0.02]
    best2 = max(ok2, key=lambda r: r["coverage"]) if ok2 else None
    return rows, best, best2


def head_stats(Sa, Sb, kte, te_head, thr):
    """头框里被双片接受多少、其中落到宿主同簇多少。"""
    kidx = {k: i for i, k in enumerate(kte)}
    n = acc = same_as_host = 0
    for k, host in te_head.items():
        if k not in kidx:
            continue
        i = kidx[k]
        n += 1
        if Sa[i].max() >= thr and Sb[i].max() >= thr:
            acc += 1
            kh = (k[0], host)
            if kh in kidx:
                j = kidx[kh]
                if int(Sa[i].argmax()) == int(Sa[j].argmax()):
                    same_as_host += 1
    return dict(n_eval=n, accepted=acc, same_cluster_as_host=same_as_host)


def main():
    tr_head, te_head = head_maps()
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr_all = _norm((Va_tr + Vb_tr) / 2.0)

    # 负对：dirty（全部共现对） / clean（剔除 头框↔宿主）
    keys_ok = set(kte)
    dirty = []
    for seg in S.TEST_SEGS:
        for a, b in cooc_pairs(os.path.join(DET, seg, f"{seg}_detections.jsonl"), keys_ok, seg, 5):
            dirty.append(((seg, a), (seg, b)))
    clean, n_fake = [], 0
    for u, v in dirty:
        if (u in te_head and te_head[u] == v[1]) or (v in te_head and te_head[v] == u[1]):
            n_fake += 1
            continue
        clean.append((u, v))
    print(f"[neg] 全部共现对 {len(dirty)}，剔除头框↔宿主假负对 {n_fake} → 干净口径 {len(clean)}")

    all_variants = {}
    for mode in ("camera", "global"):
        variants = {}
        d = build(ktr, Vtr_all, drop_clusters=False, head_tr=tr_head, mode=mode)
        variants["base"] = (d, ktr)
        d2 = build(ktr, Vtr_all, drop_clusters=True, head_tr=tr_head, mode=mode)
        variants["drop_head_clusters"] = (d2, ktr)
        keepi = [i for i, k in enumerate(ktr) if k not in tr_head]
        ktr3 = [ktr[i] for i in keepi]
        Vtr3 = _norm(Vtr_all[keepi])
        d3 = build(ktr3, Vtr3, drop_clusters=False, head_tr=frozenset(), mode=mode)
        variants["drop_head_tracks"] = (d3, ktr3)
        variants["inherit"] = (d, ktr)
        for name, (dd, _kt) in variants.items():
            all_variants[f"{mode}:{name}"] = (dd, name)

    res = {"negatives": {"n_all": len(dirty), "n_fake_head_host": n_fake, "n_clean": len(clean)},
           "variants": {}}
    for vname, (dd, base_name) in all_variants.items():
        Qa, Qb = Va_te, Vb_te
        Sa, Sb = scores_for(dd, Qa, Qb)
        kids = {k: i for i, k in enumerate(kte)}
        thr0 = 2.5
        _, b0, _ = evaluate(Sa, Sb, kte, dirty)
        if b0:
            thr0 = b0["thr"]
        n_inh = 0
        if base_name == "inherit":
            Sa, Sb, n_inh = apply_inherit(Sa, Sb, kte, kids, te_head, thr0)
        entry = dict(K=dd["K"], n_head_clusters=dd["n_head_clusters"],
                     head_cluster_tracks=dd["head_cluster_tracks"], n_inherited=n_inh, neg={})
        for negname, neg in (("dirty", dirty), ("clean", clean)):
            rows, best, best2 = evaluate(Sa, Sb, kte, neg)
            hs = head_stats(Sa, Sb, kte, te_head, best["thr"]) if best else None
            entry["neg"][negname] = dict(self_cal=best, with_fm2=best2, head=hs)
        res["variants"][vname] = entry
        b = entry["neg"]["clean"]["self_cal"]
        bd = entry["neg"]["dirty"]["self_cal"]
        print("%-26s K=%-4d 头簇%-2d | dirty cov=%.4f same=%.4f fm=%.3f%% | clean cov=%.4f same=%.4f fm=%.3f%% (ev %d) | 头框接受 %d/%d 同簇 %d"
              % (vname, dd["K"], dd["n_head_clusters"],
                 bd["coverage"], bd["same"], 100 * bd["false_merge"] if np.isfinite(bd["false_merge"]) else -1,
                 b["coverage"], b["same"], 100 * b["false_merge"] if np.isfinite(b["false_merge"]) else -1,
                 b["evaluable"],
                 entry["neg"]["clean"]["head"]["accepted"], entry["neg"]["clean"]["head"]["n_eval"],
                 entry["neg"]["clean"]["head"]["same_cluster_as_host"]))
        for negname in ("dirty", "clean"):
            w = entry["neg"][negname]["with_fm2"]
            if w:
                print("    └ %s + fm≤2%%: thr=%.2f cov=%.4f same=%.4f fm=%.3f%% ev=%d"
                      % (negname, w["thr"], w["coverage"], w["same"], 100 * w["false_merge"], w["evaluable"]))

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
