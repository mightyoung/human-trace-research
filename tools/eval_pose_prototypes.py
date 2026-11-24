#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两个提案的实测：(1) 多原型/姿态感知原型；(2) 姿态变化能否解释 a/b 不一致。

背景
----
提案 A：不同镜头下的同一 ReID「默认归到一起」——即跨机位合并更激进。
        生产已经是机位内 0.15 + 跨机位 complete-linkage 0.25，本节顺带扫 thr_out。
提案 B：一个人坐下前后的向量差别大，应把「站着的那段」和「坐下的那段」归到一起。

本脚本不依赖新的检测/编码，只用现有框几何（h/w 高宽比判站/坐）与既有嵌入。

实验
----
E-A  a/b 两片的姿态是否真的不同？姿态差大的轨迹，same 是否更低？（诊断，不改系统）
E-B  每身份多原型：max over members / top-3 均值 / 2-medoid / 姿态双原型（站+坐）
     在**训练集**上按姿态切原型，测试集完全不参与构建。
E-C  跨机位合并阈值 thr_out 扫描（0.20 / 0.25 / 0.30 / 0.35）

口径：生产配置（机位感知聚类、thr_in 0.15）、self-cal = same>=0.90 且 evaluable>=100 取覆盖最大；
      负对同时报 dirty（1120 对）与 clean（1100 对，剔除头框↔宿主假负对）。

用法：python tools/eval_pose_prototypes.py
输出：selector_osnet_1012/pose_proto.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
import cluster_camera_aware as CA  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/pose_proto.json")
GRID = np.round(np.arange(-2.0, 8.01, 0.05), 3)

STAND_HW = 2.2      # h/w >= 2.2 视为站姿（站立 2.5-3.5，坐姿 1.3-1.8）
SIT_HW = 2.0        # h/w <= 2.0 视为坐姿


def box_pose_map(segs):
    """每条轨迹的 h/w 序列 → 姿态统计。返回 {(seg,tid): dict}"""
    out = {}
    for seg in segs:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        if not os.path.exists(p):
            continue
        per = defaultdict(list)
        for line in open(p, encoding="utf-8"):
            o = json.loads(line)
            for tid, b in zip(o["track_ids"], o["boxes"]):
                w, h = b[2] - b[0], b[3] - b[1]
                if w > 2 and h > 2:
                    per[int(tid)].append((o["frame"], h / w))
        for tid, v in per.items():
            v.sort()
            hw = np.array([x[1] for x in v])
            half = len(hw) // 2
            a, b = hw[:max(half, 1)], hw[max(half, 1):] if len(hw) > 1 else hw
            out[(seg, tid)] = dict(
                n=len(hw), hw_all=float(np.median(hw)),
                hw_a=float(np.median(a)),
                hw_b=float(np.median(b)) if len(b) else float(np.median(a)),
                hw_std=float(hw.std()),
            )
    return out


def posture(hw):
    if hw >= STAND_HW:
        return "stand"
    if hw <= SIT_HW:
        return "sit"
    return "mid"


def build_protos(V, lab, cams, kept, kind, mu_g=None, pose=None, k=2):
    """按 kind 构造多原型。返回 protos, pcl, meta。"""
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        idx = np.where(lab == c)[0]
        if idx.size == 0:
            continue
        if kind == "mean":
            protos.append(_norm(V[idx].mean(0, keepdims=True))[0])
            pcl.append(ci)
        elif kind == "knn":
            for i in idx:
                protos.append(V[i])
                pcl.append(ci)
        elif kind == "pose":
            groups = defaultdict(list)
            for i in idx:
                p = pose.get(i)
                groups["sit" if p == "sit" else "stand"].append(i)
            for g, ii in groups.items():
                if len(ii) >= 2:
                    protos.append(_norm(V[ii].mean(0, keepdims=True))[0])
                    pcl.append(ci)
            if not groups:
                protos.append(_norm(V[idx].mean(0, keepdims=True))[0])
                pcl.append(ci)
        elif kind == "medoid":
            sub = V[idx]
            Sim = sub @ sub.T
            centers = [int(np.argmax(Sim.sum(1)))]
            for _ in range(k - 1):
                d = 1.0 - Sim[:, centers].max(1)
                centers.append(int(np.argmax(d)))
            for j in centers:
                protos.append(_norm(sub[j:j + 1])[0])
                pcl.append(ci)
    P = np.asarray(protos, np.float32)
    if mu_g is not None:
        P = _norm(P - mu_g)
    return P, np.asarray(pcl)


def score_with(X, protos, pcl, K, mu, calib, topk=None):
    Xa = _norm(X - mu) if mu is not None else _norm(X)
    if topk is None:
        out = S.cluster_scores(Xa, protos, pcl, K)
        if calib is not None:
            out = (out - calib[0]) / calib[1]
        return out
    Sfull = Xa @ protos.T
    out = np.full((len(Xa), K), -np.inf, np.float32)
    for k in range(K):
        cols = np.where(pcl == k)[0]
        if cols.size:
            v = np.sort(Sfull[:, cols], axis=1)[:, ::-1][:, :min(topk, cols.size)]
            out[:, k] = v.mean(1)
    if calib is not None:
        out = (out - calib[0]) / calib[1]
    return out


def eval_variant(protos, pcl, K, Vtr, lab, kept, Qa, Qb, kte, cooc, topk=None):
    mu_g = Vtr.mean(0)
    m_hat, s_hat, _ = S.fit_ident_stats(score_with(Vtr, protos, pcl, K, mu_g, None, topk),
                                        lab, kept)
    mu_q = _norm((Qa + Qb) / 2.0).mean(0)
    Sa = score_with(Qa, protos, pcl, K, mu_q, (m_hat, s_hat), topk)
    Sb = score_with(Qb, protos, pcl, K, mu_q, (m_hat, s_hat), topk)
    rows = S.sweep(Sa, Sb, kte, cooc, GRID)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    return best


def main():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Qa, Qb = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    cams_tr = np.array([k[0].split("_")[0] for k in ktr])

    keys_ok = set(kte)
    dirty = []
    for seg in S.TEST_SEGS:
        for a, b in cooc_pairs(os.path.join(DET, seg, f"{seg}_detections.jsonl"), keys_ok, seg, 5):
            dirty.append(((seg, a), (seg, b)))

    pose_tr = box_pose_map(S.TRAIN_SEGS)
    pose_te = box_pose_map(S.TEST_SEGS)
    Ki = {k: i for i, k in enumerate(ktr)}
    pose_idx_tr = {Ki[k]: posture(v["hw_all"]) for k, v in pose_tr.items() if k in Ki}

    res = {"config": dict(thr_in=0.15, same_min=0.90, stand_hw=STAND_HW, sit_hw=SIT_HW),
           "E_A": {}, "E_B": {}, "E_C": {}}

    # ---------- E-A：姿态变化能否解释 same ----------
    lab, kept = CA.camera_aware_labels(Vtr, cams_tr, 0.15, 0.25, S.MIN_CLUSTER)
    K = len(kept)
    protos, pcl = build_protos(Vtr, lab, cams_tr, kept, "mean", mu_g=Vtr.mean(0))
    mu_g = Vtr.mean(0)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Qa + Qb) / 2.0).mean(0)
    Sa = S.score_rows(Qa, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Qb, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    thr = 1.85
    acc = (Sa.max(1) >= thr) & (Sb.max(1) >= thr)
    agree = Sa.argmax(1) == Sb.argmax(1)
    d = []
    for i, k in enumerate(kte):
        if not acc[i]:
            continue
        v = pose_te.get(k)
        if not v:
            continue
        pa, pb = posture(v["hw_a"]), posture(v["hw_b"])
        tag = "同站" if (pa == "stand" and pb == "stand") else               "同坐" if (pa == "sit" and pb == "sit") else               "站↔坐" if {pa, pb} == {"stand", "sit"} else "含中间"
        d.append((tag, float(agree[i])))
    grp = {}
    for tag, ag in d:
        grp.setdefault(tag, []).append(ag)
    res["E_A"] = dict(
        n_accepted=int(acc.sum()),
        same_overall=float(agree[acc].mean()),
        by_group={k: dict(n=len(v), same=float(np.mean(v)) if v else None)
                  for k, v in sorted(grp.items(), key=lambda x: -len(x[1]))},
    )
    print("[E-A] 接受 %d 条，整体 same=%.3f；按 a/b 姿态转移分组：%s"
          % (acc.sum(), agree[acc].mean(),
             {k: (len(v), round(float(np.mean(v)), 3) if v else None) for k, v in grp.items()}))

    # ---------- E-B：多原型 ----------
    # clean 负对：剔除头框↔宿主
    from render_head_track_audit import find_head_tracks
    te_head = {}
    for seg in S.TEST_SEGS:
        for t, v in find_head_tracks(seg).items():
            te_head[(seg, t)] = v[0]
    clean = [(u, v) for u, v in dirty
             if not ((u in te_head and te_head[u] == v[1]) or (v in te_head and te_head[v] == u[1]))]

    for kind, topk in (("mean", None), ("knn", None), ("knn", 3), ("medoid", None), ("pose", None)):
        pp, pc = build_protos(Vtr, lab, cams_tr, kept, kind, mu_g=Vtr.mean(0), pose=pose_idx_tr, k=2)
        tag = f"{kind}" + (f"_top{topk}" if topk else "") + (f"_P{len(set(pc.tolist()))}" if False else "")
        b_d = eval_variant(pp, pc, K, Vtr, lab, kept, Qa, Qb, kte, dirty, topk)
        b_c = eval_variant(pp, pc, K, Vtr, lab, kept, Qa, Qb, kte, clean, topk)
        if not b_d:
            print(f"[E-B] {tag}: 无自洽点"); continue
        res["E_B"][tag] = dict(n_proto=int(len(pp)), dirty=b_d, clean=b_c)
        print("[E-B] %-8s 原型 %4d | dirty cov=%.4f same=%.4f fm=%.3f%% | clean cov=%.4f same=%.4f fm=%.3f%%"
              % (tag, len(pp), b_d["coverage"], b_d["same"], 100 * b_d["false_merge"],
                 b_c["coverage"], b_c["same"], 100 * b_c["false_merge"]))

    # ---------- E-C：跨机位合并阈值 ----------
    for thr_out in (0.20, 0.25, 0.30, 0.35):
        lab2, kept2 = CA.camera_aware_labels(Vtr, cams_tr, 0.15, thr_out, S.MIN_CLUSTER)
        K2 = len(kept2)
        p2, pc2 = build_protos(Vtr, lab2, cams_tr, kept2, "mean", mu_g=Vtr.mean(0))
        b = eval_variant(p2, pc2, K2, Vtr, lab2, kept2, Qa, Qb, kte, clean)
        if b:
            res["E_C"][f"thr_out={thr_out}"] = dict(K=K2, **b)
            print("[E-C] thr_out=%.2f K=%3d | cov=%.4f same=%.4f fm=%.3f%%"
                  % (thr_out, K2, b["coverage"], b["same"], 100 * b["false_merge"]))

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[out]", OUT)


if __name__ == "__main__":
    main()
