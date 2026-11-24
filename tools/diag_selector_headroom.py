#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断：判定策略（selector）还有多少可改进空间。

目的：回答"要不要为选择器上学习型/RL 策略"，先量化三件事：
  1) 固定分数下，策略可动的空间有多大（阈值前沿的陡峭程度）；
  2) 被拒轨迹里有多少是"差一点"（决策噪声）vs"差很远"（图库没有/表征失败）；
  3) a/b 冲突样本是"两个簇分数接近"（决策问题）还是"一半完全没分"（表征问题）。

口径与 eval_det_switch.run() 完全一致（生产配置 yolo8dd，same>=0.90，min_eval 100）。
只读，不改任何既有产物；输出 JSON 到 selector_osnet_1012/。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/headroom_yolo8dd.json")

THR = 2.5          # deploy_eval.json 的 self_cal 工作点
NEAR = 0.5         # z 单位：与门槛距离 <= NEAR 视为"差一点"
SAME_MIN = 0.90


def build():
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(
        S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    return ktr, kte, Sa, Sb, K


def top2_margin(Srow):
    """top1 - top2 的分差；分差小 = 两个候选簇难分。"""
    s = np.sort(Srow)[::-1]
    return float(s[0] - s[1])


def main():
    ktr, kte, Sa, Sb, K = build()
    ia, ib = Sa.argmax(1), Sb.argmax(1)
    sa, sb = Sa.max(1), Sb.max(1)
    n = len(kte)
    acca, accb = sa >= THR, sb >= THR
    both = acca & accb

    cats = {"both_same": [], "both_conflict": [], "one_side": [],
            "rej_near": [], "rej_far": []}
    rows = []
    for i in range(n):
        best = max(sa[i], sb[i])
        if acca[i] and accb[i]:
            cat = "both_same" if ia[i] == ib[i] else "both_conflict"
        elif acca[i] or accb[i]:
            cat = "one_side"
        else:
            cat = "rej_near" if best >= THR - NEAR else "rej_far"
        cats[cat].append(i)
        rows.append({"key": list(kte[i]), "sa": float(sa[i]), "sb": float(sb[i]),
                     "ia": int(ia[i]), "ib": int(ib[i]), "cat": cat,
                     "margin_a": top2_margin(Sa[i]), "margin_b": top2_margin(Sb[i])})

    summary = {"n_test": n, "thr": THR, "near_band": NEAR, "K": K}
    for c, idx in cats.items():
        if not idx:
            summary[c] = {"n": 0}
            continue
        summary[c] = {
            "n": len(idx),
            "share": round(len(idx) / n, 4),
            "best_score_median": round(float(np.median([max(sa[i], sb[i]) for i in idx])), 3),
            "best_score_p10": round(float(np.percentile([max(sa[i], sb[i]) for i in idx], 10)), 3),
            "margin_median": round(float(np.median([min(rows[i]["margin_a"], rows[i]["margin_b"])
                                                    for i in idx])), 3),
        }

    # 1) 阈值前沿：只改决策规则时，覆盖换稳定率的代价
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    cooc = []
    keys_ok = set(kte)
    for s in S.TEST_SEGS:
        p = os.path.join(DET, s, f"{s}_detections.jsonl")
        cooc += [((s, u), (s, v)) for u, v in cooc_pairs(p, keys_ok, s, 5)]
    sweep = S.sweep(Sa, Sb, kte, cooc, grid)
    frontier = []
    for tgt in (0.55, 0.568, 0.60, 0.62, 0.65, 0.70, 0.75):
        cand = [r for r in sweep if r["coverage"] >= tgt and r["evaluable"] >= 100
                and np.isfinite(r["same"])]
        if not cand:
            frontier.append({"target": tgt, "reachable": False})
            continue
        r = min(cand, key=lambda x: x["coverage"])
        frontier.append({"target": tgt, "reachable": True, "thr": r["thr"],
                         "coverage": round(r["coverage"], 4), "same": round(r["same"], 4),
                         "fm": round(r["false_merge"], 4)})
    summary["frontier"] = frontier

    # 2) 只靠 decision rule 的理论上限：把 rej_near 全部"正确"捞回（乐观上界，需 oracle）
    base = len(cats["both_same"]) + len(cats["both_conflict"]) + len(cats["one_side"])
    summary["oracle_near_miss_lift"] = round(len(cats["rej_near"]) / n, 4)
    summary["oracle_near_miss_coverage"] = round((base + len(cats["rej_near"])) / n, 4)

    # 3) 按机位拆分（哪台机器贡献了主要损失）
    by_cam = {}
    for i, r in enumerate(rows):
        cam = kte[i][0].split("_")[0]
        d = by_cam.setdefault(cam, {})
        d[r["cat"]] = d.get(r["cat"], 0) + 1
    summary["by_cam"] = by_cam

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
