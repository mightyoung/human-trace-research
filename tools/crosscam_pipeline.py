#!/usr/bin/env python
"""跨画面候选生产流水线：时间门候选 → 时空一致性过滤 → 外观辅助 → 分层输出

定位（REPORT §6.3/§8"生产架构"的工程化落地）：
  输入 = 已有时间门候选（crosscam_candidates.json，来自 crosscam_analysis.py）
  流程 = 用确认对拟合 A→B 映射（训练对 LOO 防自证）→ 逐候选计算映射残差 median_r
         → 分层决策：
           AUTO   : median_r <= --sp-threshold            （时空一致性通过）
           REVIEW : 残差灰区 (阈, 阈*mult] / 双门限通过 / 残差不可算但有外观证据
           DROP   : 其余
  输出 = candidates_pipeline.json + candidates_review.csv + 控制台校验报告

用法（318×320 现有数据）:
python tools/crosscam_pipeline.py \
  --candidates output_crosscam/analysis/crosscam_candidates.json \
  --a-jsonl output_crosscam/318_detections.jsonl --a-wh 1920 1080 \
  --b-jsonl output_crosscam/320_detections.jsonl --b-wh 2560 1440 \
  --pairs-csv output_crosscam/analysis/confirmed_pairs.csv \
  --reid-emb output_crosscam/analysis/reid_emb.npz \
  --labels output_crosscam/analysis/candidates_combined.json \
  --out-dir output_crosscam/analysis
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscam_spatiotemporal import (load_tracks, norm_uv, fit_map,  # noqa: E402
                                     load_train_points, residual_series,
                                     TRAIN_PAIRS, SHIFT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-wh", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--b-wh", nargs=2, type=int, default=[2560, 1440])
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--reid-emb", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--sp-threshold", type=float, default=0.125,
                    help="时空残差自动通过阈值（本批判读的分离间隙 0.109↔0.140）")
    ap.add_argument("--auto-reid-min", type=float, default=0.5,
                    help="AUTO 层的外观下限（2026-10-12 压力测试：边界区 2 例误放 "
                         "reid=0.33/0.35，8 个确认对 reid≥0.628 → 取 0.5 判别）")
    ap.add_argument("--gray-mult", type=float, default=2.0,
                    help="灰区上限 = 阈值 × 本系数")
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ta = norm_uv(load_tracks(args.a_jsonl), *args.a_wh)
    tb = norm_uv(load_tracks(args.b_jsonl), *args.b_wh)

    train_pts = load_train_points(args.pairs_csv, set(TRAIN_PAIRS))
    bootstrap = len(TRAIN_PAIRS) < args.min_train
    coef_full = fit_map(train_pts) if train_pts else None
    coef_loo = {}
    for pr in TRAIN_PAIRS:
        pts = load_train_points(args.pairs_csv, set(TRAIN_PAIRS) - {pr})
        if pts:
            coef_loo[pr] = fit_map(pts)
    print(f"[map] 训练对 {len(TRAIN_PAIRS)} 个 / {len(train_pts)} 点"
          f"{'（bootstrap：训练不足，跳过时空过滤）' if bootstrap else ''}")

    emb = np.load(args.reid_emb) if args.reid_emb and os.path.exists(args.reid_emb) else None
    labels = {}
    if args.labels and os.path.exists(args.labels):
        for r in json.loads(Path(args.labels).read_text(encoding="utf-8")):
            labels[(r["tid_a"], r["tid_b"])] = r.get("label")

    cand = json.loads(Path(args.candidates).read_text(encoding="utf-8"))["pairs"]
    rows = []
    for p in cand:
        a, b = int(p["tid_a"]), int(p["tid_b"])
        sync = p.get("sync_r")
        covis = p.get("covis_sec")
        reid = None
        if emb is not None:
            ka, kb = f"a{a}", f"b{b}"
            if ka in emb.files and kb in emb.files:
                reid = float(np.dot(emb[ka], emb[kb]))
        mr = p90 = None
        n_pts = 0
        if not bootstrap and coef_full is not None and a in ta and b in tb:
            coef = coef_loo.get((a, b), coef_full)
            r, n = residual_series(coef, ta[a], tb[b])
            if r is not None:
                mr = float(np.median(r))
                p90 = float(np.percentile(r, 90))
                n_pts = n

        note = []
        sp_pass = mr is not None and mr <= args.sp_threshold
        if sp_pass and reid is not None and reid >= args.auto_reid_min:
            tier = "AUTO"
            note.append("sp_pass+reid_ok")
        elif sp_pass:
            tier = "REVIEW"
            note.append("sp_pass_reid_low" if reid is not None else "sp_pass_no_reid")
        else:
            if mr is not None and mr <= args.sp_threshold * args.gray_mult:
                tier = "REVIEW"
                note.append("sp_gray")
            else:
                tier = None
                if mr is not None:
                    note.append("sp_fail")
            dbl = (reid is not None and reid >= 0.6
                   and sync is not None and sync >= 0.5)
            if dbl:
                tier = "REVIEW"
                note = ["double_gate"]
            elif tier is None:
                if reid is not None and reid >= 0.6:
                    tier = "REVIEW"
                    note = ["reid_only"]
                else:
                    tier = "DROP"
                    if not note:
                        note = ["no_evidence"]

        rows.append({
            "tid_a": a, "tid_b": b,
            "sync_r": sync,
            "reid_cos": None if reid is None else round(reid, 3),
            "sp_median": None if mr is None else round(mr, 4),
            "sp_p90": None if p90 is None else round(p90, 4),
            "sp_n": n_pts,
            "covis_sec": covis,
            "tier": tier, "note": "+".join(note),
            "label": labels.get((a, b)),
        })

    # ---------- 校验（对已判读子集）----------
    lab = [r for r in rows if r["label"] in ("CONF", "REJ")]
    conf = {"AUTO": {"P": 0, "N": 0}, "REVIEW": {"P": 0, "N": 0}, "DROP": {"P": 0, "N": 0}}
    for r in lab:
        conf[r["tier"]][("P" if r["label"] == "CONF" else "N")] += 1

    print(f"\n=== 分层统计（全部 {len(rows)} 候选）===")
    for t in ("AUTO", "REVIEW", "DROP"):
        n = sum(1 for r in rows if r["tier"] == t)
        print(f"  {t:<7}: {n}")

    print(f"\n=== 校验（{len(lab)} 个已判读对：{sum(1 for r in lab if r['label']=='CONF')} CONF / "
          f"{sum(1 for r in lab if r['label']=='REJ')} REJ）===")
    for t in ("AUTO", "REVIEW", "DROP"):
        c = conf[t]
        print(f"  {t:<7}: 确认 {c['P']} / 否决 {c['N']}")

    print(f"\n=== AUTO 层候选（{sum(1 for r in rows if r['tier']=='AUTO')} 个）===")
    for r in sorted([r for r in rows if r["tier"] == "AUTO"],
                    key=lambda r: (r["sp_median"] if r["sp_median"] is not None else 9)):
        lab_s = r["label"] or "未核验"
        print(f"  A{r['tid_a']:>4}:{r['tid_b']:<4} sp={r['sp_median']} "
              f"reid={r['reid_cos']} sync={r['sync_r']} [{lab_s}]")

    print(f"\n=== REVIEW 层（{sum(1 for r in rows if r['tier']=='REVIEW')} 个，"
          f"建议下次人工核验顺序）===")
    for r in sorted([r for r in rows if r["tier"] == "REVIEW"],
                    key=lambda r: (r["sp_median"] if r["sp_median"] is not None else 9)):
        lab_s = r["label"] or "未核验"
        print(f"  A{r['tid_a']:>4}:{r['tid_b']:<4} tier={r['tier']} note={r['note']} "
              f"sp={r['sp_median']} reid={r['reid_cos']} sync={r['sync_r']} [{lab_s}]")

    out = {
        "sp_threshold": args.sp_threshold,
        "auto_reid_min": args.auto_reid_min,
        "gray_mult": args.gray_mult,
        "bootstrap": bootstrap,
        "train_pairs": TRAIN_PAIRS,
        "shift": SHIFT,
        "summary": {t: sum(1 for r in rows if r["tier"] == t)
                    for t in ("AUTO", "REVIEW", "DROP")},
        "validation": conf,
        "rows": rows,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    pj = os.path.join(args.out_dir, "candidates_pipeline.json")
    Path(pj).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {pj}")

    pc = os.path.join(args.out_dir, "candidates_review.csv")
    with open(pc, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["tid_a", "tid_b", "tier", "note", "sync_r",
                                          "reid_cos", "sp_median", "sp_p90", "sp_n",
                                          "covis_sec", "label"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["tier"] != "AUTO", r["tier"] != "REVIEW",
                                             r["sp_median"] if r["sp_median"] is not None else 9)):
            w.writerow(r)
    print(f"[out] {pc}")


if __name__ == "__main__":
    main()
