#!/usr/bin/env python
"""全量轨迹清洗：用"镜像运动学检验"扫描 316/323 所有轨迹对 -> 标记倒影 -> 去镜校正计数。

原理：倒影轨迹与其源轨迹满足运动学镜像关系（关于触地线）：
  |du| < 0.06（水平对齐）
  pred_err = |v_R − (2·v_S_bot − v_S_center)| / h_S <= 0.10（镜像位置预测残差，单位=人体高）
  0.35 <= (v_R_c − v_S_bot)/h_S <= 0.60（倒影中心位于源脚下方半身高处）
验证（mirror_kinematics）：29 真对通过 28（96.6%）；408 自然共现对仅 7 通过（且均几何近乎完美）。

本工具对每个镜头全量轨迹做同样检验（轨道对=共现≥4 帧），输出：
  - 每镜头的 倒影轨迹清单（含源、pred_err）
  - 去镜校正计数（原始轨迹数 -> 清洗后）
  - 清洗版 jsonl（删除倒影轨迹的框，供后续跨机流程复用）

用法：
  python tools/track_cleaning.py \
    --a-jsonl output_crosscam/316_detections.jsonl \
    --b-jsonl output_crosscam/323_detections.jsonl \
    --out-dir output_crosscam/analysis/track_cleaning
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from mirror_kinematics import load_tracks, pair_metrics

DIMS = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
        "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
RULE = dict(du_max=0.06, perr_max=0.10, off_lo=0.35, off_hi=0.60, n_min=4)


def rule_pass(m):
    if m is None or m.get("pred_err") is None:
        return False
    if m["n_common"] < RULE["n_min"]:
        return False
    return (abs(m["du_med"]) < RULE["du_max"] and m["pred_err"] <= RULE["perr_max"]
            and RULE["off_lo"] <= m["offset_ratio"] <= RULE["off_hi"])


def cooccur(tracks, n_min=4):
    """{(a,b): n} 共现对计数（a<b）"""
    fmap = {}
    for tid, lst in tracks.items():
        if len(lst) < 4:
            continue
        for f, b in lst:
            fmap.setdefault(f, []).append(tid)
    cnt = {}
    for f, tids in fmap.items():
        tids = sorted(set(tids))
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                k = (tids[i], tids[j])
                cnt[k] = cnt.get(k, 0) + 1
    return {k: v for k, v in cnt.items() if v >= n_min}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True,
                    help='格式 "tag=path,tag=path"（如 316=output_crosscam/316_detections.jsonl）')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="", help="逗号分隔的 tag 子集（默认全部）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = []
    for part in args.jobs.split(","):
        tag, path = part.split("=", 1)
        jobs.append((tag.strip(), path.strip()))
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        jobs = [j for j in jobs if j[0] in keep]

    # 已知真对（验证用，仅 316/323 有标注）
    known = {}
    kcsv = Path("output_crosscam/analysis/spread/depth_filter/shortlist_audited.csv")
    if kcsv.exists():
        import csv
        with open(kcsv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                for side in ("disp", "app"):
                    if row.get(f"{side}_sib") == "1":
                        tag, tid = row[side].split("#")
                        known.setdefault(tag, set()).add(int(tid))

    summary = {}
    for tag, jpath in jobs:
        tracks = load_tracks(jpath)
        W, H = DIMS[tag]
        cnt = cooccur(tracks)
        # 每个轨迹找最佳"源"（若作为倒影被某源完美解释）
        flags = {}   # tid -> {"source": s, "perr": p, "off": o, "n": n}
        for (a, b), n in cnt.items():
            for r, s in ((a, b), (b, a)):
                m = pair_metrics(tracks, W, H, r, s)
                if not rule_pass(m):
                    continue
                cur = flags.get(r)
                if cur is None or m["pred_err"] < cur["perr"]:
                    flags[r] = {"source": s, "perr": round(m["pred_err"], 4),
                                "off": round(m["offset_ratio"], 3), "n": m["n_common"]}
        all_tids = sorted([t for t, l in tracks.items() if len(l) >= 4])
        mirror_tids = sorted(flags.keys())
        # 已知验证
        k_hit = len(known.get(tag, set()) & set(mirror_tids))
        k_tot = len(known.get(tag, set()))
        print(f"\n######## {tag} ({jpath}) ########")
        print(f"  轨迹总数(n>=4): {len(all_tids)}   判为倒影: {len(mirror_tids)}  "
              f"-> 去镜校正计数: {len(all_tids) - len(mirror_tids)}")
        print(f"  已知倒影验证: {k_hit}/{k_tot} 被捕获")
        hi = [t for t in mirror_tids if flags[t]["perr"] <= 0.05]
        print(f"  其中高置信(perr<=0.05): {len(hi)}")
        print(f"  倒影清单（tid -> 源, perr, off, n）：")
        for t in mirror_tids:
            fl = flags[t]
            print(f"    #{t:<4} -> 源 #{fl['source']:<4} perr={fl['perr']:.3f} off={fl['off']:.2f} n={fl['n']}")

        # 清洗版 jsonl（逐框契约：boxes/track_ids/scores 同长度；count == len(boxes)）
        clean_path = os.path.join(args.out_dir, f"{tag}_cleaned.jsonl")
        removed_boxes = 0
        with open(clean_path, "w", encoding="utf-8") as fo, open(jpath, encoding="utf-8") as fi:
            for line in fi:
                r = json.loads(line)
                tids = r.get("track_ids", []) or []
                boxes = r.get("boxes", []) or []
                scores = r.get("scores", []) or []
                keep_t, keep_b, keep_s = [], [], []
                for i, b in enumerate(boxes):
                    tid = tids[i] if i < len(tids) else -1
                    if tid is not None and int(tid) in flags:
                        removed_boxes += 1
                        continue
                    keep_b.append(b)
                    keep_t.append(tid)
                    if i < len(scores):
                        keep_s.append(scores[i])
                r["track_ids"] = keep_t
                r["boxes"] = keep_b
                r["scores"] = keep_s
                r["count"] = len(keep_b)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [out] {clean_path}  (删除 {removed_boxes} 个框)")

        summary[tag] = {
            "tracks_total": len(all_tids), "mirror_tracks": len(mirror_tids),
            "cleaned_count": len(all_tids) - len(mirror_tids),
            "known_hit": f"{k_hit}/{k_tot}",
            "high_conf": len(hi),
            "flags": {str(t): flags[t] for t in mirror_tids},
        }

    out = {"rule": RULE, "summary": summary}
    Path(os.path.join(args.out_dir, "track_cleaning.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {args.out_dir}/track_cleaning.json")

    # 简单汇总图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        cams = [t for t, _ in jobs]
        raw = [summary[c]["tracks_total"] for c in cams]
        mir = [summary[c]["mirror_tracks"] for c in cams]
        fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=120)
        x = np.arange(len(cams))
        ax.bar(x - 0.2, raw, 0.4, label="raw tracks (n>=4)", color="#888")
        ax.bar(x + 0.2, [summary[c]["cleaned_count"] for c in cams], 0.4,
               label="after mirror cleanup", color="#2ca02c")
        for i, c in enumerate(cams):
            ax.text(i - 0.2, raw[i] + 1, str(raw[i]), ha="center", fontsize=10)
            ax.text(i + 0.2, summary[c]["cleaned_count"] + 1, str(summary[c]["cleaned_count"]),
                    ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}  (mirror removed: {m})" for c, m in zip(cams, mir)])
        ax.set_ylabel("unique tracks (10-min)")
        ax.set_title("Track cleanup by kinematic mirror test")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_track_cleaning.png"), facecolor="white")
        print(f"[out] {args.out_dir}/fig_track_cleaning.png")
    except Exception as e:  # noqa: BLE001
        print("[warn] fig failed:", str(e)[:160])


if __name__ == "__main__":
    main()
