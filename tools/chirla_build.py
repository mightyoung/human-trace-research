#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CHIRLA 自动真值：把跟踪器输出的轨迹与数据集标注做逐帧 IoU 匹配，生成
下游 `eval_precision` 要的两张表 + VALIDATION §5.2 要求的 tracklet_gt.csv。

口径（VALIDATION §4.1 第 2 步，预注册）：
  - 逐采样帧把检测框与同帧标注框做 IoU ≥ --iou 的匈牙利匹配
  - 轨迹身份 = 匹配帧上的多数 id；purity = 多数 id 帧数 / 匹配帧数
  - purity < --purity-min → 轨迹内串人（对应现场 id_switch）
  - 匹配帧占比 < --matched-min → no-match（对应"看不清/非人"，进 U）
  - 仅 purity 与 matched_frac 都达标才写 person_id，否则留空（与现场 GT 同口径）

段名约定：`c<cam>_<seq>`（如 `c1_000`）——下游用 `seg.split("_")[0]` 取机位，
所以机位必须放在第一段。标注文件按 rename_map 的连字符命名。

用法：
  PY tools/chirla_build.py --det validation/chirla/det --ann datasets/CHIRLA/annotations \
     --out validation/chirla --gallery-seq 000 --query-seq 001,002
"""
import argparse
import collections
import csv
import json
import os
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)), np.float32)
    A = np.asarray(a, np.float32)[:, None, :]
    B = np.asarray(b, np.float32)[None, :, :]
    x1 = np.maximum(A[..., 0], B[..., 0])
    y1 = np.maximum(A[..., 1], B[..., 1])
    x2 = np.minimum(A[..., 2], B[..., 2])
    y2 = np.minimum(A[..., 3], B[..., 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    aa = np.clip(A[..., 2] - A[..., 0], 0, None) * np.clip(A[..., 3] - A[..., 1], 0, None)
    bb = np.clip(B[..., 2] - B[..., 0], 0, None) * np.clip(B[..., 3] - B[..., 1], 0, None)
    return inter / np.maximum(aa + bb - inter, 1e-9)


def find_ann(ann_root, seq, cam, map_rows):
    """seg -> 标注文件（按 rename_map 的 original→saved 对应）"""
    key_prefix = f"annotations/seq_{seq}/camera_{cam}_"
    for r in map_rows:
        if r["original_path"].startswith(key_prefix):
            return os.path.join(os.path.dirname(os.path.abspath(ann_root)), r["saved_path"])
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="validation/chirla/det")
    ap.add_argument("--ann", default="datasets/CHIRLA/annotations")
    ap.add_argument("--out", default="validation/chirla")
    ap.add_argument("--gallery-seq", default="000")
    ap.add_argument("--query-seq", default="001,002")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--purity-min", type=float, default=0.8)
    ap.add_argument("--matched-min", type=float, default=0.5)
    ap.add_argument("--min-frames", type=int, default=6)
    ap.add_argument("--fps", type=float, default=2.5, help="仅写入 run_info：采样帧率")
    ap.add_argument("--stride", type=int, default=12, help="仅写入 run_info：原始帧步长")
    ap.add_argument("--frame-offset", type=int, default=1,
                    help="标注帧号 = 检测帧号 + offset（标注 1 起，检测 0 起 → 1）")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "gt"), exist_ok=True)
    map_rows = list(csv.DictReader(open(
        os.path.join(os.path.dirname(os.path.abspath(a.ann)), "rename_map.csv"), encoding="utf-8")))

    gq = a.gallery_seq.split(",")
    qq = a.query_seq.split(",")
    rows, per_seg = [], {}
    for seg_dir in sorted(os.listdir(a.det)):
        # 段名两种：`c3_000`（按 seq 分图库/查询）与 `c3_000g` / `c3_000q`（同一 seq 内按时间切）
        m = re.match(r"^c(\d)_(\d{3})([gq]?)$", seg_dir)
        if not m:
            continue
        cam, seq, suf = m.group(1), m.group(2), m.group(3)
        if suf == "g":
            split = "gallery"
        elif suf == "q":
            split = "query"
        else:
            split = "gallery" if seq in gq else ("query" if seq in qq else "other")
        jp = os.path.join(a.det, seg_dir, f"{seg_dir}_detections.jsonl")
        fp = find_ann(a.ann, seq, cam, map_rows)
        if not (os.path.exists(jp) and fp and os.path.exists(fp)):
            print(f"[skip] {seg_dir}: jsonl={os.path.exists(jp)} ann={fp}")
            continue
        ann = {int(k): v for k, v in json.load(open(fp, encoding="utf-8")).items()}
        frames = [json.loads(l) for l in open(jp, encoding="utf-8")]
        # track -> [(frame, box)]
        agg_n = agg_m = 0
        obs = collections.defaultdict(list)
        for r in frames:
            for t, b in zip(r.get("track_ids") or [], r.get("boxes") or []):
                if t is None:
                    continue
                obs[int(t)].append((int(r["frame"]), b))
        n_tr = 0
        for tid, lst in sorted(obs.items()):
            lst.sort()
            if len(lst) < a.min_frames:
                continue
            n_tr += 1
            votes = collections.Counter()
            n_match = 0
            for fr, box in lst:
                objs = ann.get(fr + a.frame_offset)
                if not objs:
                    continue
                ids = [str(o["id"]) for o in objs]
                boxes = [o["BboxP"] for o in objs]
                M = iou_matrix([box], boxes)[0]
                j = int(M.argmax())
                if M[j] >= a.iou:
                    votes[ids[j]] += 1
                    n_match += 1
            n = len(lst)
            matched_frac = n_match / n
            if votes:
                pid, ntop = votes.most_common(1)[0]
                purity = ntop / max(1, n_match)
            else:
                pid, purity = "", 0.0
            if matched_frac < a.matched_min:
                ev, why = False, "no-match"
            elif purity < a.purity_min:
                ev, why = False, "id_switch"
            else:
                ev, why = True, ""
            agg_n += n
            agg_m += n_match
            rows.append(dict(track_id=f"{seg_dir}#{tid}", seg=seg_dir, split=split,
                             cam=f"c{cam}", n_frames=n, pid=pid if ev else "",
                             raw_pid=pid, purity=round(purity, 3),
                             matched_frac=round(matched_frac, 3),
                             evaluable=int(ev), exclude_reason=why, box_type="body"))
        per_seg[seg_dir] = n_tr
        print(f"[{seg_dir}] 轨迹 {n_tr} 条｜框匹配率 {agg_m/max(1,agg_n):.3f} "
              f"({agg_m}/{agg_n})（段内 <{a.min_frames} 帧的轨迹已丢）")

    with open(os.path.join(out, "gt", "tracklet_gt.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gal = [r for r in rows if r["split"] == "gallery" and r["evaluable"]]
    qry = [r for r in rows if r["split"] == "query" and r["evaluable"]]
    with open(os.path.join(out, "gt", "gt_gallery_members_FILLED.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["track_id", "seg", "cam", "tid", "n_frames",
                                           "box_type", "person_id", "note", "prompt_jpg"])
        w.writeheader()
        for r in gal:
            w.writerow(dict(track_id=r["track_id"], seg=r["seg"], cam=r["cam"],
                            tid=r["track_id"].split("#")[1], n_frames=r["n_frames"],
                            box_type="body", person_id=r["pid"],
                            note=f"purity={r['purity']}", prompt_jpg=""))
    with open(os.path.join(out, "gt", "gt_query_gt_FILLED.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_id", "person_id"])
        w.writeheader()
        for r in qry:
            w.writerow(dict(query_id=r["track_id"], person_id=r["pid"]))

    gal_ids = sorted({r["pid"] for r in gal if r["pid"]})
    qry_ids = sorted({r["pid"] for r in qry if r["pid"]})
    print(f"\n[gt] 图库可用 {len(gal)} 条 / {len(gal_ids)} 身份 {gal_ids}")
    print(f"     查询可用 {len(qry)} 条 / {len(qry_ids)} 身份 {qry_ids}")
    print(f"     图库外查询身份 {sorted(set(qry_ids) - set(gal_ids))}")
    print(f"     被排除: no-match {sum(1 for r in rows if r['exclude_reason']=='no-match')}"
          f"｜id_switch {sum(1 for r in rows if r['exclude_reason']=='id_switch')}")

    run_info = dict(
        dataset="CHIRLA",
        subset=dict(gallery=sorted({r["seg"] for r in rows if r["split"] == "gallery"}),
                    query=sorted({r["seg"] for r in rows if r["split"] == "query"})),
        counts=dict(gallery_tracks=len(gal), gallery_labeled=len(gal),
                    query_tracks=len(qry), query_identities=len(qry_ids),
                    answerable=sum(1 for r in qry if r["pid"] in set(gal_ids))),
        gt_match=dict(iou_min=a.iou, purity_switch=a.purity_min,
                      matched_min=a.matched_min, frame_offset=a.frame_offset,
                      no_match=sum(1 for r in rows if r["exclude_reason"] == "no-match"),
                      id_switch=sum(1 for r in rows if r["exclude_reason"] == "id_switch")),
        params=dict(fps=a.fps, stride=a.stride, imgsz=1280, thr_in=0.15, thr_out=0.25,
                    same_min=0.90, min_frames=a.min_frames),
        deviations=[
            "CHIRLA 30 FPS，按每 12 帧取 1 → 2.5 FPS（现场 25 FPS 每 10 帧取 1）",
            "seq_001/002 的 camera_5/6/7 标注为空，只跑 camera_1..4（查询侧实际 4 机位）",
            "特征口径沿用历史 BGR，与冻结基线一致，未开 --rgb",
            "轨迹真值为自动 IoU 匹配（非人工标注），purity/matched_frac 阈值见 gt_match",
        ],
    )
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
