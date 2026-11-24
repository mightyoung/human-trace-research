#!/usr/bin/env python
"""本地链路 vs 远程历史产物 对账工具。

两级对账：
  detect：逐帧比较检测产物（raw/deduped/count、boxes、scores、track_ids），
          并做"阈值边缘框"分析（解释 ±1 级差异的来源）。
  embed ：轨迹嵌入 npz 的 key 覆盖与余弦一致性。

用法:
  python tools/local_check.py detect \
      --local-dir output_crosscam_local/detect --remote-dir output_crosscam \
      --cams 305,316,318,320,323,343 [--json out.json]

  python tools/local_check.py embed \
      --pairs 305:output_crosscam_local/embed/reid_emb_305_d3b_local.npz:output_crosscam/analysis/relay/remote_emb/reid_emb_305_d3b.npz \
      ...
  或  --local-dir/--remote-dir 自动匹配 reid_emb_{cam}_d3b*.npz
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def _kid(k):
    """从 npz key 提取 track id（尾部数字，兼容 v305_12 / a150 / b29）。"""
    m = re.search(r"(\d+)$", k)
    return int(m.group(1)) if m else -1


# ---------------- 通用 ----------------
def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def greedy_match(A, B):
    """A/B: list of boxes；返回贪心配对的 IoU 列表。"""
    ious = []
    used = set()
    for a in A:
        best_j, best_v = -1, -1.0
        for j, b in enumerate(B):
            if j in used:
                continue
            v = box_iou(a, b)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            ious.append(best_v)
    return ious


# ---------------- detect 对账 ----------------
def check_detect_cam(local_dir, remote_dir, cam, n_frames_limit=0):
    lp = Path(local_dir) / f"{cam}_detections.jsonl"
    rp = Path(remote_dir) / f"{cam}_detections.jsonl"
    if not lp.exists():
        return {"cam": cam, "status": "local_missing"}
    if not rp.exists():
        return {"cam": cam, "status": "remote_missing"}
    loc = load_jsonl(lp)
    rem = load_jsonl(rp)
    if n_frames_limit:
        loc, rem = loc[:n_frames_limit], rem[:n_frames_limit]
    n = min(len(loc), len(rem))
    res = {"cam": cam, "status": "ok", "local_rows": len(loc), "remote_rows": len(rem)}

    # 帧号对齐检查
    frame_mismatch = sum(1 for a, b in zip(loc[:n], rem[:n]) if a["frame"] != b["frame"])
    res["frame_mismatch_rows"] = frame_mismatch

    for tag in ("raw_count", "deduped_count", "count"):
        eq = sum(1 for a, b in zip(loc[:n], rem[:n]) if a[tag] == b[tag])
        d1 = sum(1 for a, b in zip(loc[:n], rem[:n]) if abs(a[tag] - b[tag]) <= 1)
        mae = float(np.mean([abs(a[tag] - b[tag]) for a, b in zip(loc[:n], rem[:n])]))
        res[f"{tag}_exact"] = round(eq / n, 4)
        res[f"{tag}_within1"] = round(d1 / n, 4)
        res[f"{tag}_mae"] = round(mae, 4)

    # track_ids 完全一致率（帧级）
    tid_eq = sum(1 for a, b in zip(loc[:n], rem[:n]) if a["track_ids"] == b["track_ids"])
    res["track_ids_frame_exact"] = round(tid_eq / n, 4)
    res["track_ids_set_overlap"] = _tid_overlap(loc[:n], rem[:n])

    # boxes IoU（数量相等的帧做配对；数量不等帧计数）
    ious, mismatch_frames, score_deltas = [], [], []
    for i, (a, b) in enumerate(zip(loc[:n], rem[:n])):
        A, B = a["boxes"], b["boxes"]
        if len(A) == len(B):
            if A:
                ious.extend(greedy_match(A, B))
                if len(a["scores"]) == len(b["scores"]) and a["scores"]:
                    score_deltas.extend(np.abs(np.sort(a["scores"]) -
                                               np.sort(b["scores"])).tolist())
        else:
            mismatch_frames.append({"frame": a["frame"], "local": len(A), "remote": len(B)})
    ious = np.asarray(ious) if ious else np.zeros(0)
    res["box_pairs_total"] = int(len(ious))
    if len(ious):
        res["iou_mean"] = round(float(ious.mean()), 4)
        res["iou_p10"] = round(float(np.percentile(ious, 10)), 4)
        res["iou_min"] = round(float(ious.min()), 4)
        res["iou_ge_0.9"] = round(float(np.mean(ious >= 0.9)), 4)
        res["iou_ge_0.7"] = round(float(np.mean(ious >= 0.7)), 4)
        # 直方图（供可视化）：<0.5 / 0.5-0.7 / 0.7-0.9 / 0.9-0.95 / 0.95-0.99 / >=0.99
        edges = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0001]
        hist, _ = np.histogram(ious, bins=edges)
        res["iou_hist"] = {"edges": edges, "counts": hist.tolist()}
    # 逐帧 count 序列（供可视化；n 最大 3000，体积可控）
    res["count_series"] = {
        "local": [r["count"] for r in loc[:n]],
        "remote": [r["count"] for r in rem[:n]],
    }
    sd = np.asarray(score_deltas) if score_deltas else np.zeros(0)
    if len(sd):
        res["score_absd_mean"] = round(float(sd.mean()), 5)
        res["score_absd_p90"] = round(float(np.percentile(sd, 90)), 5)
        res["score_absd_max"] = round(float(sd.max()), 5)
    res["count_mismatch_frames"] = len(mismatch_frames)
    res["count_mismatch_examples"] = mismatch_frames[:10]

    # summary 对比
    ls = Path(local_dir) / f"{cam}_summary.json"
    rs = Path(remote_dir) / f"{cam}_summary.json"
    if ls.exists() and rs.exists():
        lsum = json.loads(ls.read_text(encoding="utf-8"))
        rsum = json.loads(rs.read_text(encoding="utf-8"))
        res["summary"] = {
            k: {"local": lsum.get(k), "remote": rsum.get(k)}
            for k in ("frames_processed", "count_mean", "count_max", "count_min",
                      "unique_track_ids", "deduped_count_mean", "raw_count_mean",
                      "boxes_removed_pct")
        }

    # 边缘框分析（本地 raw jsonl）：统计 [0.27,0.33] 区间的框数量（用本地 raw）
    lraw = Path(local_dir) / f"{cam}_raw.jsonl"
    if lraw.exists():
        edge_lo, edge_hi = 0.27, 0.33
        n_edge, n_all = 0, 0
        for r in load_jsonl(lraw):
            for s in r.get("scores", []):
                n_all += 1
                if edge_lo <= s <= edge_hi:
                    n_edge += 1
        res["edge_boxes"] = {"lo": edge_lo, "hi": edge_hi, "count": n_edge,
                             "total": n_all,
                             "frac": round(n_edge / n_all, 4) if n_all else None}
    return res


def _tid_overlap(loc, rem):
    """track 集合重叠（Jaccard）。"""
    tl = set(); tr = set()
    for r in loc:
        tl.update(r["track_ids"])
    for r in rem:
        tr.update(r["track_ids"])
    if not tl and not tr:
        return 1.0
    return round(len(tl & tr) / len(tl | tr), 4)


def cmd_detect(args):
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    out = {}
    for cam in cams:
        print(f"===== detect check: {cam} =====", flush=True)
        r = check_detect_cam(args.local_dir, args.remote_dir, cam,
                             n_frames_limit=args.limit)
        out[cam] = r
        if r.get("status") == "ok":
            print(f"  rows local/remote: {r['local_rows']}/{r['remote_rows']}  "
                  f"frame_mismatch={r['frame_mismatch_rows']}")
            for tag in ("raw_count", "deduped_count", "count"):
                print(f"  {tag:<14} exact={r[f'{tag}_exact']:.3f} "
                      f"within1={r[f'{tag}_within1']:.3f} MAE={r[f'{tag}_mae']}")
            print(f"  track_ids frame-exact={r['track_ids_frame_exact']:.3f} "
                  f"set-Jaccard={r['track_ids_set_overlap']}")
            if "iou_mean" in r:
                print(f"  boxes n={r['box_pairs_total']} IoU mean={r['iou_mean']} "
                      f"p10={r['iou_p10']} min={r['iou_min']} "
                      f">=0.9:{r['iou_ge_0.9']:.3f} >=0.7:{r['iou_ge_0.7']:.3f}")
            if "score_absd_mean" in r:
                print(f"  scores |d| mean={r['score_absd_mean']} "
                      f"p90={r['score_absd_p90']} max={r['score_absd_max']}")
            print(f"  count mismatch frames={r['count_mismatch_frames']}")
            if "summary" in r:
                print(f"  summary: {json.dumps(r['summary'], ensure_ascii=False)}")
        else:
            print(f"  {r['status']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n[json] {args.json}")


# ---------------- embed 对账 ----------------
def _load_tracks_from_jsonl(p):
    """tid -> set(frames)"""
    t = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for i, tid in enumerate(r.get("track_ids", []) or []):
                if tid is not None and tid >= 0:
                    t.setdefault(int(tid), set()).add(int(r["frame"]))
    return t


def check_embed_content(local_npz, remote_npz, local_jsonl, remote_jsonl,
                        min_overlap=0.2):
    """内容级嵌入对账：track id 因浮点边缘而错位时，按"帧重叠 Jaccard"
    为每个本地 track 找远程对应 track，再比较嵌入 cos。"""
    loc = np.load(local_npz)
    rem = np.load(remote_npz)
    tl = _load_tracks_from_jsonl(local_jsonl)
    tr = _load_tracks_from_jsonl(remote_jsonl)
    rem_by_tid = {_kid(k): k for k in rem.files}
    matched, unmatched = [], []
    for k in loc.files:
        tid = _kid(k)
        fl = tl.get(tid, set())
        best_t, best_ov = None, 0.0
        for t2, fr in tr.items():
            if t2 not in rem_by_tid:
                continue
            denom = len(fl | fr)
            ov = len(fl & fr) / denom if denom else 0.0
            if ov > best_ov:
                best_ov, best_t = ov, t2
        if best_t is not None and best_ov >= min_overlap:
            cos = float(np.dot(loc[k], rem[rem_by_tid[best_t]]))
            matched.append({"local": k, "remote": rem_by_tid[best_t],
                            "overlap": round(best_ov, 3), "cos": round(cos, 5),
                            "same_id": k == rem_by_tid[best_t]})
        else:
            unmatched.append({"local": k, "best_overlap": round(best_ov, 3)})
    r = {"mode": "content_match", "n_local": len(loc.files), "n_remote": len(rem.files),
         "n_matched": len(matched), "n_unmatched": len(unmatched),
         "unmatched": unmatched[:20]}
    r["matched_all"] = matched  # 全量（供可视化；约 n_remote 条）
    if matched:
        coss = np.array([m["cos"] for m in matched])
        ovs = np.array([m["overlap"] for m in matched])
        r.update({
            "cos_mean": round(float(coss.mean()), 5),
            "cos_min": round(float(coss.min()), 5),
            "cos_p10": round(float(np.percentile(coss, 10)), 5),
            "cos_ge_0.99": round(float(np.mean(coss >= 0.99)), 4),
            "cos_ge_0.95": round(float(np.mean(coss >= 0.95)), 4),
            "same_id_frac": round(float(np.mean([m["same_id"] for m in matched])), 4),
            "overlap_mean": round(float(ovs.mean()), 3),
        })
        r["worst5"] = sorted(matched, key=lambda m: m["cos"])[:5]
    return r


def check_embed_pair(local_p, remote_p):
    loc = np.load(local_p)
    rem = np.load(remote_p)
    kl, kr = set(loc.files), set(rem.files)
    common = sorted(kl & kr, key=_kid)
    r = {"local": str(local_p), "remote": str(remote_p),
         "n_local": len(kl), "n_remote": len(kr),
         "n_common": len(common),
         "only_local": sorted(kl - kr)[:20], "only_remote": sorted(kr - kl)[:20]}
    if common:
        coss = np.array([float(np.dot(loc[k], rem[k])) for k in common])
        r["cos_mean"] = round(float(coss.mean()), 5)
        r["cos_min"] = round(float(coss.min()), 5)
        r["cos_p10"] = round(float(np.percentile(coss, 10)), 5)
        r["cos_ge_0.999"] = round(float(np.mean(coss >= 0.999)), 4)
        r["cos_ge_0.99"] = round(float(np.mean(coss >= 0.99)), 4)
        r["cos_lt_0.9_count"] = int((coss < 0.9).sum())
        if len(common) >= 5:
            ks = common[:min(60, len(common))]
            Ml = np.stack([loc[k] for k in ks])
            Mr = np.stack([rem[k] for k in ks])
            Sl = Ml @ Ml.T; Sr = Mr @ Mr.T
            iu = np.triu_indices(len(ks), 1)
            r["sim_matrix_pearson"] = round(float(np.corrcoef(Sl[iu], Sr[iu])[0, 1]), 5)
    return r


def cmd_embed(args):
    out = {}
    if args.pairs:
        for spec in args.pairs:
            parts = spec.split(":")
            if len(parts) != 3:
                raise SystemExit(f"--pairs 格式: cam:local.npz:remote.npz，收到 {spec}")
            cam, lp, rp = parts
            print(f"===== embed check: {cam} =====", flush=True)
            r = check_embed_pair(lp, rp)
            out[cam] = r
            print(json.dumps({k: v for k, v in r.items()
                              if k not in ("local", "remote")},
                             ensure_ascii=False, indent=1))
    elif args.local_dir and args.remote_dir:
        ldir = Path(args.local_dir); rdir = Path(args.remote_dir)
        content_mode = bool(args.local_jsonl and args.remote_jsonl)
        for lp in sorted(ldir.glob("reid_emb_*_d3b_local.npz")):
            cam = lp.stem.split("_")[2]
            cands = list(rdir.glob(f"reid_emb_{cam}_d3b.npz"))
            if not cands:
                out[cam] = {"status": "remote_missing"}
                continue
            if content_mode:
                ljsonl = Path(args.local_jsonl) / f"{cam}_detections.jsonl"
                rjsonl = Path(args.remote_jsonl) / f"{cam}_detections.jsonl"
                if not (ljsonl.exists() and rjsonl.exists()):
                    out[cam] = {"status": "jsonl_missing"}
                    continue
                print(f"===== embed content check: {cam} =====", flush=True)
                r = check_embed_content(str(lp), str(cands[0]),
                                        str(ljsonl), str(rjsonl))
                out[cam] = r
                print(json.dumps({k: v for k, v in r.items()
                                  if k != "matched_all"},
                                 ensure_ascii=False, indent=1))
            else:
                print(f"===== embed check: {cam} =====", flush=True)
                r = check_embed_pair(lp, cands[0])
                out[cam] = r
                print(json.dumps({k: v for k, v in r.items()
                                  if k not in ("local", "remote")},
                                 ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\n[json] {args.json}")


def main():
    ap = argparse.ArgumentParser(description="本地 vs 远程 对账")
    sub = ap.add_subparsers(dest="kind", required=True)

    d = sub.add_parser("detect")
    d.add_argument("--local-dir", default="output_crosscam_local/detect")
    d.add_argument("--remote-dir", default="output_crosscam")
    d.add_argument("--cams", default="305,316,318,320,323,343")
    d.add_argument("--limit", type=int, default=0, help="只比较前 N 帧（0=全部）")
    d.add_argument("--json", default="")
    d.set_defaults(func=cmd_detect)

    e = sub.add_parser("embed")
    e.add_argument("--pairs", nargs="*", default=None,
                   help="cam:local.npz:remote.npz（可多个）")
    e.add_argument("--local-dir", default="")
    e.add_argument("--remote-dir", default="")
    e.add_argument("--local-jsonl", default="",
                   help="内容级模式：本地 detections jsonl 所在目录（按 {cam}_detections.jsonl 匹配）")
    e.add_argument("--remote-jsonl", default="",
                   help="内容级模式：远程 detections jsonl 所在目录")
    e.add_argument("--json", default="")
    e.set_defaults(func=cmd_embed)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
