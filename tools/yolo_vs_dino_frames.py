#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO-seg vs GroundingDINO：同帧框匹配率对照（C1 决策依据）。

背景（2026-10-18）：A 方案（几何收紧）已否决——0.85 就切头顶，几何收缩无法
只去椅子不切人。候选 C1 = 换 YOLO11/26-seg（COCO person 检测 + 实例掩膜），
风险是远距离小目标召回可能低于 GroundingDINO。本脚本先量化这个风险。

对照口径：
  - 基准：现有生产检测 output_crosscam_local/1012_ab/{seg}_a.jsonl
    （GroundingDINO-base ONNX / 1333x750 / box_thr 0.30 / containment-NMS 0.85
     / ByteTrack 激活后输出，每 5 帧一条）。
  - 候选：YOLO-seg（默认 yolo26s-seg.pt, imgsz=1280 与 DINO 有效分辨率对齐）。
  - 帧：从基准帧号里**均匀抽 N 帧**，YOLO 在完全相同的 frame 号上推理。
  - 匹配：一对一贪心，IoU>=0.5（主）/ 0.3、0.7（敏感性）。

输出：
  output_crosscam_local/yolo_vs_dino/{seg}_yolo.jsonl   每帧 YOLO 框+分数+掩膜占比
  output_crosscam_local/yolo_vs_dino/{seg}_match.json   统计
  output_crosscam_local/yolo_vs_dino/{seg}_compare.jpg  代表帧并排图

用法:
  python tools/yolo_vs_dino_frames.py --seg 305_test_544 --n 300
  python tools/yolo_vs_dino_frames.py --all
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEG_DIR = ROOT / "media-resource" / "1012" / "segments"
DINO_DIR = ROOT / "output_crosscam_local" / "1012_ab"
OUT_DIR = ROOT / "output_crosscam_local" / "yolo_vs_dino"
DEFAULT_MODEL = ROOT / "models" / "Ultralytics--YOLO26" / "snapshots" / "master" / "yolo26s-seg.pt"

PERSON_CLASS = 0


# ------------------------------------------------------------------ 基础几何
def iou_matrix(a, b):
    """a:[N,4] b:[M,4] xyxy -> [N,M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    aa = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    ba = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = aa + ba[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(iou, thr):
    """按 IoU 降序一对一匹配。返回 [(i,j,iou)]"""
    if iou.size == 0:
        return []
    idx = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_a, used_b, pairs = set(), set(), []
    for i, j in idx:
        v = float(iou[i, j])
        if v < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(int(i))
        used_b.add(int(j))
        pairs.append((int(i), int(j), v))
    return pairs


def load_dino(jsonl_path, stride_keep=5):
    """读基准 jsonl -> {frame: np.array[N,4]}"""
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            b = r.get("boxes") or []
            out[int(r["frame"])] = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    return out


def pick_frames(frames, n):
    """在基准帧号上均匀抽样 n 个（排序、去重）。"""
    fs = sorted(frames)
    if len(fs) <= n:
        return fs
    idx = np.linspace(0, len(fs) - 1, n).round().astype(int)
    return [fs[i] for i in sorted(set(idx.tolist()))]


# ------------------------------------------------------------------ YOLO 推理
def run_yolo(video, targets, model, imgsz, conf, device):
    """顺序解码到目标帧并推理。返回 [{frame, boxes, scores, mask_frac}]"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tgt = sorted(targets)
    ti = 0
    frame_idx = 0
    rows = []
    t0 = time.time()
    next_report = 0
    while ti < len(tgt):
        ok = cap.grab()
        if not ok:
            print(f"[warn] video ended at frame {frame_idx}, targets left={len(tgt)-ti}", flush=True)
            break
        if frame_idx < tgt[ti]:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            frame_idx += 1
            continue
        res = model.predict(frame, imgsz=imgsz, conf=conf, classes=[PERSON_CLASS],
                            verbose=False, device=device, retina_masks=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32) if res.boxes is not None else np.zeros((0, 4), np.float32)
        scores = res.boxes.conf.cpu().numpy().astype(np.float32) if res.boxes is not None else np.zeros((0,), np.float32)
        # 掩膜在框内的填充率（人体像素 / 框面积），按坐标比例换算，不做全图 resize
        mask_frac = np.zeros((len(boxes),), dtype=np.float32)
        if res.masks is not None and len(boxes):
            m = res.masks.data.cpu().numpy()  # [N, Mh, Mw] (float 0/1)
            Mh, Mw = m.shape[1], m.shape[2]
            for k, b in enumerate(boxes):
                bw = max((b[2] - b[0]) * Mw / W, 1e-6)
                bh = max((b[3] - b[1]) * Mh / H, 1e-6)
                mask_frac[k] = float(m[k].sum()) / (bw * bh)
        rows.append({
            "frame": int(frame_idx),
            "boxes": [[round(float(v), 1) for v in b] for b in boxes],
            "scores": [round(float(s), 4) for s in scores],
            "mask_frac": [round(float(v), 4) for v in mask_frac],
        })
        ti += 1
        if ti >= next_report:
            el = time.time() - t0
            print(f"[yolo] {ti}/{len(tgt)} frames  {el:.0f}s  {ti/max(el,1e-6):.2f} f/s", flush=True)
            next_report = ti + max(20, len(tgt) // 10)
        frame_idx += 1
    cap.release()
    return {"W": W, "H": H, "rows": rows}


# ------------------------------------------------------------------ 统计
def analyze(dino, yolo_rows, W, H, thresholds=(0.3, 0.5, 0.7), det_confs=(0.10, 0.25, 0.40)):
    rec = {}
    fmap = {r["frame"]: r for r in yolo_rows}
    frames = [f for f in fmap if f in dino]
    A_total = float(W * H)

    for dc in det_confs:
        per_thr = {}
        all_sizes = []
        for thr in (0.3, 0.5, 0.7):
            d_tot = y_tot = matched = 0
            ious = []
            frame_recall = []
            yolo_only = 0
            area_ratio = []
            per_bucket = {"tiny": [0, 0], "small": [0, 0], "mid": [0, 0], "large": [0, 0]}
            for f in frames:
                db = dino[f]
                yb_all = np.asarray(fmap[f]["boxes"], dtype=np.float32).reshape(-1, 4)
                ysc = np.asarray(fmap[f]["scores"], dtype=np.float32).reshape(-1)
                keep = ysc >= dc
                yb = yb_all[keep]
                if len(yb):
                    iou = iou_matrix(db, yb)
                    pairs = greedy_match(iou, thr)
                else:
                    iou = np.zeros((len(db), 0), np.float32)
                    pairs = []
                d_tot += len(db)
                y_tot += len(yb)
                matched += len(pairs)
                yolo_only += len(yb) - len(pairs)
                for i, j, v in pairs:
                    ious.append(v)
                    b = db[i]
                    a = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
                    area_ratio.append(float(a) / A_total)
                frame_recall.append(len(pairs) / len(db) if len(db) else 1.0)
                # 尺寸分桶（DINO 框相对画面面积）
                for i in range(len(db)):
                    b = db[i]
                    rel = max((b[2] - b[0]) * (b[3] - b[1]), 0) / A_total
                    key = "tiny" if rel < 0.001 else "small" if rel < 0.005 else "mid" if rel < 0.02 else "large"
                    per_bucket[key][1] += 1
                    if any(p[0] == i for p in pairs):
                        per_bucket[key][0] += 1
            per_thr[f"iou{thr}"] = {
                "dino_boxes": d_tot,
                "yolo_boxes": y_tot,
                "matched": matched,
                "recall": round(matched / d_tot, 4) if d_tot else None,
                "yolo_only_boxes": yolo_only,
                "iou_median": round(float(np.median(ious)), 4) if ious else None,
                "iou_p10": round(float(np.percentile(ious, 10)), 4) if ious else None,
                "frame_recall_mean": round(float(np.mean(frame_recall)), 4) if frame_recall else None,
                "frame_full_recall_pct": round(100.0 * float(np.mean([1.0 if r >= 0.999 else 0.0 for r in frame_recall])), 2) if frame_recall else None,
                "bucket": {k: {"hit": v[0], "n": v[1], "recall": round(v[0] / v[1], 4) if v[1] else None}
                           for k, v in per_bucket.items()},
            }
        rec[f"conf{dc}"] = per_thr
    rec["_meta"] = {
        "frames": len(frames),
        "dino_frames_available": len(dino),
        "resolution": [W, H],
    }
    return rec


# ------------------------------------------------------------------ 可视化
def _draw(img, boxes, color, thick=3, label=None, scores=None):
    out = img.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        if label:
            txt = label if scores is None else f"{label}{scores[i]:.2f}"
            cv2.putText(out, txt, (x1 + 3, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
    return out


def make_compare(video, dino, yolo_rows, frames_pick, out_path, imgsz_note, max_w=640):
    """代表帧：左 DINO、右 YOLO，各缩放到 max_w 宽。"""
    fmap = {r["frame"]: r for r in yolo_rows}
    tiles = []
    cap = cv2.VideoCapture(str(video))
    tgt = sorted(frames_pick)
    ti = 0
    fi = 0
    while ti < len(tgt) and cap.grab():
        if fi == tgt[ti]:
            ok, frame = cap.retrieve()
            if not ok:
                break
            db = dino.get(fi, np.zeros((0, 4), np.float32))
            yb = np.asarray(fmap[fi]["boxes"], dtype=np.float32).reshape(-1, 4) if fi in fmap else np.zeros((0, 4), np.float32)
            ys = np.asarray(fmap[fi]["scores"], dtype=np.float32) if fi in fmap else np.zeros((0,), np.float32)
            left = _draw(frame, db, (0, 200, 0), 3, "DINO")
            right = _draw(frame, yb, (0, 0, 255), 3, "YOLO", ys)
            for im in (left, right):
                cv2.putText(im, f"f{fi}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            pair = np.hstack([left, right])
            h, w = pair.shape[:2]
            scale = max_w / w
            pair = cv2.resize(pair, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            tiles.append(pair)
            ti += 1
        fi += 1
    cap.release()
    if tiles:
        grid = np.vstack(tiles)
        if grid.shape[0] > 4000:
            grid = grid[:4000]
        cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return True
    return False


# ------------------------------------------------------------------ 主流程
def run_seg(seg, args, model):
    video = SEG_DIR / f"{seg}.mp4"
    dino_jsonl = DINO_DIR / f"{seg}_a.jsonl"
    if not video.exists():
        raise SystemExit(f"missing video {video}")
    if not dino_jsonl.exists():
        raise SystemExit(f"missing dino jsonl {dino_jsonl}")
    dino = load_dino(dino_jsonl)
    targets = pick_frames(list(dino.keys()), args.n)
    print(f"[{seg}] dino frames={len(dino)} picked={len(targets)} "
          f"range={targets[0]}..{targets[-1]}", flush=True)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.out_suffix
    t0 = time.time()
    res = run_yolo(video, targets, model, args.imgsz, args.conf, args.device)
    el = time.time() - t0
    W, H = res["W"], res["H"]
    rows = res["rows"]

    with (out_dir / f"{seg}{suffix}_yolo.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = analyze(dino, rows, W, H)
    stats["_meta"].update({
        "seg": seg,
        "video": str(video.relative_to(ROOT)),
        "dino_jsonl": str(dino_jsonl.relative_to(ROOT)),
        "model": str(Path(args.model).name),
        "imgsz": args.imgsz,
        "conf_floor": args.conf,
        "device": args.device,
        "yolo_fps": round(len(rows) / max(el, 1e-6), 3),
        "wall_sec": round(el, 1),
    })
    (out_dir / f"{seg}{suffix}_match.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 代表帧：按"匹配数最少/最多"各取若干，外加均匀抽样
    fmap = {r["frame"]: r for r in rows}
    scored = []
    for f in targets:
        if f not in fmap:
            continue
        db = dino[f]
        yb = np.asarray(fmap[f]["boxes"], np.float32).reshape(-1, 4)
        ys = np.asarray(fmap[f]["scores"], np.float32)
        yb = yb[ys >= 0.25]
        m = len(greedy_match(iou_matrix(db, yb), 0.5)) if len(yb) else 0
        scored.append((m - len(db), m, f))
    scored.sort()
    worst = [s[2] for s in scored[:3]]
    best = [s[2] for s in scored[-3:]]
    rng = np.random.default_rng(0)
    mid = list(rng.choice(targets, size=min(4, len(targets)), replace=False)) if targets else []
    pick = sorted(set(worst + best + [int(x) for x in mid]))
    ok = make_compare(video, dino, rows, pick, out_dir / f"{seg}{suffix}_compare.jpg", args.imgsz)
    print(f"[{seg}] compare img={'ok' if ok else 'fail'} ({len(pick)} frames)", flush=True)

    k = stats["conf0.1"]["iou0.5"]
    print(f"[{seg}] DONE  recall(iou.5,conf.1)={k['recall']}  "
          f"dino={k['dino_boxes']} yolo={k['yolo_boxes']} matched={k['matched']} "
          f"yolo_only={k['yolo_only_boxes']}  wall={el:.0f}s", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="", help="段名，如 305_test_544")
    ap.add_argument("--all", action="store_true", help="跑 305_test_544 + 323_test_527")
    ap.add_argument("--n", type=int, default=300, help="抽样帧数")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05, help="YOLO 推理下限（统计时再按阈值筛）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out-suffix", default="", help="输出文件名后缀，如 _i1920")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    segs = ["305_test_544", "323_test_527"] if args.all else [args.seg]
    for s in segs:
        run_seg(s, args, model)


if __name__ == "__main__":
    main()
