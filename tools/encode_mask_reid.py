#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""掩膜重编码：用 YOLO26s-seg 的实例掩膜替代/净化 DINO 框裁剪，重算轨迹表征。

背景（2026-10-18）：
  - 几何收紧（A 方案）已否决——0.85 就切头顶，缩小框只能二选一（留椅子 or 切人）。
  - 实测 DINO 框内人体像素仅 51.7%（305 俯视工位）/ 60.5%（323 走廊），
    约一半像素是座椅/桌面/背景，把 OSNet 的轨迹均值拉向"工位特征"。
  - 掩膜是唯一能"只留人体、不切人"的手段。

本脚本与前序编码严格同口径（可直接进 eval_two_encodings 对照）：
  - 同一 a/b 分片、同一 np.linspace 8 点采样、同一 OSNet 权重、L2 归一后按轨迹均值。

两种掩膜变体（一次 YOLO 推理同时产出，避免重复算力）：
  V1 maskfill：裁剪仍用 DINO 框，把掩膜外的像素置黑（保留原尺度/长宽比）。
  V2 maskbbox：裁剪改为掩膜外接框（略扩边），掩膜外置黑（尺度归一到人体本体）。
  未匹配到掩膜（YOLO 漏检）时回退为原始裁剪，并记录 used_mask=0。

匹配口径：同帧内 DINO 框 ↔ YOLO 实例，一对一贪心 IoU>=0.3（取掩膜用，非召回评估）。

输出：
  output_crosscam_local/1012_osnet_emb_maskfill/{seg}_emb_{a,b}.npz
  output_crosscam_local/1012_osnet_emb_maskbbox/{seg}_emb_{a,b}.npz
  output_crosscam_local/1012_osnet_crops_mask/{seg}_crops.npz   (逐裁: eid/emb/质量/iou)
  output_crosscam_local/1012_osnet_crops_mask/{seg}_maskstats.json
  output_crosscam_local/1012_osnet_crops_mask/{seg}_check.jpg   (目检三联图)

用法：
  python tools/encode_mask_reid.py --seg 305_test_544 --max-frames 40 --limit-tracks 12
  python tools/encode_mask_reid.py --all
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encode_1012_osnet_seq as E  # noqa: E402
from yolo_vs_dino_frames import iou_matrix, greedy_match  # noqa: E402

ROOT = E.ROOT
DEFAULT_MODEL = ROOT / "models" / "Ultralytics--YOLO26" / "snapshots" / "master" / "yolo26s-seg.pt"
OUT_ROOT = ROOT / "output_crosscam_local" / "1012_osnet_crops_mask"
VARIANTS = ("maskfill", "maskbbox")
MIN_W, MIN_H = E.MIN_W, E.MIN_H
EXPAND = 0.04          # V2 掩膜外接框扩边比例
MIN_MASK_PIX = 40      # 掩膜内有效像素下限，低于此视为无效


def clip_box(box, W, H):
    x1, y1, x2, y2 = [float(v) for v in box]
    return (max(0, int(round(x1))), max(0, int(round(y1))),
            min(W, int(round(x2))), min(H, int(round(y2))))


def build_crops(frame, box, mask_full):
    """返回 {'maskfill': img|None, 'maskbbox': img|None, 'mask_frac': float, 'used': int}

    mask_full: uint8 {0,1} 与 frame 同尺寸；None 表示无可用掩膜。
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(box, W, H)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    out = {"maskfill": None, "maskbbox": None, "mask_frac": 0.0, "used": 0}
    crop = frame[y1:y2, x1:x2]

    if mask_full is None:
        out["maskfill"] = crop if crop.shape[0] >= MIN_H and crop.shape[1] >= MIN_W else None
        return out

    m = mask_full[y1:y2, x1:x2]
    msum = int(m.sum())
    out["mask_frac"] = float(msum) / float(max(1, (x2 - x1) * (y2 - y1)))
    if msum < MIN_MASK_PIX:
        out["maskfill"] = crop if crop.shape[0] >= MIN_H and crop.shape[1] >= MIN_W else None
        return out
    out["used"] = 1

    # V1：原框 + 掩膜置黑
    v1 = (crop * (m > 0)[..., None]).astype(np.uint8)
    if v1.shape[0] >= MIN_H and v1.shape[1] >= MIN_W:
        out["maskfill"] = v1

    # V2：掩膜外接框 + 掩膜置黑
    ys, xs = np.where(m > 0)
    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
    by1, by2 = int(ys.min()), int(ys.max()) + 1
    pw, ph = int(round((bx2 - bx1) * EXPAND)), int(round((by2 - by1) * EXPAND))
    bx1, by1 = max(0, bx1 - pw), max(0, by1 - ph)
    bx2, by2 = min(m.shape[1], bx2 + pw), min(m.shape[0], by2 + ph)
    if bx2 - bx1 >= MIN_W and by2 - by1 >= MIN_H:
        out["maskbbox"] = (crop[by1:by2, bx1:bx2] * (m[by1:by2, bx1:bx2] > 0)[..., None]).astype(np.uint8)
    return out


def collect_requests(seg, samples, limit_tracks=0):
    """-> frame -> [(ab, tid, k, box)]，按帧号升序"""
    by_frame = defaultdict(list)
    n_tr = 0
    for ab, p in (("a", E.AB_DIR / f"{seg}_a.jsonl"), ("b", E.AB_DIR / f"{seg}_b.jsonl")):
        tracks = E.load_ab(p)
        for tid in sorted(tracks):
            tf = tracks[tid]
            if len(tf) < 2:
                continue
            n_tr += 1
            if limit_tracks and n_tr > limit_tracks:
                continue
            idxs = np.linspace(0, len(tf) - 1, min(samples, len(tf))).astype(int)
            for k, ix in enumerate(idxs):
                fnum, box, _sc = tf[ix]
                by_frame[int(fnum)].append((ab, int(tid), k, box))
    return by_frame, n_tr


def grid_check(tiles, path, cols=6, cell=(64, 128)):
    """目检图：每行 3 联（原裁剪 / maskfill / maskbbox）。"""
    if not tiles:
        return False
    rows = []
    for raw, v1, v2 in tiles:
        ims = []
        for im in (raw, v1, v2):
            if im is None:
                im = np.zeros((cell[1], cell[0], 3), np.uint8)
            ims.append(cv2.resize(im, cell, interpolation=cv2.INTER_AREA))
        rows.append(np.hstack([np.pad(im, ((2, 2), (2, 2), (0, 0))) for im in ims]))
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
    cv2.imwrite(str(path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return True


def run_seg(seg, args, model):
    out_root = Path(args.out_root)
    emb_root = Path(args.emb_root)
    out_dirs = {v: emb_root / f"1012_osnet_emb_{v}" for v in VARIANTS}
    for v in VARIANTS:
        out_dirs[v].mkdir(parents=True, exist_ok=True)
    done = all((out_dirs[v] / f"{seg}_emb_a.npz").exists() and
               (out_dirs[v] / f"{seg}_emb_b.npz").exists() for v in VARIANTS)
    if done and not args.force:
        print(f"[skip] {seg}", flush=True)
        return None

    out_root.mkdir(parents=True, exist_ok=True)
    va = E.VID_DIR / f"{seg}.mp4"
    if not va.exists():
        raise SystemExit(f"missing video {va}")

    by_frame, n_tr = collect_requests(seg, args.samples, args.limit_tracks)
    frames = sorted(by_frame)
    if args.max_frames and len(frames) > args.max_frames:
        frames = frames[:args.max_frames]
    n_req = sum(len(by_frame[f]) for f in frames)
    print(f"[{seg}] tracks={n_tr} frames={len(frames)} requests={n_req}", flush=True)
    if not frames:
        return None

    enc = E.OSNetEnc(args.device)
    cap = cv2.VideoCapture(str(va))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {va}")

    CHUNK = 256
    feats = {v: [] for v in VARIANTS}           # 逐裁特征（顺序与 owners 对齐）
    owners = {v: [] for v in VARIANTS}
    rows = []                                   # 逐裁诊断（两变体共用请求集合）
    buf = {v: [] for v in VARIANTS}             # 待编码裁剪
    buf_keys = {v: [] for v in VARIANTS}        # 与 buf 一一对应的 (ab, tid, k)
    states = {"used": 0, "n": 0, "iou": [], "mask_frac": [], "no_mask_box": 0}
    checks = []
    cur, frame = -1, None
    t0 = time.time()

    def flush():
        for v in VARIANTS:
            if not buf[v]:
                continue
            V = enc.encode(buf[v])
            feats[v].extend(V)
            owners[v].extend(buf_keys[v])
            buf[v], buf_keys[v] = [], []

    visited = 0
    for fnum in frames:
        while cur < fnum:
            if not cap.grab():
                break
            cur += 1
        if cur < fnum:
            print(f"[warn] video ended at {cur}, want {fnum}", flush=True)
            break
        ok, frame = cap.retrieve()
        if not ok:
            continue
        H, W = frame.shape[:2]
        # YOLO 恒在 CPU（ultralytics 不支持 DirectML）；--device 只影响 OSNet 编码
        res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, classes=[0],
                            verbose=False, device="cpu", retina_masks=True,
                            max_det=100)[0]
        yb = res.boxes.xyxy.cpu().numpy().astype(np.float32) if res.boxes is not None \
            else np.zeros((0, 4), np.float32)
        ymasks = []
        if res.masks is not None and len(yb):
            for m in res.masks.data.cpu().numpy():
                # retina_masks=True 时已是原图尺寸；防御性 resize
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                ymasks.append((m > 0.5).astype(np.uint8))
        else:
            ymasks = [None] * len(yb)
        visited += 1

        reqs = by_frame[fnum]
        db = np.asarray([r[3] for r in reqs], dtype=np.float32).reshape(-1, 4)
        pairs = greedy_match(iou_matrix(db, yb), args.match_iou) if len(yb) and len(db) else []
        pmap = {i: (j, v) for i, j, v in pairs}

        for i, (ab, tid, k, box) in enumerate(reqs):
            j, iou = pmap.get(i, (-1, 0.0))
            mf = ymasks[j] if j >= 0 else None
            crops = build_crops(frame, box, mf)
            states["n"] += 1
            if crops is None:
                continue
            if mf is None:
                states["no_mask_box"] += 1
            else:
                states["iou"].append(float(iou))
            states["mask_frac"].append(crops["mask_frac"])
            states["used"] += crops["used"]
            X1, Y1, X2, Y2 = clip_box(box, W, H)
            raw = frame[Y1:Y2, X1:X2]
            area = ((X2 - X1) * (Y2 - Y1)) / float(W * H)
            g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.size else np.zeros((2, 2), np.uint8)
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            for v in VARIANTS:
                if crops[v] is None:
                    continue
                buf[v].append(crops[v])
                buf_keys[v].append((ab, tid, k))
            rows.append({"ab": ab, "tid": tid, "k": k, "frame": int(fnum),
                         "iou": round(float(iou), 3), "mask_frac": round(crops["mask_frac"], 3),
                         "used": int(crops["used"])})
            if len(checks) < 36 and crops["maskfill"] is not None and crops["maskbbox"] is not None:
                checks.append((raw, crops["maskfill"], crops["maskbbox"]))
            if len(rows) % CHUNK == 0:
                flush()
        if visited % 50 == 0:
            el = time.time() - t0
            print(f"[{seg}] {visited}/{len(frames)} frames  {el:.0f}s  "
                  f"{visited/max(el,1e-6):.2f} f/s  crops={len(rows)}", flush=True)
    cap.release()
    flush()
    print(f"[{seg}] yolo+decode {visited} frames, {len(rows)} crops, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---------- 汇总 ----------
    res = {v: {"a": {}, "b": {}} for v in VARIANTS}
    qmap = {(r["ab"], r["tid"], r["k"]): (r["iou"], r["mask_frac"], r["used"]) for r in rows}
    for v in VARIANTS:
        if not feats[v]:
            continue
        V = np.stack(feats[v]).astype(np.float32)
        keys = owners[v]
        np.savez(out_root / f"{seg}_crops_{v}.npz",
                 eids=np.asarray([f"{b}#{t}#{k}" for b, t, k in keys]), emb=V,
                 q=np.asarray([qmap[k] for k in keys], np.float32).reshape(-1, 3))
        by_track = defaultdict(list)
        for key, vec in zip(keys, V):
            by_track[(key[0], key[1])].append(vec)
        for (ab, tid), vs in by_track.items():
            m = np.mean(np.stack(vs), 0)
            m = m / (np.linalg.norm(m) + 1e-9)
            res[v][ab][f"{seg}#{tid}"] = m.astype(np.float32)
        np.savez(out_dirs[v] / f"{seg}_emb_a.npz", **res[v]["a"])
        np.savez(out_dirs[v] / f"{seg}_emb_b.npz", **res[v]["b"])

    ious = np.asarray(states["iou"], np.float32) if states["iou"] else np.zeros(0, np.float32)
    mfs = np.asarray(states["mask_frac"], np.float32) if states["mask_frac"] else np.zeros(0, np.float32)
    stats = {
        "seg": seg, "tracks": n_tr, "sampled_frames": len(frames),
        "crops": states["n"], "mask_used_pct": round(100.0 * states["used"] / max(1, states["n"]), 2),
        "no_yolo_instance_pct": round(100.0 * states["no_mask_box"] / max(1, states["n"]), 2),
        "iou_median": round(float(np.median(ious)), 4) if ious.size else None,
        "mask_frac_median": round(float(np.median(mfs)), 4) if mfs.size else None,
        "mask_frac_p10": round(float(np.percentile(mfs, 10)), 4) if mfs.size else None,
        "tracks_a": {v: len(res[v]["a"]) for v in VARIANTS},
        "tracks_b": {v: len(res[v]["b"]) for v in VARIANTS},
        "imgsz": args.imgsz, "conf": args.conf, "match_iou": args.match_iou,
        "device": args.device, "wall_sec": round(time.time() - t0, 1),
    }
    (out_root / f"{seg}_maskstats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    grid_check(checks, out_root / f"{seg}_check.jpg")
    print(f"[{seg}] DONE mask_used={stats['mask_used_pct']}% "
          f"mask_frac_med={stats['mask_frac_median']} "
          f"a/b={stats['tracks_a']}/{stats['tracks_b']}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="")
    ap.add_argument("--segs", default="", help="逗号分隔的多段，用于并行分片")
    ap.add_argument("--all", action="store_true", help="跑 TRAIN_SEGS + TEST_SEGS 全 8 段")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--match-iou", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--max-frames", type=int, default=0, help="冒烟用：只看前 N 个采样帧")
    ap.add_argument("--limit-tracks", type=int, default=0, help="冒烟用：只取前 N 条轨迹")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--emb-root", default=str(ROOT / "output_crosscam_local"))
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    if args.all:
        import session_selector_v1 as S
        segs = S.TRAIN_SEGS + S.TEST_SEGS
    elif args.segs:
        segs = [x.strip() for x in args.segs.split(",") if x.strip()]
    else:
        segs = [args.seg]
    for s in segs:
        run_seg(s, args, model)


if __name__ == "__main__":
    main()
