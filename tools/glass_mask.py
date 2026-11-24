#!/usr/bin/env python
"""玻璃反光区域掩膜工具：预览草稿 / 过滤检测

背景：楼层多玻璃，检测/跟踪会把玻璃反光（镜像）当人。本工具提供：
  --preview : 在参考帧上画出掩膜区域（enabled=红、disabled=橙草稿），供人工圈改确认
  --apply   : 按掩膜过滤 detections.jsonl（剔除"落脚点落入区域内"的检测），输出过滤版 jsonl + 统计

用法：
  python tools/glass_mask.py --masks output_crosscam/glass_mask/glass_masks.json --preview \
      --refs-dir output_crosscam/glass_mask/refs --out-dir output_crosscam/glass_mask
  python tools/glass_mask.py --masks ... --apply --cam 316 \
      --in output_crosscam/316_detections.jsonl --out output_crosscam/glass_mask/316_masked.jsonl \
      --wh 2560 1440
"""
import argparse
import json
import os

import cv2
import numpy as np


def pt_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def load_masks(path):
    return json.load(open(path, encoding="utf-8"))


def preview(args):
    m = load_masks(args.masks)
    tiles = []
    for cid, cfg in m["cameras"].items():
        ref = os.path.join(args.refs_dir, f"{cid}_grid.jpg")
        img = cv2.imread(ref)
        if img is None:
            print(f"[skip] {cid} no ref")
            continue
        H, W = img.shape[:2]
        overlay = img.copy()
        for i, poly in enumerate(cfg["regions"]):
            pts = np.array([[int(u * W), int(v * H)] for u, v in poly], np.int32)
            color = (0, 0, 255) if cfg["enabled"] else (0, 165, 255)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], True, color, 2)
        img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        bar = np.full((30, W, 3), 252, np.uint8)
        st = "ENABLED" if cfg["enabled"] else "draft(off)"
        cv2.putText(bar, f"{cid}  [{st}]  {cfg['conf'][:40]}", (6, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
    if tiles:
        W = max(t.shape[1] for t in tiles)
        H = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, H - t.shape[0]), (0, W - t.shape[1]), (0, 0)),
                        constant_values=255) for t in tiles]
        vg = np.full((12, W, 3), 255, np.uint8)
        out = tiles[0]
        for t in tiles[1:]:
            out = np.vstack([out, vg, t])
        p = os.path.join(args.out_dir, "glass_mask_preview_all.png")
        cv2.imwrite(p, out)
        print(f"[out] {p}  ({out.shape})")
        # 分相机单图
        for cid, cfg in m["cameras"].items():
            ref = os.path.join(args.refs_dir, f"{cid}_grid.jpg")
            img = cv2.imread(ref)
            if img is None:
                continue
            H, W = img.shape[:2]
            overlay = img.copy()
            for poly in cfg["regions"]:
                pts = np.array([[int(u * W), int(v * H)] for u, v in poly], np.int32)
                color = (0, 0, 255) if cfg["enabled"] else (0, 165, 255)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], True, color, 2)
            img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
            cv2.imwrite(os.path.join(args.out_dir, f"preview_{cid}.png"), img)


def apply_mask(args):
    m = load_masks(args.masks)
    cfg = m["cameras"][args.cam]
    polys = cfg["regions"] if cfg["enabled"] else []
    W, H = args.wh
    n_rows = n_boxes = n_removed = 0
    tracks_before = {}
    tracks_after = {}
    out_lines = []
    with open(args.in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_rows += 1
            boxes = r.get("boxes", []) or []
            tids = r.get("track_ids", []) or []
            keep_boxes, keep_tids = [], []
            for i, b in enumerate(boxes):
                n_boxes += 1
                tid = tids[i] if i < len(tids) else None
                if tid is not None and int(tid) >= 0:
                    tracks_before[int(tid)] = tracks_before.get(int(tid), 0) + 1
                foot = (((b[0] + b[2]) / 2) / W, b[3] / H)
                if any(pt_in_poly(foot, poly) for poly in polys):
                    n_removed += 1
                    continue
                keep_boxes.append(b)
                keep_tids.append(tid)
                if tid is not None and int(tid) >= 0:
                    tracks_after[int(tid)] = tracks_after.get(int(tid), 0) + 1
            r["boxes"] = keep_boxes
            r["track_ids"] = keep_tids
            out_lines.append(json.dumps(r, ensure_ascii=False))
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    gone = [t for t in tracks_before if t not in tracks_after]
    print(f"[apply] cam={args.cam}  regions={len(polys)}")
    print(f"  rows        : {n_rows}")
    print(f"  boxes       : {n_boxes} -> {n_boxes - n_removed}  (removed {n_removed}, "
          f"{100.0 * n_removed / max(1, n_boxes):.2f}%)")
    print(f"  tracks      : {len(tracks_before)} -> {len(tracks_after)}  (完全消失 {len(gone)})")
    print(f"  [out] {args.out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--refs-dir", default="output_crosscam/glass_mask/refs")
    ap.add_argument("--out-dir", default="output_crosscam/glass_mask")
    ap.add_argument("--cam", default="")
    ap.add_argument("--in-path", dest="in_path", default="")
    ap.add_argument("--out_path", dest="out_path", default="")
    ap.add_argument("--wh", nargs=2, type=int, default=[0, 0])
    args = ap.parse_args()
    if args.preview:
        preview(args)
    if args.apply:
        assert args.cam and args.in_path and args.out_path and args.wh[0]
        apply_mask(args)


if __name__ == "__main__":
    main()
