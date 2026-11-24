#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检测器切换试点：YOLO26s-seg + ByteTrack 出轨迹 jsonl（与 DINO 版同格式）。

口径与 1012_ab 基线对齐：
  stride 5（每 5 帧一次）、classes=[0]、跟踪整段视频（不是逐帧独立检测）
差异（切换点）：
  model = yolo26s-seg（imgsz 1280, conf 0.10），tracker = ByteTrack(自定义阈值)
说明：
  DINO 基线的 track_activation 0.10 只影响新轨迹起点，这里用 new_track_thresh 对应；
  stride=5 意味着两次观测间隔 200ms，track_buffer 按"处理帧"计数，设为 60（≈3s 真实时间）。

输出：
  output_crosscam_local/1012_yolo/{seg}/{seg}_detections.jsonl
用法：
  python tools/yolo_track_det.py --seg 305_test_544 --max-frames 40   # 冒烟
  python tools/yolo_track_det.py --seg 305_test_544
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
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _vid_dir():
    """默认视频目录。延迟导入：算力机只放本脚本时也能用（显式传 --video 即可）。"""
    try:
        import encode_1012_osnet_seq as E
        return Path(E.VID_DIR)
    except Exception:
        return ROOT / "media-resource" / "1012"

DEFAULT_MODEL = ROOT / "models" / "Ultralytics--YOLO26" / "snapshots" / "master" / "yolo26s-seg.pt"
TRACKER_CFG = Path(__file__).resolve().parent / "bytetrack_1012.yaml"


def ensure_tracker_cfg():
    if TRACKER_CFG.exists():
        return TRACKER_CFG
    TRACKER_CFG.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.10\n"
        "track_low_thresh: 0.05\n"
        "new_track_thresh: 0.12\n"
        "track_buffer: 60\n"
        "match_thresh: 0.85\n"
        "fuse_score: True\n",
        encoding="utf-8")
    return TRACKER_CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="NMS IoU 阈值。实测 0.7 会留下同一人的头/全身两个框"
                         "（305_train 26%% 帧有重复框），0.5 可完全抑制，且框集合与"
                         "'0.7+后处理去重'逐帧一致（1000 帧仅 4 vs 12 个独有框），零召回代价")
    ap.add_argument("--dedup-mask", type=float, default=0.0,
                    help="同帧掩膜包含率阈值（0=关闭）。同一人的头/身两个框，小掩膜几乎完全落在"
                         "大掩膜内（实测包含率 >=0.9 的 52 对掩膜 IoU 中位 0.64）；而前后景两个"
                         "真人的掩膜互斥（包含率 <0.3 的 120 对掩膜 IoU 中位 0.00）。"
                         "建议 0.9：只删重复框，不误删真人投影重叠")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--tracker", default="",
                    help="ByteTrack 配置路径（默认 tools/bytetrack_1012.yaml）。"
                         "track_buffer 按'处理帧'计，改 stride 时须等比缩放以维持真实时间窗"
                         "（基线 stride5→60 ≈ 12 s，故通用式 = 300/stride）")
    ap.add_argument("--device", default="cpu",
                    help="推理设备：cpu（默认，与历史基线一致）或 cuda:0/cuda:1/0/1。"
                         "★ 换设备会改变检测框集合（浮点差异），同一批实验必须统一设备")
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 个采样帧（冒烟）")
    ap.add_argument("--out-dir", default=str(ROOT / "output_crosscam_local/1012_yolo"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--video", default="",
                    help="直接指定 mp4 路径（默认 <VID_DIR>/<seg>.mp4）。用于原卷局部区间诊断，免切片")
    ap.add_argument("--start-sec", type=float, default=0.0, help="起始秒（0=卷首）")
    ap.add_argument("--dur-sec", type=float, default=0.0, help="处理时长秒（0=到卷尾）")
    a = ap.parse_args()

    out_dir = Path(a.out_dir) / a.seg
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"{a.seg}_detections.jsonl"
    if outp.exists() and not a.force and not a.max_frames:
        print(f"[skip] {outp} 已存在")
        return

    va = Path(a.video) if a.video else _vid_dir() / f"{a.seg}.mp4"
    if not va.exists():
        raise SystemExit(f"missing video {va}")

    from ultralytics import YOLO
    model = YOLO(a.model)
    cfg = Path(a.tracker) if a.tracker else ensure_tracker_cfg()

    cap = cv2.VideoCapture(str(va))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {va}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = 0
    end_frame = None
    if a.start_sec > 0:
        start_frame = (int(round(a.start_sec * fps)) // a.stride) * a.stride
    if a.dur_sec > 0:
        end_frame = start_frame + int(round(a.dur_sec * fps))
    if start_frame:
        for _ in range(start_frame):          # grab 跳过（不解码，快）
            if not cap.grab():
                break

    fnum, sampled = start_frame, 0
    t0 = time.time()
    n_boxes, n_ids, ids_seen = 0, 0, set()
    n_dup = 0
    fh = open(outp, "w", encoding="utf-8")
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if end_frame is not None and fnum >= end_frame:
                break
            if fnum % a.stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                r = model.track(frame, persist=True, classes=[0], conf=a.conf, iou=a.iou,
                                imgsz=a.imgsz, device=a.device, tracker=str(cfg),
                                verbose=False)[0]
                bx = r.boxes
                boxes, ids, scores = [], [], []
                if bx is not None and len(bx):
                    xyxy = bx.xyxy.cpu().numpy().astype(np.float32)
                    cf = bx.conf.cpu().numpy().astype(np.float32)
                    ii = bx.id.cpu().numpy().astype(np.int32) if bx.id is not None else None
                    items = [j for j in range(len(xyxy))
                             if ii is not None and ii[j] is not None]
                    if a.dedup_mask > 0 and len(items) > 1 and r.masks is not None:
                        ms = r.masks.data.cpu().numpy()
                        if len(ms) >= len(xyxy):
                            keep = []
                            for j in sorted(items, key=lambda k: -float(cf[k])):
                                dup = False
                                for k in keep:
                                    m1, m2 = ms[j] > 0.5, ms[k] > 0.5
                                    s1, s2 = int(m1.sum()), int(m2.sum())
                                    if not s1 or not s2:
                                        continue
                                    small, big = (m1, m2) if s1 <= s2 else (m2, m1)
                                    ctn = float(np.logical_and(small, big).sum()) / min(s1, s2)
                                    if ctn >= a.dedup_mask:
                                        dup = True
                                        n_dup += 1
                                        break
                                if not dup:
                                    keep.append(j)
                            items = sorted(keep)
                    for j in items:
                        boxes.append([round(float(v), 1) for v in xyxy[j]])
                        ids.append(int(ii[j]))
                        scores.append(round(float(cf[j]), 4))
                n_boxes += len(boxes)
                n_ids += len(set(ids))
                ids_seen.update(ids)
                fh.write(json.dumps({"frame": fnum, "track_ids": ids,
                                     "boxes": boxes, "scores": scores}) + "\n")
                sampled += 1
                if sampled % 50 == 0:
                    el = time.time() - t0
                    print(f"[{a.seg}] {sampled} frames {el:.0f}s "
                          f"{sampled/max(el,1e-6):.2f} f/s boxes={n_boxes} "
                          f"ids={len(ids_seen)}", flush=True)
                if a.max_frames and sampled >= a.max_frames:
                    break
            fnum += 1
    finally:
        fh.close()
        cap.release()

    print(f"[{a.seg}] DONE sampled={sampled} video_frames={fnum} boxes={n_boxes} "
          f"unique_tracks={len(ids_seen)} mask_dedup_dropped={n_dup} "
          f"wall={time.time()-t0:.0f}s", flush=True)
    if a.max_frames:
        print(f"[smoke] 只写了前 {a.max_frames} 帧到 {outp}，正式跑请加 --force")


if __name__ == "__main__":
    main()
