#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""六路视频流并行运行 → 检测/跟踪/轨迹绘制 → 拼成单视图。

真实运行约束（不要误解本脚本的产出）：
  - 单张 AMD 5700XT 上 GroundingDINO-base（1333×750）约 0.6 s/帧；
  - 六路同时跑，推理在**单会话内串行**（GPU 只有一张，锁保护），这才是真实部署形态；
  - 因此视频不是 25 fps 实时，而是按"源时间步进"抽样的时序拼图：
    每路按同一时间表（step 秒/步）取帧，六路同步推进，合成一帧；
    输出帧数 = 步数，播放速度由 --out-fps 决定。

结构：
  6 个 worker 线程（解码 + 推理 + 跟踪 + 画框/轨迹）→ 按步号排队
  1 个合成器（主线程）：取六路同一步的画面 → 2×3 拼接 + HUD → 写 mp4

用法：
  python tools/six_stream_parallel.py --steps 90 --step-sec 0.4 --out-fps 5
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from containment_nms import containment_nms  # noqa: E402
from local_detect import build_session, get_size_with_aspect_ratio, postprocess, preprocess_bgr  # noqa: E402

VID = os.path.join(ROOT, "media-resource/multi-cemra/synced-10min")
CAMS = [
    ("316", "01_行政走廊通道_316", "316_085500-090500.mp4"),
    ("305", "02_办公区1_305", "305_085500-090500.mp4"),
    ("318", "03_办公区2_318", "318_085500-090500.mp4"),
    ("320", "04_办公区球机_320", "320_085500-090500.mp4"),
    ("323", "05_茶水间_323", "323_085500-090500.mp4"),
    ("343", "06_电梯间_343", "343_085500-090500.mp4"),
]
MODEL = os.path.join(ROOT, "models/onnx/groundingdino_base.onnx")
OUT_DIR = os.path.join(ROOT, "output_crosscam_local/demo_six")
TILE_W, TILE_H = 640, 360
TOKEN_IDS = [101, 2711, 1012, 102]   # "person" 的 BERT 分词
BOX_THR = 0.30
CONTAIN_THR = 0.85


class IoUTracker:
    """轻量跟踪：IoU + 中心距离双重门控（检测是抽样的，纯 IoU 会断）。"""

    def __init__(self, iou_thr=0.25, dist_frac=0.12, max_age=3):
        self.tracks = {}   # id -> {"box":..., "last":step, "trail": deque, "hits": int}
        self.next_id = 1
        self.iou_thr = iou_thr
        self.dist_frac = dist_frac
        self.max_age = max_age

    @staticmethod
    def _iou(a, b):
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(1e-6, ua)

    @staticmethod
    def _center(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def update(self, boxes, step, frame_w):
        used = set()
        out = []
        for b in boxes:
            cx, cy = self._center(b)
            best, best_key = None, None
            for tid, t in self.tracks.items():
                if tid in used or step - t["last"] > self.max_age:
                    continue
                iou = self._iou(b, t["box"])
                tx, ty = self._center(t["box"])
                dist = float(np.hypot(cx - tx, cy - ty)) / max(1, frame_w)
                if iou >= self.iou_thr or dist <= self.dist_frac:
                    key = (iou, -dist)
                    if best_key is None or key > best_key:
                        best, best_key = tid, key
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"box": b, "last": step, "trail": deque(maxlen=40), "hits": 1}
            else:
                t = self.tracks[best]
                t["box"], t["last"] = b, step
                t["hits"] += 1
            used.add(best)
            self.tracks[best]["trail"].append(self._center(b))
            out.append((best, b))
        return out


def infer(sess, frame, imgsz):
    x, oh, ow = preprocess_bgr(frame, *imgsz)
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    feed = {"pixel_values": x,
            "input_ids": np.asarray([TOKEN_IDS], dtype=np.int64),
            "attention_mask": np.ones((1, len(TOKEN_IDS)), dtype=np.int64),
            "pixel_mask": np.ones((1, 750, 1333), dtype=bool)}
    outs = sess.run(out_names, {k: feed[k] for k in in_names if k in feed})
    got = dict(zip(out_names, outs))
    h, w = frame.shape[:2]
    boxes, scores = postprocess(got["pred_logits"], got["pred_boxes"], orig_w=w, orig_h=h,
                                box_threshold=BOX_THR)
    if len(boxes) > 1:
        keep = containment_nms(boxes, scores, containment_threshold=CONTAIN_THR, keep="larger")
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores


def worker(cam, rel, fn, steps, step_frames, sess, lock, out_q, stats, imgsz, a_start_frames):
    path = os.path.join(VID, rel, fn)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        out_q.put(("ERR", cam, 0, None))
        return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = IoUTracker()
    box_ann = sv.BoxAnnotator(thickness=2)
    lbl_ann = sv.LabelAnnotator(text_scale=0.45, text_thickness=1, text_padding=4)

    pos = 0
    if a_start_frames > 0:
        while pos < a_start_frames:
            if not cap.grab():
                break
            pos += 1
    t_start = time.time()
    n_det = 0
    lat = []
    for step in range(steps):
        target = int(a_start_frames) + step * step_frames
        ok = True
        while pos < target:
            ok = cap.grab()
            if not ok:
                break
            pos += 1
        if not ok:
            out_q.put(("ERR", cam, step, None))
            break
        ok, frame = cap.read()
        if not ok:
            out_q.put(("ERR", cam, step, None))
            break
        pos += 1

        t0 = time.time()
        with lock:
            boxes, scores = infer(sess, frame, imgsz)
        lat.append(time.time() - t0)
        n_det += 1

        tracks = tracker.update(boxes, step, W)
        scale = TILE_W / float(W)
        tile = cv2.resize(frame, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

        # 轨迹（在缩放后的画布上画）
        for tid, t in tracker.tracks.items():
            if len(t["trail"]) < 2:
                continue
            pts = np.asarray([[int(x * scale), int(y * scale)] for x, y in t["trail"]], np.int32)
            h = (tid * 47) % 180
            bgr = cv2.cvtColor(np.uint8([[[h, 230, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            col = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            cv2.polylines(tile, [pts], False, col, 2, cv2.LINE_AA)
            cv2.circle(tile, tuple(pts[-1]), 3, col, -1)

        if len(boxes):
            xy = boxes * scale
            det = sv.Detections(xyxy=xy.astype(np.float32),
                                class_id=np.zeros(len(xy), int))
            tile = box_ann.annotate(tile, det)
            labels = [f"#{tid}" for tid, _b in tracks]
            if len(labels) == len(det):
                tile = lbl_ann.annotate(tile, det, labels=labels)

        # HUD
        alive = sum(1 for t in tracker.tracks.values() if step - t["last"] <= tracker.max_age)
        cv2.rectangle(tile, (0, 0), (TILE_W, 26), (30, 30, 30), -1)
        cv2.putText(tile, f"{cam}  step {step+1}/{steps}  now={alive}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out_q.put(("FRAME", cam, step, tile))

    stats[cam] = {"frames": n_det, "wall_s": time.time() - t_start,
                  "mean_lat_ms": float(np.mean(lat) * 1000) if lat else None,
                  "tracks": tracker.next_id - 1}
    cap.release()
    out_q.put(("DONE", cam, steps, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=90, help="源时间步数")
    ap.add_argument("--step-sec", type=float, default=0.4, help="每步代表的源时长（秒）")
    ap.add_argument("--out-fps", type=float, default=5.0)
    ap.add_argument("--start-sec", type=float, default=20.0)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "six_stream_parallel.mp4"))
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sess = build_session(MODEL, "dml")
    print("[model]", sess.get_providers(), flush=True)
    # 模型只吃 1333×750（16:9）
    cap0 = cv2.VideoCapture(os.path.join(VID, CAMS[0][1], CAMS[0][2]))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 25.0
    W0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()
    oh, ow = get_size_with_aspect_ratio(H0, W0, 800, 1333)
    if (oh, ow) != (750, 1333):
        raise SystemExit(f"非 16:9 源：{W0}x{H0} -> {ow}x{oh}，模型仅支持 1333x750")
    print(f"[src] {W0}x{H0} @ {fps:.2f}fps, step={a.step_sec}s "
          f"({int(a.step_sec*fps)} frames)", flush=True)

    step_frames = max(1, int(round(a.step_sec * fps)))
    n_total = len(CAMS) * a.steps
    print(f"[plan] steps={a.steps} step_frames={step_frames} 预计检测次数={n_total} "
          f"(单次约 0.6s → 约 {n_total*0.6/60:.1f} 分钟)", flush=True)

    lock = threading.Lock()
    out_q = queue.Queue(maxsize=64)
    stats = {}
    threads = []
    t_all = time.time()
    for cam, rel, fn in CAMS:
        th = threading.Thread(target=worker, daemon=True,
                              args=(cam, rel, fn, a.steps, step_frames, sess, lock,
                                    out_q, stats, (800, 1333),
                                    int(round(a.start_sec * fps))))
        th.start()
        threads.append(th)

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.out_fps,
                         (TILE_W * 3, TILE_H * 2))
    tiles = {}
    done = 0
    written = 0
    errors = []
    deadline = time.time() + 3600
    while done < len(CAMS) and time.time() < deadline:
        try:
            kind, cam, step, payload = out_q.get(timeout=10)
        except queue.Empty:
            if all(not t.is_alive() for t in threads):
                break
            continue
        if kind == "DONE":
            done += 1
            continue
        if kind == "ERR":
            errors.append({"cam": cam, "step": step})
            done += 1
            continue
        tiles[cam] = payload
        if len(tiles) == len(CAMS):
            grid = np.zeros((TILE_H * 2, TILE_W * 3, 3), np.uint8)
            for i, (c, _r, _f) in enumerate(CAMS):
                r, col = divmod(i, 3)
                grid[r * TILE_H:(r + 1) * TILE_H, col * TILE_W:(col + 1) * TILE_W] = tiles[c]
            cv2.putText(grid, f"6-stream parallel  step {step+1}/{a.steps}  "
                              f"src {a.start_sec + step*a.step_sec:5.1f}s  "
                              f"(single-GPU serialized inference)",
                        (10, TILE_H * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            vw.write(grid)
            written += 1
            tiles = {}
            if written % 10 == 0:
                print(f"[合成] {written}/{a.steps} 帧  {time.time()-t_all:.0f}s", flush=True)
    vw.release()
    wall = time.time() - t_all

    report = {"steps": a.steps, "step_sec": a.step_sec, "step_frames": step_frames,
              "out_fps": a.out_fps, "frames_written": written, "wall_s": wall,
              "n_detections": sum(v["frames"] for v in stats.values()),
              "mean_latency_ms": {k: v["mean_lat_ms"] for k, v in stats.items()},
              "tracks_total": {k: v["tracks"] for k, v in stats.items()},
              "errors": errors, "providers": sess.get_providers()}
    json.dump(report, open(os.path.join(OUT_DIR, "six_stream_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[done] {a.out}  frames={written}  wall={wall:.0f}s  "
          f"det={report['n_detections']}  errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
