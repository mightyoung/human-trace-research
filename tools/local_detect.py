#!/usr/bin/env python
"""本机版检测环节（ONNX GroundingDINO-base + ORT-DirectML）。

定位：替代远程 L20 版 tools/infer_supervision.py 中的推理核心——
在不再使用远程服务器的前提下，于本机（AMD RX 5700 XT / DirectML）跑出与
远程历史产物**同口径**的检测 jsonl + summary，供"本地链路 vs 远程产物"对账。

模型（2026-10-17 实锤）：
  远程生产模型 dino-model = IDEA-Research/grounding-dino-base（HF 格式，
  sha1 f9ce46b65eda3420734c…）。本脚本加载由 tools/export_onnx_hf.py
  从该权重导出的 ONNX（models/onnx/groundingdino_base.onnx）。

口径对齐（与远程 HF + L20 逐项对齐）：
  1. 预处理：复刻 HF GroundingDINOImageProcessor——
     shortest_edge=800 / longest_edge=1333 等比缩放（16:9 源恒为 1333×750），
     PIL BILINEAR；1/255 + ImageNet mean/std；RGB、NCHW、float32。
  2. 文本：prompt 固定 "person."（input_ids = [101, 2711, 1012, 102]，
     换 prompt 需重新导出 ONNX 并更新 --token-ids）。
  3. 后处理：复刻 HF post_process_grounded_object_detection——
     sigmoid(logits) 在 256 个 text token 维取 max 作为 score；score > box_threshold
     才保留；归一化 cxcywh → 原图坐标（**不 clip**，与 HF 一致）。
  4. Containment-NMS（0.85, keep=larger）与 ByteTrack（activation=0.10 /
     matching=0.8 / lost_buffer=90 / min_consec=1）复用相同实现
     （supervision==0.24.0，与远程 requirements 一致）。
  5. 输出 jsonl / summary 与远程逐字段同构（latency 除外，属本机指标）。

用法:
  python tools/local_detect.py --video media-resource/.../305_*.mp4 --cam 305
  python tools/local_detect.py --all
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from containment_nms import containment_nms  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SYNCED_CAMS = ["305", "316", "318", "320", "323", "343"]


# ---------------------------------------------------------------- 预处理
def get_size_with_aspect_ratio(h, w, shortest_edge=800, longest_edge=1333):
    """复刻 HF transformers get_size_with_aspect_ratio（GroundingDINO 图像处理）。

    16:9 源（如 2560×1440 / 1920×1080）输出恒为 (750, 1333)。
    注意与 Rust src/preprocess.rs 的细节差异：HF 用未取整的 raw_size 计算另一维
    （int 截断），两者对 16:9 源结果一致。
    """
    size = float(shortest_edge)
    raw_size = None
    min_o = float(min(h, w))
    max_o = float(max(h, w))
    if max_o / min_o * size > longest_edge:
        raw_size = longest_edge * min_o / max_o
        size = int(round(raw_size))  # Python round 为「四舍六入五成双」
    if (h <= w and h == size) or (w <= h and w == size):
        oh, ow = h, w
    elif w < h:
        ow = size
        oh = int(raw_size * h / w) if raw_size is not None else int(size * h / w)
    else:
        oh = size
        ow = int(raw_size * w / h) if raw_size is not None else int(size * w / h)
    return oh, ow


def preprocess_bgr(frame, shortest_edge=800, longest_edge=1333):
    """BGR 帧 -> ONNX 输入 (1,3,oh,ow) float32。与 HF 图像处理逐步骤对齐。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    oh, ow = get_size_with_aspect_ratio(h, w, shortest_edge, longest_edge)
    pil = Image.fromarray(rgb).resize((ow, oh), Image.BILINEAR)  # PIL: (W, H)
    x = np.asarray(pil, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    return x, oh, ow


def postprocess(logits, pred_boxes, orig_w, orig_h, box_threshold):
    """复刻 HF post_process_grounded_object_detection（框选择部分）。

    logits: [1,900,256]；pred_boxes: [1,900,4] 归一化 cxcywh。
    返回 (boxes[N,4] xyxy 原图像素坐标, scores[N])，均保持 query 升序。
    """
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits[0], -50.0, 50.0)))
    scores = prob.max(axis=1)
    keep = scores > box_threshold
    if not keep.any():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    b = pred_boxes[0][keep]
    s = scores[keep]
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    boxes = np.stack([(cx - bw / 2) * orig_w, (cy - bh / 2) * orig_h,
                      (cx + bw / 2) * orig_w, (cy + bh / 2) * orig_h], axis=1)
    return boxes.astype(np.float32), s.astype(np.float32)


# ---------------------------------------------------------------- 构造 session
def build_session(model_path, provider):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if provider == "dml":
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), so, providers=providers)
    return sess


# ---------------------------------------------------------------- 单路主流程
def run_one(args, video_path, cam, out_dir, sess, sv_module):
    import supervision as sv

    stem = cam
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[video] {video_path}  {w}x{h} @ {fps:.3f}fps  frames={total}", flush=True)

    # ByteTrack（参数与远程一致；frame_rate 用原始视频帧率）
    tracker = sv.ByteTrack(
        track_activation_threshold=args.track_activation,
        lost_track_buffer=int(args.lost_buffer),
        minimum_matching_threshold=0.8,
        frame_rate=float(fps),
        minimum_consecutive_frames=1,
    )
    print(f"[track] ByteTrack fps={fps} stride={args.stride} "
          f"activation={args.track_activation} (det_thresh={tracker.det_thresh}) "
          f"lost_buffer={args.lost_buffer} -> max_time_lost="
          f"{int(float(fps)/30.0*args.lost_buffer)} frames", flush=True)

    jsonl_path = out_dir / f"{stem}_detections.jsonl"
    jf = jsonl_path.open("w", encoding="utf-8")
    raw_jf = None
    if args.dump_raw:
        raw_jf = (out_dir / f"{stem}_raw.jsonl").open("w", encoding="utf-8")

    frame_idx = 0
    processed = 0
    t_start = time.time()
    latencies = []
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    if set(in_names) != {"pixel_values", "input_ids", "attention_mask", "pixel_mask"}:
        raise SystemExit(
            f"[fatal] 模型输入为 {in_names}，期望 base 版 4 输入 "
            f"(pixel_values/input_ids/attention_mask/pixel_mask)。"
            f"请用 tools/export_onnx_hf.py 导出。")
    # 文本输入固定：prompt "person." 的 BERT 编码（换 prompt 需重新导出 ONNX）
    token_ids = np.asarray([args.token_ids], dtype=np.int64)          # [1, L]
    attn_mask = np.ones_like(token_ids)
    pixel_mask = np.ones((1, 750, 1333), dtype=bool)                  # 与预处理输出同形状

    while True:
        ok = cap.grab()
        if not ok:
            break
        if args.max_frames and processed >= args.max_frames:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue

        ok, frame = cap.retrieve()
        if not ok:
            break

        t1 = time.time()
        x, oh, ow = preprocess_bgr(frame, args.shortest_edge, args.longest_edge)
        if (oh, ow) != (750, 1333):
            raise SystemExit(
                f"[fatal] 预处理输出 {ow}x{oh}，模型（Swin 常量折叠）仅支持 1333×750"
                f"（16:9 源）。当前视频 {w}x{h} 非 16:9。")
        outs = sess.run(out_names, {
            "pixel_values": x,
            "input_ids": token_ids,
            "attention_mask": attn_mask,
            "pixel_mask": pixel_mask,
        })
        got = dict(zip(out_names, outs))
        dt = time.time() - t1
        latencies.append(dt)

        boxes, scores = postprocess(got["pred_logits"], got["pred_boxes"],
                                    orig_w=w, orig_h=h, box_threshold=args.box_threshold)
        raw_count = len(boxes)

        # 诊断 dump：Containment-NMS 之前的模型原始输出（threshold 过滤后全量框）
        if raw_jf is not None:
            raw_jf.write(json.dumps({
                "frame": frame_idx,
                "raw_count": raw_count,
                "boxes": [[round(float(v), 2) for v in b] for b in boxes],
                "scores": [round(float(s), 5) for s in scores],
            }, ensure_ascii=False) + "\n")

        # Containment-NMS（与远程同一实现/参数）
        if raw_count > 1:
            if args.dedup_policy == "hybrid":
                keep_idx = containment_nms(boxes, scores,
                                           containment_threshold=args.containment_threshold,
                                           keep="larger",
                                           hy_score_k=args.hy_score_k,
                                           hy_min_h=args.hy_min_h)
            else:
                keep_idx = containment_nms(boxes, scores,
                                           containment_threshold=args.containment_threshold,
                                           keep=args.dedup_policy)
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
        deduped_count = len(boxes)

        det = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
        )
        det = tracker.update_with_detections(det)
        tracked_count = len(det)
        track_ids = det.tracker_id if det.tracker_id is not None else np.array([])
        labels = ["person"] * len(det)

        jf.write(json.dumps({
            "frame": frame_idx,
            "count": tracked_count,
            "deduped_count": deduped_count,
            "raw_count": raw_count,
            "removed": raw_count - deduped_count,
            "track_ids": [int(t) for t in track_ids.tolist()] if len(track_ids) else [],
            "latency_ms": round(dt * 1000, 1),
            "boxes": [[round(float(v), 1) for v in b] for b in det.xyxy],
            "scores": [round(float(s), 4) for s in det.confidence],
            "labels": labels,
        }, ensure_ascii=False) + "\n")

        processed += 1
        frame_idx += 1
        if processed % 20 == 0:
            el = time.time() - t_start
            print(f"[prog] {stem} processed={processed} fps={processed/el:.2f} "
                  f"last_latency={dt*1000:.0f}ms", flush=True)

    cap.release()
    jf.close()
    if raw_jf is not None:
        raw_jf.close()

    el = time.time() - t_start
    providers = sess.get_providers()
    summary = {
        "video": str(video_path),
        "cam": cam,
        "resolution": [w, h],
        "fps": round(fps, 3),
        "frames_total": total,
        "frames_processed": processed,
        "stride": args.stride,
        "device": "dml" if "DmlExecutionProvider" in providers else "cpu",
        "providers": providers,
        "model": str(args.model),
        "prompt": args.prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "elapsed_sec": round(el, 2),
        "throughput_fps": round(processed / el, 3) if el > 0 else None,
        "latency_ms_mean": round(float(np.mean(latencies)) * 1000, 1) if latencies else None,
        "latency_ms_p50": round(float(np.percentile(latencies, 50)) * 1000, 1) if latencies else None,
        "latency_ms_p90": round(float(np.percentile(latencies, 90)) * 1000, 1) if latencies else None,
        "tracking": "ByteTrack",
        "tracker_version": sv.__version__,
        "dedup": "containment_nms",
        "containment_threshold": args.containment_threshold,
        "dedup_keep": args.dedup_keep,
        "dedup_policy": args.dedup_policy,
        **({"hy_score_k": args.hy_score_k, "hy_min_h": args.hy_min_h}
           if args.dedup_policy == "hybrid" else {}),
        "count_mean": None,
    }
    counts, dedups, raws, all_ids = [], [], [], set()
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            counts.append(r["count"])
            dedups.append(r["deduped_count"])
            raws.append(r["raw_count"])
            all_ids.update(r["track_ids"])
    if counts:
        summary["count_mean"] = round(float(np.mean(counts)), 2)
        summary["count_max"] = int(max(counts))
        summary["count_min"] = int(min(counts))
        summary["count_std"] = round(float(np.std(counts)), 3)
        summary["deduped_count_mean"] = round(float(np.mean(dedups)), 2)
        summary["raw_count_mean"] = round(float(np.mean(raws)), 2)
        summary["boxes_removed_total"] = int(sum(raws) - sum(dedups))
        summary["boxes_removed_pct"] = (
            round(100.0 * (1 - sum(dedups) / sum(raws)), 2) if sum(raws) else 0.0
        )
        summary["unique_track_ids"] = len(all_ids)

    sp = out_dir / f"{stem}_summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {jsonl_path}", flush=True)
    print(f"[out] {sp}", flush=True)
    print(f"[done] {stem}: {processed} frames in {el:.0f}s "
          f"({processed/el:.2f} fps, lat_p50={summary['latency_ms_p50']}ms)", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="本机版检测（ONNX-DirectML）")
    ap.add_argument("--video", default="", help="单路视频路径")
    ap.add_argument("--cam", default="", help="机位号（305/316/318/320/323/343）")
    ap.add_argument("--all", action="store_true", help="跑全部 6 路（扫描 --synced-dir）")
    ap.add_argument("--synced-dir", default="media-resource/multi-cemra/synced-10min")
    ap.add_argument("--model", default="models/onnx/groundingdino_base.onnx",
                    help="base ONNX（tools/export_onnx_hf.py 从 HF base 导出）")
    ap.add_argument("--out-dir", default="output_crosscam_local/detect")
    ap.add_argument("--prompt", default="person", help="记录用（prompt 已烘焙进 ONNX）")
    ap.add_argument("--token-ids", default=[101, 2711, 1012, 102],
                    help="prompt 'person.' 的 BERT 编码 [CLS] person . [SEP]")
    ap.add_argument("--box-threshold", type=float, default=0.30,
                    help="与远程 infer_supervision.py 默认一致")
    ap.add_argument("--text-threshold", type=float, default=0.25, help="记录用")
    ap.add_argument("--stride", type=int, default=5, help="与远程一致")
    ap.add_argument("--max-frames", type=int, default=0, help="处理帧数上限（0=全部）")
    ap.add_argument("--track-activation", type=float, default=0.10)
    ap.add_argument("--lost-buffer", type=int, default=90)
    ap.add_argument("--containment-threshold", type=float, default=0.85)
    ap.add_argument("--dedup-keep", choices=["larger", "higher"], default="larger")
    ap.add_argument("--dedup-policy", choices=["larger", "higher", "hybrid"], default="larger",
                    help="C 方案接入点（与 infer_supervision 同义）："
                         "hybrid = 大框分数不足时让小框占位，仅对高≥--hy-min-h 的框生效")
    ap.add_argument("--hy-score-k", type=float, default=0.8)
    ap.add_argument("--hy-min-h", type=float, default=200.0)
    ap.add_argument("--shortest-edge", type=int, default=800)
    ap.add_argument("--longest-edge", type=int, default=1333)
    ap.add_argument("--provider", choices=["dml", "cpu"], default="dml")
    ap.add_argument("--dump-raw", action="store_true", default=True,
                    help="同时输出 tracker 前（containment 后）raw 序列，供诊断与 tracker 重放")
    ap.add_argument("--no-dump-raw", dest="dump_raw", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    sess = build_session(args.model, args.provider)
    providers = sess.get_providers()
    print(f"[model] {args.model} providers={providers}", flush=True)
    print(f"[onnx] inputs={[ (i.name, i.shape) for i in sess.get_inputs() ]} "
          f"outputs={[ (o.name, o.shape) for o in sess.get_outputs() ]}", flush=True)

    import supervision as sv  # 导入一次给 run_one 用

    jobs = []
    if args.all:
        root = Path(args.synced_dir)
        for cam in SYNCED_CAMS:
            matches = sorted(root.glob(f"*_{cam}/*.mp4"))
            if not matches:
                raise SystemExit(f"未找到 cam={cam} 的视频（{root}/*_{cam}/*.mp4）")
            jobs.append((cam, matches[0]))
    else:
        if not args.video:
            raise SystemExit("需要 --video 或 --all")
        cam = args.cam or Path(args.video).stem.split("_")[0]
        jobs.append((cam, Path(args.video)))

    for cam, vp in jobs:
        print(f"\n===== [{cam}] {vp} =====", flush=True)
        run_one(args, vp, cam, out_dir, sess, sv)


if __name__ == "__main__":
    main()
