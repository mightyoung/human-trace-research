#!/usr/bin/env python
"""低光/逆光增强对检测的影响 A/B（本机 ONNX-DirectML）。

问题：在低光照机位（343 电梯间=最暗；320 球机=逆光）上，对输入帧做光照增强，
      能否提升 GroundingDINO-base 的检出质量？

设计：
  - 抽帧 stride=5，取 --frames 帧（默认 240）
  - 三个条件：orig（原图）/ lab_gamma（LAB-L 自适应 gamma，算法同 P1 人脸增强）/
    clahe（LAB-L CLAHE clip=2.5）
  - 同一 ONNX 模型、同一后处理（box_threshold=0.30）；不做跟踪/去重（只看检测响应）
指标：
  - 每条件：总框数、帧均、检出帧率、conf mean/median/p90
  - 条件间：IoU>0.5 匹配后的新增/丢失框数、新增框 conf 分布
  - 亮度：全图 LAB-L 均值（增强前后）
输出：
  - output_crosscam_local/lowlight_ab/{cam}_ab.json
  - {cam}_montage.jpg（抽帧三条件拼图，画框，供人工抽检）

用法:
  python tools/lowlight_detect_ab.py --cam 343 --frames 240
  python tools/lowlight_detect_ab.py --cam 320 --frames 240
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_detect import build_session, postprocess, preprocess_bgr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VIDEO = {
    "305": "media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4",
    "316": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    "318": "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4",
    "320": "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4",
    "323": "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
    "343": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
}


# ---------------------------------------------------------------- 增强算子
def apply_l_gamma(bgr, gamma):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lut = ((np.arange(256) / 255.0) ** gamma * 255).clip(0, 255).astype(np.uint8)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_l(bgr, clip=2.5, tile=4):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def luma_mean(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(lab[:, :, 0].mean())


def enhance(frame, mode):
    """返回 (enhanced_bgr, meta)。lab_gamma 算法与 gil_face_enhance.enhance 的
    dark_gamma 一致：目标亮度 128、gamma 限制 [0.45, 1]；极暗（<60）加双边滤波。"""
    mu = luma_mean(frame)
    if mode == "orig":
        return frame, {"mode": "orig", "luma": round(mu, 2)}
    if mode == "lab_gamma":
        target = 128.0
        g = math.log(target / 255.0) / math.log(max(mu, 1.0) / 255.0)
        g = min(max(g, 0.45), 1.0)
        out = apply_l_gamma(frame, g)
        meta = {"mode": "lab_gamma", "luma_in": round(mu, 2), "gamma": round(g, 3)}
        if mu < 60:
            out = cv2.bilateralFilter(out, 5, 25, 25)
            meta["bilateral"] = 5
        meta["luma_out"] = round(luma_mean(out), 2)
        return out, meta
    if mode == "clahe":
        out = clahe_l(frame, clip=2.5, tile=4)
        return out, {"mode": "clahe", "luma_in": round(mu, 2),
                     "luma_out": round(luma_mean(out), 2)}
    raise ValueError(mode)


# ---------------------------------------------------------------- 检测
def detect(sess, frame, token_ids, box_threshold):
    in_names = [i.name for i in sess.get_inputs()]
    out_names = [o.name for o in sess.get_outputs()]
    h, w = frame.shape[:2]
    x, oh, ow = preprocess_bgr(frame, 800, 1333)
    if (oh, ow) != (750, 1333):
        raise SystemExit(f"预处理输出 {ow}x{oh} != 1333x750（源非 16:9）")
    tok = np.asarray([token_ids], dtype=np.int64)
    outs = sess.run(out_names, {
        "pixel_values": x,
        "input_ids": tok,
        "attention_mask": np.ones_like(tok),
        "pixel_mask": np.ones((1, 750, 1333), dtype=bool),
    })
    got = dict(zip(out_names, outs))
    return postprocess(got["pred_logits"], got["pred_boxes"],
                       orig_w=w, orig_h=h, box_threshold=box_threshold)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xx1 = np.maximum(ax1[:, None], bx1[None, :])
    yy1 = np.maximum(ay1[:, None], by1[None, :])
    xx2 = np.minimum(ax2[:, None], bx2[None, :])
    yy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_diff(boxes_a, sa, boxes_b, sb, thr=0.5):
    """a 为参考；返回 (matched, added(b), lost(a), added_confs, lost_confs)。"""
    M = iou_matrix(boxes_a, boxes_b)
    matched = int((M > thr).any(axis=1).sum())
    lost = int((M > thr).any(axis=1).__invert__().sum())
    added = int((M > thr).any(axis=0).__invert__().sum())
    added_confs = [float(s) for j, s in enumerate(sb) if not (M[:, j] > thr).any()]
    lost_confs = [float(s) for i, s in enumerate(sa) if not (M[i, :] > thr).any()]
    return matched, added, lost, added_confs, lost_confs


def draw_boxes(img, boxes, scores, color, scale=0.5):
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = [int(v * scale) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(out, f"{s:.2f}", (x1 + 2, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="低光增强检测 A/B")
    ap.add_argument("--cam", default="343", choices=list(VIDEO.keys()))
    ap.add_argument("--video", default="", help="覆盖默认视频路径")
    ap.add_argument("--frames", type=int, default=240, help="处理帧数（stride 后）")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="起始帧号（用于避开空场景；须为 stride 的倍数）")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--model", default="models/onnx/groundingdino_base.onnx")
    ap.add_argument("--provider", choices=["dml", "cpu"], default="dml")
    ap.add_argument("--token-ids", default=[101, 2711, 1012, 102])
    ap.add_argument("--out-dir", default="output_crosscam_local/lowlight_ab")
    args = ap.parse_args()

    video = Path(args.video) if args.video else ROOT / VIDEO[args.cam]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = build_session(ROOT / args.model, args.provider)
    print(f"[model] providers={sess.get_providers()}", flush=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[video] {video} {W}x{H}@{fps:.2f}", flush=True)

    modes = ["orig", "lab_gamma", "clahe"]
    stats = {m: {"boxes": 0, "frames": 0, "conf": [], "det_frames": 0} for m in modes}
    diffs = {"gamma": {"matched": 0, "added": 0, "lost": 0,
                       "added_conf": [], "lost_conf": []},
             "clahe": {"matched": 0, "added": 0, "lost": 0,
                       "added_conf": [], "lost_conf": []}}
    per_frame = []
    montage = []
    start = args.start_frame - (args.start_frame % args.stride)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        print(f"[start] frame={start}", flush=True)
    frame_idx = start
    n_done = 0
    t0 = time.time()

    while n_done < args.frames:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        res = {}
        metas = {}
        for m in modes:
            img, meta = enhance(frame, m)
            boxes, scores = detect(sess, img, args.token_ids, args.box_threshold)
            res[m] = (boxes, scores)
            metas[m] = meta
            st = stats[m]
            st["frames"] += 1
            st["boxes"] += len(boxes)
            st["conf"].extend([float(s) for s in scores])
            st["det_frames"] += 1 if len(boxes) > 0 else 0

        # 以 orig 为参考，算新增/丢失
        ob, os_ = res["orig"]
        for key, m in (("gamma", "lab_gamma"), ("clahe", "clahe")):
            bb, bs = res[m]
            matched, added, lost, ac, lc = match_diff(ob, os_, bb, bs, thr=0.5)
            d = diffs[key]
            d["matched"] += matched
            d["added"] += added
            d["lost"] += lost
            d["added_conf"].extend(ac)
            d["lost_conf"].extend(lc)

        per_frame.append({
            "frame": frame_idx,
            "counts": {m: len(res[m][0]) for m in modes},
            "luma": {m: metas[m].get("luma_out", metas[m].get("luma")) for m in modes},
        })

        # 拼图：每 30 帧取一帧，最多 8 帧
        if len(montage) < 8 and (n_done % 30 == 0):
            row = []
            colors = {"orig": (80, 200, 80), "lab_gamma": (60, 160, 240),
                      "clahe": (220, 160, 80)}
            for m in modes:
                bb, bs = res[m]
                row.append(draw_boxes(frame, bb, bs, colors[m], scale=0.25))
            montage.append(np.hstack(row))

        n_done += 1
        frame_idx += 1
        if n_done % 20 == 0:
            el = time.time() - t0
            print(f"  [{n_done}/{args.frames}] {el:.0f}s "
                  f"({el/(n_done*3)*1000:.0f} ms/infer)", flush=True)

    cap.release()

    def agg(vals):
        a = np.asarray(vals, dtype=np.float32)
        if a.size == 0:
            return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
        return {"n": int(a.size), "mean": round(float(a.mean()), 4),
                "median": round(float(np.median(a)), 4),
                "p90": round(float(np.percentile(a, 90)), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}

    report = {
        "cam": args.cam, "video": str(video.relative_to(ROOT))
        if video.is_relative_to(ROOT) else str(video),
        "resolution": [W, H], "fps": fps,
        "stride": args.stride, "start_frame": start, "frames_requested": args.frames,
        "frames_done": stats["orig"]["frames"],
        "box_threshold": args.box_threshold,
        "conditions": {
            m: {
                "frames": stats[m]["frames"],
                "total_boxes": stats[m]["boxes"],
                "mean_boxes_per_frame": round(stats[m]["boxes"] / max(stats[m]["frames"], 1), 3),
                "detect_rate": round(stats[m]["det_frames"] / max(stats[m]["frames"], 1), 4),
                "conf": agg(stats[m]["conf"]),
            } for m in modes
        },
        "diff_vs_orig": {
            k: {
                "matched": v["matched"], "added": v["added"], "lost": v["lost"],
                "added_conf": agg(v["added_conf"]),
                "lost_conf": agg(v["lost_conf"]),
            } for k, v in diffs.items()
        },
        "per_frame": per_frame,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "指标为检测响应层面代理；新增/丢失以 IoU>0.5 与 orig 匹配判定；"
                "未做人工真值，抽检见拼图。",
    }

    jpath = out_dir / f"{args.cam}_ab.json"
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if montage:
        M = np.vstack(montage)
        mpath = out_dir / f"{args.cam}_montage.jpg"
        cv2.imwrite(str(mpath), M, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"[out] {mpath}", flush=True)

    print(f"[out] {jpath}")
    print(json.dumps({k: report["conditions"][k]["mean_boxes_per_frame"] for k in modes},
                     ensure_ascii=False))
    print("ALL_DONE")


if __name__ == "__main__":
    main()
