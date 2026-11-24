#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C 方案诊断/全量：dump **Containment-NMS 之前**的 GroundingDINO 原始框。

why:
  `{cam}_detections.jsonl` 只保留 NMS 后的框（305 被删 13.7%）。要判断「两人重合时
  检测器到底出了几个框、合并框分数多高、小框是被 NMS 删的还是根本没检出」，
  必须在 NMS 前落盘。这是 C（检测层治本）的唯一入口。

模式:
  --mode probe   只跑指定帧（诊断，秒级）
  --mode full    全量 stride 跑，输出与 {cam}_detections.jsonl 同构 + 原始框

输出（每行）:
  frame, raw_count, boxes/scores（threshold 生效后的全部框，未 NMS）, kept（NMS 保留下标),
  suppressed（被删框下标 + 吞掉它的框下标 + 包含度）

用法:
  python tools/dino_raw_dump.py --video vid/305_long35.mp4 --cam 305 \
      --frames 4380,4405,4435,4460,4490 --out out/305_probe.jsonl --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def containment_of(inner, outer) -> float:
    """inner 被 outer 包含的面积比（inter / area(inner)）。"""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih / a) if a > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--model", default="/share/data4/yhh/models/dino-model")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--box-threshold", type=float, default=0.10,
                    help="dump 阈值；生产是 0.30，诊断用 0.10 看被截断的框")
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--containment-threshold", type=float, default=0.85)
    ap.add_argument("--mode", choices=["probe", "full"], default="probe")
    ap.add_argument("--frames", default="", help="probe 模式：逗号分隔帧号")
    ap.add_argument("--range", default="", help="start:end（含 start 不含 end），按 stride 采样")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fp16", action="store_true", help="autocast fp16（仅测速用，口径会变）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="out 已存在则从最后一行 frame+stride 续跑（append）")
    ap.add_argument("--fsync-every", type=int, default=50,
                    help="每 N 帧 fsync（NFS 上缓冲不落地，中断会丢；0=关闭）")
    ap.add_argument("--save-video", default="", help="可选：画框输出 mp4（probe 目检用）")
    a = ap.parse_args()

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    from containment_nms import containment_nms

    dev = a.device
    if dev.startswith("cuda"):
        idx = int(dev.split(":")[1]) if ":" in dev else 0
        torch.cuda.set_device(idx)
        print(f"[env] torch={torch.__version__} cuda={torch.version.cuda} "
              f"dev={dev} name={torch.cuda.get_device_name(idx)}", flush=True)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(a.model).to(dev).eval()
    if a.fp16:
        model = model.half()
    print(f"[model] loaded {time.time()-t0:.1f}s fp16={a.fp16}", flush=True)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[video] {W}x{H} @{fps:.2f} frames={total}", flush=True)

    if a.mode == "probe":
        want = sorted({int(x) for x in a.frames.split(",") if x.strip()})
    elif a.range:
        s, e = (int(v) for v in a.range.split(":"))
        want = list(range(s, min(e, total), a.stride))
    else:
        want = list(range(0, total, a.stride))

    # 续跑：out 已有内容 → 从最后一行 frame 之后接着来
    append = False
    out = Path(a.out)
    if a.resume and out.exists() and out.stat().st_size > 0:
        last = -1
        with out.open("rb") as fr:
            try:
                fr.seek(-4096, 2)
            except OSError:
                fr.seek(0)
            for ln in fr.read().splitlines():
                if ln.strip():
                    try:
                        last = int(json.loads(ln)["frame"])
                    except Exception:
                        pass
        if last >= 0:
            want = [f for f in want if f > last]
            append = True
            print(f"[resume] 从 frame>{last} 继续，剩 {len(want)} 帧", flush=True)
    if a.max_frames:
        want = want[:a.max_frames]
    print(f"[plan] {len(want)} 帧", flush=True)

    writer = None
    if a.save_video:
        writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps / a.stride), (W, H))

    out.parent.mkdir(parents=True, exist_ok=True)
    lat = []
    n_written = 0

    def iter_frames():
        """full 模式顺序读，probe 模式才 seek。

        ★ 逐帧 `cap.set(POS_FRAMES)` 每帧都要从最近关键帧解码过来，2560×1440 实测
        0.84s/帧（其中推理只占 0.19s），比顺序读慢 4 倍以上。顺序读要先把前面
        的帧解掉（resume 到 f13440 约 9 分钟），之后按 5 帧/s 走。
        """
        if a.mode == "full":
            want_set = set(want)
            fi = 0
            got = 0
            while got < len(want):
                ok_, fr_ = cap.read()
                if not ok_:
                    return
                if fi in want_set:
                    got += 1
                    yield fi, fr_
                fi += 1
            return
        for f_ in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_)
            ok_, fr_ = cap.read()
            if not ok_:
                print(f"[warn] frame {f_} read fail", flush=True)
                continue
            yield f_, fr_

    with out.open("a" if append else "w", encoding="utf-8") as fo:
        for frame_idx, frame in iter_frames():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            t1 = time.time()
            inputs = processor(images=pil, text=a.prompt + ".",
                               return_tensors="pt").to(dev)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16):
                    outputs = model(**inputs)
            res = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=a.box_threshold,
                text_threshold=a.text_threshold,
                target_sizes=[pil.size[::-1]],
            )[0]
            if dev != "cpu":
                torch.cuda.synchronize()
            lat.append(time.time() - t1)

            boxes = res["boxes"].detach().float().cpu().numpy()
            scores = res["scores"].detach().float().cpu().numpy()

            # NMS 复算（仅用于诊断：谁被删、被谁吞）
            kept: list[int] = []
            supp: list[list] = []
            if len(boxes) > 1:
                kept = [int(i) for i in containment_nms(
                    boxes, scores, containment_threshold=a.containment_threshold,
                    keep="larger")]
                ks = set(kept)
                for i in range(len(boxes)):
                    if i in ks:
                        continue
                    best, bestc = -1, 0.0
                    for j in kept:
                        c = containment_of(boxes[i], boxes[j])
                        if c > bestc:
                            best, bestc = j, c
                    supp.append([int(i), int(best), round(float(bestc), 4)])
            else:
                kept = list(range(len(boxes)))

            rec = {
                "frame": frame_idx,
                "raw_count": int(len(boxes)),
                "kept": kept,
                "suppressed": supp,
                "boxes": [[round(float(v), 1) for v in b] for b in boxes],
                "scores": [round(float(v), 4) for v in scores],
                "latency_ms": round(lat[-1] * 1000, 1),
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # NFS 上块缓冲不落地（实测跑 459 帧后文件仍 0 字节）→ 每帧 flush，
            # 每 fsync_every 帧 fsync，保证 kill/掉线后可 --resume 续跑。
            fo.flush()
            if a.fsync_every and n_written % a.fsync_every == 0:
                os.fsync(fo.fileno())
            n_written += 1

            if writer is not None:
                vis = frame.copy()
                for i, b in enumerate(boxes):
                    x1, y1, x2, y2 = (int(v) for v in b)
                    is_kept = i in set(kept)
                    col = (0, 200, 0) if is_kept else (60, 60, 220)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                    cv2.putText(vis, f"{i}:{scores[i]:.2f}", (x1 + 3, max(14, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                writer.write(vis)

            if n_written % 20 == 0 or a.mode == "probe":
                print(f"[{frame_idx}] raw={len(boxes)} kept={len(kept)} "
                      f"supp={len(supp)} {lat[-1]*1000:.0f}ms", flush=True)

    if writer is not None:
        writer.release()
    if lat:
        arr = np.array(lat) * 1000
        print(f"[done] {n_written} 帧  均值 {arr.mean():.0f}ms  "
              f"P50 {np.percentile(arr,50):.0f}  P90 {np.percentile(arr,90):.0f}  "
              f"吞吐 {1000/arr.mean():.2f} fps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
