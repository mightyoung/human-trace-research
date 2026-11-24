# -*- coding: utf-8 -*-
"""TOP-2 探针：开放词表"门"检测（GroundingDINO 复用于门/门框/开口）。

对六路 10min 视频各抽 8 帧 → GDINO 提示 "door. doorway. opening. entrance." →
输出框+分数 JSON（供与 C0 v7 门坐标对照）。远程运行。
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

CAMS = ["305", "316", "318", "320", "323", "343"]
VDIR = "/share/data4/yhh/data/human-count/video-6way"
OUT = "/share/data4/yhh/data/human-count/top2-door"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/share/data4/yhh/models/dino-model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default="door. doorway. opening. entrance.")
    ap.add_argument("--box-thr", type=float, default=0.25)
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(args.device).eval()

    res = {}
    for cam in cams:
        vp = f"{VDIR}/{cam}.mp4"
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = nfr / fps
        times = [min(dur - 3, 20 + i * max(10, (dur - 25) / 7)) for i in range(8)]
        frames = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if ok:
                frames.append((round(t, 1), f))
        cap.release()
        rows = []
        for t, f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            inputs = proc(images=rgb, text=args.prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = model(**inputs)
            r = proc.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=args.box_thr, text_threshold=0.2,
                target_sizes=[rgb.shape[:2]])[0]
            boxes = r["boxes"].detach().cpu().numpy().astype(float).tolist()
            scores = r["scores"].detach().cpu().numpy().astype(float).tolist()
            labels = r["labels"]
            H, W = f.shape[:2]
            rows.append(dict(t=t, W=W, H=H, boxes=boxes, scores=scores, labels=labels))
            # 标注样例（第一帧每机位）
            if len(rows) == 1 and boxes:
                vis = f.copy()
                for b, s in zip(boxes, scores):
                    x1, y1, x2, y2 = [int(v) for v in b]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(vis, f"{s:.2f}", (x1, max(30, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imwrite(f"{OUT}/{cam}_t{rows[0]['t']:.0f}.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"[{cam}] t={t}s n_box={len(boxes)}", flush=True)
        res[cam] = rows

    with open(f"{OUT}/door_probe{args.out_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(dict(prompt=args.prompt, box_thr=args.box_thr, cams=res), f,
                  ensure_ascii=False, indent=1)
    print("saved", f"{OUT}/door_probe{args.out_suffix}.json")


if __name__ == "__main__":
    main()
