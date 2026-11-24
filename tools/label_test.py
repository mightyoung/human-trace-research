# -*- coding: utf-8 -*-
"""label_test.py — 对 raw/test 抽帧执行 GroundingDINO 伪标签（评估参照，生产口径）。

输出：{OUT}/pseudo/labels_test.jsonl（与 labels_{train,val}.jsonl 同 schema）
口径：与 det_prep_dataset.py 完全一致（fp32 + box_thr=0.30 + text_thr=0.25
      + containment_nms=0.85 larger）

用法（远程 rfdetr env，建议 cuda:1 避开训练卡）：
  ~/miniconda3/envs/rfdetr/bin/python label_test.py --device cuda:1
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.insert(0, "/home/r4900g3/yhh/human-count")
from containment_nms import containment_nms  # noqa: E402

OUT = "/share/data4/yhh/data/human-count/det-bench"


def load_detector(model_dir, device):
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir).to(device).eval()
    return proc, model


@torch.no_grad()
def detect_one(proc, model, pil, device):
    inputs = proc(images=pil, text="person.", return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.30, text_threshold=0.25,
        target_sizes=[pil.size[::-1]])[0]
    boxes = res["boxes"].detach().cpu().numpy()
    scores = res["scores"].detach().cpu().numpy()
    if len(boxes) > 1:
        keep = containment_nms(boxes, scores, containment_threshold=0.85, keep="larger")
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores


def label_frames(files, proc, model, device, jsonl_path):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except Exception:
                    pass
    todo = [f for f in files if os.path.basename(f) not in done]
    print(f"[label_test] todo={len(todo)} done={len(done)}", flush=True)
    t0 = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as wf:
        for k, fp in enumerate(todo):
            im = cv2.imread(fp)
            h, w = im.shape[:2]
            pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            boxes, scores = detect_one(proc, model, pil, device)
            rec = {"file": os.path.basename(fp), "w": w, "h": h,
                   "boxes": boxes.tolist(), "scores": scores.tolist()}
            wf.write(json.dumps(rec) + "\n")
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                print(f"[label_test] {k+1}/{len(todo)}  {el:.0f}s  "
                      f"{(k+1)/el:.2f} fps", flush=True)
    print(f"[label_test] DONE {len(todo)} frames in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/share/data4/yhh/models/dino-model")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    torch.cuda.set_device(int(args.device.split(":")[1]))
    os.makedirs(f"{OUT}/pseudo", exist_ok=True)

    files = sorted(glob.glob(f"{OUT}/raw/test/*.jpg"))
    print(f"[label_test] files={len(files)} device={args.device}", flush=True)
    proc, model = load_detector(args.model, args.device)
    label_frames(files, proc, model, args.device, f"{OUT}/pseudo/labels_test.jsonl")


if __name__ == "__main__":
    main()
