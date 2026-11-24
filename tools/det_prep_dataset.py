# -*- coding: utf-8 -*-
"""det_prep_dataset.py — 检测头实验数据集构建（1012 train/val 段，GroundingDINO 伪标签）。

流程：
  阶段1 抽帧：video-1012 的 4 机位 train 段（每 3s 一帧 ≈1200/段 → 4800）
              + 4 机位 val 段（每 6s 一帧 ≈600/段 → 2400）
  阶段2 打标：GroundingDINO（生产口径 fp32 + box_thr=0.3 + text_thr=0.25
              + containment_nms=0.85 larger）→ 伪标签
  阶段3 导出：RF-DETR COCO（dataset/train|valid/_annotations.coco.json + 图片同目录）
              + YOLO（yolo/images hardlink + labels txt + dataset.yaml）

可重入：抽帧跳过已存在文件；打标跳过 labels.jsonl 已记录帧。

用法（远程 rfdetr env）：
  ~/miniconda3/envs/rfdetr/bin/python det_prep_dataset.py --device cuda:0
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

sys.path.insert(0, "/home/r4900g3/yhh/human-count")
from containment_nms import containment_nms  # noqa: E402

VIDEO_DIR = "/share/data4/yhh/data/human-count/video-1012"
OUT = "/share/data4/yhh/data/human-count/det-bench"
CAMS = ["305", "318", "320", "323"]
TRAIN_STRIDE = 75    # 25fps -> 每 3s
VAL_STRIDE = 150     # 每 6s

JPEG_Q = 88


def list_videos(split):
    vids = []
    for cam in CAMS:
        g = sorted(glob.glob(f"{VIDEO_DIR}/{cam}_{split}_*.mp4"))
        if g:
            vids.append((cam, g[0]))
    return vids


def extract_frames(vids, stride, out_dir):
    """grab 跳帧抽取。返回 (新增数, 总文件列表)。"""
    os.makedirs(out_dir, exist_ok=True)
    n_new = 0
    for cam, vpath in vids:
        stem = os.path.splitext(os.path.basename(vpath))[0]
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        i = 0
        kept = 0
        tries = 0
        while i < total and tries < 5:
            if i % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    tries += 1
                    i += 1
                    continue
                tries = 0
                fn = f"{stem}_{i:06d}.jpg"
                fp = os.path.join(out_dir, fn)
                if not os.path.exists(fp):
                    cv2.imwrite(fp, frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                    n_new += 1
                kept += 1
            else:
                cap.grab()
            i += 1
        cap.release()
        print(f"[extract] {stem}: frames={total} kept={kept}", flush=True)
    return n_new


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
    print(f"[label] todo={len(todo)} done={len(done)}", flush=True)
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
                print(f"[label] {k+1}/{len(todo)}  {el:.0f}s  "
                      f"{(k+1)/el:.2f} fps", flush=True)
    return len(todo)


def export_coco(records, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images, anns = [], []
    aid = 1
    for i, r in enumerate(records, 1):
        # 图片：hardlink 或 copy 到 out_dir
        src = os.path.join(img_dir, r["file"])
        dst = os.path.join(out_dir, r["file"])
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        images.append({"id": i, "file_name": r["file"],
                       "width": r["w"], "height": r["h"]})
        for (x1, y1, x2, y2) in r["boxes"]:
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1 or h < 1:
                continue
            anns.append({"id": aid, "image_id": i, "category_id": 1,
                         "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0})
            aid += 1
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "person", "supercategory": "none"}]}
    with open(os.path.join(out_dir, "_annotations.coco.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    return len(images), len(anns)


def export_yolo(records, img_dir, split):
    yimg = f"{OUT}/yolo/images/{split}"
    ylab = f"{OUT}/yolo/labels/{split}"
    os.makedirs(yimg, exist_ok=True)
    os.makedirs(ylab, exist_ok=True)
    for r in records:
        src = os.path.join(img_dir, r["file"])
        dst = os.path.join(yimg, r["file"])
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        lines = []
        for (x1, y1, x2, y2) in r["boxes"]:
            cx = (x1 + x2) / 2 / r["w"]
            cy = (y1 + y2) / 2 / r["h"]
            bw = (x2 - x1) / r["w"]
            bh = (y2 - y1) / r["h"]
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(ylab, r["file"].replace(".jpg", ".txt")), "w") as f:
            f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/share/data4/yhh/models/dino-model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-extract", action="store_true",
                    help="跳过抽帧阶段（配合 ffmpeg 抽帧脚本使用）")
    ap.add_argument("--skip-label", action="store_true")
    args = ap.parse_args()

    torch.cuda.set_device(int(args.device.split(":")[1]))
    os.makedirs(f"{OUT}/pseudo", exist_ok=True)

    # ---- 阶段1：抽帧 ----
    if not args.skip_extract:
        print("=== STAGE1 extract ===", flush=True)
        tr_vids = list_videos("train")
        va_vids = list_videos("val")
        print(f"[videos] train={tr_vids} val={va_vids}", flush=True)
        n1 = extract_frames(tr_vids, TRAIN_STRIDE, f"{OUT}/raw/train")
        n2 = extract_frames(va_vids, VAL_STRIDE, f"{OUT}/raw/val")
        print(f"[extract] new train={n1} val={n2}", flush=True)
    else:
        print("=== STAGE1 skipped (ffmpeg 抽帧) ===", flush=True)

    if args.skip_label:
        return

    # ---- 阶段2：打标 ----
    print("=== STAGE2 label ===", flush=True)
    proc, model = load_detector(args.model, args.device)
    for split in ["train", "val"]:
        files = sorted(glob.glob(f"{OUT}/raw/{split}/*.jpg"))
        label_frames(files, proc, model, args.device, f"{OUT}/pseudo/labels_{split}.jsonl")

    # ---- 阶段3：导出 ----
    print("=== STAGE3 export ===", flush=True)
    stats = {}
    for split, cname in [("train", "train"), ("val", "valid")]:
        recs = []
        with open(f"{OUT}/pseudo/labels_{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                recs.append(json.loads(line))
        recs = [r for r in recs if os.path.exists(os.path.join(f"{OUT}/raw/{split}", r["file"]))]
        ni, na = export_coco(recs, f"{OUT}/raw/{split}", f"{OUT}/dataset/{cname}")
        export_yolo(recs, f"{OUT}/raw/{split}", split)
        nbox = [len(r["boxes"]) for r in recs]
        stats[split] = {"images": ni, "boxes": na,
                        "box_per_img_mean": round(float(np.mean(nbox)), 2),
                        "empty_ratio": round(float(np.mean([b == 0 for b in nbox])), 3),
                        "score_mean": round(float(np.mean([s for r in recs for s in r["scores"]])) if na else 0, 3)}
    with open(f"{OUT}/pseudo/stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[stats] {stats}", flush=True)

    # YOLO dataset.yaml
    yaml = f"""path: {OUT}/yolo
train: images/train
val: images/val
names:
  0: person
"""
    with open(f"{OUT}/yolo/dataset.yaml", "w") as f:
        f.write(yaml)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
