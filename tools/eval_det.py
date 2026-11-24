# -*- coding: utf-8 -*-
"""eval_det.py — 检测头实验评估：YOLO26 / RF-DETR vs GroundingDINO（参照）。

对比口径：
  - 参照（reference）：GroundingDINO 伪标签（labels_val.jsonl，生产口径 box_thr=0.3）
  - 指标：双向一对一贪心匹配（IoU≥0.5 / ≥0.75）的 precision/recall/F1、
    count MAE、逐帧数量一致率、GDINO 高置信子集 recall
  - 可视化：抽样帧三列对比图（GDINO | YOLO26 | RF-DETR）供人工核验
  - 吞吐：各模型 batch=1 fp16 的 fps（L20，与检测基准同口径）

用法：
  python eval_det.py --device cuda:0 [--limit 0]
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

D = "/share/data4/yhh/data/human-count/det-bench"
B = "/home/r4900g3/yhh/human-count/bench/det"

REF_CONF = 0.30   # 评估主口径（与 GDINO 伪标签一致）
COLLECT_CONF = 0.05


# ---------------- 工具 ----------------
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match_counts(pred, gt, thr=0.5):
    """greedy 一对一匹配，返回 (tp, fp, fn)。"""
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    ious = iou_matrix(pred, gt)
    order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
    used_p, used_g = set(), set()
    tp = 0
    for i, j in order:
        if ious[i, j] < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def load_ref(split="val"):
    ref = {}
    with open(f"{D}/pseudo/labels_{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ref[r["file"]] = (np.array(r["boxes"], np.float32).reshape(-1, 4),
                              np.array(r["scores"], np.float32))
    return ref


# ---------------- 各模型推理适配 ----------------
def run_yolo(weights, files, device, imgsz, chunk=64):
    from ultralytics import YOLO
    m = YOLO(weights)
    dev = int(str(device).split(":")[1]) if ":" in str(device) else 0
    out = {}
    t0 = time.time()
    # 分块推理（必须）：ultralytics 流式模式会把“传入列表”的图像内容整体转入显存，
    # 峰值随列表总张数线性增长（602 张 2560×1440 ≈ 26.6GB，实测 warmup 即 OOM），
    # 因此按 chunk 拆分调用；单块 64 张对应峰值 ≈ 3-4GB。
    for i in range(0, len(files), chunk):
        part = files[i:i + chunk]
        results = m.predict(part, imgsz=imgsz, conf=COLLECT_CONF, device=dev,
                            batch=16, verbose=False, stream=True)
        for fp, r in zip(part, results):
            b = r.boxes
            out[os.path.basename(fp)] = (
                b.xyxy.detach().cpu().numpy().astype(np.float32),
                b.conf.detach().cpu().numpy().astype(np.float32))
    el = time.time() - t0
    return out, el


def run_rfdetr(ckpt, files, device):
    import torch
    from rfdetr import RFDETR
    m = RFDETR.from_checkpoint(ckpt)
    try:  # 推理设备：显式设置（否则默认 cuda:0，可能干扰训练卡）
        m.model.device = torch.device(device)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] rfdetr device set failed: {e}", flush=True)
    out = {}
    t0 = time.time()
    for fp in files:
        im = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        dets = m.predict(im, threshold=COLLECT_CONF, include_source_image=False)
        b = np.asarray(dets.xyxy, np.float32).reshape(-1, 4)
        s = np.asarray(dets.confidence, np.float32)
        out[os.path.basename(fp)] = (b, s)
    el = time.time() - t0
    return out, el


# ---------------- 指标 ----------------
def evaluate(model_out, ref):
    rows = {"tp": 0, "fp": 0, "fn": 0, "tp75": 0, "fp75": 0, "fn75": 0,
            "n_pred": 0, "n_ref": 0, "n_frames": 0, "absdiff": 0, "exact": 0,
            "ref_hi_tp": 0, "ref_hi_n": 0, "per_cam": {}}
    detail = {}
    for fname, (pb_all, ps_all) in model_out.items():
        if fname not in ref:
            continue
        gb_all, gs_all = ref[fname]
        keep = ps_all >= REF_CONF
        pb, ps = pb_all[keep], ps_all[keep]
        gk = gb_all[gs_all >= REF_CONF]
        tp, fp, fn = match_counts(pb, gk, 0.5)
        tp75, fp75, fn75 = match_counts(pb, gk, 0.75)
        rows["tp"] += tp; rows["fp"] += fp; rows["fn"] += fn
        rows["tp75"] += tp75; rows["fp75"] += fp75; rows["fn75"] += fn75
        rows["n_pred"] += len(pb); rows["n_ref"] += len(gk)
        rows["n_frames"] += 1
        rows["absdiff"] += abs(len(pb) - len(gk))
        rows["exact"] += int(len(pb) == len(gk))
        # GDINO 高置信子集（score>=0.5）的覆盖
        hi = gs_all >= 0.5
        rows["ref_hi_n"] += int(hi.sum())
        if hi.sum() and len(pb):
            _, _, fn_hi = match_counts(pb, gb_all[hi], 0.5)
            rows["ref_hi_tp"] += int(hi.sum()) - fn_hi
        cam = fname.split("_")[0] if "_" in fname else "?"
        pc = rows["per_cam"].setdefault(cam, {"tp": 0, "fp": 0, "fn": 0,
                                              "absdiff": 0, "n": 0})
        pc["tp"] += tp; pc["fp"] += fp; pc["fn"] += fn
        pc["absdiff"] += abs(len(pb) - len(gk)); pc["n"] += 1
        detail[fname] = {"n_pred": len(pb), "n_ref": len(gk), "tp": tp}

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return round(p, 4), round(r, 4), round(2 * p * r / max(p + r, 1e-9), 4)

    p, r, f1 = prf(rows["tp"], rows["fp"], rows["fn"])
    p75, r75, f175 = prf(rows["tp75"], rows["fp75"], rows["fn75"])
    res = {
        "frames": rows["n_frames"],
        "precision@0.5": p, "recall@0.5": r, "f1@0.5": f1,
        "precision@0.75": p75, "recall@0.75": r75, "f1@0.75": f175,
        "count_mae": round(rows["absdiff"] / max(rows["n_frames"], 1), 3),
        "count_exact_ratio": round(rows["exact"] / max(rows["n_frames"], 1), 3),
        "ref_hi_recall": round(rows["ref_hi_tp"] / max(rows["ref_hi_n"], 1), 4),
        "n_pred_total": rows["n_pred"], "n_ref_total": rows["n_ref"],
        "per_cam": {k: {"precision": prf(v["tp"], v["fp"], v["fn"])[0],
                        "recall": prf(v["tp"], v["fp"], v["fn"])[1],
                        "count_mae": round(v["absdiff"] / max(v["n"], 1), 3)}
                    for k, v in rows["per_cam"].items()},
    }
    return res, detail


# ---------------- 可视化 ----------------
def render_compare(rows_list, ref, model_outs, out_path, n_frames=24):
    """rows_list: 待渲染帧文件列表；三列：GDINO | YOLO | RF-DETR。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = list(model_outs.keys())
    ncol = 1 + len(names)
    sample = rows_list[:n_frames] if len(rows_list) >= n_frames else rows_list
    tiles = []
    for fp in sample:
        im = cv2.imread(fp)
        h, w = im.shape[:2]
        scale = 720 / h
        base = cv2.resize(im, (int(w * scale), 720))
        row_tiles = []
        # 参照列
        gb, gs = ref.get(os.path.basename(fp), (np.zeros((0, 4)), np.zeros(0)))
        lk = gs >= REF_CONF
        row_tiles.append(_draw(base, gb[lk], f"GDINO(ref) n={int(lk.sum())}", (0, 0, 255)))
        for nm in names:
            mo = model_outs[nm].get(os.path.basename(fp))
            if mo is None:
                row_tiles.append(_draw(base, np.zeros((0, 4)), f"{nm} n=?", (255, 128, 0)))
                continue
            mb, ms = mo
            lk2 = ms >= REF_CONF
            row_tiles.append(_draw(base, mb[lk2], f"{nm} n={int(lk2.sum())}", (255, 128, 0)))
        tiles.append(np.hstack(row_tiles))
    grid = np.vstack(tiles)
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_path


def _draw(im, boxes, label, color):
    im = im.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(im, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return im


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="限制帧数（调试）；0=全量")
    ap.add_argument("--only", default="", help="只评指定模型（逗号分隔）")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="评估集：val（同时段）/ test（跨时段）")
    args = ap.parse_args()

    os.makedirs(f"{D}/eval", exist_ok=True)
    ref = load_ref(args.split)
    files = sorted(glob.glob(f"{D}/raw/{args.split}/*.jpg"))
    if args.limit:
        files = files[:args.limit]
    print(f"[eval] split={args.split} frames={len(files)} ref_files={len(ref)}", flush=True)

    # 模型清单（存在才评）
    model_map = {
        "yolo26s": ("yolo", f"{D}/runs/yolo26s/weights/best.pt", 960),
        "yolo26m": ("yolo", f"{D}/runs/yolo26m/weights/best.pt", 960),
        "yolo26x": ("yolo", f"{D}/runs/yolo26x/weights/best.pt", 960),
        "rfdetr_small": ("rfdetr", None, None),
        "rfdetr_medium": ("rfdetr", None, None),
        "rfdetr_large": ("rfdetr", None, None),
    }
    for v in ["small", "medium", "large"]:
        cands = sorted(glob.glob(f"{D}/runs/rfdetr_{v}/**/*best*.pth", recursive=True)
                       + glob.glob(f"{D}/runs/rfdetr_{v}/*.pth"))
        if cands:
            model_map[f"rfdetr_{v}"] = ("rfdetr", cands[-1], None)

    only = set(args.only.split(",")) if args.only else None
    model_outs, timings, summary = {}, {}, {}
    for name, (kind, path, imgsz) in model_map.items():
        if only and name not in only:
            continue
        if path is None or not os.path.exists(path):
            print(f"[eval] skip {name} (no weight at {path})", flush=True)
            continue
        print(f"[eval] running {name} ...", flush=True)
        try:
            if kind == "yolo":
                out, el = run_yolo(path, files, args.device, imgsz)
            else:
                out, el = run_rfdetr(path, files, args.device)
            model_outs[name] = out
            timings[name] = round(len(files) / el, 2)
            res, detail = evaluate(out, ref)
            res["fps_end2end"] = timings[name]
            res["weight"] = path
            summary[name] = res
            print(f"[eval] {name}: {json.dumps(res, ensure_ascii=False)}", flush=True)
        except Exception as e:
            import traceback
            print(f"[eval] {name} FAILED: {e}", flush=True)
            traceback.print_exc()
            summary[name] = {"error": str(e)[:300]}

    # 可视化（前 24 帧，跨机位抽样）
    if model_outs:
        sample = files[::max(1, len(files) // 24)][:24]
        p = render_compare(sample, ref, model_outs, f"{D}/eval/compare_grid_{args.split}.jpg", 24)
        print(f"[eval] grid -> {p}", flush=True)

    with open(f"{D}/eval/summary_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "ref_conf": REF_CONF, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    print("[eval] done", flush=True)


if __name__ == "__main__":
    main()
