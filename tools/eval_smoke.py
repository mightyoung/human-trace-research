# -*- coding: utf-8 -*-
"""eval_smoke.py — 评估链路端到端冒烟（20 帧 val，预训练权重，cuda:1）。

验证：ref 加载 / yolo 推理 / rfdetr 推理 / 指标计算 / 对比图渲染 全链路无 bug。
用法：python eval_smoke.py --device cuda:1
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/home/r4900g3/yhh/human-count/bench/det")
from eval_det import D, evaluate, load_ref, render_compare, run_rfdetr, run_yolo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    os.makedirs(f"{D}/eval", exist_ok=True)

    files = sorted(glob.glob(f"{D}/raw/val/*.jpg"))[:20]
    ref = load_ref("val")
    print(f"[smoke] files={len(files)} ref_entries={len(ref)}", flush=True)
    assert len(ref) > 0, "ref empty!"

    # 1) YOLO26s 预训练（COCO，非本次训练权重）
    yo, el = run_yolo("/home/r4900g3/yhh/human-count/bench/det/yolo26s.pt",
                      files, args.device, 960)
    r1, _ = evaluate(yo, ref)
    print("[smoke] yolo26s(pre):", json.dumps(
        {k: r1[k] for k in ["frames", "precision@0.5", "recall@0.5", "f1@0.5",
                            "count_mae", "per_cam"]}, ensure_ascii=False), flush=True)
    print(f"[smoke] yolo time {el:.1f}s = {len(files)/el:.1f} img/s", flush=True)

    # 2) RF-DETR-small 预训练（官方 COCO 权重，直接加载缓存 .pth）
    ro, el2 = run_rfdetr("/home/r4900g3/.roboflow/models/rf-detr-small.pth",
                         files, args.device)
    r2, _ = evaluate(ro, ref)
    print("[smoke] rfdetr-sm(pre):", json.dumps(
        {k: r2[k] for k in ["frames", "precision@0.5", "recall@0.5", "f1@0.5",
                            "count_mae", "per_cam"]}, ensure_ascii=False), flush=True)
    print(f"[smoke] rfdetr time {el2:.1f}s = {len(files)/el2:.1f} img/s", flush=True)

    # 3) 对比图
    p = render_compare(files[:8], ref,
                       {"yolo26s_pre": yo, "rfdetr_sm_pre": ro},
                       f"{D}/eval/smoke_grid.jpg", 8)
    print(f"[smoke] grid -> {p}", flush=True)
    print("[smoke] ALL OK", flush=True)


if __name__ == "__main__":
    main()
