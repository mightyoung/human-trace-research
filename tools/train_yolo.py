# -*- coding: utf-8 -*-
"""train_yolo.py — YOLO26 单类 person 训练（ultralytics）。

用法：
  python train_yolo.py yolo26s --epochs 100 --imgsz 960 --batch 16
"""
import argparse
import json
import os

from ultralytics import YOLO

D = "/share/data4/yhh/data/human-count/det-bench"
B = "/home/r4900g3/yhh/human-count/bench/det"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["yolo26s", "yolo26m", "yolo26x"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    m = YOLO(os.path.join(B, f"{args.model}.pt"))
    results = m.train(
        data=f"{D}/yolo/dataset.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=8,
        seed=0,
        project=f"{D}/runs",
        name=args.model,
        exist_ok=True,
        patience=args.patience,
        val=True,
        plots=True,
    )
    # 汇总 best 指标
    try:
        rd = results.save_dir
        summary = {
            "model": args.model, "save_dir": str(rd),
            "best": str(rd / "weights" / "best.pt"),
            "fitness": float(results.fitness) if results.fitness is not None else None,
        }
        with open(f"{D}/runs/{args.model}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[yolo] summary:", summary, flush=True)
    except Exception as e:
        print("[yolo] summary failed:", e, flush=True)


if __name__ == "__main__":
    main()
