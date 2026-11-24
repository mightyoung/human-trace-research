#!/bin/bash
# YOLO26s 增量评估：val（同时段）→ test（跨时段），GPU1（避开 GPU0 训练）。
B=/home/r4900g3/yhh/human-count/bench/det
P=/home/r4900g3/miniconda3/envs/rfdetr/bin/python
cd "$B" || exit 1
"$P" eval_det.py --split val --only yolo26s --device cuda:1 > eval_yolo26s_val.log 2>&1
echo "[eval] val exit=$? $(date '+%F %T')"
"$P" eval_det.py --split test --only yolo26s --device cuda:1 > eval_yolo26s_test.log 2>&1
echo "[eval] test exit=$? $(date '+%F %T')"
echo "[eval] yolo26s evaluation done"
