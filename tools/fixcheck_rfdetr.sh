#!/bin/bash
# RF-DETR device 修复的快速验证：跑 1 epoch 小批量（GPU1），260s 后强制超时。
# 判定：日志中出现训练启动迹象（无 AttributeError）即视为修复通过。
B=/home/r4900g3/yhh/human-count/bench/det
D=/share/data4/yhh/data/human-count/det-bench
P=/home/r4900g3/miniconda3/envs/rfdetr/bin/python
cd "$B" || exit 1
echo "[fixcheck] start $(date '+%F %T')"
timeout 260 "$P" train_rfdetr.py small --epochs 1 --batch 2 --device cuda:1 --out-dir "$D/runs/_fixcheck"
echo "[fixcheck] exit=$? $(date '+%F %T')"
