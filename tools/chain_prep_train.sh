#!/usr/bin/env bash
# chain_prep_train.sh — 自动衔接：等抽帧 → 打标 → 启动串行训练
set -u
B=/home/r4900g3/yhh/human-count/bench/det
P="$HOME/miniconda3/envs/rfdetr/bin/python"
cd "$B"

echo "[chain] wait extract ... $(date '+%F %T')"
for i in $(seq 1 180); do
  grep -q '\[ff\] DONE' extract.log 2>/dev/null && break
  sleep 20
done
if ! grep -q '\[ff\] DONE' extract.log 2>/dev/null; then
  echo "[chain] extract TIMEOUT, abort"; exit 1
fi
echo "[chain] extract done -> start labeling $(date '+%F %T')"
tail -3 extract.log

"$P" det_prep_dataset.py --skip-extract --device cuda:0 > label.log 2>&1
echo "[chain] labeling exit=$? $(date '+%F %T')"
tail -8 label.log

if [ ! -f /share/data4/yhh/data/human-count/det-bench/pseudo/stats.json ]; then
  echo "[chain] stats.json missing, abort"; exit 1
fi
nohup bash train_all.sh > train_all.log 2>&1 &
echo "[chain] train_all launched $(date '+%F %T')"
