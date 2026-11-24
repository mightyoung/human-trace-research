#!/bin/bash
# 等待主训练队列（train_all.sh）完成（[6/6] 出 exit 行）后，自动执行 RF-DETR 补跑。
# 说明：若 [4/6] rfdetr_large / [6/6] rfdetr_medium 已成功，补跑脚本会按 best 权重跳过。
B=/home/r4900g3/yhh/human-count/bench/det
LOG=$B/watch_backfill.log
echo "[watch] start $(date '+%F %T')" >> "$LOG"
for i in $(seq 1 900); do
  if grep -q 'rfdetr_medium exit=' "$B/train_all.log" 2>/dev/null; then
    echo "[watch] queue done detected $(date '+%F %T')" >> "$LOG"
    break
  fi
  sleep 120
done
bash "$B/train_rfdetr_backfill.sh" >> "$LOG" 2>&1
echo "[watch] backfill finished $(date '+%F %T')" >> "$LOG"
