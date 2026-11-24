#!/bin/bash
# RF-DETR 训练补跑（device-list bug 修复后）：small → large → medium
# 幂等：已有 best 权重则跳过；由 watch_backfill.sh 在主队列结束后自动调用。
B=/home/r4900g3/yhh/human-count/bench/det
D=/share/data4/yhh/data/human-count/det-bench
P=/home/r4900g3/miniconda3/envs/rfdetr/bin/python
cd "$B" || exit 1
for v in small large medium; do
  BEST=$(find "$D/runs/rfdetr_${v}" -name '*best*.pth' 2>/dev/null | head -1)
  if [ -n "$BEST" ]; then
    echo "[backfill] rfdetr_${v} 已有 best（$BEST），跳过"
    continue
  fi
  echo "===== [backfill] rfdetr_${v} START $(date '+%F %T') ====="
  "$P" train_rfdetr.py "${v}" --epochs 80 > "$B/train_rfdetr_${v}.retry.log" 2>&1
  echo "[backfill] rfdetr_${v} exit=$? $(date '+%F %T')"
done
echo "[backfill] ALL DONE $(date '+%F %T')"
