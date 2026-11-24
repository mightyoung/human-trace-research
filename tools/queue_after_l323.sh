#!/usr/bin/env bash
# 排队器：等 l-seg 323 端到端出结果后，自动开始 305 掩膜去重验证，避免两个重任务抢 CPU。
# 最多等 5 小时；超时（或 l323 失败无产物）则放弃并报错，不会盲目开跑。
set -u
cd /d/human-count
EVAL=output_crosscam_local/selector_osnet_1012/l323_vs_s323.json
echo "[wait] 等 $EVAL $(date '+%F %H:%M:%S')"
for i in $(seq 1 300); do
  if [ -f "$EVAL" ]; then
    echo "[wait] l323 已出结果，启动 305 掩膜去重验证 $(date '+%F %H:%M:%S')"
    bash tools/run_mask_dedup_305.sh
    exit $?
  fi
  sleep 60
done
echo "[wait] 超时 5h 未等到 l323 结果，放弃（请检查 l323 任务）"
exit 1
