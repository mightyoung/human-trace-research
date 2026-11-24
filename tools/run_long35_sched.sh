#!/bin/bash
# 长窗检测调度（3 路 × 35 分钟，stride=5）
# 目的：多目标轨迹追踪研究——比 10 分钟窗长 3.5 倍的长时段数据
# 调度：318@GPU0 立即（GPU0 与他方服务共享、避开 d3b 占用的 GPU1）
#       d3b 完成后 → 316@GPU1；318 完成后 → 320@GPU0
cd /home/r4900g3/yhh/human-count || exit 1
PY=~/miniconda3/envs/rfdetr/bin/python
VID=/share/data4/yhh/data/human-count/video-long35
OUT=/share/data4/yhh/data/human-count/inference-out-long35
BASE="--model /share/data4/yhh/models/dino-model --prompt person --stride 5 --box-threshold 0.30 --track-activation 0.10 --containment-threshold 0.85 --no-save-video"
mkdir -p $OUT/316 $OUT/318 $OUT/320

# GPU 监控（每 15s）
( while true; do echo "=== $(date '+%H:%M:%S')"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 15; done ) > $OUT/gpu_monitor.log 2>&1 &
MONPID=$!

echo "[long35] start $(date '+%H:%M:%S')  318@cuda0 先行"

# 1) 318 立即启动（GPU0，轻量 1080p）
$PY infer_supervision.py --video $VID/318_long35.mp4 --device cuda:0 $BASE \
    --out-dir $OUT/318 > $OUT/318/run.log 2>&1 &
P318=$!

# 2) 等 d3b（dinov3_bridge_remote.py）结束（最多等 90 分钟兜底）
WAITED=0
while pgrep -f "dinov3_bridge_remote.py" > /dev/null 2>&1; do
  sleep 30; WAITED=$((WAITED+30))
  if [ $WAITED -ge 5400 ]; then echo "[long35] d3b-wait timeout"; break; fi
done
echo "[long35] d3b done (waited ${WAITED}s) $(date '+%H:%M:%S') -> launch 316@cuda1"
sleep 10

# 3) 316 启动（GPU1）
$PY infer_supervision.py --video $VID/316_long35.mp4 --device cuda:1 $BASE \
    --out-dir $OUT/316 > $OUT/316/run.log 2>&1 &
P316=$!

# 4) 318 完成后 → 320（GPU0）
wait $P318; RC318=$?
echo "[long35] 318 exit rc=$RC318 $(date '+%H:%M:%S') -> launch 320@cuda0"
$PY infer_supervision.py --video $VID/320_long35.mp4 --device cuda:0 $BASE \
    --out-dir $OUT/320 > $OUT/320/run.log 2>&1 &
P320=$!

wait $P316 $P320
kill $MONPID 2>/dev/null
echo "[long35] ALL_DONE $(date '+%H:%M:%S')"
for c in 316 318 320; do
  echo "--- $c ---"
  grep -E "throughput_fps|latency_ms_mean|latency_ms_p90|frames_processed|elapsed_sec|unique_track_ids|count_max" \
      $OUT/${c}/${c}_long35_summary.json 2>/dev/null | head -9
done
echo "LONG35_DONE" > $OUT/done.txt
