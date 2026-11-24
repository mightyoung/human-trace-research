#!/bin/bash
# R3.1 六路并行基线测试（双卡 3+3 同时推理）
# 目的：测 6 路「同时输入」时的吞吐/延迟/显存，对比历史分波数据（每卡 2 路时 latency≈191ms）
# 注意：GPU0 与服务器 asr/locate 服务共享（~12GB 已占，利用率空闲）
cd /home/r4900g3/yhh/human-count || exit 1
PY=~/miniconda3/envs/rfdetr/bin/python
BASE="--model /share/data4/yhh/models/dino-model --prompt person --stride 5 --box-threshold 0.30 --track-activation 0.10 --containment-threshold 0.85 --no-save-video"
OUT=/tmp/p6test
mkdir -p $OUT
rm -f $OUT/*.log $OUT/done.txt 2>/dev/null

# GPU 监控（每 10s 采样）
( while true; do echo "=== $(date '+%H:%M:%S')"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 10; done ) > $OUT/gpu_monitor.log 2>&1 &
MONPID=$!

echo "[p6test] start $(date '+%H:%M:%S')  max_frames=600  cuda0:305,323,343  cuda1:316,318,320"
PIDS=""
for CAM in 305 323 343; do
  $PY infer_supervision.py --video crosscam/in/$CAM.mp4 --device cuda:0 $BASE --max-frames 600 --out-dir $OUT > $OUT/$CAM.log 2>&1 &
  PIDS="$PIDS $!"
done
for CAM in 316 318 320; do
  $PY infer_supervision.py --video crosscam/in/$CAM.mp4 --device cuda:1 $BASE --max-frames 600 --out-dir $OUT > $OUT/$CAM.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $MONPID 2>/dev/null

echo "[p6test] ALL_DONE $(date '+%H:%M:%S')"
for CAM in 305 316 318 320 323 343; do
  echo "--- $CAM ---"
  grep -E "throughput_fps|latency_ms_mean|latency_ms_p90|frames_processed|elapsed_sec" $OUT/${CAM}_summary.json 2>/dev/null | head -6
done
echo "P6TEST_DONE" > $OUT/done.txt
