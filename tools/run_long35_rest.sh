#!/bin/bash
# 长窗检测补跑：305 / 323 / 343 三路（各 ~35min，stride=5）
# 口径与 316/318/320 + 1012 完全一致：GroundingDINO + ByteTrack, stride=5,
#   box-thr 0.30, activate 0.10, containment 0.85, no-save-video
# 目的：补齐六路 35min 同步检测 → 解锁六机位拓扑显著性检验
#
# 调度：当前 1012 批次占用 GPU0(320@1440p,318@1080p) / GPU1(323@1080p,305@1440p) 各 2 进程，
#       显存 48GB/卡仍有大量余量，故并发插入并按负载均衡：
#         GPU0 ← 305(1440p) + 323(1080p)
#         GPU1 ← 343(1440p)
#       启动间隔 60s，避免 3 进程同时抢模型加载。
cd /home/r4900g3/yhh/human-count || exit 1
PY=~/miniconda3/envs/rfdetr/bin/python
VID=/share/data4/yhh/data/human-count/video-long35
OUT=/share/data4/yhh/data/human-count/inference-out-long35
BASE="--model /share/data4/yhh/models/dino-model --prompt person --stride 5 --box-threshold 0.30 --track-activation 0.10 --containment-threshold 0.85 --no-save-video"
mkdir -p $OUT/305 $OUT/323 $OUT/343

run_one() {  # $1=cam  $2=cuda
  local cam=$1 dev=$2
  mkdir -p $OUT/$cam
  echo "[long35-rest $(date '+%H:%M:%S')] START $cam @ $dev"
  $PY infer_supervision.py --video $VID/${cam}_long35.mp4 --device $dev $BASE \
      --out-dir $OUT/$cam > $OUT/$cam/run.log 2>&1
  echo "[long35-rest $(date '+%H:%M:%S')] DONE $cam rc=$?"
}

echo "=== long35-rest ALL START $(date '+%H:%M:%S') ==="
run_one 305 cuda:0 & P1=$!
sleep 60
run_one 323 cuda:0 & P2=$!
sleep 60
run_one 343 cuda:1 & P3=$!

wait $P1 $P2 $P3
echo "=== long35-rest ALL DONE $(date '+%H:%M:%S') ==="
for c in 305 323 343; do
  echo "--- $c ---"
  grep -E "throughput_fps|frames_processed|elapsed_sec|unique_track_ids|count_max" \
      $OUT/${c}/${c}_long35_summary.json 2>/dev/null | head -6
  n=$(wc -l < $OUT/${c}/${c}_long35_detections.jsonl 2>/dev/null)
  echo "jsonl_lines=$n"
done
echo "LONG35_REST_DONE" > $OUT/done_rest.txt
