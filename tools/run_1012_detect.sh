#!/bin/bash
# 1012 切段双 GPU 并发检测调度（4 并发 = 2 进程/GPU）
# 口径与 long35 完全一致：GroundingDINO + ByteTrack, stride=5, box-thr 0.30, activate 0.10, containment 0.85
# 输入: /share/data4/yhh/data/human-count/video-1012/*.mp4
# 输出: /share/data4/yhh/data/human-count/inference-out-1012/{seg}/{seg}_detections.jsonl
cd /home/r4900g3/yhh/human-count || exit 1
PY=~/miniconda3/envs/rfdetr/bin/python
VID=/share/data4/yhh/data/human-count/video-1012
OUT=/share/data4/yhh/data/human-count/inference-out-1012
mkdir -p $OUT

COMMON="--model /share/data4/yhh/models/dino-model --prompt person --stride 5 --box-threshold 0.30 --track-activation 0.10 --containment-threshold 0.85 --no-save-video"

run_one() {  # $1=seg  $2=cuda
  local seg=$1 dev=$2
  mkdir -p $OUT/$seg
  echo "[$(date '+%H:%M:%S')] START $seg @ $dev"
  $PY infer_supervision.py --video $VID/${seg}.mp4 --device $dev $COMMON --out-dir $OUT/$seg \
      > $OUT/$seg/run.log 2>&1
  echo "[$(date '+%H:%M:%S')] DONE $seg rc=$?"
}

# ---- 4 队列（贪心均衡：60min 段权重 4、15min 段权重 1；每队列权重 6）----
q1() { run_one 320_train_514 cuda:0; run_one 323_val_538 cuda:0; run_one 305_test_544 cuda:0; }
q2() { run_one 318_train_525 cuda:0; run_one 320_val_519 cuda:0; run_one 323_test_527 cuda:0; }
q3() { run_one 323_train_510 cuda:1; run_one 318_val_525 cuda:1; run_one 305_val_517 cuda:1; }
q4() { run_one 305_train_529 cuda:1; run_one 320_test_518 cuda:1; run_one 318_test_534 cuda:1; }

echo "=== 1012 detect ALL START $(date '+%H:%M:%S') ==="
q1 & P1=$!
sleep 45   # 错峰启动，避免 4 进程同时抢模型加载
q2 & P2=$!
sleep 45
q3 & P3=$!
sleep 45
q4 & P4=$!

wait $P1 $P2 $P3 $P4
echo "=== 1012 detect ALL DONE $(date '+%H:%M:%S') ==="
for d in $OUT/*/; do
  seg=$(basename $d)
  if [ -f "$d/${seg}_detections.jsonl" ]; then
    n=$(wc -l < "$d/${seg}_detections.jsonl")
    echo "OK $seg lines=$n"
  else
    echo "MISS $seg"
  fi
done
