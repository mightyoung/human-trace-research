#!/bin/bash
# 4 路剩余摄像头推理（远程 192.168.11.16）——双 GPU 分两波
# 波1: 305(办公区1, cuda:0) + 316(行政走廊, cuda:1)
# 波2: 323(茶水间, cuda:0) + 343(电梯间, cuda:1)
# 用法（远程）: nohup bash run_wave4.sh > wave4.log 2>&1 < /dev/null &
cd /home/r4900g3/yhh/human-count || exit 1
PY=~/miniconda3/envs/rfdetr/bin/python
BASE="--model /share/data4/yhh/models/dino-model --prompt person --out-dir crosscam/out --stride 5 --box-threshold 0.30 --track-activation 0.10 --containment-threshold 0.85"

echo "[wave4] start wave1: 305 + 316  $(date '+%H:%M:%S')"
$PY infer_supervision.py --video crosscam/in/305.mp4 --device cuda:0 $BASE > crosscam/305.log 2>&1 &
P1=$!
$PY infer_supervision.py --video crosscam/in/316.mp4 --device cuda:1 $BASE > crosscam/316.log 2>&1 &
P2=$!
wait $P1 $P2

echo "[wave4] start wave2: 323 + 343  $(date '+%H:%M:%S')"
$PY infer_supervision.py --video crosscam/in/323.mp4 --device cuda:0 $BASE > crosscam/323.log 2>&1 &
P3=$!
$PY infer_supervision.py --video crosscam/in/343.mp4 --device cuda:1 $BASE > crosscam/343.log 2>&1 &
P4=$!
wait $P3 $P4

echo "[wave4] ALL_DONE $(date '+%H:%M:%S')"
echo "ALL_WAVE4_DONE $(date '+%H:%M:%S')" > crosscam/wave4_done.txt
