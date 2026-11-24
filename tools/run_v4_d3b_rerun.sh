#!/bin/bash
# d3b v4 重跑排队（修正事故：首轮误用 raw 数据——远程脚本为旧版漏传）
# 等待 long35 检测完成（done.txt）后启动，避免 GPU 竞争
cd /home/r4900g3/yhh/human-count/crosscam_run || exit 1
OUT=/share/data4/yhh/data/human-count/inference-out-long35
W=0
while [ ! -f $OUT/done.txt ]; do
  sleep 60; W=$((W+60))
  if [ $W -ge 14400 ]; then echo "[d3b-rerun] wait-timeout ${W}s"; break; fi
done
sleep 30
export DET_VIEW=clean
export D3B_OUT=dinov3_scale_v4
export FUSION_DEV=cuda:1
PY=~/yhh/human-count/embed_venv/bin/python

echo "[d3b-rerun] start $(date '+%H:%M:%S') view=clean"
$PY dinov3_bridge_remote.py collect > $D3B_OUT/collect.log 2>&1
echo "[d3b-rerun] collect rc=$? $(date '+%H:%M:%S')"
$PY dinov3_bridge_remote.py solve   > $D3B_OUT/solve.log 2>&1
echo "[d3b-rerun] solve   rc=$? $(date '+%H:%M:%S')"
$PY dinov3_bridge_remote.py encode  > $D3B_OUT/encode.log 2>&1
echo "[d3b-rerun] encode  rc=$? $(date '+%H:%M:%S')"
echo "[d3b-rerun] ALL_DONE $(date '+%H:%M:%S')"
