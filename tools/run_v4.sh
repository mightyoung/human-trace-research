#!/bin/bash
# R4.1 v4 重跑：clean 视图（cleanset_v1）下的 C-RADIOv4×Gemma2 桥接全链
# collect(W 训练) -> solve(W) -> encode(v4 嵌入)
cd /home/r4900g3/yhh/human-count/crosscam_run || exit 1
export DET_VIEW=clean
export SCALE_OUT=cradio_scale_v4
export FUSION_DEV=cuda:1
PY=~/yhh/human-count/embed_venv/bin/python
mkdir -p $SCALE_OUT

echo "[v4] start $(date '+%H:%M:%S') | DET_VIEW=$DET_VIEW SCALE_OUT=$SCALE_OUT DEV=$FUSION_DEV"
$PY cradio_scale_remote.py collect > $SCALE_OUT/collect.log 2>&1
echo "[v4] collect rc=$? $(date '+%H:%M:%S')"
$PY cradio_scale_remote.py solve   > $SCALE_OUT/solve.log 2>&1
echo "[v4] solve   rc=$? $(date '+%H:%M:%S')"
$PY cradio_scale_remote.py encode  > $SCALE_OUT/encode.log 2>&1
echo "[v4] encode  rc=$? $(date '+%H:%M:%S')"
echo "[v4] ALL_DONE $(date '+%H:%M:%S')"
