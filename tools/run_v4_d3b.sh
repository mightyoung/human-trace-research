#!/bin/bash
# R4.1 v4 追加：dinov3_bridge（备选桥）clean 视图重跑
cd /home/r4900g3/yhh/human-count/crosscam_run || exit 1
export DET_VIEW=clean
export D3B_OUT=dinov3_scale_v4
export FUSION_DEV=cuda:1
PY=~/yhh/human-count/embed_venv/bin/python
mkdir -p $D3B_OUT

echo "[v4-d3b] start $(date '+%H:%M:%S')"
$PY dinov3_bridge_remote.py collect > $D3B_OUT/collect.log 2>&1
echo "[v4-d3b] collect rc=$? $(date '+%H:%M:%S')"
$PY dinov3_bridge_remote.py solve   > $D3B_OUT/solve.log 2>&1
echo "[v4-d3b] solve   rc=$? $(date '+%H:%M:%S')"
$PY dinov3_bridge_remote.py encode  > $D3B_OUT/encode.log 2>&1
echo "[v4-d3b] encode  rc=$? $(date '+%H:%M:%S')"
echo "[v4-d3b] ALL_DONE $(date '+%H:%M:%S')"
