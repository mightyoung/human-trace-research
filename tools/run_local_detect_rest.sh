#!/bin/bash
# 本地检测链路：串行跑其余 5 路（316/318/320/323/343）。
# 前置：305 全量已通过对账。用法: bash tools/run_local_detect_rest.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml/Scripts/python.exe"
ROOT="media-resource/multi-cemra/synced-10min"
OUT="output_crosscam_local/detect"
mkdir -p "$OUT" output_crosscam_local/logs

for spec in "316:01_行政走廊通道_316" "318:03_办公区2_318" "320:04_办公区球机_320" "323:05_茶水间_323" "343:06_电梯间_343"; do
  cam="${spec%%:*}"; dir="${spec##*:}"
  v=$(ls "$ROOT/$dir"/*.mp4 2>/dev/null | head -1)
  if [ -z "$v" ]; then echo "[skip] $cam: video not found"; continue; fi
  if [ -f "$OUT/${cam}_summary.json" ]; then echo "[skip] $cam: already done"; continue; fi
  echo "===== [$cam] $v ====="
  "$PY" tools/local_detect.py --video "$v" --cam "$cam" --out-dir "$OUT" \
      > "output_crosscam_local/logs/detect_${cam}.log" 2>&1
  echo "[$cam] EXIT $?"
  tail -2 "output_crosscam_local/logs/detect_${cam}.log"
done
echo "ALL_DONE"
