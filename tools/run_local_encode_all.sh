#!/bin/bash
# 本地嵌入链路：6 路全量（本地检测 jsonl -> 本地 DINOv3-B + DML 嵌入）。
# 前置：output_crosscam_local/detect/{cam}_detections.jsonl 已产出。
# 用法: bash tools/run_local_encode_all.sh
set -u
cd "$(dirname "$0")/.." || exit 1
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
ROOT="media-resource/multi-cemra/synced-10min"
DET="output_crosscam_local/detect"
OUT="output_crosscam_local/embed"
mkdir -p "$OUT" output_crosscam_local/logs

declare -A DIRMAP=(
  [305]="02_办公区1_305" [316]="01_行政走廊通道_316" [318]="03_办公区2_318"
  [320]="04_办公区球机_320" [323]="05_茶水间_323" [343]="06_电梯间_343"
)
for cam in 305 316 318 320 323 343; do
  dir="${DIRMAP[$cam]}"
  v=$(ls "$ROOT/$dir"/*.mp4 2>/dev/null | head -1)
  jsonl="$DET/${cam}_detections.jsonl"
  if [ -z "$v" ] || [ ! -f "$jsonl" ]; then echo "[skip] $cam: missing input"; continue; fi
  if [ -f "$OUT/reid_emb_${cam}_d3b_local.npz" ]; then echo "[skip] $cam: already done"; continue; fi
  echo "===== [$cam] encode ====="
  "$PY" tools/local_encode.py --jsonl "$jsonl" --video "$v" --prefix "v${cam}_" \
      --out "$OUT/reid_emb_${cam}_d3b_local.npz" \
      > "output_crosscam_local/logs/encode_${cam}.log" 2>&1
  echo "[$cam] EXIT $?"
  tail -1 "output_crosscam_local/logs/encode_${cam}.log"
done
echo "ALL_DONE"
