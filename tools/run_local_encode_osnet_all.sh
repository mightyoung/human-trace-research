#!/usr/bin/env bash
# 本机 6 路 OSNet 轨迹编码（专用 ReID 对照/替换候选）
# 输入：output_crosscam_local/detect/{cam}_detections.jsonl（本地链路检测产物）
# 输出：output_crosscam_local/embed_osnet/reid_emb_{cam}_osnet_local.npz（512d）
set -euo pipefail
cd "$(dirname "$0")/.."

P="${PY:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/default/Scripts/python.exe}"
OUT_DIR="output_crosscam_local/embed_osnet"
mkdir -p "$OUT_DIR"

declare -A DIR=(
  [305]="02_办公区1_305"
  [316]="01_行政走廊通道_316"
  [318]="03_办公区2_318"
  [320]="04_办公区球机_320"
  [323]="05_茶水间_323"
  [343]="06_电梯间_343"
)

for cam in 305 316 318 320 323 343; do
  vid="media-resource/multi-cemra/synced-10min/${DIR[$cam]}/${cam}_085500-090500.mp4"
  if [ ! -f "$vid" ]; then
    echo "[skip] $cam video not found: $vid"
    continue
  fi
  echo "===== cam $cam ====="
  "$P" tools/local_encode_osnet.py \
    --jsonl "output_crosscam_local/detect/${cam}_detections.jsonl" \
    --video "$vid" \
    --prefix "v${cam}_" \
    --out "${OUT_DIR}/reid_emb_${cam}_osnet_local.npz"
done

echo "ALL_DONE"
