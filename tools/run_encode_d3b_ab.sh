#!/bin/bash
# d3b a/b 嵌入批量编码（EVAL-3c 长流实验数据底座）
# 任务：1012-train 4 路（选择器构建）+ long35 4 路（跨 session 长流）
# 输出：output_crosscam_local/d3b_ab/{1012_train,long35}/*.npz (+ _a/_b)
set -u
cd /d/human-count || exit 1
PY=C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe
OUT=output_crosscam_local/d3b_ab
LOG=output_crosscam_local/logs
mkdir -p "$OUT/1012_train" "$OUT/long35" "$LOG"

run() {  # $1=tag $2=jsonl $3=video $4=prefix $5=out
  local tag="$1" jsonl="$2" video="$3" prefix="$4" out="$5"
  if [ -f "$out" ] && [ -f "${out%.npz}_a.npz" ]; then
    echo "[$tag] skip (exists)"; return 0
  fi
  echo "===== [$tag] encode ====="
  "$PY" tools/local_encode.py --jsonl "$jsonl" --video "$video" \
      --prefix "$prefix" --out "$out" --ab --samples 8 --device dml \
      > "$LOG/enc3c_$tag.log" 2>&1
  local rc=$?
  echo "[$tag] EXIT $rc"
  tail -2 "$LOG/enc3c_$tag.log"
  return $rc
}

echo "===== 1012-train (selector base) $(date '+%H:%M:%S') ====="
run 305tr output_crosscam/1012_final/305_train_529.jsonl media-resource/1012/segments/305_train_529.mp4 v305_ "$OUT/1012_train/305_train_529_emb_d3b.npz"
run 318tr output_crosscam/1012_final/318_train_525.jsonl media-resource/1012/segments/318_train_525.mp4 v318_ "$OUT/1012_train/318_train_525_emb_d3b.npz"
run 320tr output_crosscam/1012_final/320_train_514.jsonl media-resource/1012/segments/320_train_514.mp4 v320_ "$OUT/1012_train/320_train_514_emb_d3b.npz"
run 323tr output_crosscam/1012_final/323_train_510.jsonl media-resource/1012/segments/323_train_510.mp4 v323_ "$OUT/1012_train/323_train_510_emb_d3b.npz"

echo "===== long35 (stream) $(date '+%H:%M:%S') ====="
run 305L output_crosscam/long35/305_long35_detections.jsonl media-resource/multi-cemra/long-35min/305_long35.mp4 v305_ "$OUT/long35/305_emb_long35_d3b.npz"
run 318L output_crosscam/long35/318_long35_detections.jsonl media-resource/multi-cemra/long-35min/318_long35.mp4 v318_ "$OUT/long35/318_emb_long35_d3b.npz"
run 320L output_crosscam/long35/320_long35_detections.jsonl media-resource/multi-cemra/long-35min/320_long35.mp4 v320_ "$OUT/long35/320_emb_long35_d3b.npz"
run 323L output_crosscam/long35/323_long35_detections.jsonl media-resource/multi-cemra/long-35min/323_long35.mp4 v323_ "$OUT/long35/323_emb_long35_d3b.npz"

echo "===== ALL_DONE $(date '+%H:%M:%S') ====="
ls -la "$OUT/1012_train" "$OUT/long35"
