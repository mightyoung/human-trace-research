#!/usr/bin/env bash
# extract_ffmpeg.sh — ffmpeg 并行抽帧（每 split 内 4 机位并发）
set -u
D=/share/data4/yhh/data/human-count/det-bench
V=/share/data4/yhh/data/human-count/video-1012

extract_split() {
  local split=$1 stride=$2 outdir=$3
  mkdir -p "$outdir"
  for cam in 305 318 320 323; do
    local vid
    vid=$(ls $V/${cam}_${split}_*.mp4 2>/dev/null | head -1)
    [ -z "$vid" ] && continue
    local stem
    stem=$(basename "$vid" .mp4)
    (
      ffmpeg -hide_banner -loglevel error -i "$vid" \
        -vf "select='not(mod(n\,$stride))'" -fps_mode passthrough \
        -q:v 2 "$outdir/${stem}_%06d.jpg"
      echo "[ff] $stem done $(date +%T) n=$(ls $outdir/${stem}_*.jpg 2>/dev/null | wc -l)"
    ) &
  done
  wait
}

echo "[ff] START $(date +%T)"
echo "=== TRAIN stride=75 ==="
extract_split train 75 "$D/raw/train"
echo "=== VAL stride=150 ==="
extract_split val 150 "$D/raw/val"
echo "[ff] TOTAL train=$(ls $D/raw/train/*.jpg 2>/dev/null | wc -l) val=$(ls $D/raw/val/*.jpg 2>/dev/null | wc -l)"
echo "[ff] DONE $(date +%T)"
