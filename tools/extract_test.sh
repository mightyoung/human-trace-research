#!/usr/bin/env bash
# extract_test.sh — test 段抽帧（stride=100，4 机位并行）
set -u
D=/share/data4/yhh/data/human-count/det-bench
V=/share/data4/yhh/data/human-count/video-1012
mkdir -p "$D/raw/test"
for cam in 305 318 320 323; do
  vid=$(ls $V/${cam}_test_*.mp4 2>/dev/null | head -1)
  [ -z "$vid" ] && continue
  stem=$(basename "$vid" .mp4)
  (
    ffmpeg -hide_banner -loglevel error -i "$vid" \
      -vf "select='not(mod(n\,100))'" -fps_mode passthrough \
      -q:v 2 "$D/raw/test/${stem}_%06d.jpg"
    echo "[ff] $stem done n=$(ls $D/raw/test/${stem}_*.jpg 2>/dev/null | wc -l)"
  ) &
done
wait
echo "[ff] TEST DONE total=$(ls $D/raw/test/*.jpg 2>/dev/null | wc -l)"
