#!/bin/bash
# 把 human-count 样本数据迁移到 /share/data4/yhh/data/human-count/
# 策略：cp -a → 校验（文件数+字节数）→ rm 源 → ln -s 软链回原位（保持旧路径可用）
set -e
SRC=/home/r4900g3/yhh/human-count
DST=/share/data4/yhh/data/human-count

echo "=== before ==="
df -h /share/data4 | tail -1
mkdir -p "$DST"

move_link() {
  local src="$1" dst="$2"
  echo ""
  echo "--- $src -> $dst ---"
  if [ -L "$src" ]; then echo "SKIP: already a symlink"; return 0; fi
  if [ -e "$dst" ]; then echo "ERROR: destination already exists: $dst"; return 1; fi
  cp -a "$src" "$dst"
  local n1 n2 b1 b2
  n1=$(find "$src" -type f | wc -l); n2=$(find "$dst" -type f | wc -l)
  b1=$(du -sb "$src" | cut -f1); b2=$(du -sb "$dst" | cut -f1)
  echo "files: $n1 -> $n2 | bytes: $b1 -> $b2"
  if [ "$n1" != "$n2" ] || [ "$b1" != "$b2" ]; then
    echo "!!! MISMATCH - source kept, manual inspection needed"; return 1
  fi
  rm -rf "$src"
  ln -s "$dst" "$src"
  echo "OK: $src => $dst"
}

move_link "$SRC/crosscam/in"               "$DST/video-6way"
move_link "$SRC/crosscam/out"              "$DST/inference-out"
move_link "$SRC/crosscam_run"              "$DST/crosscam-run"
move_link "$SRC/crosscam/reflection_test"  "$DST/reflection-test"

echo ""
echo "=== after ==="
echo "[home/crosscam]"
ls -la "$SRC/crosscam/"
echo "[data dir]"
ls -la "$DST/"
echo "[df]"
df -h /share/data4 | tail -1
echo "=== MIGRATE_DONE ==="
