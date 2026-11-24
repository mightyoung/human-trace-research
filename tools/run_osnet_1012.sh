#!/usr/bin/env bash
# 1012 数据集的 OSNet 编码（a/b 分片，时间隔离口径）
# 用途：训练 OSNet 版选择器（train 段=上午 60min），在 test 段=下午 15min 评估
# 注意：1012 train 段与 6 路 synced-10min 时间重叠 → 不得用所产嵌入在 synced-10min 上评估
set -euo pipefail
cd "$(dirname "$0")/.."

PY_NATIVE="${PY_NATIVE:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/default/Scripts/python.exe}"
PY_GPU="${PY_GPU:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe}"
AB_DIR="output_crosscam_local/1012_ab"
EMB_DIR="output_crosscam_local/1012_osnet_emb"
mkdir -p "$AB_DIR" "$EMB_DIR"

SEGS=(
  305_train_529 318_train_525 320_train_514 323_train_510
  305_test_544  318_test_534  320_test_518  323_test_527
)

for seg in "${SEGS[@]}"; do
  src="output_crosscam/1012_final/${seg}.jsonl"
  vid="media-resource/1012/segments/${seg}.mp4"
  if [ ! -f "$src" ] || [ ! -f "$vid" ]; then
    echo "[skip] $seg (missing jsonl or video)"
    continue
  fi
  echo "===== $seg ====="
  if [ ! -f "$AB_DIR/${seg}_a.jsonl" ]; then
    "$PY_NATIVE" tools/make_ab_jsonl.py --in-jsonl "$src" --out-prefix "$AB_DIR/${seg}" --min-frames 6
  fi
  for ab in a b; do
    out="$EMB_DIR/${seg}_emb_${ab}.npz"
    if [ -f "$out" ]; then
      echo "[skip] $out exists"
      continue
    fi
    "$PY_GPU" tools/local_encode_osnet.py \
      --jsonl "$AB_DIR/${seg}_${ab}.jsonl" \
      --video "$vid" \
      --prefix "${seg}#" \
      --samples 8 \
      --out "$out" 2>&1 | grep -v "Warning\|warnings.warn\|FutureWarning\|torch.load" | tail -3
  done
done

echo "ALL_DONE"
