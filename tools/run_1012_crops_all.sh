#!/usr/bin/env bash
# 8 段逐裁特征（含质量三要素）编码；跳过已完成段
set -u
PY=${PY:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe}
OUT=output_crosscam_local/1012_osnet_crops
mkdir -p "$OUT"
SEGS=(305_train_529 318_train_525 320_train_514 323_train_510 305_test_544 318_test_534 320_test_518 323_test_527)
for seg in "${SEGS[@]}"; do
  if [ -f "$OUT/${seg}_crops.npz" ]; then echo "[skip] $seg"; continue; fi
  echo "[run ] $seg"
  "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" --device dml --save-crops --out-dir "$OUT" \
    2>&1 | grep -v "Warning\|warnings.warn\|FutureWarning\|torch.load\|Successfully loaded\|following layers"
done
echo "[done] crops all"
