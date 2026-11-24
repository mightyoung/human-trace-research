#!/usr/bin/env bash
set -u
PY=${PY:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe}
OUT=output_crosscam_local/1012_osnet_crops_ac
mkdir -p "$OUT"
SEGS=(305_train_529 318_train_525 320_train_514 323_train_510 305_test_544 318_test_534 320_test_518 323_test_527)
for seg in "${SEGS[@]}"; do
  echo "[run ] $seg"
  "$PY" tools/encode_ac.py --seg "$seg" --device dml --tighten --topk 4 --out-dir "$OUT" \
    2>&1 | grep -v "Warning\|warnings.warn\|FutureWarning\|torch.load\|Successfully loaded\|following layers"
done
echo "[done] AC all"
