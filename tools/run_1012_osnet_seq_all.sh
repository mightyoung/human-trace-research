#!/usr/bin/env bash
# 1012 八段 OSNet 顺序编码（跳过已完成的段）
set -u
PY=${PY:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe}
SEGS=(305_train_529 318_train_525 320_train_514 323_train_510 305_test_544 318_test_534 320_test_518 323_test_527)
for seg in "${SEGS[@]}"; do
  if [ -f "output_crosscam_local/1012_osnet_emb/${seg}_emb_a.npz" ] && [ -f "output_crosscam_local/1012_osnet_emb/${seg}_emb_b.npz" ]; then
    echo "[skip] $seg already done"
    continue
  fi
  echo "[run ] $seg"
  "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" --device dml 2>&1 | grep -v "Warning\|warnings.warn\|FutureWarning\|torch.load"
done
echo "[done] all segments"
