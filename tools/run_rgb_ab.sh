#!/usr/bin/env bash
# RGB 口径 A/B：只换编码颜色通道，轨迹（ab jsonl）与真值完全不动。
#   基线 = output_crosscam_local/1012_yolo8_dd_emb（BGR 直喂，历史口径）
#   实验 = output_crosscam_local/1012_yolo8_dd_emb_rgb（--rgb）
#
# 注意：--device 必须用 cpu。存档 1012_yolo8_dd_emb 是 run_yolo_u1.sh 用 cpu 编的，
#       换 dml 会把设备差异混进颜色差异里。
# 预计：8 段编码 ~20–40 min（CPU），d7_sweep 42 配置 ~5–15 min。
set -u
cd /d/human-count || exit 1
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
OUT=output_crosscam_local
AB=$OUT/1012_yolo8_dd_ab
DET=$OUT/1012_yolo8_dd
EMB_RGB=$OUT/1012_yolo8_dd_emb_rgb
PKG=tmp/rgb_pkg
LOG=$OUT/_rgb_run
SEGS="305_train_529 318_train_525 320_train_514 323_train_510 305_test_544 318_test_534 320_test_518 323_test_527"
mkdir -p "$LOG" "$EMB_RGB" "$PKG"

echo "=== [1/3] 编码 RGB（device cpu）  $(date '+%F %H:%M:%S') ==="
for seg in $SEGS; do
  "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" --device cpu --rgb \
      --ab-dir "$AB" --out-dir "$EMB_RGB" > "$LOG/enc_$seg.log" 2>&1
  echo "[enc-rgb] $seg rc=$? $(date '+%H:%M:%S')"
done

for seg in $SEGS; do
  for h in a b; do
    f="$EMB_RGB/${seg}_emb_$h.npz"
    if [ ! -f "$f" ]; then echo "MISSING $f  -> 中止"; exit 1; fi
  done
done
echo "  npz 齐全：$(ls "$EMB_RGB" | wc -l) 个"

echo "=== [2/3] 扫 42 配置（只换 emb）  $(date '+%F %H:%M:%S') ==="
"$PY" tools/d7_sweep.py --emb "$EMB_RGB" --det "$DET" --out "$PKG/results"

echo "=== [3/3] 评估 + 配对比较  $(date '+%F %H:%M:%S') ==="
"$PY" tools/eval_precision.py --deliv "$PKG" --gt-dir deliverable/data-request-v2/gt \
    --run results --c-denom 229
"$PY" tools/compare_cfgs.py --deliv "$PKG" --gt-dir deliverable/data-request-v2/gt \
    --run results --c-denom 229 | tail -20
"$PY" tools/ab_emb_report.py \
    --run-a deliverable/data-request-v2/results --run-b "$PKG/results" \
    --gt deliverable/data-request-v2/gt/gt_query_gt_FILLED.csv \
    --cfg A_baseline --label-a BGR --label-b RGB --c-denom 229 \
    --out "$PKG/ab_report.md"

echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
echo "报告：$PKG/ab_report.md；逐配置：$PKG/results/precision.csv"
