#!/usr/bin/env bash
# U1：YOLO26s-seg 扩到 4 机位（8 段），统一产物目录 1012_yolo8*
# 参数口径（按交接第 7 节，事先写定）：
#   test 段 15min -> stride 5（4500 采样帧）
#   train 段 60min -> stride 10（9000 采样帧）
#   2 进程 × 7 线程，两条车道均衡
# 2026-10-19 追加：检测后再做同帧去重（IoU>=0.5 丢低分框），
#   消融显示只去掉 3.8% 框、fm 3.25%->0.75%、覆盖 0.577->0.568。后续一律用 dd 目录。
set -u
cd /d/human-count || exit 1
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
OUT=output_crosscam_local
DET=$OUT/1012_yolo8
DET_DD=$OUT/1012_yolo8_dd
AB=$OUT/1012_yolo8_dd_ab
EMB=$OUT/1012_yolo8_dd_emb
LOG=$OUT/_yolo_run
mkdir -p $LOG $DET $AB $EMB

echo "=== [1/3] detect  $(date '+%F %H:%M:%S') ==="
run_det () {  # seg stride
  # --iou 0.7 必须显式写：2026-10 起 yolo_track_det.py 的默认值改成了 0.5，
  # 而存档 1012_yolo8_dd 是「0.7 + 下面 dedup iou 0.5」跑出来的。不写就复现不了基线。
  "$PY" tools/yolo_track_det.py --seg "$1" --stride "$2" --iou 0.7 --force \
      --out-dir $DET > $LOG/det_$1.log 2>&1
  echo "[det] $1 rc=$? $(date '+%H:%M:%S')"
}

# 车道 A：50 + 37 + 25 + 19 ≈ 131 min
( run_det 320_train_514 10
  run_det 318_train_525 10
  run_det 320_test_518   5
  run_det 323_test_527   5 ) &
LA=$!
# 车道 B：50 + 37 + 25 + 19 ≈ 131 min
( run_det 305_train_529 10
  run_det 323_train_510 10
  run_det 305_test_544   5
  run_det 318_test_534   5 ) &
LB=$!
wait $LA; echo "[laneA] rc=$?"
wait $LB; echo "[laneB] rc=$?"
echo "=== detect done $(date '+%F %H:%M:%S') ==="
for s in 305_train_529 318_train_525 320_train_514 323_train_510 \
         305_test_544 318_test_534 320_test_518 323_test_527; do
  f=$DET/$s/${s}_detections.jsonl
  if [ -f "$f" ]; then echo "  $s rows=$(wc -l < $f)"; else echo "  $s MISSING"; fi
done

echo "=== [1.5/3] dedup  $(date '+%F %H:%M:%S') ==="
"$PY" tools/dedup_detections.py --in-dir $DET --out-dir $DET_DD --iou 0.5 | tail -3

echo "=== [2/3] split a/b  $(date '+%F %H:%M:%S') ==="
for seg in 305_train_529 318_train_525 320_train_514 323_train_510 \
           305_test_544 318_test_534 320_test_518 323_test_527; do
  "$PY" tools/make_ab_jsonl.py --in-jsonl $DET_DD/$seg/${seg}_detections.jsonl \
      --out-prefix $AB/$seg --min-frames 6 >> $LOG/split_u1.log 2>&1
  echo "[split] $seg a=$(wc -l < $AB/${seg}_a.jsonl) b=$(wc -l < $AB/${seg}_b.jsonl)"
done

echo "=== [3/3] encode  $(date '+%F %H:%M:%S') ==="
enc () {  # seg
  "$PY" tools/encode_1012_osnet_seq.py --seg "$1" --device cpu \
      --ab-dir $AB --out-dir $EMB > $LOG/enc_$1.log 2>&1
  echo "[enc] $1 rc=$? $(date '+%H:%M:%S')"
}
( enc 305_train_529; enc 320_train_514; enc 305_test_544; enc 320_test_518 ) &
EA=$!
( enc 318_train_525; enc 323_train_510; enc 318_test_534; enc 323_test_527 ) &
EB=$!
wait $EA; wait $EB
ls -l $EMB | tail -20
echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
