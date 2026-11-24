#!/usr/bin/env bash
# 检测模型（无分割头）8 段流水线：yolo26s.pt + ByteTrack + 同帧去重
# 与 run_yolo_u1.sh 唯一的差别就是 --model 换成 yolo26s.pt（boxes only）。
# 参数口径保持一致：test stride 5 / train stride 10，2 进程 × 7 线程。
set -u
cd /d/human-count || exit 1
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
MODEL=models/Ultralytics--YOLO26/snapshots/master/yolo26s.pt
OUT=output_crosscam_local
DET=$OUT/1012_ydet
DET_DD=$OUT/1012_ydet_dd
AB=$OUT/1012_ydet_dd_ab
EMB=$OUT/1012_ydet_dd_emb
LOG=$OUT/_yolo_run
mkdir -p $LOG $AB $EMB

echo "=== [1/4] detect $(date '+%F %H:%M:%S') ==="
run_det () {
  "$PY" tools/yolo_track_det.py --seg "$1" --stride "$2" --force --iou 0.5 \
      --model $MODEL --out-dir $DET > $LOG/ydet_$1.log 2>&1
  echo "[det] $1 rc=$? $(date '+%H:%M:%S')"
}
( run_det 320_train_514 10
  run_det 318_train_525 10
  run_det 320_test_518   5
  run_det 323_test_527   5 ) &
( run_det 305_train_529 10
  run_det 323_train_510 10
  run_det 305_test_544   5
  run_det 318_test_534   5 ) &
wait
echo "=== detect done $(date '+%F %H:%M:%S') ==="

echo "=== [2/4] dedup $(date '+%H:%M:%S') ==="
"$PY" tools/dedup_detections.py --in-dir $DET --out-dir $DET_DD --iou 0.5 | tail -2

echo "=== [3/4] split + encode $(date '+%H:%M:%S') ==="
for seg in 305_train_529 318_train_525 320_train_514 323_train_510 \
           305_test_544 318_test_534 320_test_518 323_test_527; do
  "$PY" tools/make_ab_jsonl.py --in-jsonl $DET_DD/$seg/${seg}_detections.jsonl \
      --out-prefix $AB/$seg --min-frames 6 >> $LOG/split_ydet.log 2>&1
  echo "[split] $seg a=$(wc -l < $AB/${seg}_a.jsonl) b=$(wc -l < $AB/${seg}_b.jsonl)"
done
enc () { "$PY" tools/encode_1012_osnet_seq.py --seg "$1" --device cpu --ab-dir $AB --out-dir $EMB > $LOG/enc_ydet_$1.log 2>&1; echo "[enc] $1 rc=$?"; }
( enc 305_train_529; enc 318_train_525 ) &
( enc 320_train_514; enc 323_train_510 ) &
wait
( enc 305_test_544; enc 318_test_534 ) &
( enc 320_test_518; enc 323_test_527 ) &
wait

echo "=== [4/4] eval $(date '+%H:%M:%S') ==="
"$PY" tools/eval_det_switch.py \
    --variant baseline=$OUT/1012_osnet_emb \
    --variant yolo8dd=$OUT/1012_yolo8_dd_emb \
    --variant yolo8det=$EMB \
    --cooc-map yolo8dd=$OUT/1012_yolo8_dd \
    --cooc-map yolo8det=$DET_DD \
    --same-min 0.90 --out $OUT/selector_osnet_1012/det_vs_seg.json 2>&1 | grep -E "@full|avail"
echo "=== ALL DONE $(date '+%F %H:%M:%S') ==="
