#!/usr/bin/env bash
# 323 一对段：yolo26l-seg 端到端（与生产 s-seg 同采样、同 iou、同去重）
#   323_train_510 stride 10 (9000 采样帧) / 323_test_527 stride 5 (4500 采样帧)
#   两条车道并行，单进程 7 线程
# 目的：回答「l 能不能救 323 覆盖」，口径 = 2 段诊断（min-eval 20），与 s 的 0.588 可比
set -u
cd /d/human-count
export PYTHONPATH=D:/human-count/pylibs TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
M=models/Ultralytics--YOLO26/snapshots/master/yolo26l-seg.pt

OUT=output_crosscam_local
DET=$OUT/1012_yolol323
DD=$OUT/1012_yolol323_dd
AB=$OUT/1012_yolol323_dd_ab
EMB=$OUT/1012_yolol323_dd_emb
LOG=$OUT/_yolo_run
mkdir -p "$DET" "$DD" "$AB" "$EMB" "$LOG" tmp

echo "=== [1/4] detect $(date '+%F %H:%M:%S') ==="
( "$PY" tools/yolo_track_det.py --seg 323_test_527 --model $M --stride 5 --iou 0.5 \
      --out-dir "$DET" > "$LOG/yl_323_test_527.log" 2>&1; \
  echo "[det] 323_test rc=$? $(date +%H:%M:%S)" ) &
( "$PY" tools/yolo_track_det.py --seg 323_train_510 --model $M --stride 10 --iou 0.5 \
      --out-dir "$DET" > "$LOG/yl_323_train_510.log" 2>&1; \
  echo "[det] 323_train rc=$? $(date +%H:%M:%S)" ) &
wait

echo "=== [2/4] dedup $(date '+%F %H:%M:%S') ==="
"$PY" tools/dedup_detections.py --in-dir "$DET" --out-dir "$DD" --iou 0.5 \
    --segs 323_train_510,323_test_527

echo "=== [3/4] split a/b $(date '+%F %H:%M:%S') ==="
for s in 323_train_510 323_test_527; do
  "$PY" tools/make_ab_jsonl.py --in-jsonl "$DD/$s/${s}_detections.jsonl" \
      --out-prefix "$AB/$s" --min-frames 6
  echo "[split] $s a=$(wc -l < "$AB/${s}_a.jsonl") b=$(wc -l < "$AB/${s}_b.jsonl")"
done

echo "=== [4/4] encode $(date '+%F %H:%M:%S') ==="
( "$PY" tools/encode_1012_osnet_seq.py --seg 323_test_527 --device cpu \
      --ab-dir "$AB" --out-dir "$EMB" > tmp/enc_l_323_test.log 2>&1; \
  echo "[enc] 323_test rc=$? $(date +%H:%M:%S)" ) &
( "$PY" tools/encode_1012_osnet_seq.py --seg 323_train_510 --device cpu \
      --ab-dir "$AB" --out-dir "$EMB" > tmp/enc_l_323_train.log 2>&1; \
  echo "[enc] 323_train rc=$? $(date +%H:%M:%S)" ) &
wait

echo "=== [eval] $(date '+%F %H:%M:%S') ==="
"$PY" tools/eval_det_switch.py \
    --variant yolo8dd=$OUT/1012_yolo8_dd_emb \
    --variant yolol323=$EMB \
    --segs 323_train_510,323_test_527 --min-eval 20 \
    --cooc-map yolo8dd=$OUT/1012_yolo8_dd \
    --cooc-map yolol323=$DD \
    --out $OUT/selector_osnet_1012/l323_vs_s323.json 2>&1 | grep -E "@full|Error|Traceback"

echo "=== DONE $(date '+%F %H:%M:%S') ==="
