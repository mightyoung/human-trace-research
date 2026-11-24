#!/usr/bin/env bash
# 检测器切换试点：305 一对段（60min 训练 / 15min 测试）
# 链路：YOLO26s-seg + ByteTrack -> 轨迹 jsonl -> a/b 分片 -> OSNet 编码 -> 与 DINO 基线同口径评估
set -u
cd /d/human-count || exit 1
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
OUT=output_crosscam_local
LOG=$OUT/_yolo_run
mkdir -p $LOG $OUT/1012_yolo_ab

echo "=== [1/4] detect $(date '+%H:%M:%S') ==="
"$PY" tools/yolo_track_det.py --seg 305_train_529 --stride 10 --force > $LOG/det_train.log 2>&1 &
P1=$!
"$PY" tools/yolo_track_det.py --seg 305_test_544  --stride 10 --force > $LOG/det_test.log 2>&1 &
P2=$!
wait $P1; echo "[detect] train rc=$?"
wait $P2; echo "[detect] test  rc=$?"
tail -2 $LOG/det_train.log; tail -2 $LOG/det_test.log

echo "=== [2/4] split a/b $(date '+%H:%M:%S') ==="
for seg in 305_train_529 305_test_544; do
  "$PY" tools/make_ab_jsonl.py --in-jsonl $OUT/1012_yolo/$seg/${seg}_detections.jsonl \
      --out-prefix $OUT/1012_yolo_ab/$seg --min-frames 6 >> $LOG/split.log 2>&1
  echo "[split] $seg rows_a=$(wc -l < $OUT/1012_yolo_ab/${seg}_a.jsonl) rows_b=$(wc -l < $OUT/1012_yolo_ab/${seg}_b.jsonl)"
done

echo "=== [3/4] encode $(date '+%H:%M:%S') ==="
"$PY" tools/encode_1012_osnet_seq.py --seg 305_test_544 --device cpu \
    --ab-dir $OUT/1012_yolo_ab --out-dir $OUT/1012_yolo_emb > $LOG/enc_test.log 2>&1
echo "[enc] test rc=$?"; tail -2 $LOG/enc_test.log
"$PY" tools/encode_1012_osnet_seq.py --seg 305_train_529 --device cpu \
    --ab-dir $OUT/1012_yolo_ab --out-dir $OUT/1012_yolo_emb > $LOG/enc_train.log 2>&1
echo "[enc] train rc=$?"; tail -2 $LOG/enc_train.log

echo "=== [4/4] eval $(date '+%H:%M:%S') ==="
"$PY" tools/eval_det_switch.py \
    --variant baseline=$OUT/1012_osnet_emb \
    --variant yolo26s=$OUT/1012_yolo_emb \
    --segs 305_train_529,305_test_544 \
    --cooc-map yolo26s=$OUT/1012_yolo --min-eval 60 > $LOG/eval.log 2>&1
echo "[eval] rc=$?"; cat $LOG/eval.log
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
