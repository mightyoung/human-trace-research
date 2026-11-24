#!/usr/bin/env bash
# 掩膜去重验证（305 一对段）：与生产配置唯一差别 = 检测时多加 --dedup-mask 0.9
#   305 的重复框最密集（87% 帧有 IoU0.2-0.5 重叠、21 个持续重叠轨迹对），是检验"删得对不对"的最佳场地
#
# 对照组（生产 yolo8dd，2 段诊断口径 min-eval 20，已实测）：
#   305: cov 1.000 / same 0.922 / fm 0.41% (3 对 / 736)
#
# 判据（先写死，避免事后找理由）：
#   cov  >= 0.990   （现 1.000，已封顶，只容忍 1pp）
#   same >= 0.912   （现 0.922，容忍 1pp）
#   fm   <= 0.55%   （现 0.41%＝3 对，最多容忍 +1 对）
#   三项全过 → 扩到 8 段做最终验收；任一项不过 → 弃用该规则
set -u
cd /d/human-count
export PYTHONPATH=D:/human-count/pylibs TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
M=models/Ultralytics--YOLO26/snapshots/master/yolo26s-seg.pt

OUT=output_crosscam_local
DET=$OUT/1012_yolom850
DD=$OUT/1012_yolom850_dd
AB=$OUT/1012_yolom850_dd_ab
EMB=$OUT/1012_yolom850_dd_emb
LOG=$OUT/_yolo_run
mkdir -p "$DET" "$DD" "$AB" "$EMB" "$LOG" tmp

echo "=== [1/4] detect (mask-dedup 0.9) $(date '+%F %H:%M:%S') ==="
( "$PY" tools/yolo_track_det.py --seg 305_test_544 --model $M --stride 5 --iou 0.5 \
      --dedup-mask 0.9 --out-dir "$DET" > "$LOG/md_305_test_544.log" 2>&1; \
  echo "[det] 305_test rc=$? $(date +%H:%M:%S)" ) &
( "$PY" tools/yolo_track_det.py --seg 305_train_529 --model $M --stride 10 --iou 0.5 \
      --dedup-mask 0.9 --out-dir "$DET" > "$LOG/md_305_train_529.log" 2>&1; \
  echo "[det] 305_train rc=$? $(date +%H:%M:%S)" ) &
wait

echo "=== [2/4] dedup(框 IoU0.5，保持与生产一致) $(date '+%F %H:%M:%S') ==="
"$PY" tools/dedup_detections.py --in-dir "$DET" --out-dir "$DD" --iou 0.5 \
    --segs 305_train_529,305_test_544

echo "=== [3/4] split a/b $(date '+%F %H:%M:%S') ==="
for s in 305_train_529 305_test_544; do
  "$PY" tools/make_ab_jsonl.py --in-jsonl "$DD/$s/${s}_detections.jsonl" \
      --out-prefix "$AB/$s" --min-frames 6
  echo "[split] $s a=$(wc -l < "$AB/${s}_a.jsonl") b=$(wc -l < "$AB/${s}_b.jsonl")"
done

echo "=== [4/4] encode $(date '+%F %H:%M:%S') ==="
( "$PY" tools/encode_1012_osnet_seq.py --seg 305_test_544 --device cpu \
      --ab-dir "$AB" --out-dir "$EMB" > tmp/enc_md_305_test.log 2>&1; \
  echo "[enc] test rc=$? $(date +%H:%M:%S)" ) &
( "$PY" tools/encode_1012_osnet_seq.py --seg 305_train_529 --device cpu \
      --ab-dir "$AB" --out-dir "$EMB" > tmp/enc_md_305_train.log 2>&1; \
  echo "[enc] train rc=$? $(date +%H:%M:%S)" ) &
wait

echo "=== [eval] $(date '+%F %H:%M:%S') ==="
"$PY" tools/eval_det_switch.py \
    --variant yolo8dd=$OUT/1012_yolo8_dd_emb \
    --variant m850=$EMB \
    --segs 305_train_529,305_test_544 --min-eval 20 \
    --cooc-map yolo8dd=$OUT/1012_yolo8_dd \
    --cooc-map m850=$DD \
    --out $OUT/selector_osnet_1012/maskdedup_305.json 2>&1 | grep -E "@full|Traceback|Error"

echo "=== DONE $(date '+%F %H:%M:%S') ==="
