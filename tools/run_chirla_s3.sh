#!/usr/bin/env bash
# CHIRLA 加密采样版（stride 3 = 10 FPS）：把轨迹从 5–12 帧拉到 20–50 帧，
# 目的只有一个 —— 让 a/b 两半各有 10+ 帧，看 same（a/b 同簇率）能不能上到 0.90。
#
# 设计：seq_000 七机位同步，前 60% 帧作图库、后 40% 作查询（与 validation/chirla_time 同构，
#       唯一区别是采样密度 4×）。因此结果可与 stride12 版并排比较，唯一变量就是采样率。
#
# 用法：
#   bash tools/run_chirla_s3.sh det       # 只做检测+去重+切分+a/b+编码
#   bash tools/run_chirla_s3.sh tail      # 只做 GT + 扫描 + 评测
#   bash tools/run_chirla_s3.sh all
set -u
cd /d/human-count || exit 1
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
MODEL="models/Ultralytics--YOLO26/snapshots/master/yolo26s-seg.pt"
V="datasets/CHIRLA/videos/seq_000"
PKG="validation/chirla_s3"
DET=$PKG/det
DD=$PKG/det_dd
DTM=$PKG/det_time
AB=$PKG/ab_time
EMB=$PKG/emb_time
LOG=$PKG/log
STRIDE=3
mkdir -p "$DET" "$DD" "$DTM" "$AB" "$EMB" "$LOG"

CAMS="c1 c2 c3 c4 c5 c6 c7"
SEGS=""
for c in $CAMS; do SEGS="$SEGS ${c}_000"; done
SEGS=${SEGS# }
TSEGS=""
for c in $CAMS; do TSEGS="$TSEGS ${c}_000g ${c}_000q"; done
TSEGS=${TSEGS# }

video_of () { ls "$V"/camera_${1#c}_*.avi 2>/dev/null | head -1; }

one_seg () {   # seg = c<cam>_000
  local seg=$1 cam=${1%%_*} video
  video=$(video_of "$cam")
  [ -f "$video" ] || { echo "[$seg] 视频缺失"; return 1; }
  "$PY" tools/yolo_track_det_v2.py --seg "$seg" --video "$video" --model "$MODEL" \
      --stride $STRIDE --imgsz 1280 --iou 0.7 --out-dir "$DET" --tag default --force \
      > "$LOG/det_$seg.log" 2>&1
  echo "[det] $seg rc=$? $(date '+%H:%M:%S') $(grep -o 'boxes=[0-9]* tracks=[0-9]*' "$LOG/det_$seg.log" | tail -1)"
  "$PY" tools/dedup_detections.py --in-dir "$DET/default" --out-dir "$DD" --iou 0.5 \
      --segs "$seg" >> "$LOG/dedup.log" 2>&1
  echo "[dedup] $seg rc=$?"
}

enc_one () {   # seg = c<cam>_000g / c<cam>_000q
  local seg=$1 cam=${1%%_*} video
  video=$(video_of "$cam")
  "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" --device dml --ab-dir "$AB" \
      --video "$video" --out-dir "$EMB" >> "$LOG/enc.log" 2>&1
  echo "[enc] $seg rc=$? $(date '+%H:%M:%S')"
}

stage_det () {
  echo "=== [1/4] 检测 stride=$STRIDE（10 FPS，7 机位，2 lane）$(date '+%F %H:%M:%S') ==="
  set -- $SEGS
  half=$(( ($# + 1) / 2 ))
  L1=""; L2=""; i=0
  for s in $SEGS; do
    i=$((i+1)); if [ $i -le $half ]; then L1="$L1 $s"; else L2="$L2 $s"; fi
  done
  ( for s in $L1; do OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 one_seg "$s"; done ) & PA=$!
  ( for s in $L2; do OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 one_seg "$s"; done ) & PB=$!
  wait $PA; echo "[lane1] rc=$?"; wait $PB; echo "[lane2] rc=$?"
  echo "=== 检测完成 $(date '+%F %H:%M:%S') ==="

  echo "=== [2/4] 时间切分（前 60% 图库 / 后 40% 查询）$(date '+%H:%M:%S') ==="
  "$PY" tools/split_seg_by_time.py --in-dir "$DD" --out-dir "$DTM" \
      --segs "$(echo $SEGS | tr ' ' ',')" --split-frac 0.6 | tee "$LOG/split_time.log"

  echo "=== [3/4] a/b 切分 $(date '+%H:%M:%S') ==="
  for s in $TSEGS; do
    "$PY" tools/make_ab_jsonl.py --in-jsonl "$DTM/$s/${s}_detections.jsonl" \
        --out-prefix "$AB/$s" --min-frames 6 >> "$LOG/split_ab.log" 2>&1
    echo -n "$s:$(wc -l < "$AB/$s"_a.jsonl 2>/dev/null) "
  done; echo

  echo "=== [4/4] 编码（dml）$(date '+%H:%M:%S') ==="
  for s in $TSEGS; do enc_one "$s"; done
  echo "=== 编码完成 $(date '+%F %H:%M:%S') ==="
}

stage_tail () {
  echo "=== [5] 自动真值 $(date '+%H:%M:%S') ==="
  "$PY" tools/chirla_build.py --det "$DTM" --ann datasets/CHIRLA/annotations --out "$PKG" \
      --fps 10 --stride $STRIDE > "$LOG/gt.log" 2>&1
  tail -10 "$LOG/gt.log"

  GAL=""; QRY=""
  for c in $CAMS; do
    [ -f "$EMB/${c}_000g_emb_a.npz" ] && GAL="$GAL,${c}_000g"
    [ -f "$EMB/${c}_000q_emb_a.npz" ] && QRY="$QRY,${c}_000q"
  done
  GAL=${GAL#,}; QRY=${QRY#,}
  NQ=$(( $(wc -l < "$PKG/gt/gt_query_gt_FILLED.csv") - 1 ))
  echo "=== [6] 扫描：图库 [$GAL]｜查询 [$QRY]｜NQ=$NQ $(date '+%H:%M:%S') ==="
  "$PY" tools/d7_sweep.py --emb "$EMB" --det "$DTM" --train-segs "$GAL" --test-segs "$QRY" \
      --out "$PKG/results" --min-eval 0 > "$LOG/sweep.log" 2>&1
  head -6 "$LOG/sweep.log"

  echo "=== [7] 精度 + 判定 $(date '+%H:%M:%S') ==="
  "$PY" tools/eval_precision.py --deliv "$PKG" --run results --c-denom "$NQ" 2>&1 | head -8
  "$PY" tools/public_eval.py --pkg "$PKG" --emb-dir "$EMB" 2>&1 | tail -12
  echo "=== 全部完成 $(date '+%F %H:%M:%S') ==="
}

case "${1:-all}" in
  det)  stage_det ;;
  tail) stage_tail ;;
  all)  stage_det; stage_tail ;;
  *) echo "用法: $0 [det|tail|all]"; exit 2 ;;
esac
