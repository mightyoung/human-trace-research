#!/usr/bin/env bash
# CHIRLA 端到端复现：检测 → 去重 → a/b 切分 → 编码 → 自动真值 → 42 配置扫描 → 判定
#
# 段名约定 `c<cam>_<seq>`：下游用 seg.split("_")[0] 取机位，机位必须放第一段。
# 采样：30 FPS，每 12 帧取 1 → 2.5 FPS（VALIDATION §4.1 第 1 步）。
# 查询侧只用 camera_1..4：seq_001/002 的 camera_5/6/7 标注为空（已核，5 个文件零框）。
#
# 用法：
#   bash tools/run_chirla.sh                 # 全量 15 段
#   SEGS="c1_000" bash tools/run_chirla.sh   # 只跑单段（冒烟）
set -u
cd /d/human-count || exit 1
export PYTHONPATH=D:/human-count/pylibs
export TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
MODEL="models/Ultralytics--YOLO26/snapshots/master/yolo26s-seg.pt"
V="datasets/CHIRLA/videos"
PKG="validation/chirla"
DET=$PKG/det
DD=$PKG/det_dd
AB=$PKG/ab
EMB=$PKG/emb
LOG=$PKG/log
mkdir -p "$DET" "$DD" "$AB" "$EMB" "$LOG"

seg_video () {   # seg -> 视频路径
  local seg=$1 cam=${1%%_*} seq=${1##*_}
  local d=$V/seq_$seq
  local f
  f=$(ls "$d"/camera_${cam#c}_*.avi 2>/dev/null | head -1)
  echo "$f"
}

one_seg () {     # seg
  local seg=$1 video
  video=$(seg_video "$seg")
  if [ -z "$video" ] || [ ! -f "$video" ]; then echo "[$seg] 视频缺失"; return 1; fi
  "$PY" tools/yolo_track_det_v2.py --seg "$seg" --video "$video" --model "$MODEL" \
      --stride 12 --imgsz 1280 --iou 0.7 --out-dir "$DET" --tag default \
      > "$LOG/det_$seg.log" 2>&1
  echo "[det] $seg rc=$?"
  "$PY" tools/dedup_detections.py --in-dir "$DET/default" --out-dir "$DD" --iou 0.5 \
      --segs "$seg" >> "$LOG/dedup.log" 2>&1
  "$PY" tools/make_ab_jsonl.py --in-jsonl "$DD/$seg/${seg}_detections.jsonl" \
      --out-prefix "$AB/$seg" --min-frames 6 >> "$LOG/split.log" 2>&1
  local na nb
  na=$(wc -l < "$AB/${seg}_a.jsonl" 2>/dev/null || echo 0)
  nb=$(wc -l < "$AB/${seg}_b.jsonl" 2>/dev/null || echo 0)
  echo "[split] $seg a=$na b=$nb"
  "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" --device dml --ab-dir "$AB" \
      --video "$video" --out-dir "$EMB" > "$LOG/enc_$seg.log" 2>&1
  echo "[enc] $seg rc=$? $(date '+%H:%M:%S')"
}

ALL="c1_000 c2_000 c3_000 c4_000 c5_000 c6_000 c7_000 c1_001 c2_001 c3_001 c4_001 c1_002 c2_002 c3_002 c4_002"
SEGS="${SEGS:-$ALL}"

echo "=== 待跑段: $SEGS  $(date '+%F %H:%M:%S') ==="
lane () { for s in $@; do one_seg "$s"; done; }
set -- $SEGS
half=$(( ($# + 1) / 2 ))
L1=""; L2=""; i=0
for s in $SEGS; do
  i=$((i+1))
  if [ $i -le $half ]; then L1="$L1 $s"; else L2="$L2 $s"; fi
done
( OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 lane $L1 ) & PA=$!
( OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 lane $L2 ) & PB=$!
wait $PA; echo "[lane1] rc=$?"
wait $PB; echo "[lane2] rc=$?"
echo "=== 检测/编码完成 $(date '+%F %H:%M:%S') ==="

if [ -z "${SKIP_TAIL:-}" ]; then
  "$PY" tools/chirla_build.py --det "$DD" --ann datasets/CHIRLA/annotations --out "$PKG" \
      > "$LOG/gt.log" 2>&1
  tail -8 "$LOG/gt.log"
  NQ=$(($(wc -l < "$PKG/gt/gt_query_gt_FILLED.csv") - 1))
  ME=$(( NQ / 2 )); [ "$ME" -lt 20 ] && ME=20
  # 只把确实产出 emb 的段交给 d7_sweep（缺段会让 np.load 直接报错）
  GAL=""; QRY=""
  for s in c1_000 c2_000 c3_000 c4_000 c5_000 c6_000 c7_000; do
    [ -f "$EMB/${s}_emb_a.npz" ] && GAL="$GAL,$s"
  done
  for s in c1_001 c2_001 c3_001 c4_001 c1_002 c2_002 c3_002 c4_002; do
    [ -f "$EMB/${s}_emb_a.npz" ] && QRY="$QRY,$s"
  done
  GAL=${GAL#,}; QRY=${QRY#,}
  echo "=== d7_sweep：查询 $NQ 条，min-eval=$ME（VALIDATION §4.1：可评估数的 50%） ==="
  "$PY" tools/d7_sweep.py --emb "$EMB" --det "$DD" --train-segs "$GAL" --test-segs "$QRY" \
      --out "$PKG/results" --min-eval "$ME" > "$LOG/sweep.log" 2>&1
  tail -4 "$LOG/sweep.log"
  "$PY" tools/eval_precision.py --deliv "$PKG" --run results --c-denom "$NQ" > "$LOG/precision.log" 2>&1
  head -8 "$LOG/precision.log"
  "$PY" tools/public_eval.py --pkg "$PKG" >> "$LOG/public_eval.log" 2>&1
  tail -12 "$PKG/VERDICTS.md"
fi
