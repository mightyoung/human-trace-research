#!/usr/bin/env bash
# 全机位统一换 l-seg（方案 B）：8 段重跑检测 → 去重 → 分片 → 编码 → 8 段评估
#
# 背景：隔离验证显示 l-seg 在 323 上 0.588→0.913，但把 l 的 323 轨迹混进 s 的共享图库会
#   把 θ 从 1.85 抬到 2.25、整体覆盖 −7.0pp（docs/l323-mix-2026-10-19.md）。
#   方案 B = 图库内只保留一种检测器的特征，所以 8 段全部用 l 重跑。
#
# 复用：323 两段的 l-seg 检测已在 14:05 跑过（同模型/同 stride/同 iou），已复制进 1012_yolol_all，
#   检测阶段会自动跳过，省约 3h。
#
# 调度：xargs -P 2 动态队列（长段优先），两进程 × 7 线程。实测 0.83 f/s/进程。
# 断点：检测脚本对已有输出 skip；本脚本额外校验行数（train≥8800 / test≥4400），不足则删除重跑。
#
# 用法：
#   bash tools/run_yolo_l_all.sh          # 全流程（检测→后处理→评估），中断后重跑即续
#   bash tools/run_yolo_l_all.sh post     # 只跑去重/分片/编码/评估
set -u
cd /d/human-count
export PYTHONPATH=D:/human-count/pylibs TMPDIR=D:/human-count/tmp TEMP=D:/human-count/tmp
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7
PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
M=models/Ultralytics--YOLO26/snapshots/master/yolo26l-seg.pt
export PY M

OUT=output_crosscam_local
DET=$OUT/1012_yolol_all
DD=$OUT/1012_yolol_all_dd
AB=$OUT/1012_yolol_all_dd_ab
EMB=$OUT/1012_yolol_all_dd_emb
LOG=$OUT/_yolo_run
export DET LOG
mkdir -p "$DET" "$DD" "$AB" "$EMB" "$LOG"

TRAIN_SEGS="305_train_529 318_train_525 320_train_514 323_train_510"
TEST_SEGS="305_test_544 318_test_534 320_test_518 323_test_527"

min_lines () { case "$1" in *_train_*) echo 8800;; *) echo 4400;; esac; }
export -f min_lines

detect_one () {
  local seg=$1 stride=$2
  local f="$DET/$seg/${seg}_detections.jsonl"
  if [ -f "$f" ]; then
    local n; n=$(wc -l < "$f")
    if [ "$n" -ge "$(min_lines "$seg")" ]; then
      echo "[skip] $seg 已有 $n 行"
      return 0
    fi
    echo "[warn] $seg 只有 $n 行，判为未完成，删除重跑"
    rm -rf "$DET/$seg"
  fi
  local t0; t0=$(date +%s)
  "$PY" tools/yolo_track_det.py --seg "$seg" --model "$M" --stride "$stride" --iou 0.5 \
      --out-dir "$DET" > "$LOG/la_$seg.log" 2>&1
  local rc=$?
  local dt; dt=$(( $(date +%s) - t0 ))
  echo "[det] $seg rc=$rc ${dt}s $(wc -l < "$f" 2>/dev/null || echo 0) 行"
  return $rc
}
export -f detect_one

if [ "${1:-all}" = "all" ]; then
  echo "=== [1/4] detect l-seg 8 段（动态队列，2 并发）$(date '+%F %H:%M:%S') ==="
  # 长段优先，减少尾部串行；323 已在队列里但会立即 skip
  printf '%s %s\n' \
    305_train_529 10 318_train_525 10 320_train_514 10 323_train_510 10 \
    305_test_544 5 318_test_534 5 320_test_518 5 323_test_527 5 \
    | xargs -P 2 -n 2 bash -c 'detect_one "$0" "$1"'
  echo "=== detect 全部结束 $(date '+%F %H:%M:%S') ==="
fi

echo "=== [2/4] dedup $(date '+%F %H:%M:%S') ==="

# 门禁：检测被中断时（例如 kill 掉子进程）xargs 会正常返回，脚本会带着残缺产物
# 继续往下跑，最后 eval 出来一版没意义的结果。所以进后处理前必须逐段校验行数。
short=0
for s in $TRAIN_SEGS $TEST_SEGS; do
  n=$(wc -l < "$DET/$s/${s}_detections.jsonl" 2>/dev/null || echo 0)
  if [ "$n" -lt "$(min_lines "$s")" ]; then
    echo "[ABORT] $s 只有 $n 行（要求 $(min_lines "$s")），检测未完成，停止后处理"
    short=1
  fi
done
[ "$short" -ne 0 ] && exit 2

"$PY" tools/dedup_detections.py --in-dir "$DET" --out-dir "$DD" --iou 0.5 \
    --segs "$(echo $TRAIN_SEGS $TEST_SEGS | tr ' ' ',')"

echo "=== [3/4] split a/b $(date '+%F %H:%M:%S') ==="
for s in $TRAIN_SEGS $TEST_SEGS; do
  "$PY" tools/make_ab_jsonl.py --in-jsonl "$DD/$s/${s}_detections.jsonl" \
      --out-prefix "$AB/$s" --min-frames 6
  echo "[split] $s a=$(wc -l < "$AB/${s}_a.jsonl") b=$(wc -l < "$AB/${s}_b.jsonl")"
done

echo "=== [4/4] encode $(date '+%F %H:%M:%S') ==="
# encode 脚本遇到已存在的 npz 会 skip。中断重跑时这些 npz 可能是上一轮的残缺结果
# （实测：305_train 只有 11 条轨迹、另 3 段是 0 条的空壳），会静默污染评估。
# 所以编码前先删掉比对应 ab 文件更旧的 npz。
enc () {
  local seg=$1
  local ref="$AB/${seg}_a.jsonl"
  if [ -f "$ref" ]; then
    for side in a b; do
      local p="$EMB/${seg}_emb_${side}.npz"
      if [ -f "$p" ] && [ "$p" -ot "$ref" ]; then
        echo "[stale] $seg 的 ${side} 侧 npz 早于分片文件，删除重编码"
        rm -f "$p"
      fi
    done
  fi
  "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" --device cpu --ab-dir "$AB" --out-dir "$EMB" \
          > "tmp/enc_la_$seg.log" 2>&1; echo "[enc] $seg rc=$? $(date +%H:%M:%S)"
}
( enc 305_train_529; enc 305_test_544; enc 318_train_525; enc 318_test_534 ) &
( enc 320_train_514; enc 320_test_518; enc 323_train_510; enc 323_test_527 ) &
wait

echo "=== [eval] l 全量 vs 生产 s $(date '+%F %H:%M:%S') ==="
"$PY" tools/eval_det_switch.py \
    --variant prod=output_crosscam_local/1012_yolo8_dd_emb \
    --variant lall=$EMB \
    --cluster-mode camera --thr-in 0.15 --thr-out 0.25 \
    --cooc-map prod=output_crosscam_local/1012_yolo8_dd \
    --cooc-map lall=$DD \
    --out output_crosscam_local/selector_osnet_1012/l_all_vs_prod.json 2>&1 | grep -E "@full|Traceback|Error"

echo "=== DONE $(date '+%F %H:%M:%S') ==="
