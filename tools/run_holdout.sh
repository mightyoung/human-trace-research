#!/usr/bin/env bash
# ============================================================================
# 留出复验运行脚本（冻结版）—— 对应预注册 §2.3 的处理链
#
# 目的：把"运行参数"也锁进冻结清单。只锁文件哈希不够——漏传 --iou 0.7
#       不会改变任何哈希，却会悄悄改掉结果（脚本默认值是 0.5）。
#
# 本脚本所有参数写死，不接受环境变量覆盖；任何一项要改，必须改脚本并重算冻结哈希。
#
# 用法：
#   bash tools/run_holdout.sh --print-plan          # 只打印将执行的命令，不跑
#   bash tools/run_holdout.sh --smoke               # 冒烟：1 个片段 × 40 采样帧
#   bash tools/run_holdout.sh --stage detect        # 只跑检测
#   bash tools/run_holdout.sh --stage all           # 全链（detect→dedup→split→encode）
#
# 退出码：0 成功；2 冻结哈希不一致；3 参数/前置条件不满足
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- 冻结参数
# 以下六项来自预注册 §2.3 / §7，改动即视为偏离（§6），必须记录并重算冻结哈希。
readonly DET_IOU=0.7            # 检测 NMS；脚本默认 0.5，必须显式传
readonly DEDUP_IOU=0.5          # 追踪后同帧去重
readonly STRIDE_GAL=10          # 图库 25fps → 2.5fps
readonly STRIDE_QRY=5           # 查询 25fps → 5fps
readonly TRACKER_CFG="tools/bytetrack_1012.yaml"
readonly IMGSZ=1280
readonly CONF=0.10
readonly MIN_FRAMES=6
readonly ENC_DEVICE="cpu"

# ---------------------------------------------------------------- 片段表
# 格式：段名 机位 窗口 卷文件 卷内起点(秒) 时长(秒) 采样步长
# 段名必须以机位号开头：下游（d7_sweep.py 第 67 行等）用 seg.split("_")[0] 取机位，
#   机位感知聚类依赖它。2026-10-24 由 H_gal_*/H_qry_* 改为 {机位}_hgal*/{机位}_hqry*。
# 数值直接取自 docs/preregistration_holdout_2026-10-24.json 的 segments.*.per_camera
# 305 图库跨 529→517：529 卷末帧与 517 卷首帧是同一时刻（11:24:52），故 p2 从 0s 起
# 320 图库跨 540→543：543 卷首帧（11:26:12）早于 540 卷末帧（11:26:16），故 p2 从 4s 起避开重叠
PIECES=(
  "305_hgal_p1  305 gallery 0100000252900000000.mp4 4680 2293 $STRIDE_GAL"
  "305_hgal_p2  305 gallery 0100000251700000000.mp4    0 1307 $STRIDE_GAL"
  "318_hgal     318 gallery 0100000252500000000.mp4 8590 3600 $STRIDE_GAL"
  "320_hgal_p1  320 gallery 0100000254000000000.mp4 1500 2359 $STRIDE_GAL"
  "320_hgal_p2  320 gallery 0100000254300000000.mp4    4 1200 $STRIDE_GAL"
  "323_hgal     323 gallery 0100000251300000000.mp4    0 3600 $STRIDE_GAL"
  "305_hqry     305 query   0100000251700000000.mp4 5580  900 $STRIDE_QRY"
  "318_hqry     318 query   0100000252500000000.mp4 16450 900 $STRIDE_QRY"
  "320_hqry     320 query   0100000251900000000.mp4 1080  900 $STRIDE_QRY"
  "323_hqry     323 query   0100000252700000000.mp4    0  900 $STRIDE_QRY"
)

# ---------------------------------------------------------------- 运行时
# 与主实验同一套环境（run_yolo_u1.sh 第 11–14 行）：ultralytics 在 pylibs 下，
# 不设 PYTHONPATH 会 ModuleNotFoundError。
# 注意：Git Bash 的 ROOT 是 /d/... 形式，Windows 的 Python 不认，须转成 D:/...
if command -v cygpath >/dev/null 2>&1; then
  ROOT_WIN="$(cygpath -m "$ROOT")"
else
  ROOT_WIN="$ROOT"
fi
export PYTHONPATH="$ROOT_WIN/pylibs${PYTHONPATH:+;$PYTHONPATH}"
export TMPDIR="$ROOT_WIN/tmp" TEMP="$ROOT_WIN/tmp"
export OMP_NUM_THREADS=7 MKL_NUM_THREADS=7

PY="${HOLDOUT_PY:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe}"
OUT_BASE="output_crosscam_local/holdout"
STAGE="all"
SMOKE=0
PRINT_PLAN=0
FORCE=""
ONLY=""
MAX_FRAMES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)      SMOKE=1; STAGE="detect"; MAX_FRAMES=40; OUT_BASE="tmp/holdout_smoke"; shift ;;
    --stage)      STAGE="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --out)        OUT_BASE="$2"; shift 2 ;;
    --force)      FORCE="--force"; shift ;;
    --print-plan) PRINT_PLAN=1; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 3 ;;
  esac
done

DET="$OUT_BASE/det"
DET_DD="$OUT_BASE/det_dd"
AB="$OUT_BASE/ab"
EMB="$OUT_BASE/emb"
LOG="$OUT_BASE/log"

# ---------------------------------------------------------------- 前置校验
echo "=== 冻结参数 ==="
cat <<EOF
  detection NMS iou : $DET_IOU      （必须显式传；脚本默认 0.5）
  same-frame dedup  : $DEDUP_IOU
  stride gallery    : $STRIDE_GAL  (2.5 fps)
  stride query      : $STRIDE_QRY  (5 fps)
  tracker config    : $TRACKER_CFG
  imgsz / conf      : $IMGSZ / $CONF
  encode device     : $ENC_DEVICE
  min frames (a/b)  : $MIN_FRAMES
  输出目录          : $OUT_BASE
EOF

[ -f "$TRACKER_CFG" ] || { echo "[ABORT] 跟踪配置缺失：$TRACKER_CFG" >&2; exit 3; }
[ -x "$PY" ] || { echo "[ABORT] Python 不存在：$PY（可用 HOLDOUT_PY 覆盖）" >&2; exit 3; }

echo
echo "=== 冻结哈希校验（仓库副本 ↔ 导出包存档副本） ==="
if ! "$PY" tools/verify_preeg_hashes.py > "$OUT_BASE.verify.log" 2>&1; then
  mv "$OUT_BASE.verify.log" "$OUT_BASE.verify.log.fail" 2>/dev/null || true
  echo "[ABORT] 哈希与预注册不一致 → 按 §6 记为偏离，不得静默运行。" >&2
  echo "        详情见 $OUT_BASE.verify.log.fail" >&2
  exit 2
fi
grep -E "逐字节相同|全部一致" "$OUT_BASE.verify.log" | tail -4
rm -f "$OUT_BASE.verify.log"

mkdir -p "$DET" "$DET_DD" "$AB" "$EMB" "$LOG"
CMDS="$LOG/commands.log"
: > "$CMDS"

run_det() {  # seg cam vol offset dur stride
  local seg=$1 vol=$2 off=$3 dur=$4 stride=$5
  local cmd=("$PY" tools/yolo_track_det.py
             --seg "$seg" --video "media-resource/1012/$vol"
             --start-sec "$off" --dur-sec "$dur" --stride "$stride"
             --imgsz "$IMGSZ" --conf "$CONF" --iou "$DET_IOU"
             --out-dir "$DET")
  [ "$MAX_FRAMES" -gt 0 ] && cmd+=(--max-frames "$MAX_FRAMES")
  [ -n "$FORCE" ] && cmd+=("$FORCE")
  printf '%q ' "${cmd[@]}" >> "$CMDS"; echo >> "$CMDS"
  if [ "$PRINT_PLAN" -eq 1 ]; then printf '  '; printf '%q ' "${cmd[@]}"; echo; return 0; fi
  "${cmd[@]}" > "$LOG/det_${seg}.log" 2>&1
  echo "  [det] $seg rc=$? $(date '+%H:%M:%S')"
}

selected() {
  [ -z "$ONLY" ] && return 0
  [ "$1" = "$ONLY" ] && return 0
  return 1
}

# ---------------------------------------------------------------- 1) detect
if [ "$STAGE" = "all" ] || [ "$STAGE" = "detect" ]; then
  echo
  echo "=== [1/4] detect  $(date '+%F %H:%M:%S') ==="
  for row in "${PIECES[@]}"; do
    # shellcheck disable=SC2086
    set -- $row
    seg=$1 cam=$2 win=$3 vol=$4 off=$5 dur=$6 stride=$7
    selected "$seg" || continue
    run_det "$seg" "$vol" "$off" "$dur" "$stride"
    [ "$SMOKE" -eq 1 ] && break   # 冒烟只跑第一个片段
  done
  [ "$PRINT_PLAN" -eq 1 ] && { echo; echo "（--print-plan：未执行，命令已写入 $CMDS）"; exit 0; }
fi

SEGS=$(printf '%s\n' "${PIECES[@]}" | awk '{print $1}' | paste -sd, -)
[ -n "$ONLY" ] && SEGS="$ONLY"   # --only 时只处理该片，便于逐步冒烟

# ---------------------------------------------------------------- 2) dedup
if [ "$STAGE" = "all" ] || [ "$STAGE" = "dedup" ]; then
  echo
  echo "=== [2/4] dedup iou=$DEDUP_IOU  $(date '+%F %H:%M:%S') ==="
  printf '%q ' "$PY" tools/dedup_detections.py --in-dir "$DET" --out-dir "$DET_DD" \
      --iou "$DEDUP_IOU" --segs "$SEGS" >> "$CMDS"; echo >> "$CMDS"
  "$PY" tools/dedup_detections.py --in-dir "$DET" --out-dir "$DET_DD" \
      --iou "$DEDUP_IOU" --segs "$SEGS" | tail -5
fi

# ---------------------------------------------------------------- 3) split a/b
if [ "$STAGE" = "all" ] || [ "$STAGE" = "split" ]; then
  echo
  echo "=== [3/4] split a/b  min-frames=$MIN_FRAMES  $(date '+%F %H:%M:%S') ==="
  for row in "${PIECES[@]}"; do
    set -- $row; seg=$1
    selected "$seg" || continue
    printf '%q ' "$PY" tools/make_ab_jsonl.py \
        --in-jsonl "$DET_DD/$seg/${seg}_detections.jsonl" \
        --out-prefix "$AB/$seg" --min-frames "$MIN_FRAMES" >> "$CMDS"; echo >> "$CMDS"
    "$PY" tools/make_ab_jsonl.py \
        --in-jsonl "$DET_DD/$seg/${seg}_detections.jsonl" \
        --out-prefix "$AB/$seg" --min-frames "$MIN_FRAMES" >> "$LOG/split.log" 2>&1
    echo "  [split] $seg a=$(wc -l < "$AB/${seg}_a.jsonl") b=$(wc -l < "$AB/${seg}_b.jsonl")"
  done
fi

# ---------------------------------------------------------------- 4) encode
if [ "$STAGE" = "all" ] || [ "$STAGE" = "encode" ]; then
  echo
  echo "=== [4/4] encode  $(date '+%F %H:%M:%S') ==="
  for row in "${PIECES[@]}"; do
    set -- $row; seg=$1 vol=$4
    selected "$seg" || continue
    printf '%q ' "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" \
        --video "media-resource/1012/$vol" --device "$ENC_DEVICE" \
        --ab-dir "$AB" --out-dir "$EMB" >> "$CMDS"; echo >> "$CMDS"
    "$PY" tools/encode_1012_osnet_seq.py --seg "$seg" \
        --video "media-resource/1012/$vol" --device "$ENC_DEVICE" \
        --ab-dir "$AB" --out-dir "$EMB" > "$LOG/enc_${seg}.log" 2>&1
    echo "  [enc] $seg rc=$? $(date '+%H:%M:%S')"
  done
fi

echo
echo "=== DONE stage=$STAGE  $(date '+%F %H:%M:%S') ==="
echo "命令清单：$CMDS"
