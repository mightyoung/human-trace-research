#!/usr/bin/env bash
# ============================================================================
# 留出复验分析脚本（冻结版）—— 对应预注册 §3.1 标注协议、§4 假设、§5 分析计划
#
# 接在 tools/run_holdout.sh（检测→去重→切 a/b→编码）之后执行：
#   bash tools/run_holdout_analysis.sh --stage verify      # 只核对分析代码哈希
#   bash tools/run_holdout_analysis.sh --stage sweep       # 42 配置扫描（d7_sweep，参数写死）
#   bash tools/run_holdout_analysis.sh --stage templates   # 生成盲标模板（范围规则写死）
#   （人工标注：另存为 gt/gt_gallery_members_FILLED.csv 与 gt/gt_query_gt_FILLED.csv）
#   bash tools/run_holdout_analysis.sh --stage eval        # 判定 H1a/H1b/H1c/H2c/H3
#   bash tools/run_holdout_analysis.sh --print-plan        # 只打印命令
#
# 所有参数写死；任何一项要改，必须改脚本并重算冻结哈希（预注册 §9）。
# 退出码：0 成功；2 分析代码哈希不一致；3 参数/前置条件不满足
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- 冻结参数
readonly OUT_BASE="output_crosscam_local/holdout"     # 与 run_holdout.sh 一致
readonly EMB="$OUT_BASE/emb"
readonly DET_DD="$OUT_BASE/det_dd"                     # 共现对取自去重后的查询检测
readonly PKG="$OUT_BASE/eval"
readonly SAME_MIN=0.90
readonly MIN_EVAL=100
readonly ROBUST_MARGIN=0.10                            # 仅补充表，不参与判定
readonly GAL_SEGS="305_hgal_p1,305_hgal_p2,318_hgal,320_hgal_p1,320_hgal_p2,323_hgal"
readonly QRY_SEGS="305_hqry,318_hqry,320_hqry,323_hqry"

# 分析代码的完整 sha256（取自 2026-10-24 导出包存档副本，前 16 位与其 MANIFEST.md 一致）
readonly -a DEP_HASHES=(
  "bfb1b2650d756d06196898ba1243aceb17a208c37483ca68e3abafc06f1b3e60  tools/d7_sweep.py"
  "3a344cbc59a7d27ccba54e072e6af8a1ead42457c0ee9ac8bd50acd0ea3d14dc  tools/session_selector_v1.py"
  "549c12f622145627b67756201140bf1981aeb6495c9902daf5e54fa19ae9d171  tools/eval_mask_variants.py"
  "85eb338b8ad48b430edd3687d52a3d1b2deb575c5b64cf9515b2889d0c4ad1f2  tools/cluster_camera_aware.py"
  "854314b9d5be0ee41e4665072dc611686d33bf4b0a4b2bd225f5386f06e3b0b6  tools/eval_selector_1012test.py"
  "57e83fb7477db10a2dfdccf8bd76ec837bf3d760aad11f7d5c73d72946e5042f  tools/selector.py"
)

# ---------------------------------------------------------------- 运行时（同 run_holdout.sh）
if command -v cygpath >/dev/null 2>&1; then ROOT_WIN="$(cygpath -m "$ROOT")"; else ROOT_WIN="$ROOT"; fi
export PYTHONPATH="$ROOT_WIN/pylibs${PYTHONPATH:+;$PYTHONPATH}"
export TMPDIR="$ROOT_WIN/tmp" TEMP="$ROOT_WIN/tmp"
PY="${HOLDOUT_PY:-C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe}"

STAGE=""
PRINT_PLAN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage)      STAGE="$2"; shift 2 ;;
    --print-plan) PRINT_PLAN=1; shift ;;
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 3 ;;
  esac
done
[ -n "$STAGE" ] || [ "$PRINT_PLAN" -eq 1 ] || { echo "须指定 --stage 或 --print-plan" >&2; exit 3; }

SWEEP=("$PY" tools/d7_sweep.py --emb "$EMB" --det "$DET_DD"
       --train-segs "$GAL_SEGS" --test-segs "$QRY_SEGS"
       --out "$PKG/results" --min-eval "$MIN_EVAL" --same-min "$SAME_MIN")
SWEEP_ROBUST=("$PY" tools/d7_sweep.py --emb "$EMB" --det "$DET_DD"
       --train-segs "$GAL_SEGS" --test-segs "$QRY_SEGS"
       --out "$PKG/results_robust" --min-eval "$MIN_EVAL" --same-min "$SAME_MIN"
       --robust-margin "$ROBUST_MARGIN")
TPL=("$PY" tools/holdout_make_templates.py --run "$PKG/results" --out "$PKG/gt")
EVAL=("$PY" tools/holdout_eval.py --pkg "$PKG")

if [ "$PRINT_PLAN" -eq 1 ]; then
  for c in SWEEP SWEEP_ROBUST TPL EVAL; do
    eval "arr=(\"\${${c}[@]}\")"; printf '  '; printf '%q ' "${arr[@]}"; echo
  done
  exit 0
fi

verify_deps() {
  echo "=== 分析代码哈希核对 ==="
  local bad=0 line want rel got
  for line in "${DEP_HASHES[@]}"; do
    want="${line%%  *}"; rel="${line##*  }"
    if [ ! -f "$rel" ]; then echo "  MISSING  $rel"; bad=1; continue; fi
    got="$(sha256sum "$rel" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then echo "  OK       $rel"; else echo "  FAIL     $rel"; echo "           实测 $got"; bad=1; fi
  done
  [ "$bad" -eq 0 ] || { echo "[ABORT] 分析代码与冻结记录不一致 → 按 §6 记为偏离，不得静默运行。" >&2; exit 2; }
}

mkdir -p "$PKG"
verify_deps

case "$STAGE" in
  verify) ;;
  sweep)
    for s in ${GAL_SEGS//,/ } ${QRY_SEGS//,/ }; do
      [ -f "$EMB/${s}_emb_a.npz" ] && [ -f "$EMB/${s}_emb_b.npz" ] || { echo "[ABORT] 缺少编码 $EMB/${s}_emb_{a,b}.npz" >&2; exit 3; }
    done
    echo "=== sweep（主口径，参与判定） ==="; "${SWEEP[@]}"
    echo "=== sweep（稳健口径 ±$ROBUST_MARGIN，仅补充表） ==="; "${SWEEP_ROBUST[@]}"
    ;;
  templates)
    [ -f "$PKG/results/queries/A_baseline.csv" ] || { echo "[ABORT] 先跑 --stage sweep" >&2; exit 3; }
    "${TPL[@]}"
    ;;
  eval)
    for f in gt_gallery_members_FILLED.csv gt_query_gt_FILLED.csv; do
      [ -f "$PKG/gt/$f" ] || { echo "[ABORT] 缺少人工标注 $PKG/gt/$f" >&2; exit 3; }
    done
    "${EVAL[@]}"
    ;;
  *) echo "未知 stage：$STAGE" >&2; exit 3 ;;
esac
echo "=== DONE stage=$STAGE ==="
