#!/bin/bash
# C 方案第一步：6 路 35min 全量 dump Containment-NMS 之前的原始框（GPU1，串行）。
# 为什么必须重跑：历史 detections.jsonl 只存 NMS 后的框（305 被删 13.7%），
#   要判断「谁被删、被谁吞、分数多少」只能回到 NMS 前。
# 输出 /share/data4/yhh/data/human-count/inference-out-long35-raw/{cam}_raw10.jsonl
#   threshold=0.10（生产 0.30 的候选是其子集，本地可任意切阈值复现）
set -u
PY=~/miniconda3/envs/rfdetr/bin/python
VDIR=/share/data4/yhh/data/human-count/video-long35
OUT=/share/data4/yhh/data/human-count/inference-out-long35-raw
mkdir -p "$OUT"
cd ~/human-count || exit 1

# 用法: bash run_dino_raw_all.sh [cam...]   默认 6 路；可拆成两个进程并行
# 环境变量 MAX_FRAMES=N：每路只跑前 N 个处理帧（泛化验证用，省 GPU）
CAMS="${*:-305 316 318 320 323 343}"
MAXF="${MAX_FRAMES:-0}"
EXTRA=""
[ "$MAXF" -gt 0 ] 2>/dev/null && EXTRA="--max-frames $MAXF"

for cam in $CAMS; do
  f="$OUT/${cam}_raw10.jsonl"
  if [ -f "$f" ]; then
    n=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$n" -ge 10000 ]; then echo "[skip] $cam rows=$n"; continue; fi
    echo "[redo] $cam 半成品 rows=$n"
  fi
  echo "===== [$cam] start $(date +%H:%M:%S) ====="
  "$PY" tools/dino_raw_dump.py --video "$VDIR/${cam}_long35.mp4" --cam "$cam" \
      --device cuda:1 --mode full --stride 5 --box-threshold 0.10 --resume $EXTRA \
      --out "$f" >> "$OUT/${cam}.log" 2>&1
  echo "[$cam] exit=$? rows=$(wc -l < "$f" 2>/dev/null || echo 0) $(date +%H:%M:%S)"
done
echo ALL_DONE
