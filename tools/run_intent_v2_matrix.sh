#!/bin/bash
# 意图模型 v2 训练矩阵：4 配置 × 3 seeds（社交邻域 / 行为聚类条件化）
cd "$(dirname "$0")"
PY=~/miniconda3/envs/rfdetr/bin/python
mkdir -p out_v2

for seed in 42 43 44; do
  for cfg in base social behavior both; do
    case $cfg in
      base)     FLAGS="" ;;
      social)   FLAGS="--social" ;;
      behavior) FLAGS="--behavior" ;;
      both)     FLAGS="--social --behavior" ;;
    esac
    echo "=== [$cfg s$seed] start $(date +%H:%M:%S) ==="
    $PY intent_model_social.py \
        --train 1012_train_v2.npz --val 1012_val_v2.npz --test 1012_test_v2.npz \
        --out-dir out_v2/${cfg}_s${seed} $FLAGS --epochs 40 --seed $seed \
        > out_v2/${cfg}_s${seed}.log 2>&1
    echo "=== [$cfg s$seed] rc=$? end $(date +%H:%M:%S) ==="
  done
done
echo "ALL DONE"
