#!/usr/bin/env bash
# 意图预测：摄像头分组 × 身份条件化 训练矩阵（10 模型）—— 远程执行
# 用法: cd ~/human-count/intent_cc && nohup bash run_intent_cc_matrix.sh > matrix.log 2>&1 &
set -u
PY=$HOME/miniconda3/envs/rfdetr/bin/python
D=$HOME/human-count/intent_cc
OUT=$D/out
mkdir -p $OUT

run() {
  local name=$1 mode=$2 cams=$3
  local cargs=""
  [ -n "$cams" ] && cargs="--cams $cams"
  echo "=== [$name] mode=$mode cams=${cams:-ALL} ==="
  $PY $D/intent_model_cc.py \
    --train $D/1012_train_cc.npz --val $D/1012_val_cc.npz --test $D/1012_test_cc.npz \
    --out-dir $OUT/$name --id-mode $mode --n-proto 125 $cargs \
    > $OUT/$name.log 2>&1
  echo "[done] $name rc=$?"
}

run global_none none ""
run global_proto proto ""
for c in 305 318 320 323; do
  run cam${c}_none none $c
  run cam${c}_proto proto $c
done
echo "ALL DONE"
