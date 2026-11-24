#!/usr/bin/env bash
# v2 补充实验：①训练充分版（epochs=160，320 用 batch 64）②关键配置 seed 重复
set -u
PY=$HOME/miniconda3/envs/rfdetr/bin/python
D=$HOME/human-count/intent_cc
OUT=$D/out2
mkdir -p $OUT
B="--train $D/1012_train_cc.npz --val $D/1012_val_cc.npz --test $D/1012_test_cc.npz"

run() {
  local name=$1; shift
  echo "=== [$name] $* ==="
  $PY $D/intent_model_cc.py $B --out-dir $OUT/$name "$@" > $OUT/$name.log 2>&1
  echo "[done] $name rc=$?"
}

echo "== A. 训练充分版 (epochs=160) =="
run cam305_none_v2 --cams 305 --id-mode none --epochs 160
run cam318_none_v2 --cams 318 --id-mode none --epochs 160
run cam320_none_v2 --cams 320 --id-mode none --epochs 160 --batch 64
run cam323_none_v2 --cams 323 --id-mode none --epochs 160

echo "== B. seed 重复 (v1 口径) =="
run global_none_s0 --id-mode none --seed 0
run global_none_s123 --id-mode none --seed 123
run cam318_none_s0 --cams 318 --id-mode none --seed 0
run cam318_none_s123 --cams 318 --id-mode none --seed 123
run cam323_none_s0 --cams 323 --id-mode none --seed 0
run cam323_none_s123 --cams 323 --id-mode none --seed 123
run cam305_proto_s0 --cams 305 --id-mode proto --seed 0
run cam305_proto_s123 --cams 305 --id-mode proto --seed 123

echo "ALL DONE V2"
