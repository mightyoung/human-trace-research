#!/usr/bin/env bash
# run_all_bench.sh — 串行执行三个基准（同一张 GPU，避免测速互相干扰）
set -u
cd /home/r4900g3/yhh/human-count/bench
PY_RF="$HOME/miniconda3/envs/rfdetr/bin/python"
PY_EMB="$HOME/yhh/human-count/embed_venv/bin/python"

echo "[run_all] START $(date '+%F %T')"
echo "=== STEP1 gemm_probe (embed_venv) ==="
"$PY_EMB" gemm_probe.py --out gemm_probe.json --device cuda:0
echo "[run_all] step1 exit=$?"
echo "=== STEP2 det_batch_bench (rfdetr) ==="
"$PY_RF" det_batch_bench.py --video /share/data4/yhh/data/human-count/video-6way/318.mp4 --out det_bench.json --device cuda:0
echo "[run_all] step2 exit=$?"
echo "=== STEP3 emb_quant_probe (embed_venv) ==="
"$PY_EMB" emb_quant_probe.py --out emb_quant_probe.json --device cuda:0
echo "[run_all] step3 exit=$?"
echo "[run_all] DONE $(date '+%F %T')"
