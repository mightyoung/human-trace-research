#!/usr/bin/env bash
# train_all.sh — 检测头实验：等待数据 → 串行训练 6 个模型（单 GPU 不并发）
# 顺序（按信息价值）：YOLO26s（基准）→ RF-DETR-Small → YOLO26x → RF-DETR-Large → YOLO26m → RF-DETR-Medium
set -u
D=/share/data4/yhh/data/human-count/det-bench
B=/home/r4900g3/yhh/human-count/bench/det
P="$HOME/miniconda3/envs/rfdetr/bin/python"
cd "$B"

echo "[all] START $(date '+%F %T')"

# ---- 等待数据集就绪（打标+导出完成） ----
for i in $(seq 1 240); do
  [ -f "$D/pseudo/stats.json" ] && break
  sleep 30
done
if [ ! -f "$D/pseudo/stats.json" ]; then
  echo "[all] dataset NOT ready after wait, abort"
  exit 1
fi
echo "[all] dataset ready $(date '+%F %T')"
cat "$D/pseudo/stats.json"

echo "===== [1/6] YOLO26s (imgsz960 b16) ====="
"$P" train_yolo.py yolo26s --epochs 100 --imgsz 960 --batch 16 > "$B/train_yolo26s.log" 2>&1
echo "[all] yolo26s exit=$? $(date '+%F %T')"

echo "===== [2/6] RF-DETR-Small ====="
"$P" train_rfdetr.py small --epochs 80 > "$B/train_rfdetr_small.log" 2>&1
echo "[all] rfdetr_small exit=$? $(date '+%F %T')"

echo "===== [3/6] YOLO26x (imgsz960 b6) ====="
"$P" train_yolo.py yolo26x --epochs 80 --imgsz 960 --batch 6 > "$B/train_yolo26x.log" 2>&1
echo "[all] yolo26x exit=$? $(date '+%F %T')"

echo "===== [4/6] RF-DETR-Large ====="
"$P" train_rfdetr.py large --epochs 80 > "$B/train_rfdetr_large.log" 2>&1
echo "[all] rfdetr_large exit=$? $(date '+%F %T')"

echo "===== [5/6] YOLO26m (imgsz960 b8) ====="
"$P" train_yolo.py yolo26m --epochs 100 --imgsz 960 --batch 8 > "$B/train_yolo26m.log" 2>&1
echo "[all] yolo26m exit=$? $(date '+%F %T')"

echo "===== [6/6] RF-DETR-Medium ====="
"$P" train_rfdetr.py medium --epochs 80 > "$B/train_rfdetr_medium.log" 2>&1
echo "[all] rfdetr_medium exit=$? $(date '+%F %T')"

echo "[all] ALL DONE $(date '+%F %T')"
