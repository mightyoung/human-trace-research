#!/usr/bin/env bash
# env_setup.sh — 检测头实验环境搭建（后台跑）
set -u
P="$HOME/miniconda3/envs/rfdetr/bin/python"
D="$HOME/yhh/human-count/bench/det"
mkdir -p "$D"
LOG="$D/env_setup.log"
{
echo "[env] START $(date '+%F %T')"
echo "=== STEP1 pip install ultralytics ==="
$P -m pip install ultralytics 2>&1 | tail -4
echo "[env] pip exit=$?"
echo "=== STEP2 ultralytics version ==="
$P -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
echo "=== STEP3 yolo26 weights download ==="
cd "$D"
$P -c "
from ultralytics import YOLO
for n in ['yolo26s.pt','yolo26m.pt']:
    m = YOLO(n)
    print(n, 'OK', m.task)
"
echo "[env] yolo exit=$?"
echo "=== STEP4 rfdetr small weight init ==="
$P -c "
from rfdetr import RFDETRSmall
m = RFDETRSmall()
print('RFDETRSmall init OK')
" 2>&1 | tail -5
echo "[env] rfdetr exit=$?"
ls -la "$D"/*.pt 2>/dev/null
echo "[env] DONE $(date '+%F %T')"
} > "$LOG" 2>&1
