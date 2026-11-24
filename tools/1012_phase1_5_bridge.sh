#!/bin/bash
# 1012 深夜流水线桥接：等 Phase1（下载/清洗/分片/上传）DONE → 自动启动 Phase2（编码）
# 与 phase1_phase2 两个脚本不同，本脚本可安全后台运行（不改动运行中的 watcher）
cd /d/dev/human-count || exit 1
LOG=output_crosscam/analysis/1012_pipeline/phase1_5.log
echo "===== BRIDGE start $(date '+%H:%M:%S') =====" >> $LOG

while true; do
  if grep -q "PHASE1 DONE" output_crosscam/analysis/1012_pipeline/phase1.log 2>/dev/null; then
    echo "[bridge] PHASE1 DONE detected $(date '+%H:%M:%S')" >> $LOG
    break
  fi
  sleep 120
done

# 启动 phase2（内部含 health 预检 + 编码提交 + 轮询 + 下载 24 npz）
bash tools/1012_phase2_encode.sh
echo "[bridge] PHASE2 returned rc=$? $(date '+%H:%M:%S')" >> $LOG
echo "===== BRIDGE DONE $(date '+%H:%M:%S') =====" >> $LOG
