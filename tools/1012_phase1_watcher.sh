#!/bin/bash
# 1012 深夜流水线 Phase 1：等检测完成 → 下载 → 清洗(323) → final/分片 → 上传
# 后台运行；日志 output_crosscam/analysis/1012_pipeline/phase1.log
cd /d/dev/human-count || exit 1
mkdir -p output_crosscam/analysis/1012_pipeline
LOG=output_crosscam/analysis/1012_pipeline/phase1.log
PY=/c/Users/might/.workbuddy/binaries/python/envs/default/Scripts/python.exe

echo "===== PHASE1 watcher start $(date '+%H:%M:%S') =====" >> $LOG

# 0) 等检测全部完成
while true; do
  HIT=$(MSYS_NO_PATHCONV=1 python tools/remote_run.py "grep -q 'ALL DONE' /home/r4900g3/yhh/human-count/crosscam_run/run_1012_detect.log && echo WATCHER_MARKER_HIT" 2>/dev/null | grep -c WATCHER_MARKER_HIT)
  echo "[wait] $(date '+%H:%M:%S') marker_hit=$HIT" >> $LOG
  if [ "$HIT" -ge 1 ] 2>/dev/null; then break; fi
  sleep 60
done
echo "[done-wait] detect finished $(date '+%H:%M:%S')" >> $LOG

# 1) 下载 12 段
$PY tools/1012_post_detect.py download >> $LOG 2>&1
echo "[download] rc=$? $(date '+%H:%M:%S')" >> $LOG

# 2) 清洗 323（运动学 + 玻璃掩膜）
$PY tools/1012_post_detect.py clean >> $LOG 2>&1
echo "[clean] rc=$? $(date '+%H:%M:%S')" >> $LOG

# 3) final 集 + a/b 分片
$PY tools/1012_post_detect.py final >> $LOG 2>&1
echo "[final] rc=$? $(date '+%H:%M:%S')" >> $LOG

# 4) 上传 a/b（24 个 jsonl）
$PY tools/1012_post_detect.py upload >> $LOG 2>&1
echo "[upload] rc=$? $(date '+%H:%M:%S')" >> $LOG

# 5) 上传编码提交脚本（不启动，留给主流程启动）
MSYS_NO_PATHCONV=1 python tools/remote_run.py --put tools/encode_1012_submit.py /home/r4900g3/yhh/human-count/crosscam_run/encode_1012_submit.py >> $LOG 2>&1
echo "[put-encoder] rc=$? $(date '+%H:%M:%S')" >> $LOG

echo "===== PHASE1 DONE $(date '+%H:%M:%S') =====" >> $LOG
