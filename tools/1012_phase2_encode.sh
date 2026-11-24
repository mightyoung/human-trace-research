#!/bin/bash
# 1012 深夜流水线 Phase 2：启动编码（24 任务）→ 轮询 → 下载 24 npz
# 运行时机：Phase1 DONE 后手动启动
cd /d/dev/human-count || exit 1
mkdir -p output_crosscam/analysis/1012_pipeline
LOG=output_crosscam/analysis/1012_pipeline/phase2.log
PY=/c/Users/might/.workbuddy/binaries/python/envs/default/Scripts/python.exe

echo "===== PHASE2 start $(date '+%H:%M:%S') =====" >> $LOG

# 0) embed_server 健康预检（失败则尝试重启；凌晨无人值守关键节点）
hcheck() {
  MSYS_NO_PATHCONV=1 python tools/remote_run.py "~/yhh/human-count/embed_venv/bin/python -c \"import urllib.request,json; h=json.load(urllib.request.urlopen('http://127.0.0.1:8791/health', timeout=10)); print('HEALTH_OK')\"" 2>/dev/null | grep -q HEALTH_OK
}
if hcheck; then
  echo "[health] embed_server OK" >> $LOG
else
  echo "[health] DOWN — restarting embed_server" >> $LOG
  MSYS_NO_PATHCONV=1 python tools/remote_run.py "cd /home/r4900g3/yhh/human-count/crosscam_run && nohup ~/yhh/human-count/embed_venv/bin/python embed_server.py --port 8791 --device cuda:1 --w-dir cradio_scale_v4 --workers 3 --jobs-log embed_server_jobs.jsonl >> embed_server_v11.log 2>&1 & echo relaunched" >> $LOG 2>&1
  sleep 45
  if hcheck; then
    echo "[health] restarted OK" >> $LOG
  else
    echo "[health] STILL DOWN — abort phase2" >> $LOG
    exit 1
  fi
fi

# 1) 启动编码提交
MSYS_NO_PATHCONV=1 python tools/remote_run.py "cd /home/r4900g3/yhh/human-count/crosscam_run && nohup ~/yhh/human-count/embed_venv/bin/python encode_1012_submit.py > encode_1012_submit.log 2>&1 & echo launched" >> $LOG 2>&1
echo "[launch-encoder] rc=$? $(date '+%H:%M:%S')" >> $LOG
sleep 30

# 2) 轮询至完成（检测日志中出现 [done] 24/24 ok 或 npz 数量=24）
while true; do
  TAIL=$(MSYS_NO_PATHCONV=1 python tools/remote_run.py "tail -3 /home/r4900g3/yhh/human-count/crosscam_run/encode_1012_submit.log 2>/dev/null; ls /share/data4/yhh/data/human-count/inference-out-1012-emb/ 2>/dev/null | wc -l" 2>/dev/null)
  N=$(echo "$TAIL" | tail -1 | tr -dc '0-9')
  echo "[poll] $(date '+%H:%M:%S') npz_count=$N" >> $LOG
  if echo "$TAIL" | grep -q '\[done\]'; then break; fi
  if [ -n "$N" ] && [ "$N" -ge 24 ]; then break; fi
  sleep 120
done
echo "[encode-done] $(date '+%H:%M:%S')" >> $LOG

# 3) 下载 24 npz
mkdir -p output_crosscam/analysis/cradio_1012_emb
for seg in 320_train_514 318_train_525 323_train_510 305_train_529 320_val_519 318_val_525 323_val_538 305_val_517 320_test_518 318_test_534 323_test_527 305_test_544; do
  for ab in a b; do
    F=${seg}_emb_${ab}.npz
    MSYS_NO_PATHCONV=1 python tools/remote_run.py --get /share/data4/yhh/data/human-count/inference-out-1012-emb/$F output_crosscam/analysis/cradio_1012_emb/$F >> $LOG 2>&1
    echo "[get] $F rc=$?" >> $LOG
  done
done

echo "===== PHASE2 DONE $(date '+%H:%M:%S') =====" >> $LOG
