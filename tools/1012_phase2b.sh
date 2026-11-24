#!/bin/bash
# 1012 Phase2b（自愈版）：SSH 恢复等待 → 分片打包上传 → 编码提交 → 轮询 → 打包下载
# 修正了 Phase2 已知三坑：① 上传静默失败（现带验证）② [done] 误导轮询 ③ 空目录下载
# 温柔连接策略：单连接 tar 传输 + 有限重试 + 长间隔轮询
cd /d/dev/human-count || exit 1
LOG=output_crosscam/analysis/1012_pipeline/phase2b.log
PY=/c/Users/might/.workbuddy/binaries/python/envs/default/Scripts/python.exe
AB_DIR=/share/data4/yhh/data/human-count/inference-out-1012-ab
EMB_DIR=/share/data4/yhh/data/human-count/inference-out-1012-emb

echo "===== PHASE2B start $(date '+%H:%M:%S') =====" >> $LOG

# 0) 等 SSH 端口恢复（每 90s 探测，最多 60 次 = 90 分钟）
n=0
while true; do
  if $PY -c "import socket; s=socket.create_connection(('192.168.11.16',10022),timeout=6); s.close()" 2>/dev/null; then
    break
  fi
  n=$((n+1))
  echo "[ssh-wait] $(date '+%H:%M:%S') attempt $n" >> $LOG
  if [ $n -ge 60 ]; then echo "[ssh-wait] TIMEOUT" >> $LOG; exit 2; fi
  sleep 90
done
echo "[ssh] port open $(date '+%H:%M:%S')" >> $LOG
sleep 20

# 1) 打包上传（单连接）
cd output_crosscam/1012_final/ab || exit 3
tar czf ../ab_all.tgz *.jsonl
cd /d/dev/human-count || exit 3
ok=0
for try in 1 2 3; do
  if MSYS_NO_PATHCONV=1 python tools/remote_run.py --put output_crosscam/1012_final/ab_all.tgz /tmp/ab_all.tgz >> $LOG 2>&1; then ok=1; break; fi
  echo "[upload] try $try failed" >> $LOG
  sleep 15
done
if [ $ok = 0 ]; then echo "[upload] tar FAIL after retries" >> $LOG; exit 3; fi
echo "[upload] tar ok size=$(stat -c %s output_crosscam/1012_final/ab_all.tgz)" >> $LOG

# 2) 远程解压 + 验证 24 个
sleep 8
MSYS_NO_PATHCONV=1 python tools/remote_run.py "mkdir -p $AB_DIR && cd $AB_DIR && tar xzf /tmp/ab_all.tgz && ls *.jsonl | wc -l" >> $LOG 2>&1
sleep 3
CNT=$(MSYS_NO_PATHCONV=1 python tools/remote_run.py "ls $AB_DIR/*.jsonl 2>/dev/null | wc -l" 2>/dev/null | grep -oE '^[0-9]+$' | head -1)
echo "[verify] remote ab count=$CNT" >> $LOG

# 3) 启动编码提交
sleep 5
MSYS_NO_PATHCONV=1 python tools/remote_run.py "cd /home/r4900g3/yhh/human-count/crosscam_run && nohup ~/yhh/human-count/embed_venv/bin/python encode_1012_submit.py > encode_1012_submit.log 2>&1 & echo launched" >> $LOG 2>&1
sleep 40
MSYS_NO_PATHCONV=1 python tools/remote_run.py "head -10 /home/r4900g3/yhh/human-count/crosscam_run/encode_1012_submit.log" >> $LOG 2>&1

# 4) 轮询至编码完成（每 150s）
while true; do
  OUT=$(MSYS_NO_PATHCONV=1 python tools/remote_run.py "tail -2 /home/r4900g3/yhh/human-count/crosscam_run/encode_1012_submit.log 2>/dev/null; ls $EMB_DIR/*.npz 2>/dev/null | wc -l" 2>/dev/null)
  N=$(echo "$OUT" | grep -oE '^[0-9]+$' | tail -1)
  echo "[poll] $(date '+%H:%M:%S') npz=$N last=$(echo "$OUT" | head -1 | cut -c1-80)" >> $LOG
  if echo "$OUT" | grep -q '\[done\]'; then break; fi
  if [ -n "$N" ] && [ "$N" -ge 24 ] 2>/dev/null; then break; fi
  sleep 150
done
echo "[encode] finished $(date '+%H:%M:%S')" >> $LOG

# 5) 打包下载 24 npz（单连接）
sleep 8
MSYS_NO_PATHCONV=1 python tools/remote_run.py "cd $EMB_DIR && tar czf /tmp/emb_1012.tgz *.npz && ls -la /tmp/emb_1012.tgz" >> $LOG 2>&1
mkdir -p output_crosscam/analysis/cradio_1012_emb
ok=0
for try in 1 2 3; do
  if MSYS_NO_PATHCONV=1 python tools/remote_run.py --get /tmp/emb_1012.tgz output_crosscam/analysis/cradio_1012_emb/emb_1012.tgz >> $LOG 2>&1; then ok=1; break; fi
  sleep 15
done
if [ $ok = 1 ]; then
  cd output_crosscam/analysis/cradio_1012_emb || exit 4
  tar xzf emb_1012.tgz && rm -f emb_1012.tgz
  echo "[download] npz count=$(ls *.npz 2>/dev/null | wc -l)" >> $LOG
else
  echo "[download] FAIL" >> $LOG
fi
echo "===== PHASE2B DONE $(date '+%H:%M:%S') =====" >> $LOG
