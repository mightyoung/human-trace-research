#!/bin/bash
# 等 305 单路跑完，再串行跑其余 5 路（实测两进程并行收益仅 16% 且互相拖慢，改串行）。
set -u
OUT=/share/data4/yhh/data/human-count/inference-out-long35-raw
cd ~/human-count || exit 1
while pgrep -f 'dino_raw_dump.*305_long35' > /dev/null; do sleep 60; done
sleep 5
echo "=== 305 finished, start rest $(date '+%m-%d %H:%M:%S') ===" >> "$OUT/chain.log"
# 其余 5 路只跑前 3000 个处理帧（≈10min 视频）用于**泛化验证**，不做全量；
# 全量留给"策略确认后重跑下游"那一步，避免白占 GPU 两小时。
MAX_FRAMES=3000 bash tools/run_dino_raw_all.sh 316 318 320 323 343 >> "$OUT/chain.log" 2>&1
echo "=== ALL DONE $(date '+%m-%d %H:%M:%S') ===" >> "$OUT/chain.log"
