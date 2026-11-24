#!/bin/bash
echo "=== [1] /share/data4/yhh/ ==="
ls -la /share/data4/yhh/ 2>&1 | head -25
echo ""
echo "=== [2] /share/data4/yhh/data 存在? ==="
ls -la /share/data4/yhh/data 2>&1 | head -25
echo ""
echo "=== [3] /home/r4900g3/yhh/ ==="
ls -la /home/r4900g3/yhh/ 2>&1 | head -30
echo ""
echo "=== [4] human-count 目录树（2层）==="
find /home/r4900g3/yhh/human-count -maxdepth 2 -type d 2>/dev/null | head -30
echo ""
echo "=== [5] crosscam_run ==="
ls -la /home/r4900g3/yhh/human-count/crosscam_run/ 2>&1
echo ""
echo "=== [6] 视频文件 ==="
find /home/r4900g3/yhh -name "*.mp4" 2>/dev/null | head -30
echo ""
echo "=== [7] df 空间 ==="
df -h /share/data4 2>&1 | head -5
echo ""
echo "=== [8] du 占用 ==="
du -sh /home/r4900g3/yhh/human-count 2>/dev/null
du -sh /home/r4900g3/yhh/human-count/crosscam_run 2>/dev/null
