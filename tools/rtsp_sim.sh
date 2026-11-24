#!/usr/bin/env bash
# 用本地视频模拟海康 RTSP 摄像头（无需真实设备即可端到端联调）
#
# 原理：ffmpeg 以 RTSP 服务端模式监听 8554 端口，把 mp4 循环推成 RTSP 流。
# 参数刻意对齐海康子码流：704x576 / 25fps。
#
# 用法：
#   1) 先运行本脚本（保持不关）
#   2) 另一个终端运行服务：
#        cargo run -- --rtsp "rtsp://127.0.0.1:8554/live"
#
# 换其他视频：修改 SRC 即可，或作为第 1 个参数传入。

set -euo pipefail

# ffmpeg 路径：优先环境变量 FFMPEG，其次本机默认安装位置，最后回退 PATH 里的 ffmpeg
FFMPEG="${FFMPEG:-D:/dev/ffmpeg/bin/ffmpeg.exe}"
[ -x "$FFMPEG" ] || FFMPEG="ffmpeg"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-$HERE/../media-resource/192.168.20.159_02_20261010161653728.mp4}"
PORT="${PORT:-8554}"
MOUNT="${MOUNT:-live}"

echo "[rtsp-sim] source = $SRC"
echo "[rtsp-sim] listen = rtsp://127.0.0.1:$PORT/$MOUNT"
echo "[rtsp-sim] 按 Ctrl+C 停止"

exec "$FFMPEG" -hide_banner -loglevel info \
  -re -stream_loop -1 -i "$SRC" \
  -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -g 25 -pix_fmt yuv420p \
  -f rtsp -rtsp_flags listen -rtsp_transport tcp \
  "rtsp://127.0.0.1:$PORT/$MOUNT"
