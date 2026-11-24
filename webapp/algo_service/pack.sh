#!/bin/bash
# 离线组装镜像：把宿主上已跑通的环境/权重拷成 build context（同盘硬链接，不占空间），再 build。
# 无外网 → 不做任何 apt/pip，缺的系统库从宿主 /lib 里补。
set -euo pipefail

SRC=/share/data4/yhh/data/human-count/algo_image
CTX=/tmp/hc_img
E="$HOME/miniconda3/envs/rfdetr"
V="$HOME/venvs/embgemma/lib/python3.11/site-packages"
TAG="${1:-hc-attr:1.0}"

echo "[1/4] 清理旧 context"
rm -rf "$CTX"
mkdir -p "$CTX/libs" "$CTX/app" "$CTX/models" "$CTX/env" "$CTX/venv"

echo "[2/4] 硬链接环境与权重（同盘，不占额外空间）"
cp -al "$E" "$CTX/env/rfdetr"
cp -al "$V" "$CTX/venv/site-packages"
cp -al "$HOME/work/eg2/model" "$CTX/models/eg2"
cp -al "$HOME/work/eg2/attr_heads.joblib" "$CTX/models/"
cp -al "$HOME/.cache/ultralytics/yolo11m.pt" "$CTX/models/"

echo "[3/4] 代码与缺失的系统库"
cp "$SRC/app.py" "$CTX/app/"
cp "$SRC/README.md" "$CTX/app/"
cp "$SRC/Dockerfile" "$CTX/"
for f in libGL.so.1 libGLX.so.0 libGLdispatch.so.0 libglib-2.0.so.0 libgthread-2.0.so.0 \
         libX11.so.6 libxcb.so.1 libXau.so.6 libXdmcp.so.6 libbsd.so.0 libmd.so.0; do
  src=$(ls "/lib/x86_64-linux-gnu/$f" 2>/dev/null || ls "/usr/lib/x86_64-linux-gnu/$f" 2>/dev/null || true)
  if [ -n "$src" ]; then cp -L "$src" "$CTX/libs/$f"; else echo "  [warn] 缺 $f"; fi
done
ls -1 "$CTX/libs" | wc -l | xargs echo "  libs 数量:"

echo "[4/4] docker build -> $TAG"
cd "$CTX"
docker build -t "$TAG" . 2>&1 | tail -30
docker images "$TAG" --format '{{.Repository}}:{{.Tag}} {{.Size}}'
