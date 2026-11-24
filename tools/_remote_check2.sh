#!/bin/bash
echo "=== 网络详测 ==="
for url in https://pypi.org/simple/ https://files.pythonhosted.org https://github.com https://mirrors.aliyun.com/pypi/simple/ https://ghproxy.net; do
  code=$(curl -sI --max-time 6 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
  echo "$url -> $code"
done
echo "=== pip 配置 ==="
~/miniconda3/envs/rfdetr/bin/pip config list 2>/dev/null
cat ~/.pip/pip.conf 2>/dev/null
cat ~/.config/pip/pip.conf 2>/dev/null
echo "=== GPU 完整列表 ==="
nvidia-smi -L
echo "=== torchvision ==="
~/miniconda3/envs/rfdetr/bin/python -c "import torchvision; print('tv', torchvision.__version__)" 2>&1 | tail -1
~/miniconda3/envs/rfdetr/bin/python -c "import einops; print('einops', einops.__version__)" 2>&1 | tail -1
~/miniconda3/envs/rfdetr/bin/python -c "import safetensors; print('safetensors ok')" 2>&1 | tail -1
echo "=== 磁盘 ==="
df -h /home | tail -1
echo "done"
