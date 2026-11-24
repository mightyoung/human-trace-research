#!/bin/bash
echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== Python/torch ==="
~/miniconda3/envs/rfdetr/bin/python -V
~/miniconda3/envs/rfdetr/bin/python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())"
echo "=== timm/transformers ==="
~/miniconda3/envs/rfdetr/bin/python -c "import timm; print('timm', timm.__version__)" 2>&1 | tail -1
~/miniconda3/envs/rfdetr/bin/python -c "import transformers; print('transformers', transformers.__version__)" 2>&1 | tail -1
echo "=== 工作目录 ==="
ls /home/r4900g3/yhh/human-count/ 2>/dev/null | head -25
echo "--- crosscam/in ---"
ls /home/r4900g3/yhh/human-count/crosscam/in/ 2>/dev/null | head -15
echo "--- output_crosscam ---"
ls /home/r4900g3/yhh/human-count/output_crosscam/ 2>/dev/null | head -20
echo "=== 网络 ==="
echo "hf-mirror:"; curl -sI --max-time 8 https://hf-mirror.com 2>&1 | head -1
echo "tuna pypi:"; curl -sI --max-time 8 https://pypi.tuna.tsinghua.edu.cn/simple/ 2>&1 | head -1
echo "done"
