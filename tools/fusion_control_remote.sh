#!/bin/bash
# 对照实验：随机正交 W 跑同一融合管线（验证 ridge-W 的信息传递作用）
set -e
cd /home/r4900g3/yhh/human-count/crosscam_run
PY=/home/r4900g3/yhh/human-count/embed_venv/bin/python

# 备份真实产物
mv fusion/bridge_emb_cache.npz fusion/bridge_emb_cache_T3real.npz
cp fusion/bridge_W.npy fusion/bridge_W_T3real.npy

# 生成随机正交矩阵替换 W
$PY -c "import numpy as np; rng = np.random.RandomState(0); Q, _ = np.linalg.qr(rng.randn(768, 768)); np.save('fusion/bridge_W.npy', Q.astype(np.float32)); print('rand W saved', Q.shape)"

# 跑 eval（随机 W）
$PY fusion_dinogemma_remote.py bridge --stage eval

# 移走随机 W 的缓存并打分
mv fusion/bridge_emb_cache.npz fusion/bridge_emb_cache_T3rand.npz
$PY fusion_rescore.py fusion/bridge_emb_cache_T3rand.npz fusion/bridge_result_rand.json

# 恢复真实产物
mv fusion/bridge_emb_cache_T3real.npz fusion/bridge_emb_cache.npz
cp fusion/bridge_W_T3real.npy fusion/bridge_W.npy
echo CONTROL_DONE
