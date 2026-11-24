#!/bin/bash
cd /share/data4/yhh/data/human-count/crosscam-run
PY=~/miniconda3/envs/rfdetr/bin/python
V=/share/data4/yhh/data/human-count/video-6way
D=/share/data4/yhh/data/human-count/inference-out
mkdir -p logs

echo "=== 分辨率核查 ==="
$PY - <<'PYEOF'
import cv2
for c in ['305','316','318','320','323','343']:
    cap = cv2.VideoCapture(f'/share/data4/yhh/data/human-count/video-6way/{c}.mp4')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(c, f'{w}x{h}', 'frames:', n)
    cap.release()
PYEOF

echo "=== 启动四路编码 ==="
( CUDA_VISIBLE_DEVICES=0 $PY tools/encode_tracks.py --jsonl $D/305_detections.jsonl --prefix v305_ --video $V/305.mp4 --out out/reid_emb_305_d3b.npz --samples 8 --backend dinov3b --device cuda > logs/enc305.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY tools/encode_tracks.py --jsonl $D/316_detections.jsonl --prefix v316_ --video $V/316.mp4 --out out/reid_emb_316_d3b.npz --samples 8 --backend dinov3b --device cuda > logs/enc316.log 2>&1
  echo GPU0_DONE > logs/enc_gpu0.done ) &
( CUDA_VISIBLE_DEVICES=1 $PY tools/encode_tracks.py --jsonl $D/323_detections.jsonl --prefix v323_ --video $V/323.mp4 --out out/reid_emb_323_d3b.npz --samples 8 --backend dinov3b --device cuda > logs/enc323.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY tools/encode_tracks.py --jsonl $D/343_detections.jsonl --prefix v343_ --video $V/343.mp4 --out out/reid_emb_343_d3b.npz --samples 8 --backend dinov3b --device cuda > logs/enc343.log 2>&1
  echo GPU1_DONE > logs/enc_gpu1.done ) &
sleep 15
echo "=== 15s 后状态 ==="
ps aux | grep encode_tracks | grep -v grep | wc -l
tail -3 logs/enc305.log 2>/dev/null
tail -3 logs/enc323.log 2>/dev/null
echo "=== ENCODE6_LAUNCHED ==="
