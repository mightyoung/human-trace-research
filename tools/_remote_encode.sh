#!/bin/bash
set -e
cd /home/r4900g3/yhh/human-count/crosscam_run
PY=~/miniconda3/envs/rfdetr/bin/python
echo "=== 0. weights ==="
ls -la models/dinov3_vitb16_timm/
echo "=== 1. frame count check ==="
$PY -c "
import cv2
for name, exp in [('318', 14914), ('320', 14901)]:
    cap = cv2.VideoCapture('/home/r4900g3/yhh/human-count/crosscam/in/' + name + '.mp4')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(name, 'frames:', n, '(expected ~%d)' % exp)
    cap.release()
"
echo "=== 2. probe DINOv3-B on GPU ==="
$PY -c "
import sys; sys.path.insert(0, 'tools')
import numpy as np
from crosscam_reid import Embedder
e = Embedder(device='cuda', backend='dinov3b')
v = e.embed_batch([np.zeros((224,224,3), dtype=np.uint8), np.zeros((224,224,3), dtype=np.uint8)+100])
print('probe ok', v.shape, v.dtype)
"
echo "=== 3. encode 318 -> a* ==="
$PY tools/encode_tracks.py --jsonl 318_detections.jsonl --prefix a \
  --video /home/r4900g3/yhh/human-count/crosscam/in/318.mp4 --wh 1920 1080 \
  --out out/reid_emb_318_d3b.npz --samples 8 --backend dinov3b --device cuda
echo "=== 4. encode 320 -> b* ==="
$PY tools/encode_tracks.py --jsonl 320_detections.jsonl --prefix b \
  --video /home/r4900g3/yhh/human-count/crosscam/in/320.mp4 --wh 2560 1440 \
  --out out/reid_emb_320_d3b.npz --samples 8 --backend dinov3b --device cuda
echo "=== 5. canonical reid_scores ==="
$PY tools/crosscam_reid.py --a-jsonl 318_detections.jsonl --b-jsonl 320_detections.jsonl \
  --a-video /home/r4900g3/yhh/human-count/crosscam/in/318.mp4 \
  --b-video /home/r4900g3/yhh/human-count/crosscam/in/320.mp4 \
  --a-wh 1920 1080 --b-wh 2560 1440 --backend dinov3b --device cuda \
  --out out/reid_scores.json --save-emb out/reid_emb.npz
echo "=== 6. verify ==="
$PY -c "
import numpy as np
for f in ['out/reid_emb_318_d3b.npz', 'out/reid_emb_320_d3b.npz', 'out/reid_emb.npz']:
    z = np.load(f)
    ks = list(z.files)
    print(f.split('/')[-1], '-', len(ks), 'keys, dim', z[ks[0]].shape)
"
echo "ALL_DONE"
