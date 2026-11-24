#!/bin/bash
cd /share/data4/yhh/data/human-count/crosscam-run
PY=~/miniconda3/envs/rfdetr/bin/python

echo "########## 320 v2 ##########"
$PY tools/track_stitching.py --jsonl stitch_in/320_cleaned.jsonl --wh 2560 1440 \
  --emb out/reid_emb_320_d3b.npz --prefix b \
  --out-csv stitch_out/stitch_320.csv --out-groups stitch_out/stitch_320_groups.json --top 35

echo ""
echo "########## 318 v2 ##########"
$PY tools/track_stitching.py --jsonl stitch_in/318_cleaned.jsonl --wh 1920 1080 \
  --emb out/reid_emb_318_d3b.npz --prefix a \
  --out-csv stitch_out/stitch_318.csv --out-groups stitch_out/stitch_318_groups.json --top 45
echo "########## STITCH3_DONE ##########"
