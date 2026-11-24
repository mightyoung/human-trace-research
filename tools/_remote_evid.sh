#!/bin/bash
cd /share/data4/yhh/data/human-count/crosscam-run
PY=~/miniconda3/envs/rfdetr/bin/python
V=/share/data4/yhh/data/human-count/video-6way

echo "########## 320 证据图（含 18/22/23 手动指定）##########"
$PY tools/stitch_evidence.py --video $V/320.mp4 \
  --jsonl stitch_in/320_cleaned.jsonl --wh 2560 1440 \
  --csv stitch_out/stitch_320.csv \
  --pairs "8:9,9:13,13:16,16:17,18:23,18:22,22:23,19:20,25:26,26:27" \
  --out-prefix stitch_out/evid_320 --per-page 12

echo ""
echo "########## 318 证据图（全部 37 links）##########"
$PY tools/stitch_evidence.py --video $V/318.mp4 \
  --jsonl stitch_in/318_cleaned.jsonl --wh 1920 1080 \
  --csv stitch_out/stitch_318.csv \
  --links stitch_out/stitch_318_groups.json \
  --out-prefix stitch_out/evid_318 --per-page 12

echo "########## 文件清单 ##########"
ls -la stitch_out/evid_*.png
echo "########## EVID_DONE ##########"
