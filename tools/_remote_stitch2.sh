#!/bin/bash
cd /share/data4/yhh/data/human-count/crosscam-run
PY=~/miniconda3/envs/rfdetr/bin/python

echo "===== P2 STITCH: 318 ====="
$PY tools/track_stitching.py --jsonl stitch_in/318_cleaned.jsonl --wh 1920 1080 \
  --emb out/reid_emb_318_d3b.npz --prefix a \
  --out-csv stitch_out/stitch_318.csv --top 60

echo ""
echo "===== 320 CSV 深挖 ====="
$PY - <<'PYEOF'
import csv
rows = list(csv.DictReader(open('stitch_out/stitch_320.csv', encoding='utf-8')))
print('total pairs:', len(rows))

def show(r):
    print(f"{r['tid_a']:>3}->{r['tid_b']:<3} gap={r['gap_s']:>6} "
          f"pe_min={r['pe_min']:>6} pe_fwd={r['pe_fwd']:>6} pe_bwd={r['pe_bwd']:>6} "
          f"cos={r['cos'] or 'n/a':>6} scale={r['scale'] or 'n/a':>5} n={r['n_a']}/{r['n_b']}")

print('--- [A] 18 的后继候选（18->*） ---')
for r in rows:
    if r['tid_a'] == '18':
        show(r)
print('--- [B] 23 的前驱候选（*->23） ---')
for r in rows:
    if r['tid_b'] == '23':
        show(r)
print('--- [C] 8/9/13/16/17 组内两两 ---')
grp = {'8','9','13','16','17'}
for r in rows:
    if r['tid_a'] in grp and r['tid_b'] in grp:
        show(r)
print('--- [D] 26/27 周边 ---')
for r in rows:
    if r['tid_a'] in ('25','26','27') or r['tid_b'] in ('25','26','27'):
        show(r)
PYEOF
echo "===== STITCH2_DONE ====="
