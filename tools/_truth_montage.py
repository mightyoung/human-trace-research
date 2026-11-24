# -*- coding: utf-8 -*-
"""生成 truth 事件的缩略图接触表（供人工快速浏览）。一次性辅助脚本。"""
import csv, os, sys
from PIL import Image, ImageDraw, ImageFont

TRUTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "output_crosscam", "analysis", "truth")
OUT = os.path.join(TRUTH, "_review")
os.makedirs(OUT, exist_ok=True)


def rd(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


seen = {}
for r in rd(os.path.join(TRUTH, "events_all.csv")):
    seen.setdefault(r["event_id"], r)
ids = sorted(seen, key=lambda k: float(seen[k]["t"]))

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
except Exception:
    font = ImageFont.load_default()

CW, CH, COLS, PAD, LBL = 132, 176, 9, 4, 32


def grid(items, out):
    rows = (len(items) + COLS - 1) // COLS
    W = COLS * (CW + PAD) + PAD
    H = rows * (CH + LBL + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (eid, meta) in enumerate(items):
        cx = PAD + (i % COLS) * (CW + PAD)
        cy = PAD + (i // COLS) * (CH + LBL + PAD)
        p = os.path.join(TRUTH, "thumbs", eid + ".jpg")
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((CW, CH))
            canvas.paste(im, (cx + (CW - im.width) // 2, cy + (CH - im.height) // 2))
        except Exception:
            d.rectangle([cx, cy, cx + CW, cy + CH], outline=(255, 0, 0))
        d.multiline_text((cx + 2, cy + CH + 1),
                         "%s\n%s/%s t=%.0f" % (eid.split("_E_")[1], meta["kind"], meta["edge"], float(meta["t"])),
                         fill=(0, 0, 0), font=font)
    canvas.save(out)
    print(out, canvas.size, len(items))


for cam in ("343", "316"):
    grid([(e, seen[e]) for e in ids if seen[e]["cam"] == cam],
         os.path.join(OUT, "contact_%s.png" % cam))
