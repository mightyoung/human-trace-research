# -*- coding: utf-8 -*-
"""把高信号配对候选拼成 A|B 对照图（供人工/机器初判）。一次性辅助脚本。"""
import csv, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = os.path.join(ROOT, "output_crosscam", "analysis", "truth")
REV = os.path.join(TRUTH, "_review")

rows = list(csv.DictReader(open(os.path.join(REV, "pair_machine.csv"), encoding="utf-8-sig")))
rows.sort(key=lambda r: -float(r["cos_appear"]))
sel, seen_a = [], set()
for r in rows:
    if len(sel) < 18 or r["in_relay_v5"] == "1":
        sel.append(r)
sel = sel[:22]

try:
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
except Exception:
    font = ImageFont.load_default()

IH, GAP, PAD, LBL = 230, 10, 8, 34
W = PAD * 2 + (IH + GAP) * 2 + IH  # 两图 + 元信息列


def load(eid):
    p = os.path.join(TRUTH, "thumbs", eid + ".jpg")
    im = Image.open(p).convert("RGB")
    im.thumbnail((IH, IH))
    return im


CHUNK = 7
for ci in range(0, len(sel), CHUNK):
    part = sel[ci:ci + CHUNK]
    canvas = Image.new("RGB", (W, PAD * 2 + len(part) * (IH + LBL + GAP)), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, r in enumerate(part):
        y = PAD + i * (IH + LBL + GAP)
        a, b = load(r["event_a"]), load(r["event_b"])
        canvas.paste(a, (PAD, y + (IH - a.height) // 2))
        canvas.paste(b, (PAD + IH + GAP, y + (IH - b.height) // 2))
        x = PAD + (IH + GAP) * 2
        d.multiline_text((x, y + 6), "cos=%.3f  dt=%.1fs\n%s\n%s\nrelay=%s  %s"
                         % (float(r["cos_appear"]), float(r["dt"]),
                            r["event_a"].split("_E_")[1], r["event_b"].split("_E_")[1],
                            r["in_relay_v5"], r["machine"]), fill=(0, 0, 0), font=font)
    out = os.path.join(REV, "pair_sheet_%d.png" % (ci // CHUNK + 1))
    canvas.save(out)
    print("wrote", out, canvas.size, "rows", len(part))
