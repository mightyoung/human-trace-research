# -*- coding: utf-8 -*-
"""冲突裁决证据图：对每个「同一 343 事件被判给多个 316 事件」的组，
把 343 端 + 全部候选 316 端并排，标注 dt / cos / pred / relay / 人工判定。

输出：output_crosscam/analysis/truth/_review/conflict_sheet.png
"""
import json
import os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = os.path.join(ROOT, "output_crosscam/analysis/truth")
REV = os.path.join(TR, "_review")
STATE = os.path.join(ROOT, "webapp/backend/data/truth/state.json")

state = json.load(open(STATE, encoding="utf-8"))
pairs = {p["pair_id"]: p for p in state["pairs"]}
seed = json.load(open(os.path.join(ROOT, "webapp/backend/data/truth/seed_labels.json"),
                      encoding="utf-8"))
lab = dict(seed)
lab.update((json.load(open(os.path.join(TR, "remote/labels.json"), encoding="utf-8")) or {})
           .get("pair", {}))
V = {1: "同人", 0: "异人", 2: "看不清"}


def font(sz):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
              "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


F = font(15)
FB = font(18)
TH = os.path.join(TR, "thumbs")


def thumb(eid, h):
    im = Image.open(os.path.join(TH, eid + ".jpg")).convert("RGB")
    w = max(1, round(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)


# 分组
groups = defaultdict(list)
for pid, p in pairs.items():
    if (lab.get(pid) or {}).get("v") == 1:
        groups[p["a"]].append(pid)
groups = {k: v for k, v in groups.items() if len(v) > 1}
# 按 343 事件时间排序
groups = dict(sorted(groups.items(), key=lambda kv: pairs[kv[1][0]]["ta"]))

IH, GAP, PAD, LBL = 240, 14, 16, 46
rows = []
for a, pids in groups.items():
    pids = sorted(pids, key=lambda x: pairs[x]["dt"])
    rows.append((a, pids))

W = PAD * 2 + IH + GAP + max(len(p) for _, p in rows) * (IH + GAP)
H = PAD + 34 + sum(IH + LBL + GAP for _ in rows) + 26
canvas = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(canvas)
d.text((PAD, PAD), "冲突裁决：同一 343 事件被判给多个 316 事件（每组只能留 1 条，或都不是）",
       fill=(20, 20, 20), font=FB)

y = PAD + 34
for a, pids in rows:
    p0 = pairs[pids[0]]
    im = thumb(p0["a"], IH)
    canvas.paste(im, (PAD + (IH - im.width) // 2, y))
    d.text((PAD, y + IH + 3), "343 端", fill=(180, 30, 30), font=F)
    d.text((PAD, y + IH + 20), a.replace("343_E_343-", ""), fill=(20, 20, 20), font=F)
    x = PAD + IH + GAP
    for pid in pids:
        p = pairs[pid]
        im = thumb(p["b"], IH)
        canvas.paste(im, (x + (IH - im.width) // 2, y))
        cos = ("%.3f" % p["cos"]) if p["cos"] > -8 else "缺"
        pred = ("%.3f" % p["pred"]) if p["pred"] > -8 else "缺"
        L = [p["sb"].replace("316_E_316-", "316-"), "dt %.1fs   cos %s" % (p["dt"], cos),
             "pred %s   relay %d" % (pred, p["relay"]),
             "判定: " + V.get((lab.get(pid) or {}).get("v"), "未标")]
        col = (180, 30, 30) if p["relay"] else (20, 20, 20)
        for i, s in enumerate(L):
            d.text((x + 2, y + IH + 3 + i * 14), s, fill=col, font=F)
        x += IH + GAP
    y += IH + LBL + GAP

out = os.path.join(REV, "conflict_sheet.png")
canvas.save(out)
print("wrote", out, canvas.size, "组数", len(rows))
