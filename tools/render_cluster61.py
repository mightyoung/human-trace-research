"""渲染 318 簇 61（椅子上的帽子+背包）的对照图：4 条图库轨迹 + 5 条命中查询。

用途：M3 非人门控 / 真值口径讨论时给标注人看的证据图。
用法：PY tools/render_cluster61.py
输出：deliverable/data-request-v2/gt/cluster61_hat/cluster61_hat.jpg
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "deliverable/data-request-v2/gt")
OUT = os.path.join(GT, "cluster61_hat")
FONT = "C:/Windows/Fonts/msyh.ttc"

MEMBERS = [(1302, 100), (5270, 164), (6394, 43), (6992, 112)]
QUERIES = [(1501, "5.43/5.29"), (916, "4.39/2.86"), (2555, "3.42/4.19"),
           (1951, "2.14/3.01"), (2819, "2.22/2.00")]


def tile(seg, tid):
    """取 8 帧拼图的左上第一帧（270x317），标题条去掉。"""
    im = Image.open(os.path.join(GT, "crops", seg, str(tid) + ".jpg")).convert("RGB")
    return im.crop((0, 32, 270, 32 + 317))


def main():
    os.makedirs(OUT, exist_ok=True)
    f = ImageFont.truetype(FONT, 17) if os.path.exists(FONT) else None
    f2 = ImageFont.truetype(FONT, 20) if os.path.exists(FONT) else None

    items = [("图库 #%d · %d 帧" % (t, n), tile("318_train_525", t)) for t, n in MEMBERS]
    items += [("查询 #%d · %s" % (t, s), tile("318_test_534", t)) for t, s in QUERIES]

    cols, pad, lab = 4, 10, 26
    rows = (len(items) + cols - 1) // cols
    W = cols * (270 + pad) + pad
    H = rows * (317 + lab + pad) + pad + 34
    sheet = Image.new("RGB", (W, H), (22, 24, 28))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), "318 簇 61 = 椅子上的帽子 + 背包：4 条图库轨迹 + 5 条查询，都是这同一件静物",
           fill=(240, 200, 80), font=f2)
    for k, (t, im) in enumerate(items):
        x = pad + (k % cols) * (270 + pad)
        y = pad + 34 + (k // cols) * (317 + lab + pad)
        sheet.paste(im, (x, y))
        d.text((x + 2, y + 319), t, fill=(230, 230, 230), font=f)
    p = os.path.join(OUT, "cluster61_hat.jpg")
    sheet.save(p, quality=90)
    print("→ %s（%dx%d）" % (p, W, H))


if __name__ == "__main__":
    main()
