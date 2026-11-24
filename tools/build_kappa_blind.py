#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「时空矛盾对」的**第二标注人盲标页**，用来给 153 对补 Cohen's κ。

为什么需要它
------------
论文的 153 对「同人/异人」判读由一个人完成，没有 κ —— 这是审稿人一定会问的点
（图库层已有 κ=0.925，见 `results/kappa_a3.json`；**对级那层没有**）。

这份页面只做一件事：让第二个人在**看不到第一人结论**的情况下独立判一遍。

设计约束（每条都对应踩过的坑）
------------------------------
- **不复用原页面**：原页面的项顺序、缓存键与第一人已看过的界面相同，容易产生
  记忆污染；这里重新洗牌并用固定 seed 记录映射。
- **不显示任何标签/组号/重叠时长**：重叠时长会暗示「同一空间同时在场 →
  不同人」，必须藏起来。只给 A/B 两侧相同时刻的裁图。
- **同一时刻三帧**：同一人在同一瞬间长相不会变；两个人三个时刻都应能看出差异。
- **必须能离线**：复核人不在工作机上，图片 base64 内嵌。
- **导出 JSON + CSV**：CSV 供人工核对，JSON 供脚本算 κ。
"""
import argparse
import base64
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "deliverable/spacetime_review"
CACHE = SRC / ".cache"
OUT = SRC / "kappa2"

CSS = """
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
     background:#f6f7f9;color:#1a1c1f}
header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #e3e6ea;
       padding:10px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0}
.bar{height:6px;background:#e3e6ea;border-radius:3px;flex:1;min-width:120px}
.bar>i{display:block;height:100%;background:#2f6f4f;border-radius:3px;width:0}
.tag{font-size:12px;color:#5b6470}
.wrap{max-width:900px;margin:0 auto;padding:14px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:10px;margin:0 0 16px;overflow:hidden}
.card h2{font-size:14px;margin:0;padding:10px 14px;background:#fafbfc;border-bottom:1px solid #eef0f3;
         display:flex;justify-content:space-between;font-weight:600}
.grid{display:grid;grid-template-columns:76px repeat(3,1fr);gap:6px;padding:10px 12px}
.side{font-weight:600;font-size:13px;color:#39414b;display:flex;align-items:center}
.grid img{width:100%;border-radius:6px;background:#111;display:block;cursor:zoom-in}
.cap{grid-column:1/-1;font-size:12px;color:#6b7480;padding-top:2px}
.btns{display:flex;gap:8px;padding:0 12px 12px;flex-wrap:wrap}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #cfd5dc;background:#fff;
       cursor:pointer}
button.on{background:#2f6f4f;color:#fff;border-color:#2f6f4f}
button.un{background:#8a5a1f;color:#fff;border-color:#8a5a1f}
button.di{background:#8d2f2f;color:#fff;border-color:#8d2f2f}
footer{padding:18px 0 60px;display:flex;gap:10px;flex-wrap:wrap}
#zoom{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:99}
#zoom img{max-width:96vw;max-height:96vh}
.note{background:#fff8e6;border:1px solid #f0dfae;border-radius:8px;padding:10px 12px;font-size:13px}
"""

JS = """
const KEY='kappa2_spacetime_v1';
let st=JSON.parse(localStorage.getItem(KEY)||'{}');
if(!st.annotator) st.annotator='';
function save(){localStorage.setItem(KEY,JSON.stringify(st));}
function paint(){
  document.querySelectorAll('.card').forEach(c=>{
    const id=c.dataset.id, v=st.ans?st.ans[id]:null;
    c.querySelectorAll('button').forEach(b=>b.className=(b.dataset.v===v?'on':''));
  });
  const n=Object.keys(st.ans||{}).length, t=document.querySelectorAll('.card').length;
  document.querySelector('.bar>i').style.width=(100*n/t)+'%';
  document.querySelector('#cnt').textContent=n+' / '+t;
}
function setv(id,v){st.ans=st.ans||{};st.ans[id]=v;
  if(v!==null){const n=Object.keys(st.ans).length;const t=document.querySelectorAll('.card').length;
   if(n===t){autoUp();}} save();paint();}
function autoUp(){}
function exportJSON(){
  const blob=new Blob([JSON.stringify({annotator:st.annotator,ans:st.ans||{}},null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='kappa2_verdicts.json';a.click();
}
function exportCSV(){
  let r=['item_id,verdict\\n'];
  document.querySelectorAll('.card').forEach(c=>{
    const id=c.dataset.id; r.push(id+','+((st.ans||{})[id]||'')+'\\n');});
  const blob=new Blob([r.join('')],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='kappa2_verdicts.csv';a.click();
}
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelector('#who').value=st.annotator||'';
  document.querySelector('#who').addEventListener('input',e=>{st.annotator=e.target.value;save();});
  document.querySelectorAll('.card').forEach(c=>{
    c.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
      const v=b.dataset.v==='clear'?null:b.dataset.v;
      setv(c.dataset.id,v);}));
  });
  document.addEventListener('click',e=>{
    if(e.target.tagName==='IMG'&&e.target.closest('.grid')){
      const z=document.querySelector('#zoom');
      z.querySelector('img').src=e.target.src;z.style.display='flex';}
    if(e.target.id==='zoom'||e.target.parentElement?.id==='zoom'){
      document.querySelector('#zoom').style.display='none';}
  });
  paint();
});
"""


def b64(p):
    return "data:image/webp;base64," + base64.b64encode(Path(p).read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--per", type=int, default=0, help="只取前 N 条（0=全部）")
    a = ap.parse_args()

    items = json.load(open(SRC / "items.json", encoding="utf-8"))["items"]
    if a.per:
        items = items[:a.per]
    order = list(range(len(items)))
    random.Random(a.seed).shuffle(order)

    cards, mapping = [], []
    for shown, idx in enumerate(order, 1):
        it = items[idx]
        iid = it["id"]
        rows = []
        for si, s in enumerate(it["sides"]):
            side_name = "A" if si == 0 else "B"
            imgs = []
            for k in range(3):
                # 文件名形如 sp01_305173_detail0_f23485.webp
                tag = "%s_%s%s" % (iid, s["cam"], s["tid"])
                cand = list(CACHE.glob("%s_detail%d_*.webp" % (tag, k)))
                if not cand:
                    continue
                imgs.append('<img src="%s" alt="">' % b64(cand[0]))
            rows.append('<div class="side">%s</div>%s' % (side_name, "".join(imgs)))
        cards.append("""
<div class="card" data-id="%s">
  <h2>第 %d 项<span class="tag">两侧同一时刻，共 3 个时刻</span></h2>
  <div class="grid">%s
    <div class="cap">判断两侧是否同一个人，点图可放大。</div>
  </div>
  <div class="btns">
    <button data-v="same">同一人</button>
    <button data-v="diff">不同人</button>
    <button data-v="unk">看不清</button>
    <button data-v="clear">清除</button>
  </div>
</div>""" % (iid, shown, "".join(rows)))
        mapping.append(dict(shown=shown, id=iid))

    html = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>时空矛盾对盲标 · 第二标注人</title><style>%s</style></head><body>
<header><h1>时空矛盾对盲标</h1><div class="bar"><i></i></div>
<span class="tag" id="cnt">0 / %d</span>
<input id="who" placeholder="标注人代号（必填）" style="padding:6px 8px;border:1px solid #cfd5dc;border-radius:8px">
</header>
<div class="wrap">
<div class="note">每项只判一件事：<b>两侧是不是同一个人</b>。依据长相、发型、衣着、体型。
不要因为「两边同时出现在画面里」就判不同人——同一片开放空间会被两侧同时拍到。
看不清就选「看不清」，不要猜。全程约 20–30 分钟。</div>
%s
<footer><button onclick="exportJSON()">导出 JSON</button>
<button onclick="exportCSV()">导出 CSV</button>
<span class="tag">导出后把文件交回即可，进度自动存在本机浏览器。</span></footer>
</div>
<div id="zoom"><img alt=""></div>
<script>%s</script></body></html>""" % (CSS, len(cards), "".join(cards), JS)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "kappa2_blind.html"
    p.write_text(html, encoding="utf-8")
    json.dump(dict(seed=a.seed, mapping=mapping),
              open(OUT / "shuffle_map.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s  (%.1f MB, %d 项)" % (p, p.stat().st_size / 1e6, len(cards)))


if __name__ == "__main__":
    main()
