#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成人工复核网页。

用法：
  python tools/make_review_page.py          # 引用相对路径（同目录直接打开有效）
  python tools/make_review_page.py --embed  # 图片压成 base64 内嵌（预览面板/单文件传输可用）

输出：
  selector_osnet_1012/identity_audit/review.html
  selector_osnet_1012/identity_audit/review_embed.html   （--embed）
"""
import base64
import csv
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/identity_audit")
if "--dir" in sys.argv:
    D = sys.argv[sys.argv.index("--dir") + 1]
    if not os.path.isabs(D):
        D = os.path.join(ROOT, D)
rows = list(csv.DictReader(open(os.path.join(D, "index.csv"), encoding="utf-8-sig")))
EMBED = "--embed" in sys.argv
SAVE_KEY = "rev_" + os.path.basename(D.rstrip("/\\"))
TITLE = "身份复核"
if "--title" in sys.argv:
    TITLE = sys.argv[sys.argv.index("--title") + 1]


def data_uri(path, max_w=1280, q=72):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


DATA = json.dumps([{
    "img": data_uri(os.path.join(D, r["image"])) if EMBED else r["image"],
    "seg": r["seg"], "tid": r["tid"], "cam": r["cam"],
    "ca": r["abs_cos_a"], "cb": r["abs_cos_b"], "cl": r["cluster"],
    "sz": r["cluster_size"], "ai": r["verdict"].replace("AI预判:", ""), "note": r["note"],
} for r in rows], ensure_ascii=False)

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>__TITLE__ · __N__ 条</title>
<style>
:root{--bg:#1e1f22;--panel:#2b2d31;--line:#3a3d43;--fg:#e6e6e6;--dim:#9aa0a6;
--ok:#3fb950;--bad:#f85149;--unk:#d29922;--acc:#4c8dff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:16px;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
header b{font-size:15px}
.bar{flex:1;height:6px;background:#3a3d43;border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .2s}
main{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:14px;padding:14px;height:calc(100vh - 58px)}
.view{background:#111;border:1px solid var(--line);border-radius:8px;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:280px}
.view img{max-width:100%;max-height:100%;object-fit:contain;display:block}
@media (max-width:900px){
  main{grid-template-columns:1fr;height:auto}
  .view{height:58vh;min-height:260px}
  aside{order:2}
}
aside{display:flex;flex-direction:column;gap:10px}
.meta{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:13px}
.meta div{display:flex;justify-content:space-between;padding:2px 0;color:var(--dim)}
.meta div span:last-child{color:var(--fg)}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{cursor:pointer;border:1px solid var(--line);background:#33363c;color:var(--fg);
padding:10px 8px;border-radius:8px;font-size:14px}
button:hover{background:#3d4046}
button.ok{border-color:var(--ok)} button.bad{border-color:var(--bad)} button.unk{border-color:var(--unk)}
button.wide{grid-column:1/-1}
.hint{font-size:12px;color:var(--dim)}
.ai{background:#24262a;border:1px dashed var(--line);border-radius:8px;padding:8px 10px;font-size:12px;color:var(--dim)}
.ai b{color:#d0d0d0}
.strip{display:flex;flex-wrap:wrap;gap:4px}
.strip b{width:26px;height:22px;display:grid;place-items:center;border-radius:4px;background:#33363c;
font:11px monospace;color:var(--dim);cursor:pointer}
.strip b.ok{background:#1f3d24;color:#7ee787}
.strip b.bad{background:#4a1f1f;color:#ff9b95}
.strip b.unk{background:#43391a;color:#e3b341}
textarea{width:100%;height:74px;background:#111;color:var(--fg);border:1px solid var(--line);border-radius:8px;
padding:8px;font:12px monospace;resize:none}
kbd{background:#3d4046;border-radius:4px;padding:1px 5px;font:11px monospace}
</style></head><body>
<header><b>__TITLE__</b><span id="pos" style="color:#9aa0a6"></span><div class="bar"><i id="pb"></i></div>
<button id="copy" style="padding:6px 12px">复制结果</button></header>
<main>
<div class="view"><img id="img" alt=""></div>
<aside>
  <div class="meta" id="meta"></div>
  <div class="btns">
    <button class="ok" data-v="ok">对 <kbd>1</kbd></button>
    <button class="bad" data-v="bad">错 <kbd>2</kbd></button>
    <button class="unk" data-v="unk">看不清 <kbd>3</kbd></button>
    <button class="wide" data-v="skip">跳过 <kbd>空格</kbd></button>
  </div>
  <button class="wide" id="prev">上一条 <kbd>←</kbd></button>
  <div class="ai"><b>AI 预判（先别看）</b><br><span id="ai">点击右侧「显示」再展开，避免先入为主</span>
    <button id="showai" style="margin-top:6px;padding:4px 8px;font-size:12px">显示</button></div>
  <div class="hint">判断标准：只问一件事——<b>第 3 行的 gallery 里有没有和第 1/2 行同一个人</b>。
  衣着、体型、发型、步态任一明显不同就是「错」；框里没有完整人形就是「看不清」。</div>
  <div class="strip" id="strip"></div>
  <textarea id="out" readonly placeholder="完成后这里会出现结果串，点右上角复制"></textarea>
</aside>
</main>
<script>
const D = __DATA__, K = ['','ok','bad','unk'];
let i = 0, ans = new Array(D.length).fill(0);
const el = id => document.getElementById(id);
function draw(){
  const d = D[i];
  el('img').src = d.img;
  el('pos').textContent = (i+1) + ' / ' + D.length;
  el('pb').style.width = (100*(ans.filter(x=>x).length/D.length)) + '%';
  el('meta').innerHTML = [
    ['轨迹', d.seg + ' #' + d.tid], ['机位', d.cam],
    ['绝对余弦 a/b', d.ca + ' / ' + d.cb], ['分配簇', d.cl + '（' + d.sz + ' 条）']
  ].map(x=>`<div><span>${x[0]}</span><span>${x[1]}</span></div>`).join('');
  el('ai').textContent = d.ai + (d.note ? '｜' + d.note : '');
  el('ai').style.display = 'none';
  el('strip').querySelectorAll('b').forEach((b,k)=>b.className = K[ans[k]] || '');
  el('out').value = outStr();
}
function outStr(){
  return D.map((d,k)=> (k+1).toString().padStart(2,'0') + '=' + (K[ans[k]] || '?')).join(' ');
}
function set(v,adv=true){
  ans[i] = v==='ok'?1:v==='bad'?2:v==='unk'?3:0;
  if(adv){ if(i<D.length-1){ i++; draw(); } else draw(); } else draw();
  save();
}
function save(){ try{ localStorage.setItem('__KEY__', JSON.stringify(ans)); }catch(e){} }
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>set(b.dataset.v, b.dataset.v!=='skip'));
el('prev').onclick=()=>{ if(i>0){i--;draw();} };
el('showai').onclick=()=>{ const a=el('ai'); a.style.display = a.style.display==='none'?'block':'none'; };
el('copy').onclick=()=>{ el('out').select(); document.execCommand('copy');
  el('copy').textContent='已复制'; setTimeout(()=>el('copy').textContent='复制结果',1200); };
document.addEventListener('keydown', e=>{
  if(e.key==='1') set('ok'); else if(e.key==='2') set('bad'); else if(e.key==='3') set('unk');
  else if(e.key===' '){ e.preventDefault(); set('skip'); } else if(e.key==='ArrowLeft' && i>0){ i--; draw(); }
});
const strip = el('strip');
D.forEach((d,k)=>{ const b=document.createElement('b'); b.textContent=k+1;
  b.onclick=()=>{ i=k; draw(); }; strip.appendChild(b); });
try{ const s=JSON.parse(localStorage.getItem('__KEY__')||'[]'); if(s.length===D.length) ans=s; }catch(e){}
draw(); el('out').value = outStr();
</script></body></html>"""

html = (HTML.replace("__DATA__", DATA)
        .replace("__KEY__", SAVE_KEY)
        .replace("__TITLE__", TITLE)
        .replace("__N__", str(len(rows))))
name = "review_embed.html" if EMBED else "review.html"
open(os.path.join(D, name), "w", encoding="utf-8").write(html)
print("[out]", os.path.join(D, name), len(html) // 1024, "KB")
