#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 1-NN 复核网页（单文件，图片 base64 内嵌）。

与 make_review_page.py 的区别：
  1. 读 knn_hit_audit/index.csv（字段名 cos_a/cos_b/hit_a/hit_b 与旧版不同，自动适配）；
  2. 右栏顶部多一块「判决对比」——把 query 代表帧与命中成员代表帧并排裁出放大，
     这是判「对/错」时唯一需要看清的两个对象；
  3. localStorage key 独立（idrev_knn），不会与上一轮 33 条的记录串。

用法：
  python tools/make_knn_review_page.py --embed
输出：
  selector_osnet_1012/knn_hit_audit/review_embed.html
"""
import base64
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/knn_hit_audit")
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
VID = os.path.join(ROOT, "media-resource/1012/segments")
EMBED = "--embed" in sys.argv

rows = list(csv.DictReader(open(os.path.join(D, "index.csv"), encoding="utf-8-sig")))


def b64(path, max_w=1280, q=72, reencode=True):
    if not reencode:
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def file_uri(path):
    """小图直接用原字节，不二次压缩。"""
    return b64(path, reencode=False)


# ---------- 判决对比小图：query 中间帧 vs 命中成员中间帧 ----------
_det_cache, _cap_cache = {}, {}


def load_dets(seg):
    if seg not in _det_cache:
        d = {}
        with open(os.path.join(DET, seg, f"{seg}_detections.jsonl"), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                d[o["frame"]] = (o["track_ids"], o["boxes"])
        _det_cache[seg] = d
    return _det_cache[seg]


def crop_of(seg, tid, size=300):
    dets = load_dets(seg)
    fr = sorted(f for f, (ids, _) in dets.items() if tid in ids)
    if not fr:
        return None
    f = fr[len(fr) // 2]
    if seg not in _cap_cache:
        _cap_cache[seg] = cv2.VideoCapture(os.path.join(VID, f"{seg}.mp4"))
    cap = _cap_cache[seg]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        return None
    ids, boxes = dets[f]
    x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[ids.index(tid)]]
    H, W = img.shape[:2]
    pad = max(8, int(0.12 * max(x2 - x1, y2 - y1)))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    s = size / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_CUBIC)
    out = np.full((size, size, 3), 22, np.uint8)
    y0, x0 = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return out


def pair_image(seg, tid, hit):
    """左 = query，右 = 命中成员；同一高度并排。"""
    a = crop_of(seg, tid)
    hs, ht = hit.split("#")
    b = crop_of(hs, int(ht))
    if a is None or b is None:
        return None
    S = 300
    canvas = np.full((S + 26, S * 2, 3), 245, np.uint8)
    canvas[26:26 + S, 0:S] = a
    canvas[26:26 + S, S:S * 2] = b
    cv2.putText(canvas, "QUERY", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    cv2.putText(canvas, "GALLERY HIT", (S + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    cv2.rectangle(canvas, (0, 26), (S - 1, 26 + S - 1), (80, 200, 80), 2)
    cv2.rectangle(canvas, (S, 26), (S * 2 - 1, 26 + S - 1), (60, 200, 230), 2)
    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


DATA = []
for r in rows:
    d = dict(seg=r["seg"], tid=r["tid"], cam=r["cam"], cl=r["cluster"],
             sz=r.get("cluster_size", ""), hit=r.get("hit_a", ""), hb=r.get("hit_b", ""),
             ca=r.get("cos_a") or r.get("abs_cos_a", ""),
             cb=r.get("cos_b") or r.get("abs_cos_b", ""),
             note=r.get("note", ""))
    p = os.path.join(D, r["image"])
    d["img"] = b64(p, q=76) if EMBED else r["image"]
    d["pair"] = pair_image(r["seg"], int(r["tid"]), r["hit_a"]) if EMBED else ""
    DATA.append(d)
DATA = json.dumps(DATA, ensure_ascii=False)

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>1-NN 复核 · 18 条</title>
<style>
:root{--bg:#1e1f22;--panel:#2b2d31;--line:#3a3d43;--fg:#e6e6e6;--dim:#9aa0a6;
--ok:#3fb950;--bad:#f85149;--unk:#d29922;--acc:#4c8dff;--hit:#e8c33f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif}
header{display:flex;align-items:center;gap:14px;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:9}
header b{font-size:15px;white-space:nowrap}
.bar{flex:1;height:6px;background:#3a3d43;border-radius:3px;overflow:hidden;min-width:60px}
.bar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .2s}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px;padding:14px}
.view{background:#111;border:1px solid var(--line);border-radius:8px;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:300px}
.view img{max-width:100%;max-height:calc(100vh - 120px);object-fit:contain;display:block}
@media (max-width:1000px){
  main{grid-template-columns:1fr}
  aside{order:2}
  .view img{max-height:none;width:100%}
}
aside{display:flex;flex-direction:column;gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card h4{margin:0 0 8px;font-size:13px;color:var(--dim);font-weight:600}
.meta div{display:flex;justify-content:space-between;padding:2px 0;color:var(--dim);font-size:13px;gap:10px}
.meta div span:last-child{color:var(--fg);text-align:right}
.pair img{width:100%;border-radius:6px;display:block}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{cursor:pointer;border:1px solid var(--line);background:#33363c;color:var(--fg);
padding:11px 8px;border-radius:8px;font-size:14px}
button:hover{background:#3d4046}
button.ok{border-color:var(--ok)} button.bad{border-color:var(--bad)} button.unk{border-color:var(--unk)}
button.wide{grid-column:1/-1}
.hint{font-size:12px;color:var(--dim);line-height:1.65}
.hint b{color:#d8d8d8}
.strip{display:flex;flex-wrap:wrap;gap:4px}
.strip b{width:27px;height:22px;display:grid;place-items:center;border-radius:4px;background:#33363c;
font:11px monospace;color:var(--dim);cursor:pointer;border:1px solid transparent}
.strip b.cur{border-color:var(--acc)}
.strip b.ok{background:#1f3d24;color:#7ee787}
.strip b.bad{background:#4a1f1f;color:#ff9b95}
.strip b.unk{background:#43391a;color:#e3b341}
textarea{width:100%;height:70px;background:#111;color:var(--fg);border:1px solid var(--line);border-radius:8px;
padding:8px;font:12px monospace;resize:none}
kbd{background:#3d4046;border-radius:4px;padding:1px 5px;font:11px monospace}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--dim)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style></head><body>
<header><b>1-NN 复核</b><span id="pos" style="color:#9aa0a6;white-space:nowrap"></span>
<div class="bar"><i id="pb"></i></div>
<button id="copy" style="padding:6px 12px;white-space:nowrap">复制结果</button></header>
<main>
<div class="view"><img id="img" alt=""></div>
<aside>
  <div class="card pair"><h4>判决对比（只看这两个）</h4><img id="pair" alt=""></div>
  <div class="card meta" id="meta"></div>
  <div class="btns">
    <button class="ok" data-v="ok">对 <kbd>1</kbd></button>
    <button class="bad" data-v="bad">错 <kbd>2</kbd></button>
    <button class="unk" data-v="unk">看不清 <kbd>3</kbd></button>
    <button class="wide" data-v="skip">跳过 <kbd>空格</kbd></button>
  </div>
  <button class="wide" id="prev">上一条 <kbd>←</kbd></button>
  <div class="card hint">
    <b>判据只有一条</b>：上面「判决对比」里，右边那个人（黄框）和左边（绿框）是不是同一个人。<br>
    衣着、体型、发型、姿态任一明显不同 → <b>错</b>；只有轮廓/剪影、看不清 → <b>看不清</b>。<br>
    下图第 3 行：<span style="color:var(--hit)">黄=真正命中的成员</span>，橙=后半段命中（若不同），
    蓝=同簇其他成员（只作参考，系统没用它们比较）。
  </div>
  <div class="card">
    <div class="legend"><span><i style="background:#50c850"></i>绿=query</span>
      <span><i style="background:#e8c33f"></i>黄=命中</span>
      <span><i style="background:#ffa53c"></i>橙=后半段命中</span>
      <span><i style="background:#3ca6e6"></i>蓝=同簇其他</span></div>
    <div class="strip" id="strip" style="margin-top:8px"></div>
  </div>
  <textarea id="out" readonly placeholder="完成后这里出现结果串，点右上角复制"></textarea>
</aside>
</main>
<script>
const D = __DATA__, K = ['','ok','bad','unk'], SAVE='idrev_knn';
let i = 0, ans = new Array(D.length).fill(0);
const el = id => document.getElementById(id);
function draw(){
  const d = D[i];
  el('img').src = d.img;
  if(d.pair) el('pair').src = d.pair;
  el('pos').textContent = (i+1) + ' / ' + D.length;
  el('pb').style.width = (100*(ans.filter(x=>x).length/D.length)) + '%';
  el('meta').innerHTML = [
    ['轨迹', d.seg + ' #' + d.tid],
    ['机位', d.cam],
    ['判给簇', 'c' + d.cl + '（' + d.sz + ' 条成员）'],
    ['命中成员', d.hit + (d.hb && d.hb !== d.hit ? ' ／ ' + d.hb : '')],
    ['余弦 a/b', d.ca + ' / ' + d.cb]
  ].map(x=>`<div><span>${x[0]}</span><span>${x[1]}</span></div>`).join('');
  el('strip').querySelectorAll('b').forEach((b,k)=>{
    b.className = (K[ans[k]] || '') + (k===i ? ' cur' : '');
  });
  el('out').value = outStr();
}
function outStr(){ return D.map((d,k)=>String(k+1).padStart(2,'0')+'='+(K[ans[k]]||'?')).join(' '); }
function set(v,adv=true){
  ans[i] = v==='ok'?1:v==='bad'?2:v==='unk'?3:0;
  if(adv && i<D.length-1) i++;
  draw(); save();
}
function save(){ try{ localStorage.setItem(SAVE, JSON.stringify(ans)); }catch(e){} }
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>set(b.dataset.v, b.dataset.v!=='skip'));
el('prev').onclick=()=>{ if(i>0){i--;draw();} };
el('copy').onclick=()=>{
  const t = el('out'); t.removeAttribute('readonly'); t.select(); t.setSelectionRange(0,99999);
  try{ document.execCommand('copy'); }catch(e){}
  if(navigator.clipboard) navigator.clipboard.writeText(t.value).catch(()=>{});
  t.setAttribute('readonly','');
  el('copy').textContent='已复制'; setTimeout(()=>el('copy').textContent='复制结果',1400);
};
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='TEXTAREA') return;
  if(e.key==='1'){e.preventDefault();set('ok');}
  else if(e.key==='2'){e.preventDefault();set('bad');}
  else if(e.key==='3'){e.preventDefault();set('unk');}
  else if(e.key===' '){e.preventDefault();set('skip');}
  else if(e.key==='ArrowLeft' && i>0){e.preventDefault();i--;draw();}
  else if(e.key==='ArrowRight' && i<D.length-1){e.preventDefault();i++;draw();}
});
const strip = el('strip');
D.forEach((d,k)=>{ const b=document.createElement('b'); b.textContent=k+1;
  b.onclick=()=>{ i=k; draw(); }; strip.appendChild(b); });
try{ const s=JSON.parse(localStorage.getItem(SAVE)||'[]'); if(s.length===D.length) ans=s; }catch(e){}
draw();
</script></body></html>"""

name = "review_embed.html" if EMBED else "review.html"
html = HTML.replace("__DATA__", DATA)
out = os.path.join(D, name)
open(out, "w", encoding="utf-8").write(html)
print("[out]", out, len(html) // 1024, "KB")

for c in _cap_cache.values():
    c.release()
