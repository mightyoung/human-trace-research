#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 D3 收尾 20 条的专用标注页。

对象：4 个从未标注的孤立簇（61 / 32 / 34 / 16）。
设计要点：
  - 按簇分组、簇内按间隙升序，便于「同簇是否同一人」一眼判断
  - 支持「本簇全部同一人」批量确认（簇 61 的 d_max 仅 0.018，极可能成立）
  - 主图用原分辨率，可点击开新窗看原图（323 远场必须能放大）
  - 身份库只放已建档的 29 个身份，点图即选
"""
import base64
import collections
import csv
import io
import json
import os

from PIL import Image

GT = "deliverable/data-request-v2/gt"
OUT = os.path.join(GT, "review_d3_tail.html")

# 簇顺序：越紧越先标（d_max 升序），便于建立信心
ORDER = [61, 32, 34, 16]


def b64(jpg_rel, maxw=None, quality=72, fmt="WEBP"):
    p = os.path.join(GT, jpg_rel)
    if not os.path.exists(p):
        return ""
    im = Image.open(p).convert("RGB")
    if maxw:
        im.thumbnail((maxw, 20000), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, fmt, quality=quality, method=4)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    d3 = list(csv.DictReader(open(os.path.join(GT, "gt_gallery_members_FILLED.csv"),
                                  encoding="utf-8-sig")))
    tpl = list(csv.DictReader(open(os.path.join(GT, "gallery_members_TEMPLATE.csv"),
                                   encoding="utf-8-sig")))
    done = {r["track_id"] for r in d3}
    blank = [r for r in d3 if not r["person_id"].strip()]
    miss = [r for r in tpl if r["track_id"] not in done]
    for r in miss:
        r["person_id"] = ""
        r["unk_reason"] = ""
        r["n_frames"] = ""
        r["box_type"] = "body"

    # 用 tracks.csv 补 n_frames / box_type
    tracks = {r["track_id"]: r for r in csv.DictReader(
        open("deliverable/data-request-v2/ledger/tracks.csv", encoding="utf-8-sig"))}
    items = blank + miss
    for r in items:
        t = tracks.get(r["track_id"], {})
        if not r.get("n_frames"):
            r["n_frames"] = t.get("n_frames", "")
        if not r.get("box_type"):
            r["box_type"] = t.get("box_type_auto", "body")

    geo = {int(r["cluster_idx"]): r for r in csv.DictReader(
        open("deliverable/data-request-v2/ledger/clusters_geom.csv", encoding="utf-8-sig"))
        if r["config"] == "baseline"}

    # 排序：簇优先级 → 段 → track_id
    def key(r):
        c = int(r["cluster_idx"])
        p = ORDER.index(c) if c in ORDER else 99
        seg, tid = r["track_id"].split("#")
        return (p, c, seg, int(tid))

    items.sort(key=key)

    imgs = {}
    out = []
    for r in items:
        jp = r.get("prompt_jpg") or ""
        if not jp:
            seg, tid = r["track_id"].split("#")
            jp = f"crops/{seg}/{tid}.jpg"
        if jp not in imgs:
            imgs[jp] = b64(jp, quality=74)
        c = int(r["cluster_idx"])
        out.append({
            "k": r["track_id"], "hits": r.get("hits_by", ""),
            "cl": c, "cam": r["track_id"].split("_")[0],
            "n": r.get("n_frames", ""), "box": r.get("box_type", ""),
            "old_unk": r.get("unk_reason", "") or "",
            "jpg": jp,
        })

    # 簇内兄弟（同簇其他待标成员），供横向比对
    for x in out:
        x["sibs"] = [y["k"] for y in out if y["cl"] == x["cl"] and y["k"] != x["k"]]

    groups = []
    for c in ORDER:
        sub = [x for x in out if x["cl"] == c]
        if not sub:
            continue
        g = geo.get(c, {})
        groups.append({"cl": c, "n": len(sub), "dmax": g.get("d_max", ""),
                       "cams": g.get("cams", ""), "keys": [x["k"] for x in sub]})

    # 身份库
    pool = json.load(open("tmp/pool.json", encoding="utf-8"))
    pimgs = {}
    for p in pool:
        if p["jpg"] not in pimgs:
            pimgs[p["jpg"]] = b64(p["jpg"], maxw=300, quality=62)
        imgs.setdefault(p["jpg"], pimgs[p["jpg"]])
        p["img"] = p["jpg"]

    data = {"items": out, "groups": groups, "pool": pool,
            "imgs": imgs, "order": ORDER}
    tpl_html = TPL.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                                   separators=(",", ":")))
    open(OUT, "w", encoding="utf-8").write(tpl_html)
    total = os.path.getsize(OUT)
    print(f"[out] {OUT}")
    print(f"  {len(out)} 条 / {len(groups)} 簇 / 图 {len(imgs)} 张 / {total/1024/1024:.1f} MB")
    for g in groups:
        print(f"    簇 {g['cl']:>3}: {g['n']} 条  d_max={g['dmax']}  {g['cams']}")


TPL = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>D3 收尾 20 条</title>
<style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;font:14px/1.6 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;background:#f4f5f7;color:#171a20}
header{background:#181d27;color:#fff;padding:10px 16px;position:sticky;top:0;z-index:20;
       display:flex;align-items:center;gap:14px;flex-wrap:wrap}
header h1{margin:0;font-size:15px;font-weight:600;white-space:nowrap}
.gtabs{display:flex;gap:6px;flex-wrap:wrap}
.gtab{background:#2a3242;color:#c7d0de;border:1px solid #39445a;border-radius:14px;
      padding:3px 10px;font-size:12px;cursor:pointer;white-space:nowrap}
.gtab.on{background:#2563eb;border-color:#2563eb;color:#fff}
.gtab .done{color:#7ee2a8;margin-left:4px}
#bar{margin-left:auto;font-size:12.5px;color:#9fb0c6}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:14px;padding:14px;align-items:start}
.panel{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:12px}
.meta{font-size:13px;color:#4b5563;margin-bottom:8px;line-height:1.7}
.meta b{color:#0f172a}
.tag{display:inline-block;padding:1px 7px;border-radius:5px;font-size:12px;background:#e8ebf0;
     color:#3b4557;margin-right:5px}
.tag.hot{background:#fee2e2;color:#991b1b}
.tag.cool{background:#e0f2fe;color:#075985}
#shot{width:100%;display:block;border-radius:8px;background:#0d1016;cursor:zoom-in}
.hint{font-size:12px;color:#8b95a5;margin-top:6px}
.sibs{margin-top:10px}
.sibs h3{font-size:12.5px;color:#6b7280;margin:0 0 6px;font-weight:600}
.sibrow{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
.sibrow figure{margin:0;flex:0 0 148px;cursor:pointer;border:2px solid transparent;border-radius:7px;overflow:hidden;background:#f8f9fb}
.sibrow figure.sel{border-color:#2563eb}
.sibrow img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#111}
.sibrow cap,.sibrow span{display:block;font-size:11px;color:#6b7280;padding:2px 4px}
h2{font-size:13px;margin:0 0 8px;color:#374151;font-weight:600;display:flex;align-items:center;gap:8px}
h2 input{flex:1;font:inherit;font-size:12.5px;padding:4px 8px;border:1px solid #d8dde5;border-radius:6px;min-width:0}
#pool{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:7px;max-height:calc(100vh - 220px);overflow:auto}
.card{border:1px solid #e3e7ee;border-radius:7px;overflow:hidden;background:#fafbfc;cursor:pointer}
.card:hover{border-color:#2563eb;box-shadow:0 0 0 2px #dbeafe}
.card img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#111}
.card b{display:block;font-size:11.5px;color:#111827;padding:2px 4px}
.card i{display:block;font-size:10px;color:#8b95a5;padding:0 4px 3px;font-style:normal}
.ops{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}
button{font:inherit;padding:7px 13px;border-radius:8px;border:1px solid #d3d9e3;background:#fff;cursor:pointer}
button:hover{background:#f1f4f8}
button.primary{background:#2563eb;border-color:#2563eb;color:#fff;font-weight:600}
button.primary:hover{background:#1d4ed8}
button.batch{background:#f59e0b;border-color:#f59e0b;color:#fff;font-weight:600}
button.batch:hover{background:#d97706}
button.ghost{color:#6b7280}
#free{font:inherit;padding:7px 11px;border:1px solid #d3d9e3;border-radius:8px;width:140px}
#done{margin-top:8px;font-size:12.5px;color:#059669}
#tip{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);background:#111827ee;color:#fff;
     padding:12px 20px;border-radius:10px;font-size:14px;opacity:0;transition:opacity .18s;pointer-events:none;z-index:60}
#tip.on{opacity:1}
dialog{border:0;border-radius:10px;padding:0;max-width:96vw;max-height:96vh;background:#000}
dialog img{display:block;max-width:96vw;max-height:92vh}
kbd{background:#eef1f5;border:1px solid #d6dbe3;border-bottom-width:2px;border-radius:4px;
    padding:0 4px;font-size:11px;font-family:inherit;color:#374151}
</style></head><body>
<header>
  <h1>D3 收尾 · 20 条</h1>
  <div class="gtabs" id="gtabs"></div>
  <div id="bar"></div>
</header>
<main>
  <div class="panel">
    <div class="meta" id="meta"></div>
    <img id="shot" alt="" onclick="zoom()">
    <div class="hint">点图放大　<kbd>Enter</kbd> 确认并下一条　<kbd>U</kbd> 看不清　<kbd>←</kbd><kbd>→</kbd> 翻页　<kbd>B</kbd> 本簇全部同一人</div>
    <div class="sibs"><h3>同簇其他待标成员（点图跳到该条）</h3><div class="sibrow" id="sibrow"></div></div>
    <div class="ops">
      <input id="free" placeholder="输入身份（回车）" autocomplete="off" list="pidlist" onkeydown="if(event.key==='Enter')freePick()">
      <datalist id="pidlist"></datalist>
      <button class="primary" onclick="keepCurrent()">沿用本条已选</button>
      <button class="batch" onclick="batchCluster()">本簇全部同一人</button>
      <button class="ghost" onclick="markUnk()">看不清（U）</button>
      <button class="ghost" onclick="exportCSV()">导出 CSV</button>
    </div>
    <div id="done"></div>
  </div>
  <div class="panel">
    <h2>身份库 <input id="q" placeholder="筛选…" oninput="renderPool()"><span class="tiny" id="pn"></span></h2>
    <div id="pool"></div>
  </div>
</main>
<dialog id="dlg" onclick="this.close()"><img id="dlgimg" alt=""></dialog>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const IMG = DATA.imgs, ITEMS = DATA.items, GROUPS = DATA.groups, POOL = DATA.pool;
const KEY = 'd3tail_v1';
let store = JSON.parse(localStorage.getItem(KEY) || '{}');
let idx = 0, filter = '';

function cur(){ return ITEMS[idx]; }
function esc(s){ return String(s==null?'':s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function tip(t){ const e=document.getElementById('tip'); e.textContent=t; e.classList.add('on');
  clearTimeout(window._t); window._t=setTimeout(()=>e.classList.remove('on'),900); }

function renderTabs(){
  document.getElementById('gtabs').innerHTML = GROUPS.map(g=>{
    const doneN = g.keys.filter(k=>store[k]).length;
    const on = cur().cl===g.cl ? ' on' : '';
    return `<div class="gtab${on}" onclick="jumpCl(${g.cl})">簇${g.cl} · ${g.n}条 · dmax ${g.dmax}`+
           (doneN?`<span class="done">✓${doneN}</span>`:'')+`</div>`;
  }).join('');
  const n=Object.keys(store).length;
  document.getElementById('bar').textContent = `已标 ${n} / ${ITEMS.length}　当前 ${idx+1}`;
}

function jumpCl(c){ const i=ITEMS.findIndex(x=>x.cl===c && !store[x.k]); idx = i>=0?i:ITEMS.findIndex(x=>x.cl===c); render(); }

function render(){
  const it = cur();
  const g = GROUPS.find(x=>x.cl===it.cl) || {};
  const hits = it.hits ? `<span class="tag cool">${esc(it.hits)}</span>` : '';
  document.getElementById('meta').innerHTML =
    `<b>${esc(it.k)}</b>　<span class="tag">${esc(it.cam)}</span>`+
    `<span class="tag">${esc(it.box)}</span><span class="tag">${it.n} 帧</span>`+
    `<span class="tag hot">簇 ${it.cl} · d_max ${g.dmax}</span>${hits}`+
    (it.old_unk?`<span class="tag">上轮：${esc(it.old_unk)}</span>`:'')+
    (store[it.k]?`<div id="selnow" style="color:#059669;margin-top:3px">当前已选：<b>${esc(store[it.k])}</b></div>`:'');
  document.getElementById('shot').src = IMG[it.jpg] || '';
  document.getElementById('sibrow').innerHTML = (it.sibs||[]).map(k=>{
    const s = ITEMS.find(x=>x.k===k);
    return `<figure onclick="go('${k}')"><img loading="lazy" src="${IMG[s.jpg]||''}"><span>${esc(k.split('#')[1])}${store[k]?' · '+esc(store[k]):''}</span></figure>`;
  }).join('') || '<span class="tiny">（本簇只有这一条）</span>';
  document.getElementById('done').textContent = store[it.k] ? '本条已完成：'+store[it.k] : '';
  renderPool(); renderTabs();
}

function go(k){ const i=ITEMS.findIndex(x=>x.k===k); if(i>=0){ idx=i; render(); } }

function renderPool(){
  const q=(document.getElementById('q').value||'').toLowerCase();
  const list = POOL.filter(p=>!q||p.pid.toLowerCase().includes(q));
  document.getElementById('pool').innerHTML = list.map(p=>
    `<div class="card" onclick="pick('${p.pid}')"><img loading="lazy" src="${IMG[p.img]||''}">`+
    `<b>${esc(p.pid)}</b><i>${esc(p.tid.split('#')[1])} · ${p.n}帧</i></div>`).join('');
  document.getElementById('pidlist').innerHTML = POOL.map(p=>`<option value="${p.pid}">`).join('');
  document.getElementById('pn').textContent = `(${list.length})`;
}

function pick(pid){
  store[cur().k]=pid; persist();
  tip('✓ '+pid);
  advance();
}
function freePick(){
  const v=(document.getElementById('free').value||'').trim();
  if(!v) return;
  document.getElementById('free').value='';
  pick(v);
}
function keepCurrent(){
  const v = store[cur().k];
  if(!v){ tip('本条还没选'); return; }
  advance();
}
function markUnk(){ store[cur().k]='UNK'; persist(); tip('记为看不清'); advance(); }

function batchCluster(){
  const it=cur();
  const done = ITEMS.filter(x=>x.cl===it.cl && store[x.k]);
  const pending = ITEMS.filter(x=>x.cl===it.cl && !store[x.k]).map(x=>x.k);
  if(!pending.length){ tip('本簇都已标'); return; }
  if(!done.length){ tip('本簇还没有已标的，先标一条再批量'); return; }
  const vals = [...new Set(done.map(x=>store[x.k]))];
  if(vals.length>1){ tip('本簇已有多个身份：'+vals.join(' / ')+'，请逐条标'); return; }
  const v = vals[0];
  if(v==='UNK'){ tip('本簇已标的是「看不清」，批量请手动确认'); return; }
  if(!confirm(`把簇 ${it.cl} 剩余的 ${pending.length} 条都标成「${v}」？\n` + pending.join('\n'))) return;
  pending.forEach(k=>store[k]=v);
  persist(); render(); tip(`已批量标记 ${pending.length} 条 → ${v}`);
}

function advance(){
  let i = idx;
  for(let d=1; d<=ITEMS.length; d++){
    const j = (idx+d) % ITEMS.length;
    if(!store[ITEMS[j].k]){ i=j; break; }
  }
  idx = i; render();
}

function persist(){ localStorage.setItem(KEY, JSON.stringify(store)); }

function zoom(){
  document.getElementById('dlgimg').src = IMG[cur().jpg]||'';
  document.getElementById('dlg').showModal();
}

function exportCSV(){
  const head='track_id,cluster_idx,person_id,box_type,unk_reason,note';
  const lines=[head];
  ITEMS.forEach(it=>{
    const v = store[it.k];
    const pid = v && v!=='UNK' ? v : (v==='UNK' ? 'UNK' : '');
    const unk = v==='UNK' ? 'far' : '';
    lines.push([it.k, it.cl, pid, it.box, unk, v? 'd3_tail':'skipped']
      .map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(','));
  });
  const csv='\ufeff'+lines.join('\r\n');
  const box=window.prompt('全选复制（Ctrl+A / Ctrl+C），或点确定尝试下载：', csv);
  if(box!==null){
    const a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
    a.download='d3_tail_FILLED.csv'; a.click();
  }
}

document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='ArrowRight'){ idx=(idx+1)%ITEMS.length; render(); }
  else if(e.key==='ArrowLeft'){ idx=(idx-1+ITEMS.length)%ITEMS.length; render(); }
  else if(e.key==='u'||e.key==='U'){ markUnk(); }
  else if(e.key==='b'||e.key==='B'){ batchCluster(); }
  else if(e.key==='Enter'){ keepCurrent(); }
});
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
