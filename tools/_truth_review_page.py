# -*- coding: utf-8 -*-
"""生成离线可标注复核页（图片内嵌 base64，可导出 CSV）。一次性辅助脚本。

输入：output_crosscam/analysis/truth/_review/pair_machine.csv、truth/events_all.csv、truth/thumbs/
输出：output_crosscam/analysis/truth/_review/review.html
"""
import base64, csv, io, json, os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = os.path.join(ROOT, "output_crosscam", "analysis", "truth")
REV = os.path.join(TRUTH, "_review")


def rd(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def b64(eid, maxw=190, q=78):
    p = os.path.join(TRUTH, "thumbs", eid + ".jpg")
    if not os.path.exists(p):
        return ""
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, max(1, round(im.height * maxw / im.width))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_cache = {}


def img(eid):
    if eid not in _cache:
        _cache[eid] = b64(eid)
    return _cache[eid]


# 唯一事件（取首行）
seen = {}
for r in rd(os.path.join(TRUTH, "events_all.csv")):
    seen.setdefault(r["event_id"], r)
events = []
for eid, r in sorted(seen.items(), key=lambda kv: kv[1]["cam"] + "%08.1f" % float(kv[1]["t"])):
    events.append(dict(eid=eid, short=eid.split("_E_")[1], cam=r["cam"], kind=r["kind"],
                       edge=r["edge"], t=float(r["t"]), u=r["u"], v=r["v"], score=r["score"],
                       n_obs=r["n_obs"], dur_s=r["dur_s"]))

pairs = []
for r in rd(os.path.join(REV, "pair_machine.csv")):
    a, b = r["event_a"], r["event_b"]
    pairs.append(dict(pair_id=r["pair_id"], a=a, b=b, sa=a.split("_E_")[1], sb=b.split("_E_")[1],
                      ta=float(r["t_a"]), tb=float(r["t_b"]), dt=float(r["dt"]),
                      relay=int(r["in_relay_v5"]), cos=float(r["cos_appear"]),
                      machine=r["machine"]))

THUMB = {eid: img(eid) for eid in seen}

# 预填：已有的人工判定（pair_candidates_labeled.csv）
SEED = {}
_lp = os.path.join(TRUTH, "pair_candidates_labeled.csv")
if os.path.exists(_lp):
    for r in rd(_lp):
        v = r.get("human_same_person", "")
        if v in ("1", "0", "?"):
            SEED[r["pair_id"]] = {"v": 1 if v == "1" else (0 if v == "0" else 2),
                                  "c": r.get("human_confidence", "") or "",
                                  "n": r.get("human_note", "") or ""}

HTML = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>truth 人工判定复核页（343 × 316）</title>
<style>
:root{--bd:#e3e6ea;--mut:#6b7280;--bg:#f7f8fa;--c1:#0b6bcb;--ok:#0a7d33;--no:#b3261e;--un:#8a6d00}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:#1a1a1a}
header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid var(--bd);padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0} .mut{color:var(--mut);font-size:12px}
button{font:inherit;padding:5px 12px;border:1px solid var(--bd);background:#fff;border-radius:6px;cursor:pointer}
button.p{background:var(--c1);color:#fff;border-color:var(--c1)}
select,input{font:inherit;padding:4px 8px;border:1px solid var(--bd);border-radius:6px;background:#fff}
main{padding:14px 16px 60px}
.tabs{display:flex;gap:6px;margin-bottom:10px}
.tab{padding:6px 14px;border:1px solid var(--bd);background:#fff;border-radius:8px;cursor:pointer}
.tab.on{background:#e8f0fb;border-color:var(--c1);color:var(--c1);font-weight:600}
.card{background:#fff;border:1px solid var(--bd);border-radius:10px;padding:10px 12px;margin-bottom:10px;display:grid;grid-template-columns:190px 190px 1fr;gap:14px;align-items:start}
.card.done{box-shadow:inset 4px 0 0 var(--ok)} .card.same{box-shadow:inset 4px 0 0 var(--ok)} .card.diff{box-shadow:inset 4px 0 0 var(--no)} .card.unk{box-shadow:inset 4px 0 0 var(--un)}
.ph{width:190px;min-height:120px;border:1px solid var(--bd);border-radius:8px;background:#fafbfc;display:flex;align-items:center;justify-content:center;overflow:hidden}
.ph img{max-width:100%;max-height:260px;display:block}
.cap{font-size:12px;color:var(--mut);margin-top:4px}
.meta{font-size:12.5px} .meta b{font-size:13.5px}
.k{color:var(--mut)}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11.5px;border:1px solid var(--bd)}
.b-same{background:#e7f5ec;color:var(--ok);border-color:#bfe3cc}.b-diff{background:#fdecea;color:var(--no);border-color:#f3c6c1}
.b-un{background:#fdf6e0;color:var(--un);border-color:#ecdca6}.b-r{background:#e8f0fb;color:var(--c1);border-color:#c3d8f5}
.b-x{background:#fdecea;color:#b3261e;border-color:#f0b4ae;font-weight:700}
.opt{display:flex;gap:8px;margin:8px 0}
.opt label{cursor:pointer;border:1px solid var(--bd);border-radius:8px;padding:5px 12px;background:#fff}
.opt input{margin-right:5px}
.opt label.sel{border-color:var(--c1);background:#e8f0fb;font-weight:600}
.note{width:100%;padding:5px 8px;border:1px solid var(--bd);border-radius:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.ecard{background:#fff;border:1px solid var(--bd);border-radius:10px;padding:8px}
.ecard img{max-width:100%;max-height:210px;display:block;margin:0 auto 6px}
.pid{width:74px;padding:3px 6px;border:1px solid var(--bd);border-radius:6px}
.hide{display:none}
</style></head><body>
<header>
  <h1>truth 人工判定复核页 <span class="mut">343 × 316 · synced-10min</span></h1>
  <div class="tabs"><div class="tab on" id="t1" onclick="tab(1)">配对判定 (<span id="np"></span>)</div><div class="tab" id="t2" onclick="tab(2)">事件人物编号 (<span id="ne"></span>)</div></div>
  <span class="mut" id="prog"></span>
  <button class="p" onclick="exp('pair')">导出配对 CSV</button>
  <button class="p" onclick="exp('event')">导出事件 CSV</button>
  <button onclick="if(confirm('清空本页全部标注？')){lsDel(KEY);location.reload()}">清空</button>
  <span class="mut">| 过滤</span>
  <select id="f" onchange="render(1)"><option value="all">全部</option><option value="relay">relay v5 支持</option><option value="hi">cos ≥ 0.95</option><option value="gray">灰区 0.85–0.95</option><option value="lo">cos &lt; 0.85</option><option value="nd">只看未标注</option><option value="todo">优先带（relay｜cos≥.94｜dt≤15s）</option></select>
  <label class="mut"><input type="checkbox" id="sortdt" onchange="render(1)"> 按 dt 升序</label>
</header>
<main>
  <div id="err" style="display:none;background:#fdecea;border:1px solid #f0b4ae;color:#b3261e;padding:8px 10px;border-radius:8px;margin-bottom:10px;font-size:12.5px"></div>
  <noscript><div style="background:#fff4e5;border:1px solid #f0c98a;padding:10px;border-radius:8px;margin-bottom:10px">
  当前环境禁用了脚本，页面内容无法渲染。<b>请直接用浏览器打开本文件</b>（双击 <code>output_crosscam/analysis/truth/_review/review.html</code>）。</div></noscript>
  <section id="s1"><div id="pairs"></div></section>
  <section id="s2" class="hide"><div class="mut" style="margin-bottom:8px">给同一人的事件填相同编号（如 1,2,3…）；反光伪影/误检填 <b>FP</b>。编号可跨机位、跨时间。</div><div class="grid" id="events"></div></section>
</main>
<script>
const KEY='truth_review_v1';
window.onerror=function(m,s,l,c,e){var b=document.getElementById('err');if(b){b.style.display='block';b.textContent='脚本错误: '+m+' @'+l+':'+c;}};
const PAIRS=__PAIRS__, EVENTS=__EVENTS__, THUMB=__THUMB__, SEED=__SEED__;
const MEM={};
function lsGet(k){try{return localStorage.getItem(k)}catch(e){return (k in MEM)?MEM[k]:null}}
function lsSet(k,v){try{localStorage.setItem(k,v)}catch(e){MEM[k]=v}}
function lsDel(k){try{localStorage.removeItem(k)}catch(e){delete MEM[k]}}
let ST=JSON.parse(lsGet(KEY)||'{}')||{};
if(!ST.pair)ST.pair={}; if(!ST.event)ST.event={};
for(const k in SEED){ if(!ST.pair[k]) ST.pair[k]=SEED[k]; }
const save=()=>lsSet(KEY,JSON.stringify(ST));
document.getElementById('np').textContent=PAIRS.length; document.getElementById('ne').textContent=EVENTS.length;
function tab(n){document.getElementById('t1').classList.toggle('on',n==1);document.getElementById('t2').classList.toggle('on',n==2);
 document.getElementById('s1').classList.toggle('hide',n!=1);document.getElementById('s2').classList.toggle('hide',n!=2);}
function cls(m){return m=='same'?'b-same':m=='diff'?'b-diff':'b-un'}
function prog(){let a=0,b=0;PAIRS.forEach(p=>{if(ST.pair[p.pair_id]&&ST.pair[p.pair_id].v!==undefined)a++});
 EVENTS.forEach(e=>{const r=ST.event[e.eid];if(r&&(r.pid||r.fp))b++});
 document.getElementById('prog').textContent='配对 '+a+'/'+PAIRS.length+' · 事件 '+b+'/'+EVENTS.length;}
function pass(p){const m=p.machine,f=document.getElementById('f').value,st=ST.pair[p.pair_id];
 if(f=='relay')return p.relay==1; if(f=='hi')return p.cos>=0.95; if(f=='gray')return p.cos>=0.85&&p.cos<0.95;
 if(f=='lo')return p.cos<0.85; if(f=='nd')return !(st&&st.v!==undefined);
 if(f=='todo')return p.relay==1||p.cos>=0.94||p.dt<=15; return true;}
function render(n){
 if(n!=1)return;
 let list=PAIRS.filter(pass);
 if(document.getElementById('sortdt').checked)list=[...list].sort((x,y)=>x.dt-y.dt);
 const el=document.getElementById('pairs');el.innerHTML='';
 const SA={},SB={};PAIRS.forEach(p=>{if((ST.pair[p.pair_id]||{}).v==1){SA[p.a]=(SA[p.a]||0)+1;SB[p.b]=(SB[p.b]||0)+1;}});
 list.forEach(p=>{const st=ST.pair[p.pair_id]||{};const conf=(st.v==1&&(SA[p.a]>1||SB[p.b]>1))?'<span class="badge b-x">⚠ 互斥冲突</span>':'';
  const d=document.createElement('div');d.className='card '+(st.v==1?'same':st.v==0?'diff':st.v==2?'unk':'');
  d.innerHTML=`<div><div class="ph">${THUMB[p.a]?`<img src="${THUMB[p.a]}">`:''}</div><div class="cap">A · 343 · ${p.sa} · t=${p.ta.toFixed(0)}s</div></div>
  <div><div class="ph">${THUMB[p.b]?`<img src="${THUMB[p.b]}">`:''}</div><div class="cap">B · 316 · ${p.sb} · t=${p.tb.toFixed(0)}s</div></div>
  <div class="meta"><b>${p.sa} ↔ ${p.sb}</b><br>
   <span class="k">dt</span> ${p.dt.toFixed(1)}s &nbsp; <span class="k">外观 cos</span> <b>${p.cos.toFixed(3)}</b>
   &nbsp; <span class="badge ${cls(p.machine)}">机器建议 ${p.machine}</span> ${p.relay?`<span class="badge b-r">relay v5</span>`:''} ${conf}
   <div class="opt" data-pid="${p.pair_id}">
    <label class="${st.v==1?'sel':''}"><input type="radio" name="r_${p.pair_id}" value="1" ${st.v==1?'checked':''} onchange="setp('${p.pair_id}',1,this)">同人</label>
    <label class="${st.v==0?'sel':''}"><input type="radio" name="r_${p.pair_id}" value="0" ${st.v==0?'checked':''} onchange="setp('${p.pair_id}',0,this)">异人</label>
    <label class="${st.v==2?'sel':''}"><input type="radio" name="r_${p.pair_id}" value="2" ${st.v==2?'checked':''} onchange="setp('${p.pair_id}',2,this)">看不清</label>
   </div>
   <div class="opt"><span class="k">置信</span>
    ${['high','mid','low'].map(c=>`<label class="${st.c==c?'sel':''}"><input type="radio" name="c_${p.pair_id}" value="${c}" ${st.c==c?'checked':''} onchange="setp('${p.pair_id}',null,this,'c')">${c}</label>`).join('')}
   </div>
   <input class="note" placeholder="备注（可选）" value="${(st.n||'').replace(/"/g,'&quot;')}" oninput="setp('${p.pair_id}',null,null,'n',this.value)">
  </div>`;el.appendChild(d);});
 prog();}
function setp(pid,v,el,k){const st=ST.pair[pid]||{};k=k||'v';
 if(k=='v'){st.v=v; [...document.querySelectorAll(`[data-pid="${pid}"] label`)].forEach((l,i)=>l.classList.toggle('sel',i==v));}
 else if(k=='c'){st.c=el.value; el.closest('.opt').querySelectorAll('label').forEach(l=>l.classList.toggle('sel',l.contains(el)));}
 else st.n=el; ST.pair[pid]=st;save();prog();
 const card=el?el.closest('.card'):null; if(card&&k=='v'){card.className='card '+(v==1?'same':v==0?'diff':v==2?'unk':'');}}
function renderEv(){const el=document.getElementById('events');el.innerHTML='';
 EVENTS.forEach(e=>{const r=ST.event[e.eid]||{};
  const d=document.createElement('div');d.className='ecard';
  d.innerHTML=`<img src="${THUMB[e.eid]||''}"><div class="cap">${e.short} · ${e.cam} · ${e.kind}/${e.edge} · t=${e.t.toFixed(0)}s · n=${e.n_obs}</div>
  <div style="margin-top:4px">人物 <input class="pid" value="${r.pid||''}" placeholder="—" oninput="sete('${e.eid}','pid',this.value)">
   <label class="mut" style="margin-left:6px"><input type="checkbox" ${r.fp?'checked':''} onchange="sete('${e.eid}','fp',this.checked)"> FP</label></div>`;el.appendChild(d);});
 prog();}
function sete(eid,k,v){const r=ST.event[eid]||{};r[k]=v;ST.event[eid]=r;save();prog();}
function dl(name,txt){const b=new Blob(['\ufeff'+txt],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=name;a.click();}
function exp(what){let s='';
 if(what=='pair'){s='pair_id,cam_a,event_a,t_a,cam_b,event_b,t_b,dt,in_relay_v5,cos_appear,machine,human_same_person,human_confidence,human_note\n';
  PAIRS.forEach(p=>{const st=ST.pair[p.pair_id]||{};const v=st.v===1?'1':st.v===0?'0':st.v===2?'?':'';
   s+=[p.pair_id,'343',p.a,p.ta,'316',p.b,p.tb,p.dt,p.relay,p.cos,p.machine,v,st.c||'',(st.n||'').replace(/,/g,'，')].join(',')+'\n';});dl('truth_labels_pair.csv',s);}
 else{s='event_id,cam,kind,edge,t,human_person_id\n';EVENTS.forEach(e=>{const r=ST.event[e.eid]||{};const pid=r.fp?'FP':(r.pid||'');
   s+=[e.eid,e.cam,e.kind,e.edge,e.t,pid].join(',')+'\n';});dl('truth_labels_event.csv',s);}}
render(1);renderEv();
</script></body></html>"""

html = (HTML.replace("__PAIRS__", json.dumps(pairs, ensure_ascii=False))
            .replace("__EVENTS__", json.dumps(events, ensure_ascii=False))
            .replace("__THUMB__", json.dumps(THUMB, ensure_ascii=False))
            .replace("__SEED__", json.dumps(SEED, ensure_ascii=False)))
out = os.path.join(REV, "review.html")
open(out, "w", encoding="utf-8").write(html)
print("wrote", out, round(len(html) / 1e6, 2), "MB", "pairs", len(pairs), "events", len(events))
