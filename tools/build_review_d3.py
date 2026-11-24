#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3 的收尾页：把「身份复核」与「补标」合成一个页面。

两类任务性质不同，分两个页签：

  复核 A（25 条）：female4 / male7 / male9 / male4 的全部轨迹。
      显示原标注，右侧给同簇全部成员的原标注做横向对比。
  补标 B（20 条）：上次标成「看不清」的 16 条（簇 16/32/34，323 远场）
      + 上次漏提交的 4 条（簇 61，318）。空白待填，可从头标。

导出：
  点「导出」先弹一个文本框（内容已全选），同时给下载与复制两个按钮——
  预览面板里 a.click() 下载可能被拦，文本框是兜底。
  CSV 用标准 quoting（含逗号/引号的字段加双引号并转义），不再把逗号替换成空格。

输出  gt/review_d3.html
"""
import base64
import collections
import csv
import io
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GT = os.path.join(ROOT, "deliverable/data-request-v2/gt")

FILLED = os.path.join(GT, "gt_gallery_members_FILLED.csv")
TEMPLATE = os.path.join(GT, "gallery_members_TEMPLATE.csv")
DST = os.path.join(GT, "review_d3.html")

REVIEW_PIDS = ["female4", "male7", "male9", "male4"]

TPL = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>D3 收尾：复核 + 补标</title>
<style>
:root{--bg:#0f1216;--card:#171b21;--line:#262c35;--fg:#e6e9ee;--dim:#8b95a5;
      --hot:#ff6b6b;--ok:#3ddc97;--acc:#4d9fff;--warn:#f5a524}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.6 -apple-system,"Segoe UI",system-ui,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:10px;
       padding:10px 16px;background:#12161c;border-bottom:1px solid var(--line);flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
.tabs{display:flex;gap:4px;margin-left:6px}
.tab{padding:5px 12px;border-radius:7px;background:#1d232b;color:var(--dim);
     cursor:pointer;font-size:13px;border:1px solid transparent}
.tab.on{background:#25303d;color:var(--fg);border-color:#3a4a5e}
.sp{flex:1}
.pill{padding:4px 10px;border-radius:20px;background:#1d232b;color:var(--dim);font-size:12px}
button{font-family:inherit;cursor:pointer;border:0;border-radius:8px;
       padding:8px 14px;font-size:13px;background:#232b35;color:var(--fg)}
button:hover{background:#2c3644}
button.pri{background:var(--acc);color:#04101f;font-weight:600}
main{padding:16px;max-width:1500px;margin:0 auto}
.wrap{display:grid;grid-template-columns:minmax(0,1.02fr) minmax(0,1fr);gap:16px}
@media(max-width:1100px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.shot{width:100%;border-radius:9px;display:block;background:#0a0d11}
.meta{color:var(--dim);font-size:12.5px;margin-top:8px;line-height:1.8}
.meta b{color:var(--fg);font-weight:600}
.old{display:inline-block;padding:2px 9px;border-radius:6px;background:#2a1c1c;
     color:#ffb3b3;font-weight:600;font-size:13px}
.old.none{background:#2a2318;color:#f5cf8a}
h2{font-size:13px;color:var(--dim);margin:0 0 10px;font-weight:600;letter-spacing:.4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:8px}
.tile{position:relative;border-radius:8px;overflow:hidden;background:#0a0d11;
      outline:1px solid var(--line);cursor:zoom-in}
.tile img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
.tile span{position:absolute;left:0;right:0;bottom:0;padding:3px 5px;font-size:11px;
           background:linear-gradient(transparent,#000c);color:#fff;font-weight:600}
.tile.me{outline:2px solid var(--hot)}
.tile.me span{background:linear-gradient(transparent,#5a1212)}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px;
     padding-top:14px;border-top:1px solid var(--line)}
.opt{padding:9px 15px;border-radius:9px;background:#1d232b;color:var(--fg);
     border:1px solid var(--line);font-weight:600;font-size:13.5px}
.opt:hover{background:#28313c}
.opt.on{background:#14342a;border-color:var(--ok);color:var(--ok)}
.opt.keep{border-style:dashed}
.opt.skip{border-color:#4a3a1a;color:#f5cf8a}
.opt kbd{font-size:10px;opacity:.55;margin-left:5px;font-family:inherit}
input[type=text]{padding:9px 12px;border-radius:9px;border:1px solid var(--line);
                 background:#0e1218;color:var(--fg);font-size:13.5px;font-family:inherit;width:150px}
.hint{color:var(--dim);font-size:12.5px;margin-top:10px;line-height:1.8}
.hint kbd{background:#252d38;padding:1px 6px;border-radius:4px;font-family:inherit;font-size:12px}
.ids{display:grid;gap:14px}
.idrow{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.idhead{display:flex;align-items:baseline;gap:10px;margin-bottom:10px}
.idhead b{font-size:15px}
.idhead em{color:var(--dim);font-style:normal;font-size:12.5px}
.lightbox{position:fixed;inset:0;background:#000d;display:none;z-index:99;
          align-items:center;justify-content:center;padding:24px;cursor:zoom-out}
.lightbox.on{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:10px}
.sheet{position:fixed;inset:0;background:#000c;display:none;z-index:120;
       align-items:center;justify-content:center;padding:24px}
.sheet.on{display:flex}
.sheet .inner{background:#151a21;border:1px solid #2c3542;border-radius:14px;
              width:min(920px,100%);max-height:88vh;display:flex;flex-direction:column;padding:16px}
.sheet h3{margin:0 0 10px;font-size:15px}
.sheet textarea{flex:1;min-height:320px;width:100%;background:#0b0e13;color:#d7dee8;
                border:1px solid #2c3542;border-radius:9px;padding:11px;
                font:12px/1.5 ui-monospace,Consolas,monospace;resize:vertical}
.sheet .row{display:flex;gap:9px;margin-top:12px;flex-wrap:wrap}
.ok{color:var(--ok)}
</style></head><body>

<header>
  <h1>D3 收尾</h1>
  <div class="tabs" id="tabs"></div>
  <span class="pill" id="prog">0 / 0</span>
  <span class="pill" id="chg">已处理 0</span>
  <span class="sp"></span>
  <button onclick="openExport()" class="pri">导出 CSV</button>
</header>

<main>
  <div id="panel">
    <div class="wrap">
      <div class="card">
        <img id="shot" class="shot" alt="">
        <div class="meta" id="meta"></div>
      </div>
      <div class="card">
        <h2>同簇成员（灰字=原标注，红边=当前这条）</h2>
        <div class="grid" id="sibs"></div>
        <div class="hint" id="sibhint"></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="meta" id="origline"></div>
      <div class="bar" id="opts"></div>
      <div class="hint" id="keyhint"></div>
    </div>
  </div>

  <div id="overview" style="display:none">
    <div class="hint" style="margin-bottom:12px">
      同一 person_id 的全部轨迹并排。若这些不像同一个人，说明这个编号被用混了。
    </div>
    <div class="ids" id="idlist"></div>
  </div>
</main>

<div class="lightbox" id="lb" onclick="this.classList.remove('on')"><img id="lbi" alt=""></div>

<div class="sheet" id="sheet">
  <div class="inner">
    <h3>导出（内容已全选，可直接 Ctrl+C 复制）</h3>
    <textarea id="csvout" onclick="this.select()" readonly></textarea>
    <div class="row">
      <button class="pri" onclick="doDownload()">下载 CSV</button>
      <button onclick="copyCSV()">复制到剪贴板</button>
      <button onclick="closeExport()">关闭</button>
      <span id="expinfo" class="hint" style="margin:0"></span>
    </div>
    <div class="hint">
      预览面板里「下载」可能被拦。<b>若点了没反应，用「复制」或直接在文本框里 Ctrl+A / Ctrl+C</b>，
      粘贴到记事本另存为 <code>review_d3_FILLED.csv</code> 即可。
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const IMG  = __IMGS__;
const KEY  = 'review_d3_v2';
let mode = 'review';
let idx = 0;
let store = JSON.parse(localStorage.getItem(KEY) || '{}');

const ITEMS = DATA.items;
const srcOf = it => IMG[it.jpg] || it.jpg;

function persist(){ localStorage.setItem(KEY, JSON.stringify(store)); }
function cur(){ return ITEMS[idx]; }
function done(){
  return ITEMS.filter(it => {
    const r = store[it.k];
    return r && (r.pid || r.skip);
  }).length;
}

function renderTabs(){
  const a = ITEMS.filter(i=>i.mode==='review').length;
  const b = ITEMS.filter(i=>i.mode==='new').length;
  document.getElementById('tabs').innerHTML =
    `<div class="tab ${mode==='review'?'on':''}" onclick="setMode('review')">A 复核 ${a}</div>
     <div class="tab ${mode==='new'?'on':''}" onclick="setMode('new')">B 补标 ${b}</div>
     <div class="tab ${mode==='overview'?'on':''}" onclick="setMode('overview')">身份总览</div>`;
}
function setMode(m){
  mode = m;
  document.getElementById('panel').style.display    = m==='overview'?'none':'';
  document.getElementById('overview').style.display = m==='overview'?'':'none';
  renderTabs();
  if(m==='overview'){ renderOverview(); return; }
  const first = ITEMS.findIndex(i => i.mode === m);
  const mark  = ITEMS.findIndex((i,j) => j>=first && i.mode===m &&
                  !(store[i.k] && (store[i.k].pid || store[i.k].skip)));
  idx = mark>=0 ? mark : (first>=0 ? first : 0);
  renderItem();
}

function renderItem(){
  const it = cur();
  if(!it) return;
  const r = store[it.k] || {};
  document.getElementById('shot').src = srcOf(it);

  document.getElementById('meta').innerHTML =
    `簇 <b>${it.cl}</b> · <b>${it.k}</b> · ${it.cam} 机位 · ${it.n} 帧 · ${it.box}`
    + (it.sw ? `<br><span style="color:#ffb3b3">轨迹内换人 → ${it.sw}</span>` : '')
    + `<br>簇内身份：${it.brief}`;

  const o = document.getElementById('origline');
  if(it.mode === 'review'){
    o.innerHTML = `原标注：<span class="old">${it.pid}</span>`
      + (it.note ? `<span style="margin-left:8px">（备注：${it.note}）</span>` : '');
  }else{
    o.innerHTML = `待补标 · <span class="old none">${it.why}</span>`;
  }

  document.getElementById('sibs').innerHTML = it.sibs.map(s =>
    `<div class="tile ${s.k===it.k?'me':''}" onclick="lb('${srcOf(s)}')">
       <img loading="lazy" src="${srcOf(s)}" alt="">
       <span>${s.k===it.k?'▶ ':''}${s.pid || '未标'}</span></div>`).join('');
  document.getElementById('sibhint').textContent =
    `本簇共 ${it.sibs.length} 条，已标 ${it.sibs.filter(s=>s.pid).length} 条。`;

  const now = r.pid || '';
  let html = '';
  if(it.mode === 'review'){
    html = REVIEW_PIDS.map((p,i) =>
      `<button class="opt ${now===p?'on':''}" onclick="pick('${p}')">${p}<kbd>${i+1}</kbd></button>`).join('')
      + `<button class="opt keep" onclick="keep()">保持不变<kbd>⏎</kbd></button>`
      + `<input type="text" id="free" list="pids" placeholder="其他身份…"
                onchange="pick(this.value.trim())"
                value="${now && !REVIEW_PIDS.includes(now) ? now : ''}">`;
    document.getElementById('keyhint').innerHTML =
      `<kbd>1</kbd> female4 · <kbd>2</kbd> male7 · <kbd>3</kbd> male9 · <kbd>4</kbd> male4 ·
       <kbd>Enter</kbd> 保持不变 · <kbd>←</kbd><kbd>→</kbd> 翻页`;
  }else{
    html = `<input type="text" id="free" list="pids" placeholder="输入身份，如 male7 / female2"
                  onkeydown="if(event.key==='Enter')submitFree()"
                  value="${now}">
            <button class="opt" onclick="submitFree()">确认<kbd>⏎</kbd></button>
            <button class="opt skip" onclick="skip()">确实看不清<kbd>U</kbd></button>`;
    document.getElementById('keyhint').innerHTML =
      `输入框里打字，<kbd>Enter</kbd> 确认并下一条；确实看不清就按 <kbd>U</kbd>（记为 UNK）·
       <kbd>←</kbd><kbd>→</kbd> 翻页`;
    setTimeout(()=>{ const f=document.getElementById('free'); if(f) f.focus(); }, 30);
  }
  document.getElementById('opts').innerHTML = html;

  const inMode = ITEMS.filter(i=>i.mode===it.mode);
  const pos = inMode.findIndex(i=>i.k===it.k)+1;
  document.getElementById('prog').textContent = `${pos} / ${inMode.length}`;
  document.getElementById('chg').textContent = `已处理 ${done()} / ${ITEMS.length}`;
  window.scrollTo({top:0,behavior:'smooth'});
}

function pick(p){
  const it = cur();
  if(!p) return;
  store[it.k] = {pid:p, old:it.pid, mode:it.mode};
  persist(); next();
}
function keep(){
  const it = cur();
  store[it.k] = {pid:it.pid, old:it.pid, mode:it.mode};
  persist(); next();
}
function skip(){
  const it = cur();
  store[it.k] = {pid:'UNK', old:it.pid, skip:1, mode:it.mode};
  persist(); next();
}
function submitFree(){
  const f = document.getElementById('free');
  const v = (f && f.value || '').trim();
  if(!v){ f && f.focus(); return; }
  pick(v);
}
function next(){
  const m = cur().mode;
  let j = idx;
  do { j++; } while (j < ITEMS.length && ITEMS[j].mode !== m);
  if(j >= ITEMS.length){
    const first = ITEMS.findIndex(i=>i.mode===m);
    if(first>=0){ idx = first; renderItem(); } else renderItem();
    return;
  }
  idx = j; renderItem();
}
function jump(d){
  const m = cur().mode;
  let j = idx + d;
  while(j>=0 && j<ITEMS.length && ITEMS[j].mode!==m) j += d;
  if(j<0 || j>=ITEMS.length) return;
  idx = j; renderItem();
}

function renderOverview(){
  const box = document.getElementById('idlist');
  box.innerHTML = DATA.groups.map(g => {
    const tiles = g.items.map(it =>
      `<div class="tile" onclick="lb('${srcOf(it)}')">
         <img loading="lazy" src="${srcOf(it)}" alt="">
         <span>簇${it.cl} · ${it.k.split('#')[1]}</span></div>`).join('');
    return `<div class="idrow">
      <div class="idhead"><b>${g.pid}</b><em>${g.items.length} 条 · 跨 ${g.ncl} 个簇</em></div>
      <div class="grid">${tiles}</div></div>`;
  }).join('');
}

function lb(src){
  document.getElementById('lbi').src = src;
  document.getElementById('lb').classList.add('on');
}

const q = v => { const s = String(v==null?'':v);
  return /[",\n]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; };

function buildCSV(){
  const head = ['cluster_idx','track_id','seg','cam','person_id','box_type',
                'id_switch','id_switch_frame','unk_reason','note','task'];
  const out = [head.join(',')];
  ITEMS.forEach(it => {
    const r = store[it.k];
    if(!r) return;
    const pid = r.skip ? 'UNK' : (r.pid || '');
    out.push([it.cl, it.k, it.seg, it.cam, pid, it.box,
              it.isw, it.iswf, r.skip?'unsure':'', it.note || '', it.mode].map(q).join(','));
  });
  return out.join('\n');
}

function openExport(){
  const csv = buildCSV();
  const n = csv.split('\n').length - 1;
  document.getElementById('csvout').value = csv;
  document.getElementById('expinfo').textContent =
    `已处理 ${done()} / ${ITEMS.length} 条，导出 ${n} 行`;
  document.getElementById('sheet').classList.add('on');
  const ta = document.getElementById('csvout');
  setTimeout(()=>{ ta.focus(); ta.select(); }, 50);
}
function closeExport(){ document.getElementById('sheet').classList.remove('on'); }
function doDownload(){
  const csv = document.getElementById('csvout').value;
  try{
    const blob = new Blob(['\ufeff'+csv], {type:'text/csv;charset=utf-8'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'review_d3_FILLED.csv';
    document.body.appendChild(a); a.click();
    setTimeout(()=>{ document.body.removeChild(a); URL.revokeObjectURL(a.href); }, 100);
    document.getElementById('expinfo').textContent = '已触发下载；若无文件，请用复制';
  }catch(e){
    document.getElementById('expinfo').textContent = '下载失败：' + e.message + '，请用复制';
  }
}
async function copyCSV(){
  const csv = document.getElementById('csvout').value;
  try{
    await navigator.clipboard.writeText(csv);
    document.getElementById('expinfo').textContent = '已复制到剪贴板';
  }catch(e){
    const ta = document.getElementById('csvout');
    ta.select();
    document.getElementById('expinfo').textContent = '请按 Ctrl+C 手动复制（已全选）';
  }
}

const REVIEW_PIDS = __REVIEW__;
document.addEventListener('keydown', e => {
  if(mode === 'overview') return;
  if(document.getElementById('sheet').classList.contains('on')) return;
  const typing = e.target && e.target.tagName === 'INPUT';
  if(e.key === 'ArrowRight'){ jump(1); return; }
  if(e.key === 'ArrowLeft'){ jump(-1); return; }
  if(mode === 'review'){
    if(typing) return;
    if(e.key === '1') pick(REVIEW_PIDS[0]);
    else if(e.key === '2') pick(REVIEW_PIDS[1]);
    else if(e.key === '3') pick(REVIEW_PIDS[2]);
    else if(e.key === '4') pick(REVIEW_PIDS[3]);
    else if(e.key === 'Enter') keep();
  }else{
    if(e.key === 'Enter'){ e.preventDefault(); submitFree(); }
    else if(e.key === 'u' || e.key === 'U'){ if(!typing) skip(); }
  }
});

const dl = document.createElement('datalist');
dl.id = 'pids';
dl.innerHTML = DATA.pids.map(p=>`<option value="${p}">`).join('');
document.body.appendChild(dl);

renderTabs(); setMode('review');
</script></body></html>
"""


def build():
    filled = list(csv.DictReader(open(FILLED, encoding="utf-8-sig")))
    tpl = list(csv.DictReader(open(TEMPLATE, encoding="utf-8-sig")))
    got = {r["track_id"] for r in filled}

    by = collections.defaultdict(list)
    for r in filled:
        by[int(r["cluster_idx"])].append(r)

    all_pids = sorted({r["person_id"].strip() for r in filled
                       if r["person_id"].strip() and r["person_id"].strip().upper() != "UNK"})

    items, need = [], set()

    def sibs_of(cluster, skip_key):
        out = []
        for s in by.get(cluster, []):
            if s["track_id"] == skip_key:
                continue
            out.append({"k": s["track_id"], "pid": s["person_id"].strip(),
                        "jpg": s["prompt_jpg"]})
            need.add(s["prompt_jpg"])
        # 把当前这条也放进对比条（前端按 k 高亮），排在最前
        return out

    # A 复核
    for r in filled:
        if r["person_id"].strip() not in REVIEW_PIDS:
            continue
        c = int(r["cluster_idx"])
        cnt = collections.Counter(s["person_id"].strip() or "未标" for s in by[c])
        brief = "、".join(f"{k}×{v}" for k, v in cnt.most_common())
        sibs = [{"k": r["track_id"], "pid": r["person_id"].strip(),
                 "jpg": r["prompt_jpg"]}] + sibs_of(c, r["track_id"])
        items.append({
            "k": r["track_id"], "cl": c, "seg": r["seg"], "cam": r["cam"],
            "n": r["n_frames"], "box": r["box_type"] or "body",
            "pid": r["person_id"].strip(), "note": (r["note"] or "").strip(),
            "sw": (r["note"] or "").strip() if (r["id_switch"] or "").strip() not in ("", "0") else "",
            "isw": (r["id_switch"] or "").strip(), "iswf": (r["id_switch_frame"] or "").strip(),
            "jpg": r["prompt_jpg"], "brief": brief, "sibs": sibs, "mode": "review",
            "why": "",
        })
        need.add(r["prompt_jpg"])

    # B 补标：16 条空 + 4 条缺失
    for r in filled:
        if r["person_id"].strip():
            continue
        c = int(r["cluster_idx"])
        cnt = collections.Counter(s["person_id"].strip() or "未标" for s in by[c])
        brief = "、".join(f"{k}×{v}" for k, v in cnt.most_common())
        sibs = [{"k": r["track_id"], "pid": "", "jpg": r["prompt_jpg"]}] + sibs_of(c, r["track_id"])
        items.append({
            "k": r["track_id"], "cl": c, "seg": r["seg"], "cam": r["cam"],
            "n": r["n_frames"], "box": r["box_type"] or "body",
            "pid": "", "note": (r["note"] or "").strip(),
            "sw": "", "isw": (r["id_switch"] or "").strip(),
            "iswf": (r["id_switch_frame"] or "").strip(),
            "jpg": r["prompt_jpg"], "brief": brief, "sibs": sibs, "mode": "new",
            "why": f'上次标为看不清（{r["unk_reason"] or "远场"}）',
        })
        need.add(r["prompt_jpg"])

    for r in tpl:                     # 完全没提交的 4 条
        if r["track_id"] in got:
            continue
        c = int(r["cluster_idx"])
        cnt = collections.Counter(s["person_id"].strip() or "未标" for s in by.get(c, []))
        brief = "、".join(f"{k}×{v}" for k, v in cnt.most_common()) or "本簇尚无已标成员"
        sibs = [{"k": r["track_id"], "pid": "", "jpg": r["prompt_jpg"]}] + sibs_of(c, r["track_id"])
        items.append({
            "k": r["track_id"], "cl": c, "seg": r["seg"], "cam": r["cam"],
            "n": r["n_frames"], "box": "", "pid": "", "note": "",
            "sw": "", "isw": "", "iswf": "",
            "jpg": r["prompt_jpg"], "brief": brief, "sibs": sibs, "mode": "new",
            "why": "上次漏提交",
        })
        need.add(r["prompt_jpg"])

    items.sort(key=lambda x: (0 if x["mode"] == "review" else 1, x["cl"], x["k"]))

    groups = []
    for p in REVIEW_PIDS:
        its = [r for r in filled if r["person_id"].strip() == p]
        its.sort(key=lambda r: (int(r["cluster_idx"]), r["track_id"]))
        groups.append({"pid": p, "ncl": len({int(r["cluster_idx"]) for r in its}),
                       "items": [{"k": r["track_id"], "cl": int(r["cluster_idx"]),
                                  "jpg": r["prompt_jpg"]} for r in its]})

    print(f"[embed] {len(need)} 张图 …")
    imgs = {}
    for i, p in enumerate(sorted(need), 1):
        im = Image.open(os.path.join(GT, p)).convert("RGB")
        im.thumbnail((1000, 20000), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=68, method=4)
        imgs[p] = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
        if i % 10 == 0 or i == len(need):
            print(f"  {i}/{len(need)}", end="\r", flush=True)
    print()

    html = (TPL
            .replace("__DATA__", json.dumps({"items": items, "groups": groups, "pids": all_pids},
                                            ensure_ascii=False, separators=(",", ":")))
            .replace("__IMGS__", json.dumps(imgs, ensure_ascii=False, separators=(",", ":")))
            .replace("__REVIEW__", json.dumps(REVIEW_PIDS, ensure_ascii=False)))
    open(DST, "w", encoding="utf-8").write(html)

    a = sum(1 for i in items if i["mode"] == "review")
    b = len(items) - a
    print(f"[out] {DST}  {os.path.getsize(DST)/1024/1024:.1f} MB")
    print(f"  A 复核 {a} 条｜B 补标 {b} 条｜图片 {len(imgs)} 张｜已有身份 {len(all_pids)} 个")


if __name__ == "__main__":
    build()
