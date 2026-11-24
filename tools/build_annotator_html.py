#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 D3 / D4 / D5 三份标注模板 + 874 张拼图打包成一个可离线使用的标注页。

设计要点（对应清单里的纪律）：
  - D3 / D4 / D5 三个独立标签页，各自独立的存储命名空间；D5 默认看不到 D3/D4 的身份库
  - 界面**不显示簇号、分数、接受状态**，只给轨迹 id、机位、帧数与 8 帧拼图
  - 顺序按「机位 → 时间」排列（自然的浏览顺序），不按簇排（避免被系统分组带偏）
  - 身份库：标注过程中自动累积，缩略图 + 编号，点击即选用
  - 导出直接生成**完整 CSV**（模板原有列 + 人工列），不需要再跑 merge 脚本

用法：python tools/build_annotator_html.py
输出：deliverable/data-request-v2/gt/annotate.html
"""
from __future__ import annotations

import argparse
import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GT = os.path.join(ROOT, "deliverable/data-request-v2/gt")
LEDGER = os.path.join(ROOT, "deliverable/data-request-v2/ledger")

TPL = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>1012 轨迹身份标注 · D3/D4/D5</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#171a1f;--mut:#6b7280;--line:#e2e5ea;--acc:#2563eb;--accbg:#eff6ff;--ok:#15803d;--warn:#b45309;--bad:#b91c1c}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:20;background:var(--card);border-bottom:1px solid var(--line);padding:8px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:650}
.tabs{display:flex;gap:4px;background:#eef1f5;padding:3px;border-radius:9px}
.tab{padding:5px 12px;border-radius:7px;cursor:pointer;font-size:13px;color:var(--mut);white-space:nowrap}
.tab.on{background:#fff;color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.08);font-weight:600}
.tab b{font-weight:600;color:var(--acc)}
.sp{flex:1}
.pill{border:1px solid var(--line);border-radius:999px;padding:2px 9px;color:var(--mut);font-size:12px;white-space:nowrap}
.pill.on{background:var(--accbg);border-color:#bfdbfe;color:var(--acc)}
button{padding:7px 12px;border-radius:8px;border:1px solid var(--line);background:#fff;cursor:pointer;font-size:13px;color:var(--ink)}
button:hover{border-color:#c3c9d1}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button.primary:hover{background:#1d4ed8}
button.ghost{color:var(--mut)}
main{display:grid;grid-template-columns:minmax(520px,1fr) 380px;gap:14px;padding:14px;max-width:1560px;margin:0 auto;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:12px}
#imgwarn{display:none;padding:11px 16px;background:#3a1418;border-bottom:1px solid #7a2a33;
  color:#ffd9dd;font-size:13px;line-height:1.7}
#imgwarn b{color:#ff9aa6}
#imgwarn code{background:#1a0d10;padding:1px 5px;border-radius:4px;color:#ffc4cc;
  font-family:ui-monospace,Consolas,monospace;font-size:12px}
#shot{width:100%;border-radius:8px;display:block;background:#14171c;min-height:300px;object-fit:contain}
.head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
#item{font-size:16px;font-weight:650;letter-spacing:.2px}
.meta{color:var(--mut);font-size:12px}
.bar{height:5px;background:#e9ecf1;border-radius:999px;overflow:hidden;margin-top:10px}
.bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
h2{font-size:12px;color:var(--mut);font-weight:650;margin:0 0 8px;letter-spacing:.3px;text-transform:uppercase}
#idgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(74px,1fr));gap:6px;max-height:min(46vh,420px);overflow:auto;padding-right:2px}
.idcard{border:1px solid var(--line);border-radius:8px;overflow:hidden;cursor:pointer;background:#fafbfc;position:relative}
.idcard:hover{border-color:var(--acc)}
.idcard.sel{border-color:var(--acc);box-shadow:0 0 0 2px #dbeafe}
.idcard img{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#14171c}
.idcard span{display:block;font-size:10px;text-align:center;padding:2px 0 3px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.idcard i{position:absolute;top:2px;right:2px;background:#111a;color:#fff;font-style:normal;font-size:10px;padding:0 4px;border-radius:6px}
.idcard .mag{position:absolute;top:2px;left:2px;background:#111c;color:#fff;border:0;border-radius:6px;font-size:10px;line-height:1;padding:3px 5px;cursor:pointer;opacity:.85;font-family:inherit}
.idcard .mag:hover{opacity:1;background:var(--acc)}
#shot{cursor:zoom-in}
.zoom{position:fixed;inset:0;background:#0b0d10ee;display:none;z-index:200;padding:16px;overflow:auto}
.zoom.on{display:block}
.zoombar{position:sticky;top:0;z-index:2;display:flex;gap:10px;align-items:center;justify-content:space-between;background:#11151c;border:1px solid #2a2f38;border-radius:10px;padding:8px 12px;margin-bottom:12px;color:#e5e7eb;font-size:13px}
.zoombar button{background:#1f2937;color:#e5e7eb;border:1px solid #374151;border-radius:7px;padding:5px 10px;font-size:12px;cursor:pointer}
.zoombar button:hover{background:#374151}
.zoomgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.zcell{background:#11151c;border:1px solid #2a2f38;border-radius:10px;overflow:hidden;cursor:pointer}
.zcell:hover{border-color:#3b82f6}
.zcell img{width:100%;height:300px;object-fit:contain;background:#0b0d10;display:block}
.zcell div{font-size:12px;color:#cbd5e1;padding:6px 8px;display:flex;justify-content:space-between;gap:8px}
.zcell b{color:#fff;font-weight:600;font-family:ui-monospace,Menlo,Consolas,monospace}
.zsec{display:flex;align-items:center;gap:8px;margin:16px 0 7px;color:#e5e7eb;font-size:13px}
.zsec b{color:#fff}
.zoomone{display:flex;justify-content:center;align-items:center;min-height:74vh}
.zoomone img{max-width:96vw;max-height:84vh;object-fit:contain;display:block}
input[type=text],select{width:100%;padding:8px 9px;border:1px solid var(--line);border-radius:8px;font-size:14px;background:#fff;color:var(--ink)}
.row{display:flex;gap:7px;margin-top:8px;flex-wrap:wrap;align-items:center}
.seg{display:flex;gap:0;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:6px 10px;font-size:12px;color:var(--mut)}
.seg button.on{background:var(--accbg);color:var(--acc);font-weight:600}
label.chk{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--mut);cursor:pointer}
details{margin-top:10px;border-top:1px dashed var(--line);padding-top:8px}
summary{font-size:12px;color:var(--mut);cursor:pointer;user-select:none}
.note{color:var(--mut);font-size:12px;line-height:1.65}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;font-size:11px;background:#fafbfc;font-family:inherit}
.hintbox{background:var(--accbg);border:1px solid #bfdbfe;border-radius:9px;padding:9px 11px;font-size:12px;line-height:1.65;color:#1e3a8a}
.hintbox b{color:#1e40af}
.alertbox{background:#fff7ed;border:1px solid #fed7aa;border-radius:9px;padding:9px 11px;font-size:12px;line-height:1.65;color:#7c2d12}
.tiny{font-size:11px;color:var(--mut)}
table.kv{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}
table.kv td{padding:3px 4px;border-bottom:1px solid #f1f3f5;vertical-align:top}
table.kv td.k{color:var(--mut);white-space:nowrap;width:78px}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#111827e6;color:#fff;padding:8px 16px;border-radius:9px;font-size:13px;opacity:0;transition:opacity .2s;pointer-events:none;z-index:99}
.toast.on{opacity:1}
@media (max-width:1080px){main{grid-template-columns:1fr}}
</style></head><body>
<header>
  <h1>轨迹身份标注</h1>
  <div class="tabs" id="tabs"></div>
  <span class="pill" id="prog">0 / 0</span>
  <span class="pill" id="idcnt">身份 0</span>
  <span class="sp"></span>
  <button class="ghost" onclick="jumpUnmarked()">跳到未标</button>
  <button class="ghost" onclick="exportCur()">导出本页 CSV</button>
</header>

<div id="imgwarn">
  <b>图片没有加载出来（已失败 <span id="imgfailcnt">0</span> 张）。</b>
  最常见的原因是用<b>预览面板</b>打开——它只提供这一个 HTML，<code>crops/</code> 下的图片取不到。
  改用任一种方式即可：<br>
  ① 双击同目录的 <code>annotate_standalone.html</code>（图片已内嵌，任何打开方式都能显示，<b>推荐</b>）；<br>
  ② 双击 <code>annotate.html</code>，用浏览器本地打开；<br>
  ③ 双击 <code>启动标注页.bat</code>（自动起本地服务再开浏览器）。
</div>

<main>
  <div class="card">
    <div class="head">
      <div>
        <div id="item">—</div>
        <div class="meta" id="meta"></div>
      </div>
      <div class="row" style="margin:0">
        <button class="ghost" onclick="jump(-1)">← 上一条</button>
        <button class="ghost" onclick="jump(1)">下一条 →</button>
      </div>
    </div>
    <img id="shot" alt="" onerror="imgErr()" onclick="zoomOne()" title="点击放大"> 
    <div class="bar"><i id="bar"></i></div>
    <div class="hintbox" style="margin-top:12px" id="hint"></div>
    <div class="note" style="margin-top:10px">
      <kbd>Enter</kbd> 确认并下一条 ·
      <kbd>U</kbd> 看不清 ·
      <kbd>S</kbd> 跳过 ·
      <kbd>←</kbd><kbd>→</kbd> 翻页 ·
      <kbd>1</kbd>–<kbd>9</kbd> 选身份库前 9 个 ·
      <kbd>Z</kbd> 核对大图
    </div>
  </div>

  <div class="card">
    <h2>身份库 <span class="tiny" id="poolhint"></span></h2>
    <div style="display:flex;gap:6px;margin-bottom:8px">
      <input type="text" id="idsearch" placeholder="筛选编号…" oninput="renderPool()" style="flex:1">
      <button class="ghost" onclick="newId()" title="按下一个可用编号新建">新建</button>
      <button class="ghost" onclick="zoomAll()" title="把全部已标条目按身份铺开看大图，核对标错没有">核对</button>
    </div>
    <div id="idgrid"></div>

    <h2 style="margin-top:14px">录入</h2>
    <input type="text" id="pid" placeholder="输入新编号，或点上方身份卡" autocomplete="off">
    <div class="row">
      <div class="seg" id="boxtype">
        <button data-v="body" class="on" onclick="setBox('body')">全身</button>
        <button data-v="head" onclick="setBox('head')">头/半身</button>
        <button data-v="partial" onclick="setBox('partial')">残缺</button>
      </div>
      <select id="unkreason" style="width:auto;flex:1;min-width:110px">
        <option value="far">远场模糊</option>
        <option value="occluded">遮挡</option>
        <option value="back">只有背影</option>
        <option value="other">其他</option>
      </select>
    </div>
    <div class="row">
      <button class="primary" onclick="save('ok')" style="flex:2">确认并下一条（Enter）</button>
      <button onclick="save('unk')" style="flex:1">看不清 U</button>
      <button class="ghost" onclick="save('skip')">跳过 S</button>
    </div>

    <details>
      <summary>更多字段（轨迹内换人 / 重复检测 / 备注）</summary>
      <div class="row">
        <label class="chk"><input type="checkbox" id="idswitch"> 轨迹内换人</label>
        <input type="text" id="swframe" placeholder="换人帧号（如 26660）" style="flex:1;min-width:120px">
      </div>
      <div class="row" id="duprow">
        <input type="text" id="dup" placeholder="duplicate_of：同一人的另一条轨迹 id（可选）">
      </div>
      <div class="row">
        <input type="text" id="note" placeholder="备注（可选）">
      </div>
    </details>

    <h2 style="margin-top:14px">当前条目</h2>
    <table class="kv">
      <tr><td class="k">条目</td><td id="s_id">—</td></tr>
      <tr><td class="k">机位</td><td id="s_cam">—</td></tr>
      <tr><td class="k">帧数</td><td id="s_n">—</td></tr>
      <tr><td class="k">已标</td><td id="s_v">未标</td></tr>
    </table>

    <div class="row" style="margin-top:12px">
      <input type="text" id="annotator" placeholder="标注人（写入 CSV 的 annotator 列）" oninput="saveAnnotator()" style="flex:1">
    </div>
    <div class="row" style="margin-top:8px">
      <label class="chk"><input type="checkbox" id="showref" onchange="onShowRef()"> D5 参考 D3/D4 身份库</label>
    </div>
    <div class="alertbox" id="d5warn" style="display:none;margin-top:8px">
      D5 应由<b>第二个人</b>独立完成。开启参考后本页会在导出文件里记一笔，供论文声明一致性时核对。
    </div>
  </div>
</main>
<div class="toast" id="toast"></div>
<div class="zoom" id="zoom" onclick="if(event.target.id==='zoom')zoomClose()">
  <div class="zoombar">
    <div id="zoomtitle"></div>
    <button onclick="zoomClose()">关闭（Esc）</button>
  </div>
  <div id="zoombody"></div>
</div>

<script>
const DATA = __DATA__;
const IMGS = __IMGS__;          /* 内嵌版：{相对路径: data:image/webp;base64,...}；外链版为空对象 */
function srcOf(it){ return IMGS[it.jpg] || it.jpg; }
let imgFail = 0;
function imgErr(){
  imgFail++;
  const w = document.getElementById('imgwarn');
  if(w){ w.style.display = 'block';
         document.getElementById('imgfailcnt').textContent = imgFail; }
}
const MODES = [
  {k:'d3', name:'D3 图库层', file:'gallery_members'},
  {k:'d4', name:'D4 查询层', file:'query_gt'},
  {k:'d5', name:'D5 复标',   file:'annotator2'}
];
const KEY = m => '__NS__' + m;
const store = {};
MODES.forEach(m => { try { store[m.k] = JSON.parse(localStorage.getItem(KEY(m.k)) || '{}'); } catch(e){ store[m.k] = {}; } });
let mode = 'd3', idx = 0, selPid = '', curBox = 'body';

function items(){ return DATA[mode]; }
function cur(){ return items()[idx]; }
function modeMeta(){ return MODES.find(m => m.k === mode); }
function persist(){ localStorage.setItem(KEY(mode), JSON.stringify(store[mode])); }
function saveAnnotator(){ localStorage.setItem('__NS__annotator', document.getElementById('annotator').value.trim()); }
let tEnter = Date.now();

/* ---------------- 身份库 ---------------- */
function poolSrc(){
  if(mode === 'd5') return document.getElementById('showref').checked ? ['d5','d3','d4'] : ['d5'];
  return ['d3','d4'];
}
function idPool(){
  const pool = new Map();
  poolSrc().forEach(m => {
    (DATA[m]||[]).forEach(it => {
      const r = store[m][it.k];
      if(!r || !r.pid) return;
      if(pool.has(r.pid)) pool.get(r.pid).n++;
      else pool.set(r.pid, {jpg: it.jpg, n: 1});
    });
  });
  return [...pool.entries()].sort((a,b) => a[0].localeCompare(b[0], 'zh'));
}
function nextId(){
  let mx = 0;
  Object.keys(store).forEach(m => Object.keys(store[m]).forEach(k => {
    const pid = store[m][k].pid; if(!pid) return;
    const g = /P1012-(\d+)/.exec(pid); if(g) mx = Math.max(mx, +g[1]);
  }));
  return 'P1012-' + String(mx + 1).padStart(3, '0');
}
function newId(){ document.getElementById('pid').value = nextId(); document.getElementById('pid').focus(); }
function pick(pid){
  selPid = pid;
  document.getElementById('pid').value = pid;
  document.querySelectorAll('.idcard').forEach(el => el.classList.toggle('sel', el.dataset.pid === pid));
}

/* ---------------- 渲染 ---------------- */
function renderTabs(){
  document.getElementById('tabs').innerHTML = MODES.map(m => {
    const done = Object.keys(store[m.k]).length, tot = (DATA[m.k]||[]).length;
    return `<div class="tab ${m.k===mode?'on':''}" onclick="switchMode('${m.k}')">${m.name} <b>${done}</b>/${tot}</div>`;
  }).join('');
}
function renderPool(){
  const q = (document.getElementById('idsearch').value || '').trim().toLowerCase();
  const pool = idPool().filter(([pid]) => !q || pid.toLowerCase().includes(q));
  document.getElementById('poolhint').textContent = mode==='d5'
    ? (document.getElementById('showref').checked ? '（含 D3/D4）' : '（独立，不看 D3/D4）')
    : '（D3+D4 共用）';
  document.getElementById('idcnt').textContent = '身份 ' + idPool().length;
  document.getElementById('idgrid').innerHTML = pool.length ? pool.map(([pid, info]) =>
    `<div class="idcard ${pid===selPid?'sel':''}" data-pid="${pid}" onclick="pick('${pid}')" title="${pid}">
       <img loading="lazy" src="${IMGS[info.jpg]||info.jpg}" alt="${pid}" onerror="this.style.opacity=.25"><i>${info.n}</i>
       <button class="mag" title="放大核对（${info.n} 条）" onclick="event.stopPropagation();zoomPid('${pid}')">放大</button><span>${pid}</span>
     </div>`).join('')
    : `<div class="note" style="grid-column:1/-1;padding:8px 2px">还没有身份。输入编号或点「新建」开始建档。</div>`;
}
function setBox(v){
  curBox = v;
  document.querySelectorAll('#boxtype button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
}
function renderState(){
  const it = cur();
  if(!it){ document.getElementById('item').textContent = '（本页已无条目）'; return; }
  document.getElementById('item').textContent = it.k;
  document.getElementById('meta').textContent = it.cam + ' · ' + it.n + ' 帧';
  const im = document.getElementById('shot');
  const _s = srcOf(it);
  if(im.getAttribute('src') !== _s) im.setAttribute('src', _s);
  document.getElementById('s_id').textContent = it.k;
  document.getElementById('s_cam').textContent = it.cam;
  document.getElementById('s_n').textContent = it.n;
  const r = store[mode][it.k];
  document.getElementById('s_v').textContent = r
    ? ((r.unk ? 'UNK（' + (r.unkreason||'') + '）' : r.pid) + (r.box ? ' · ' + r.box : ''))
    : '未标';
  document.getElementById('pid').value = (r && r.pid) ? r.pid : '';
  selPid = (r && r.pid) ? r.pid : '';
  setBox((r && r.box) ? r.box : (it.boxauto || 'body'));
  document.getElementById('unkreason').value = (r && r.unkreason) ? r.unkreason : 'far';
  document.getElementById('idswitch').checked = !!(r && r.idswitch);
  document.getElementById('swframe').value = (r && r.idswitchframe) ? r.idswitchframe : '';
  document.getElementById('dup').value = (r && r.duplicate_of) ? r.duplicate_of : '';
  document.getElementById('note').value = (r && r.note) ? r.note : '';
  document.getElementById('duprow').style.display = (mode === 'd4') ? 'flex' : 'none';

  const done = Object.keys(store[mode]).length, tot = items().length;
  document.getElementById('prog').textContent = (idx+1) + ' / ' + tot + ' · 已标 ' + done;
  document.getElementById('bar').style.width = (100 * done / tot).toFixed(1) + '%';
  document.getElementById('hint').innerHTML = hintText();
  tEnter = Date.now();
  renderPool(); renderTabs();
}
function hintText(){
  if(mode === 'd3') return '<b>图库层：只回答「这个图库成员是谁」。</b>不要看 D4 的查询结果。同一人在不同机位出现时，从右侧身份库选同一个编号。';
  if(mode === 'd4') return '<b>查询层：只回答「这条查询是谁」。</b>图库里没有这个人就新建 <b>NEW-xxx</b>。只看图，本页不显示系统给了什么结果。';
  return '<b>复标：独立重标，不看第一遍的结果。</b>前 104 条是关键簇的图库成员——先把它们标完，你就有了自己的身份库；后面 139 条查询用它来判。建议用<b>无痕窗口</b>打开，别勾右侧的「参考 D3/D4」。';
}
function onShowRef(){
  document.getElementById('d5warn').style.display = document.getElementById('showref').checked ? 'block' : 'none';
  renderPool();
}
function switchMode(k){ mode = k; idx = 0; renderState(); }
function jump(d){ idx = Math.max(0, Math.min(items().length - 1, idx + d)); renderState(); }
function jumpUnmarked(){
  const i = items().findIndex(it => !store[mode][it.k]);
  if(i < 0){ toast('本页已全部标完'); return; }
  idx = i; renderState();
}

/* ---------------- 保存 ---------------- */
function save(kind){
  const it = cur(); if(!it) return;
  if(kind === 'skip'){ jump(1); return; }
  const pid = document.getElementById('pid').value.trim();
  const rec = {
    box: curBox,
    idswitch: document.getElementById('idswitch').checked ? 1 : 0,
    idswitchframe: document.getElementById('swframe').value.trim(),
    duplicate_of: document.getElementById('dup').value.trim(),
    note: document.getElementById('note').value.trim()
  };
  if(kind === 'unk'){
    rec.pid = ''; rec.unk = 1; rec.unkreason = document.getElementById('unkreason').value;
  } else {
    if(!pid){ document.getElementById('pid').focus(); toast('先填编号，或按 U 记看不清'); return; }
    rec.pid = pid; rec.unk = 0; rec.unkreason = '';
  }
  if(mode === 'd5' && document.getElementById('showref').checked) rec.saw_ref = 1;
  rec.sec = Math.max(0, Math.round((Date.now() - tEnter) / 1000));
  store[mode][it.k] = rec;
  persist(); next();
  toast((kind==='unk' ? '已记 UNK' : '已标 ' + rec.pid) + ' · ' + it.k);
}
function next(){
  const i = items().findIndex((x, k) => k > idx && !store[mode][x.k]);
  idx = (i >= 0) ? i : Math.min(idx + 1, items().length - 1);
  renderState();
}
let toastT = null;
function toast(s){
  const el = document.getElementById('toast');
  el.textContent = s; el.classList.add('on');
  clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('on'), 1400);
}

/* ---------------- 导出 ---------------- */
function toCSV(rows){
  if(!rows.length) return '';
  const head = Object.keys(rows[0]);
  const esc = v => '"' + String(v === undefined || v === null ? '' : v).replace(/"/g, '""') + '"';
  const lines = [head.map(esc).join(',')];
  rows.forEach(r => lines.push(head.map(h => esc(r[h])).join(',')));
  return '\ufeff' + lines.join('\n');
}
function download(name, text){
  const blob = new Blob([text], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
function exportCur(){
  const m = modeMeta();
  const rows = items().filter(it => store[mode][it.k]).map(it => {
    const r = store[mode][it.k];
    const ann = {
      person_id: r.pid || '',
      box_type: r.box || '',
      id_switch: r.idswitch || '',
      id_switch_frame: r.idswitchframe || '',
      duplicate_of: r.duplicate_of || '',
      unk_reason: r.unk ? (r.unkreason || 'unsure') : '',
      evidence: (mode === 'd4') ? (r.note || '') : '',
      annotator: (document.getElementById('annotator').value || '').trim(),
      time_spent_s: r.sec || '',
      note: (r.note || '') + (r.saw_ref ? ' [saw_d34_ids]' : '')
    };
    // 只输出模板原有的列（保持列顺序与列集合完全一致），人工列覆盖同名列
    const out = {};
    Object.keys(it.raw).forEach(k => { out[k] = (k in ann) ? ann[k] : it.raw[k]; });
    return out;
  });
  if(!rows.length){ toast('本页还没有已标条目'); return; }
  download('gt_' + m.file + '_FILLED.csv', toCSV(rows));
  toast('已导出 ' + rows.length + ' 行');
}

/* ---------------- 放大核对 ---------------- */
const h = v => String(v === undefined || v === null ? '' : v)
  .replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function membersByPid(){
  const out = new Map();
  poolSrc().forEach(m => {
    (DATA[m]||[]).forEach(it => {
      const r = store[m][it.k];
      if(!r || !r.pid) return;
      if(!out.has(r.pid)) out.set(r.pid, []);
      out.get(r.pid).push({m: m, it: it, r: r});
    });
  });
  return out;
}
function zcellHTML(m, it, r){
  const src = IMGS[it.jpg] || it.jpg;
  const lab = r.unk ? 'UNK' : r.pid;
  const mark = store[m][it.k] ? '' : ' style="opacity:.45"';
  return `<div class="zcell"${mark} onclick="gotoItem('${m}','${h(it.k)}')" title="跳到 ${h(it.k)} 重新判断">
     <img loading="lazy" src="${src}" alt="" onerror="this.style.opacity=.2">
     <div><b>${h(it.k)}</b><span style="color:#94a3b8">${h(it.cam)} · ${h(lab)}</span></div>
   </div>`;
}
function zoomOpen(title, html){
  const z = document.getElementById('zoom');
  document.getElementById('zoomtitle').innerHTML = title;
  document.getElementById('zoombody').innerHTML = html;
  z.classList.add('on');
  z.scrollTop = 0;
}
function zoomClose(){
  const z = document.getElementById('zoom');
  if(!z.classList.contains('on')) return;
  z.classList.remove('on');
  document.getElementById('zoombody').innerHTML = '';
}
function zoomOne(){
  const it = cur(); if(!it) return;
  const r = store[mode][it.k];
  const lab = r ? (r.unk ? 'UNK（' + (r.unkreason||'') + '）' : r.pid) : '未标';
  zoomOpen(`放大 · <b>${h(it.k)}</b> <span style="color:#94a3b8">${h(it.cam)} · ${h(it.n)} 帧 · 当前标注：${h(lab)}</span>`,
    `<div class="zoomone"><img src="${srcOf(it)}" alt=""></div>`);
}
function zoomPid(pid){
  const arr = membersByPid().get(pid) || [];
  const body = arr.length
    ? `<div class="zoomgrid">${arr.map(x => zcellHTML(x.m, x.it, x.r)).join('')}</div>`
    : `<div style="color:#94a3b8;font-size:13px">该身份下没有条目。</div>`;
  zoomOpen(`核对 · <b>${h(pid)}</b> · 共 ${arr.length} 条 <span style="color:#94a3b8">（点任一图跳到该条）</span>`, body);
}
function zoomAll(){
  const mb = membersByPid();
  const pids = [...mb.keys()].sort((a,b) => a.localeCompare(b, 'zh'));
  if(!pids.length){ toast('还没有已标条目'); return; }
  let n = 0;
  const body = pids.map(pid => {
    const arr = mb.get(pid); n += arr.length;
    return `<div class="zsec"><b>${h(pid)}</b><span style="color:#94a3b8">· ${arr.length} 条</span></div>
      <div class="zoomgrid">${arr.map(x => zcellHTML(x.m, x.it, x.r)).join('')}</div>`;
  }).join('');
  zoomOpen(`核对 · 全部已标 <b>${pids.length}</b> 个身份 / <b>${n}</b> 条 <span style="color:#94a3b8">（点图跳转）</span>`, body);
}
function gotoItem(m, k){
  zoomClose();
  const i = (DATA[m]||[]).findIndex(x => x.k === k);
  if(i < 0) return;
  mode = m; idx = i;
  renderState();
  window.scrollTo({top: 0, behavior: 'smooth'});
  toast('已跳到 ' + k);
}

/* ---------------- 键盘 ---------------- */
document.addEventListener('keydown', e => {
  if(document.getElementById('zoom').classList.contains('on')){
    if(e.key === 'Escape'){ e.preventDefault(); zoomClose(); }
    return;
  }
  const tag = (e.target.tagName || '').toLowerCase();
  if(tag === 'input' || tag === 'select'){
    if(e.key === 'Enter'){ e.preventDefault(); save('ok'); }
    return;
  }
  if(e.key === 'Enter'){ e.preventDefault(); save('ok'); }
  else if(e.key === 'u' || e.key === 'U'){ save('unk'); }
  else if(e.key === 's' || e.key === 'S'){ save('skip'); }
  else if(e.key === 'ArrowRight'){ jump(1); }
  else if(e.key === 'ArrowLeft'){ jump(-1); }
  else if(e.key === 'z' || e.key === 'Z'){ zoomAll(); }
  else if(/^[1-9]$/.test(e.key)){
    const cards = [].slice.call(document.querySelectorAll('.idcard'));
    const c = cards[+e.key - 1];
    if(c) pick(c.dataset.pid);
  }
});

renderState();
document.getElementById('annotator').value = localStorage.getItem('__NS__annotator') || '';
</script></body></html>
"""


def read_csv(name):
    p = os.path.join(GT, name)
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(GT, "annotate.html"))
    ap.add_argument("--embed", action="store_true",
                    help="把每张拼图压成 WebP 内嵌进 HTML，产出单文件离线版")
    ap.add_argument("--maxw", type=int, default=960, help="内嵌时的最大宽度")
    ap.add_argument("--quality", type=int, default=58, help="内嵌时的 WebP 质量")
    ap.add_argument("--a3", action="store_true",
                    help="产出第二标注人盲标页：图库= a3_gallery_TEMPLATE.csv（257 条全量）、"
                         "查询= a3_query_TEMPLATE.csv（100 条分层抽样）、无 D5；自动内嵌")
    a = ap.parse_args()
    if a.a3:
        if os.path.abspath(a.out) == os.path.abspath(os.path.join(GT, "annotate.html")):
            a.out = os.path.join(GT, "annotate_a3.html")
        a.embed = True

    tstart, tbox, tlen = {}, {}, {}
    for r in csv.DictReader(open(os.path.join(LEDGER, "tracks.csv"), encoding="utf-8-sig")):
        tstart[r["track_id"]] = int(r["t_start_frame"])
        tbox[r["track_id"]] = r["box_type_auto"]
        tlen[r["track_id"]] = r["n_frames"]

    def norm(rows, idd, jpgd):
        out = []
        for r in rows:
            key = r[idd]
            seg = r.get("seg") or key.split("#")[0]
            tid = key.split("#")[1]
            out.append({
                "k": key, "jpg": r[jpgd],
                "cam": r.get("cam", ""),
                "n": (r.get("n_frames") or "").strip(),
                "boxauto": "head" if tbox.get(f"{seg}#{tid}") == "head_like" else "body",
                "raw": r,
                "_t": tstart.get(f"{seg}#{tid}", 0),
            })
        out.sort(key=lambda x: (x["cam"], x["_t"], x["k"]))
        for x in out:
            x.pop("_t", None)
        return out

    d3 = norm(read_csv("a3_gallery_TEMPLATE.csv" if a.a3 else "gallery_members_TEMPLATE.csv"),
              "track_id", "prompt_jpg")
    d4 = norm(read_csv("a3_query_TEMPLATE.csv" if a.a3 else "query_gt_TEMPLATE.csv"),
              "query_id", "prompt_jpg")
    d5 = []
    if not a.a3:
        for r in read_csv("annotator2_TEMPLATE.csv"):
            seg, tid = r["item_id"].split("#")[0], r["item_id"].split("#")[1]
            d5.append({"k": r["item_id"], "jpg": r["prompt_jpg"], "cam": r["cam"],
                       "n": tlen.get(r["item_id"], ""), "boxauto": "body", "raw": r,
                       "_t": tstart.get(f"{seg}#{tid}", 0)})
        d5.sort(key=lambda x: (x["raw"].get("layer", ""), x["cam"], x["_t"], x["k"]))
        for x in d5:
            x.pop("_t", None)

    data = {"d3": d3, "d4": d4, "d5": d5}
    imgs = {}
    if a.embed:
        from PIL import Image
        import base64
        import io
        keys = sorted({x["jpg"] for m in data.values() for x in m})
        print(f"[embed] 去重后 {len(keys)} 张，压到宽 ≤{a.maxw} WebP q{a.quality} …")
        for i, p in enumerate(keys, 1):
            fp = os.path.join(GT, p)
            im = Image.open(fp).convert("RGB")
            im.thumbnail((a.maxw, 20000), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=a.quality, method=4)
            imgs[p] = "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()
            if i % 100 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", end="\r", flush=True)
        print()

    html = (TPL
            .replace("__NS__", "a3_blind_" if a.a3 else "ann1012_v2_")
            .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            .replace("__IMGS__", json.dumps(imgs, ensure_ascii=False, separators=(",", ":"))))
    if a.a3:
        # 导出文件名加 a3_ 前缀，避免与第一标注人的 gt_*_FILLED.csv 冲突
        html = (html.replace("file:'gallery_members'", "file:'a3_gallery_members'")
                    .replace("file:'query_gt'", "file:'a3_query_gt'"))
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"\n[out] {a.out}")
    print(f"  D3 {len(d3)} 条 / D4 {len(d4)} 条 / D5 {len(d5)} 条"
          f" · 内嵌图 {len(imgs)} 张 · {os.path.getsize(a.out)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
