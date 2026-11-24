#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为「female4 / male7 / male9 混淆」生成专项复核页。

与 D3 标注页的区别：
  - 它显示**原标注**（这是纠错，不是盲标）
  - 每条轨迹右侧给出**同簇其他成员**的原标注，用于横向对比
  - 另有一个「身份总览」页：把同一 person_id 的全部轨迹并排，
    用来判断「这些到底是不是同一个人」

输出  gt/review_female4_male7_male9.html（图片内嵌，双击即用）
导出  review_female4_male7_male9_FILLED.csv（track_id, old_pid, new_pid, note）
"""
import base64
import collections
import csv
import io
import json
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GT = os.path.join(ROOT, "deliverable/data-request-v2/gt")

TARGET = ["female4", "male7", "male9"]
SRC = os.path.join(GT, "gt_gallery_members_FILLED.csv")
DST = os.path.join(GT, "review_female4_male7_male9.html")

TPL = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>复核 female4 / male7 / male9</title>
<style>
:root{--bg:#0f1216;--card:#171b21;--line:#262c35;--fg:#e6e9ee;--dim:#8b95a5;
      --hot:#ff6b6b;--ok:#3ddc97;--acc:#4d9fff}
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
button.pri:hover{filter:brightness(1.1)}
main{padding:16px;max-width:1500px;margin:0 auto}
.wrap{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:16px}
@media(max-width:1100px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.shot{width:100%;border-radius:9px;display:block;background:#0a0d11}
.meta{color:var(--dim);font-size:12.5px;margin-top:8px}
.meta b{color:var(--fg);font-weight:600}
.old{display:inline-block;padding:2px 9px;border-radius:6px;background:#2a1c1c;
     color:#ffb3b3;font-weight:600;font-size:13px}
h2{font-size:13px;color:var(--dim);margin:0 0 10px;font-weight:600;letter-spacing:.4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(118px,1fr));gap:8px}
.tile{position:relative;border-radius:8px;overflow:hidden;background:#0a0d11;
      outline:1px solid var(--line);cursor:zoom-in}
.tile img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
.tile span{position:absolute;left:0;right:0;bottom:0;padding:3px 5px;font-size:11px;
           background:linear-gradient(transparent,#000c);color:#fff;font-weight:600}
.tile.me{outline:2px solid var(--hot)}
.tile.me span{background:linear-gradient(transparent,#5a1212)}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:14px;
     padding-top:14px;border-top:1px solid var(--line)}
.opt{padding:9px 16px;border-radius:9px;background:#1d232b;color:var(--fg);
     border:1px solid var(--line);font-weight:600;font-size:13.5px}
.opt:hover{background:#28313c}
.opt.on{background:#14342a;border-color:var(--ok);color:var(--ok)}
.opt.keep{border-style:dashed}
.opt kbd{font-size:10px;opacity:.55;margin-left:5px;font-family:inherit}
input[type=text]{padding:9px 12px;border-radius:9px;border:1px solid var(--line);
                 background:#0e1218;color:var(--fg);font-size:13.5px;font-family:inherit;width:130px}
.hint{color:var(--dim);font-size:12.5px;margin-top:10px}
.ids{display:grid;gap:14px}
.idrow{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
.idhead{display:flex;align-items:baseline;gap:10px;margin-bottom:10px}
.idhead b{font-size:15px}
.idhead em{color:var(--dim);font-style:normal;font-size:12.5px}
.lightbox{position:fixed;inset:0;background:#000d;display:none;z-index:99;
          align-items:center;justify-content:center;padding:24px;cursor:zoom-out}
.lightbox.on{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:10px}
.done{background:#14342a!important;border-color:var(--ok)!important}
</style></head><body>

<header>
  <h1>复核 female4 / male7 / male9</h1>
  <div class="tabs" id="tabs"></div>
  <span class="pill" id="prog">0 / 0</span>
  <span class="pill" id="chg">改动 0</span>
  <span class="sp"></span>
  <button onclick="exportCSV()" class="pri">导出 CSV</button>
</header>

<main>
  <div id="review">
    <div class="wrap">
      <div class="card">
        <img id="shot" class="shot" alt="">
        <div class="meta" id="meta"></div>
      </div>
      <div class="card">
        <h2>同簇其他成员（灰框=当前，红边=你正在看的这条）</h2>
        <div class="grid" id="sibs"></div>
        <div class="hint">先看同簇成员，判断这条和谁更像；再决定它是不是原来标的那个身份。</div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="meta">原标注：<span class="old" id="oldpid">—</span>
        <span id="orignote" style="margin-left:8px"></span></div>
      <div class="bar" id="opts"></div>
      <div class="hint">
        <kbd>1</kbd> female4 · <kbd>2</kbd> male7 · <kbd>3</kbd> male9 ·
        <kbd>Enter</kbd> 保持原标注 · <kbd>←</kbd><kbd>→</kbd> 翻页 ·
        也可在右侧输入任意身份
      </div>
    </div>
  </div>

  <div id="overview" style="display:none">
    <div class="hint" style="margin-bottom:12px">
      同一 person_id 的全部轨迹并排。若它们不像同一个人，说明这个编号被用混了——
      回到「逐条复核」改掉不合适的那几条。
    </div>
    <div class="ids" id="idlist"></div>
  </div>
</main>

<div class="lightbox" id="lb" onclick="this.classList.remove('on')"><img id="lbi" alt=""></div>

<script>
const DATA = __DATA__;
const IMG  = __IMGS__;
const KEY  = 'review_f4m7m9_v1';
let mode = 'review';
let idx = 0;
let store = JSON.parse(localStorage.getItem(KEY) || '{}');

const ITEMS = DATA.items;
const srcOf = it => IMG[it.jpg] || it.jpg;

function persist(){ localStorage.setItem(KEY, JSON.stringify(store)); }
function cur(){ return ITEMS[idx]; }

function renderTabs(){
  document.getElementById('tabs').innerHTML =
    `<div class="tab ${mode==='review'?'on':''}" onclick="setMode('review')">逐条复核 ${ITEMS.length}</div>
     <div class="tab ${mode==='overview'?'on':''}" onclick="setMode('overview')">身份总览</div>`;
}
function setMode(m){
  mode = m;
  document.getElementById('review').style.display   = m==='review'?'':'none';
  document.getElementById('overview').style.display = m==='overview'?'':'none';
  renderTabs();
  if(m==='overview') renderOverview();
}

function renderReview(){
  const it = cur();
  if(!it) return;
  document.getElementById('shot').src = srcOf(it);
  const r = store[it.k] || {};
  document.getElementById('meta').innerHTML =
    `簇 <b>${it.cl}</b> · <b>${it.k}</b> · ${it.cam} 机位 · ${it.n} 帧 · ${it.box}`
    + (it.sw ? ` · <span style="color:#ffb3b3">轨迹内换人 → ${it.sw}</span>` : '')
    + `<br>簇内身份：${it.brief}`;
  document.getElementById('oldpid').textContent = it.pid;
  document.getElementById('orignote').textContent = it.note ? `（原备注：${it.note}）` : '';

  document.getElementById('sibs').innerHTML = it.sibs.map(s =>
    `<div class="tile ${s.k===it.k?'me':''}" onclick="lb('${srcOf(s)}')">
       <img loading="lazy" src="${srcOf(s)}" alt="">
       <span>${s.k===it.k?'▶ ':''}${s.pid || '未标'}</span>
     </div>`).join('');

  const now = r.pid || '';
  const opts = TARGETS.map((p,i) =>
    `<button class="opt ${now===p?'on':''}" onclick="pick('${p}')">${p}<kbd>${i+1}</kbd></button>`).join('')
    + `<button class="opt keep ${now===''?'on':''}" onclick="keep()">保持不变<kbd>⏎</kbd></button>
       <input type="text" id="free" placeholder="其他身份…" onchange="pick(this.value.trim())"
              value="${now && !TARGETS.includes(now) ? now : ''}">`;

  document.getElementById('opts').innerHTML = opts;
  document.getElementById('prog').textContent = `${idx+1} / ${ITEMS.length}`;
  document.getElementById('chg').textContent =
    '改动 ' + Object.values(store).filter(v => v.pid && v.pid !== v.old).length;
  window.scrollTo({top:0,behavior:'smooth'});
}

function pick(p){
  const it = cur();
  if(!p) return;
  store[it.k] = {pid:p, old:it.pid};
  persist(); renderReview();
  setTimeout(() => jump(1), 120);
}
function keep(){
  const it = cur();
  store[it.k] = {pid:it.pid, old:it.pid};
  persist(); renderReview();
  setTimeout(() => jump(1), 120);
}
function jump(d){
  idx = Math.min(ITEMS.length-1, Math.max(0, idx+d));
  renderReview();
}

function renderOverview(){
  const box = document.getElementById('idlist');
  box.innerHTML = DATA.groups.map(g => {
    const tiles = g.items.map(it => {
      const r = store[it.k];
      const changed = r && r.pid && r.pid !== r.old;
      return `<div class="tile ${changed?'':''}" onclick="lb('${srcOf(it)}')">
        <img loading="lazy" src="${srcOf(it)}" alt="">
        <span>簇${it.cl} · ${it.k.split('#')[1]}</span></div>`;
    }).join('');
    return `<div class="idrow">
      <div class="idhead"><b>${g.pid}</b><em>${g.items.length} 条轨迹 · 跨 ${g.ncl} 个簇</em></div>
      <div class="grid">${tiles}</div></div>`;
  }).join('');
}

function lb(src){
  document.getElementById('lbi').src = src;
  document.getElementById('lb').classList.add('on');
}

function exportCSV(){
  const head = ['track_id','old_pid','new_pid','changed','note'];
  const lines = [head.join(',')];
  ITEMS.forEach(it => {
    const r = store[it.k] || {};
    const np = (r.pid !== undefined && r.pid !== '') ? r.pid : it.pid;
    const ch = np && np !== it.pid ? 1 : 0;
    lines.push([it.k, it.pid, np, ch, (it.note||'').replace(/[",\n]/g,' ')].join(','));
  });
  const blob = new Blob(['\ufeff'+lines.join('\n')], {type:'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'review_female4_male7_male9_FILLED.csv';
  a.click();
}

const TARGETS = __TARGETS__;
document.addEventListener('keydown', e => {
  if(mode !== 'review') return;
  if(e.key === '1') pick(TARGETS[0]);
  else if(e.key === '2') pick(TARGETS[1]);
  else if(e.key === '3') pick(TARGETS[2]);
  else if(e.key === 'Enter') keep();
  else if(e.key === 'ArrowRight') jump(1);
  else if(e.key === 'ArrowLeft') jump(-1);
});

renderTabs(); renderReview();
</script></body></html>
"""


def build():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["cluster_idx"])].append(r)

    target = [r for r in rows if r["person_id"].strip() in TARGET]
    target.sort(key=lambda r: (int(r["cluster_idx"]), r["track_id"]))

    items, need = [], set()
    for r in target:
        c = int(r["cluster_idx"])
        sibs = [s for s in by[c]]
        cnt = collections.Counter(s["person_id"].strip() or "未标" for s in sibs)
        brief = "、".join(f"{k}×{v}" for k, v in cnt.most_common())
        items.append({
            "k": r["track_id"], "cl": c, "cam": r["cam"],
            "n": r["n_frames"], "box": r["box_type"] or "body",
            "pid": r["person_id"].strip(),
            "note": (r["note"] or "").strip(),
            "sw": (r["note"] or "").strip() if (r["id_switch"] or "").strip() not in ("", "0") else "",
            "jpg": r["prompt_jpg"],
            "brief": brief,
            "sibs": [{"k": s["track_id"], "pid": s["person_id"].strip(),
                      "jpg": s["prompt_jpg"]} for s in sibs],
        })
        need.add(r["prompt_jpg"])
        for s in sibs:
            need.add(s["prompt_jpg"])

    # 身份总览
    groups = []
    for p in TARGET:
        its = [r for r in rows if r["person_id"].strip() == p]
        its.sort(key=lambda r: (int(r["cluster_idx"]), r["track_id"]))
        groups.append({
            "pid": p, "ncl": len({int(r["cluster_idx"]) for r in its}),
            "items": [{"k": r["track_id"], "cl": int(r["cluster_idx"]),
                       "jpg": r["prompt_jpg"]} for r in its],
        })

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
            .replace("__DATA__", json.dumps({"items": items, "groups": groups},
                                            ensure_ascii=False, separators=(",", ":")))
            .replace("__IMGS__", json.dumps(imgs, ensure_ascii=False, separators=(",", ":")))
            .replace("__TARGETS__", json.dumps(TARGET, ensure_ascii=False)))
    open(DST, "w", encoding="utf-8").write(html)
    print(f"[out] {DST}  {os.path.getsize(DST)/1024/1024:.1f} MB")
    print(f"  待复核 {len(items)} 条｜涉及 {len({i['cl'] for i in items})} 个簇"
          f"｜参考成员 {len(need)} 张")


if __name__ == "__main__":
    build()
