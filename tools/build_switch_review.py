#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-A：为 46 条串人成员生成「重标切换时刻」的复核页（带跟踪框）。

为什么需要：`id_switch_frame` 里 19 条恰好等于轨迹末帧、4 条是多值文本，
用它们做「切点命中 ±1s」判据等于在轨迹末尾切，判据不成立（见 docs/m1-kickoff §1.3）。

做法：
  1. 原视频没有跟踪框，人眼无法判断「框在哪一刻换了人」；
     因此先用 `tools/render_switch_clips.py`（supervision 画框 + ffmpeg）渲染两条片段：
       focus  原记录点 ±8s（不可用时取轨迹末 20s；覆盖过低时滑到最密处），原生 25fps
       scan   整条轨迹，5fps 抽样，时长与真实一致
  2. 本脚本生成页面，播放这两条片段；帧号口径与 ledger 一致（25fps）
  3. 空格播放/暂停，←/→ ±0.5s，Enter 记录，X 标「无法判定」，Tab 下一条
     localStorage 续标，可导出 CSV（含 note 列）

窗口用 `tools/switch_windows.py` 的同一份 JSON，避免页面与渲染对不上。

用法：
  PY tools/switch_windows.py        # 1) 算窗口
  PY tools/render_switch_clips.py   # 2) 渲染片段
  PY tools/build_switch_review.py   # 3) 生成页面
输出：deliverable/data-request-v2/gt/switch_review.html
"""
import json
import os

from switch_windows import load_items, FPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, 'deliverable/data-request-v2')
OUT = os.path.join(DELIV, 'gt/switch_review.html')

TPL = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>M1-A 换人时刻复核（46 条 · 带框）</title>
<style>
:root{--bg:#0f1218;--card:#171c24;--line:#2a3240;--ink:#e8edf5;--mut:#93a1b5;--acc:#4f8cff;--ok:#2ec27e;--warn:#f5a524}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:5;background:#121820;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:650}
.pill{border:1px solid var(--line);border-radius:999px;padding:2px 10px;color:var(--mut);font-size:12px}
button{background:#1e2734;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 11px;font-size:13px;cursor:pointer}
button:hover{border-color:#3a4a63}
button.on{background:#1b2a44;border-color:#2f4a75;color:#cfe0ff}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
button.ok{background:#1d3a2c;border-color:#2b6b4d;color:#9ff0c2}
button.bad{background:#3a1f1f;border-color:#6b2b2b;color:#f0a2a2}
main{display:grid;grid-template-columns:1fr 350px;gap:14px;padding:14px;max-width:1560px;margin:0 auto;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
video{width:100%;border-radius:8px;background:#000;display:block}
#item{font-size:17px;font-weight:650;letter-spacing:.2px}
.meta{color:var(--mut);font-size:12.5px;margin-top:3px}
.kbd{display:inline-block;border:1px solid var(--line);border-bottom-width:2px;border-radius:5px;padding:0 6px;font:12px ui-monospace,Consolas,monospace;color:var(--mut)}
#cur{font:13px ui-monospace,Consolas,monospace;color:var(--acc);margin-top:6px}
.bar{height:5px;background:#222b38;border-radius:999px;overflow:hidden;margin:10px 0 2px}
.bar>i{display:block;height:100%;background:var(--ok);width:0}
ol{list-style:none;padding:0;margin:0;max-height:600px;overflow:auto}
li{padding:7px 10px;border-radius:8px;cursor:pointer;display:flex;justify-content:space-between;gap:8px;font-size:13px;border:1px solid transparent}
li:hover{background:#1c2532}
li.on{background:#1b2a44;border-color:#2f4a75}
li.done{color:#9ff0c2}
li.und{color:#f0a2a2}
li .tag{font-size:11px;color:var(--mut)}
.warn{color:var(--warn);font-size:12px;margin-top:8px;line-height:1.5}
.hint{color:var(--mut);font-size:12px;margin-top:6px}
</style></head><body>
<header>
  <h1>M1-A 换人时刻复核</h1>
  <span class="pill" id="prog">0 / 0</span>
  <span class="pill"><span class="kbd">空格</span> 播放 · <span class="kbd">←</span><span class="kbd">→</span> 0.5s · <span class="kbd">Enter</span> 记录 · <span class="kbd">X</span> 无法判定 · <span class="kbd">Tab</span> 下一条</span>
  <button id="btnFocus" class="on" onclick="setView('focus')">聚焦片段</button>
  <button id="btnScan" onclick="setView('scan')">全轨迹扫描</button>
  <button onclick="jump(-0.5)">−0.5s</button>
  <button onclick="jump(0.5)">+0.5s</button>
  <button class="ok" onclick="mark()">记录换人点</button>
  <button class="bad" onclick="markBad()">无法判定</button>
  <button class="primary" onclick="exportCSV()">导出 CSV</button>
</header>
<main>
  <div class="card">
    <div id="item">—</div>
    <div class="meta" id="meta"></div>
    <div class="bar"><i id="bar"></i></div>
    <div id="cur">—</div>
    <video id="v" controls preload="metadata" style="margin-top:8px"></video>
    <div class="hint" id="hint"></div>
    <div class="warn" id="warn"></div>
  </div>
  <div class="card">
    <div class="meta" style="margin-bottom:6px">46 条串人轨迹（绿＝已记录，红＝无法判定）</div>
    <ol id="list"></ol>
  </div>
</main>
<script>
const ITEMS = __ITEMS__;
const CLIPS = '__CLIPS_BASE__';
const FPS = 25;
const KEY = 'm1_switch_review_v3';
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { store = {}; }
let idx = 0, view = 'focus', curSrc = '';
const v = document.getElementById('v');

function item(){ return ITEMS[idx]; }
function startF(){ return item().start_f[view]; }
function clipURL(){ return CLIPS + '/' + item().file_base + '_' + view + '.mp4'; }
function realFrame(){ return Math.round(startF() + v.currentTime * FPS); }

function render(seekReal){
  const it = item();
  document.getElementById('item').textContent = it.track_id;
  document.getElementById('meta').textContent =
    it.seg + ' · ' + it.cam + ' · ' + it.n_frames + ' 帧 · 轨迹 ' + it.t0.toFixed(1) + '–' + it.t1.toFixed(1) + 's'
    + ' · 检测点 ' + it.n_det
    + (it.old_frame !== null ? ' · 原记录 ' + it.old_frame + ' 帧' + (it.old_usable ? '' : '（不可用）') : ' · 原记录：无');
  document.getElementById('hint').textContent =
    view === 'focus'
      ? '聚焦片段：' + it.win_label + '（原生 25fps）。绿粗框＝该轨迹，灰细框＝同场其他人，右上角是目标放大。'
      : '全轨迹扫描：整条轨迹 5fps 抽样，时长与真实一致；先扫全程再切回聚焦片段定位。';
  const notes = [];
  if (!it.old_usable) notes.push('原记录不可用（末帧/越界/无法解析），请扫完整条轨迹再记录。');
  if (it.coverage < 0.4) notes.push('注意：这条轨迹检测点很稀疏（窗口覆盖率 ' + Math.round(it.coverage*100) + '%），可能本来就判不出换人点——这种情况用「无法判定」。');
  document.getElementById('warn').textContent = notes.join(' ');

  const src = encodeURI(clipURL());
  if (curSrc !== src) {
    curSrc = src;
    v.onerror = () => {
      document.getElementById('warn').textContent =
        '片段还没生成：' + decodeURI(src).replace(/^.*\//, '') + '（先跑 tools/render_switch_clips.py）';
    };
    v.src = src;
    v.onloadedmetadata = () => {
      const base = startF() / FPS;
      const target = (seekReal !== undefined) ? seekReal : (it.old_usable ? it.old_frame / FPS : base + 0.5);
      v.currentTime = Math.max(0, Math.min(v.duration - 0.05, target - base));
    };
  } else if (seekReal !== undefined) {
    const base = startF() / FPS;
    v.currentTime = Math.max(0, Math.min(v.duration - 0.05, seekReal - base));
  }
  document.getElementById('btnFocus').classList.toggle('on', view === 'focus');
  document.getElementById('btnScan').classList.toggle('on', view === 'scan');
  document.querySelectorAll('#list li').forEach((el, i) => {
    const r = store[ITEMS[i].track_id];
    el.classList.toggle('on', i === idx);
    el.classList.toggle('done', !!r && !r.undecidable);
    el.classList.toggle('und', !!r && !!r.undecidable);
    const tag = el.querySelector('.tag');
    if (tag) tag.textContent = ITEMS[i].cam + ' · ' + (r ? (r.undecidable ? '无法判定' : ('已记 ' + r.switch_frame + ' 帧'))
      : (ITEMS[i].old_frame === null ? '无原记录' : '原 ' + ITEMS[i].old_frame + ' 帧'));
  });
  document.getElementById('prog').textContent = Object.keys(store).length + ' / ' + ITEMS.length;
}

v.ontimeupdate = () => {
  const it = item(), f = realFrame();
  document.getElementById('cur').textContent =
    '帧 ' + f + '（' + (f / FPS).toFixed(2) + 's）　轨迹区间 ' + it.t0.toFixed(1) + '–' + it.t1.toFixed(1) + 's';
  const t0 = it.t0 - 1, t1 = it.t1 + 1;
  document.getElementById('bar').style.width =
    Math.max(0, Math.min(100, 100 * (f / FPS - t0) / Math.max(1, t1 - t0))) + '%';
};

function setView(nv){
  if (nv === view) return;
  const rf = startF() / FPS + v.currentTime;
  view = nv;
  render(rf);
}
function jump(d){ v.pause(); v.currentTime = Math.max(0, v.currentTime + d); }
function mark(){
  const f = realFrame();
  store[item().track_id] = { switch_s: +(f / FPS).toFixed(3), switch_frame: f, view: view };
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}
function markBad(){
  store[item().track_id] = { undecidable: true, switch_s: '', switch_frame: '', view: view };
  localStorage.setItem(KEY, JSON.stringify(store));
  render();
}
function go(i){ idx = Math.max(0, Math.min(ITEMS.length - 1, i)); render(); }
function exportCSV(){
  const rows = [['track_id','seg','cam','switch_s','switch_frame','old_switch_frame','note','view']];
  ITEMS.forEach(it => {
    const r = store[it.track_id];
    rows.push([it.track_id, it.seg, it.cam,
      r && !r.undecidable ? r.switch_s : '', r && !r.undecidable ? r.switch_frame : '',
      it.old_frame === null ? '' : it.old_frame,
      r ? (r.undecidable ? '无法判定' : '') : '未标注',
      r ? r.view : '']);
  });
  const csv = rows.map(r => r.map(x => /[",\n]/.test(String(x)) ? '"' + String(x).replace(/"/g,'""') + '"' : x).join(',')).join('\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,\ufeff' + encodeURIComponent(csv);
  a.download = 'a3_switch_review_FILLED.csv';
  a.click();
}
function buildList(){
  const ol = document.getElementById('list');
  ol.innerHTML = '';
  ITEMS.forEach((it, i) => {
    const li = document.createElement('li');
    li.innerHTML = '<span>' + it.track_id + '</span><span class="tag"></span>';
    li.onclick = () => go(i);
    ol.appendChild(li);
  });
}
document.addEventListener('keydown', e => {
  if (e.code === 'Space'){ e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.code === 'ArrowLeft'){ e.preventDefault(); jump(-0.5); }
  else if (e.code === 'ArrowRight'){ e.preventDefault(); jump(0.5); }
  else if (e.code === 'Enter'){ e.preventDefault(); mark(); }
  else if (e.code === 'KeyX'){ e.preventDefault(); markBad(); }
  else if (e.code === 'Tab'){ e.preventDefault(); go(idx + (e.shiftKey ? -1 : 1)); }
});
buildList(); render();
</script></body></html>
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--clips-base', default='clips', help='页面里片段目录（相对页面）')
    args = ap.parse_args()

    items = []
    for it in load_items():
        items.append(dict(
            track_id=it['track_id'], file_base=it['file_base'], seg=it['seg'], cam=it['cam'],
            n_frames=it['n_frames'], n_det=it['n_det'], t0=it['t0'], t1=it['t1'],
            old_frame=it['old_frame'], old_usable=it['old_usable'],
            coverage=it['coverage'], win_mode=it['win_mode'], win_label=it['win_label'],
            start_f={'focus': it['focus_start'], 'scan': it['scan_start']}))

    html = (TPL.replace('__ITEMS__', json.dumps(items, ensure_ascii=False))
               .replace('__CLIPS_BASE__', args.clips_base))
    open(args.out, 'w', encoding='utf-8').write(html)
    print('[out] %s（%d 条，%d KB）' % (args.out, len(items), os.path.getsize(args.out) // 1024))
    print('      原记录不可用 %d 条｜窗口覆盖 <40%% 的 %d 条'
          % (sum(1 for it in items if not it['old_usable']),
             sum(1 for it in items if it['coverage'] < 0.4)))


if __name__ == '__main__':
    main()
