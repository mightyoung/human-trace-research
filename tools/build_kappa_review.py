#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 D5 分歧定案页：3 条真实分歧 + 各自所在簇的全部成员。

只做一件事：让标注人对着图确认「这条查询是不是簇主那个人」。
"""
import base64
import collections
import csv
import io
import json
import os

from PIL import Image

GT = "deliverable/data-request-v2/gt"
RES = "deliverable/data-request-v2/results"
LED = "deliverable/data-request-v2/ledger"
OUT = os.path.join(GT, "review_kappa.html")

CASES = ["305_test_544#440", "305_test_544#1153", "323_test_527#1669"]


def load(p, key):
    with open(p, encoding="utf-8-sig") as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def b64(jpg_rel, maxw=920, q=60):
    p = os.path.join(GT, jpg_rel)
    if not os.path.exists(p):
        return ""
    im = Image.open(p).convert("RGB")
    im.thumbnail((maxw, 20000), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=q, method=4)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    v = load(os.path.join(RES, "verdicts", "A_baseline.csv"), "query_id")
    mem = list(csv.DictReader(open(os.path.join(RES, "members", "A_baseline.csv"),
                                   encoding="utf-8-sig")))
    geo = {int(r["cluster_idx"]): r
           for r in csv.DictReader(open(os.path.join(LED, "clusters_geom.csv"),
                                        encoding="utf-8-sig"))
           if r["config"] == "baseline"}
    lab2gid = {int(g["cluster_lab"]): int(g["cluster_idx"]) for g in geo.values()}
    loc2gid = {int(r["cluster_local"]): lab2gid[int(r["cluster_lab"])] for r in mem}

    d3 = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(GT, "gt_gallery_members_FILLED.csv"),
                                 encoding="utf-8-sig")):
        d3[r["track_id"]].append(r)
    d4 = load(os.path.join(GT, "gt_query_gt_FILLED.csv"), "query_id")
    d5 = load(os.path.join(GT, "gt_annotator2_FILLED.csv"), "item_id")

    items, imgs = [], {}
    for k in CASES:
        loc = int(v[k]["cluster_local"])
        gid = loc2gid[loc]
        g = geo[gid]
        a1, a2 = d4[k], d5[k]
        jq = a1["prompt_jpg"]
        imgs[jq] = b64(jq)
        sibs = []
        for m in sorted({r["track_id"] for r in mem if int(r["cluster_local"]) == loc}):
            r = d3.get(m, [{}])
            r = r[0] if r else {}
            jp = f"crops/{m.split('#')[0]}/{m.split('#')[1]}.jpg"
            if jp in imgs:
                uri = imgs[jp]
            else:
                uri = b64(jp, maxw=460, q=52)
                imgs[jp] = uri
            sibs.append({"k": m, "pid": r.get("person_id", "(未标)"),
                         "box": r.get("box_type", ""), "jpg": jp,
                         "is_self": False})
        items.append({
            "k": k, "cam": a1["cam"], "box": a1["box_type"],
            "pid1": a1["person_id"].strip() or "UNK",
            "pid2": a2["person_id"].strip() or "UNK",
            "c": v[k]["c_pid"], "n_lab": v[k]["cluster_n_lab"],
            "n_ids": v[k]["cluster_n_ids"], "share": v[k]["top_share"],
            "dmax": g["d_max"], "cams": g["cams"], "verdict_old": g["purity_verdict"],
            "jpg": jq, "sibs": sibs,
        })

    data = {"items": items, "imgs": imgs}
    html = TPL.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    open(OUT, "w", encoding="utf-8").write(html)
    print(f"[out] {OUT}")
    print(f"  {len(items)} 条分歧｜图片 {len(imgs)} 张｜{os.path.getsize(OUT)/1024:.0f} KB")


TPL = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>D5 分歧定案（3 条）</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:14px/1.6 system-ui,"Segoe UI","Microsoft YaHei",sans-serif;
     background:#f5f6f8;color:#1a1d22}
header{background:#1f2430;color:#fff;padding:12px 18px;position:sticky;top:0;z-index:9}
header h1{margin:0;font-size:16px;font-weight:600}
header .sub{font-size:12px;color:#9aa4b5;margin-top:3px}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:16px;padding:16px;align-items:start}
.panel{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.07)}
#qimg{width:100%;border-radius:8px;display:block;background:#0f1218}
.meta{font-size:13px;color:#4b5563;margin:8px 0}
.meta b{color:#111827}
.tag{display:inline-block;padding:1px 7px;border-radius:5px;font-size:12px;
     background:#e5e7eb;color:#374151;margin-right:5px}
.tag.warn{background:#fee2e2;color:#991b1b}
.tag.ok{background:#dcfce7;color:#166534}
h2{font-size:14px;margin:0 0 10px;color:#374151;font-weight:600}
.idgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
.card{border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;background:#fafbfc}
.card img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#111}
.card .b{padding:4px 6px;font-size:11.5px;line-height:1.35}
.card .b b{display:block;color:#111827;font-size:12.5px}
.card .b i{color:#6b7280;font-style:normal}
.btns{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid #d1d5db;
       background:#fff;cursor:pointer}
button:hover{background:#f3f4f6}
button.p1{border-color:#2563eb;color:#1d4ed8;font-weight:600}
button.p2{border-color:#b45309;color:#92400e;font-weight:600}
button.p3{border-color:#6b7280;color:#374151}
.prog{font-size:13px;color:#6b7280;margin-top:6px}
.done{background:#ecfdf5;border-color:#10b981}
</style></head><body>
<header>
  <h1>D5 分歧定案 · 3 条</h1>
  <div class="sub">只判断一件事：这条查询，是不是它所在簇的那几个人之一？右侧是簇内全部成员与第一轮的标注。</div>
</header>
<main>
  <div class="panel">
    <div class="meta" id="meta"></div>
    <img id="qimg" alt="">
    <div class="btns">
      <button class="p1" onclick="pick(1)">1 · 第一人（<span id="l1"></span>）对</button>
      <button class="p2" onclick="pick(2)">2 · 第二人（<span id="l2"></span>）对</button>
      <button class="p3" onclick="pick(3)">3 · 两个都不对</button>
      <button onclick="pick(0)">跳过</button>
    </div>
    <div class="prog" id="prog"></div>
  </div>
  <div class="panel">
    <h2>该簇全部成员（标注来自第一轮）</h2>
    <div class="idgrid" id="sibs"></div>
  </div>
</main>
<script>
const DATA = __DATA__;
const IMG = DATA.imgs, ITEMS = DATA.items;
const KEY = 'd5kappa_review_v1';
let store = JSON.parse(localStorage.getItem(KEY) || '{}');
let idx = 0;

function cur(){ return ITEMS[idx]; }
function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

function render(){
  const it = cur();
  document.getElementById('meta').innerHTML =
    `<span class="tag">${esc(it.cam)}</span><span class="tag">${esc(it.box)}</span>` +
    `<span class="tag ${it.n_ids >= 2 ? 'warn' : 'ok'}">簇内 ${it.n_lab} 已标 / ${it.n_ids} 身份</span>` +
    `<span class="tag">d_max ${it.dmax}</span>` +
    `<div style="margin-top:6px">簇主 <b>${esc(it.c)}</b>　主占比 ${esc(it.share)}　簇跨机位 ${esc(it.cams)}　旧判定 ${esc(it.verdict_old)}</div>` +
    `<div style="margin-top:4px">第一轮 <b>${esc(it.pid1)}</b>　第二轮 <b>${esc(it.pid2)}</b></div>`;
  document.getElementById('qimg').src = IMG[it.jpg] || '';
  document.getElementById('l1').textContent = it.pid1;
  document.getElementById('l2').textContent = it.pid2;
  document.getElementById('sibs').innerHTML = it.sibs.map(s =>
    `<div class="card"><img loading="lazy" src="${IMG[s.jpg] || ''}" alt="">` +
    `<div class="b"><b>${esc(s.pid)}</b><i>${esc(s.k)} · ${esc(s.box)}</i></div></div>`).join('');
  renderProg();
}

function renderProg(){
  const n = Object.keys(store).length;
  document.getElementById('prog').textContent =
    `已定 ${n} / ${ITEMS.length}　当前第 ${idx + 1} 条　${store[cur().k] ? '（本条已定：' + ['跳过','第一人对','第二人对','都不对'][store[cur().k]] + '）' : ''}`;
}

function pick(v){
  const k = cur().k;
  if (v === 0) delete store[k]; else store[k] = v;
  localStorage.setItem(KEY, JSON.stringify(store));
  if (v !== 0 && idx < ITEMS.length - 1) idx++;
  render();
}

function exportCSV(){
  const head = 'item_id,cam,box_type,pid_annotator1,pid_annotator2,cluster_master,n_labelled,n_ids,top_share,d_max,decision';
  const name = ['', 'annotator1', 'annotator2', 'neither'];
  const lines = [head];
  ITEMS.forEach(it => {
    const d = store[it.k] ? name[store[it.k]] : 'skipped';
    lines.push([it.k, it.cam, it.box, it.pid1, it.pid2, it.c, it.n_lab, it.n_ids, it.share, it.dmax, d]
      .map(x => '"' + String(x).replace(/"/g, '""') + '"').join(','));
  });
  const csv = '\ufeff' + lines.join('\r\n');
  const box = window.prompt('全选复制（Ctrl+A / Ctrl+C），或点确定尝试下载：', csv);
  if (box !== null) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
    a.download = 'review_kappa_FILLED.csv';
    a.click();
  }
}

document.addEventListener('keydown', e => {
  if (e.key === '1') pick(1);
  else if (e.key === '2') pick(2);
  else if (e.key === '3') pick(3);
  else if (e.key === 'ArrowRight' && idx < ITEMS.length - 1) { idx++; render(); }
  else if (e.key === 'ArrowLeft' && idx > 0) { idx--; render(); }
  else if (e.key === 's') exportCSV();
});
render();
const btn = document.createElement('button');
btn.textContent = '导出定案 CSV（或按 s）';
btn.style.cssText = 'position:fixed;right:18px;bottom:18px;background:#2563eb;color:#fff;border:0;padding:10px 16px;border-radius:24px;box-shadow:0 3px 10px rgba(37,99,235,.4);z-index:99';
btn.onclick = exportCSV;
document.body.appendChild(btn);
</script></body></html>"""


if __name__ == "__main__":
    main()
