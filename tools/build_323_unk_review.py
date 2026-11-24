"""生成 323 机位 31 条「远场未标查询」的补标页（含身份参考图）。

为什么单独做一页：这 31 条在 D4 里被标成 unk_reason=far，但 8 帧拼图其实能看清
（见 `crops/323_test_527/2107.jpg`）。补标后它们从「无真值」变成「可答/图库外」，
会改变可答上限，所以必须独立成页、独立导出，不能混进第一轮真值。

页面特性：
  - 左：8 帧拼图（点击放大）；右：25 个身份参考图（每个身份 2 张，来自图库已标成员）
  - 快捷键：U=看不清 / P=非人（影子·物件）/ N=图库外新人 / ←→ 翻页
  - 不显示任何系统判决（簇号、分数、接受与否）
  - 导出 CSV：query_id, person_id, note

用法：PY tools/build_323_unk_review.py
输出：deliverable/data-request-v2/gt/review_323_unk.html
"""
import base64
import collections
import csv
import io
import json
import os

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
GT = os.path.join(DELIV, "gt")
OUT = os.path.join(GT, "review_323_unk.html")
SEG = "323_test_527"
MAX_W = 1100


def b64(path, max_w=MAX_W, quality=82):
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def det_frames(det_root, seg, tid):
    out = {}
    p = os.path.join(det_root, seg, "%s_detections.jsonl" % seg)
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        for t, b in zip(r["track_ids"], r["boxes"]):
            if int(t) == tid:
                out[r["frame"]] = b
    return out


def portrait(seg, tid, vcap, box, pad=0.12, w=150):
    import cv2
    ok, img = vcap.read()
    if not ok:
        return None
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * pad); x2 = min(img.shape[1], x2 + bw * pad)
    y1 = max(0, y1 - bh * pad); y2 = min(img.shape[0], y2 + bh * pad)
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return None
    im = Image.fromarray(crop[:, :, ::-1])
    im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def is_nonperson(p):
    p = (p or "").strip().lower()
    return p == "no-human" or p.startswith("shadow")


def main():
    q = [r for r in rd(os.path.join(GT, "gt_query_gt_FILLED.csv")) if r["seg"] == SEG]
    items = [r for r in q if not (r["person_id"] or "").strip()]
    items.sort(key=lambda r: -int(r["n_frames"]))
    print("%s 未标 %d 条" % (SEG, len(items)))

    gal = rd(os.path.join(GT, "gt_gallery_members_FILLED.csv"))
    by_id = collections.defaultdict(list)
    for r in gal:
        pid = (r["person_id"] or "").strip()
        if not pid or pid.upper() == "UNK" or is_nonperson(pid):
            continue
        p = os.path.join(GT, r["prompt_jpg"])
        if os.path.exists(p):
            by_id[pid].append((int(r["n_frames"] or 0), r["box_type"], p))
    # 每个身份生成一张干净半身像：取 body 型、帧数最多的那条图库轨迹的中间帧
    import cv2
    det_root = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
    seg_dir = os.path.join(ROOT, "media-resource/1012/segments")
    caps = {}
    refs = {}
    for pid, v in by_id.items():
        v.sort(key=lambda x: (x[1] != "body", -x[0]))
        got = None
        for _, _, path in v[:4]:
            rel = os.path.relpath(path, GT).replace("\\", "/")
            parts = rel.split("/")
            if len(parts) < 3:
                continue
            seg, tid = parts[1], int(os.path.splitext(parts[2])[0])
            frames = det_frames(det_root, seg, tid)
            if not frames:
                continue
            mid = sorted(frames)[len(frames) // 2]
            if seg not in caps:
                cap = cv2.VideoCapture(os.path.join(seg_dir, seg + ".mp4"))
                caps[seg] = cap
            cap = caps[seg]
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            img = portrait(seg, tid, cap, frames[mid])
            if img:
                got = img
                break
        refs[pid] = [got] if got else [b64(v[0][2], 240, 78)]
    for c in caps.values():
        c.release()
    ids = sorted(refs, key=lambda s: (s.startswith("female"), int("".join(c for c in s if c.isdigit()) or 0)))
    print("参考身份 %d 个：%s" % (len(ids), ids))

    data = []
    for r in items:
        p = os.path.join(GT, r["prompt_jpg"])
        data.append(dict(q=r["query_id"], n=int(r["n_frames"]), ev=r["evaluable"],
                         img=b64(p), old=(r["unk_reason"] or "")))

    html = TPL.replace("__IDS__", json.dumps(ids, ensure_ascii=False)) \
              .replace("__REFS__", json.dumps(refs, ensure_ascii=False)) \
              .replace("__DATA__", json.dumps(data, ensure_ascii=False))
    open(OUT, "w", encoding="utf-8").write(html)
    print("→ %s（%.1f MB）" % (OUT, os.path.getsize(OUT) / 1048576))


TPL = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>323 远场补标</title>
<style>
:root{--bg:#14161a;--panel:#1c1f26;--line:#2c313b;--fg:#e8eaed;--dim:#9aa0aa;--acc:#4c9aff;--ok:#3ddc84;--warn:#ffb020}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,"Segoe UI",sans-serif}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
header b{font-size:15px}
.bar{flex:1;height:6px;background:#242832;border-radius:3px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok);width:0}
button{background:#262b34;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer}
button:hover{border-color:var(--acc)}
button.pri{background:var(--acc);border-color:var(--acc);color:#08121f;font-weight:600}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:16px;padding:16px}
img.big{width:100%;border-radius:8px;border:1px solid var(--line);cursor:zoom-in;background:#000}
.meta{color:var(--dim);font-size:13px;margin:8px 0}
aside{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;max-height:52vh;overflow:auto}
.pick{display:flex;gap:8px;align-items:center;padding:6px;border-radius:8px;border:1px solid transparent;cursor:pointer}
.pick:hover{background:#262b34;border-color:var(--acc)}
.pick img{width:56px;height:84px;object-fit:cover;border-radius:4px;background:#000}
.pick span{font-size:13px}
.pick.on{background:#1f3a5f;border-color:var(--acc)}
.sp{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.tag{font-size:12px;color:var(--dim)}
.zoom{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:20}
.zoom img{max-width:96vw;max-height:96vh}
.hint{color:var(--dim);font-size:12px}
</style></head><body>
<header>
  <b>323 远场补标</b>
  <span class="hint">U 看不清 · P 非人 · N 图库外新人 · ← → 翻页 · Backspace 撤销上一条</span>
  <div class="bar"><i id="pb"></i></div>
  <span id="cnt" class="hint"></span>
  <button id="exp" class="pri">导出 CSV</button>
</header>
<main>
  <section>
    <img id="big" class="big">
    <div class="meta" id="meta"></div>
  </section>
  <aside>
    <div class="tag">点选身份（右侧是该身份的图库参考图）</div>
    <div class="grid" id="grid"></div>
    <div class="sp">
      <button data-v="UNK">U 看不清</button>
      <button data-v="no-human">P 非人/影子</button>
      <button data-v="__NEW__">N 图库外新人</button>
      <button id="prev">← 上一条</button>
      <button id="next">下一条 →</button>
    </div>
  </aside>
</main>
<div class="zoom" id="zoom"><img id="zi"></div>
<script>
const IDS = __IDS__, REFS = __REFS__, DATA = __DATA__;
const NS = 'rev323unk_';
let marks = JSON.parse(localStorage.getItem(NS+'marks')||'{}');
let i = 0, cur = null;
const $ = s => document.querySelector(s);
function render(){
  const it = DATA[i];
  $('#big').src = it.img;
  $('#meta').textContent = `${it.q} · n_frames=${it.n} · ${it.ev==='1'?'可评估':'不可评估'} · 原标注=${it.old}`;
  const g = $('#grid'); g.innerHTML='';
  for(const id of IDS){
    const d = document.createElement('div');
    d.className='pick'+(marks[it.q]===id?' on':'');
    d.innerHTML = `<img src="${(REFS[id]||[])[0]||''}"><span>${id}</span>`;
    d.onclick = ()=>set(id);
    g.appendChild(d);
  }
  $('#cnt').textContent = `${i+1}/${DATA.length} · 已标 ${Object.keys(marks).length}`;
  $('#pb').style.width = (100*Object.keys(marks).length/DATA.length)+'%';
}
function set(v){
  marks[DATA[i].q]=v;
  localStorage.setItem(NS+'marks', JSON.stringify(marks));
  if(i<DATA.length-1){i++;render();}else{render();}
}
$('#prev').onclick=()=>{if(i>0){i--;render();}};
$('#next').onclick=()=>{if(i<DATA.length-1){i++;render();}};
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>set(b.dataset.v));
$('#big').onclick=()=>{$('#zi').src=$('#big').src;$('#zoom').style.display='flex';};
$('#zoom').onclick=()=>$('#zoom').style.display='none';
// 按住键会重复触发 keydown；靠 e.repeat + 250ms 节流挡住连标（曾导致一次按住标完整批）
let lastKey = 0;
addEventListener('keydown',e=>{
  if(e.repeat) return;
  const now = Date.now(); if(now-lastKey < 250) return; lastKey = now;
  if(e.key==='ArrowLeft'){$('#prev').click();}
  else if(e.key==='ArrowRight'){$('#next').click();}
  else if(e.key==='Backspace'){ if(i>0){i--; delete marks[DATA[i].q]; localStorage.setItem(NS+'marks',JSON.stringify(marks)); render();} e.preventDefault(); }
  else if(e.key.toLowerCase()==='u'){set('UNK');}
  else if(e.key.toLowerCase()==='p'){set('no-human');}
  else if(e.key.toLowerCase()==='n'){set('__NEW__');}
});
$('#exp').onclick=()=>{
  const vals=DATA.map(it=>marks[it.q]||'').filter(Boolean);
  const c={}; vals.forEach(v=>c[v]=(c[v]||0)+1);
  const top=Object.entries(c).sort((a,b)=>b[1]-a[1])[0];
  const miss=DATA.length-vals.length;
  if(top && top[1]/vals.length>=0.9 && vals.length>=10){
    if(!confirm(`注意：已标的 ${vals.length} 条里有 ${top[1]} 条都是「${top[0]}」（≥90%）。\n如果是按住键连标出来的，请点取消，用 Backspace 退回去重标。\n仍要导出吗？`)) return;
  }
  if(miss>0 && !confirm(`还有 ${miss} 条未标，导出后这些为空。继续？`)) return;
  const rows=[['query_id','person_id','note']];
  for(const it of DATA) rows.push([it.q, marks[it.q]||'', marks[it.q]==='__NEW__'?'gallery_out':(marks[it.q]?'':'unmarked')]);
  const csv='\ufeff'+rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(',')).join('\n');
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
  a.download='rev323_unk_FILLED.csv'; a.click();
};
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
