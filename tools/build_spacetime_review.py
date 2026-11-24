"""把时空矛盾（非共视机位同时在场）做成离线复核队列。

输入  tmp/full/spacetime_audit.json     27 处违规（15 个身份组）
输出  deliverable/spacetime_review/spacetime_review_offline.html
      deliverable/spacetime_review/items.json

为什么这样设计（每条都对应本项目踩过的坑）：
  - ★ 只判「长相/衣着/体型是否同一人」，不用重叠时长当答案；重叠时长只作为元数据展示。
    理由：305/318/320 是同一开放空间的分区，人走过交界处会被两侧同时拍十几秒
    （D1 投影相关 0.00–0.21），短重叠本身不能证明是两个人。
  - ★ 每个案例取「同一 OSD 绝对时刻」的三帧（窗口 8% / 50% / 92%），A/B 两行并排。
    同一人在同一瞬间长相不会变；如果两行是两个人，三个时刻应该都看得出差异。
  - ★★★ 必须用原生 1440p：media-resource/multi-cemra/long-35min/{cam}_long35.mp4
    webapp 下的 media 都是 960×540 代理，看不出人脸/衣着细节。
  - ★ 必须画目标框，且留 30% 边距，否则框贴边等于没画。
  - ★ 同一进程不要并发打开同一个 1440p h264 → 每机位打开一次、按帧号升序 seek。
"""
import base64
import json
import os
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "media-resource/multi-cemra/long-35min"
TRACKS = ROOT / "webapp/backend/data/tracks"
AUDIT = ROOT / "tmp/full/spacetime_audit.json"
GRAPH = ROOT / "tmp/full/long35_crosscam_attr8.json"
GT = ROOT / "deliverable/gt_long35/gt_long35.json"
OUT = ROOT / "deliverable/spacetime_review"
CACHE = OUT / ".cache"

OSD = {"318": 0.0, "320": 1.0, "323": 2.0, "316": 16.0, "305": 25.0, "343": 121.0}
FPS = 25.0
DETAIL_W, DETAIL_Q = 260, 72      # 单帧细节裁图：手机 3 列约 105px 显示，点开看原图
SCENE_W, SCENE_Q = 340, 70
N_T = 3                            # 每个案例取三个时刻
RAW_ON = bool(os.environ.get("SP_RAW"))   # SP_RAW=1：把裁图另存 PNG 便于人眼核对


def load_tracks(cam):
    doc = json.loads((TRACKS / f"{cam}.json").read_text(encoding="utf-8"))
    return doc["tracks"]


def nearest(pts, f):
    """取离 f 最近的采样点（stride=5）。"""
    return min(pts, key=lambda p: abs(p[0] - f))


def box_h_at(pts, f):
    p = nearest(pts, f)
    return max(2.0, p[4] - p[2]), abs(p[0] - f)


def pick_times(lo, hi, tra, osda, trb, osdb, n=N_T):
    """在重叠窗口里挑「两边都看得最清楚」的 n 个时刻（两侧必须是同一个绝对时刻）。

    办公区座位视角，人是坐着的：均匀取样经常取到被显示器挡住/侧身重叠的瞬间。
    打分 = min(A 框高, B 框高)（较小的一侧是可见性瓶颈），再贪心保证时间分散。
    """
    w = hi - lo
    nc = 40 if w > 6 else max(6, int(w))
    cands = []
    for k in range(nc):
        t = lo + w * (k + 0.5) / nc
        ha, da = box_h_at(tra, int(round((t - osda) * FPS)))
        hb, db = box_h_at(trb, int(round((t - osdb) * FPS)))
        pen = (da > 6) + (db > 6)          # 采样点离请求帧太远 = 该时刻没有观测
        cands.append((min(ha, hb) - 400 * pen, t))
    cands.sort(reverse=True)
    picked, gap = [], max(0.6, w * 0.12)
    for _sc, t in cands:
        if all(abs(t - p) >= gap for p in picked):
            picked.append(t)
        if len(picked) == n:
            break
    while len(picked) < n:
        picked.append((picked[-1] if picked else lo) + 0.4)
    return sorted(picked[:n])


def detail_crop(frame, box, margin=0.30):
    """人物细节：框外留 30% 边距，保证框在画面内可见。"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
    bw, bh = max(2, x2 - x1), max(2, y2 - y1)
    mx, my = bw * margin, bh * margin * 0.5
    # 只裁到合理比例，避免做成细长条
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cw, ch = max(bw + 2 * mx, bh * 0.42), bh + 2 * my
    x0 = int(max(0, min(W - cw, cx - cw / 2)))
    y0 = int(max(0, min(H - ch, cy - ch / 2)))
    x3 = int(min(W, max(x0 + cw, 2)))
    y3 = int(min(H, max(y0 + ch, 2)))
    crop = frame[y0:y3, x0:x3].copy()
    cv2.rectangle(crop, (int(x1 - x0), int(y1 - y0)),
                  (int(x2 - x0), int(y2 - y0)), (0, 235, 0), 2)
    # 办公区座位视角的人很小（305 中位 164 px 高，318 只有 65 px）→ 轻微放大 + 锐化，
    # 否则手机上根本看不清衣着。只放大不放飞：上限 3×，目标高 230 px。
    h, w = crop.shape[:2]
    if h < 230:
        sc = min(3.0, 230 / h)
        crop = cv2.resize(crop, (max(2, int(w * sc)), max(2, int(h * sc))),
                          interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(crop, (0, 0), 1.2)
        crop = cv2.addWeighted(crop, 1.35, blur, -0.35, 0)
    return crop


def scene_crop(frame, box, pad=1.9, out_w=SCENE_W):
    """16:9 环境窗口（pad 倍数在细长框上会退化成竖条，所以先按框高等比算宽度）。"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box[1], box[2], box[3], box[4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = max((y2 - y1) * pad, 140)
    w = h * 16 / 9
    x0 = int(max(0, min(W - w, cx - w / 2)))
    y0 = int(max(0, min(H - h, cy - h / 2)))
    crop = frame[y0:int(y0 + h), x0:int(x0 + w)].copy()
    cv2.rectangle(crop, (int(x1 - x0), int(y1 - y0)),
                  (int(x2 - x0), int(y2 - y0)), (0, 235, 0), 3)
    sc = out_w / crop.shape[1]
    return cv2.resize(crop, (out_w, max(2, int(crop.shape[0] * sc))))


def enc(img, quality):
    ok, buf = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise RuntimeError("webp 编码失败")
    return "data:image/webp;base64," + base64.b64encode(buf).decode("ascii")


def grab(vc, fidx, retries=8):
    """h264 偶发解码失败 → 先后退再重试（本项目实测有效）。"""
    for k in range(retries):
        f = max(0, int(fidx) - k * 5)
        vc.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = vc.read()
        if ok and img is not None:
            return img, f
    return None, None


def main():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    viol = audit["violations"]
    if len(sys.argv) > 1:                      # 只跑前 N 条，便于先验通链路
        viol = viol[:int(sys.argv[1])]
    groups = json.loads(GRAPH.read_text(encoding="utf-8"))["groups"]
    # ★ v["group"] 是「跨机位组子集」的下标，不是全 443 组的下标；
    #   早期版本直接拿它索引 groups，导致页面上「共合并 N 段 / 同组机位」显示错组。
    multi = [g for g in groups if len({c for c, _ in g}) > 1]
    gt = json.loads(GT.read_text(encoding="utf-8"))["items"]

    gt_cover = {}
    for it in gt:
        for side in ("a", "b"):
            gt_cover.setdefault(it[side], set()).add(it["label"])

    # ---- 1. 生成任务清单：每个案例三个同一 OSD 时刻 ----
    items, jobs = [], []
    for i, v in enumerate(viol):
        ca, ta = v["a"].split(":")
        cb, tb = v["b"].split(":")
        lo = max(v["a_span"][0], v["b_span"][0])
        hi = min(v["a_span"][1], v["b_span"][1])
        if hi <= lo:
            print(f"[skip] {v['a']} vs {v['b']} 重叠窗口为空")
            continue
        iid = f"sp{i + 1:02d}"
        trs = {}
        for cam, tid in ((ca, ta), (cb, tb)):
            trs[cam] = load_tracks(cam)[str(tid)]
        times = pick_times(lo, hi, trs[ca], OSD[ca], trs[cb], OSD[cb])
        sides = []
        for cam, tid in ((ca, ta), (cb, tb)):
            tr = trs[cam]
            f0, f1 = tr[0][0], tr[-1][0]
            cells = []
            for k, t in enumerate(times):
                want = int(round((t - OSD[cam]) * FPS))
                want = max(int(f0), min(int(f1), want))
                jobs.append({"iid": iid, "cam": cam, "tid": tid, "slot": f"{k}",
                             "frame": want, "pts": tr, "kind": "detail"})
                cells.append({"t": round(t, 1), "frame": want})
            # 环境帧取「看得最清楚」的那个时刻
            want = max(int(f0), min(int(f1), int(round((times[0] - OSD[cam]) * FPS))))
            jobs.append({"iid": iid, "cam": cam, "tid": tid, "slot": "s",
                         "frame": want, "pts": tr, "kind": "scene"})
            sides.append({"cam": cam, "tid": tid, "f0": f0, "f1": f1,
                          "span": [round(OSD[cam] + f0 / FPS, 1),
                                   round(OSD[cam] + f1 / FPS, 1)],
                          "cells": cells,
                          "gt": sorted(gt_cover.get(f"{cam}:{tid}", []))})
        gmem = multi[int(v["group"])] if int(v["group"]) < len(multi) else []
        cams = sorted({c for c, _ in gmem})
        items.append({
            "id": iid, "group": v["group"], "pair": v["pair"],
            "ov": round(v["overlap_s"], 1), "a": v["a"], "b": v["b"],
            "times": [round(t, 1) for t in times],
            "sides": sides,
            "gsize": len(gmem), "gcams": cams,
        })

    # ---- 2. 按机位归并、帧号升序解码（避免反复 seek 与并发打开） ----
    jobs.sort(key=lambda j: (j["cam"], j["frame"]))
    caps, got, fails = {}, {}, []
    for j in jobs:
        cam = j["cam"]
        if cam not in caps:
            caps[cam] = cv2.VideoCapture(str(VID / f"{cam}_long35.mp4"))
        cap = caps[cam]
        if not cap.isOpened():
            fails.append(f"{cam} 打不开")
            continue
        ckey = CACHE / f"{j['iid']}_{j['cam']}{j['tid']}_{j['kind']}{j['slot']}_f{j['frame']}.webp"
        if ckey.exists():                       # 缓存命中：调样式/文案时不必重解码 216 帧
            got[(j["iid"], j["cam"], j["slot"])] = (
                "data:image/webp;base64," + base64.b64encode(ckey.read_bytes()).decode("ascii"))
            continue
        img, used = grab(cap, j["frame"])
        if img is None:
            fails.append(f"{j['iid']} {cam}:{j['tid']} f={j['frame']}")
            continue
        bx = nearest(j["pts"], used)
        crop = (detail_crop(img, bx) if j["kind"] == "detail"
                else scene_crop(img, bx))
        if RAW_ON:
            d = OUT / "raw"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{j['iid']}_{cam}{j['tid']}_{j['kind']}{j['slot']}.png"), crop)
        q = DETAIL_Q if j["kind"] == "detail" else SCENE_Q
        if j["kind"] == "detail" and crop.shape[1] > DETAIL_W:
            h = crop.shape[0]
            crop = cv2.resize(crop, (DETAIL_W, max(2, int(h * DETAIL_W / crop.shape[1]))))
        got[(j["iid"], cam, j["slot"])] = enc(crop, q)
        if not RAW_ON:
            ckey.parent.mkdir(parents=True, exist_ok=True)
            ckey.write_bytes(base64.b64decode(got[(j["iid"], cam, j["slot"])].split(",", 1)[1]))
        print(f"[{len(got):3d}/{len(jobs)}] {j['iid']} {cam}:{j['tid']} "
              f"{j['kind']:6s} f={used} 实际t={OSD[cam]+used/FPS:7.1f}", flush=True)
    for c in caps.values():
        c.release()
    if fails:
        print("\n[失败] " + "; ".join(fails[:10]))

    # ---- 3. 组装 HTML ----
    for it in items:
        for sd in it["sides"]:
            sd["imgs"] = [got.get((it["id"], sd["cam"], str(k))) for k in range(N_T)]
            sd["scene"] = got.get((it["id"], sd["cam"], "s"))

    OUT.mkdir(parents=True, exist_ok=True)
    # items.json 只存元数据：图已经在 HTML 里 base64 内联，再抄一份会让这个文件也变 2 MB
    meta = [{k: v for k, v in it.items() if k != "sides"} |
            {"sides": [{kk: vv for kk, vv in sd.items() if kk not in ("imgs", "scene")}
                       for sd in it["sides"]]}
            for it in items]
    (OUT / "items.json").write_text(
        json.dumps({"items": meta, "osd": OSD, "fps": FPS},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(items, ensure_ascii=False))
    html = html.replace("__N__", str(len(items)))
    (OUT / "spacetime_review_offline.html").write_text(html, encoding="utf-8")
    mb = (OUT / "spacetime_review_offline.html").stat().st_size / 1e6
    print(f"\n{len(items)} 个案例 / {len(got)} 张图 → {OUT/'spacetime_review_offline.html'} ({mb:.2f} MB)")
    miss = sum(1 for it in items for sd in it["sides"]
               if any(x is None for x in sd["imgs"]) or sd["scene"] is None)
    if miss:
        print(f"[warn] {miss} 侧存在缺图（页面会显示占位）")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#f4f5f7">
<title>时空矛盾复核 __N__ 条</title>
<style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;background:#f4f5f7;color:#16181d;
  font:16px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
body{padding-bottom:158px}
header{position:sticky;top:0;z-index:20;background:rgba(244,245,247,.96);
  backdrop-filter:saturate(1.6) blur(10px);border-bottom:1px solid #e2e5ea;padding:9px 13px 8px}
.hrow{display:flex;align-items:baseline;gap:9px}
h1{margin:0;font-size:16px;font-weight:650}
.cnt{margin-left:auto;font-size:13.5px;color:#6b7280;font-variant-numeric:tabular-nums}
.cnt b{color:#16181d;font-size:15px}
.bar{height:4px;background:#e2e5ea;border-radius:99px;margin-top:7px;overflow:hidden}
.bar i{display:block;height:100%;width:0;background:#2563eb;border-radius:99px;transition:width .25s}
.chips{display:flex;gap:6px;margin-top:8px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px}
.chips::-webkit-scrollbar{display:none}
.chips button{flex:0 0 auto;font:inherit;font-size:12.5px;padding:5px 11px;border-radius:99px;
  border:1px solid #d7dae0;background:#fff;color:#4b5563;white-space:nowrap}
.chips button[aria-pressed=true]{background:#16181d;border-color:#16181d;color:#fff;font-weight:600}
main{padding:10px 12px 0;max-width:880px;margin:0 auto}
.card{background:#fff;border:1px solid #e2e5ea;border-radius:12px;overflow:hidden;
  box-shadow:0 1px 2px rgba(16,24,40,.04)}
.top{display:flex;align-items:center;gap:7px;padding:9px 11px;border-bottom:1px solid #f0f1f4;flex-wrap:wrap}
.top .id{font-weight:700;font-size:15px;letter-spacing:.3px;font-variant-numeric:tabular-nums}
.tag{font-size:11.5px;padding:2px 8px;border-radius:99px;background:#eef2f7;color:#4b5563;white-space:nowrap}
.tag.blue{background:#e0edff;color:#1d4ed8}
.tag.warn{background:#fef3c7;color:#92400e}
.tag.gray{background:#f1f3f6;color:#6b7280}
.top .sp{margin-left:auto}
.top .verdict{font-size:12px;font-weight:600;padding:2px 9px;border-radius:99px}
.v-same{background:#dcfce7;color:#15803d}
.v-diff{background:#fee2e2;color:#b91c1c}
.v-unsure{background:#fef3c7;color:#92400e}
.grid{padding:8px;display:grid;gap:6px}
.row{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.cell{background:#111418;border-radius:8px;overflow:hidden;position:relative;cursor:zoom-in}
.cell img{display:block;width:100%;height:190px;object-fit:contain}
.cell .cap{position:absolute;left:0;right:0;bottom:0;font-size:10px;color:#fff;
  background:linear-gradient(transparent,rgba(0,0,0,.72));padding:9px 5px 2px;
  font-variant-numeric:tabular-nums;text-align:center}
.slabel{display:flex;align-items:center;gap:6px;margin:6px 0 2px;font-size:12.5px;color:#4b5563}
.slabel .cam{font-weight:700;color:#16181d;font-variant-numeric:tabular-nums}
.slabel .ln{flex:1;height:1px;background:#eef0f3}
.scenes{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:0 8px 8px}
.scenes figure{margin:0;background:#111418;border-radius:8px;overflow:hidden;position:relative}
.scenes img{display:block;width:100%;height:auto;cursor:zoom-in}
.scenes figcaption{position:absolute;left:0;bottom:0;font-size:10px;color:#fff;
  background:rgba(0,0,0,.55);padding:2px 6px;border-radius:0 6px 0 0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 10px;padding:0 11px 10px;font-size:12.5px}
.kv .k{color:#9aa1ab;white-space:nowrap}
.kv .v{color:#16181d;font-variant-numeric:tabular-nums;word-break:break-all}
.note{margin:0 11px 10px;padding:8px 10px;background:#f8fafc;border-left:3px solid #cbd5e1;
  border-radius:0 8px 8px 0;font-size:12.5px;line-height:1.55;color:#4b5563}
.note b{color:#16181d}
footer{position:fixed;left:0;right:0;bottom:0;z-index:30;background:rgba(255,255,255,.97);
  border-top:1px solid #e2e5ea;backdrop-filter:blur(10px);
  padding:8px 10px calc(9px + env(safe-area-inset-bottom))}
.nav{display:flex;align-items:center;gap:8px;margin:0 auto 7px;max-width:880px}
.nav button{font:inherit;font-size:13px;min-height:34px;border-radius:9px;border:1px solid #d7dae0;
  background:#fff;color:#4b5563;padding:0 12px}
.nav .pos{margin:0 auto;font-size:12.5px;color:#9aa1ab;font-variant-numeric:tabular-nums}
.btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;max-width:880px;margin:0 auto}
.btns button{font:inherit;font-size:15px;font-weight:600;min-height:48px;border-radius:11px;
  border:1.5px solid #d7dae0;background:#fff;color:#16181d;display:flex;align-items:center;
  justify-content:center;gap:6px}
.btns button .kb{font-size:11px;font-weight:500;color:#9aa1ab}
.btns button.same[aria-pressed=true]{background:#16a34a;border-color:#16a34a;color:#fff}
.btns button.diff[aria-pressed=true]{background:#dc2626;border-color:#dc2626;color:#fff}
.btns button.unsure[aria-pressed=true]{background:#f59e0b;border-color:#f59e0b;color:#fff}
.btns button[aria-pressed=true] .kb{color:rgba(255,255,255,.82)}
.btns.sub{margin-top:5px}
.btns.sub button{min-height:34px;font-size:12.5px;font-weight:500;background:#f6f7f9;
  border-color:#e6e8ec;color:#4b5563;border-radius:9px}
.btns.sub button[aria-pressed=true]{background:#7c3aed;border-color:#7c3aed;color:#fff;font-weight:600}
.v-boundary,.v-mixA,.v-mixB{background:#ede9fe;color:#6d28d9}
.ov{position:fixed;inset:0;z-index:60;background:rgba(10,12,16,.96);display:flex;
  align-items:flex-start;justify-content:center;overflow:auto}
.ov[hidden]{display:none}
.ov img{display:block;width:100%;max-width:900px;height:auto}
.ov .tools{position:fixed;top:8px;right:8px;display:flex;gap:7px;z-index:61}
.ov .tools button{font:inherit;font-size:12.5px;padding:6px 11px;border-radius:99px;
  border:1px solid rgba(255,255,255,.35);background:rgba(0,0,0,.55);color:#fff}
body.bright .ov img,body.bright .cell img,body.bright .scenes img{filter:brightness(1.35) contrast(1.08)}
.sheet{position:fixed;inset:0;z-index:70;background:rgba(16,24,40,.45);display:flex;align-items:flex-end}
.sheet[hidden]{display:none}
.panel{background:#fff;width:100%;border-radius:16px 16px 0 0;
  padding:14px 14px calc(14px + env(safe-area-inset-bottom));max-height:88vh;
  display:flex;flex-direction:column;gap:9px}
.panel h2{margin:0;font-size:15px}
.panel .sub{font-size:12.5px;color:#6b7280;margin:0}
.panel textarea{width:100%;min-height:110px;max-height:26vh;font:12px/1.45 ui-monospace,Menlo,monospace;
  border:1px solid #d7dae0;border-radius:10px;padding:9px;background:#fafbfc;color:#16181d;resize:none}
.panel .row{display:flex;gap:8px}
.panel .row button{flex:1;font:inherit;font-size:14.5px;min-height:44px;border-radius:10px;
  border:1px solid #d7dae0;background:#fff;color:#16181d}
.panel .row button.pri{background:#2563eb;border-color:#2563eb;color:#fff;font-weight:650}
.stats{font-size:12.5px;line-height:1.65;color:#374151}
#toast{position:fixed;left:50%;bottom:120px;transform:translate(-50%,14px);z-index:90;
  background:rgba(22,24,29,.94);color:#fff;font-size:13.5px;padding:9px 15px;border-radius:99px;
  opacity:0;transition:.2s;pointer-events:none;max-width:86vw;text-align:center}
#toast.on{opacity:1;transform:translate(-50%,0)}
@media(min-width:720px){.cell img{height:280px}}
</style>
</head>
<body>
<header>
  <div class="hrow">
    <h1>时空矛盾复核</h1>
    <span class="cnt"><b id="done">0</b> / __N__</span>
  </div>
  <div class="bar"><i id="fill"></i></div>
  <div class="chips" id="chips">
    <button data-f="TODO" aria-pressed="true">待判</button>
    <button data-f="ALL">全部</button>
    <button data-f="STRONG">强证据 ≥60s</button>
    <button data-f="MID">10–60s</button>
    <button data-f="same">已判同一人</button>
    <button data-f="diff">已判不同人</button>
  </div>
</header>

<main id="main"></main>

<div class="ov" id="ov" hidden>
  <div class="tools">
    <button id="ovB">提亮</button>
    <button id="ovC">关闭</button>
  </div>
  <img id="ovI" alt="" />
</div>

<footer>
  <div class="nav">
    <button id="btnPrev">← 上一条</button>
    <span class="pos" id="pos"></span>
    <button id="btnUndo">撤销 U</button>
    <button id="btnNext">下一条 →</button>
    <button id="btnOut">导出</button>
  </div>
  <div class="btns">
    <button class="same" data-v="same">同一人<span class="kb">1</span></button>
    <button class="diff" data-v="diff">不同人<span class="kb">2</span></button>
    <button class="unsure" data-v="unsure">看不清<span class="kb">3</span></button>
  </div>
  <div class="btns sub">
    <button data-v="boundary">疑似交界处走过<span class="kb">4</span></button>
    <button data-v="mixA">A 侧混人<span class="kb">5</span></button>
    <button data-v="mixB">B 侧混人<span class="kb">6</span></button>
  </div>
</footer>

<div class="sheet" id="sheet" hidden>
  <div class="panel">
    <h2>导出判读结果</h2>
    <p class="sub" id="outSub"></p>
    <textarea id="out" readonly></textarea>
    <div class="stats" id="stats"></div>
    <div class="row">
      <button id="btnCloseSheet">关闭</button>
      <button class="pri" id="btnCopy">复制 JSON</button>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
const ITEMS = /*__DATA__*/;
const KEY = 'hc_spacetime_review_v1';
const ZH = {same:'同一人', diff:'不同人', unsure:'看不清',
            boundary:'交界处走过', mixA:'A侧混人', mixB:'B侧混人'};
let S = {}, filter = 'TODO', curId = null;
try { S = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { S = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} }
function $(q) { return document.querySelector(q); }
function verdict(id) { const r = S[id]; return r ? r.v : null; }
function isJudge(v) { return v === 'same' || v === 'diff' || v === 'unsure'; }
function nDone() { return ITEMS.filter(function (it) { return isJudge(verdict(it.id)); }).length; }
function ovClass(ov) { return ov >= 60 ? 'warn' : ''; }

function wclock(t) {
  const base = 8 * 3600 + 55 * 60 + 20 + t;   // 318 首帧 08:55:20
  const h = Math.floor(base / 3600) % 24, m = Math.floor(base / 60) % 60, s = Math.round(base % 60);
  return [h, m, s].map(function (x) { return String(x).padStart(2, '0'); }).join(':');
}

function view() {
  if (filter === 'ALL') return ITEMS;
  if (filter === 'TODO') return ITEMS.filter(function (it) { return !isJudge(verdict(it.id)); });
  if (filter === 'STRONG') return ITEMS.filter(function (it) { return it.ov >= 60; });
  if (filter === 'MID') return ITEMS.filter(function (it) { return it.ov < 60; });
  return ITEMS.filter(function (it) { return verdict(it.id) === filter; });
}

function imgTag(src, cap, alt) {
  if (!src) return '<div class="cell"><div style="height:190px;display:flex;align-items:center;justify-content:center;color:#8b93a1;font-size:12px">缺图</div></div>';
  return '<div class="cell" data-src="' + src + '"><img src="' + src + '" alt="' + alt + '" loading="lazy" />' +
         (cap ? '<div class="cap">' + cap + '</div>' : '') + '</div>';
}

function render() {
  const v = view();
  const main = $('#main');
  if (!v.length) {
    main.innerHTML = '<div class="card"><div class="note">该筛选下没有条目。' +
      '<b>全部</b> 里换一个标签看看。</div></div>';
    $('#pos').textContent = '0 / 0';
    $('#btnPrev').disabled = $('#btnNext').disabled = true;
    paintBtns(null, null);            // ★ 否则清掉判读后按钮还亮着，看着像没清
    refreshBar(); return;
  }
  if (!v.some(function (x) { return x.id === curId; })) curId = v[0].id;
  const it = v.find(function (x) { return x.id === curId; }) || v[0];
  const vd = verdict(it.id);
  const flag = S[it.id] && S[it.id].flag;

  let h = '<div class="card"><div class="top">' +
    '<span class="id">' + it.id + '</span>' +
    '<span class="tag blue">' + it.pair + '</span>' +
    '<span class="tag ' + ovClass(it.ov) + '">同时在场 ' + it.ov.toFixed(0) + ' s</span>' +
    '<span class="tag gray">共合并 ' + it.gsize + ' 段</span>' +
    '<span class="sp"></span>' +
    (vd ? '<span class="verdict v-' + vd + '">' + ZH[vd] + '</span>' : '') +
    (flag ? '<span class="verdict v-' + flag + '">' + ZH[flag] + '</span>' : '') +
    '</div><div class="grid">';

  it.sides.forEach(function (sd, si) {
    h += '<div class="slabel"><span class="cam">' + (si ? 'B' : 'A') + ' · ' + sd.cam + ':' + sd.tid +
         '</span><span class="ln"></span><span>' + sd.span[0].toFixed(0) + '–' + sd.span[1].toFixed(0) + ' s</span></div>' +
         '<div class="row">';
    sd.cells.forEach(function (c, k) {
      h += imgTag(sd.imgs[k], wclock(c.t).slice(0, 5) + ' <span style="opacity:.7">f' + c.frame + '</span>',
                  sd.cam + ':' + sd.tid + ' t' + (k + 1));
    });
    h += '</div>';
  });
  h += '</div><div class="scenes">';
  it.sides.forEach(function (sd, si) {
    h += sd.scene
      ? '<figure data-src="' + sd.scene + '"><img src="' + sd.scene + '" alt="scene ' + sd.cam + '" loading="lazy" />' +
        '<figcaption>' + sd.cam + ':' + sd.tid + ' 环境</figcaption></figure>'
      : '';
  });
  h += '</div><div class="kv">' +
    '<span class="k">A 轨</span><span class="v">' + it.a + ' · ' + it.sides[0].f0 + '–' + it.sides[0].f1 + ' 帧' + gtText(it.sides[0]) + '</span>' +
    '<span class="k">B 轨</span><span class="v">' + it.b + ' · ' + it.sides[1].f0 + '–' + it.sides[1].f1 + ' 帧' + gtText(it.sides[1]) + '</span>' +
    '<span class="k">同组机位</span><span class="v">' + it.gcams.join(' / ') + '</span>' +
    '<span class="k">三个时刻</span><span class="v">' + it.times.map(function (t) { return wclock(t); }).join(' · ') + '</span>' +
    '</div>' +
    '<p class="note"><b>只判长相</b>：A、B 两行是<b>同一时刻</b>的三帧。看衣着、体型、发型、配饰，判断是不是同一个人。' +
    '两侧同时在场本身就是矛盾信号，但<b>不要</b>据此直接判「不同人」——305/318/320/323 是同一开放空间的相邻分区，' +
    '人走过交界处会被两侧同时拍到十几秒（该站不住时请点下面的「疑似交界处走过」）。</p>' +
    '</div>';

  main.innerHTML = h;
  main.querySelectorAll('[data-src]').forEach(function (el) {
    el.onclick = function () { openOv(el.dataset.src); };
  });
  $('#pos').textContent = (v.findIndex(function (x) { return x.id === curId; }) + 1) + ' / ' + v.length;
  $('#btnPrev').disabled = $('#btnNext').disabled = v.length < 2;
  paintBtns(vd, flag); refreshBar();
}

function gtText(sd) {
  if (!sd.gt || !sd.gt.length) return '';
  const has1 = sd.gt.indexOf(1) >= 0, has0 = sd.gt.indexOf(0) >= 0;
  return ' · 已有真值 ' + (has1 && has0 ? '同/异都出现过' : has1 ? '同人' : '异人');
}

function paintBtns(vd, flag) {
  document.querySelectorAll('.btns button').forEach(function (b) {
    const isSub = !!b.closest('.btns.sub');
    b.setAttribute('aria-pressed', String(isSub ? flag === b.dataset.v : vd === b.dataset.v));
  });
}
function refreshBar() {
  const n = nDone();
  $('#done').textContent = n;
  $('#fill').style.width = (n / ITEMS.length * 100) + '%';
}

let undoStack = [];
function judge(v) {
  if (!curId) return;
  const it = ITEMS.find(function (x) { return x.id === curId; });
  const prev = verdict(curId), prevFlag = S[curId] && S[curId].flag;
  const idAt = curId;
  if (prev === v) {
    delete S[idAt]; save(); undoStack.push({ id: idAt, prev: prev });
    if (!view().some(function (x) { return x.id === idAt; })) setFilter('TODO');
    render(); toast('已清除本条'); return;
  }
  S[idAt] = { v: v, at: new Date().toISOString(), kind: 'spacetime',
              a: it.a, b: it.b, pair: it.pair, ov: it.ov, group: it.group,
              flag: prevFlag || null };
  save(); paintBtns(v, prevFlag);
  undoStack.push({ id: idAt, prev: prev });
  if (undoStack.length > 80) undoStack.shift();
  if (navigator.vibrate) navigator.vibrate(8);
  setTimeout(advance, 160);
}
function judgeFlag(v) {
  if (!curId) return;
  const r = S[curId];
  const cur = r && r.flag;
  if (!r) { toast('先判同一人 / 不同人'); return; }
  r.flag = cur === v ? null : v;
  save(); paintBtns(r.v, r.flag); toast(r.flag ? '标记：' + ZH[v] : '已清除标记');
}
function undo() {
  const last = undoStack.pop();
  if (!last) { toast('没有可撤销的步骤'); return; }
  if (last.prev) {
    S[last.id] = { v: last.prev, at: new Date().toISOString(), kind: 'spacetime' };
  } else delete S[last.id];
  save(); curId = last.id;
  if (!view().some(function (x) { return x.id === curId; })) setFilter('ALL');
  render();
  if (window.scrollY) window.scrollTo({ top: 0, behavior: 'smooth' });
  toast('撤销 ' + last.id + (last.prev ? '：仍为「' + ZH[last.prev] + '」' : '：已清空'));
}
function advance() {
  const v = view();
  if (!v.length) { render(); return; }
  const i = v.findIndex(function (x) { return x.id === curId; });
  if (i < 0) curId = v[0].id;
  else if (i + 1 < v.length) curId = v[i + 1].id;
  else if (v.length === 1) { /* 只剩一条 */ }
  else curId = v[0].id;
  render();
  if (window.scrollY) window.scrollTo({ top: 0, behavior: 'smooth' });
}
function step(d) {
  const v = view();
  if (v.length < 2) return;
  let i = v.findIndex(function (x) { return x.id === curId; });
  if (i < 0) i = 0;
  curId = v[(i + d + v.length) % v.length].id;
  render();
  if (window.scrollY) window.scrollTo({ top: 0, behavior: 'smooth' });
}

$('#btnPrev').onclick = function () { step(-1); };
$('#btnNext').onclick = function () { step(1); };
$('#btnUndo').onclick = undo;
document.querySelectorAll('.btns button').forEach(function (b) {
  b.onclick = function () { b.closest('.btns.sub') ? judgeFlag(b.dataset.v) : judge(b.dataset.v); };
});
function setFilter(f) {
  filter = f;
  document.querySelectorAll('#chips button').forEach(function (x) {
    x.setAttribute('aria-pressed', String(x.dataset.f === f));
  });
}
document.querySelectorAll('#chips button').forEach(function (b) {
  b.onclick = function () {
    setFilter(b.dataset.f);
    if (filter !== 'ALL' && curId && !view().some(function (x) { return x.id === curId; })) curId = null;
    render();
    if (window.scrollY) window.scrollTo({ top: 0 });
  };
});
document.addEventListener('keydown', function (e) {
  const t = e.target;
  if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return;
  if (e.key === '1') judge('same');
  else if (e.key === '2') judge('diff');
  else if (e.key === '3') judge('unsure');
  else if (e.key === '4') judgeFlag('boundary');
  else if (e.key === '5') judgeFlag('mixA');
  else if (e.key === '6') judgeFlag('mixB');
  else if (e.key === 'u' || e.key === 'U') undo();
  else if (e.key === 'j' || e.key === 'J' || e.key === 'ArrowRight') step(1);
  else if (e.key === 'k' || e.key === 'K' || e.key === 'ArrowLeft') step(-1);
});

function openOv(src) {
  $('#ovI').src = src;
  $('#ov').hidden = false;
  document.body.style.overflow = 'hidden';
}
function closeOv() { $('#ov').hidden = true; document.body.style.overflow = ''; }
$('#ovC').onclick = closeOv;
$('#ov').onclick = function (e) { if (e.target.id === 'ov') closeOv(); };
$('#ovB').onclick = function () {
  const on = document.body.classList.toggle('bright');
  try { localStorage.setItem('hc_sp_bright', on ? '1' : '0'); } catch (e) {}
  toast(on ? '已提亮' : '已还原');
};

function buildJSON() {
  const labels = {};
  ITEMS.forEach(function (it) {
    const r = S[it.id];
    if (!r) return;
    labels[it.id] = { v: r.v || null, flag: r.flag || null, at: r.at || null,
                      a: it.a, b: it.b, pair: it.pair, ov: it.ov, group: it.group };
  });
  return JSON.stringify({ version: 1, source: 'spacetime_review',
    updated: new Date().toISOString(), n: ITEMS.length,
    n_labeled: Object.keys(labels).length, labels: labels }, null, 1);
}
function statsText() {
  let n = 0, same = 0, diff = 0, uns = 0, bnd = 0;
  ITEMS.forEach(function (it) {
    const v = verdict(it.id);
    if (v === 'same') { n++; same++; }
    else if (v === 'diff') { n++; diff++; }
    else if (v === 'unsure') { uns++; }
    if (S[it.id] && S[it.id].flag === 'boundary') bnd++;
  });
  const dec = same + diff;
  return '<b>可判</b> ' + dec + ' 条：同人 ' + same + ' / 不同人 ' + diff +
    (dec ? ' → <b>误合并率 ' + (diff / dec * 100).toFixed(1) + '%</b>' : '') +
    '<br>看不清 ' + uns + '　交界处走过 <b>' + bnd + '</b>（这些不计入误合并率）' +
    '<br><span style="color:#9aa1ab">未判 ' + (ITEMS.length - nDone()) + ' 条</span>';
}
$('#btnOut').onclick = function () {
  const miss = ITEMS.length - nDone();
  $('#outSub').textContent = miss
    ? ('还有 ' + miss + ' 条没判。可以先导出——未判的不写入，之后补齐再导一次覆盖即可。')
    : (ITEMS.length + ' 条全部判完。把这段 JSON 发回来，我直接回写成跨机位真值。');
  $('#out').value = buildJSON();
  $('#stats').innerHTML = statsText();
  $('#sheet').hidden = false;
};
$('#btnCloseSheet').onclick = function () { $('#sheet').hidden = true; };
$('#btnCopy').onclick = async function () {
  const t = $('#out');
  try { await navigator.clipboard.writeText(t.value); toast('已复制 JSON'); }
  catch (e) {
    t.removeAttribute('readonly'); t.focus(); t.select();
    const ok = document.execCommand && document.execCommand('copy');
    t.setAttribute('readonly', 'readonly');
    toast(ok ? '已复制' : '长按文本框手动复制');
  }
};
let tTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg; t.classList.add('on');
  clearTimeout(tTimer);
  tTimer = setTimeout(function () { t.classList.remove('on'); }, 1700);
}

(function boot() {
  const un = ITEMS.filter(function (it) { return !isJudge(verdict(it.id)); });
  curId = (un[0] || ITEMS[0] || {}).id || null;
  if (!un.length) setFilter('ALL');
  try { if (localStorage.getItem('hc_sp_bright') === '1') document.body.classList.add('bright'); } catch (e) {}
  render();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
