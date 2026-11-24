#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""渲染「头框轨迹」人工复核图。

头框轨迹 = 同一个真人被 YOLO 同时检成「头」和「全身」两条轨迹（NMS 因 IoU 只有 0.14 去不掉）。
本工具把每条头框轨迹和它的「宿主」（包含它的那条更大的轨迹）以及它在图库中被分配到的簇
放在同一张图上，用来回答一个问题：

    这条头框轨迹被「接受」时，落到的那一簇，是不是宿主那个人？

三行布局（同一张图）：
  第 1 行：头框轨迹本身（绿框）4 帧 —— 看它到底框住了什么
  第 2 行：同一批帧里的「宿主」全身（红框），头框以细绿线叠在里面 —— 看两者是否同一人
  第 3 行：所分配簇的 gallery 成员（蓝框）—— 看这个簇是不是宿主本人

用法：
    python tools/render_head_track_audit.py
输出：
    selector_osnet_1012/head_track_audit/{NN}_{seg}_{tid}.jpg
    selector_osnet_1012/head_track_audit/index.csv   （verdict/note 留空供人工填）
    selector_osnet_1012/head_track_audit/head_track_audit.pdf
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402

DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
EMB = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
VID = os.path.join(ROOT, "media-resource/1012/segments")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/head_track_audit")

CELL = 340
HDR = 58
COLS = 4
RATIO, CONTAIN = 0.70, 0.80      # 与 eval_head_exclusion.py 完全一致


def load_dets(seg, cache):
    if seg not in cache:
        p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
        d = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                d[o["frame"]] = (o["track_ids"], o["boxes"])
        cache[seg] = d
    return cache[seg]


def find_head_tracks(seg, det_path=None):
    """返回 {head_tid: (host_tid, ratio, obs)}，判据与 eval_head_exclusion 一致。"""
    p = det_path or os.path.join(DET, seg, f"{seg}_detections.jsonl")
    if not os.path.exists(p):
        return {}
    fr = [json.loads(l) for l in open(p, encoding="utf-8")]
    total, nested, host_cnt = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(int))
    for o in fr:
        for t in o["track_ids"]:
            total[t] += 1
    for o in fr:
        ts = list(zip(o["track_ids"], o["boxes"]))
        for ti, bi in ts:
            hb, wb = bi[3] - bi[1], bi[2] - bi[0]
            if hb <= 0 or wb <= 0:
                continue
            best = None
            for tj, bj in ts:
                if ti == tj:
                    continue
                if bj[3] - bj[1] <= hb * 1.2:
                    continue
                ix1, iy1 = max(bi[0], bj[0]), max(bi[1], bj[1])
                ix2, iy2 = min(bi[2], bj[2]), min(bi[3], bj[3])
                if max(0, ix2 - ix1) * max(0, iy2 - iy1) / (hb * wb) >= CONTAIN:
                    best = tj
                    break
            if best is not None:
                nested[ti] += 1
                host_cnt[ti][best] += 1
    out = {}
    for t in total:
        if total[t] >= 20 and nested[t] / total[t] >= RATIO:
            host = max(host_cnt[t].items(), key=lambda kv: kv[1])[0]
            out[t] = (host, nested[t] / total[t], total[t])
    return out


def tile(img, box, label, color, extra_box=None, extra_color=(80, 220, 80)):
    out = np.full((CELL, CELL, 3), 20, np.uint8)
    if img is None:
        return out
    h, w = img.shape[:2]
    box = [int(round(float(v))) for v in box]
    x1, y1, x2, y2 = box
    pad = max(10, int(0.22 * max(x2 - x1, y2 - y1)))
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    if cx2 - cx1 < 12 or cy2 - cy1 < 12:
        return out
    crop = img[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), color, 2)
    if extra_box is not None:
        eb = [int(round(float(v))) for v in extra_box]
        cv2.rectangle(crop, (eb[0] - cx1, eb[1] - cy1), (eb[2] - cx1, eb[3] - cy1), extra_color, 1)
    ch, cw = crop.shape[:2]
    s = CELL / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (CELL - crop.shape[0]) // 2, (CELL - crop.shape[1]) // 2
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    cv2.putText(out, label, (6, CELL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def pick_even(seq, k):
    if not seq:
        return []
    if len(seq) <= k:
        return list(seq)
    return [seq[int(round(i * (len(seq) - 1) / (k - 1)))] for i in range(k)]


def build_scores():
    """复现生产口径：8 段、生产 profile 聚类、self-cal 取 same>=0.90 且 evaluable>=100 下覆盖最大。"""
    ktr, Va_tr, Vb_tr = M.load_split(EMB, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(EMB, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        if m.sum() == 0:
            continue
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    from eval_selector_1012test import cooc_pairs
    keys_ok = set(kte)
    cooc = []
    for seg in S.TEST_SEGS:
        for a, b in cooc_pairs(os.path.join(DET, seg, f"{seg}_detections.jsonl"), keys_ok, seg, 5):
            cooc.append(((seg, a), (seg, b)))
    rows = S.sweep(Sa, Sb, kte, cooc, np.round(np.arange(-2.0, 8.01, 0.05), 3))
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
    thr = max(ok, key=lambda r: r["coverage"])["thr"]
    print("[cal] 自洽点 coverage=%.4f same=%.4f evaluable=%d"
          % (max(ok, key=lambda r: r["coverage"])["coverage"],
             max(ok, key=lambda r: r["coverage"])["same"], max(ok, key=lambda r: r["coverage"])["evaluable"]))
    if not np.isfinite(thr):
        thr = 2.5
    return dict(ktr=ktr, kte=kte, lab=lab, kept=kept, K=K, Sa=Sa, Sb=Sb, thr=thr,
                Va=Va_te, Vb=Vb_te)


def main():
    st = build_scores()
    ktr, kte, lab, kept, K = st["ktr"], st["kte"], st["lab"], st["kept"], st["K"]
    Sa, Sb, thr = st["Sa"], st["Sb"], st["thr"]
    print(f"[cal] 生产工作点 thr={thr:.2f}  n_test={len(kte)}  K={K}")

    kidx = {k: i for i, k in enumerate(kte)}
    det_cache, caps = {}, {}
    items = []
    for seg in S.TEST_SEGS:
        for tid, (host, ratio, obs) in find_head_tracks(seg).items():
            k = (seg, tid)
            kh = (seg, host)
            if k not in kidx:
                continue
            i = kidx[k]
            cos_host = None
            if kh in kidx:
                j = kidx[kh]
                v1 = _norm((st["Va"][i] + st["Vb"][i]) / 2.0)
                v2 = _norm((st["Va"][j] + st["Vb"][j]) / 2.0)
                cos_host = round(float(v1 @ v2), 3)
            ca, cb = int(Sa[i].argmax()), int(Sb[i].argmax())
            acc_a, acc_b = float(Sa[i].max()) >= thr, float(Sb[i].max()) >= thr
            items.append(dict(seg=seg, tid=int(tid), host=int(host), host_ratio=round(ratio, 2),
                              obs=int(obs), cam=seg.split("_")[0], cos_host=cos_host,
                              score_a=round(float(Sa[i].max()), 2), score_b=round(float(Sb[i].max()), 2),
                              cluster=ca, cluster_b=cb, same_cluster=int(ca == cb),
                              cluster_size=int((lab == kept[ca]).sum()),
                              accepted=int(acc_a and acc_b),
                              half_accepted=int(acc_a or acc_b)))
    items.sort(key=lambda r: (-r["accepted"], -min(r["score_a"], r["score_b"])))
    print(f"[find] 测试集头框轨迹 {len(items)} 条，其中双片接受 {sum(r['accepted'] for r in items)} 条")

    os.makedirs(OUT, exist_ok=True)

    index = []
    for n, r in enumerate(items, 1):
        seg, tid = r["seg"], r["tid"]
        dets = load_dets(seg, det_cache)
        frames = sorted(f for f, (ids, _) in dets.items() if tid in ids and r["host"] in ids)
        if not frames:
            print(f"[miss] {seg}#{tid}")
            continue
        vp = os.path.join(VID, f"{seg}.mp4")
        if seg not in caps:
            caps[seg] = cv2.VideoCapture(vp)
        cap = caps[seg]

        cells = []
        fc = {}
        for f in pick_even(frames, COLS):
            if f not in fc:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                okf, im = cap.read()
                fc[f] = im if okf else None
            img = fc[f]
            ids, boxes = dets[f]
            bh = boxes[ids.index(tid)]
            bb = boxes[ids.index(r["host"])]
            cells.append(tile(img, bh, f"H f{f}", (80, 220, 80)))
        while len(cells) < COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))
        for f in pick_even(frames, COLS):
            img = fc.get(f)
            ids, boxes = dets[f]
            bh = boxes[ids.index(tid)]
            bb = boxes[ids.index(r["host"])]
            cells.append(tile(img, bb, f"BODY #{r['host']} f{f}", (60, 60, 230),
                              extra_box=bh, extra_color=(80, 220, 80)))
        while len(cells) < 2 * COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))

        mem = [ktr[i] for i in range(len(ktr)) if lab[i] == kept[r["cluster"]]]
        mem.sort(key=lambda k: (k[0].split("_")[0], k[1]))
        pick, seen = [], set()
        for m in mem:
            cam = m[0].split("_")[0]
            if cam in seen:
                continue
            seen.add(cam)
            pick.append(m)
            if len(pick) >= COLS:
                break
        for m in mem:
            if len(pick) >= COLS:
                break
            if m not in pick:
                pick.append(m)
        for mseg, mtid in pick[:COLS]:
            md = load_dets(mseg, det_cache)
            mf = sorted(f for f, (ids, _) in md.items() if mtid in ids)
            if not mf:
                continue
            f = mf[len(mf) // 2]
            ids, boxes = md[f]
            if mseg not in caps:
                caps[mseg] = cv2.VideoCapture(os.path.join(VID, f"{mseg}.mp4"))
            c2 = caps[mseg]
            c2.set(cv2.CAP_PROP_POS_FRAMES, f)
            okf, im = c2.read()
            cells.append(tile(im if okf else None, boxes[ids.index(mtid)],
                              f"G {mseg}#{mtid}", (230, 170, 60)))
        while len(cells) < 3 * COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))

        grid = np.vstack([np.hstack(cells[i * COLS:(i + 1) * COLS]) for i in range(3)])
        hdr = np.full((HDR, grid.shape[1], 3), 245, np.uint8)
        t1 = (f"{n:02d}/{len(items)}  {seg} #{tid}  host=#{r['host']}（{int(r['host_ratio']*100)}%帧被其包含）  "
              f"obs={r['obs']}  cos(头,宿主)={r['cos_host'] if r['cos_host'] is not None else 'n/a'}"
              f"  [同人参考 0.915 / 异人 0.541]")
        state = "双片接受" if r["accepted"] else ("单片接受" if r["half_accepted"] else "拒识")
        tail = "" if r["same_cluster"] else "  b片->%d" % r["cluster_b"]
        t2 = ("score a=%.2f b=%.2f  thr=%.2f  %s  cluster=%d（size %d）%s"
              % (r["score_a"], r["score_b"], thr, state, r["cluster"], r["cluster_size"], tail))
        cv2.putText(hdr, t1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (10, 10, 10), 2)
        cv2.putText(hdr, t2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (10, 10, 10), 2)
        img = np.vstack([hdr, grid])

        name = f"{n:02d}_{seg}_{tid}.jpg"
        cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        index.append(dict(image=name, n=n, **{k: r[k] for k in
                         ("seg", "cam", "tid", "host", "host_ratio", "obs",
                          "score_a", "score_b", "cluster", "cluster_b", "same_cluster",
                          "cluster_size", "accepted", "half_accepted")},
                          verdict="", note=""))
        print(f"[{n:02d}/{len(items)}] {name}  acc={r['accepted']}")

    for c in caps.values():
        c.release()

    with open(os.path.join(OUT, "index.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    json.dump(items, open(os.path.join(OUT, "head_tracks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    try:
        from PIL import Image
        fs = sorted(os.path.join(OUT, x) for x in os.listdir(OUT)
                    if x[:2].isdigit() and x.endswith(".jpg"))
        if fs:
            imgs = [Image.open(x).convert("RGB") for x in fs]
            imgs[0].save(os.path.join(OUT, "head_track_audit.pdf"), save_all=True,
                         append_images=imgs[1:], resolution=100.0)
            print("[pdf]", os.path.join(OUT, "head_track_audit.pdf"), len(fs), "pages")
    except Exception as e:      # noqa: BLE001
        print("[pdf] skip:", e)
    print(f"[out] {OUT}  ({len(index)} 张)")


if __name__ == "__main__":
    main()
