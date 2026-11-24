#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 R3 新增接受的轨迹渲染成人工复核图（一张图 = 一条 query 轨迹 + 它被分配到的 gallery 簇）。

三行布局：
  第 1 行：query 轨迹前半段（a）最多 4 帧，绿框
  第 2 行：query 轨迹后半段（b）最多 4 帧，绿框
  第 3 行：所分配簇的 gallery 成员，最多 4 条（优先不同机位），蓝框

输入：r3_audit_added_tracks.csv（seg,tid,cam,abs_cos_a,abs_cos_b,assigned_cluster,...）
输出：identity_audit/{NN}_{seg}_{tid}.jpg + index.csv（含空的 verdict/note 两列供人工填）

用法：
  python tools/render_identity_audit.py
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S  # noqa: E402
import eval_mask_variants as M  # noqa: E402
from selector import _norm  # noqa: E402

ROOT = S.ROOT
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
VID = os.path.join(ROOT, "media-resource/1012/segments")
CSV_IN = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/r3_audit_added_tracks.csv")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/identity_audit")
if "--csv" in sys.argv:
    CSV_IN = sys.argv[sys.argv.index("--csv") + 1]
if "--out" in sys.argv:
    OUT = sys.argv[sys.argv.index("--out") + 1]
CLUSTER_MODE = "camera" if "--cluster-mode" in sys.argv and "camera" in sys.argv[sys.argv.index("--cluster-mode") + 1] else "global"

CELL = 340
HDR = 34
COLS = 4


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


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    return img if ok else None


def tile(img, box, label, color):
    if img is None:
        return np.full((CELL, CELL, 3), 30, np.uint8)
    h, w = img.shape[:2]
    if box is not None:
        box = [int(round(float(v))) for v in box]
        x1, y1, x2, y2 = box
        pad = max(8, int(0.15 * max(x2 - x1, y2 - y1)))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 - x1 < 12 or y2 - y1 < 12:
            x1, y1, x2, y2 = 0, 0, w, h
        crop = img[y1:y2, x1:x2].copy()
        cv2.rectangle(crop, (box[0] - x1, box[1] - y1), (box[2] - x1, box[3] - y1), color, 2)
    else:
        crop = img.copy()
    ch, cw = crop.shape[:2]
    s = CELL / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_LINEAR)
    out = np.full((CELL, CELL, 3), 20, np.uint8)
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


def main():
    ktr, Va, Vb = M.load_split(os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb"), S.TRAIN_SEGS)
    Vtr = _norm((Va + Vb) / 2.0)
    if CLUSTER_MODE == "camera":
        import cluster_camera_aware as CA
        cams = np.array([k[0].split("_")[0] for k in ktr])
        lab, kept = CA.camera_aware_labels(Vtr, cams, 0.15, 0.25, S.MIN_CLUSTER)
    else:
        lab, kept = S.hier_labels(Vtr)

    rows = list(csv.DictReader(open(CSV_IN, encoding="utf-8-sig")))
    rows.sort(key=lambda r: -(float(r["abs_cos_a"]) + float(r["abs_cos_b"])) / 2)

    os.makedirs(OUT, exist_ok=True)
    det_cache, caps = {}, {}
    index = []

    for n, r in enumerate(rows, 1):
        seg, tid = r["seg"], int(r["tid"])
        c = int(r["assigned_cluster"])
        dets = load_dets(seg, det_cache)
        frames = sorted(f for f, (ids, _) in dets.items() if tid in ids)
        if not frames:
            print(f"[miss] {seg}#{tid} 无检测帧")
            continue
        fa, fb = frames[: len(frames) // 2], frames[len(frames) // 2:]

        vp = os.path.join(VID, f"{seg}.mp4")
        if seg not in caps:
            caps[seg] = cv2.VideoCapture(vp)
        cap = caps[seg]

        cells = []
        for f in pick_even(fa, COLS):
            ids, boxes = dets[f]
            b = boxes[ids.index(tid)]
            cells.append(tile(grab(cap, f), b, f"Q-a f{f}", (80, 220, 80)))
        while len(cells) < COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))
        for f in pick_even(fb, COLS):
            ids, boxes = dets[f]
            b = boxes[ids.index(tid)]
            cells.append(tile(grab(cap, f), b, f"Q-b f{f}", (80, 220, 80)))
        while len(cells) < 2 * COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))

        # gallery 成员：该簇在 train 段的轨迹，优先不同机位
        mem = [ktr[i] for i in range(len(ktr)) if lab[i] == kept[c]]
        mem.sort(key=lambda k: (k[0].split("_")[0], k[1]))
        pick, seen_cam = [], set()
        for m in mem:                       # 先保证机位多样
            cam = m[0].split("_")[0]
            if cam in seen_cam:
                continue
            seen_cam.add(cam)
            pick.append(m)
            if len(pick) >= COLS:
                break
        for m in mem:                       # 不足再补同机位
            if len(pick) >= COLS:
                break
            if m not in pick:
                pick.append(m)
        pick = pick[:COLS]
        for mseg, mtid in pick:
            md = load_dets(mseg, det_cache)
            mf = sorted(f for f, (ids, _) in md.items() if mtid in ids)
            if not mf:
                continue
            f = mf[len(mf) // 2]
            ids, boxes = md[f]
            b = boxes[ids.index(mtid)]
            vp2 = os.path.join(VID, f"{mseg}.mp4")
            if mseg not in caps:
                caps[mseg] = cv2.VideoCapture(vp2)
            cells.append(tile(grab(caps[mseg], f), b, f"G {mseg}#{mtid}", (230, 170, 60)))
        while len(cells) < 3 * COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))

        grid = np.vstack([np.hstack(cells[i * COLS:(i + 1) * COLS]) for i in range(3)])
        hdr = np.full((HDR, grid.shape[1], 3), 245, np.uint8)
        txt = (f"{n:02d}/{len(rows)}  {seg} #{tid}  cosA={float(r['abs_cos_a']):.2f} "
               f"cosB={float(r['abs_cos_b']):.2f}  cluster={c} (size {r['cluster_size']})")
        cv2.putText(hdr, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2)
        img = np.vstack([hdr, grid])

        name = f"{n:02d}_{seg}_{tid}.jpg"
        cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        index.append({"image": name, "seg": seg, "tid": tid, "cam": r["cam"],
                      "abs_cos_a": r["abs_cos_a"], "abs_cos_b": r["abs_cos_b"],
                      "cluster": c, "cluster_size": r["cluster_size"],
                      "verdict": "", "note": ""})
        print(f"[{n:02d}/{len(rows)}] {name}")

    with open(os.path.join(OUT, "index.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    for c in caps.values():
        c.release()
    print(f"[out] {OUT}  ({len(index)} 张)")


if __name__ == "__main__":
    main()
