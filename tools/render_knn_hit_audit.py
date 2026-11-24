#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1-NN 复核图（标出真正命中的 gallery 原型）。

与 render_identity_audit.py 的差别：knn 模式下判决依据不是「簇均值」，
而是「簇内与 query 最像的那一条成员」。原渲染器按「机位多样性」挑 4 个成员，
18 条里有 5 条根本没画到命中那条 → 看图像是「全都不像」，实际判据没进画面。

布局（3 行 × 4 列）：
  第 1 行  Q-a   query 前半段，绿框；角标 = 与命中原型的余弦
  第 2 行  Q-b   query 后半段，绿框
  第 3 行  HIT   a 片命中成员（黄框）；若 b 片命中不同则第 2 格再放一张（橙框）
           其余格 = 同簇其它成员（蓝框），角标 = 与 query 的余弦

用法：
  python tools/render_knn_hit_audit.py --csv .../knn_audit_added.csv --out .../knn_hit_audit
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
import cluster_camera_aware as CA  # noqa: E402
import eval_pose_prototypes as P  # noqa: E402
from selector import _norm  # noqa: E402

ROOT = S.ROOT
DET = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
VID = os.path.join(ROOT, "media-resource/1012/segments")

CELL = 320
HDR = 52
COLS = 4

C_QUERY = (80, 220, 80)      # 绿：query
C_HIT = (60, 220, 235)       # 黄：a 片命中原型
C_HIT_B = (60, 165, 255)     # 橙：b 片命中原型（与 a 不同时）
C_OTHER = (230, 170, 60)     # 蓝：同簇其它成员


def _arg(name, dflt):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else dflt


CSV_IN = _arg("--csv", os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/knn_audit_added.csv"))
OUT = _arg("--out", os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/knn_hit_audit"))


def load_dets(seg, cache):
    if seg not in cache:
        d = {}
        with open(os.path.join(DET, seg, f"{seg}_detections.jsonl"), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                d[o["frame"]] = (o["track_ids"], o["boxes"])
        cache[seg] = d
    return cache[seg]


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    return img if ok else None


def tile(img, box, label, color, sub=None):
    """sub: 右下角第二行小字（余弦等）"""
    if img is None:
        out = np.full((CELL, CELL, 3), 30, np.uint8)
        cv2.putText(out, "no frame", (6, CELL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out
    h, w = img.shape[:2]
    if box is not None:
        b = [int(round(float(v))) for v in box]
        x1, y1, x2, y2 = b
        pad = max(8, int(0.15 * max(x2 - x1, y2 - y1)))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 - x1 < 12 or y2 - y1 < 12:
            x1, y1, x2, y2 = 0, 0, w, h
        crop = img[y1:y2, x1:x2].copy()
        cv2.rectangle(crop, (b[0] - x1, b[1] - y1), (b[2] - x1, b[3] - y1), color, 3)
    else:
        crop = img.copy()
    ch, cw = crop.shape[:2]
    s = CELL / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_LINEAR)
    out = np.full((CELL, CELL, 3), 20, np.uint8)
    y0, x0 = (CELL - crop.shape[0]) // 2, (CELL - crop.shape[1]) // 2
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    cv2.rectangle(out, (0, 0), (CELL - 1, CELL - 1), color, 2)
    # 顶部半透明条上写标签，避免遮住人
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (CELL, 26), (0, 0, 0), -1)
    out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
    cv2.putText(out, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
    if sub:
        ov = out.copy()
        cv2.rectangle(ov, (0, CELL - 24), (CELL, CELL), (0, 0, 0), -1)
        out = cv2.addWeighted(ov, 0.55, out, 0.45, 0)
        cv2.putText(out, sub, (5, CELL - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


def pick_even(seq, k):
    if not seq:
        return []
    if len(seq) <= k:
        return list(seq)
    return [seq[int(round(i * (len(seq) - 1) / (k - 1)))] for i in range(k)]


def main():
    ktr, Va, Vb = M.load_split(os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb"), S.TRAIN_SEGS)
    kte, Qa, Qb = M.load_split(os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb"), S.TEST_SEGS)
    Vtr = _norm((Va + Vb) / 2.0)
    cams = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams, 0.15, 0.25, S.MIN_CLUSTER)
    K = len(kept)
    mu_g, mu_q = Vtr.mean(0), _norm((Qa + Qb) / 2.0).mean(0)

    # 构造与 ktr 同索引空间的原型表（protos 的顺序 = ktr 的顺序）。
    # 注意：不能直接用 build_protos("knn") —— 它按簇分组 append，索引空间与 ktr 不同，
    # 用它反查「命中的是谁」会张冠李戴（曾导致 18 张图全部标错成员）。
    T = _norm(Vtr - mu_g)
    col_of = np.full(len(ktr), -1, np.int32)
    for ci, c in enumerate(kept):
        col_of[lab == c] = ci
    keep_idx = np.where(col_of >= 0)[0]          # 与 build_protos 相同的成员集合
    protos, pcl = T[keep_idx], col_of[keep_idx]
    Sfa = _norm(Qa - mu_q) @ protos.T
    Sfb = _norm(Qb - mu_q) @ protos.T

    rows = list(csv.DictReader(open(CSV_IN, encoding="utf-8-sig")))
    rows.sort(key=lambda r: -(float(r["cos_a"]) if "cos_a" in r else (float(r["abs_cos_a"]) + float(r["abs_cos_b"])) / 2))
    kidx = {k: i for i, k in enumerate(kte)}

    os.makedirs(OUT, exist_ok=True)
    det_cache, caps, index = {}, {}, []

    for n, r in enumerate(rows, 1):
        seg, tid = r["seg"], int(r["tid"])
        c = int(r["assigned_cluster"])
        i = kidx[(seg, tid)]
        cols = np.where(pcl == c)[0]              # protos 空间的位置下标
        ja_pos = int(cols[int(np.argmax(Sfa[i, cols]))])
        jb_pos = int(cols[int(np.argmax(Sfb[i, cols]))])
        ja, jb = int(keep_idx[ja_pos]), int(keep_idx[jb_pos])   # → ktr 索引空间
        ca, cb = float(Sfa[i, ja_pos]), float(Sfb[i, jb_pos])
        rank_a = int((Sfa[i, cols] > ca).sum()) + 1
        rank_b = int((Sfb[i, cols] > cb).sum()) + 1

        dets = load_dets(seg, det_cache)
        frames = sorted(f for f, (ids, _) in dets.items() if tid in ids)
        if not frames:
            continue
        fa, fb = frames[: len(frames) // 2], frames[len(frames) // 2:]
        if seg not in caps:
            caps[seg] = cv2.VideoCapture(os.path.join(VID, f"{seg}.mp4"))

        cells = []
        for f in pick_even(fa, COLS):
            ids, boxes = dets[f]
            cells.append(tile(grab(caps[seg], f), boxes[ids.index(tid)],
                              f"Q-a f{f}", C_QUERY, f"cos2hit {ca:.2f}"))
        while len(cells) < COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))
        for f in pick_even(fb, COLS):
            ids, boxes = dets[f]
            cells.append(tile(grab(caps[seg], f), boxes[ids.index(tid)],
                              f"Q-b f{f}", C_QUERY, f"cos2hit {cb:.2f}"))
        while len(cells) < 2 * COLS:
            cells.append(np.full((CELL, CELL, 3), 20, np.uint8))

        # ---- 第 3 行 ----
        def draw_member(j, tag, color, cos_val, rank):
            mseg, mtid = ktr[j]
            md = load_dets(mseg, det_cache)
            mf = sorted(f for f, (ids, _) in md.items() if mtid in ids)
            if not mf:
                return np.full((CELL, CELL, 3), 20, np.uint8)
            f = mf[len(mf) // 2]
            ids, boxes = md[f]
            if mseg not in caps:
                caps[mseg] = cv2.VideoCapture(os.path.join(VID, f"{mseg}.mp4"))
            return tile(grab(caps[mseg], f), boxes[ids.index(mtid)],
                        f"{tag} {mseg}#{mtid}", color, f"cos {cos_val:.2f}  rank{rank}/{len(cols)}")

        def draw_pos(p, tag, color):
            """p 为 protos 空间位置，转成 ktr 索引后画。"""
            return draw_member(int(keep_idx[p]), tag, color, float(Sfa[i, p]),
                               int((Sfa[i, cols] > Sfa[i, p]).sum()) + 1)

        cells.append(draw_member(ja, "HIT-a", C_HIT, ca, rank_a))
        order_pos = [int(cols[p]) for p in np.argsort(-Sfa[i, cols])]
        if jb_pos != ja_pos:
            cells.append(draw_member(jb, "HIT-b", C_HIT_B, cb, rank_b))
            rest = [q for q in order_pos if q not in (ja_pos, jb_pos)]
        else:
            # a/b 命中同一条：第 2 格画簇内次相似，用来显示判决余量
            rest = [q for q in order_pos if q != ja_pos]
            if rest:
                cells.append(draw_pos(rest[0], "RUNNER-UP", C_OTHER))
                rest = rest[1:]
            else:
                pad = np.full((CELL, CELL, 3), 20, np.uint8)
                cv2.putText(pad, "only 1 member", (6, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (70, 70, 70), 1)
                cells.append(pad)
        for p in rest[:2]:
            cells.append(draw_pos(p, "G", C_OTHER))
        while len(cells) < 3 * COLS:
            pad = np.full((CELL, CELL, 3), 20, np.uint8)
            cv2.putText(pad, "same cluster", (6, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1)
            cells.append(pad)

        grid = np.vstack([np.hstack(cells[i2 * COLS:(i2 + 1) * COLS]) for i2 in range(3)])
        hdr = np.full((HDR, grid.shape[1], 3), 245, np.uint8)
        t1 = (f"{n:02d}/{len(rows)}  {seg} #{tid}   cluster={c} (size {len(cols)})   "
              f"cosA={float(r['abs_cos_a']):.2f} cosB={float(r['abs_cos_b']):.2f}")
        t2 = "row1-2 = query (green) | row3 = gallery of THAT cluster: yellow = argmax hit, blue = other members"
        cv2.putText(hdr, t1, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
        cv2.putText(hdr, t2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (90, 90, 90), 1)
        img = np.vstack([hdr, grid])

        name = f"{n:02d}_{seg}_{tid}.jpg"
        cv2.imwrite(os.path.join(OUT, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        index.append(dict(image=name, seg=seg, tid=tid, cam=r["cam"], cluster=c,
                          cluster_size=len(cols),
                          hit_a=f"{ktr[ja][0]}#{ktr[ja][1]}", cos_a=round(ca, 3), rank_a=rank_a,
                          hit_b=f"{ktr[jb][0]}#{ktr[jb][1]}", cos_b=round(cb, 3), rank_b=rank_b,
                          verdict="", note=""))
        print(f"[{n:02d}/{len(rows)}] {name}  hit_a={ktr[ja][0]}#{ktr[ja][1]} cos={ca:.3f} rank{rank_a}")

    with open(os.path.join(OUT, "index.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    for cp in caps.values():
        cp.release()
    print(f"[out] {OUT}  ({len(index)} 张)")


if __name__ == "__main__":
    main()
