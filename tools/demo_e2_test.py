#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E2 选择器的测试级演示：身份卡 + 标注视频。

说明（口径）：1012 的 4 个 test 段时段互不重叠（323 12:57 / 305 13:28 / 318 14:22 / 320 15:03），
因此本演示**不是**"同一时刻跨机位比对"，而是"下午的轨迹被归到上午画廊里的哪个身份"。
这正好是 E2 的真实用途（跨会话身份延续）。

产物：
  demo_cards_<seg>.jpg   身份卡：每个簇一行，左列=上午画廊成员，右列=下午被归入的轨迹
  demo_video_<seg>.mp4   标注视频：框上画簇 ID 与校准分（z）
  demo_<seg>.json        该段统计（接受率、各簇成员数、使用的原型数）

用法：
  python tools/demo_e2_test.py --seg 305_test_544 --start 0 --dur 40 --topk 8
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
VID = os.path.join(ROOT, "media-resource/1012/segments")
AB = os.path.join(ROOT, "output_crosscam_local/1012_ab")
CROPS = os.path.join(ROOT, "output_crosscam_local/1012_osnet_crops")  # 可被 --crops-dir 覆盖
ART = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/selector_osnet_1012_e2.npz")
OUT = os.path.join(ROOT, "output_crosscam_local/demo_e2")
SAMPLES = 8
THETA = 2.75  # E2 在 test 上的 self-cal 阈值（见 docs/e1-e3-results-2026-10-17.md）

_cache_vid = {}
_cache_tracks = {}


def tracks_of(seg):
    if seg in _cache_tracks:
        return _cache_tracks[seg]
    t = {}
    for ab in ("a", "b"):
        p = os.path.join(AB, f"{seg}_{ab}.jsonl")
        d = defaultdict(list)
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                tids = r.get("track_ids") or []
                for i, b in enumerate(r.get("boxes") or []):
                    if i < len(tids) and tids[i] is not None and int(tids[i]) >= 0:
                        d[int(tids[i])].append((int(r["frame"]), b))
        for k in d:
            d[k].sort()
        t["a" if ab == "a" else "b"] = d
    _cache_tracks[seg] = t
    return t


def video_of(seg):
    if seg not in _cache_vid:
        _cache_vid[seg] = cv2.VideoCapture(os.path.join(VID, f"{seg}.mp4"))
    return _cache_vid[seg]


def get_crop(seg, tid, ab, k, pad=0.08):
    tf = tracks_of(seg)[ab].get(tid)
    if not tf:
        return None
    idxs = np.linspace(0, len(tf) - 1, min(SAMPLES, len(tf))).astype(int)
    if k >= len(idxs):
        return None
    fnum, box = tf[idxs[k]]
    cap = video_of(seg)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fnum))
    ok, fr = cap.read()
    if not ok:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    x1 -= w * pad
    x2 += w * pad
    y1 -= h * pad
    y2 += h * pad
    X1, Y1 = max(0, int(x1)), max(0, int(y1))
    X2, Y2 = min(fr.shape[1], int(x2)), min(fr.shape[0], int(y2))
    c = fr[Y1:Y2, X1:X2]
    if c.size == 0:
        return None
    return fit_pad(c)


def load_e2():
    d = np.load(ART)
    meta = json.loads(str(d["meta"]))
    return {"protos": d["protos"], "pcl": d["proto_cluster"], "m_hat": d["m_hat"],
            "s_hat": d["s_hat"], "K": int(meta["n_clusters"])}



def fit_model(crops_dir):
    """从指定逐裁特征目录重建 E2 模型（画廊 + 每身份校准 + self-cal θ）。"""
    def agg(seg):
        d = np.load(os.path.join(crops_dir, f"{seg}_crops.npz"))
        eids, emb = d["eids"], d["emb"]
        buck = defaultdict(list)
        for i, e in enumerate(eids):
            ab, tid, _k = str(e).split("#")
            buck[(ab, int(tid))].append(i)
        out = {}
        for (ab, tid), idx in buck.items():
            v = emb[np.asarray(idx)].mean(0)
            out[(ab, tid)] = v / (np.linalg.norm(v) + 1e-9)
        return out

    def split(segs):
        keys, Va, Vb = [], [], []
        for s in segs:
            a = agg(s)
            for t in sorted({t for (_ab, t) in a}):
                if ("a", t) in a and ("b", t) in a:
                    keys.append((s, t)); Va.append(a[("a", t)]); Vb.append(a[("b", t)])
        return keys, np.asarray(Va, np.float32), np.asarray(Vb, np.float32)

    ktr, Va_tr, Vb_tr = split(S.TRAIN_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr); K = len(kept); mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        if m.sum() == 0:
            continue
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0]); pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g); pcl = np.asarray(pcl)
    mh, sh, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)

    kte, Va_te, Vb_te = split(S.TEST_SEGS)
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(mh, sh))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(mh, sh))
    cooc = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte), s, 5):
            cooc.append(((s, u), (s, v)))
    rows = S.sweep(Sa, Sb, kte, cooc, np.round(np.arange(-2.0, 8.01, 0.05), 3))
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90 and r["evaluable"] >= 100]
    theta = max(ok, key=lambda r: r["coverage"])["thr"] if ok else 2.75
    print(f"[fit] K={K} theta={theta:.2f}")
    return {"protos": protos, "pcl": pcl, "K": K, "m_hat": mh, "s_hat": sh,
            "theta": float(theta), "mu": mu_q}


def compute_assign(model, seg, mu, crops_dir=None, theta=None):
    """对该 test 段做对齐+z 校准+阈值判定。mu 为会话均值（四个 test 段合并估计）。"""
    crops = np.load(os.path.join(crops_dir or CROPS, f"{seg}_crops.npz"))
    eids, emb = crops["eids"], crops["emb"]
    buckets = defaultdict(list)
    for i, e in enumerate(eids):
        ab, tid, k = str(e).split("#")
        buckets[(ab, int(tid))].append(i)
    keys = sorted({t for (_ab, t) in buckets})
    Va, Vb, valid = [], [], []
    for t in keys:
        ia, ib = buckets.get(("a", t)), buckets.get(("b", t))
        if not ia or not ib:
            continue
        ma = _norm(emb[ia].mean(0, keepdims=True))[0]
        mb = _norm(emb[ib].mean(0, keepdims=True))[0]
        Va.append(ma)
        Vb.append(mb)
        valid.append(t)
    Va, Vb = np.asarray(Va, np.float32), np.asarray(Vb, np.float32)
    Sa = S.score_rows(Va, model["protos"], model["pcl"], model["K"], mu=mu,
                      calib=(model["m_hat"], model["s_hat"]))
    Sb = S.score_rows(Vb, model["protos"], model["pcl"], model["K"], mu=mu,
                      calib=(model["m_hat"], model["s_hat"]))
    ia, za = Sa.argmax(1), Sa.max(1)
    ib, zb = Sb.argmax(1), Sb.max(1)
    out = {}
    for i, t in enumerate(valid):
        th = THETA if theta is None else theta
        accepted = za[i] >= th and zb[i] >= th and ia[i] == ib[i]
        out[t] = {"cid": int(ia[i]) if accepted else -1,
                  "z": float(min(za[i], zb[i])),
                  "consistent": bool(ia[i] == ib[i])}
    return out, float(np.mean([v["cid"] >= 0 for v in out.values()]))


def session_mu():
    """四个 test 段合并的会话均值（与 final_e0_e3.py 的 mu_q 口径一致）。"""
    vecs = []
    for s in S.TEST_SEGS:
        p = os.path.join(CROPS, f"{s}_crops.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        eids, emb = d["eids"], d["emb"]
        buck = defaultdict(list)
        for i, e in enumerate(eids):
            ab, tid, _k = str(e).split("#")
            buck[(ab, int(tid))].append(i)
        for t in {tid for (_ab, tid) in buck}:
            ia, ib = buck.get(("a", t)), buck.get(("b", t))
            if not ia or not ib:
                continue
            ma = _norm(emb[ia].mean(0, keepdims=True))[0]
            mb = _norm(emb[ib].mean(0, keepdims=True))[0]
            vecs.append(_norm(((ma + mb) / 2.0)[None, :])[0])
    return np.asarray(vecs, np.float32).mean(0)


def fit_pad(img, w=96, h=224, pad_color=235):
    """保持长宽比缩放后居中填充，避免拉伸变形。"""
    ih, iw = img.shape[:2]
    s = min(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), pad_color, np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def draw_video(seg, assign, start, dur, path, theta=THETA):
    """用 supervision 渲染：BoxAnnotator 画框、LabelAnnotator 画标签，按身份配色。"""
    cap = video_of(seg)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1280.0 / W
    outw, outh = 1280, int(H * scale)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (outw, outh))

    box_acc = sv.BoxAnnotator(thickness=3)
    lbl_acc = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_padding=6,
                                text_position=sv.Position.TOP_LEFT, smart_position=True)
    gray = sv.Color(r=150, g=150, b=150)
    box_unk = sv.BoxAnnotator(color=gray, thickness=2)
    lbl_unk = sv.LabelAnnotator(color=gray, text_scale=0.5, text_thickness=1,
                                text_padding=5, text_position=sv.Position.TOP_LEFT)

    tr = tracks_of(seg)
    by_frame = defaultdict(list)
    for ab in ("a", "b"):
        for tid, lst in tr[ab].items():
            if tid not in assign:
                continue
            for fnum, box in lst:
                by_frame[fnum].append((tid, box))

    f0, f1 = int(start * fps), int((start + dur) * fps)
    # 检测是每 5 帧抽样一次 → 把每个检测结果保持到下一个检测帧，保证逐帧可见
    disp = {}
    cur = {}
    for f in sorted(by_frame):
        if f <= f0:
            for tid, box in by_frame[f]:
                cur[tid] = box
    for f in range(f0, f1):
        if f in by_frame:
            for tid, box in by_frame[f]:
                cur[tid] = box
        disp[f] = list(cur.items())

    idx = 0
    written = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        if idx - 1 < f0:
            continue
        if idx - 1 >= f1:
            break
        out = cv2.resize(fr, (outw, outh), interpolation=cv2.INTER_AREA)

        acc_xy, acc_cid, acc_lab, unk_xy, unk_lab = [], [], [], [], []
        for tid, box in disp.get(idx - 1, []):
            a = assign[tid]
            xy = np.asarray(box, np.float32) * scale
            if a["cid"] >= 0:
                acc_xy.append(xy)
                acc_cid.append(int(a["cid"]) % len(sv.ColorPalette.DEFAULT))
                acc_lab.append(f"ID{a['cid']} {a['z']:.1f}")
            else:
                unk_xy.append(xy)
                unk_lab.append(f"t{tid} ?")
        if acc_xy:
            det = sv.Detections(xyxy=np.stack(acc_xy), class_id=np.asarray(acc_cid, int))
            out = box_acc.annotate(out, det)
            out = lbl_acc.annotate(out, det, labels=acc_lab)
        if unk_xy:
            det2 = sv.Detections(xyxy=np.stack(unk_xy),
                                 class_id=np.zeros(len(unk_xy), int))
            out = box_unk.annotate(out, det2)
            out = lbl_unk.annotate(out, det2, labels=unk_lab)

        msg = f"{seg}  t={(idx-1)/fps:6.1f}s  z>={theta}  gray=unknown"
        cv2.putText(out, msg, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(out, msg, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        vw.write(out)
        written += 1
    vw.release()
    print(f"[draw] frames={written} by_frame={len(by_frame)} assign={len(assign)}")
    return written


def draw_cards(model, seg, assign, topk, path):
    """每个簇一张行卡：上午画廊成员 vs 下午被归入的轨迹。"""
    members = defaultdict(list)
    # 上午画廊成员
    ktr, Va, Vb, ctr = S.load(S.TRAIN_SEGS)
    Vtr = _norm((Va + Vb) / 2.0)
    lab2, kept = S.hier_labels(Vtr)
    idx_of = {k: i for i, k in enumerate(kept)}
    for i, c in enumerate(lab2):
        if c in idx_of:
            s, tid = ktr[i]
            members[idx_of[c]].append(("M", s, tid, "a", 0))
    # 下午被归入的轨迹
    for tid, a in assign.items():
        if a["cid"] >= 0:
            members[a["cid"]].append(("T", seg, tid, "a", 0))

    rows = [(cid, v) for cid, v in members.items() if sum(1 for x in v if x[0] == "T") >= 1]
    rows.sort(key=lambda kv: -sum(1 for x in kv[1] if x[0] == "T"))
    rows = rows[:topk]
    if not rows:
        return None
    # 每行固定：左 3 张"跨机位画廊"（非本机位优先）+ 右 3 张"下午轨迹"，构成可直接判读的对照
    q_cam = seg.split("_")[0]
    trimmed = []
    for cid, v in rows:
        gal = [x for x in v if x[0] == "M"]
        gal.sort(key=lambda x: (0 if x[1].split("_")[0] != q_cam else 1, x[1], x[2]))
        tst = [x for x in v if x[0] == "T"][:3]
        trimmed.append((cid, gal[:3] + tst))
    rows = trimmed

    cw, ch, gap = 96, 224, 8
    label_w = 220
    maxrow = max(len(v) for _c, v in rows)
    cols = min(6, maxrow)
    row_h = ch + gap + 22
    W = label_w + cols * (cw + gap) + gap
    H = 62 + len(rows) * row_h + gap
    canvas = np.full((H, W, 3), 255, np.uint8)
    n_acc = sum(1 for a in assign.values() if a["cid"] >= 0)
    cv2.putText(canvas, f"E2 identity cards - {seg}   tracks={len(assign)}  "
                        f"matched={n_acc} ({n_acc/max(1,len(assign)):.1%})  "
                        f"identities={len(rows)}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(canvas, "M=morning gallery (cross-camera first)   T=afternoon track matched here   "
                        f"theta(z)={THETA}",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    for r, (cid, v) in enumerate(rows):
        y = 62 + r * row_h
        nT = sum(1 for x in v if x[0] == "T")
        nG = len(v) - nT
        cv2.putText(canvas, f"ID{cid}  gallery={nG} matched={nT}",
                    (8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for c, (sp, s, tid, ab, k) in enumerate(v[:cols]):
            crop = get_crop(s, tid, ab, k)
            if crop is None:
                continue
            x = label_w + c * (cw + gap)
            canvas[y + 20:y + 20 + ch, x:x + cw] = crop
            tag = f"{sp} {s.split('_')[0]}#{tid}"
            cv2.putText(canvas, tag, (x, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0) if sp == "M" else (0, 0, 200), 1)
            cv2.rectangle(canvas, (x, y + 20), (x + cw, y + 20 + ch),
                          (180, 180, 180) if sp == "M" else (0, 0, 200), 1)
    cv2.imwrite(path, canvas)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="305_test_544")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=40.0)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--crops-dir", default=None, help="逐裁特征目录（默认基线 1012_osnet_crops）")
    ap.add_argument("--suffix", default="", help="输出文件名后缀")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.crops_dir:
        model = fit_model(a.crops_dir)
        mu = model["mu"]
        assign, cov = compute_assign(model, a.seg, mu, a.crops_dir, model["theta"])
        TH = model["theta"]
    else:
        model = load_e2()
        mu = session_mu()
        assign, cov = compute_assign(model, a.seg, mu, a.crops_dir)
        TH = THETA
    n_acc = sum(1 for v in assign.values() if v["cid"] >= 0)
    print(f"[{a.seg}] tracks={len(assign)} accepted={n_acc} ({cov:.3f})")

    vp = os.path.join(OUT, f"demo_video_{a.seg}{a.suffix}.mp4")
    n = draw_video(a.seg, assign, a.start, a.dur, vp, TH)
    print(f"[video] {vp} ({n} frames)")

    cp = os.path.join(OUT, f"demo_cards_{a.seg}{a.suffix}.jpg")
    r = draw_cards(model, a.seg, assign, a.topk, cp)
    print(f"[cards] {r}")

    stat = {"seg": a.seg, "n_tracks": len(assign), "n_accepted": n_acc,
            "coverage": cov, "theta": THETA,
            "clusters_used": len({v["cid"] for v in assign.values() if v["cid"] >= 0}),
            "assign": {str(k): v for k, v in sorted(assign.items())}}
    json.dump(stat, open(os.path.join(OUT, f"demo_{a.seg}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[json] {os.path.join(OUT, f'demo_{a.seg}.json')}")


if __name__ == "__main__":
    main()
