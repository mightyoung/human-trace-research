#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E21 三天六机位的人脸抽取（跨天身份绑定用）。

为什么必须重抽
--------------
e21 里没有任何人脸产物：只有 dd（去重后的逐帧框）、emb（OSNet a/b 分片）、
results（阈值扫描）。要问「今天的脸能不能绑到昨天的身份簇」，就得先在原生
视频上把脸抽出来。

口径（与 `tools/face_query_scan.py` 一致，便于横向比较）
--------------------------------------------------------
- **按框裁 ROI 再检测**：整帧 letterbox 到 640，60 px 的脸只剩 15–20 px，
  SCRFD 在 1080p 整帧上实测零检出。
- **按框高挑帧**：面部像素预算 ≈ 0.054 × 框高，抽框最高的帧把单帧可用率
  从 6% 抬到 47%。
- ROI = 框的上 55%，左右各放 25%；脸心必须落在本轨框内、脸宽不得超过框宽。
- 质量门 pd ≥ 15 / 最小边 ≥ 40 / 清晰度 ≥ 40，宁可少不要脏模板。

时间轴
------
tr = 0–720 s，te = 810 s 到当日窗口尾（见 e21_run.py 的 SPLITS）。
det jsonl 的 `frame` 是**视频绝对帧号**（te 从 20250 起 = 810 s × 25 fps），
所以直接用它跳读，不要再叠加任何偏移。

用法
----
    python e21_face_scan.py --segs 2026-10-22_茶水间_tr --dry
    python e21_face_scan.py --segs 2026-10-22_茶水间_tr --limit 20
    python e21_face_scan.py            # 两天 24 段全跑
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scrfd_dnn import ScrfdDNN  # noqa: E402

DS = "/share/data4/yhh/data/human-count/e21"
MODELS = "/share/data4/yhh/human-count/models/face"
OUT = os.path.join(DS, "face")

DAYS = ("2026-10-22", "2026-10-23")
DAY_CAMS = {
    "2026-10-22": ["办公区1", "办公区2", "办公区球机", "茶水间", "行政走廊通道", "电梯间"],
    "2026-10-23": ["办公区1", "办公区2", "办公区球机", "茶水间", "走廊通道", "电梯间"],
}
SPLITS = ("tr", "te")

TPL112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
GATE = dict(pd=15.0, w=40.0, lap=40.0)
DET_THR = 0.40


def slice_of(day, cam):
    d = os.path.join(DS, day)
    if not os.path.isdir(d):
        return None
    files = [f for f in sorted(os.listdir(d))
             if f.endswith(".mp4") and f.startswith(f"{day}_{cam}_")]
    for f in files:
        if "merged" in f:
            return os.path.join(d, f)
    return os.path.join(d, files[0]) if files else None


def align112(img, kps5):
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), TPL112,
                                       method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def pd_of(kps):
    d = np.asarray(kps[0], np.float64) - np.asarray(kps[1], np.float64)
    return float(np.sqrt((d * d).sum()))


class Embedder:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(4, (os.cpu_count() or 8) // 2)
        prov = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, sess_options=so, providers=prov)
        self.iname = self.sess.get_inputs()[0].name
        self.dim = int(self.sess.get_outputs()[0].shape[-1])

    def __call__(self, aligned_bgr):
        blob = ((aligned_bgr[:, :, ::-1].astype(np.float32) - 127.5) / 127.5)
        v = self.sess.run(None, {self.iname: blob.transpose(2, 0, 1)[None]})[0][0]
        v = v.astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-9)


class FrameReader:
    """按帧号跳读；`cur` = 下一次 read 会拿到的帧号（grab 只返回 bool）。"""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.cur = -1
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self, f):
        f = int(max(0, min(f, max(0, self.n - 1))))
        if self.cur < 0 or f < self.cur or f - self.cur > 40:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            self.cur = f
        while self.cur < f:
            if not self.cap.grab():
                return None
            self.cur += 1
        ok, img = self.cap.read()
        if ok:
            self.cur += 1
        return img if ok else None

    def close(self):
        self.cap.release()


def pick_frames(pts, n):
    """按框高挑帧：先取最高的，再按时间均匀补足。"""
    if not pts:
        return []
    by_h = sorted(pts, key=lambda p: -(p[1][3] - p[1][1]))
    chosen, seen = [], set()

    def push(p):
        f = int(p[0])
        if any(abs(f - g) < 5 for g in seen):
            return
        seen.add(f)
        chosen.append(f)

    for p in by_h[: max(1, n - n // 3)]:
        push(p)
    step = max(1, len(pts) // max(1, n // 2))
    for i in range(0, len(pts), step):
        if len(chosen) >= n:
            break
        push(pts[i])
    return sorted(chosen)


def load_tracks(seg):
    p = os.path.join(DS, "dd", seg, f"{seg}_detections.jsonl")
    if not os.path.exists(p):
        return None
    per = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for t, b in zip(d["track_ids"], d["boxes"]):
                per.setdefault(int(t), []).append((fr, b))
    return per


def scan_seg(seg, day, cam, det, embed, args):
    per = load_tracks(seg)
    if per is None:
        return None
    rows = {}
    for t, pts in per.items():
        hs = [p[1][3] - p[1][1] for p in pts]
        rows[t] = float(np.median(hs))
    todo = sorted([t for t, h in rows.items() if h >= args.min_h])
    if args.limit:
        todo = todo[: args.limit]

    need = {}   # frame -> [(tid, box, bh)]
    picked = {}
    for t in todo:
        pts = per[t]
        fs = pick_frames(pts, args.shots)
        picked[t] = fs
        for f in fs:
            p = min(pts, key=lambda q: abs(int(q[0]) - f))
            bx = p[1]
            need.setdefault(int(f), []).append((t, bx, bx[3] - bx[1]))

    vid = slice_of(day, cam)
    if not vid:
        print(f"  [MISS video] {seg}", flush=True)
        return None
    rd = FrameReader(vid)
    shots = {t: [] for t in todo}
    n_read = n_roi = n_face = 0
    t0 = time.time()
    for f in sorted(need):
        img = rd.read(f)
        if img is None:
            continue
        n_read += 1
        H, W = img.shape[:2]
        for tid, (bx1, by1, bx2, by2), bh in need[f]:
            bw = bx2 - bx1
            if bh < 24 or bw < 10:
                continue
            rx0 = int(max(0, bx1 - 0.25 * bw))
            rx1 = int(min(W, bx2 + 0.25 * bw))
            ry0 = int(max(0, by1 - 0.10 * bh))
            ry1 = int(min(H, by1 + 0.55 * bh))
            roi = img[ry0:ry1, rx0:rx1]
            if roi.size == 0 or min(roi.shape[:2]) < 24:
                continue
            n_roi += 1
            faces = det.detect(roi)
            if not len(faces):
                continue
            faces = np.asarray(faces, np.float64).copy()
            faces[:, [0, 2, 4, 6, 8, 10, 12]] += rx0
            faces[:, [1, 3, 5, 7, 9, 11, 13]] += ry0
            cand = None
            for fc in faces:
                fx, fy = (fc[0] + fc[2]) / 2.0, (fc[1] + fc[3]) / 2.0
                if not (bx1 - 0.08 * bw <= fx <= bx2 + 0.08 * bw):
                    continue
                if not (by1 - 0.10 * bh <= fy <= by1 + 0.62 * bh):
                    continue
                if (fc[2] - fc[0]) > 0.95 * bw:
                    continue
                if cand is None or fc[14] > cand[14]:
                    cand = fc
            if cand is None:
                continue
            n_face += 1
            kps = np.asarray(cand[4:14], np.float32).reshape(5, 2)
            fw, fh = float(cand[2] - cand[0]), float(cand[3] - cand[1])
            pd = pd_of(kps)
            ali = align112(img, kps)
            if ali is None:
                continue
            lap = float(cv2.Laplacian(cv2.cvtColor(ali, cv2.COLOR_BGR2GRAY),
                                      cv2.CV_64F).var())
            usable = (pd >= GATE["pd"] and min(fw, fh) >= GATE["w"]
                      and lap >= GATE["lap"])
            q = (min(pd / 40.0, 1.5) * 0.45 + min(min(fw, fh) / 100.0, 1.5) * 0.35
                 + min(lap / 400.0, 1.5) * 0.20)
            shots[tid].append(dict(frame=int(f), bh=round(float(bh), 1),
                                   fw=round(fw, 1), fh=round(fh, 1),
                                   pd=round(pd, 1), lap=round(lap, 1),
                                   q=round(float(q), 4), usable=bool(usable),
                                   _ali=ali))
    rd.close()

    out = {}
    for t in todo:
        ok = sorted([s for s in shots[t] if s["usable"]],
                    key=lambda s: -s["q"])[: args.topk]
        # 诊断：无论可用与否都记下最高分那张的三项质量值，便于校准门限
        top = sorted(shots[t], key=lambda s: -s["q"])[:1]
        probe = ({k: top[0][k] for k in ("frame", "bh", "fw", "fh", "pd", "lap", "q")}
                 if top else None)
        if not ok:
            out[f"{seg}#{t}"] = dict(seg=seg, day=day, cam=cam, tid=int(t),
                                     h_med=round(rows[t], 1),
                                     n_face=len(shots[t]), n_used=0, emb=None,
                                     probe=probe)
            continue
        vecs = [embed(s["_ali"]) for s in ok]
        e = np.mean(vecs, axis=0)
        e /= (np.linalg.norm(e) + 1e-9)
        out[f"{seg}#{t}"] = dict(
            seg=seg, day=day, cam=cam, tid=int(t), h_med=round(rows[t], 1),
            n_face=len(shots[t]), n_used=len(ok),
            pd=ok[0]["pd"], fw=ok[0]["fw"], lap=ok[0]["lap"],
            frames=[s["frame"] for s in ok],
            emb=[round(float(x), 5) for x in e])
    print("  %-28s 轨 %3d → 有脸 %3d（%.0f%%）｜读帧 %5d ROI %5d 检出 %5d｜%.0f s"
          % (seg, len(todo), sum(1 for r in out.values() if r["n_used"]),
             100.0 * sum(1 for r in out.values() if r["n_used"]) / max(1, len(todo)),
             n_read, n_roi, n_face, time.time() - t0), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", default="", help="逗号分隔；空=两天全跑")
    ap.add_argument("--min-h", type=float, default=130.0)
    ap.add_argument("--shots", type=int, default=8)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="每段只跑前 N 条轨（冒烟）")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.segs:
        segs = [s.strip() for s in args.segs.split(",") if s.strip()]
    else:
        segs = [f"{d}_{c}_{s}" for d in DAYS for c in DAY_CAMS[d] for s in SPLITS]

    if args.dry:
        for seg in segs:
            parts = seg.split("_")
            day, cam = parts[0], "_".join(parts[1:-1])
            per = load_tracks(seg)
            if per is None:
                print(f"  {seg:<32} MISS")
                continue
            hs = [float(np.median([b[3] - b[1] for _, b in pts])) for pts in per.values()]
            ok = sum(1 for h in hs if h >= args.min_h)
            has_vid = "有" if slice_of(day, cam) else "缺"
            print(f"  {seg:<32} 轨 {len(hs):>4} 合格 {ok:>4} "
                  f"({100*ok/max(1,len(hs)):>5.1f}%) 视频{has_vid}")
        return

    det = ScrfdDNN(os.path.join(MODELS, "scrfd_2.5g_bnkps.onnx"),
                   thresh=DET_THR, nms=0.4)
    embed = Embedder(os.path.join(MODELS, "w600k_mbf.onnx"))
    print("SCRFD thresh=%.2f｜w600k_mbf dim=%d｜min_h=%.0f shots=%d topk=%d"
          % (DET_THR, embed.dim, args.min_h, args.shots, args.topk), flush=True)

    t_all = time.time()
    for seg in segs:
        parts = seg.split("_")
        day, cam, split = parts[0], "_".join(parts[1:-1]), parts[-1]
        out = scan_seg(seg, day, cam, det, embed, args)
        if out is None:
            continue
        p = os.path.join(args.out, f"{seg}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dict(seg=seg, day=day, cam=cam, split=split,
                           min_h=args.min_h, shots=args.shots, topk=args.topk,
                           det_thr=DET_THR, gates=GATE,
                           embedder="w600k_mbf", tracks=out),
                      f, ensure_ascii=False)
    print("总耗时 %.0f s → %s" % (time.time() - t_all, args.out), flush=True)


if __name__ == "__main__":
    main()
