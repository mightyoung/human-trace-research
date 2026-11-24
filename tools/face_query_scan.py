#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 ReID 库的轨迹，在**原生视频**上采集人脸证据（供自动人脸触发使用）。

数据来源（为什么是这三个）
--------------------------
1. `webapp/backend/data/reid/store_meta.json` —— 库内 539 条轨（cam/tid/t0/t1/h_med）。
2. `webapp/backend/data/tracks/{cam}.json` —— **逐帧框**（stride 5），也是 `f_rep` 的
   出处。库里的 `u/v` 是**中位位置**，人走动后与具体某一帧对不上（实测 323#2 的
   (u,v) 落在空走廊上，人已经走到画面左侧），所以定位必须用逐帧框。
3. `media-resource/multi-cemra/synced-10min/{cam}_*.mp4` —— 原生视频
   （305/316/320/343 为 2560×1440；318/323 为 1920×1080）。

★ 三条必须遵守的口径
  - **不复用 `face_best_p0.json`**：那份用 `cleanset_v2` 轨号，本库用 `yolo8_dd`
    轨号。实测 115 条同号轨里 63 条时间对不上（轨号重号 + 各机位 t 轴偏移不同），
    属于项目铁律「按轨号 join 必须带时间校验」的教科书案例。
  - **必须按框裁 ROI 再检测**：整帧 letterbox 到 640 后 60 px 的脸只剩 15–20 px，
    SCRFD 在 1080p 整帧上实测 112 帧**零检出**。
  - **按框高挑帧**，不是等间隔抽帧：面部像素预算 ≈ 0.054×框高，抽「框最高的帧」
    把单帧可用率从 6% 抬到 47%（见 mainline-face-anchor-2026-11-10 §0.5）。

用法
----
    python tools/face_query_scan.py --min-h 150 --shots 8 --topk 3
    python tools/face_query_scan.py --cams 305,343 --min-h 250 --dry
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from scrfd_dnn import ScrfdDNN  # noqa: E402

FPS = 25.0
DATA = os.path.join(ROOT, "webapp", "backend", "data")

# 两个数据包的口径完全不同，别混：
#   store   10 分钟同步包，539 轨，key = "date|cam|tid"，视频 synced-10min/*/*.mp4
#   long35  35 分钟长卷，1574 轨（1894 原始），key = "cam:tid"，视频 long-35min/{cam}_long35.mp4
PACKS = {
    "store": dict(
        store=os.path.join(DATA, "reid", "store_meta.json"),
        tracks_dir=os.path.join(DATA, "tracks"),
        vid_glob=os.path.join(ROOT, "media-resource", "multi-cemra", "synced-10min", "*", "*.mp4"),
        out=os.path.join(DATA, "reid", "face_query.json"),
        crops=os.path.join(DATA, "reid", "face_q"),
    ),
    "long35": dict(
        store=None,
        tracks_dir=os.path.join(ROOT, "webapp", "backend", "data_long35", "tracks"),
        vid_glob=os.path.join(ROOT, "media-resource", "multi-cemra", "long-35min", "*_long35.mp4"),
        out=os.path.join(DATA, "reid", "face_query_long35.json"),
        crops=os.path.join(DATA, "reid", "face_q_l35"),
    ),
}
SCRFD_MODEL = os.path.join(ROOT, "models", "face", "scrfd_2.5g_bnkps.onnx")

TPL112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

# 质量门：P0 口径的 pd≥20 在本库（框高中位 142 px）过不去，pd 放宽到 15；
# 「脸宽 ≥40 / 清晰度 ≥40」两道不放宽 —— 宁可少，不要脏模板。
GATE = dict(pd=15.0, w=40.0, lap=40.0)
DET_THR = 0.40


def align112(img, kps5):
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), TPL112, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def pd_of(kps):
    """瞳距 = 两眼中点距离（kps 顺序：左眼、右眼、鼻、左嘴角、右嘴角）。"""
    d = np.asarray(kps[0], np.float64) - np.asarray(kps[1], np.float64)
    return float(np.sqrt((d * d).sum()))


class Embedder:
    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(4, (os.cpu_count() or 8) // 2)
        self.sess = ort.InferenceSession(path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.dim = int(self.sess.get_outputs()[0].shape[-1])

    def __call__(self, aligned_bgr):
        blob = ((aligned_bgr[:, :, ::-1].astype(np.float32) - 127.5) / 127.5)
        v = self.sess.run(None, {self.iname: blob.transpose(2, 0, 1)[None]})[0][0]
        v = v.astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-9)


class FrameReader:
    """按帧号跳读；`cur` = 下一次 read 会拿到的帧号。

    ★ cv2 的 seek 在 1440p H.264 上约 20–50 ms，±40 帧内顺序读更省。
    ★ `grab()` 只返回 bool（不是 (ok, frame)），这里踩过一次。
    """

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
    """按框高挑帧：先取最高的，再按时间均匀补足；去重后按帧号排序。"""
    if not pts:
        return []
    by_h = sorted(pts, key=lambda p: -(p[4] - p[2]))
    chosen, seen = [], set()

    def push(p):
        f = int(p[0])
        if any(abs(f - g) < 5 for g in seen):   # stride=5，同一格不重复
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="store", choices=sorted(PACKS),
                    help="store=10min 包 / long35=35min 长卷（两套轨号不通用）")
    ap.add_argument("--min-h", type=float, default=150.0, help="只采框高中位 ≥ 此值的轨")
    ap.add_argument("--shots", type=int, default=8, help="每轨挑几帧")
    ap.add_argument("--topk", type=int, default=3, help="每轨保留几张可用脸")
    ap.add_argument("--cams", default="", help="逗号分隔，限定机位")
    ap.add_argument("--keys", default="", help="定向补采：只采这些轨，如 305:173,320:45")
    ap.add_argument("--emb", default="mbf", choices=["mbf", "r50"])
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    pack = PACKS[args.pack]
    vids = {os.path.basename(p)[:3]: p for p in glob.glob(pack["vid_glob"])}
    vids = {c: p for c, p in vids.items() if c.isdigit()}
    cams = [c.strip() for c in args.cams.split(",") if c.strip()] or sorted(vids)

    tj = {}
    for cam in cams:
        p = os.path.join(pack["tracks_dir"], f"{cam}.json")
        if os.path.exists(p):
            tj[cam] = json.load(open(p, encoding="utf-8"))

    if pack["store"]:
        tracks = json.load(open(pack["store"], encoding="utf-8"))["tracks"]
    else:
        # 长卷没有 store_meta，轨清单从逐帧框现算（h_med = 中位框高，口径同 store）
        tracks = []
        for cam, d in tj.items():
            for tid, pts in d["tracks"].items():
                if not pts:
                    continue
                hs = [p[4] - p[2] for p in pts]
                tracks.append(dict(cam=cam, tid=int(tid), h_med=float(np.median(hs))))

    if args.keys:
        want = {(k.split(":")[0], int(k.split(":")[1]))
                for k in args.keys.split(",") if k.strip()}
        todo = [t for t in tracks if (t["cam"], int(t["tid"])) in want]
    else:
        todo = [t for t in tracks if t["cam"] in cams and float(t["h_med"]) >= args.min_h]
    print("包 %s：库内 %d 轨；进入采集 %d 轨" % (args.pack, len(tracks), len(todo)))
    print("按机位:", {c: sum(1 for t in todo if t["cam"] == c) for c in cams})
    print("逐帧框轨数:", {c: len(tj[c]["tracks"]) for c in cams if c in tj})
    if args.dry:
        return

    det = ScrfdDNN(SCRFD_MODEL, thresh=DET_THR, nms=0.4)
    mname = "w600k_mbf.onnx" if args.emb == "mbf" else "w600k_r50.onnx"
    mpath = os.path.join(ROOT, "models", "face", mname)
    embed = Embedder(mpath)
    os.makedirs(pack["crops"], exist_ok=True)
    print("嵌入模型 %s dim=%d，检测阈值 %.2f" % (mname, embed.dim, DET_THR))

    readers = {c: FrameReader(vids[c]) for c in cams if c in vids}
    recs, t_start = {}, time.time()
    n_read = n_det = n_linked = 0

    for cam in cams:
        rd = readers.get(cam)
        if rd is None or cam not in tj:
            continue
        for t in [x for x in todo if x["cam"] == cam]:
            tid = int(t["tid"])
            pts = tj[cam]["tracks"].get(str(tid)) or []
            if not pts:
                continue
            shots = []
            for f in pick_frames(pts, args.shots):
                img = rd.read(f)
                if img is None:
                    continue
                n_read += 1
                H, W = img.shape[:2]
                p = min(pts, key=lambda q: abs(int(q[0]) - f))
                bx1, by1, bx2, by2 = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
                bh, bw = by2 - by1, bx2 - bx1
                if bh < 24 or bw < 10:
                    continue
                # 头部 ROI = 框的上 55%，左右各放 25%
                rx0 = int(max(0, bx1 - 0.25 * bw))
                rx1 = int(min(W, bx2 + 0.25 * bw))
                ry0 = int(max(0, by1 - 0.10 * bh))
                ry1 = int(min(H, by1 + 0.55 * bh))
                roi = img[ry0:ry1, rx0:rx1]
                if roi.size == 0 or min(roi.shape[:2]) < 24:
                    continue
                faces = det.detect(roi)
                n_det += len(faces)
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
                    if (fc[2] - fc[0]) > 0.95 * bw:      # 比人还宽 → 不是脸
                        continue
                    if cand is None or fc[14] > cand[14]:
                        cand = fc
                if cand is None:
                    continue
                kps = np.asarray(cand[4:14], np.float32).reshape(5, 2)
                fw, fh = float(cand[2] - cand[0]), float(cand[3] - cand[1])
                pd = pd_of(kps)
                ali = align112(img, kps)
                if ali is None:
                    continue
                lap = float(cv2.Laplacian(cv2.cvtColor(ali, cv2.COLOR_BGR2GRAY),
                                          cv2.CV_64F).var())
                usable = (pd >= GATE["pd"] and min(fw, fh) >= GATE["w"] and lap >= GATE["lap"])
                q = (min(pd / 40.0, 1.5) * 0.45 + min(min(fw, fh) / 100.0, 1.5) * 0.35
                     + min(lap / 400.0, 1.5) * 0.20)
                n_linked += 1
                shots.append(dict(frame=int(f), bh=round(bh, 1), score=round(float(cand[14]), 3),
                                  fw=round(fw, 1), fh=round(fh, 1), pd=round(pd, 1),
                                  lap=round(lap, 1), q=round(q, 4), usable=bool(usable),
                                  _ali=ali))

            ok = sorted([s for s in shots if s["usable"]], key=lambda s: -s["q"])[: args.topk]
            key = "%s:%d" % (cam, tid)
            clean = lambda ss: [{k: v for k, v in s.items() if not k.startswith("_")} for s in ss]
            if not ok:
                recs[key] = dict(cam=cam, tid=tid, h_med=round(float(t["h_med"]), 1),
                                 n_face=len(shots), n_used=0, emb=None, shots=clean(shots[:6]))
                print("  %s#%-4d h=%4.0f 关联 %d 张，无可用"
                      % (cam, tid, t["h_med"], len(shots)), flush=True)
                continue
            vecs = [embed(s["_ali"]) for s in ok]
            e = np.mean(vecs, axis=0)
            e /= (np.linalg.norm(e) + 1e-9)
            ck = "%s_%d" % (cam, tid)
            for i, s in enumerate(ok):
                fn = "%s_%d.jpg" % (ck, i)
                cv2.imwrite(os.path.join(pack["crops"], fn), cv2.resize(s["_ali"], (128, 128)))
                s["crop"] = fn
            recs[key] = dict(cam=cam, tid=tid, h_med=round(float(t["h_med"]), 1),
                             n_face=len(shots), n_used=len(ok),
                             emb=[round(float(x), 5) for x in e], shots=clean(ok))
            print("  %s#%-4d h=%4.0f 关联 %d → 用 %d（pd %.0f / lap %.0f）"
                  % (cam, tid, t["h_med"], len(shots), len(ok), ok[0]["pd"], ok[0]["lap"]),
                  flush=True)

    for r in readers.values():
        r.close()

    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                pack=args.pack,
                store=os.path.relpath(pack["store"] or pack["tracks_dir"], ROOT).replace("\\", "/"),
                boxes_from="%s/{cam}.json (stride 5, 逐帧框)"
                           % os.path.relpath(pack["tracks_dir"], ROOT).replace("\\", "/"),
                detector="scrfd_2.5g_bnkps", det_thr=DET_THR,
                embedder=mname, emb_dim=embed.dim,
                fps=FPS, min_h=args.min_h, shots=args.shots, topk=args.topk,
                cams=cams, gates=GATE, frames_read=n_read,
                faces_detected=n_det, faces_linked=n_linked,
                n_track=len(todo), n_with_face=sum(1 for r in recs.values() if r["n_used"]),
                elapsed_s=round(time.time() - t_start, 1),
                note=("按本库轨迹在原生视频上重采。帧按框高挑（面部像素预算≈0.054×框高），"
                      "ROI 取框上 55%，脸须落在本轨框内。"))
    json.dump(dict(meta=meta, tracks=recs), open(pack["out"], "w", encoding="utf-8"),
              ensure_ascii=False)
    n_ok = meta["n_with_face"]
    print("\n写盘 %s" % os.path.relpath(pack["out"], ROOT))
    print("读 %d 帧 / 检出 %d 张脸 / 关联进轨 %d 张 → %d/%d 轨有可用脸（%.1f%%），%.1f s"
          % (n_read, n_det, n_linked, n_ok, meta["n_track"],
             100 * n_ok / max(1, meta["n_track"]), meta["elapsed_s"]))


if __name__ == "__main__":
    main()
