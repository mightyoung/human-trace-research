#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0 = 人脸抓取 + 质量池（6 路 synced-10min，纯规则）

链路：
  cleanset_v2/{cam}_detections.jsonl（逐帧 5fps 轨迹）
  + 视频帧（ffmpeg 管道，select n%5==0，全分辨率 bgr24）
  → SCRFD-2.5G ONNX（letterbox 640, CPU）人脸检测
  → 人脸↔轨迹关联（脸中心落框内，取最小面积框）
  → 规则质量评分（清晰度/尺寸/瞳距/姿态/光照，五因子）
  → 质量池：per tid / entity / gperson Top-K
  → 产物 analysis/face/: face_shot_p0.json + face_best_p0.json + crops/ + stats

纪律：
  - 只读不写 GIL（face 产物独立存 analysis/face/；GIL 落表留后续）
  - 阈值全部标注为"经验初值，待 val 校准"（项目铁律）
  - 时间换算：t = (frame+100)/25（obs 轴，已验证 68/68 tid 恒定偏移 4.0s）

用法：
  python tools/gil_face_capture.py --cams 316 --limit 2000     # 快速测试
  python tools/gil_face_capture.py --cams 316,305,318,320,323,343
"""
import argparse
import json
import math
import os
import shutil
import sqlite3
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OC = os.path.join(ROOT, "output_crosscam")
FACE_DIR = os.path.join(OC, "analysis", "face")
CROPS_DIR = os.path.join(FACE_DIR, "crops")
FFMPEG = "D:/dev/ffmpeg-9.0.2-full_build/bin/ffmpeg.exe"
SCRFD_MODEL = os.path.join(ROOT, "models", "face", "scrfd_2.5g_bnkps.onnx")
YUNET_MODEL = os.path.join(ROOT, "models", "face", "face_detection_yunet_2023mar.onnx")
GIL_DB = os.path.join(OC, "analysis", "gil", "store_v1", "gil.sqlite")

VIDEOS = {
    "305": "media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4",
    "316": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    "318": "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4",
    "320": "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4",
    "323": "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
    "343": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
}
STRIDE = 5          # jsonl 采样步长（帧）

# 35 分钟长卷：视频与逐帧检测都换一套。检测文件的**优先级必须与 long35_pipeline.jsonl_for
# 完全一致**，否则 track_id 对不上实体图，人脸锚点会挂到错误的轨上。
VIDEOS_LONG35 = {
    "305": "media-resource/multi-cemra/long-35min/305_long35.mp4",
    "316": "media-resource/multi-cemra/long-35min/316_long35.mp4",
    "318": "media-resource/multi-cemra/long-35min/318_long35.mp4",
    "320": "media-resource/multi-cemra/long-35min/320_long35.mp4",
    "323": "media-resource/multi-cemra/long-35min/323_long35.mp4",
    "343": "media-resource/multi-cemra/long-35min/343_long35.mp4",
}
LONG35 = os.path.join(OC, "long35")
PACK = "synced10"


def set_pack(pack):
    global PACK
    PACK = pack


def video_path(cam):
    return os.path.join(ROOT, (VIDEOS_LONG35 if PACK == "long35" else VIDEOS)[cam])


def det_path(cam, src="clean2"):
    if PACK != "long35":
        return (os.path.join(OC, f"{cam}_detections.jsonl") if src == "raw"
                else os.path.join(OC, "cleanset_v2", f"{cam}_detections.jsonl"))
    p_c = os.path.join(LONG35, f"{cam}_long35_detections_c.jsonl")
    if os.path.exists(p_c):
        return p_c
    for name in (f"{cam}_masked_cleaned_v2.jsonl", f"{cam}_masked_cleaned.jsonl",
                 f"{cam}_cleaned.jsonl"):
        q = os.path.join(LONG35, "cleaning", name)
        if os.path.exists(q):
            return q
    return os.path.join(LONG35, f"{cam}_long35_detections.jsonl")
FPS = 25.0

# ---------- 质量评分（经验初值，待 val 校准）----------
W_SHARP, W_SIZE, W_PD, W_POSE, W_LIGHT = 0.30, 0.25, 0.20, 0.15, 0.10
FACE_W_REF = 120.0     # 脸宽满分参考（px，原分辨率）
PD_REF = 60.0          # 瞳距满分参考（px）= GB/T 31488 可识别线
SHARP_REF = 200.0      # Laplacian 方差满分参考
# 硬门（usable）：脸宽 >= 40 且瞳距 >= 20 且 sharp_n >= 0.15
GATE_W, GATE_PD, GATE_SHARP = 40.0, 20.0, 0.15


def load_tracks(cam, limit_frames=None, src="clean2"):
    """jsonl -> {frame: [(tid, [x1,y1,x2,y2]), ...]}

    src: clean2（默认，清洗后轨迹，与历史口径一致）
         raw  （检测头原始输出，仅用于「不等 cleanset_v2 就能起跑」的并行前置）
    """
    path = det_path(cam, src)
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            fr = e["frame"]
            if limit_frames is not None and fr > limit_frames:
                break
            items = []
            for tid, box in zip(e["track_ids"], e["boxes"]):
                items.append((tid, [float(box[0]), float(box[1]), float(box[2]), float(box[3])]))
            d[fr] = items
    return d


def load_faces_jsonl(cam, faces_dir, limit_frames=None):
    """检测头并行人脸支路产物 {cam}_faces.jsonl -> {frame: np.ndarray[N,15]}

    行格式：{"frame":f,"w":W,"h":H,"n":k,"faces":[[x1,y1,x2,y2,kp10...,score], ...]}
    """
    path = os.path.join(faces_dir, f"{cam}_faces.jsonl")
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            fr = e["frame"]
            if limit_frames is not None and fr > limit_frames:
                break
            arr = e.get("faces") or []
            d[fr] = np.asarray(arr, np.float32).reshape(-1, 15) if arr else np.zeros((0, 15), np.float32)
    return d


def frames_cv2(cam, tracks_by_frame, limit=None, stride=STRIDE):
    """cv2 顺序读：grab 全帧推进（快），仅对「有人帧」retrieve（全分辨率 BGR）。
    yield (frame_idx, img)"""
    vp = video_path(cam)
    cap = cv2.VideoCapture(vp)
    fr = 0
    try:
        while True:
            if not cap.grab():
                break
            if fr % stride == 0:
                tracks = tracks_by_frame.get(fr)
                if tracks:
                    ok, img = cap.retrieve()
                    if ok:
                        yield fr, img
            if limit is not None and fr >= limit:
                break
            fr += 1
    finally:
        cap.release()


def frames_seek(cam, frames_needed):
    """顺序 grab 全片推进，只对 frames_needed 里的帧 retrieve（全分辨率 BGR）。

    与 frames_cv2 同构，区别是「要哪些帧」由显式集合给出 —— 用于消费检测头
    并行人脸框时（需要哪些帧由人脸框决定，而非轨迹存在与否）。
    yield (frame_idx, img)
    """
    need = set(int(x) for x in frames_needed)
    if not need:
        return
    hi = max(need)
    vp = video_path(cam)
    cap = cv2.VideoCapture(vp)
    fr = 0
    try:
        while fr <= hi:
            if not cap.grab():
                break
            if fr in need:
                ok, img = cap.retrieve()
                if ok:
                    yield fr, img
            fr += 1
    finally:
        cap.release()


class Scrfd:
    """SCRFD-2.5G ONNX (CPU)。letterbox 640，输出 [x1,y1,x2,y2,kp10,score]（原图坐标）

    ★ 2026-10-16 安全修复（与 tools/scrfd_dnn.py 同步；来源 gpt-v2 包 + 本机实测）：
      旧版固定宽 640、高按比例取 32 倍数 → 竖幅（>约 1.05:1）越界崩溃；且统一除
      det_scale=new_h/h 使 16:9 输入 x 坐标系统性偏 +2.27%。现改为长短边联合 fit +
      x/y 分别还原 + padding 中心过滤。历史产物按旧坐标生成，**不重跑、不混用**。
    """
    def __init__(self, model=SCRFD_MODEL, thresh=0.5, nms=0.4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(4, (os.cpu_count() or 8) // 2)
        self.sess = ort.InferenceSession(model, sess_options=so, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.thresh, self.nms = thresh, nms

    def detect(self, img):
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 \
                or img.shape[2] != 3 or min(img.shape[:2]) < 1:
            raise ValueError('expected nonempty uint8 BGR HWC image')
        h, w = img.shape[:2]
        scale = min(640.0 / w, 640.0 / h)
        nw = max(1, min(640, round(w * scale))); nh = max(1, min(640, round(h * scale)))
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((640, 640, 3), np.uint8); canvas[:nh, :nw] = im
        blob = ((canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.iname: blob})
        grouped = {}
        for raw in outs:
            a = np.asarray(raw)
            if a.ndim == 3 and a.shape[0] == 1: a = a[0]
            if a.ndim != 2 or a.shape[1] not in (1, 4, 10) or not np.isfinite(a).all():
                raise ValueError(f'unsupported/nonfinite SCRFD output {a.shape}')
            key = (a.shape[0], a.shape[1])
            if key in grouped: raise ValueError('duplicate SCRFD output shape')
            grouped[key] = a.astype(np.float32)
        expected = {(2 * (640 // s) ** 2, c) for s in (8, 16, 32) for c in (1, 4, 10)}
        if set(grouped) != expected: raise ValueError('SCRFD export must provide 3 score/box/5KPS scales')
        all_b, all_k, all_s = [], [], []
        for stride in (8, 16, 32):
            g = 640 // stride; n = 2 * g * g
            s = grouped[n, 1].reshape(-1); b = grouped[n, 4]; k = grouped[n, 10]
            yy, xx = np.mgrid[:g, :g]
            centers = np.repeat(np.column_stack((xx.reshape(-1), yy.reshape(-1))) * stride, 2, axis=0).astype(np.float32)
            pos = np.flatnonzero(s > self.thresh)
            if pos.size == 0: continue
            d = b[pos] * stride; c = centers[pos]
            cb = np.column_stack((c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                                  c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]))
            ck = k[pos].reshape(-1, 5, 2) * stride + c[:, None, :]
            real = (c[:, 0] < nw) & (c[:, 1] < nh)
            all_b.append(cb[real]); all_k.append(ck[real].reshape(-1, 10)); all_s.append(s[pos][real])
        if not all_b:
            return np.zeros((0, 15), np.float32)
        B = np.concatenate(all_b); K = np.concatenate(all_k); S = np.concatenate(all_s)
        B[:, [0, 2]] /= (nw / w); B[:, [1, 3]] /= (nh / h)
        K[:, 0::2] /= (nw / w); K[:, 1::2] /= (nh / h)
        B[:, [0, 2]] = np.clip(B[:, [0, 2]], 0, w); B[:, [1, 3]] = np.clip(B[:, [1, 3]], 0, h)
        keep = (B[:, 2] > B[:, 0]) & (B[:, 3] > B[:, 1])
        B, K, S = B[keep], K[keep], S[keep]
        if len(B) == 0:
            return np.zeros((0, 15), np.float32)
        xywh = np.stack([B[:, 0], B[:, 1], B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]], 1)
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), S.tolist(), self.thresh, self.nms)).reshape(-1)
        if len(idx) == 0:
            return np.zeros((0, 15), np.float32)
        out = np.zeros((len(idx), 15), np.float32)
        for j, i in enumerate(idx):
            out[j, :4] = B[i]
            out[j, 4:14] = K[i]
            out[j, 14] = S[i]
        return out


class Yunet:
    """YuNet (OpenCV 原生)。可选 backend。输出同 [x1,y1,x2,y2,kp10,score]"""
    def __init__(self, model=YUNET_MODEL, thresh=0.5, nms=0.3, downscale=2):
        self.det = None
        self.thresh, self.nms, self.ds = thresh, nms, downscale

    def detect(self, img):
        h, w = img.shape[:2]
        if self.ds > 1:
            im = cv2.resize(img, (w // self.ds, h // self.ds))
        else:
            im = img
        hh, ww = im.shape[:2]
        if self.det is None:
            self.det = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (ww, hh), self.thresh, self.nms, 5000)
        else:
            self.det.setInputSize((ww, hh))
        _, faces = self.det.detect(im)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 15), np.float32)
        out = np.zeros((len(faces), 15), np.float32)
        for i, f in enumerate(faces):
            out[i, 0] = f[0] * self.ds; out[i, 1] = f[1] * self.ds
            out[i, 2] = (f[0] + f[2]) * self.ds; out[i, 3] = (f[1] + f[3]) * self.ds
            for j in range(5):
                out[i, 4 + j * 2] = f[4 + j * 2] * self.ds
                out[i, 5 + j * 2] = f[5 + j * 2] * self.ds
            out[i, 14] = f[14]
        return out


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def score_face(img, det):
    """五因子评分。det=[x1,y1,x2,y2,kp10,score]（原分辨率）。返回 dict"""
    x1, y1, x2, y2 = det[:4]
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    fw, fh = x2i - x1i, y2i - y1i
    if fw < 6 or fh < 6:
        return None
    roi = img[y1i:y2i, x1i:x2i]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mu, sd = float(gray.mean()), float(gray.std())
    kps = det[4:14].reshape(5, 2)
    pd = float(np.linalg.norm(kps[1] - kps[0]))
    eye_mid = (kps[0] + kps[1]) / 2.0
    mouth_mid = (kps[3] + kps[4]) / 2.0
    # 姿态近似（近似公式，P0 粗筛用；精确 yaw/pitch 待后续）
    yaw_off = abs(kps[2][0] - eye_mid[0]) / max(pd, 1.0)
    yaw_s = clamp(1 - yaw_off / 0.6)
    denom = max(mouth_mid[1] - eye_mid[1], 1.0)
    pitch_r = (kps[2][1] - eye_mid[1]) / denom
    pitch_s = 1.0 if 0.35 <= pitch_r <= 0.85 else clamp(1 - abs(pitch_r - 0.6) / 0.6)
    roll = abs(np.arctan2(kps[1][1] - kps[0][1], kps[1][0] - kps[0][0]))
    roll_s = clamp(1 - roll / (np.pi / 6))
    pose_s = clamp(0.5 * yaw_s + 0.3 * pitch_s + 0.2 * roll_s)
    # 光照
    light_mu_s = 1.0 if 70 <= mu <= 180 else clamp(1 - abs(mu - 125) / 125)
    light_s = 0.6 * light_mu_s + 0.4 * clamp(sd / 45.0)
    # 归一
    sharp_n = clamp(lap / SHARP_REF)
    size_n = clamp(fw / FACE_W_REF)
    pd_n = clamp(pd / PD_REF)
    q = (W_SHARP * sharp_n + W_SIZE * size_n + W_PD * pd_n
         + W_POSE * pose_s + W_LIGHT * light_s)
    usable = (fw >= GATE_W) and (pd >= GATE_PD) and (sharp_n >= GATE_SHARP)
    return dict(w=fw, h=fh, pd=round(pd, 1), lap=round(lap, 1), mu=round(mu, 1),
                sd=round(sd, 1), q=round(q, 4), q_parts=dict(
                    sharp=round(sharp_n, 3), size=round(size_n, 3), pd=round(pd_n, 3),
                    pose=round(pose_s, 3), light=round(light_s, 3),
                    yaw_off=round(float(yaw_off), 3),                 pitch_r=round(float(pitch_r), 3)),
                usable=bool(usable), det_score=round(float(det[14]), 3))


# ---- 关联 v2.1（头部区域扩展 + 最近框兜底；两档均含「脸不跨框顶」条件）：经验初值，待 val 校准 ----
HEAD_PAD_X, HEAD_PAD_TOP, HEAD_PAD_BOT = 0.12, 0.35, 0.08
HEAD_MIN_OVERLAP = 1.0    # 脸底边须低于框顶 ≥ 1.0×脸高（防「无框行人的脸被邻框兜底」误配）
NEAREST_D_MAX = 0.50      # 脸中心↔框中心距离 / 框对角线
NEAREST_RATIO = 1.35      # 次近 / 最近 ≥ 该比值才接受（歧义守卫）


def associate_track(face_box, tracks):
    """返回 (tid|None, reason)。三级：
    1) in_exact：脸中心落原框内 → 取最小面积（v1 旧规则，不回退）
    2) in_head ：脸中心落头部扩展框（侧 12% / 上 35% / 下 8%）+ 不跨框顶 → 头锚点距离最近
    3) nearest ：兜底：x 落框横向范围（±12%）、不跨框顶、归一距离 ≤ 0.50、歧义守卫通过
    """
    if not tracks:
        return None, "no_tracks"
    fx1, fy1, fx2, fy2 = face_box
    cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
    fh = max(fy2 - fy1, 1.0)
    exact = []
    head = []
    for tid, b in tracks:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
            exact.append((bw * bh, tid))
        if (fy2 - b[1]) / fh < HEAD_MIN_OVERLAP:
            continue
        ex1, ex2 = b[0] - HEAD_PAD_X * bw, b[2] + HEAD_PAD_X * bw
        ey1, ey2 = b[1] - HEAD_PAD_TOP * bh, b[3] + HEAD_PAD_BOT * bh
        if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
            ax, ay = (b[0] + b[2]) / 2, b[1] + 0.12 * bh   # 头锚点≈框顶中央偏下
            d = math.hypot(cx - ax, cy - ay) / max(bh, 1.0)
            head.append((d, tid))
    if exact:
        exact.sort()
        return exact[0][1], "in_exact"
    if head:
        head.sort()
        return head[0][1], "in_head"
    ds = []
    for tid, b in tracks:
        bw, bh = b[2] - b[0], b[3] - b[1]
        if not (b[0] - HEAD_PAD_X * bw <= cx <= b[2] + HEAD_PAD_X * bw):
            continue
        if (fy2 - b[1]) / fh < HEAD_MIN_OVERLAP:
            continue
        d = math.hypot(cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2) / max(math.hypot(bw, bh), 1.0)
        ds.append((d, tid))
    ds.sort()
    if ds and ds[0][0] <= NEAREST_D_MAX and (len(ds) == 1 or ds[1][0] >= NEAREST_RATIO * ds[0][0] + 0.08):
        return ds[0][1], "nearest"
    return None, "none"


def load_gil_maps(require_time=False):
    """tid -> (obs_id, entity_id, gperson_id)；来自 GIL sqlite。

    ★ GIL 的 obs 是**另一轮运行**的产物，轨号 (cam, tid) 会与本轮重号。
    实测 long35 上 546 条 obs 里有 76 条（14%）时间完全不重合 —— 那是别的轨，
    直接按 tid 查表会把 gperson 挂到错的人身上。require_time=True 时用时间重合
    做守门（交叠 / 较短的 > 50% 才算同一条轨）。
    """
    con = sqlite3.connect(GIL_DB)
    cur = con.cursor()
    obs = {}
    for r in cur.execute("select id, cam, tid, frag_id, entity_id, t_first, t_last from obs"):
        obs[(r[1], r[2])] = dict(obs_id=r[0], frag_id=r[3], entity_id=r[4],
                                 gperson_id=None, _t0=r[5], _t1=r[6])
    for r in cur.execute("select node_id, gperson_id from membership where node_kind='obs'"):
        oid, gp = r
        for k, v in obs.items():
            if v["obs_id"] == oid:
                v["gperson_id"] = gp
                break
    con.close()

    if require_time:
        import glob as _glob
        span = {}
        for jp in _glob.glob(os.path.join(ROOT, "webapp/backend/data/tracks", "*.json")):
            try:
                j = json.load(open(jp, encoding="utf-8"))
            except Exception:
                continue
            cam, off, fps = j.get("cam"), j.get("offset"), j.get("fps") or 25.0
            if not cam:
                continue
            for tid, rows in (j.get("tracks") or {}).items():
                if rows:
                    span[(cam, int(tid))] = (off + rows[0][0] / fps, off + rows[-1][0] / fps)
        kept, dropped = {}, 0
        for k, v in obs.items():
            sp = span.get(k)
            gt = (v.get("_t0"), v.get("_t1"))
            if sp is None or gt[0] is None:
                kept[k] = v
                continue
            inter = min(sp[1], gt[1]) - max(sp[0], gt[0])
            short = min(sp[1] - sp[0], gt[1] - gt[0])
            if short > 0.2 and inter / short > 0.5:
                kept[k] = v
            else:
                dropped += 1
        if dropped:
            print(f"[gil] 时间守门剔除 {dropped} 条重号 obs（共 {len(obs)}）", flush=True)
        obs = kept
    for v in obs.values():
        v.pop("_t0", None)
        v.pop("_t1", None)
    return obs


def build_best_pools(all_shots, cams, topk=3):
    """质量池聚合：per tid / entity / gperson Top-K（usable 优先、q 降序）。
    返回 (best_tid, best_ent, best_gp, ent_of_tid)。"""
    by_tid = defaultdict(list)
    for s in all_shots:
        if s["tid"] is not None:
            by_tid[(s["cam"], s["tid"])].append(s)
    best_tid = {}
    for k, lst in by_tid.items():
        lst.sort(key=lambda s: (s["usable"], s["q"]), reverse=True)
        best_tid[f"{k[0]}|{k[1]}"] = lst[:topk]
    # entity 映射（stitch_clean2）
    ent_of_tid = {}
    for cam in cams:
        p = os.path.join(OC, "analysis", "stitch_clean2", f"net_tracks_{cam}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                nt = json.load(f)
            for e in nt["entities"]:
                for t in e["tracks"]:
                    ent_of_tid[(cam, t)] = e["eid"]
    by_ent = defaultdict(list)
    for s in all_shots:
        eid = ent_of_tid.get((s["cam"], s["tid"])) if s["tid"] is not None else None
        if eid:
            by_ent[eid].append(s)
    best_ent = {}
    for k, lst in by_ent.items():
        lst.sort(key=lambda s: (s["usable"], s["q"]), reverse=True)
        best_ent[k] = lst[:topk]
    # gperson 映射
    by_gp = defaultdict(list)
    for s in all_shots:
        if s["gperson_id"] is not None:
            by_gp[s["gperson_id"]].append(s)
    best_gp = {}
    for k, lst in by_gp.items():
        lst.sort(key=lambda s: (s["usable"], s["q"]), reverse=True)
        best_gp[k] = lst[:topk]
    return best_tid, best_ent, best_gp, ent_of_tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="316")
    ap.add_argument("--detector", default="scrfd", choices=["scrfd", "yunet"])
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 jsonl 帧（快速测试）")
    ap.add_argument("--no-crops", action="store_true")
    ap.add_argument("--out-tag", default="p0")
    ap.add_argument("--pack", default="synced10", choices=["synced10", "long35"],
                    help="素材包：long35 = 35 分钟长卷（视频与检测都换一套）")
    ap.add_argument("--tracks-source", default="clean2", choices=["clean2", "raw"],
                    help="关联用轨迹来源：clean2（默认，清洗后）；"
                         "raw = 检测头原始输出，用于「不等 cleanset_v2 就能起跑」")
    ap.add_argument("--faces-dir", default="",
                    help="★ 消费检测头并行人脸支路产物（{cam}_faces.jsonl 所在目录）："
                         "给了就跳过本地 SCRFD 与整片解码，只按需 seek 读帧")
    args = ap.parse_args()

    os.makedirs(FACE_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)
    det = None if args.faces_dir else (Scrfd() if args.detector == "scrfd" else Yunet())
    gil = load_gil_maps()
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    set_pack(args.pack)
    all_shots = []
    crop_map = {}
    stats = {}

    for cam in cams:
        t0 = time.time()
        tracks_by_frame = load_tracks(cam, args.limit, src=args.tracks_source)
        faces_by_frame = (load_faces_jsonl(cam, args.faces_dir, args.limit)
                          if args.faces_dir else None)
        n_det_frames = 0
        n_faces = 0
        n_assoc = 0
        buckets = defaultdict(int)
        assoc_reasons = defaultdict(int)
        best_crop = {}   # (cam,tid) -> {q, img, fr} 内存缓存，跑完即写盘
        if faces_by_frame is not None:
            # ★ 只 retrieve「有轨迹且有人脸框」的帧（顺序 grab 推进，零随机 seek）
            need = sorted(set(tracks_by_frame) & set(faces_by_frame))
            src_iter = frames_seek(cam, need)
        else:
            src_iter = frames_cv2(cam, tracks_by_frame, args.limit)
        for fr, img in src_iter:
            tracks = tracks_by_frame.get(fr, [])
            n_det_frames += 1
            faces = faces_by_frame[fr] if faces_by_frame is not None else det.detect(img)
            for d in faces:
                s = score_face(img, d)
                if s is None:
                    continue
                cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                tid, areason = associate_track(d[:4], tracks)
                if tid is not None:
                    n_assoc += 1
                assoc_reasons[areason] += 1
                key = (cam, tid) if tid is not None else None
                g = gil.get(key, {}) if key else {}
                n_faces += 1
                if s["w"] < 40: buckets["<40"] += 1
                elif s["w"] < 80: buckets["40-80"] += 1
                elif s["w"] < 120: buckets["80-120"] += 1
                else: buckets["120+"] += 1
                all_shots.append(dict(
                    cam=cam, frame=fr, t=round((fr + 100) / FPS, 2), tid=tid,
                    assoc_reason=areason,
                    face=[round(float(v), 1) for v in d[:4]],
                    kps=[round(float(v), 1) for v in d[4:14]],
                    obs_id=g.get("obs_id"), entity_id=g.get("entity_id"),
                    gperson_id=g.get("gperson_id"), **s))
                if tid is not None and s["usable"]:
                    kk = (cam, tid)
                    old = best_crop.get(kk)
                    if old is None or s["q"] > old["q"]:
                        x1, y1, x2, y2 = d[:4]
                        mw, mh = (x2 - x1) * 0.35, (y2 - y1) * 0.45
                        a = max(0, int(x1 - mw)); b = max(0, int(y1 - mh))
                        c2 = min(img.shape[1], int(x2 + mw)); d2 = min(img.shape[0], int(y2 + mh))
                        if c2 > a and d2 > b:
                            best_crop[kk] = dict(q=s["q"], img=img[b:d2, a:c2].copy(), fr=fr)
            if fr % 1000 == 0:
                print(f"  [{cam}] fr={fr} frames_det={n_det_frames} faces={n_faces} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if not args.no_crops:
            for (c, tid), v in best_crop.items():
                fn = f"tid_{c}_{tid}_f{v['fr']}_q{v['q']:.2f}.jpg"
                if cv2.imwrite(os.path.join(CROPS_DIR, fn), v["img"], [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    crop_map[f"{c}_{tid}"] = fn
            best_crop.clear()
        stats[cam] = dict(total_jsonl_frames=len(tracks_by_frame), det_frames=n_det_frames,
                          faces=n_faces, assoc=n_assoc,
                          assoc_rate=round(n_assoc / max(n_faces, 1), 3),
                          assoc_reasons=dict(assoc_reasons),
                          buckets=dict(buckets), t_sec=round(time.time() - t0, 1))
        print(f"[{cam}] done: frames={n_det_frames} faces={n_faces} assoc={n_assoc} "
              f"buckets={dict(buckets)} {time.time()-t0:.0f}s", flush=True)

    # ---- 质量池聚合 ----
    best_tid, best_ent, best_gp, ent_of_tid = build_best_pools(all_shots, cams)

    # gperson -> label
    con = sqlite3.connect(GIL_DB)
    gp_label = {r[0]: r[1] for r in con.execute("select id, label from gperson")}
    con.close()

    # ---- 存图：entity/gperson 图 = 复制其最佳 tid 的 crop ----
    if not args.no_crops:
        for k, lst in best_ent.items():
            s0 = lst[0]
            src = crop_map.get(f'{s0["cam"]}_{s0["tid"]}') if s0.get("tid") is not None else None
            if src:
                shutil.copy(os.path.join(CROPS_DIR, src),
                            os.path.join(CROPS_DIR, f"ent_{k}.jpg"))
        for k, lst in best_gp.items():
            s0 = lst[0]
            src = crop_map.get(f'{s0["cam"]}_{s0["tid"]}') if s0.get("tid") is not None else None
            if src:
                shutil.copy(os.path.join(CROPS_DIR, src),
                            os.path.join(CROPS_DIR, f"gp_{gp_label.get(k, k)}.jpg"))

    # ---- 输出 ----
    out = dict(meta=dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), detector=args.detector,
        stride=STRIDE, cams=cams, n_shots=len(all_shots),
        tracks_source=args.tracks_source,
        faces_dir=(os.path.basename(args.faces_dir.rstrip("/")) if args.faces_dir else None),
        face_backend=("head-parallel(cv2.dnn)" if args.faces_dir else "local-scrfd"),
        scoring=dict(w=[W_SHARP, W_SIZE, W_PD, W_POSE, W_LIGHT],
                     refs=dict(face_w=FACE_W_REF, pd=PD_REF, sharp=SHARP_REF),
                     gates=dict(w=GATE_W, pd=GATE_PD, sharp=GATE_SHARP),
                     note="经验初值，待 val 校准")),
        stats=stats, shots=all_shots)
    with open(os.path.join(FACE_DIR, f"face_shot_{args.out_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    best = dict(meta=out["meta"], per_tid=best_tid, per_entity=best_ent,
                per_gperson=best_gp, gp_label=gp_label, crop_map=crop_map)
    with open(os.path.join(FACE_DIR, f"face_best_{args.out_tag}.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(f"shots={len(all_shots)} tids={len(best_tid)} entities={len(best_ent)} "
          f"gpersons={len(best_gp)} -> {FACE_DIR}")


if __name__ == "__main__":
    main()
