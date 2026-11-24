#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO26s-seg + 门控 ByteTrack（可选 OSNet 外观否决）——yolo_track_det.py 的可替换版本 v2。

为什么要 v2（"框交界时转移目标"的根因，见研究文档 §串人分析）：
  1. 旧 track_buffer=60 按"处理帧"计数：stride10 的 gallery 实际是 24s、stride5 的 query 是 12s，
     不是注释写的 3s。丢失的轨迹在原地"挂"很久，路过的人一靠近就被接上。
  2. 旧 match_thresh 0.85 + fuse_score：融合相似度只要 >0.15 就能配上，交界处几乎必配。
  3. high/new 阈值 0.10/0.12：ByteTrack 的高低分两阶段被抹平，低质量的半身/遮挡框参与第一阶段。
  4. 头/身重复框在跟踪前没去重（旧 --dedup-mask 是跟踪之后才删框，id 已经被扰动）。
  5. 跟踪只看 IoU，不看外观：两人交错时谁的框离预测近就给谁。

v2 做法（每一项都可单独开关，便于消融）：
  - 去重放在跟踪之前：掩膜包含率 >= --dedup-mask（默认 0.9）的小框删掉（头框并入身体框）。
  - --buffer-sec 按真实秒设置，自动换算成处理帧：ceil(buffer_sec * fps / stride)。
  - 阈值全部可调；默认采用推荐值 high 0.35 / low 0.10 / new 0.45 / match 0.80。
  - --reid osnet：每个检测框算 OSNet 特征，用作"否决门"——外观余弦 < --veto-sim 的配对禁止，
    哪怕 IoU 很高。注意 Ultralytics BoT-SORT 的做法是 min(IoU 代价, 外观代价)，
    外观只能让匹配"更容易"、不能否决，所以它防不住交界转移；这里改成否决门。
  - 交界冻结：当前检测框与其他框 IoU >= --contest-iou 时，不用它更新轨迹外观模板（防止模板被污染）。
  - --det-cache：检测 + 去重 + 特征缓存到 npz。参数网格只需跑一次 YOLO（本机 CPU 3.3 f/s，最贵的就是这步）。
    stride5 的缓存可以直接服务 stride10 的跟踪。

输出（与旧版逐字段一致，可直接接 make_ab_jsonl.py / encode_1012_osnet_seq.py）：
  {out_dir}/{seg}/{seg}_detections.jsonl   每行 {"frame", "track_ids", "boxes", "scores"}
  另写 {seg}_detections.meta.json（配置 + 统计），不影响下游。
  boxes 默认输出原始检测框（--box-source det），旧版输出的是卡尔曼平滑后的跟踪框；
  需要严格对齐旧版时用 --box-source track。

用法（Windows，dml_torch 环境；检测固定 CPU，OSNet 可 dml）：
  # 1) 冒烟
  python tools/yolo_track_det_v2.py --seg 323_train_510 --max-frames 40 --tag smoke
  # 2) 先建缓存（stride 5，含 OSNet 特征），之后各配置只跑跟踪
  python tools/yolo_track_det_v2.py --seg 323_train_510 --stride 5 --reid osnet \
      --det-cache output_crosscam_local/det_cache/323_train_510_s5.npz --tag v2_s10_b2_m80 --track-stride 10
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = ROOT / "models" / "Ultralytics--YOLO26" / "snapshots" / "master" / "yolo26s-seg.pt"
DEFAULT_VID_DIR = ROOT / "media-resource/1012/segments"
CACHE_VERSION = 2


# ----------------------------------------------------------------------------------------------
# 几何工具
# ----------------------------------------------------------------------------------------------
def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ua = aa[:, None] + ab[None, :] - inter
    return np.where(ua > 0, inter / np.maximum(ua, 1e-9), 0.0)


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


# ----------------------------------------------------------------------------------------------
# 卡尔曼（ByteTrack 标准 XYAH 匀速模型；自带实现，不依赖 ultralytics 内部 API，避免版本差异）
# ----------------------------------------------------------------------------------------------
class KalmanXYAH:
    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)
        self.wp, self.wv = 1.0 / 20, 1.0 / 160

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self.wp * h, 2 * self.wp * h, 1e-2, 2 * self.wp * h,
               10 * self.wv * h, 10 * self.wv * h, 1e-5, 10 * self.wv * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.wp * h, self.wp * h, 1e-2, self.wp * h,
               self.wv * h, self.wv * h, 1e-5, self.wv * h]
        Q = np.diag(np.square(std))
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, z):
        h = mean[3]
        R = np.diag(np.square([self.wp * h, self.wp * h, 1e-1, self.wp * h]))
        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (z - self.H @ mean)
        cov = cov - K @ S @ K.T
        return mean, cov


# ----------------------------------------------------------------------------------------------
# 门控 ByteTrack
# ----------------------------------------------------------------------------------------------
TRACKED, LOST, REMOVED = 0, 1, 2


class Track:
    _kf = KalmanXYAH()

    def __init__(self, box, score, feat, step):
        self.mean, self.cov = self._kf.initiate(xyxy_to_xyah(box))
        self.score = score
        self.feat = None if feat is None else feat.copy()
        self.state = TRACKED
        self.activated = False
        self.tid = -1
        self.start_step = step
        self.last_step = step
        self.hits = 1
        self.det_idx = -1

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)

    def predict(self):
        if self.state != TRACKED:
            self.mean[7] = 0.0  # 丢失期间不再外推高度速度（与 ByteTrack 一致）
        self.mean, self.cov = self._kf.predict(self.mean, self.cov)

    def update(self, box, score, feat, step, freeze_feat, alpha):
        self.mean, self.cov = self._kf.update(self.mean, self.cov, xyxy_to_xyah(box))
        self.score = score
        self.last_step = step
        self.hits += 1
        self.state = TRACKED
        if feat is not None and not freeze_feat:
            if self.feat is None:
                self.feat = feat.copy()
            else:
                f = alpha * self.feat + (1 - alpha) * feat
                self.feat = f / max(np.linalg.norm(f), 1e-9)


def _assign(cost, thresh):
    """匈牙利；cost > thresh 视为不可配。返回 matches, unmatched_rows, unmatched_cols"""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    from scipy.optimize import linear_sum_assignment
    c = np.where(np.isfinite(cost), cost, 1e6)
    r, k = linear_sum_assignment(c)
    m = [(i, j) for i, j in zip(r, k) if c[i, j] <= thresh]
    mr = {i for i, _ in m}
    mc = {j for _, j in m}
    return m, [i for i in range(cost.shape[0]) if i not in mr], [j for j in range(cost.shape[1]) if j not in mc]


class GatedByteTrack:
    def __init__(self, high=0.35, low=0.10, new=0.45, match=0.80, second_match=0.50,
                 unconfirmed_match=0.70, max_lost=5, fuse_score=True,
                 veto_sim=0.0, reid_match_sim=0.0, reid_proximity=0.25,
                 contest_iou=0.30, feat_alpha=0.9, min_hits=2):
        self.high, self.low, self.new = high, low, new
        self.match, self.second_match, self.unconfirmed_match = match, second_match, unconfirmed_match
        self.max_lost = max_lost
        self.fuse_score = fuse_score
        self.veto_sim = veto_sim              # >0 启用外观否决
        self.reid_match_sim = reid_match_sim  # >0 启用"外观找回"（丢失轨迹 IoU 低但外观像）
        self.reid_proximity = reid_proximity  # 外观找回要求的最低 IoU（防瞬移）
        self.contest_iou = contest_iou
        self.feat_alpha = feat_alpha
        self.min_hits = min_hits
        self.tracks = []
        self.step = 0
        self.next_id = 1
        self.stats = dict(veto_blocked=0, reid_rescued=0, frozen_updates=0)

    # -- 代价 --
    def _cost(self, trks, boxes, scores, feats, use_reid):
        if not trks or len(boxes) == 0:
            return np.zeros((len(trks), len(boxes)))
        tb = np.array([t.box for t in trks])
        iou = iou_matrix(tb, boxes)
        sim = iou * scores[None, :] if self.fuse_score else iou
        cost = 1.0 - sim
        if feats is not None and (self.veto_sim > 0 or (use_reid and self.reid_match_sim > 0)):
            for i, t in enumerate(trks):
                if t.feat is None:
                    continue
                for j in range(len(boxes)):
                    if feats[j] is None:
                        continue
                    cs = float(np.dot(t.feat, feats[j]))
                    if self.veto_sim > 0 and cs < self.veto_sim and cost[i, j] <= self.match:
                        cost[i, j] = np.inf
                        self.stats["veto_blocked"] += 1
                    elif (use_reid and self.reid_match_sim > 0 and cs >= self.reid_match_sim
                          and iou[i, j] >= self.reid_proximity and cost[i, j] > self.match):
                        cost[i, j] = self.match - 1e-3   # 允许配上，但排在正常 IoU 配对之后
                        self.stats["reid_rescued"] += 1
        return cost

    def update(self, boxes, scores, feats=None):
        """boxes (N,4) xyxy, scores (N,), feats list[np.ndarray|None] 或 None。
        返回 [(tid, det_idx, track_box)]（仅已确认且本帧有观测的轨迹）"""
        self.step += 1
        step = self.step
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = len(boxes)
        # 交界：与任一其他检测框 IoU >= contest_iou
        contested = np.zeros(n, dtype=bool)
        if n > 1 and self.contest_iou > 0:
            M = iou_matrix(boxes, boxes)
            np.fill_diagonal(M, 0)
            contested = M.max(axis=1) >= self.contest_iou

        hi = np.flatnonzero(scores >= self.high)
        lo = np.flatnonzero((scores >= self.low) & (scores < self.high))

        for t in self.tracks:
            t.predict()
            t.det_idx = -1

        confirmed = [t for t in self.tracks if t.activated and t.state in (TRACKED, LOST)]
        unconfirmed = [t for t in self.tracks if not t.activated and t.state == TRACKED]

        def sub(idx):
            return boxes[idx], scores[idx], (None if feats is None else [feats[k] for k in idx])

        def apply(t, j):
            t.update(boxes[j], float(scores[j]), None if feats is None else feats[j], step,
                     bool(contested[j]), self.feat_alpha)
            if contested[j] and feats is not None and feats[j] is not None:
                self.stats["frozen_updates"] += 1
            t.det_idx = int(j)

        # 第一阶段：已确认（含丢失）× 高分框
        b, s, f = sub(hi)
        c = self._cost(confirmed, b, s, f, use_reid=True)
        m, ut, ud = _assign(c, self.match)
        for i, j in m:
            apply(confirmed[i], hi[j])
        rem_hi = [hi[j] for j in ud]

        # 第二阶段：仍在跟踪（非丢失）的 × 低分框，只看 IoU（低分框外观不可靠，但仍做否决）
        left = [confirmed[i] for i in ut if confirmed[i].state == TRACKED]
        b, s, f = sub(lo)
        if left and len(lo):
            iou = iou_matrix(np.array([t.box for t in left]), b)
            c2 = 1.0 - iou
            if self.veto_sim > 0 and f is not None:
                for i, t in enumerate(left):
                    if t.feat is None:
                        continue
                    for j in range(len(lo)):
                        if f[j] is not None and float(np.dot(t.feat, f[j])) < self.veto_sim:
                            c2[i, j] = np.inf
                            self.stats["veto_blocked"] += 1
            m2, ut2, _ = _assign(c2, self.second_match)
            for i, j in m2:
                apply(left[i], lo[j])
            for i in ut2:
                left[i].state = LOST
        else:
            for t in left:
                t.state = LOST

        # 未确认轨迹 × 剩余高分框
        rem_hi = np.array(rem_hi, dtype=int)
        b, s, f = sub(rem_hi)
        c3 = self._cost(unconfirmed, b, s, f, use_reid=False)
        m3, ut3, ud3 = _assign(c3, self.unconfirmed_match)
        for i, j in m3:
            t = unconfirmed[i]
            apply(t, rem_hi[j])
            if t.hits >= self.min_hits:
                t.activated = True
                t.tid = self.next_id
                self.next_id += 1
        for i in ut3:
            unconfirmed[i].state = REMOVED

        # 新轨迹
        for j in ud3:
            k = rem_hi[j]
            if scores[k] < self.new:
                continue
            t = Track(boxes[k], float(scores[k]), None if feats is None else feats[k], step)
            t.det_idx = int(k)
            if step == 1 or self.min_hits <= 1:
                t.activated = True
                t.tid = self.next_id
                self.next_id += 1
            self.tracks.append(t)

        # 清理过期丢失轨迹
        for t in self.tracks:
            if t.state == LOST and step - t.last_step > self.max_lost:
                t.state = REMOVED
        self.tracks = [t for t in self.tracks if t.state != REMOVED]

        # 去重：跟踪中与丢失中的同一位置轨迹，保留更老的（ByteTrack remove_duplicate）
        tr = [t for t in self.tracks if t.state == TRACKED and t.activated]
        lt = [t for t in self.tracks if t.state == LOST]
        if tr and lt:
            M = iou_matrix(np.array([t.box for t in tr]), np.array([t.box for t in lt]))
            drop = set()
            for i, j in zip(*np.where(M > 0.85)):
                a, bb = tr[i], lt[j]
                # 外观明显不同 = 前后站位的两个人（后面的人被挡住而丢失），不是重复轨迹，不能删
                if (self.veto_sim > 0 and a.feat is not None and bb.feat is not None
                        and float(np.dot(a.feat, bb.feat)) < self.veto_sim):
                    continue
                if (a.last_step - a.start_step) >= (bb.last_step - bb.start_step):
                    drop.add(id(bb))
                else:
                    drop.add(id(a))
            self.tracks = [t for t in self.tracks if id(t) not in drop]

        return [(t.tid, t.det_idx, t.box) for t in self.tracks
                if t.activated and t.state == TRACKED and t.det_idx >= 0 and t.last_step == step]


# ----------------------------------------------------------------------------------------------
# 外观编码
# ----------------------------------------------------------------------------------------------
class Encoder:
    """--reid osnet：优先用项目自带的 encode_1012_osnet_seq.OSNetEnc（torchreid，可 DML）；
    --reid boxmot：BoxMOT 的 OSNet（云端冒烟用）。"""

    def __init__(self, kind, device, weights=""):
        self.kind = kind
        self.min_w, self.min_h = 16, 32
        if kind == "osnet":
            import encode_1012_osnet_seq as E
            self.enc = E.OSNetEnc(device=device, rgb=True) if "rgb" in E.OSNetEnc.__init__.__code__.co_varnames \
                else E.OSNetEnc(device=device)
            self.min_w, self.min_h = E.MIN_W, E.MIN_H
            self.fn = lambda crops: self.enc.encode(crops)
        elif kind == "boxmot":
            import torch
            from boxmot.reid.core.runtime import ReID
            m = ReID(weights=Path(weights), device="cpu", half=False).model

            def fn(crops, _m=m):
                with torch.no_grad():
                    outs = []
                    for c in crops:
                        h, w = c.shape[:2]
                        v = _m.get_features(np.array([[0, 0, w, h]], dtype=np.float32), np.ascontiguousarray(c))
                        outs.append(np.asarray(v, dtype=np.float32).reshape(-1))
                V = np.stack(outs)
                return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
            self.fn = fn
        else:
            raise ValueError(kind)

    def __call__(self, frame, boxes):
        crops, idx = [], []
        H, W = frame.shape[:2]
        for k, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(float(v))) for v in b]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < self.min_w or y2 - y1 < self.min_h:
                continue
            crops.append(frame[y1:y2, x1:x2])
            idx.append(k)
        out = [None] * len(boxes)
        if crops:
            V = self.fn(crops)
            for k, v in zip(idx, V):
                out[k] = v.astype(np.float16)
        return out


# ----------------------------------------------------------------------------------------------
# 检测 + 跟踪前去重
# ----------------------------------------------------------------------------------------------
def mask_dedup(xyxy, conf, masks, thr):
    """返回保留下标。小掩膜被大掩膜包含 >= thr 视为同一人重复框（头框/半身框）。"""
    n = len(xyxy)
    if thr <= 0 or n < 2:
        return list(range(n)), 0
    order = sorted(range(n), key=lambda k: -float(conf[k]))
    if masks is not None and len(masks) >= n:
        M = (masks[:n] > 0.5).reshape(n, -1).astype(np.float32)
        area = M.sum(1)
        inter = M @ M.T
    else:  # 无掩膜时退化为框包含率
        bx = np.asarray(xyxy, dtype=np.float64)
        area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
        x1 = np.maximum(bx[:, None, 0], bx[None, :, 0]); y1 = np.maximum(bx[:, None, 1], bx[None, :, 1])
        x2 = np.minimum(bx[:, None, 2], bx[None, :, 2]); y2 = np.minimum(bx[:, None, 3], bx[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep = []
    for j in order:
        dup = False
        for k in keep:
            mn = min(area[j], area[k])
            if mn > 0 and inter[j, k] / mn >= thr:
                dup = True
                break
        if not dup:
            keep.append(j)
    return sorted(keep), n - len(keep)


def run_detection(a, video, sample_frames_cb):
    """顺序读视频，每 a.stride 帧检测一次；yield (fnum, frame, boxes, scores, n_dup)"""
    from ultralytics import YOLO
    model = YOLO(a.model)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_frames_cb(fps)
    start_frame = (int(round(a.start_sec * fps)) // a.stride) * a.stride if a.start_sec > 0 else 0
    end_frame = start_frame + int(round(a.dur_sec * fps)) if a.dur_sec > 0 else None
    for _ in range(start_frame):
        if not cap.grab():
            break
    fnum, sampled = start_frame, 0
    try:
        while True:
            if not cap.grab():
                break
            if end_frame is not None and fnum >= end_frame:
                break
            if fnum % a.stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                r = model.predict(frame, classes=[0], conf=a.conf, iou=a.iou, imgsz=a.imgsz,
                                  device="cpu", verbose=False)[0]
                xyxy = np.zeros((0, 4), np.float32)
                cf = np.zeros((0,), np.float32)
                masks = None
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    cf = r.boxes.conf.cpu().numpy().astype(np.float32)
                    if r.masks is not None:
                        masks = r.masks.data.cpu().numpy()
                keep, ndup = mask_dedup(xyxy, cf, masks, a.dedup_mask)
                yield fnum, frame, xyxy[keep], cf[keep], ndup
                sampled += 1
                if a.max_frames and sampled >= a.max_frames:
                    break
            fnum += 1
    finally:
        cap.release()


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--video", default="", help="mp4 路径（默认 <VID_DIR>/<seg>.mp4）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10, help="检测下限（=ByteTrack 低分阈值的下限）")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU；重复框交给 --dedup-mask 在跟踪前处理")
    ap.add_argument("--dedup-mask", type=float, default=0.9, help="跟踪前掩膜包含率去重（0=关）")
    ap.add_argument("--stride", type=int, default=5, help="检测采样步长（建缓存时用 5，可服务 5/10）")
    ap.add_argument("--track-stride", type=int, default=0, help="跟踪步长（0=同 --stride；须为 stride 的倍数）")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--dur-sec", type=float, default=0.0)
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 个检测采样帧（冒烟）")
    # 跟踪
    ap.add_argument("--high", type=float, default=0.35)
    ap.add_argument("--low", type=float, default=0.10)
    ap.add_argument("--new", type=float, default=0.45)
    ap.add_argument("--match", type=float, default=0.80, help="第一阶段代价上限（1-融合相似度）")
    ap.add_argument("--no-fuse-score", action="store_true")
    ap.add_argument("--buffer-sec", type=float, default=2.0, help="丢失轨迹保留的真实秒数")
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--contest-iou", type=float, default=0.30, help="交界判定；交界帧不更新外观模板")
    # 外观
    ap.add_argument("--reid", default="none", choices=["none", "osnet", "boxmot"])
    ap.add_argument("--reid-device", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--reid-weights", default="", help="--reid boxmot 时的权重路径")
    ap.add_argument("--veto-sim", type=float, default=0.0, help="外观余弦低于此值禁止配对（0=关；建议 0.35~0.5 扫）")
    ap.add_argument("--reid-match-sim", type=float, default=0.0, help="丢失轨迹外观找回阈值（0=关）")
    # 输入输出
    ap.add_argument("--det-cache", default="", help="检测缓存 npz；存在则跳过 YOLO")
    ap.add_argument("--out-dir", default=str(ROOT / "output_crosscam_local/1012_yolo_v2"))
    ap.add_argument("--tag", default="", help="输出子目录名（不同配置放不同目录）")
    ap.add_argument("--box-source", default="det", choices=["det", "track"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    tstride = a.track_stride or a.stride
    using_cache = bool(a.det_cache) and Path(a.det_cache).exists()
    if not using_cache and tstride % a.stride:   # 用缓存时改为对照缓存里的 stride 检查
        raise SystemExit("--track-stride 必须是 --stride 的倍数")
    if a.veto_sim > 0 or a.reid_match_sim > 0:
        if a.reid == "none" and not a.det_cache:
            raise SystemExit("--veto-sim/--reid-match-sim 需要 --reid osnet|boxmot（或带特征的 --det-cache）")

    out_dir = Path(a.out_dir) / (a.tag or "default") / a.seg
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"{a.seg}_detections.jsonl"
    if outp.exists() and not a.force and not a.max_frames:
        print(f"[skip] {outp} 已存在（--force 覆盖）")
        return

    # ---------- 1) 取检测（缓存或现跑） ----------
    t0 = time.time()
    cache = Path(a.det_cache) if a.det_cache else None
    frames, B, S, F = [], [], [], []
    fps_box = {}
    total_dup = 0
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if meta.get("version") != CACHE_VERSION:
            raise SystemExit(f"缓存版本不符：{meta.get('version')}")
        if tstride % meta["stride"]:
            raise SystemExit(f"缓存 stride={meta['stride']}，跟踪步长 {tstride} 不是其倍数")
        fps_box["fps"] = meta["fps"]
        total_dup = meta.get("n_dup", 0)
        fr, off, bx, sc = z["frames"], z["offsets"], z["boxes"], z["scores"]
        ft = z["feats"] if "feats" in z.files else None
        fv = z["feat_valid"] if "feat_valid" in z.files else None
        for i, f in enumerate(fr):
            s0, s1 = off[i], off[i + 1]
            frames.append(int(f)); B.append(bx[s0:s1]); S.append(sc[s0:s1])
            if ft is not None:
                F.append([ft[k].astype(np.float32) if fv[k] else None for k in range(s0, s1)])
            else:
                F.append(None)
        if (a.veto_sim > 0 or a.reid_match_sim > 0) and ft is None:
            raise SystemExit("缓存里没有外观特征；重建缓存时加 --reid osnet")
        print(f"[cache] {cache} frames={len(frames)} stride={meta['stride']} feats={ft is not None} "
              f"dedup={meta.get('dedup_mask')}", flush=True)
    else:
        try:
            import encode_1012_osnet_seq as E
            vid_dir = E.VID_DIR
        except Exception:
            vid_dir = DEFAULT_VID_DIR
        video = Path(a.video) if a.video else vid_dir / f"{a.seg}.mp4"
        if not video.exists():
            raise SystemExit(f"missing video {video}")
        enc = Encoder(a.reid, a.reid_device, a.reid_weights) if a.reid != "none" else None
        for fnum, frame, xyxy, cf, ndup in run_detection(a, video, lambda fps: fps_box.__setitem__("fps", fps)):
            frames.append(fnum); B.append(xyxy); S.append(cf)
            F.append(enc(frame, xyxy) if enc is not None else None)
            total_dup += ndup
            if len(frames) % 50 == 0:
                el = time.time() - t0
                print(f"[{a.seg}] det {len(frames)} frames {el:.0f}s {len(frames)/max(el,1e-6):.2f} f/s "
                      f"dedup_dropped={total_dup}", flush=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            off = np.cumsum([0] + [len(b) for b in B])
            bx = np.concatenate(B) if off[-1] else np.zeros((0, 4), np.float32)
            sc = np.concatenate(S) if off[-1] else np.zeros((0,), np.float32)
            save = dict(frames=np.array(frames, np.int64), offsets=off, boxes=bx, scores=sc)
            if enc is not None:
                dim = next((len(v) for fl in F for v in fl if v is not None), 512)
                feats = np.zeros((off[-1], dim), np.float16)
                valid = np.zeros(off[-1], bool)
                k = 0
                for fl in F:
                    for v in fl:
                        if v is not None:
                            feats[k] = v; valid[k] = True
                        k += 1
                save.update(feats=feats, feat_valid=valid)
            save["meta"] = json.dumps(dict(version=CACHE_VERSION, seg=a.seg, stride=a.stride,
                                           fps=fps_box["fps"], imgsz=a.imgsz, conf=a.conf, iou=a.iou,
                                           dedup_mask=a.dedup_mask, reid=a.reid, n_dup=total_dup,
                                           max_frames=a.max_frames, start_sec=a.start_sec, dur_sec=a.dur_sec))
            np.savez_compressed(cache, **save)
            print(f"[cache] saved {cache}", flush=True)
        F = [[None if v is None else v.astype(np.float32) for v in fl] if fl is not None else None for fl in F]
    t_det = time.time() - t0

    # ---------- 2) 跟踪 ----------
    fps = fps_box["fps"]
    max_lost = max(1, math.ceil(a.buffer_sec * fps / tstride))
    trk = GatedByteTrack(high=a.high, low=a.low, new=a.new, match=a.match, max_lost=max_lost,
                         fuse_score=not a.no_fuse_score, veto_sim=a.veto_sim,
                         reid_match_sim=a.reid_match_sim, contest_iou=a.contest_iou, min_hits=a.min_hits)
    n_boxes, ids_seen = 0, {}
    with open(outp, "w", encoding="utf-8") as fh:
        for fnum, bx, sc, fl in zip(frames, B, S, F):
            if fnum % tstride:
                continue
            res = trk.update(bx, sc, fl)
            res.sort(key=lambda r: r[1])
            boxes, ids, scores = [], [], []
            for tid, j, tb in res:
                src = bx[j] if a.box_source == "det" else tb
                boxes.append([round(float(v), 1) for v in src])
                ids.append(int(tid))
                scores.append(round(float(sc[j]), 4))
                ids_seen.setdefault(int(tid), [fnum, fnum])[1] = fnum
            n_boxes += len(boxes)
            fh.write(json.dumps({"frame": fnum, "track_ids": ids, "boxes": boxes, "scores": scores}) + "\n")

    durs = sorted((e - s) / fps for s, e in ids_seen.values())
    meta = dict(seg=a.seg, args=vars(a), fps=fps, track_stride=tstride, max_lost_steps=max_lost,
                n_boxes=n_boxes, unique_tracks=len(ids_seen), dedup_dropped=total_dup,
                median_track_sec=round(durs[len(durs) // 2], 2) if durs else 0,
                tracker_stats=trk.stats, det_sec=round(t_det, 1), total_sec=round(time.time() - t0, 1))
    (out_dir / f"{a.seg}_detections.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
    print(f"[{a.seg}] DONE tag={a.tag} stride={tstride} buffer={a.buffer_sec}s→{max_lost}步 boxes={n_boxes} "
          f"tracks={len(ids_seen)} median_len={meta['median_track_sec']}s dedup={total_dup} "
          f"stats={trk.stats} wall={meta['total_sec']}s -> {outp}", flush=True)


if __name__ == "__main__":
    main()
