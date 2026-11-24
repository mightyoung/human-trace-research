#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹级「人脸 or 整体」路由可行性探针（只读生产轨迹与视频，不改任何管线）。

问题：对每条轨迹，能否快速判断"该用脸判身份，还是只能用整体 ReID"？
做法：每条轨迹在其生命周期内均匀抽 K 帧 → 头部 ROI 跑 SCRFD → 冲突仲裁把人脸
      唯一分配给某个人体框 → 质量门 → ArcFace 嵌入 → 逐轨迹聚合。

三个必须分清的层次：
  1) 检出（detected）：ROI 里检出了脸
  2) 可用（usable）  ：瞳距/清晰度/姿态达标，能用于比对
  3) 自洽（self_cos）：同一轨迹内多张可用脸彼此像不像 —— 若不自洽，说明
                       关联错了或质量不够，此时"用人脸"反而制造错误

用法：
    python tools/probe_face_routing.py                # 8 段全跑
    python tools/probe_face_routing.py --segs 305_test_544 323_test_527 --k 12

输出：output_crosscam_local/selector_osnet_1012/face_routing/
    face_obs.jsonl        逐脸观测
    face_track.csv        逐轨迹聚合
    face_vectors.npz      轨迹人脸原型 + 前后半原型
    face_routing_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from scrfd_dnn import ScrfdDNN  # noqa: E402

DET_DIR = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd")
VID_DIR = os.path.join(ROOT, "media-resource/1012/segments")
EMB_DIR = os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb")
OUT = os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/face_routing")

SEGS = ["305_train_529", "318_train_525", "320_train_514", "323_train_510",
        "305_test_544", "318_test_534", "320_test_518", "323_test_527"]

TPL112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

# ---- 质量门（多档同时记录，事后可扫；此处只做"是否参与嵌入"的粗筛）----
EMB_MIN_FW = 24      # 低于此宽度的脸不做嵌入（112 对齐后全是插值）
EMB_MIN_PD = 10      # pd<10 时 112 对齐等价于放大 11 倍以上，特征已无信息

# ---- 头部 ROI：覆盖头到肩，允许头伸出框顶（坐姿/被桌挡是已知场景）----
# BOT 收到 0.40：0.55 会把大片躯干（卫衣图案、衣领）框进来，SCRFD 在放大的
# 小 ROI 上会把这些纹理判成"脸"（323#1908 实测：真脸在左，检出框落在卫衣图案上）。
HEAD_PAD_X, HEAD_PAD_TOP, HEAD_PAD_BOT = 0.15, 0.22, 0.40
GEOM_PD_LO, GEOM_PD_HI = 0.20, 0.55   # 瞳距/脸宽 的合理区间（真实人脸约 0.35）
GEOM_ROLL_MAX = 25.0                  # 两眼连线倾角上限（度）
GEOM_AR = (0.50, 1.25)                # 脸框 宽/高 合理区间


def head_roi(box, W, H):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rx1 = int(max(0, x1 - HEAD_PAD_X * w))
    rx2 = int(min(W, x2 + HEAD_PAD_X * w))
    ry1 = int(max(0, y1 - HEAD_PAD_TOP * h))
    ry2 = int(min(H, y1 + HEAD_PAD_BOT * h))
    return rx1, ry1, rx2, ry2


def face_quality(img, det):
    """det=[x1,y1,x2,y2,kp10,score] 原图坐标。返回质量字典或 None。"""
    x1, y1, x2, y2 = [float(v) for v in det[:4]]
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(img.shape[1], int(round(x2))), min(img.shape[0], int(round(y2)))
    fw, fh = xi2 - xi1, yi2 - yi1
    if fw < 8 or fh < 8:
        return None
    roi = img[yi1:yi2, xi1:xi2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    kps = det[4:14].reshape(5, 2)
    pd = float(np.linalg.norm(kps[1] - kps[0]))          # 瞳距 px
    eye_mid = (kps[0] + kps[1]) / 2.0
    mouth_mid = (kps[3] + kps[4]) / 2.0
    yaw_off = abs(float(kps[2][0] - eye_mid[0])) / max(pd, 1.0)   # |鼻子偏离眼中线|/瞳距
    denom = max(float(mouth_mid[1] - eye_mid[1]), 1.0)
    pitch_r = float(kps[2][1] - eye_mid[1]) / denom
    roll = abs(float(np.arctan2(kps[1][1] - kps[0][1], kps[1][0] - kps[0][0])))
    ar = fw / max(fh, 1)
    pd_ratio = pd / max(fw, 1.0)
    # ★ 几何可信度：SCRFD 在小 ROI 放大后会把衣纹/图案判成脸（关键点挤成一团），
    #   分数不可靠，唯一可靠的免费信号是 5 点的几何自洽性。
    geom_ok = bool(GEOM_PD_LO <= pd_ratio <= GEOM_PD_HI
                   and float(np.degrees(roll)) <= GEOM_ROLL_MAX
                   and GEOM_AR[0] <= ar <= GEOM_AR[1])
    return dict(fw=fw, fh=fh, pd=round(pd, 2), lap=round(lap, 1),
                yaw=round(yaw_off, 3), pitch=round(pitch_r, 3),
                roll=round(float(np.degrees(roll)), 1),
                ar=round(float(ar), 3), pd_ratio=round(float(pd_ratio), 3),
                scale=round(float(112.0 / max(pd, 1.0)), 2),
                geom_ok=geom_ok,
                det_score=round(float(det[14]), 3))


def align112(img, kps5):
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), TPL112, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ---- 检测：固定网格 + 重叠 + 全局 NMS ----
# 为什么不是「每个 box 裁 ROI」：实测（305_test 16 帧 146 box）四种策略的检出/几何合格
#   全帧 1x1: 28/28    2x2: 48/46    3x3: 71/66    4x4: 67/56
#   按 box 裁 head ROI（放大到 640）: 8/5  ← 裁太紧、放大过度，召回崩
# 结论：SCRFD 需要足够上下文，裁紧 + 放大反而丢检；3x3 是这颗芯片的最优尺度。
GRID = 3
GRID_OV = 0.15


def detect_faces_grid(img, det, g=GRID, ov=GRID_OV):
    H, W = img.shape[:2]
    out = []
    ys = [int(y * H / g) for y in range(g + 1)]
    xs = [int(x * W / g) for x in range(g + 1)]
    for i in range(g):
        for j in range(g):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            py, px = int((y1 - y0) * ov / 2), int((x1 - x0) * ov)
            sy0, sy1 = max(0, y0 - py), min(H, y1 + py)
            sx0, sx1 = max(0, x0 - px), min(W, x1 + px)
            sub = img[sy0:sy1, sx0:sx1]
            if sub.shape[0] < 32 or sub.shape[1] < 32:
                continue
            fs = det.detect(sub)
            for f in fs:
                gg = f.copy()
                gg[[0, 2]] += sx0
                gg[[1, 3]] += sy0
                gg[4:14:2] += sx0
                gg[5:14:2] += sy0
                out.append(gg)
    if not out:
        return np.zeros((0, 15), np.float32)
    out = np.asarray(out, np.float32)
    xywh = np.column_stack((out[:, 0], out[:, 1], out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]))
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), out[:, 14].tolist(), 0.5, 0.4)
    if len(idx) == 0:
        return np.zeros((0, 15), np.float32)
    return out[np.asarray(idx).reshape(-1)]


def assign_faces_to_boxes(faces, obs, mode="greedy"):
    """人脸 → 人体框 的全局一对一分配（一张脸只属于一个人，一个人只取最像自己那张）。

    约束（全部几何、零成本）：
      · 覆盖：脸框面积至少 60% 落在人体框内 —— 这是主判据。
        ★ 不能用"脸中心落在框顶 +45%h"这种头带假设：320#900 实测有 3 张真脸，
        T2/T3 的框只框住头（h=88/67，人被显示器挡住），脸中心落在框中央偏下，
        头带假设把它们全判成"无脸"。
      · 纵向：脸顶不得低于「框顶 + 55%h」（脸的整体必须在上半身，防止把
        框内别处的脸拉过来）。
      · 横向：脸中心在框宽 ±20% 内。
    排序键 d = 脸中心到「框顶中央偏下 30%h」的归一化距离；只用于排序，不是硬门槛。
    """
    if len(faces) == 0 or not obs:
        return {}, 0
    cands = []
    for i, f in enumerate(faces):
        fw, fh = f[2] - f[0], f[3] - f[1]
        fa = max(fw * fh, 1.0)
        cx, cy = (f[0] + f[2]) / 2.0, (f[1] + f[3]) / 2.0
        for tid, b in obs:
            w, h = b[2] - b[0], b[3] - b[1]
            if w <= 1 or h <= 1:
                continue
            ix1, iy1 = max(f[0], b[0]), max(f[1], b[1])
            ix2, iy2 = min(f[2], b[2]), min(f[3], b[3])
            cover = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / fa
            if cover < 0.60:
                continue
            if f[1] > b[1] + 0.55 * h:
                continue
            if not (b[0] - 0.20 * w <= cx <= b[2] + 0.20 * w):
                continue
            ax, ay = (b[0] + b[2]) / 2.0, b[1] + 0.30 * h
            d = float(np.hypot((cx - ax) / max(w, 1.0), (cy - ay) / max(h, 1.0)))
            cands.append((d, i, tid, cover))
    cands.sort()
    if mode == "hungarian":
        # 全局最优一对一（代价矩阵 + 匈牙利），禁止的配对给大代价。
        # 与贪心的差别只出现在「多张脸 × 多个候选框」互相冲突的帧：
        # 贪心按 d 从小到大锁死，可能先占掉别人更合适的框。
        from scipy.optimize import linear_sum_assignment
        BIG = 1e6
        tids = [t for t, _ in obs]
        ti = {t: j for j, t in enumerate(tids)}
        C = np.full((len(faces), len(tids)), BIG, np.float64)
        det_of = {}
        for d, i, tid, cover in cands:
            j = ti[tid]
            if d < C[i, j]:
                C[i, j] = d
                det_of[(i, j)] = cover
        ri, ci = linear_sum_assignment(C)
        out = {}
        for i, j in zip(ri, ci):
            if C[i, j] >= BIG:
                continue
            out[tids[j]] = (faces[i], float(C[i, j]), float(det_of.get((i, j), 0.0)))
        return out, len(cands)
    used_f, used_t, out = set(), set(), {}
    for d, i, tid, cover in cands:
        if i in used_f or tid in used_t:
            continue
        used_f.add(i)
        used_t.add(tid)
        out[tid] = (faces[i], d, cover)
    return out, len(cands)


def load_tracks(seg):
    """frame -> [(tid, [x1,y1,x2,y2]), ...]  （生产轨迹，已做同帧去重）"""
    p = os.path.join(DET_DIR, seg, f"{seg}_detections.jsonl")
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[int(o["frame"])] = list(zip([int(t) for t in o["track_ids"]], o["boxes"]))
    return out


def sample_frames(tracks_by_frame, k):
    """每条轨迹在其出现帧中等间隔取 <=k 帧。返回 {frame: [(tid, box)]} 与 tid->总帧数。"""
    by_tid = defaultdict(list)
    box_of = {}
    for fr, obs in tracks_by_frame.items():
        for tid, b in obs:
            by_tid[tid].append(fr)
            box_of[(tid, fr)] = b
    need = defaultdict(list)
    for tid, frs in by_tid.items():
        frs.sort()
        if len(frs) <= k:
            pick = frs
        else:
            idx = np.linspace(0, len(frs) - 1, k).round().astype(int)
            pick = [frs[i] for i in sorted(set(idx.tolist()))]
        for fr in pick:
            need[fr].append((tid, box_of[(tid, fr)]))
    return dict(need), {t: len(v) for t, v in by_tid.items()}


def scan_frames(seg, need, report_every=0):
    """顺序 grab 推进，只对目标帧 retrieve 全分辨率。yield (frame, img)。"""
    target = set(need)
    if not target:
        return
    hi = max(target)
    cap = cv2.VideoCapture(os.path.join(VID_DIR, f"{seg}.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {seg}")
    fr = 0
    t0 = time.time()
    try:
        while fr <= hi:
            if not cap.grab():
                break
            if fr in target:
                ok, img = cap.retrieve()
                if ok:
                    yield fr, img
            fr += 1
    finally:
        cap.release()
    if report_every:
        print(f"  scan {seg}: {fr} frames in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segs", nargs="*", default=SEGS)
    ap.add_argument("--k", type=int, default=10, help="每条轨迹最多抽多少帧")
    ap.add_argument("--min-box-h", type=float, default=0.0,
                    help="只对框高≥该值的帧跑检测（0=全跑）。L0 预筛用，"
                         "pd≈0.06h 意味着 h<200px 不可能出可用脸")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--assign", choices=["greedy", "hungarian"], default="greedy",
                    help="脸→人归属算法：greedy=按代价升序全局一对一（历史口径）；"
                         "hungarian=代价矩阵全局最优一对一（R4-4）")
    ap.add_argument("--assign-cmp", action="store_true",
                    help="同一批帧同时跑两种归属，统计差异（不额外跑检测）")
    ap.add_argument("--out-dir", default="",
                    help="输出目录（默认 face_routing/）；做对比实验时指向临时目录，避免覆盖生产产物")
    args = ap.parse_args()
    if args.out_dir:
        global OUT
        OUT = args.out_dir

    os.makedirs(OUT, exist_ok=True)
    det = ScrfdDNN(os.path.join(ROOT, "models/face/scrfd_2.5g_bnkps.onnx"),
                   thresh=0.5, nms=0.4, threads=args.threads)
    rec = cv2.dnn.readNetFromONNX(os.path.join(ROOT, "models/face/w600k_mbf.onnx"))

    def face_emb(img, kps5):
        ali = align112(img, kps5)
        if ali is None:
            return None
        blob = ((ali[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        rec.setInput(blob)
        v = np.asarray(rec.forward()).reshape(-1).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    rows, vecs = [], {}
    summary = {}
    t_start = time.time()

    for seg in args.segs:
        cam = seg.split("_")[0]
        tb = load_tracks(seg)
        need, n_frames_of = sample_frames(tb, args.k)
        n_box = sum(len(v) for v in need.values())
        print(f"[{seg}] tracks={len(n_frames_of)} frames={len(need)} boxes={n_box}", flush=True)

        per_track = defaultdict(lambda: dict(tried=0, det=0, fake=0, usable=0,
                                             fw=[], pd=[], lap=[], yaw=[], pitch=[], score=[],
                                             geo_skip=0, emb=[], emb_u=[], frames=[]))
        n_face_raw = n_conflict = n_fake = n_frame_skip = 0
        n_cmp_frames = n_cmp_frame_diff = n_cmp_tid_diff = n_g_assign = n_h_assign = 0
        t0 = time.time()

        for fr, img in scan_frames(seg, need):
            obs = need[fr]
            bh_map = {}
            for tid, b in obs:
                per_track[tid]["tried"] += 1
                bh_map[tid] = b[3] - b[1]
            if args.min_box_h > 0 and max(bh_map.values(), default=0) < args.min_box_h:
                n_frame_skip += 1
                continue
            faces = detect_faces_grid(img, det)
            if len(faces) == 0:
                continue
            n_face_raw += len(faces)
            asg, n_cand = assign_faces_to_boxes(faces, obs, args.assign)
            if args.assign_cmp:
                asg_h, _ = assign_faces_to_boxes(faces, obs, "hungarian")
                n_cmp_frames += 1
                n_g_assign += len(asg)
                n_h_assign += len(asg_h)
                mg = {t: tuple(np.round(f[0][:4], 1)) for t, f in asg.items()}
                mh = {t: tuple(np.round(f[0][:4], 1)) for t, f in asg_h.items()}
                if mg != mh:
                    n_cmp_frame_diff += 1
                    n_cmp_tid_diff += len({t for t in set(mg) | set(mh) if mg.get(t) != mh.get(t)})
            n_conflict += max(0, n_cand - len(asg))
            for tid, (f, ad, cov) in asg.items():
                q = face_quality(img, f)
                if q is None:
                    continue
                st = per_track[tid]
                if not q["geom_ok"]:
                    st["fake"] += 1
                    n_fake += 1
                    continue
                q["assoc_d"] = round(float(ad), 4)
                q["assoc_cover"] = round(float(cov), 3)
                q["box_h"] = round(float(bh_map.get(tid, 0)), 1)
                st["det"] += 1
                for kk in ("fw", "pd", "lap", "yaw", "pitch"):
                    st[kk].append(q[kk])
                st["score"].append(q["det_score"])
                # 可用判据（宽松初筛，真正的门在聚合时分档扫描）
                # pd>=20px ≈ GB/T 31488 检出下限的一半；112 对齐放大倍数 = 112/pd，
                # pd<12 等价于把 12px 的脸插值到 112，特征无意义。
                usable = (q["pd"] >= 20) and (q["fw"] >= 32) and (q["lap"] >= 50) \
                    and (q["yaw"] <= 0.35) and (0.25 <= q["pitch"] <= 0.95)
                if usable:
                    st["usable"] += 1
                # 特征对「几何合格」的脸都提，便于事后按 pd/fw 分档扫（"多低算不可用"要靠数据定，
                # 不能先验拍一个门槛再只用达标样本评估——那样 AUC 会虚高）。
                if q["fw"] >= EMB_MIN_FW and q["pd"] >= EMB_MIN_PD:
                    kps = f[4:14].reshape(5, 2).astype(np.float32)
                    v = face_emb(img, kps)
                    if v is not None:
                        st["emb"].append(v)
                        st["emb_u"].append(bool(usable))
                        st["frames"].append(fr)
                rows.append(dict(seg=seg, cam=cam, tid=int(tid), frame=int(fr),
                                 **q, usable=bool(usable)))

        dt = time.time() - t0
        print(f"  {dt:.0f}s  raw_faces={n_face_raw} geom_reject={n_fake} conflicts={n_conflict} frame_skip={n_frame_skip}", flush=True)
        if args.assign_cmp:
            print(f"  [assign-cmp] 有脸帧={n_cmp_frames} 贪心分配={n_g_assign} 匈牙利={n_h_assign} "
                  f"结果不同的帧={n_cmp_frame_diff} ({n_cmp_frame_diff/max(1,n_cmp_frames)*100:.1f}%) "
                  f"受影响轨迹数={n_cmp_tid_diff}", flush=True)

        # --- 轨迹聚合 ---
        n_track = len(n_frames_of)
        n_det = n_use = n_self = n_self_u = 0
        for tid in sorted(n_frames_of):
            st = per_track.get(tid)
            agg = dict(seg=seg, cam=cam, tid=tid, n_frames=n_frames_of[tid],
                       tried=0, det=0, fake=0, usable=0, geo_skip=0)
            if st:
                agg.update(tried=st["tried"], det=st["det"], fake=st["fake"],
                           usable=st["usable"], geo_skip=st["geo_skip"])
                for kk in ("fw", "pd", "lap", "yaw", "pitch", "score"):
                    v = st[kk]
                    agg[f"{kk}_p50"] = round(float(np.median(v)), 3) if v else None
                E = np.asarray(st["emb"], np.float32) if st["emb"] else np.zeros((0, 512), np.float32)
                U = np.asarray(st["emb_u"], bool) if st["emb_u"] else np.zeros((0,), bool)
                Eu = E[U] if len(E) and U.any() else np.zeros((0, 512), np.float32)
                if len(E) >= 2:
                    S = E @ E.T
                    iu = np.triu_indices(len(E), 1)
                    agg["self_cos"] = round(float(np.median(S[iu])), 4)
                    n_self += 1
                if len(Eu) >= 2:
                    S = Eu @ Eu.T
                    iu = np.triu_indices(len(Eu), 1)
                    agg["self_cos_u"] = round(float(np.median(S[iu])), 4)
                    n_self_u += 1
                # 前后半分原型（按该轨迹出现帧的时间中位切）；优先用 usable 子集
                # 门槛放到「每半 ≥1 张」，因为 k 帧抽样下多数轨迹只出 1-2 张脸；
                # 每半实际张数记在 half_na/half_nb，事后按需筛。
                agg["half_cos"] = None
                fr_all = np.asarray(st["frames"])
                for tag, EE, FF in (("u", Eu, fr_all[U] if len(U) else np.array([])),
                                    ("all", E, fr_all)):
                    if len(EE) >= 2 and len(FF) == len(EE):
                        mid = (FF.min() + FF.max()) / 2.0
                        A, B = EE[FF <= mid], EE[FF > mid]
                        if len(A) and len(B):
                            a = A.mean(0); a /= np.linalg.norm(a) + 1e-9
                            b = B.mean(0); b /= np.linalg.norm(b) + 1e-9
                            agg["half_cos"] = round(float(a @ b), 4)
                            agg["half_src"] = tag
                            agg["half_na"], agg["half_nb"] = len(A), len(B)
                            vecs[f"{seg}|{tid}|a"] = a.astype(np.float32)
                            vecs[f"{seg}|{tid}|b"] = b.astype(np.float32)
                            break
                if len(E):
                    e = E.mean(0); e /= np.linalg.norm(e) + 1e-9
                    vecs[f"{seg}|{tid}|full"] = e.astype(np.float32)
                if len(Eu):
                    e = Eu.mean(0); e /= np.linalg.norm(e) + 1e-9
                    vecs[f"{seg}|{tid}|fusu"] = e.astype(np.float32)
            n_det += agg["det"] > 0
            n_use += agg["usable"] > 0
            rows_track = agg
            summary.setdefault("_tracks", []).append(rows_track)

        print(f"  tracks={n_track} 有脸={n_det} 有可用脸={n_use} 自洽可算={n_self} 可用自洽={n_self_u}", flush=True)

        # 每段落盘：中断/长任务不丢数据（train 段很长，全靠最后统一写盘会前功尽弃）
        st_seg = [t for t in summary["_tracks"] if t["seg"] == seg]
        fields = []
        for t in st_seg:
            for k in t:
                if k not in fields:
                    fields.append(k)
        with open(os.path.join(OUT, f"face_track_{seg}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(st_seg)
        with open(os.path.join(OUT, f"face_obs_{seg}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                if r["seg"] == seg:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        segv = {k: v for k, v in vecs.items() if k.split("|")[0] == seg}
        if segv:
            ks = sorted(segv)
            np.savez_compressed(os.path.join(OUT, f"face_vecs_{seg}.npz"),
                                keys=np.array(ks), vecs=np.stack([segv[k] for k in ks]))

    # --- 汇总 ---
    tracks = summary.pop("_tracks", [])
    by_cam = {}
    for cam in sorted({t["cam"] for t in tracks}):
        g = [t for t in tracks if t["cam"] == cam]
        d = dict(n_track=len(g),
                 pct_det=round(100 * sum(1 for t in g if t["det"] > 0) / len(g), 1),
                 pct_usable=round(100 * sum(1 for t in g if t["usable"] > 0) / len(g), 1),
                 pct_usable3=round(100 * sum(1 for t in g if t["usable"] >= 3) / len(g), 1))
        pd = [t["pd_p50"] for t in g if t.get("pd_p50")]
        sc = [t.get("self_cos") for t in g if t.get("self_cos") is not None]
        scu = [t.get("self_cos_u") for t in g if t.get("self_cos_u") is not None]
        hc = [t.get("half_cos") for t in g if t.get("half_cos") is not None]
        d.update(pd_p50=round(float(np.median(pd)), 1) if pd else None,
                 self_cos_p50=round(float(np.median(sc)), 3) if sc else None,
                 n_self=len(sc),
                 self_cos_u_p50=round(float(np.median(scu)), 3) if scu else None,
                 n_self_u=len(scu),
                 half_cos_p50=round(float(np.median(hc)), 3) if hc else None,
                 n_half=len(hc))
        by_cam[cam] = d
    all_t = tracks
    # 快速粗筛标定：人脸宽 / 人体框高 的实测比例（用于不跑检测就估计"这条轨迹最大可能脸多大"）
    ratio = [r["fw"] / r["box_h"] for r in rows if r.get("box_h") and r.get("fw")]
    overall = dict(n_track=len(all_t),
                   pct_det=round(100 * sum(1 for t in all_t if t["det"] > 0) / len(all_t), 1),
                   pct_usable=round(100 * sum(1 for t in all_t if t["usable"] > 0) / len(all_t), 1),
                   pct_usable3=round(100 * sum(1 for t in all_t if t["usable"] >= 3) / len(all_t), 1),
                   n_self=sum(1 for t in all_t if t.get("self_cos") is not None),
                   n_half=sum(1 for t in all_t if t.get("half_cos") is not None),
                   fw_over_boxh_p50=round(float(np.median(ratio)), 4) if ratio else None,
                   fw_over_boxh_p10=round(float(np.percentile(ratio, 10)), 4) if ratio else None)
    out = dict(meta=dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), k=args.k, segs=args.segs,
                         runtime_s=round(time.time() - t_start, 1),
                         detector="scrfd_2.5g_bnkps(cv2.dnn)", embedder="w600k_mbf(cv2.dnn)",
                         grid=GRID, grid_ov=GRID_OV,
                         usable_rule="pd>=20 & fw>=32 & lap>=50 & yaw<=0.35 & 0.25<=pitch<=0.95"),
               overall=overall, by_cam=by_cam)

    fields = []
    for t in tracks:
        for k in t:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(OUT, "face_track.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(tracks)
    with open(os.path.join(OUT, "face_obs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            r.pop("box", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if vecs:
        ks = sorted(vecs)
        np.savez_compressed(os.path.join(OUT, "face_vectors.npz"),
                            keys=np.array(ks), vecs=np.stack([vecs[k] for k in ks]))
    with open(os.path.join(OUT, "face_routing_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out["overall"], ensure_ascii=False))
    for cam, d in by_cam.items():
        print(cam, d)
    print(f"[out] {OUT}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
