#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L2 rPPG SNR 评估 v0（先评估，不做判定）

目的：在 316/343 大脸高分段与"长时长小脸"代表（305 坐姿）上评估
远程光电容积脉搏（rPPG）在本项目画质下是否留下可测信号。

方法（可行性筛查级）：
- 窗口：轨迹段内取连续窗口（face w>=70px 的段；305 坐姿另取 15s 窗口）
- ROI：面框中心 60%×60%（面框在 shot 采样间线性插值）
- 信号：G 通道与 (2G-R-B) 的 ROI 均值序列，去 1s 滑动均值（去趋势）
- SNR：Hann 窗 + 补零 FFT，PSD；SNR = 10log10(P[0.7-3.0Hz] / P([0.1-0.6]∪[3.2-8]Hz))
- 峰值频率（42-180 bpm 范围）与运动量一并记录
- 对照：① 静态背景块 ② 后脑勺轨迹（负对照）③ 合成 1Hz 正弦（指标自校验，应高 SNR）

输出：output_crosscam/analysis/face/rppg_snr_v0.json
用法：python tools/gil_face_rppg.py
"""
import json
import os
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "face")
VIDEOS = {
    "305": "media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4",
    "316": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    "323": "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
    "343": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
}
FPS = 25.0
BAND = (0.7, 3.0)      # 42 - 180 bpm
NOISE = [(0.1, 0.6), (3.2, 8.0)]


def snr_fft(x, fps=FPS):
    """去趋势 + Hann + 补零 FFT，返回 SNR(dB) / 峰值频率 / 频点功率诊断"""
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < 25:
        return None
    # 去 1s 滑动均值
    k = int(round(fps))
    kern = np.ones(k) / k
    trend = np.convolve(x, kern, mode="same")
    x = x - trend
    x = x - x.mean()
    if np.allclose(x, 0):
        return None
    w = np.hanning(n)
    nfft = int(2 ** np.ceil(np.log2(n * 4)))
    spec = np.fft.rfft(x * w, nfft)
    psd = (np.abs(spec) ** 2)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fps)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    noise = np.zeros_like(band)
    for lo, hi in NOISE:
        noise |= (freqs >= lo) & (freqs <= hi)
    pb = psd[band].sum()
    pn = psd[noise].sum()
    if pn <= 0:
        return None
    snr = 10 * np.log10(pb / pn)
    fb = freqs[band]
    peak = fb[int(np.argmax(psd[band]))]
    return dict(snr_db=round(float(snr), 2), peak_hz=round(float(peak), 3),
                peak_bpm=round(float(peak * 60), 1), n=n)


def box_at(shots, frame):
    """面框在采样间线性插值（clamp 边界）"""
    if frame <= shots[0]["frame"]:
        return shots[0]["face"]
    if frame >= shots[-1]["frame"]:
        return shots[-1]["face"]
    for a, b in zip(shots, shots[1:]):
        if a["frame"] <= frame <= b["frame"]:
            if b["frame"] == a["frame"]:
                return a["face"]
            t = (frame - a["frame"]) / (b["frame"] - a["frame"])
            return [a["face"][i] + t * (b["face"][i] - a["face"][i]) for i in range(4)]
    return shots[-1]["face"]


def read_window(cam, f0, f1, shots=None, patch=None):
    """逐帧读窗口，返回 G / 2G-R-B 序列与运动统计。patch=(x1,y1,x2,y2) 时用固定区块。"""
    vp = os.path.join(ROOT, VIDEOS[cam])
    cap = cv2.VideoCapture(vp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    gs, cs, cxs, cys, ws_ = [], [], [], [], []
    for f in range(f0, f1):
        ok, img = cap.read()
        if not ok:
            break
        if patch is not None:
            x1, y1, x2, y2 = patch
            roi = img[y1:y2, x1:x2]
        else:
            x1, y1, x2, y2 = box_at(shots, f)
            bx, by = (x2 - x1), (y2 - y1)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = bx * 0.30, by * 0.30      # 中心 60%
            rx1, ry1, rx2, ry2 = int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)
            roi = img[max(0, ry1):ry2, max(0, rx1):rx2]
            cxs.append(cx); cys.append(cy); ws_.append(bx)
        if roi.size == 0:
            continue
        b, g, r = roi[:, :, 0].mean(), roi[:, :, 1].mean(), roi[:, :, 2].mean()
        gs.append(g)
        cs.append(2 * g - r - b)
    cap.release()
    motion = float(np.hypot(np.diff(cxs), np.diff(cys)).mean()) if len(cxs) > 2 else None
    w_med = float(np.median(ws_)) if ws_ else None
    return np.array(gs), np.array(cs), motion, w_med


def main():
    shots = json.load(open(f"{FACE_DIR}/face_shot_p0c.json", encoding="utf-8"))["shots"]
    by_tid = defaultdict(list)
    for s in shots:
        if s["tid"] is not None:
            by_tid[(s["cam"], s["tid"])].append(s)
    for k in by_tid:
        by_tid[k].sort(key=lambda s: s["frame"])

    cases = []
    # A) 343 大脸最长段（行人，3-5s）
    for cam, tid, f0, f1, note in [
        ("343", 44, 9915, 10035, "big_face_4.8s"),
        ("343", 13, 4015, 4130, "big_face_4.6s"),
        ("343", 14, 4000, 4100, "big_face_4.0s"),
        ("343", 50, 10955, 11050, "big_face_3.8s"),
        ("343", 55, 12835, 12920, "big_face_3.4s"),
        ("316", 32, 4705, 4775, "big_face_2.8s"),
        ("316", 127, 13450, 13500, "big_face_2.0s"),
    ]:
        cases.append(dict(kind="face", cam=cam, tid=tid, f0=f0, f1=f1, note=note,
                          shots=by_tid[(cam, tid)]))
    # B) 305 坐姿（时长充足、脸小）
    cases.append(dict(kind="face", cam="305", tid=2, f0=2000, f1=2375, note="seated_15s_smallface",
                      shots=by_tid[("305", 2)]))
    # C) 负对照：323|2 后脑勺（长时段）
    cases.append(dict(kind="face_neg", cam="323", tid=2, f0=700, f1=1075, note="backhead_15s",
                      shots=by_tid[("323", 2)]))
    # D) 静态背景块对照（每机位两块，15s）
    ctrl_patches = {
        "316": [(80, 80, 200, 200), (2360, 80, 2480, 200)],
        "343": [(80, 80, 200, 200), (2360, 80, 2480, 200)],
        "305": [(80, 80, 200, 200)],
        "323": [(80, 80, 200, 200)],
    }
    ctrl_win = {"316": (3000, 3375), "343": (3000, 3375), "305": (3000, 3375), "323": (3000, 3375)}

    results = []
    for c in cases:
        g, csig, motion, wmed = read_window(c["cam"], c["f0"], c["f1"], shots=c["shots"])
        sg = snr_fft(g)
        sc = snr_fft(csig)
        rec = dict(kind=c["kind"], cam=c["cam"], tid=c["tid"], f0=c["f0"], f1=c["f1"],
                   dur=round((c["f1"] - c["f0"]) / FPS, 1), note=c["note"],
                   w_med=None if wmed is None else round(wmed, 0),
                   motion_px=None if motion is None else round(motion, 2),
                   snr_g=sg, snr_2grb=sc)
        results.append(rec)
        s1 = f"G SNR={sg['snr_db']:+.1f}dB peak={sg['peak_bpm']:.0f}bpm" if sg else "G SNR=n/a"
        print(f"[face] {c['cam']}|{c['tid']} {c['note']} dur={rec['dur']}s w={rec['w_med']} "
              f"motion={rec['motion_px']}px -> {s1}", flush=True)

    for cam, patches in ctrl_patches.items():
        f0, f1 = ctrl_win[cam]
        for i, p in enumerate(patches):
            g, csig, _, _ = read_window(cam, f0, f1, patch=p)
            sg = snr_fft(g)
            rec = dict(kind="ctrl_patch", cam=cam, tid=None, f0=f0, f1=f1,
                       dur=round((f1 - f0) / FPS, 1), note=f"patch{i+1}", patch=list(p),
                       w_med=None, motion_px=None, snr_g=sg, snr_2grb=snr_fft(csig))
            results.append(rec)
            print(f"[ctrl] {cam} {rec['note']} {p} -> G SNR={sg['snr_db']:+.1f}dB" if sg else "n/a", flush=True)

    # E) 合成正弦自校验：1.2Hz 正弦 + 噪声
    rng = np.random.default_rng(0)
    t = np.arange(0, 375) / FPS
    synth = 1.0 * np.sin(2 * np.pi * 1.2 * t) + 1.0 * rng.normal(0, 1, len(t))
    ss = snr_fft(synth)
    print(f"[synthetic] 1.2Hz sine+noise -> SNR={ss['snr_db']:+.1f}dB peak={ss['peak_hz']}Hz (指标自校验)", flush=True)

    out = dict(
        meta=dict(created=__import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                  note="rPPG SNR 评估（可行性筛查，非判定）；band 0.7-3.0Hz；noise 0.1-0.6+3.2-8Hz"),
        cases=results, synthetic_check=ss,
    )
    out_path = os.path.join(FACE_DIR, "rppg_snr_v0.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[rppg] -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
