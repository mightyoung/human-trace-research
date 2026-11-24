#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 = 场景诊断 + 人脸增强 + A/B 验证（离线固化路径，纯 CV 算子，无 torch）

纪律（P1 硬约束）：
  - 只对 P0 质量池选出的每 tid 最优帧（crops/ 中的图）做增强 —— 固化路径，不进实时链
  - 增强后必须「复过质量门」：q_after > q_before 才接受（accepted），否则丢弃（rejected）
  - 倒影门控：obs.flags reflection>0.5 的实体一律跳过（防把倒影增强成假证据）
  - 一切阈值/参数标注为「经验初值，待 val 校准」

输入：analysis/face/face_best_{tag}.json + crops/ + GIL sqlite
输出：face_enhance_p1.json + scene_diag_p1.json + ab/ 拼图 + crops_enhanced/

用法：
  python tools/gil_face_enhance.py --tag p0            # 全量
  python tools/gil_face_enhance.py --tag p0_test --limit 5
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gil_face_capture import score_face, clamp, STRIDE  # noqa: E402

OC = os.path.join(ROOT, "output_crosscam")
FACE_DIR = os.path.join(OC, "analysis", "face")
CROPS_DIR = os.path.join(FACE_DIR, "crops")
ENH_DIR = os.path.join(FACE_DIR, "crops_enhanced")
AB_DIR = os.path.join(FACE_DIR, "ab")
GIL_DB = os.path.join(OC, "analysis", "gil", "store_v1", "gil.sqlite")

VIDEOS = {
    "305": "media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4",
    "316": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    "318": "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4",
    "320": "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4",
    "323": "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
    "343": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
}

# ---- 场景诊断（F0）阈值：经验初值，待 val 校准 ----
DARK_MU = 65.0        # 全帧亮度中位 < 65 → dark
DARK_RATIO = 0.30     # 暗像素(<35) 占比 > 30% → dark
BACKLIT_OVER = 0.04   # 过曝像素(>245) 占比 > 4% 且中位亮度偏低 → backlit
BRIGHT_MU = 170.0

# ---- 增强路由阈值：经验初值 ----
ROI_DARK = 75.0       # 脸 ROI 亮度 < 75 → 走暗光提亮
ROI_BRIGHT = 185.0    # 脸 ROI 亮度 > 185 → 走压缩
ROI_LOWCT = 30.0      # ROI 对比 < 30 → CLAHE


def median_of(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def diagnose_cam(cam, n_samples=30):
    """抽样 n 帧 → 亮度统计 + 场景标签"""
    vp = os.path.join(ROOT, VIDEOS[cam])
    cap = cv2.VideoCapture(vp)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0.05 * n, 0.95 * n, n_samples).astype(int)
    mus, sds, overs, darks, tb = [], [], [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if not ok:
            continue
        small = cv2.resize(img, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mus.append(float(gray.mean()))
        sds.append(float(gray.std()))
        overs.append(float((gray > 245).mean()))
        darks.append(float((gray < 35).mean()))
        tb.append(float(gray[:120].mean() - gray[240:].mean()))  # 顶-底亮度差
    cap.release()
    mu_m = median_of(mus); sd_m = median_of(sds)
    over_m = median_of(overs); dark_m = median_of(darks); tb_m = median_of(tb)
    if mu_m is None:
        return dict(label="unknown")
    if dark_m > DARK_RATIO or mu_m < DARK_MU:
        label = "dark"
    elif over_m > BACKLIT_OVER and mu_m < 130:
        label = "backlit"
    elif mu_m > BRIGHT_MU:
        label = "bright"
    else:
        label = "normal"
    return dict(label=label, mu=round(mu_m, 1), sd=round(sd_m, 1),
                over_ratio=round(over_m, 4), dark_ratio=round(dark_m, 4),
                top_minus_bottom=round(tb_m, 1), n_samples=len(mus),
                note="阈值经验初值，待 val 校准")


def apply_l_gamma(bgr, gamma):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lut = ((np.arange(256) / 255.0) ** gamma * 255).clip(0, 255).astype(np.uint8)
    lab[:, :, 0] = cv2.LUT(lab[:, :, 0], lut)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_l(bgr, clip=2.0, tile=4):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def route(mu, sd, cam_label):
    if mu < ROI_DARK:
        return "backlit_gamma" if cam_label == "backlit" else "dark_gamma"
    if mu > ROI_BRIGHT:
        return "bright_compress"
    if sd < ROI_LOWCT:
        return "low_contrast_clahe"
    return None


def enhance(img, mode, mu):
    """返回 (enh_img, params)。强度全部受限（防过增强）。"""
    if mode == "dark_gamma":
        target = 128.0
        g = math.log(target / 255.0) / math.log(max(mu, 1.0) / 255.0)
        g = clamp(g, 0.45, 1.0)
        out = apply_l_gamma(img, g)
        params = dict(mode=mode, gamma=round(g, 3))
        if mu < 60:
            out = cv2.bilateralFilter(out, 5, 25, 25)
            params["bilateral"] = 5
        return out, params
    if mode == "backlit_gamma":
        target = 140.0
        g = math.log(target / 255.0) / math.log(max(mu, 1.0) / 255.0)
        g = clamp(g, 0.40, 1.0)
        out = apply_l_gamma(img, g)
        out = clahe_l(out, clip=1.5, tile=4)
        return out, dict(mode=mode, gamma=round(g, 3), clahe=1.5)
    if mode == "bright_compress":
        target = 168.0
        g = math.log(target / 255.0) / math.log(max(mu, 1.0) / 255.0)
        g = clamp(g, 1.0, 1.6)
        out = apply_l_gamma(img, g)
        return out, dict(mode=mode, gamma=round(g, 3))
    if mode == "low_contrast_clahe":
        out = clahe_l(img, clip=2.5, tile=4)
        return out, dict(mode=mode, clahe=2.5)
    return None, None


def load_reflections():
    """obs_id -> reflection flag value"""
    import sqlite3
    con = sqlite3.connect(GIL_DB)
    cur = con.cursor()
    refl = {}
    for r in cur.execute("select id, flags from obs"):
        if r[1]:
            try:
                fl = json.loads(r[1])
                if isinstance(fl, dict) and "reflection" in fl:
                    refl[r[0]] = float(fl["reflection"])
            except Exception:
                pass
    con.close()
    return refl


def make_ab(btxt, bimg, aimg, label):
    """左右拼图（英文标签；cv2 putText 不支持中文）"""
    h = max(bimg.shape[0], aimg.shape[0])
    def pad(im):
        if im.shape[0] == h:
            return im
        c = np.zeros((h, im.shape[1], 3), np.uint8)
        c[:im.shape[0]] = im
        return c
    b, a = pad(bimg), pad(aimg)
    canvas = np.zeros((h + 26, b.shape[1] + a.shape[1] + 6, 3), np.uint8)
    canvas[26:, :b.shape[1]] = b
    canvas[26:, b.shape[1] + 6:] = a
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0")
    ap.add_argument("--cams", default=None, help="默认=best json 里的 cams")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ab-top", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(ENH_DIR, exist_ok=True)
    ab_dir = AB_DIR if args.tag == "p0" else os.path.join(FACE_DIR, f"ab_{args.tag}")
    os.makedirs(ab_dir, exist_ok=True)
    with open(os.path.join(FACE_DIR, f"face_best_{args.tag}.json"), encoding="utf-8") as f:
        best = json.load(f)
    crop_map = best["crop_map"]
    gp_label = best.get("gp_label", {})
    cams = [c.strip() for c in args.cams.split(",")] if args.cams else best["meta"]["cams"]

    # ---- F0 场景诊断 ----
    scene = {}
    for cam in cams:
        scene[cam] = diagnose_cam(cam)
        print(f"[scene] {cam}: {scene[cam]['label']} mu={scene[cam]['mu']} "
              f"over={scene[cam]['over_ratio']} dark={scene[cam]['dark_ratio']}", flush=True)
    with open(os.path.join(FACE_DIR, f"scene_diag_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), scene=scene), f,
                  ensure_ascii=False, indent=1)

    refl = load_reflections()

    # ---- 逐 tid 增强 ----
    records = []
    ab_items = []
    items = sorted(crop_map.items())
    if args.limit:
        items = items[: args.limit]
    n_done = 0
    for key, fn in items:
        cam, tid_s = key.split("_", 1)
        gp = None
        lst = best["per_tid"].get(f"{cam}|{tid_s}")
        if not lst:
            continue
        s0 = lst[0]
        gp = s0.get("gperson_id")
        rec = dict(cam=cam, tid=int(tid_s), gperson_id=gp,
                   gp_label=gp_label.get(str(gp)) if gp else None,
                   frame=s0["frame"], crop=fn)
        # 倒影门控
        oid = s0.get("obs_id")
        if oid is not None and refl.get(oid, 0.0) > 0.5:
            rec["verdict"] = "skip_reflection"
            records.append(rec)
            continue
        img = cv2.imread(os.path.join(CROPS_DIR, fn))
        if img is None:
            rec["verdict"] = "missing_crop"
            records.append(rec)
            continue
        # 重建 det（视频坐标 → crop 坐标）
        x1, y1, x2, y2 = s0["face"]
        mw, mh = (x2 - x1) * 0.35, (y2 - y1) * 0.45
        a = max(0, int(x1 - mw)); b = max(0, int(y1 - mh))
        kps = np.array(s0["kps"], np.float32).reshape(5, 2)
        det = np.concatenate([[x1 - a, y1 - b, x2 - a, y2 - b],
                              (kps - np.array([a, b], np.float32)).ravel(),
                              [s0.get("det_score", 0.0)]]).astype(np.float32)
        before = score_face(img, det)
        if before is None:
            rec["verdict"] = "score_fail"
            records.append(rec)
            continue
        mode = route(before["mu"], before["sd"], scene.get(cam, {}).get("label", "normal"))
        rec["before"] = dict(q=before["q"], mu=before["mu"], sd=before["sd"], lap=before["lap"])
        if mode is None:
            rec["verdict"] = "skip_normal"
            records.append(rec)
            continue
        enh, params = enhance(img, mode, before["mu"])
        after = score_face(enh, det)
        rec["mode"] = params.get("mode", mode)
        rec["params"] = params
        rec["after"] = dict(q=after["q"], mu=after["mu"], sd=after["sd"], lap=after["lap"])
        accepted = after["q"] > before["q"]
        rec["verdict"] = "accepted" if accepted else "rejected"
        rec["dq"] = round(after["q"] - before["q"], 4)
        if accepted:
            enh_fn = fn.replace("tid_", "enh_").replace(".jpg", "_enh.jpg")
            cv2.imwrite(os.path.join(ENH_DIR, enh_fn), enh, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rec["enh_file"] = enh_fn
            label = (f"{rec['mode']} tid={cam}-{tid_s} "
                     f"q {before['q']:.2f}->{after['q']:.2f} mu {before['mu']:.0f}->{after['mu']:.0f}")
            ab_items.append((rec["dq"], label, img, enh, fn))
        records.append(rec)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  ...{n_done} processed", flush=True)

    # ---- A/B 拼图（按 dq 排序取前 N）----
    ab_items.sort(key=lambda x: -x[0])
    ab_saved = []
    for dq, label, bimg, aimg, fn in ab_items[: args.ab_top]:
        canvas = make_ab(None, bimg, aimg, label)
        out_fn = f"ab_{len(ab_saved):03d}_{fn.replace('.jpg','')}.jpg"
        cv2.imwrite(os.path.join(ab_dir, out_fn), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        ab_saved.append(out_fn)

    # ---- 汇总 ----
    agg = defaultdict(lambda: dict(n=0, acc=0, skip=0, dq_sum=0.0, modes=defaultdict(int)))
    for r in records:
        a = agg[r["cam"]]
        a["n"] += 1
        v = r["verdict"]
        if v in ("skip_normal", "skip_reflection", "missing_crop", "score_fail"):
            a["skip"] += 1
        elif v == "accepted":
            a["acc"] += 1
            a["dq_sum"] += r["dq"]
            a["modes"][r.get("mode", "?")] += 1
        else:
            a["modes"][r.get("mode", "?") + "(rej)"] += 1
    summary = {}
    for cam, a in agg.items():
        summary[cam] = dict(n=a["n"], accepted=a["acc"], skipped=a["skip"],
                            acc_rate=round(a["acc"] / max(a["n"], 1), 3),
                            mean_dq_accepted=round(a["dq_sum"] / max(a["acc"], 1), 4),
                            modes=dict(a["modes"]), scene=scene.get(cam, {}).get("label"))
    out = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                         tag=args.tag, n_records=len(records), n_ab=len(ab_saved),
                         note="增强参数与路由阈值均为经验初值，待 val 校准"),
               scene=scene, summary=summary, ab_files=ab_saved, records=records)
    with open(os.path.join(FACE_DIR, f"face_enhance_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"enhance done: {len(records)} records, {len(ab_saved)} AB images -> {FACE_DIR}")


if __name__ == "__main__":
    main()
