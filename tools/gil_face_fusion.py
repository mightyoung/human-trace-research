#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-b 多帧合成清晰化（固化路径，离线；只读 GIL）

同轨迹 top-K 帧 → 5 关键点相似变换对齐（规范模板 256×256，ArcFace 5pt 缩放）
→ 中位参考 NCC 拒外（滤转头/眨眼/异动帧）→ 加权（q×清晰度）平均
→ 复过质量门（与最优单帧同空间对比：清晰度 ≥ GATE× & 记录噪声）→ 轻量 unsharp

纪律：
  - 只在「同 tid 同人连续段」内融合（不跨轨迹/跨机位，防身份幻觉）
  - 融合失败或质量门不过 → 输出 rejected（不产图），保留最优单帧
  - 参数全部标注「经验初值，待 val 校准」

用法：
  python tools/gil_face_fusion.py --tag p0b            # 全量
  python tools/gil_face_fusion.py --tag p0b --limit 10 # 前 10 个 tid 快测
输出：fused/ + ab_fused/ + face_fusion_{tag}.json（含 per_tid 与 per_gperson_best）
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gil_face_capture import FACE_DIR  # noqa: E402
from gil_face_reassoc import iter_scanned  # noqa: E402

FUSED_DIR = os.path.join(FACE_DIR, "fused")
ABF_DIR = os.path.join(FACE_DIR, "ab_fused")

# ---- 融合参数（经验初值，待 val 校准）----
CANVAS = 256
TPL = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                [41.5493, 92.3655], [70.7299, 92.2041]], np.float32) * (CANVAS / 112.0)
RESID_MAX = 10.0         # 对齐残差门（规范空间 px，粗筛；精对齐交给 ECC）
ECC_MIN = 0.85           # ECC 相关系数门（低于则拒帧，防鬼影）
NCC_MIN = 0.80           # 与参考帧 NCC 拒外线（防表情/姿态突变帧）
FUSE_GAMMA = 1.5         # 逐像素清晰度权重指数（平坦区趋于平均，纹理区跟随最锐帧）
AGREE_T = 14.0           # 结构一致度尺度（灰度级）：|diff| 在此量级权重衰减
AGREE_P = 2.0            # 一致度掩膜指数（不一致处权重→0，输出跟随参考帧）
GATE_SHARP_RATIO = 0.55  # fused_lap ≥ 0.55 × single_lap 才接受（经 AB 目检校准；经验初值）
UNSHARP_AMT, UNSHARP_SIGMA = 0.25, 1.0


def align(img, kps5):
    """5 点相似变换对齐到规范模板。返回 (warped, resid)。"""
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), TPL, method=cv2.LMEDS)
    if M is None:
        return None, 999.0
    warped = cv2.warpAffine(img, M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    proj = cv2.transform(kps5.reshape(-1, 1, 2).astype(np.float32), M).reshape(5, 2)
    resid = float(np.mean(np.linalg.norm(proj - TPL, axis=1)))
    return warped, resid


def metrics(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    hf = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    sigma = float(np.median(np.abs(hf)) / 0.6745)
    return dict(lap=round(lap, 1), sigma=round(sigma, 2), mu=round(float(g.mean()), 1))


def pearson(a, b):
    a = a.ravel().astype(np.float32) - float(a.mean())
    b = b.ravel().astype(np.float32) - float(b.mean())
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a.dot(b) / d) if d > 1e-6 else 0.0


def ecc_align(ref_g, img_g):
    """ECC 亚像素精对齐：求 M 使 warpAffine(img, M, WARP_INVERSE_MAP) ≈ ref。"""
    try:
        cc, M = cv2.findTransformECC(ref_g, img_g, np.eye(2, 3, dtype=np.float32),
                                     cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                     None, 5)
    except cv2.error:
        return None, -1.0
    return M, float(cc)


def make_abf(left, right, label, h=300):
    def fit(im):
        sc = h / im.shape[0]
        return cv2.resize(im, (max(1, int(im.shape[1] * sc)), h))
    l, r = fit(left), fit(right)
    canvas = np.zeros((h + 28, l.shape[1] + r.shape[1] + 8, 3), np.uint8)
    canvas[28:, :l.shape[1]] = l
    canvas[28:, l.shape[1] + 8:] = r
    cv2.putText(canvas, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0b")
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--window", type=float, default=1.0, help="候选帧时间窗（秒，±）")
    ap.add_argument("--gate", type=float, default=None, help="覆盖清晰度接受线（调试用）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ab-top", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(FUSED_DIR, exist_ok=True)
    os.makedirs(ABF_DIR, exist_ok=True)
    for d_ in (FUSED_DIR, ABF_DIR):        # 清旧输出，防跨次运行混图
        for fn in os.listdir(d_):
            if fn.endswith(".jpg"):
                os.remove(os.path.join(d_, fn))
    with open(os.path.join(FACE_DIR, f"face_shot_{args.tag}.json"), encoding="utf-8") as f:
        shots = json.load(f)["shots"]
    with open(os.path.join(FACE_DIR, f"face_best_{args.tag}.json"), encoding="utf-8") as f:
        best = json.load(f)
    gp_label = best.get("gp_label", {})

    groups = defaultdict(list)
    for s in shots:
        if s.get("usable") and s.get("tid") is not None:
            groups[(s["cam"], s["tid"])].append(s)
    items = sorted(groups.items())
    if args.limit:
        items = items[: args.limit]
    print(f"tids with usable faces: {len(items)}", flush=True)

    records = []
    ab_items = []
    n_ok = n_skip = 0
    t0 = time.time()
    for (cam, tid), lst in items:
        lst.sort(key=lambda s: -s["q"])
        cand = lst[: args.max_k]
        rec = dict(cam=cam, tid=tid, gperson_id=cand[0].get("gperson_id"),
                   k_total=len(lst), k_used=len(cand))
        if len(cand) < args.min_k:
            rec["verdict"] = "skip_few_frames"
            records.append(rec)
            n_skip += 1
            continue
        # ---- 时间窗筛选（防跨时姿态漂移把平均糊掉）----
        tbest = cand[0]["t"]
        win = cand
        for W in (args.window, args.window * 2.5):
            win = [s for s in cand if abs(s["t"] - tbest) <= W]
            if len(win) >= args.min_k:
                break
        rec["window"] = W
        rec["k_window"] = len(win)
        rec["k_used"] = len(win)
        if len(win) < args.min_k:
            rec["verdict"] = "skip_few_frames"
            records.append(rec)
            n_skip += 1
            continue
        img_by_fr = dict(iter_scanned([s["frame"] for s in win], cam))
        aligned, rej = [], []
        for s in win:
            img = img_by_fr.get(s["frame"])
            if img is None:
                rej.append(dict(frame=s["frame"], why="read_fail"))
                continue
            kps = np.array(s["kps"], np.float32).reshape(5, 2)
            wimg, resid = align(img, kps)
            if wimg is None or resid > RESID_MAX:
                rej.append(dict(frame=s["frame"], why="resid", resid=round(resid, 2)))
                continue
            aligned.append(dict(s=s, img=wimg, resid=resid))
        if len(aligned) < args.min_k:
            rec["verdict"] = "skip_align"
            rec["rej"] = rej[:4]
            records.append(rec)
            n_skip += 1
            continue
        # ---- ECC 亚像素精对齐（以最高 q 帧为参考）+ NCC 拒外 ----
        refg = cv2.cvtColor(aligned[0]["img"], cv2.COLOR_BGR2GRAY).astype(np.float32)
        kept = [aligned[0]]
        aligned[0]["ecc"] = 1.0
        for a in aligned[1:]:
            g = cv2.cvtColor(a["img"], cv2.COLOR_BGR2GRAY).astype(np.float32)
            M, cc = ecc_align(refg, g)
            if M is None or cc < ECC_MIN:
                rej.append(dict(frame=a["s"]["frame"], why="ecc", ecc=round(cc, 3)))
                continue
            a["img"] = cv2.warpAffine(a["img"], M, (CANVAS, CANVAS),
                                      flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REPLICATE)
            a["ecc"] = round(cc, 3)
            ncc = pearson(refg, cv2.cvtColor(a["img"], cv2.COLOR_BGR2GRAY).astype(np.float32))
            a["ncc"] = ncc
            if ncc >= NCC_MIN:
                kept.append(a)
            else:
                rej.append(dict(frame=a["s"]["frame"], why="ncc", ncc=round(ncc, 3)))
        if len(kept) < args.min_k:
            rec["verdict"] = "skip_ncc"
            rec["rej"] = rej[:4]
            records.append(rec)
            n_skip += 1
            continue
        # 清晰度加权 × 结构一致度掩膜融合：
        #   纹理区跟随最锐帧（保清晰）；与参考帧结构不一致处（转头/位移）权重→0 → 跟随参考帧（防鬼影/混合时刻）
        refg32 = cv2.cvtColor(kept[0]["img"], cv2.COLOR_BGR2GRAY).astype(np.float32)
        base, smaps, agrees, imgs = [], [], [], []
        for a in kept:
            g = cv2.cvtColor(a["img"], cv2.COLOR_BGR2GRAY).astype(np.float64)
            a["lap"] = round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)
            gf = g.astype(np.float32)
            smap = cv2.boxFilter(np.abs(cv2.Laplacian(gf, cv2.CV_32F)), -1, (5, 5))
            adiff = cv2.boxFilter(np.abs(gf - refg32), -1, (7, 7))
            agree = 1.0 / (1.0 + (adiff / AGREE_T) ** 2)
            smaps.append(smap)
            agrees.append(agree)
            base.append(max(1.0, a["s"]["q"] * a["lap"]))
            imgs.append(a["img"].astype(np.float32))
        S = np.stack(smaps)
        A = np.stack(agrees)
        Wpx = (np.array(base, np.float32)[:, None, None]
               * np.power(S + float(np.median(S)) + 1e-3, FUSE_GAMMA)
               * np.power(A + 1e-3, AGREE_P))
        Wpx = Wpx / Wpx.sum(axis=0, keepdims=True)
        fused = (Wpx[..., None] * np.stack(imgs)).sum(axis=0).clip(0, 255).astype(np.uint8)
        m_fused = metrics(fused)
        m_single = metrics(kept[0]["img"])          # q 最高者（已对齐）
        gate = args.gate if args.gate is not None else GATE_SHARP_RATIO
        accepted = m_fused["lap"] >= gate * m_single["lap"]
        fused_fn = f"fused_{cam}_{tid}.jpg"
        if accepted:
            blur = cv2.GaussianBlur(fused, (0, 0), UNSHARP_SIGMA)
            out_img = cv2.addWeighted(fused, 1 + UNSHARP_AMT, blur, -UNSHARP_AMT, 0)
            cv2.imwrite(os.path.join(FUSED_DIR, fused_fn), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_ok += 1
            ab_items.append((m_fused["lap"] / max(m_single["lap"], 1.0), kept[0]["img"],
                             fused, cam, tid, len(kept), m_single, m_fused))
        rec.update(verdict="accepted" if accepted else "rejected",
                   k_kept=len(kept), k_rejected=len(rej),
                   resid_mean=round(float(np.mean([a["resid"] for a in kept])), 2),
                   ecc_min=round(min(a.get("ecc", -1.0) for a in kept), 3),
                   ncc_min=round(min(a["ncc"] for a in kept[1:]), 3) if len(kept) > 1 else None,
                   single=m_single, fused=m_fused,
                   lap_ratio=round(m_fused["lap"] / max(m_single["lap"], 1.0), 3),
                   fused_file=fused_fn if accepted else None, rej=rej[:4])
        records.append(rec)

    # ---- AB 拼图（前 N）----
    ab_items.sort(key=lambda x: -x[0])
    ab_saved = []
    for ratio, simg, fimg, cam, tid, kk, ms, mf in ab_items[: args.ab_top]:
        label = (f"{cam}-{tid} K={kk} r={ratio:.2f} | single lap={ms['lap']:.0f} sig={ms['sigma']:.1f}"
                 f" | fused lap={mf['lap']:.0f} sig={mf['sigma']:.1f}")
        canvas = make_abf(simg, fimg, label)
        fn = f"abf_{len(ab_saved):03d}_{cam}_{tid}.jpg"
        cv2.imwrite(os.path.join(ABF_DIR, fn), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        ab_saved.append(fn)

    # ---- gperson 级：按 q 选 tid，该 tid 有 accepted fused 则升级 ----
    per_tid_ok = {(r["cam"], r["tid"]): r for r in records if r.get("verdict") == "accepted"}
    gp_best = {}
    for key, lst in best["per_tid"].items():
        if not (lst and lst[0].get("usable")):
            continue
        s0 = lst[0]
        gpid = s0.get("gperson_id")
        if gpid is None:
            continue
        cam, tid_s = key.split("|")
        prev = gp_best.get(gpid)
        if prev is None or s0["q"] > prev["q"]:
            fr = per_tid_ok.get((cam, int(tid_s)))
            gp_best[gpid] = dict(q=s0["q"], cam=cam, tid=int(tid_s),
                                 src="fused" if fr else "single",
                                 fused_file=(fr or {}).get("fused_file"))
    # 复制被选中的 fused → fused/gp_{label}.jpg（先清旧）
    for fn in os.listdir(FUSED_DIR):
        if fn.startswith("gp_"):
            os.remove(os.path.join(FUSED_DIR, fn))
    n_gp_fused = 0
    for gpid, v in gp_best.items():
        if v["src"] == "fused" and v["fused_file"]:
            shutil.copy(os.path.join(FUSED_DIR, v["fused_file"]),
                        os.path.join(FUSED_DIR, f"gp_{gp_label.get(gpid, gpid)}.jpg"))
            n_gp_fused += 1

    out = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), tag=args.tag,
                         params=dict(canvas=CANVAS, min_k=args.min_k, max_k=args.max_k,
                                     window_sec=args.window, resid_max=RESID_MAX,
                                     ecc_min=ECC_MIN, ncc_min=NCC_MIN, fuse_gamma=FUSE_GAMMA,
                                     agree_t=AGREE_T, agree_p=AGREE_P,
                                     gate_sharp_ratio=GATE_SHARP_RATIO,
                                     unsharp=[UNSHARP_AMT, UNSHARP_SIGMA]),
                         note="融合参数均为经验初值，待 val 校准；同 tid 内融合，防身份幻觉"),
               n_tids=len(items), n_accepted=n_ok, n_skipped=n_skip,
               n_gp_fused=n_gp_fused, ab_files=ab_saved,
               per_tid=records, per_gperson_best=gp_best)
    with open(os.path.join(FACE_DIR, f"face_fusion_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"fusion done: tids={len(items)} accepted={n_ok} skipped={n_skip} "
          f"gp_fused={n_gp_fused} ab={len(ab_saved)} ({time.time()-t0:.0f}s) -> {FUSED_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
