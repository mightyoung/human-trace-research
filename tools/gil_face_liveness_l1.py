#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 活体旁路 v0：MiniFASNet-V2 ONNX 对 P0 人脸产物打 spoof score（只读，不改 GIL）

模型：models/face/minifasnet_v2.onnx（Silent-Face-Anti-Spoofing 2.7_80x80_MiniFASNetV2
的 ONNX 导出）。★ 关键口径（实测校准）：
- 预处理：2.7× 居中扩展裁剪（按图像边界封顶）→ 80×80 BGR → **raw float [0,255]**。
  （/255 归一化会让网络饱和为恒定输出，已实测排除；raw 下 Lena=Real、噪声/灰块=Fake。）
- 类序：官方 test.py 判据 argmax==1 → Real Face → **class1 = live，class0/2 = 攻击类**；
  spoof score = 1 - p(live) = p0 + p2。

打分集合：
- tid_best   ：face_best_{tag}.json 全部 per-tid 最优脸（从源视频按规范重裁）
- fused      ：多帧合成图（融合画布 256→80，方法标注为 fused_canvas）
- fp         ：L0 簇采样 + 已知非人/困难样本（后脑勺、窗框纹理、平面纹理等）
- reflection ：R4 倒影交叉引用样本（倒影里是真人 → 预期 live，用于验证正交性）

输出：analysis/face/liveness_l1_minifas_{out-tag}.json + 证据拼图
用法：python tools/gil_face_liveness_l1.py --tag p0c
"""
import argparse
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "face")
MODEL = os.path.join(ROOT, "models", "face", "minifasnet_v2.onnx")
OUT_DIR = os.path.join(FACE_DIR, "debug_assoc", "liveness")

VIDEOS = {
    "305": "media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4",
    "316": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    "318": "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4",
    "320": "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4",
    "323": "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
    "343": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
}
FPS = 25.0
# 已知/疑似非人样本，从 N（正常集）中剔除单独报告
EXCLUDE_FROM_N = {("323", 2), ("343", 12)}


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fas_crop(img, face):
    """上游 2.7× 规范：居中扩展 2.7 倍（按图边界封顶）→ 80x80 BGR"""
    x1, y1, x2, y2 = [float(v) for v in face]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = img.shape[:2]
    scale = min((W - 1) / bw, (H - 1) / bh, 2.7)
    nw, nh = bw * scale, bh * scale
    nx1 = max(0, int(round(cx - nw / 2)))
    ny1 = max(0, int(round(cy - nh / 2)))
    nx2 = min(W, int(round(cx + nw / 2)))
    ny2 = min(H, int(round(cy + nh / 2)))
    crop = img[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None, None
    crop80 = cv2.resize(crop, (80, 80))
    return crop80, crop


def score_batch(sess, imgs):
    if not imgs:
        return []
    # raw [0,255]（实测口径，勿改 /255）
    x = np.stack([im.astype(np.float32) for im in imgs]).transpose(0, 3, 1, 2)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    return softmax(out.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0c")
    ap.add_argument("--out-tag", default="v0")
    ap.add_argument("--max-fp", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    print(f"[l1] model loaded {os.path.basename(MODEL)}", flush=True)

    with open(os.path.join(FACE_DIR, f"face_best_{args.tag}.json"), encoding="utf-8") as f:
        best = json.load(f)
    with open(os.path.join(FACE_DIR, f"face_shot_{args.tag}.json"), encoding="utf-8") as f:
        shots = json.load(f)["shots"]
    with open(os.path.join(FACE_DIR, "liveness_rule_v0.json"), encoding="utf-8") as f:
        l0 = json.load(f)

    shots_by_cf = defaultdict(list)      # (cam, frame) -> shots
    shots_by_cft = defaultdict(list)     # (cam, frame, tid) -> shots
    for s in shots:
        shots_by_cf[(s["cam"], s["frame"])].append(s)
        shots_by_cft[(s["cam"], s["frame"], s.get("tid"))].append(s)

    # ---- 组装打分工单 ----
    jobs = []   # dict(cam, frame, face, set, ref, note)

    # 1) tid_best
    for key, lst in best["per_tid"].items():
        s0 = lst[0]
        if s0.get("tid") is None or s0.get("frame") is None:
            continue
        jobs.append(dict(cam=s0["cam"], frame=s0["frame"], face=s0["face"], set="tid_best",
                         ref=key, note=""))

    # 2) fused（直接读图，方法=fused_canvas）
    fused_dir = os.path.join(FACE_DIR, "fused")
    fused_items = []
    if os.path.isdir(fused_dir):
        for fn in sorted(os.listdir(fused_dir)):
            if not fn.endswith(".jpg"):
                continue
            img = cv2.imread(os.path.join(fused_dir, fn))
            if img is None:
                continue
            fused_items.append((fn, cv2.resize(img, (80, 80))))

    # 3) fp：L0 簇采样
    for rec in l0["static_clusters"]:
        for smp in rec["samples"]:
            jobs.append(dict(cam=rec["cam"], frame=smp["frame"], face=smp["face"], set="fp",
                             ref=f"{rec['cam']}c{rec['id']}", note=rec["rule"]))
    # 3b) 指定补充样本（后脑勺 tid2 / 320 背影）
    for cam, fr, note in [("323", 275, "tid323_2_backhead"), ("323", 560, "tid323_2_backhead_usable"),
                          ("323", 1150, "tid323_2_backhead"), ("323", 1720, "tid323_2_backhead"),
                          ("320", 310, "cleaner_back")]:
        cand = shots_by_cf.get((cam, fr), [])
        if not cand:
            continue
        s0 = cand[0]
        jobs.append(dict(cam=cam, frame=fr, face=s0["face"], set="fp", ref=f"{cam}f{fr}", note=note))

    # 4) reflection：R4 usable 样本（跨机位取前 8 个）
    seen_cam = defaultdict(int)
    refl_items = []
    for it in l0["reflection_linked"]["items"]:
        if not it["usable"]:
            continue
        cam = it["cam"]
        if seen_cam[cam] >= 2:
            continue
        cand = shots_by_cft.get((cam, it["frame"], it["tid"]), [])
        if not cand:
            continue
        seen_cam[cam] += 1
        refl_items.append(cand[0])
        if len(refl_items) >= 8:
            break
    for s0 in refl_items:
        jobs.append(dict(cam=s0["cam"], frame=s0["frame"], face=s0["face"], set="reflection",
                         ref=f"{s0['cam']}|{s0['tid']}", note="refl_usable"))

    print(f"[l1] jobs: {len(jobs)} + fused {len(fused_items)}", flush=True)

    # ---- 逐机位抽取 + 打分 ----
    results = {"tid_best": [], "fp": [], "reflection": []}
    by_cam = defaultdict(list)
    for j in jobs:
        by_cam[j["cam"]].append(j)

    card_imgs = []   # (img80_upscaled, label, group)
    for cam, items in sorted(by_cam.items()):
        vp = os.path.join(ROOT, VIDEOS[cam])
        cap = cv2.VideoCapture(vp)
        for j in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, j["frame"])
            ok, img = cap.read()
            if not ok:
                continue
            c80, _ = fas_crop(img, j["face"])
            if c80 is None:
                continue
            p = score_batch(sess, [c80])[0]
            # class1 = live；class0/class2 = 攻击类（官方 test.py 口径）
            rec = dict(cam=j["cam"], frame=j["frame"], ref=j["ref"], note=j["note"],
                       live=round(float(p[1]), 4), att_a=round(float(p[0]), 4),
                       att_b=round(float(p[2]), 4), spoof=round(float(p[0] + p[2]), 4),
                       face=[round(float(v), 1) for v in j["face"]])
            results[j["set"]].append(rec)
            tag = f"{j['cam']}|{j['ref']}" if j["set"] == "tid_best" else j["ref"]
            card_imgs.append((c80, f"{tag} s={p[0]+p[2]:.2f}", j["set"]))
        cap.release()
        print(f"  [{cam}] scored {len([r for r in items if True])} items", flush=True)

    # fused 打分
    fused_results = []
    for fn, c80 in fused_items:
        p = score_batch(sess, [c80])[0]
        fused_results.append(dict(file=fn, live=round(float(p[1]), 4),
                                  att_a=round(float(p[0]), 4), att_b=round(float(p[2]), 4),
                                  spoof=round(float(p[0] + p[2]), 4), method="fused_canvas"))
        card_imgs.append((c80, f"{fn.replace('.jpg','')} s={p[0]+p[2]:.2f}", "fused"))

    # ---- 统计 ----
    def stats_of(recs):
        if not recs:
            return {}
        v = np.array([r["spoof"] for r in recs])
        return dict(n=len(v), median=round(float(np.median(v)), 4),
                    mean=round(float(v.mean()), 4), p10=round(float(np.percentile(v, 10)), 4),
                    p90=round(float(np.percentile(v, 90)), 4), max=round(float(v.max()), 4),
                    frac_gt_0p5=round(float((v > 0.5).mean()), 3))

    n_recs = [r for r in results["tid_best"] if (r["cam"], int(r["ref"].split("|")[1])) not in EXCLUDE_FROM_N]
    excl_recs = [r for r in results["tid_best"] if (r["cam"], int(r["ref"].split("|")[1])) in EXCLUDE_FROM_N]

    # AUC(fp vs N)：P(spoof_fp > spoof_n)
    auc = None
    nv = np.array([r["spoof"] for r in n_recs])
    fv = np.array([r["spoof"] for r in results["fp"]])
    if len(nv) >= 5 and len(fv) >= 3:
        auc = float((fv[:, None] > nv[None, :]).mean() + 0.5 * (fv[:, None] == nv[None, :]).mean())

    out = dict(
        meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), tag=args.tag,
                  model="minifasnet_v2.onnx (Silent-Face-Anti-Spoofing 2.7_80x80 V2, ONNX)",
                  preprocess="2.7x center crop -> 80x80 BGR -> raw [0,255] (div255 会饱和，实测排除)",
                  label_map="class1=live（官方 test.py: argmax==1 -> Real Face）；spoof=p0+p2",
                  note="只读产物；跨域泛化未验证，先看分布不设阈值"),
        sets=dict(tid_best=results["tid_best"], fp=results["fp"], reflection=results["reflection"],
                  fused=fused_results),
        stats=dict(
            N_normal=stats_of(n_recs), N_excluded=stats_of(excl_recs),
            fp=stats_of(results["fp"]), reflection=stats_of(results["reflection"]),
            fused=stats_of(fused_results), auc_fp_vs_N=auc,
        ),
        exclude_from_N=[[c, t] for c, t in sorted(EXCLUDE_FROM_N)],
    )
    out_path = os.path.join(FACE_DIR, f"liveness_l1_minifas_{args.out_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 拼图（fp + reflection + 抽样 N + fused 各取若干）----
    def pick(group, k):
        return [c for c in card_imgs if c[2] == group][:k]
    picks = pick("fp", 18) + pick("reflection", 8) + pick("tid_best", 12) + pick("fused", 6)
    cell = 150
    cols = 8
    rows = (len(picks) + cols - 1) // cols
    canvas = np.full((rows * (cell + 20) + 12, cols * (cell + 8) + 8, 3), 248, np.uint8)
    for i, (c80, lab, grp) in enumerate(picks):
        big = cv2.resize(c80, (cell, cell))
        r, c = divmod(i, cols)
        y0 = r * (cell + 20) + 12
        x0 = c * (cell + 8) + 8
        canvas[y0:y0 + cell, x0:x0 + cell] = big
        color = (30, 30, 30) if grp in ("tid_best", "fused") else (0, 0, 200)
        cv2.putText(canvas, lab[:26], (x0, y0 + cell + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1, cv2.LINE_AA)
    montage = os.path.join(OUT_DIR, f"liveness_l1_cards_{args.out_tag}.jpg")
    cv2.imwrite(montage, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"[l1] N(normal) n={len(n_recs)} median_spoof={stats_of(n_recs).get('median')}", flush=True)
    print(f"[l1] fp n={len(results['fp'])} median_spoof={stats_of(results['fp']).get('median')}", flush=True)
    print(f"[l1] reflection n={len(results['reflection'])} median_spoof={stats_of(results['reflection']).get('median')}", flush=True)
    print(f"[l1] excluded n={len(excl_recs)} -> {[r['ref'] for r in excl_recs]}", flush=True)
    print(f"[l1] AUC(fp vs N) = {auc}", flush=True)
    print(f"[l1] -> {out_path}", flush=True)
    print(f"[l1] cards -> {montage}", flush=True)


if __name__ == "__main__":
    main()
