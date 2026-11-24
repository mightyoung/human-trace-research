# -*- coding: utf-8 -*-
"""倒影标记二次判据扫描：亮度比 + 运动速度。

背景：运动学规则在长窗/无镜区域出现两类假阳性：
  A. 静止对（两人都几乎不动，恒等式平凡成立）——运动门可杀；
  B. 运动巧合（两个真人运动学巧合）——需外观判据：地板倒影是暗色影像。
本脚本对既有标记（10min + long35）逐对计算：
  - 运动速度 vR / vS（中位，脚点，px/frame）
  - 亮度比 lumR / lumS（裁剪中位置中位数灰度，跨样帧取中位）
输出 analysis/reflection_gate/flags_scan.json + 控制台表。
"""
import json
import os
import sys

import cv2
import numpy as np


def prep_crop(frame, box, out_h=224, pad=0.08):
    """与 cradio_scale_remote.prep_crop 完全一致（BGR in → RGB out）"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad))
    Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad))
    Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
        return None
    s = out_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, out_h))
    if nw >= out_h:
        off = (nw - out_h) // 2
        c = c[:, off:off + out_h]
    else:
        padw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2,
                               cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (tag, 检测 jsonl, 视频, 标记来源 json)
JOBS = [
    ("316-10m", "output_crosscam/316_detections.jsonl",
     "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
     "output_crosscam/analysis/track_cleaning/track_cleaning.json", "316"),
    ("318-10m", "output_crosscam/318_detections.jsonl",
     "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4",
     "output_crosscam/analysis/track_cleaning/track_cleaning.json", "318"),
    ("305-10m", "output_crosscam/305_detections.jsonl",
     "media-resource/multi-cemra/synced-10min/02_办公区1_305/305_085500-090500.mp4",
     "output_crosscam/analysis/track_cleaning/track_cleaning.json", "305"),
    ("343-10m", "output_crosscam/343_detections.jsonl",
     "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
     "output_crosscam/analysis/track_cleaning/track_cleaning.json", "343"),
    ("323-10m", "output_crosscam/323_detections.jsonl",
     "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4",
     "output_crosscam/analysis/track_cleaning/track_cleaning.json", "323"),
]
LONG = [
    ("316-L", "output_crosscam/long35/316_long35_detections.jsonl",
     "media-resource/multi-cemra/long-35min/316_long35.mp4",
     "output_crosscam/long35/cleaning/track_cleaning.json", "316"),
    ("318-L", "output_crosscam/long35/318_long35_detections.jsonl",
     "media-resource/multi-cemra/long-35min/318_long35.mp4",
     "output_crosscam/long35/cleaning/track_cleaning.json", "318"),
]


def load(path):
    acc = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        for t, b in zip(r["track_ids"], r["boxes"]):
            if t is not None and t >= 0:
                acc.setdefault(int(t), []).append((r["frame"], b))
    for k in acc:
        acc[k].sort()
    return acc


def speed(tr):
    sp = []
    for (f0, b0), (f1, b1) in zip(tr[:-1], tr[1:]):
        if f1 == f0:
            continue
        u0, v0 = (b0[0] + b0[2]) / 2, b0[3]
        u1, v1 = (b1[0] + b1[2]) / 2, b1[3]
        sp.append(np.hypot(u1 - u0, v1 - v0) / (f1 - f0))
    return float(np.median(sp)) if sp else 0.0


def lum_pair(cap, frames, boxR, boxS, n=4):
    idxs = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    lr, ls = [], []
    for k in idxs:
        f = frames[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        cr = prep_crop(fr, boxR[f])
        cs = prep_crop(fr, boxS[f])
        if cr is None or cs is None:
            continue
        lr.append(float(np.median(cv2.cvtColor(cr, cv2.COLOR_RGB2GRAY))))
        ls.append(float(np.median(cv2.cvtColor(cs, cv2.COLOR_RGB2GRAY))))
    if not lr or not ls:
        return None
    return float(np.median(lr)), float(np.median(ls))


def scan(tag, jpath, vpath, flagpath, key):
    tracks = load(jpath)
    flags = json.load(open(flagpath, encoding="utf-8"))["summary"][key]["flags"]
    cap = cv2.VideoCapture(vpath)
    assert cap.isOpened(), vpath
    out = []
    for t, fl in sorted(flags.items(), key=lambda kv: -kv[1]["n"]):
        t, s = int(t), int(fl["source"])
        if t not in tracks or s not in tracks:
            continue
        vR, vS = speed(tracks[t]), speed(tracks[s])
        common = sorted(set(f for f, _ in tracks[t]) & set(f for f, _ in tracks[s]))
        bR = dict(tracks[t]); bS = dict(tracks[s])
        lums = lum_pair(cap, common, bR, bS)
        rec = {"tag": tag, "t": t, "s": s, "n": fl["n"], "perr": fl["perr"],
               "off": fl["off"], "vR": round(vR, 2), "vS": round(vS, 2),
               "lumR": round(lums[0]) if lums else None,
               "lumS": round(lums[1]) if lums else None,
               "lum_ratio": round(lums[0] / (lums[1] + 1e-9), 3) if lums else None}
        out.append(rec)
        print(f"  {tag} #{t:>4}->#{s:<4} n={fl['n']:>3} vR={vR:5.2f} vS={vS:5.2f} "
              f"lum {rec['lumR']} / {rec['lumS']}  ratio={rec['lum_ratio']}", flush=True)
    cap.release()
    return out


def main():
    all_recs = []
    for tag, j, v, fp, key in JOBS + LONG:
        print(f"== {tag} ==")
        all_recs += scan(tag, j, v, fp, key)
    outdir = os.path.join(ROOT, "output_crosscam", "analysis", "reflection_gate")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "flags_scan.json")
    json.dump(all_recs, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {out}  ({len(all_recs)} records)")

    # 汇总统计
    import collections
    print("\n---- 分布 ----")
    for tag in [j[0] for j in JOBS] + [l[0] for l in LONG]:
        rs = [r for r in all_recs if r["tag"] == tag and r["lum_ratio"] is not None]
        if not rs:
            continue
        static = [r for r in rs if max(r["vR"], r["vS"]) < 0.4]
        moving = [r for r in rs if max(r["vR"], r["vS"]) >= 0.4]
        def stat(rs2):
            if not rs2: return "—"
            rr = [r["lum_ratio"] for r in rs2]
            return (f"n={len(rs2)} ratio p50={np.median(rr):.2f} "
                    f"p90={np.percentile(rr,90):.2f} max={max(rr):.2f}")
        print(f"{tag:>8}: 静止对[{stat(static)}]  运动对[{stat(moving)}]")


if __name__ == "__main__":
    main()
