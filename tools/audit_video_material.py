#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""录像素材盘点：实测每卷时长、与 README 标称对比、按点位拼接时间轴、统计已用/剩余区间。

只读视频 header（CAP_PROP_FRAME_COUNT / FPS），不逐帧解码，秒级完成。

用法：
  python tools/audit_video_material.py [--json out.json]
"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_1012 = os.path.join(ROOT, "media-resource", "1012")
SRC_MULTI = os.path.join(ROOT, "media-resource", "multi-cemra")
SEG_META = os.path.join(SRC_1012, "segments", "segments_meta.json")

# 卷 -> (点位, README 标称时长 min, README 起录 OSD)
VOLS_1012 = {
    "0100000251400000000.mp4": ("320", 64.9, "08:03"),
    "0100000252500000000.mp4": ("318", 299.9, "08:22:50"),
    "0100000251000000000.mp4": ("323", 65.5, "08:35"),
    "0100000248000000000.mp4": ("320", 73.1, "09:08:53"),
    "0100000252900000000.mp4": ("305", 116.2, "09:28"),
    "0100000253600000000.mp4": ("323", 64.4, "09:40"),
    "0100000254000000000.mp4": ("320", 64.3, "10:21"),
    "0100000252100000000.mp4": ("305?", 121.7, "10:26?"),
    "0100000251300000000.mp4": ("323", 65.7, "10:46"),
    "0100000251700000000.mp4": ("305", 123.8, "11:24"),
    "0100000254300000000.mp4": ("320", 73.4, "11:26"),
    "0100000253800000000.mp4": ("323", 65.7, "11:51"),
    "0100000251900000000.mp4": ("320", 39.9, "12:39"),
    "0100000252700000000.mp4": ("323", 65.7, "12:57"),
    "0100000254400000000.mp4": ("305", 117.2, "13:28"),
    "0100000254200000000.mp4": ("320", 11.2, "14:01"),
    "0100000253400000000.mp4": ("318", 233.7, "14:22:57"),
    "0100000251800000000.mp4": ("320", 64.6, "15:03"),
    "0100000255900000000.mp4": ("305", 95.5, "15:25"),
    "0100000254900000000.mp4": ("320", 12.0, "16:07"),
    "0100000256000000000.mp4": ("320", 13.9, "17:00"),
}

VOLS_MULTI = {
    "0100000230500000000.mp4": "305",
    "0100000231600000000.mp4": "316",
    "0100000231800000000.mp4": "318",
    "0100000232000000000.mp4": "320",
    "0100000232300000000.mp4": "323",
    "0100000234300000000.mp4": "343",
}

# 当前已切片段（来自 segments_meta.json，用作已用区间）
# 其余非 1012 素材：media-resource/1012 下 2 个散卷 + multi-cemra


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0 or fps <= 0:
        return None
    return {"frames": int(n), "fps": round(fps, 3), "w": w, "h": h,
            "dur_s": round(n / fps, 1), "dur_min": round(n / fps / 60, 2)}


def fmt_hms(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "docs", "material-audit-2026-10-20.json"))
    args = ap.parse_args()

    # ---- 已切片段 ----
    used = {}
    segs = []
    if os.path.exists(SEG_META):
        meta = json.load(open(SEG_META, encoding="utf-8"))
        segs = meta["segments"]
        for s in segs:
            used.setdefault(s["vol"], []).append({
                "seg": s["seg"], "split": s["split"], "ss": s["ss"], "dur": s["dur"]})

    out = {"probed_at": "2026-10-20", "volumes_1012": [], "volumes_multi": [],
           "segments_used": segs}

    print("=" * 108)
    print("1012 卷清单（实测 vs README 标称）")
    print("=" * 108)
    print(f"{'卷':<24}{'点位':<7}{'分辨率':<12}{'实测min':>9}{'标称min':>9}{'差':>7}{'起录':>10}  已用")
    print("-" * 108)
    tot = 0.0
    by_cam = {}
    for vol, (cam, claim, osd) in VOLS_1012.items():
        p = os.path.join(SRC_1012, vol)
        info = probe(p) if os.path.exists(p) else None
        if not info:
            print(f"{vol:<24}{cam:<7}  ** 读取失败/缺失 **")
            continue
        d = info["dur_min"]
        tot += d
        by_cam.setdefault(cam, []).append((vol, info, osd))
        u = used.get(vol, [])
        utxt = " ".join(f"{x['split']}[{x['ss']}-{x['ss']+x['dur']}]" for x in u) or "-"
        diff = d - claim
        print(f"{vol:<24}{cam:<7}{str(info['w'])+'x'+str(info['h']):<12}{d:>9.1f}{claim:>9.1f}{diff:>+7.1f}{osd:>10}  {utxt}")
        out["volumes_1012"].append({"vol": vol, "cam": cam, "osd_readme": osd,
                                    "claim_min": claim, **info, "used": u})
    print("-" * 108)
    print(f"实测总时长 {tot/60:.2f} 小时（README 标称合计 {sum(v[1] for v in VOLS_1012.values())/60:.2f} 小时）")

    # ---- 已用 vs 剩余 ----
    used_by_vol = {}
    for s in segs:
        used_by_vol[s["vol"]] = used_by_vol.get(s["vol"], 0) + s["dur"]

    print()
    print("=" * 108)
    print("按点位：已切用 / 该点位总可用")
    print("=" * 108)
    print(f"{'点位':<8}{'卷数':>5}{'总时长(h)':>11}{'已切(h)':>10}{'剩余(h)':>10}")
    print("-" * 108)
    stat = {}
    for cam, items in by_cam.items():
        tv = sum(i["dur_s"] for _, i, _ in items) / 3600
        uv = sum(used_by_vol.get(v, 0) for v, _, _ in items) / 3600
        stat[cam] = {"vols": len(items), "total_h": round(tv, 2), "used_h": round(uv, 2),
                     "left_h": round(tv - uv, 2)}
        print(f"{cam:<8}{len(items):>5}{tv:>11.2f}{uv:>10.2f}{tv-uv:>10.2f}")
    g_t = sum(s["total_h"] for s in stat.values())
    g_u = sum(s["used_h"] for s in stat.values())
    print("-" * 108)
    print(f"{'合计':<8}{sum(s['vols'] for s in stat.values()):>5}{g_t:>11.2f}{g_u:>10.2f}{g_t-g_u:>10.2f}")
    out["by_cam"] = stat
    out["total_h"] = round(g_t, 2)
    out["used_h"] = round(g_u, 2)
    out["left_h"] = round(g_t - g_u, 2)

    # ---- multi-cemra ----
    print()
    print("=" * 108)
    print("multi-cemra（10-10 周六 6 路同步，公共窗 08:44:55–09:32:43 ≈ 47.8min）")
    print("=" * 108)
    mt = 0.0
    for vol, cam in VOLS_MULTI.items():
        p = os.path.join(SRC_MULTI, vol)
        info = probe(p) if os.path.exists(p) else None
        if not info:
            print(f"{cam:<6}{vol:<26}  ** 读取失败 **")
            continue
        mt += info["dur_min"]
        print(f"{cam:<6}{vol:<26}{info['w']}x{info['h']}  {info['dur_min']:>6.1f} min  {info['fps']}fps")
        out["volumes_multi"].append({"vol": vol, "cam": cam, **info})
    print(f"6 路实测合计 {mt/60:.2f} 小时；当前仅用 synced-10min(6×10min=1h) + long-35min(6×35min=3.5h)")

    # ---- 散卷 ----
    print()
    print("其它散卷：")
    for f in sorted(os.listdir(os.path.join(ROOT, "media-resource"))):
        if f.endswith(".mp4"):
            p = os.path.join(ROOT, "media-resource", f)
            i = probe(p)
            if i:
                print(f"  {f}  {i['w']}x{i['h']}  {i['dur_min']:.1f} min")
                out.setdefault("misc", []).append({"file": f, **i})

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump(out, open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[DONE] -> {args.json}")


if __name__ == "__main__":
    sys.exit(main())
