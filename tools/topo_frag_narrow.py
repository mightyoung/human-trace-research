# -*- coding: utf-8 -*-
"""六机位拓扑"窄窗"显著性检验（碎片级，修正口径 v2）。

背景（外部评审 2026-10-13 结论 + 本项目复核）：
  - 旧工具 `topology_signature.py` 默认 dt∈[0.5,60]s → 信号被淹没（最大 z≈1.8）；
  - 修正口径：**窄窗（≤5/≤10s）+ 30 对族校正** → 10min 数据上复现 3 条有序边：
      316→318、343→323、343→316（族校正 p≈0.001；碎片级同样成立：z≈5.6/4.8/4.5）。
  - 本工具 = 修正口径的**项目内自包含实现**（碎片级），供 long35 确认性检验与后续复用。

输入 data_dir 需含：
  - `splits_v4.json`（或任意含 {"train|val|test": {"tracks": {cam: [tid,...]}}} 结构的 json）
  - `{cam}_detections.jsonl`（六路，raw 或清洗版均可；靠 splits 的 keep 集过滤）

输出：JSON —— 每窗口 zmax / 族校正 p / z>=3 计数 / **全 30 对 z 表**。

用法：
  python tools/topo_frag_narrow.py <data_dir> <out_json> [--shifts 2000] [--tag v10]
                                    [--cam-shift 305=-2.5] [--wins -1:3,-1:5,-1:10,-1:15,0.5:30,0.5:60]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

CAMS = ["305", "316", "318", "320", "323", "343"]
RES = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0


def load_keep(data_dir):
    """从 splits 风格 json 读每路 keep 轨迹集合（合并 train/val/test）。"""
    p = os.path.join(data_dir, "splits_v4.json")
    s = json.load(open(p, encoding="utf-8"))
    keep = defaultdict(set)
    for part in ("test", "val", "train"):
        if part not in s:
            continue
        for cam, lst in s[part]["tracks"].items():
            keep[cam].update(int(t) for t in lst)
    return keep


def load_foot(data_dir, cam, keep=None):
    W, H = RES[cam]
    acc = defaultdict(list)
    with open(os.path.join(data_dir, f"{cam}_detections.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for tid, b in zip(d.get("track_ids", []), d.get("boxes", [])):
                if tid is None or tid < 0:
                    continue
                if keep is not None and tid not in keep:
                    continue
                x1, y1, x2, y2 = b
                acc[tid].append((d["frame"], (x1 + x2) / 2 / W, y2 / H, y2 - y1))
    return {t: np.array(sorted(r)) for t, r in acc.items()}


def frag_events(data_dir, cam_shift=None):
    keep = load_keep(data_dir)
    ex, en = {}, {}
    for c in CAMS:
        ex[c], en[c] = [], []
        off = (cam_shift or {}).get(c, 0.0)
        for tid, a in load_foot(data_dir, c, keep.get(c)).items():
            fr, u, v = a[:, 0], a[:, 1], a[:, 2]
            if (fr[-1] - fr[0]) / FPS >= 1 and np.hypot(np.ptp(u), np.ptp(v)) >= 0.05:
                en[c].append(T0[c] + fr[0] / FPS + off)
                ex[c].append(T0[c] + fr[-1] / FPS + off)
    return ex, en


def count(ta, tb, lo, hi):
    tb = np.sort(tb)
    return int(sum(np.searchsorted(tb, a + hi, "right")
                   - np.searchsorted(tb, a + lo, "left") for a in ta))


def run(ex, en, lo, hi, shifts_n, rng):
    allt = [t for c in CAMS for t in ex[c] + en[c]]
    t_min, W = min(allt), max(allt) - min(allt)
    rows, null_all = [], []
    shifts = rng.uniform(0, W, size=shifts_n)
    for A in CAMS:
        for B in CAMS:
            if A == B or len(ex[A]) < 2 or len(en[B]) < 2:
                continue
            ta, tb = np.array(ex[A]), np.array(en[B])
            n = count(ta, tb, lo, hi)
            nl = np.array([count(ta, ((tb - t_min + s) % W) + t_min, lo, hi) for s in shifts])
            sd = nl.std() or 1e-9
            rows.append(dict(A=A, B=B, n_obs=n, null_mean=round(float(nl.mean()), 2),
                             z=round(float((n - nl.mean()) / sd), 2),
                             p=round(float((1 + (nl >= n).sum()) / (shifts_n + 1)), 4)))
            null_all.append((nl - nl.mean()) / sd)
    zmax_null = np.max(np.array(null_all), axis=0)
    zmax_obs = max(r["z"] for r in rows)
    rows.sort(key=lambda r: -r["z"])
    return dict(window=[lo, hi], n_exit=sum(len(v) for v in ex.values()),
                n_entry=sum(len(v) for v in en.values()),
                zmax_obs=zmax_obs,
                p_family=round(float((1 + (zmax_null >= zmax_obs).sum()) / (shifts_n + 1)), 4),
                n_z_ge3=sum(r["z"] >= 3 for r in rows),
                rows=rows)


def parse_wins(s):
    out = []
    for part in s.split(","):
        lo, hi = part.split(":")
        out.append((float(lo), float(hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--shifts", type=int, default=2000)
    ap.add_argument("--tag", default="v10")
    ap.add_argument("--cam-shift", default="", help='如 "305=-2.5"（对该机位全部事件加偏移秒）')
    ap.add_argument("--wins", default="-1:3,-1:5,-1:10,-1:15,0.5:30,0.5:60")
    args = ap.parse_args()
    cam_shift = {}
    if args.cam_shift:
        for part in args.cam_shift.split(","):
            k, v = part.split("=")
            cam_shift[k.strip()] = float(v)
    rng = np.random.default_rng(7)
    ex, en = frag_events(args.data, cam_shift)
    res = {"tag": args.tag, "data": args.data, "level": "fragment",
           "cam_shift": cam_shift, "shifts": args.shifts,
           "note": "窄窗环形移位 + 30 对族校正（修正口径 v2，外部评审 2026-10-13 建议）"}
    for lo, hi in parse_wins(args.wins):
        r = run(ex, en, lo, hi, args.shifts, rng)
        res[f"frag[{lo},{hi}]"] = r
        print(f"frag[{lo},{hi}] zmax={r['zmax_obs']} p_family={r['p_family']} "
              f"n_z>=3={r['n_z_ge3']} top={[(x['A'] + '->' + x['B'], x['z']) for x in r['rows'][:4]]}",
              flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
