# -*- coding: utf-8 -*-
"""TOP-4 Step-1：相对尺度自标定（人体身高路线，文献 arXiv 1712.00158 简化版）。

目标（低成本、仅用既有数据）：
  1) 每机位"人体框高"分布（GIL obs.h_med，546 obs）→ 相对尺度表（以 318 为参考）；
  2) 重叠对 318×320 的 8 个匹配对（match30 正对）→ **同人同期高度比**分布（直接相对尺度测量）；
  3) 两种口径交叉对照 + 一致性指标；btw 给出"距离先验"的可行边界（诚实结论）。

输出：analysis/top4/rel_scale.json
"""
import json
import os
import sqlite3
import sys

import numpy as np

os.environ.setdefault("DET_VIEW", "clean2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from crosscam_relay import load_foot_trajs  # noqa: E402

ANA = os.path.join(ROOT, "output_crosscam", "analysis")
GIL = os.path.join(ANA, "gil", "store_v1", "gil.sqlite")
OUT = os.path.join(ANA, "top4")
CAMS = ["305", "316", "318", "320", "323", "343"]
# match30 正对（318-tid, 320-tid），来自 replay_eval.POS30
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(GIL)
    cur = con.cursor()
    h = {c: [] for c in CAMS}
    for cam, hm in cur.execute("SELECT cam, h_med FROM obs WHERE h_med IS NOT NULL"):
        if cam in h and hm is not None:
            h[cam].append(float(hm))
    con.close()

    stats = {}
    ref = "318"
    ref_med = float(np.median(h[ref])) if h[ref] else None
    for c in CAMS:
        a = np.array(h[c])
        if len(a) < 5:
            stats[c] = dict(n=len(a), note="样本不足")
            continue
        stats[c] = dict(n=len(a), med=round(float(np.median(a)), 1),
                        p25=round(float(np.quantile(a, 0.25)), 1),
                        p75=round(float(np.quantile(a, 0.75)), 1),
                        ratio_vs_318=round(float(np.median(a)) / ref_med, 3))

    # ---- 318×320 匹配对直接测量（同时刻逐帧比，T0 对齐）----
    T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
    FPS = 25.0
    foot318 = load_foot_trajs("318")
    foot320 = load_foot_trajs("320")

    def series(foot, tid, cam):
        fr, u, v, h = foot[tid]
        t = T0[cam] + fr / FPS
        return t, h

    pairs = []
    for t18, t20 in POS30:
        if t18 not in foot318 or t20 not in foot320:
            pairs.append(dict(t18=t18, t20=t20, missing=True))
            continue
        t1, h1 = series(foot318, t18, "318")
        t2, h2 = series(foot320, t20, "320")
        # 同时刻匹配（±0.5s 内最近）
        ratios = []
        for a, hh in zip(t1, h1):
            j = np.searchsorted(t2, a)
            for jj in (j - 1, j, j + 1):
                if 0 <= jj < len(t2) and abs(t2[jj] - a) <= 0.5:
                    ratios.append(h2[jj] / hh)
                    break
        ratios = np.array(ratios)
        pairs.append(dict(t18=t18, t20=t20, n_sync=len(ratios),
                          ratio_med=round(float(np.median(ratios)), 3) if len(ratios) else None,
                          ratio_iqr=round(float(np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)), 3)
                          if len(ratios) else None,
                          missing=False))
    rr = np.array([p["ratio_med"] for p in pairs if not p["missing"] and p.get("ratio_med")])
    within = np.array([p["ratio_iqr"] for p in pairs if not p["missing"] and p.get("ratio_iqr") is not None])
    direct = dict(n=len(rr), med=round(float(np.median(rr)), 3) if len(rr) else None,
                  iqr=round(float(np.quantile(rr, 0.75) - np.quantile(rr, 0.25)), 3) if len(rr) else None,
                  min=round(float(rr.min()), 3) if len(rr) else None,
                  max=round(float(rr.max()), 3) if len(rr) else None,
                  within_pair_iqr_med=round(float(np.median(within)), 3) if len(within) else None,
                  note="同时刻逐帧比；within_pair_iqr_med = 同一人跨位置比值波动（位置依赖度）")

    out = dict(version="top4/step1",
               method="人体框高（h_med）统计相对尺度；重叠对同人直接测量为交叉验证",
               per_cam=stats, match30_direct=dict(pairs=pairs, **direct),
               note="Step-1 只给相对尺度与可行性边界；未做地面平面/BEV（Step-3）")
    with open(os.path.join(OUT, "rel_scale.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"{'cam':>5} {'n':>5} {'med_h':>7} {'IQR':>12} {'ratio/318':>9}")
    for c in CAMS:
        s = stats[c]
        if "med" in s:
            print(f"{c:>5} {s['n']:>5} {s['med']:>7} [{s['p25']},{s['p75']}] {s['ratio_vs_318']:>9}")
        else:
            print(f"{c:>5} {s.get('n', 0):>5}  {s.get('note')}")
    print(f"[match30 direct] n={direct['n']} ratio med={direct['med']} IQR={direct['iqr']} "
          f"range=[{direct['min']},{direct['max']}]")
    print("saved", os.path.join(OUT, "rel_scale.json"))


if __name__ == "__main__":
    main()
