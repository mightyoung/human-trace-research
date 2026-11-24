"""跨机位出入对 · 第 3 步：时间邻接能给出多少**唯一**候选（可证伪）。

为什么这么做：第 1/2 步证明「贴边 = 离开」在走廊/茶水间失效（贴边率 0%），
画面坐标也推不出机位拓扑（走廊是纵深、电梯间左侧才是通道口）。于是退到
最不依赖先验的问法：

  某个机位的一条轨结束后，在 Δt 窗口内，另一个机位有多少条轨**刚开始**？
  如果大多数时候只有 1 条 → 时间本身就能定位；如果动辄十几条 → 时间没有区分度。

关键是对照：把 B 的时间轴整体随机平移（保持人流密度不变）后重算。
若真实数据的「唯一候选率」并不高于平移对照 → 时间邻接对配人是**无信息**的，
跨机位自动建桥必须靠外观/人脸（而这两条已分别被证不可用于跨机位）。

口径（全部沿用项目既有结论）：
  · 时间用 det 的 `t`（OSD 绝对时钟），不用帧号；跨机位可直接比较
  · 有效轨：时长 ≥ min_dur、框高 ≥ min_h（滤掉瞬时假框与远景噪点）
  · 只在两机位**同时覆盖**的时段内统计（否则 B 根本没开录像）
  · 窗口 W ∈ {30, 60, 120, 300} s
"""
import argparse
import json
from itertools import combinations

import numpy as np

CAM_GROUP = {
    "办公区1": "办公区1", "办公区1b": "办公区1",
    "办公区2": "办公区2", "办公区球机": "球机",
    "茶水间": "茶水间", "电梯间": "电梯间", "行政走廊通道": "走廊",
}


def load(path, min_dur=1.0, min_h=30.0):
    d = json.loads(open(path, encoding="utf-8").read())
    segs = {}
    for name, s in d.items():
        r = np.array(s["rows"], dtype=float)
        t0, t1, n, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        ok = ((t1 - t0) >= min_dur) & (h >= min_h)
        date, cam_raw = name.split("_", 1)
        segs[name] = dict(date=date, cam=CAM_GROUP.get(cam_raw, cam_raw),
                          t0=t0[ok], t1=t1[ok], n=int(ok.sum()))
    return segs


def group_by_day(segs):
    out = {}
    for name, s in segs.items():
        out.setdefault(s["date"], {}).setdefault(s["cam"], []).append((name, s))
    return out


def merge(cams, cam):
    items = cams.get(cam) or []
    t0 = np.concatenate([s["t0"] for _n, s in items]) if items else np.array([])
    t1 = np.concatenate([s["t1"] for _n, s in items]) if items else np.array([])
    return np.sort(t0), np.sort(t1), len(t0)


def counts_in_window(A, B, W, lo, hi):
    """对每个 a，数 B 中落在 (a, a+W] 的个数；只保留 a∈[lo, hi-W]。"""
    if len(A) == 0 or len(B) == 0:
        return np.array([])
    a = A[(A >= lo) & (A <= hi - W)]
    if len(a) == 0:
        return np.array([])
    hi_i = np.searchsorted(B, a + W, side="right")
    lo_i = np.searchsorted(B, a, side="right")
    return hi_i - lo_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="tmp/r5-e1/track_summary.json")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--min-h", type=float, default=30.0)
    ap.add_argument("--const-thr", type=int, default=10,
                    help="B 侧轨数少于此值的机位对跳过（样本不足）")
    ap.add_argument("--n-shift", type=int, default=5)
    ap.add_argument("--out", default="deliverable/e21-allday-2026-11-23/door_pairs_time.csv")
    args = ap.parse_args()

    segs = load(args.summary, args.min_dur, args.min_h)
    by_day = group_by_day(segs)
    rng = np.random.default_rng(0)
    rows = []

    print("=" * 112)
    print(f"{'日期':<12}{'A → B':<22}{'A轨':>6}{'B轨':>6}{'重叠h':>7}"
          f"{'W=30s':>18}{'W=60s':>18}{'W=120s':>18}{'W=300s':>18}")
    print("（每格 = 唯一候选率 / 候选数中位；括号内是时间平移对照的唯一候选率）")
    print("-" * 112)

    for date in sorted(by_day):
        cams = by_day[date]
        # 机位自身时段
        spans = {}
        for cam in cams:
            t0, t1, n = merge(cams, cam)
            if n:
                spans[cam] = (float(t0.min()), float(t1.max()), n)
        for a, b in combinations(sorted(spans), 2):
            (alo, ahi, an) = spans[a]
            (blo, bhi, bn) = spans[b]
            lo, hi = max(alo, blo), min(ahi, bhi)
            if hi - lo < 600:
                continue

            A0, A1, _ = merge(cams, a)   # A 的轨结束 → 离场
            B0, B1, _ = merge(cams, b)   # B 的轨开始 → 入场
            if an < args.const_thr or bn < args.const_thr:
                continue

            cells, cells_b = [], []
            for W in (30, 60, 120, 300):
                c = counts_in_window(A1, B0, W, lo, hi)
                ctrl = []
                for _ in range(args.n_shift):
                    sh = B0 + rng.uniform(0, 3600)
                    sh = sh[(sh >= lo) & (sh <= hi)]
                    cc = counts_in_window(A1, np.sort(sh), W, lo, hi)
                    if len(cc):
                        ctrl.append(float((cc == 1).mean()))
                if len(c) == 0:
                    cells.append("—")
                    cells_b.append("")
                    continue
                u = float((c == 1).mean())
                med = float(np.median(c))
                cu = float(np.mean(ctrl)) if ctrl else float("nan")
                cells.append(f"{u*100:>4.0f}% / {med:>4.1f}")
                cells_b.append(f"{cu*100:>4.0f}%")
                rows.append(dict(date=date, a=a, b=b, W=W, n_a=an, n_b=bn,
                                 overlap_h=round((hi - lo) / 3600, 2),
                                 n_events=len(c), uniq_rate=round(u, 4),
                                 cand_median=med, ctrl_uniq=round(cu, 4)))
            print(f"{date:<12}{a+' → '+b:<22}{an:>6}{bn:>6}{(hi-lo)/3600:>7.2f}"
                  + "".join(f"{x:>18}" for x in cells))
            print(f"{'':<12}{'(对照 ' + b + ' 时间平移)':<22}{'':>6}{'':>6}{'':>7}"
                  + "".join(f"{('(' + x + ')') if x else '':>18}" for x in cells_b))

    # 汇总：按窗口看整体
    print("\n" + "=" * 112)
    print("整体（所有机位对合并）：")
    for W in (30, 60, 120, 300):
        sub = [r for r in rows if r["W"] == W]
        if not sub:
            continue
        w = [r["n_events"] for r in sub]
        u = np.average([r["uniq_rate"] for r in sub], weights=w)
        cu = np.average([r["ctrl_uniq"] for r in sub], weights=w)
        med = np.average([r["cand_median"] for r in sub], weights=w)
        print(f"  W={W:>3}s  事件 {sum(w):>6}  唯一候选率 {u*100:>5.1f}%  "
              f"（平移对照 {cu*100:>5.1f}%，差 {(u-cu)*100:+.1f} pt）  候选中位 {med:.1f}")

    import csv
    from pathlib import Path
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", p, len(rows), "行")


if __name__ == "__main__":
    main()
