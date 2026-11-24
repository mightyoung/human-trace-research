"""跨机位出入对 · 第 4 步：Δt 直方图 vs 时间平移对照。

第 3 步给出「唯一候选率只比随机平移高 1-2 pt」。但那只是单一汇总量，
可能存在「峰很窄、被平均掉」的情况。这里直接看形状：

  对每个 A 轨结束时刻 t1，取 B 所有 t0 ∈ (t1, t1+W]，记录 Δt = t0 - t1。
  若真有「A 走 → 几十秒后 B 出现」的通道，Δt 直方图应在几十秒处有尖峰，
  且明显高于「B 时间轴随机平移」的对照。

同时也报告只用**贴边轨**（末帧贴边）做 A 事件的版本 —— 验证贴边能否救回信号。

用法：
  python e21_door_dt.py --pairs "办公区1:走廊,办公区1:电梯间,走廊:电梯间" --date 2026-10-22
"""
import argparse
import json
from collections import defaultdict

import numpy as np

CAM_GROUP = {
    "办公区1": "办公区1", "办公区1b": "办公区1",
    "办公区2": "办公区2", "办公区球机": "球机",
    "茶水间": "茶水间", "电梯间": "电梯间", "行政走廊通道": "走廊",
}


def load(path, min_dur=1.0, min_h=30.0):
    d = json.loads(open(path, encoding="utf-8").read())
    out = defaultdict(list)
    for name, s in d.items():
        r = np.array(s["rows"], dtype=float)
        t0, t1, n, h, u0, v0, u1, v1, e0, e1 = [r[:, i] for i in range(10)]
        ok = ((t1 - t0) >= min_dur) & (h >= min_h)
        date, cam_raw = name.split("_", 1)
        cam = CAM_GROUP.get(cam_raw, cam_raw)
        out[(date, cam)].append(dict(t0=t0[ok], t1=t1[ok], e0=e0[ok], e1=e1[ok]))
    merged = {}
    for k, items in out.items():
        merged[k] = {f: np.concatenate([it[f] for it in items]) for f in ("t0", "t1", "e0", "e1")}
    return merged


def dt_hist(A_end, B_start, lo, hi, W, bins, shift=0.0):
    """返回 (每箱计数/事件数, 事件数)。B 的 t0 平移 shift 秒。"""
    B = np.sort(B_start + shift)
    a = A_end[(A_end >= lo) & (A_end <= hi - W)]
    if len(a) == 0:
        return np.zeros(len(bins) - 1), 0
    hist = np.zeros(len(bins) - 1)
    # 一次算完：对每个 a，取 (a, a+W] 的 B
    for x in a:
        s, e = np.searchsorted(B, x), np.searchsorted(B, x + W, side="right")
        if e > s:
            h, _ = np.histogram(B[s:e] - x, bins=bins)
            hist += h
    return hist / max(len(a), 1), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="tmp/r5-e1/track_summary.json")
    ap.add_argument("--date", default="2026-10-22")
    ap.add_argument("--pairs", default="办公区1:走廊,办公区1:电梯间,走廊:电梯间")
    ap.add_argument("--W", type=float, default=180.0)
    ap.add_argument("--edge-only", action="store_true", help="A 事件只用末帧贴边的轨")
    ap.add_argument("--shift-a", type=float, default=0.0,
                    help="把 A 的事件时刻整体平移（秒）。★ 保留日变化、破坏个体转移 —— "
                         "若「实测 ≈ A 平移」而「实测 > B 平移」，说明超额来自人流共变而非同一个人")
    args = ap.parse_args()

    data = load(args.summary)
    bins = np.arange(0, args.W + 1e-6, 10.0)
    rng = np.random.default_rng(0)

    for pair in args.pairs.split(","):
        a, b = [x.strip() for x in pair.split(":")]
        A = data.get((args.date, a))
        B = data.get((args.date, b))
        if A is None or B is None:
            print(f"[{args.date}] {a}→{b}: 缺数据")
            continue
        lo = max(float(A["t0"].min()), float(B["t0"].min()))
        hi = min(float(A["t1"].max()), float(B["t1"].max()))
        A_end = A["t1"][A["e1"] == 1] if args.edge_only else A["t1"]
        if args.shift_a:
            A_end = A_end + args.shift_a
            lo, hi = lo + args.shift_a, hi + args.shift_a
            lo = max(lo, float(B["t0"].min()))
            hi = min(hi, float(B["t1"].max()))
        tag = ("（仅贴边轨）" if args.edge_only else "") + (f"（A 平移 {args.shift_a:.0f}s）" if args.shift_a else "")

        real, n_ev = dt_hist(A_end, B["t0"], lo, hi, args.W, bins)
        ctrls = []
        for _ in range(5):
            c, _n = dt_hist(A_end, B["t0"], lo, hi, args.W, bins, shift=rng.uniform(600, 3600))
            ctrls.append(c)
        ctrl = np.mean(ctrls, axis=0)

        print(f"\n=== {args.date}  {a} → {b}{tag}   事件 {n_ev}  A轨 {len(A_end)}  B轨 {len(B['t0'])}"
              f"  重叠 {(hi-lo)/3600:.1f}h  W={args.W:.0f}s ===")
        print(f"{'Δt(s)':>8}{'实测/事件':>12}{'平移对照':>10}{'比值':>8}")
        for i in range(len(real)):
            r = real[i] / ctrl[i] if ctrl[i] > 1e-9 else float("nan")
            bar = "█" * int(round(real[i] * 400))
            print(f"{bins[i]:>4.0f}-{bins[i+1]:<4.0f}{real[i]:>12.4f}{ctrl[i]:>10.4f}{r:>8.2f}  {bar}")
        tot_r, tot_c = real.sum(), ctrl.sum()
        print(f"{'合计':>8}{tot_r:>12.4f}{tot_c:>10.4f}{tot_r/max(tot_c,1e-9):>8.2f}")


if __name__ == "__main__":
    main()
