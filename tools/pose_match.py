#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""形态条件化的身份模板匹配：运行时选择器的核心算子。

为什么要它
----------
同一人在一天内的框高跨度 **中位 3.76 倍**（75% 分位 8.2×，最大 11.6×），宽高比 3.2× ——
站立整身 / 行走 / 坐姿头肩 的身体部位构成不同，同一件衣服的特征也会漂移。
用「每身份一个均值模板」在这种跨度下必然失配：8 段实测，形态差最极端的一档
（|δh|≥0.8）只靠库均值只有 **7.0%** 能判对。

核心公式
--------
给定查询 (f_q, h_q) 与身份 p 的库存轨 {(f_i, h_i)}：

    w_i  = exp(-|log(h_i / h_q)| / σ)
    T_p  = Σ w_i f_i / ‖Σ w_i f_i‖
    s(p) = f_q · T_p

实测（8 段 208 查询，179 条可评，τ=0.80 的 recall）：
    库均值            56.4%
    形态加权 σ=0.6    68.2%   ← 零训练、零人脸、零新数据
    + 实时池          79.3%

★ 三条硬约束（都来自实测，不是设计偏好）
  1. **不要「只挑最像的 1 条」**：57.5%（+1.1）—— 单条噪声大，必须加权平均。
  2. **不要用绝对形态门**（框高 ≥X px）：73%→59%，挡掉的正是坐姿样本。
  3. **池必须按机位分层**：cam 73%/94.0 > 全局 67%/91.4。

σ 的选择
--------
σ=0.2–0.6 是平台（同分 68.2%），σ=0.15 更高（69.8%）但增益更集中在 female3，
不建议。σ≥0.9 开始退化，σ→∞ 即库均值。默认 0.6。
不同机位的最优 σ 不同（跨度越大越需要小 σ），`--by-cam` 可扫。

用法
----
作为库：
    from pose_match import PoseTemplates, AnchorPool
    tpl = PoseTemplates(emb, sigma=0.6).build([(cam, pid, idx, h), ...])
    scores = tpl.score(qi, hq, cam)          # {pid: cos}

自检（需要远端 8 段数据，复现上面三行数字）：
    python pose_match.py --selfcheck --emb-dir ... --face ... --det ... --gt ... --segs "a b c"
"""
import argparse
import collections
import numpy as np

DEFAULT_SIGMA = 0.6


def l2(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


class PoseTemplates:
    """按 (cam, pid) 组织的形态条件化模板库。

    参数
    ----
    emb : (n, d) float，**已 L2 归一化**
    sigma : 核宽；0 或负数表示退化到「库均值」
    """

    def __init__(self, emb, sigma=DEFAULT_SIGMA):
        self.emb = np.ascontiguousarray(emb, dtype=np.float64)
        if self.emb.ndim != 2:
            raise ValueError("emb 必须是二维 (n, d)")
        self.sigma = float(sigma)
        self.groups = {}       # (cam, pid) -> (idx[np.int64], h[np.float64])
        self.by_cam = {}       # cam -> [pid]

    def build(self, members):
        """members: 可迭代的 (cam, pid, idx, h)。h<=0 表示未知形态（按均值处理）。"""
        buf = collections.defaultdict(list)
        for cam, pid, idx, h in members:
            buf[(str(cam), str(pid))].append((int(idx), float(h)))
        self.groups, self.by_cam = {}, {}
        for (cam, pid), vs in buf.items():
            idx = np.array([x[0] for x in vs], dtype=np.int64)
            hs = np.array([x[1] for x in vs], dtype=np.float64)
            self.groups[(cam, pid)] = (idx, hs)
            self.by_cam.setdefault(cam, []).append(pid)
        for c in self.by_cam:
            self.by_cam[c].sort()
        return self

    def pids(self, cam):
        return self.by_cam.get(str(cam), [])

    def template(self, cam, pid, hq, sigma=None):
        """返回该身份的形态加权模板向量（已归一化）。"""
        g = self.groups.get((str(cam), str(pid)))
        if g is None:
            return None
        idx, hs = g
        s = self.sigma if sigma is None else float(sigma)
        # 单条样本 / 形态未知 / σ 退化 → 均值（加权在数学上等价于不变）
        if idx.size == 1 or hq is None or hq <= 0 or s <= 0:
            v = self.emb[idx].mean(0)
        else:
            d = np.abs(np.log(np.maximum(hs, 1e-6) / float(hq)))
            w = np.exp(-d / s)
            v = (self.emb[idx] * w[:, None]).sum(0)
        return l2(v)

    def score(self, qi, hq, cam, sigma=None, pids=None):
        """对某机位的候选身份打分 → {pid: cos}。缺库的身份给 -1。"""
        out = {}
        for p in (pids if pids is not None else self.pids(cam)):
            v = self.template(cam, p, hq, sigma)
            out[p] = -1.0 if v is None else float(v @ self.emb[int(qi)])
        return out

    def top1(self, qi, hq, cam, sigma=None):
        sc = self.score(qi, hq, cam, sigma)
        if not sc:
            return None, -1.0
        p = max(sc, key=sc.get)
        return p, sc[p]


class AnchorPool:
    """运行时实时池：人脸确认后把该轨的全身特征写进来。

    刻意与聚类解耦：**不进聚类、不改簇结构**，只改选择器能看到什么模板。

    参数
    ----
    cap : 每「机位×身份」保留的样本数（FIFO）
    ttl : 过期秒数（<=0 表示不过期）

    ★ 不加绝对形态门。实测框高 ≥120 px 的门会把覆盖率从 73% 打到 59%，
      因为被筛掉的正是坐姿/头肩样本 —— 而它们恰是唯一能救回的样本。
    """

    def __init__(self, cap=8, ttl=0.0):
        self.cap = int(cap)
        self.ttl = float(ttl)
        self._d = collections.defaultdict(list)   # (cam, pid) -> [(t, idx, h)]

    def add(self, cam, pid, idx, h, t=0.0):
        lst = self._d[(str(cam), str(pid))]
        lst.append((float(t), int(idx), float(h)))
        if self.cap > 0 and len(lst) > self.cap:
            del lst[0]

    def get(self, cam, pid, now=None):
        lst = self._d.get((str(cam), str(pid)), [])
        if self.ttl > 0 and now is not None:
            lst = [x for x in lst if now - x[0] <= self.ttl]
        return lst

    def score(self, emb, qi, hq, cam, pids, k=3, now=None):
        """池内取最像的 k 条均值（★ 不能取池均值：旧场景样本会稀释新样本）。"""
        out = {}
        for p in pids:
            lst = self.get(cam, p, now)
            if not lst:
                out[p] = -1.0
                continue
            v = np.array([emb[i] @ emb[int(qi)] for _, i, _ in lst])
            if v.size > k:
                v = np.sort(v)[-k:]
            out[p] = float(v.mean())
        return out

    def __len__(self):
        return sum(len(v) for v in self._d.values())


# --------------------------------------------------------------------------
# 自检：把主实验的三个数字复现出来
# --------------------------------------------------------------------------
def _selfcheck(args):
    import json
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tmp", "full"))
    from cluster_pos import load_emb                      # noqa: E402
    from exp_pose_pool import load_det_meta, load_face    # noqa: E402

    segs = args.segs.split()
    E64, keys = load_emb(args.emb_dir, segs)
    E = l2(E64)
    eidx = {k: i for i, k in enumerate(keys)}
    gt = json.load(open(args.gt, encoding="utf-8"))
    face = load_face(args.face)
    start, shape = load_det_meta(args.det, segs, set(eidx))

    rows = [dict(k=k, i=eidx[k], pid=gt[k], cam=k.rsplit("#", 1)[0].split("_")[0],
                 tr=("_train_" in k), t=start.get(k, 0), face=(k in face),
                 h=shape.get(k, (-1.0, -1.0))[0]) for k in keys if k in gt]
    tpl = PoseTemplates(E, sigma=args.sigma).build(
        [(r["cam"], r["pid"], r["i"], r["h"]) for r in rows if r["tr"]])
    print("库：%d 个「机位×身份」组" % len(tpl.groups))

    queries = sorted([r for r in rows if not r["tr"]], key=lambda r: (r["cam"], r["t"], r["k"]))
    pool = AnchorPool(cap=args.pool_max)
    rec = []
    for q in queries:
        cands = tpl.pids(q["cam"])
        if not cands:
            continue
        s0 = tpl.score(q["i"], q["h"], q["cam"], sigma=0)      # σ=0 → 库均值
        s5 = tpl.score(q["i"], q["h"], q["cam"], sigma=args.sigma)
        sp = pool.score(E, q["i"], q["h"], q["cam"], cands, k=args.pool_k)
        s1 = {p: max(s5[p], sp[p]) for p in cands}
        rec.append(dict(q=q, s0=s0, s5=s5, s1=s1))
        if q["face"] and q["pid"] in cands:                    # ★ 因果：打分后才写
            pool.add(q["cam"], q["pid"], q["i"], q["h"], q["t"])

    ev = [r for r in rec if r["s0"].get(r["q"]["pid"], -1.0) >= 0]
    print("查询 %d ｜ 可评 %d（本人上午在该机位有库）" % (len(rec), len(ev)))

    def rec_at(rs, get, tau):
        if not rs:
            return 0.0
        a = 0
        for r in rs:
            sc = get(r)
            if sc and max(sc.values()) >= tau and max(sc, key=sc.get) == r["q"]["pid"]:
                a += 1
        return 100.0 * a / len(rs)

    print("\n  %-6s %-12s %-14s %-12s" % ("τ", "库均值", "形态加权 σ=%.2f" % args.sigma, "+实时池"))
    for tau in (0.70, 0.80, 0.85):
        a = rec_at(ev, lambda r: r["s0"], tau)
        b = rec_at(ev, lambda r: r["s5"], tau)
        c = rec_at(ev, lambda r: r["s1"], tau)
        print("  %-6.2f %6.1f%%      %6.1f%% (%+5.1f)  %6.1f%% (%+5.1f)"
              % (tau, a, b, b - a, c, c - a))

    if args.by_cam:
        print("\n== 按机位扫 σ（recall@0.80，分母=该机位可评查询）==")
        cams = sorted({r["q"]["cam"] for r in ev})
        grid = [0.15, 0.3, 0.45, 0.6, 0.9, 1.5]
        print("  %-5s %4s %-8s %s" % ("机位", "n", "库均值", "  ".join("σ%.2f" % s for s in grid)))
        for c in cams:
            rs = [r for r in ev if r["q"]["cam"] == c]
            line = []
            for s in grid:
                line.append("%5.1f" % rec_at(
                    rs, lambda r, s=s: tpl.score(r["q"]["i"], r["q"]["h"],
                                                 r["q"]["cam"], sigma=s), 0.80))
            print("  %-5s %4d %6.1f%%  %s   | σ.6+池 %5.1f%%"
                  % (c, len(rs), rec_at(rs, lambda r: r["s0"], 0.80), "  ".join(line),
                     rec_at(rs, lambda r: r["s1"], 0.80)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--emb-dir")
    ap.add_argument("--face")
    ap.add_argument("--det")
    ap.add_argument("--gt")
    ap.add_argument("--segs")
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--pool-max", type=int, default=8)
    ap.add_argument("--by-cam", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
