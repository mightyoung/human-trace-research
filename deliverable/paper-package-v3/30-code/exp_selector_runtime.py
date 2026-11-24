#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行时选择器：人脸作为可信触发器，把「最新全身特征」写进该身份的实时池。

要回答的问题（用户提出的机制）
------------------------
人可能换衣服，但脸变化不大。所以：
  1. 训练素材 → 每个身份的**历史全身特征库**（这里的"训练素材"= 上午段）
  2. 运行时，人脸识别到某人 → 立刻把这条轨的**最新全身特征**并入该身份的实时池
  3. 选择器查询时，历史库 + 实时池一起匹配 → 换衣后仍能认出
关键差别：**实时池不进聚类、不改簇结构**，只改选择器能看到什么模板。

与之前 exp_anchor_pool.py 的区别（那个有三个缺陷）
------------------------------------------------
  × 用了 ±W 对称窗 → 用到了未来信息（非因果）
  × 只报聚合指标 → 看不出增益落在哪一档
  × 没有"换衣"分层 → 无法回答核心问题
本脚本：
  ✓ **严格因果**：按时间顺序逐条处理，池里只有该查询**之前**写入的样本
  ✓ **三策略对照**：S0 仅历史库 / S1 库+池（多模板 top-k）/ S2 仅池
  ✓ **按历史库相似度分档**，单列"换衣嫌疑"档（cos < 0.60）
  ✓ 同时报 coverage / same（与 d7_sweep 口径一致）

★ 口径声明（必须随结果一起引用）
  - "人脸识别到谁" 用 **GT 身份**代替 → 这是**上限**，不是现实精度
  - "这条轨能不能被写入池" 用 **真实可用脸**（face_vecs 有 |full）→ 这部分是现实的
  - 所以结果是「人脸识别全对时」的上限；现实折损只来自覆盖，不来自误识别
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_pos import load_emb            # noqa: E402

FPS = 25.0


def cam_of(k):
    return k.rsplit("#", 1)[0].split("_")[0]


def load_face(face_dir, proto="full"):
    out = {}
    for p in sorted(glob.glob(os.path.join(face_dir, "face_vecs_*.npz"))):
        z = np.load(p, allow_pickle=True)
        for k, v in zip(z["keys"], z["vecs"]):
            k = str(k)
            if "|" not in k:
                continue
            key, tag = k.rsplit("|", 1)
            if tag != proto:
                continue
            v = np.asarray(v, dtype=np.float64)
            n = float(np.linalg.norm(v))
            if n > 0:
                out[key] = v / n
    return out


def load_start(det_dir, segs, keyset):
    """每轨的起始帧（因果排序用）。缺检测文件时退化为按 track id 排。"""
    out = {}
    for v in segs:
        p = os.path.join(det_dir, v, v + "_detections.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            f = int(r.get("frame", 0))
            for t in (r.get("track_ids") or []):
                if t is None or int(t) < 0:
                    continue
                k = "%s#%d" % (v, int(t))
                if k in keyset and (k not in out or f < out[k]):
                    out[k] = f
    return out


def topk_mean(E, qi, idx_list, k=3):
    """查询向量 qi 与池内样本的 top-k 平均余弦。

    ★ 多模板匹配的正确做法：取最像的 k 条，而不是池均值——池里可能混着
      换衣前的旧样本，取均值会把新样本的信号稀释掉。
    """
    if not idx_list:
        return -1.0
    v = E[idx_list] @ E[qi]
    if len(v) > k:
        v = np.sort(v)[-k:]
    return float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--segs", required=True, help="空格分隔的 8 段")
    ap.add_argument("--proto", default="full")
    ap.add_argument("--pool-k", type=int, default=3, help="多模板取最像的 k 条")
    ap.add_argument("--pool-max", type=int, default=8, help="每身份池容量（FIFO）")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85])
    ap.add_argument("--low-thr", type=float, default=0.60, help="换衣嫌疑档的上界")
    ap.add_argument("--lib-scope", choices=["cam", "all"], default="cam",
                    help="历史库按机位建还是全局建（跨机位外观不可比，默认 cam）")
    ap.add_argument("--pool-scope", choices=["cam", "all"], default="cam",
                    help="实时池按机位还是全局（身份级业务语义是全局，但特征跨机位不可比）")
    args = ap.parse_args()

    segs = args.segs.split()
    E64, keys = load_emb(args.emb_dir, segs)
    E = E64 / np.maximum(np.linalg.norm(E64, axis=1, keepdims=True), 1e-9)
    eidx = {k: i for i, k in enumerate(keys)}
    gt = json.load(open(args.gt, encoding="utf-8"))
    face = load_face(args.face, args.proto)
    start = load_start(args.det, segs, set(eidx))

    # 上午 = *_train_*，下午 = *_test_*（8 段命名即如此）
    def is_train(k):
        return "_train_" in k

    rows = []
    for k in keys:
        if k not in gt:
            continue
        rows.append(dict(k=k, i=eidx[k], pid=gt[k], cam=cam_of(k), tr=is_train(k),
                         t=start.get(k, 0), face=(k in face)))
    print("评估轨 %d ｜ 有人脸 %d（%.1f%%）｜ 机位 %d"
          % (len(rows), sum(r["face"] for r in rows),
             100 * sum(r["face"] for r in rows) / max(len(rows), 1),
             len({r["cam"] for r in rows})))

    # 历史库：上午段每身份均值（按 cam 分开建，或用全局）
    lib = {}
    for r in rows:
        if not r["tr"]:
            continue
        key = (r["cam"] if args.lib_scope == "cam" else "*", r["pid"])
        lib.setdefault(key, []).append(r["i"])
    libv = {}
    for key, idx in lib.items():
        v = E[idx].mean(0)
        libv[key] = v / max(np.linalg.norm(v), 1e-9)

    queries = sorted([r for r in rows if not r["tr"]], key=lambda r: (r["cam"], r["t"], r["k"]))
    print("查询轨（下午）%d ｜ 历史库（上午）%d 个「机位×身份」组"
          % (len(queries), len(libv)))

    # 池
    pool = collections.defaultdict(list)
    rec = []            # 每条查询的策略得分
    for q in queries:
        lscope = q["cam"] if args.lib_scope == "cam" else "*"
        pscope = q["cam"] if args.pool_scope == "cam" else "*"
        cands = [p for (sc, p) in libv if sc == lscope]
        if not cands:
            continue
        libsc = {p: float(E[q["i"]] @ libv[(lscope, p)]) for p in cands}
        poolsc = {p: topk_mean(E, q["i"], pool[(pscope, p)], args.pool_k) for p in cands}
        s0 = libsc
        s1 = {p: max(libsc[p], poolsc[p]) for p in cands}
        s2 = {p: poolsc[p] for p in cands}
        rec.append(dict(q=q, s0=s0, s1=s1, s2=s2,
                        own0=libsc.get(q["pid"], -1.0),
                        own1=max(libsc.get(q["pid"], -1.0), poolsc.get(q["pid"], -1.0)),
                        n_pool=len(pool[(pscope, q["pid"])])))
        # ★ 因果：打分完成后再写入池（这条查询看不到自己）
        if q["face"] and q["pid"] in cands:
            lst = pool[(pscope, q["pid"])]
            lst.append(q["i"])
            if len(lst) > args.pool_max:
                del lst[0]

    # ---------------- 报表
    def report(name):
        print("\n== %s ==" % name)
        print("  %-6s %-22s %-22s %-22s" % ("τ", "S0 仅历史库 cov/same",
                                            "S1 库+池 cov/same", "S2 仅池 cov/same"))
        for tau in args.tau:
            line = []
            for key in ("s0", "s1", "s2"):
                ans = [r for r in rec if max(r[key].values()) >= tau]
                if not ans:
                    line.append("%5s /%5s" % ("0%", "—"))
                    continue
                same = np.mean([max(r[key], key=r[key].get) == r["q"]["pid"] for r in ans])
                line.append("%4.0f%% /%5.1f%%" % (100 * len(ans) / len(rec), 100 * same))
            print("  %-6.2f %-22s %-22s %-22s" % (tau, line[0], line[1], line[2]))

    report("全部机位（库=%s / 池=%s）" % (args.lib_scope, args.pool_scope))

    # 按历史库相似度分档：看增益落在哪
    print("\n== 按「历史库本人余弦」分档（τ=0.80 判定）==")
    lo = args.low_thr
    bands = [("换衣嫌疑 <%.2f" % lo, lambda v: v < lo),
             ("中间 %.2f–0.80" % lo, lambda v: lo <= v < 0.80),
             ("高 ≥0.80", lambda v: v >= 0.80)]
    print("  %-18s %5s %-18s %-18s %-10s"
          % ("档", "n", "S0 same@0.80", "S1 same@0.80", "池覆盖"))
    for name, fn in bands:
        sub = [r for r in rec if fn(r["own0"])]
        if not sub:
            print("  %-18s %5d %-18s %-18s" % (name, 0, "—", "—"))
            continue
        out = []
        for key in ("s0", "s1"):
            ans = [r for r in sub if max(r[key].values()) >= 0.80]
            out.append("%4d/%4d = %5.1f%%" % (
                sum(1 for r in ans if max(r[key], key=r[key].get) == r["q"]["pid"]),
                len(sub), 100 * np.mean([max(r[key], key=r[key].get) == r["q"]["pid"]
                                         for r in ans]) if ans else 0.0))
        inpool = np.mean([r["n_pool"] > 0 for r in sub])
        print("  %-18s %5d %-18s %-18s %9.0f%%" % (name, len(sub), out[0], out[1], 100 * inpool))

    # 单列：换衣嫌疑档里，池救回了几条
    sub = [r for r in rec if r["own0"] < lo]
    if sub:
        print("\n== 换衣嫌疑档明细（历史库本人 cos < %.2f，共 %d 条）==" % (lo, len(sub)))
        print("  %-22s %5s %-16s %-16s" % ("身份", "n", "S0 top1 对", "S1 top1 对"))
        by = collections.defaultdict(list)
        for r in sub:
            by[r["q"]["pid"]].append(r)
        for p, rs in sorted(by.items(), key=lambda x: -len(x[1])):
            a0 = sum(1 for r in rs if max(r["s0"], key=r["s0"].get) == p)
            a1 = sum(1 for r in rs if max(r["s1"], key=r["s1"].get) == p)
            print("  %-22s %5d %-16d %-16d" % (p, len(rs), a0, a1))
        print("  合计：S0 %d/%d → S1 %d/%d"
              % (sum(1 for r in sub if max(r["s0"], key=r["s0"].get) == r["q"]["pid"]),
                 len(sub),
                 sum(1 for r in sub if max(r["s1"], key=r["s1"].get) == r["q"]["pid"]),
                 len(sub)))

    # 身份级：多少身份的 best-rank 变好
    print("\n== 身份级（每个身份的最佳排名，1=第一）==")
    by = collections.defaultdict(list)
    for r in rec:
        by[r["q"]["pid"]].append(r)
    print("  %-20s %5s %8s %8s %8s" % ("身份", "n", "S0均排名", "S1均排名", "池样本"))
    better = worse = 0
    for p, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        # ★ 该身份可能不在这个机位的历史库里（上午没出现）→ 不计排名
        rs = [r for r in rs if p in r["s0"]]
        if not rs:
            continue
        r0 = np.mean([sorted(r["s0"], key=r["s0"].get, reverse=True).index(p) + 1 for r in rs])
        r1 = np.mean([sorted(r["s1"], key=r["s1"].get, reverse=True).index(p) + 1 for r in rs])
        npl = int(np.mean([r["n_pool"] for r in rs]))
        if r1 < r0 - 0.01:
            better += 1
        elif r1 > r0 + 0.01:
            worse += 1
        print("  %-20s %5d %8.2f %8.2f %8d" % (p, len(rs), r0, r1, npl))
    n_eval = sum(1 for p, rs in by.items() if any(p in r["s0"] for r in rs))
    print("  → 排名变好 %d 个 / 变差 %d 个 / 不变 %d 个（可评 %d 个身份）"
          % (better, worse, n_eval - better - worse, n_eval))


if __name__ == "__main__":
    main()
