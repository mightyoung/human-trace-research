#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨天体检：把 10-12（8 段）上定下的阈值，拿到 10-10（long35）上量一遍。

为什么不能直接复现 56.4 / 68.2 / 79.3
------------------------------------
8 段那三档都是「识别到谁」的 recall，分母是**有 pid 真值的查询轨**。
long35 只有 **126 条对级真值**（34 same / 92 diff），**没有完整 pid**
→ 无法构造「库 → 查询」的识别任务，只能做**配对判定**。
所以本脚本不比绝对值，只比三件在两天都成立的东西：

  T1 阈值迁移：τ 在 long35 的 126 对上的 P / R —— 8 段定的 .75/.78 换天还灵不灵
  T2 判别力：AUC，按机位对分层（两天比，AUC 不受样本量影响）
  T3 S5 前提：同人对 vs 异人对的 |log 框高比| 分布
     —— S5「按形态挑模板」只有在同人形态跨度足够大、且异人形态不更相似时才成立

口径声明
--------
- long35：`deliverable/gt_long35/gt_long35.json` 的 126 条人工判读（106 轨对 + 20 实体对）
- 8 段：`gt_8seg.json` 的 pid 全展开（同 pid=1 / 异 pid=0）
- 两者**总体不同**（精选难点对 vs 全组合），只比曲线形状与 AUC，不比重合绝对值

用法
----
  python tools/xday_long35_probe.py
"""
import collections
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

L35_EMB = "output_crosscam/long35/embs_osnet"
L35_TRACKS = "webapp/backend/data_long35/tracks"
L35_GT = "deliverable/gt_long35/gt_long35.json"
S8_EMB = "tmp/full/emb12"
S8_DET = "output_crosscam_local/1012_yolo8_dd"
S8_GT = "deliverable/paper-package-2026-11-10/20-data/e4-full-corpus/gt_8seg.json"


# ---------------------------------------------------------------- 通用度量
def auc_score(pos, neg):
    """Mann-Whitney AUC，不依赖 sklearn。"""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 同分取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_at_tau(sc, lab, tau):
    sc = np.asarray(sc, dtype=float)
    lab = np.asarray(lab, dtype=int)
    pred = sc >= tau
    tp = int(((pred == 1) & (lab == 1)).sum())
    fp = int(((pred == 1) & (lab == 0)).sum())
    npos = int((lab == 1).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(npos, 1)
    return p, r, tp, fp, int(pred.sum())


def recall_at_p(sc, lab, p_target):
    """P >= p_target 时能达到的最大召回（扫所有出现的分数）。"""
    sc = np.asarray(sc, dtype=float)
    lab = np.asarray(lab, dtype=int)
    npos = int((lab == 1).sum())
    if npos == 0:
        return float("nan"), float("nan")
    best_r, best_tau = -1.0, float("nan")
    for tau in np.unique(sc)[::-1]:
        p, r, tp, fp, _ = pr_at_tau(sc, lab, tau)
        if p >= p_target and r > best_r:
            best_r, best_tau = r, tau
    return (best_r if best_r >= 0 else float("nan")), best_tau


# ---------------------------------------------------------------- long35
def load_l35():
    E, idx = [], {}
    for p in sorted(glob.glob(os.path.join(L35_EMB, "*.npz"))):
        cam = os.path.basename(p).split("_")[0]
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            tid = k.split("_", 1)[1]
            idx[f"{cam}:{tid}"] = len(E)
            E.append(np.asarray(z[k], dtype=np.float32))
    E = np.asarray(E)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

    # 框高中位（stride 5 的 tracks）
    H = {}
    for p in sorted(glob.glob(os.path.join(L35_TRACKS, "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        cam = d["cam"]
        for tid, rows in d["tracks"].items():
            hs = [abs(r[4] - r[2]) for r in rows if abs(r[4] - r[2]) > 0]
            if hs:
                H[f"{cam}:{tid}"] = float(np.median(hs))

    gt = json.load(open(L35_GT, encoding="utf-8"))["items"]
    return E, idx, H, gt


# ---------------------------------------------------------------- 8 段
def load_s8():
    E, idx, segof = [], {}, {}
    for p in sorted(glob.glob(os.path.join(S8_EMB, "*_emb.npz"))):
        seg = os.path.basename(p).replace("_emb.npz", "")
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            idx[k] = len(E)
            segof[k] = seg
            E.append(np.asarray(z[k], dtype=np.float32))
    E = np.asarray(E)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)

    pid = json.load(open(S8_GT, encoding="utf-8"))

    # 框高中位：扫 detections.jsonl
    H = {}
    acc = collections.defaultdict(list)
    for seg in sorted({v for v in segof.values()}):
        f = os.path.join(S8_DET, seg, f"{seg}_detections.jsonl")
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                for tid, bx in zip(r["track_ids"], r["boxes"]):
                    h = abs(bx[3] - bx[1])
                    if h > 0:
                        acc[f"{seg}#{tid}"].append(h)
    for k, hs in acc.items():
        H[k] = float(np.median(hs))
    return E, idx, H, pid, segof


def cam_of_seg(seg):
    return seg.split("_")[0]


def main():
    print("=" * 78)
    print("跨天体检：10-12(8 段) 定阈值 → 10-10(long35) 验收")
    print("=" * 78)

    E1, I1, H1, gt1 = load_l35()
    print(f"\n[long35] 轨向量 {E1.shape[0]}｜有框高 {len(H1)}｜真值对 {len(gt1)}")

    sc, lab, pairof, missing = [], [], [], 0
    miss_rows = []
    for it in gt1:
        a, b = it["a"], it["b"]
        if a not in I1 or b not in I1:
            missing += 1
            miss_rows.append(it)
            continue
        sc.append(float(E1[I1[a]] @ E1[I1[b]]))
        lab.append(int(it["label"]))
        pairof.append(it.get("pair", "?"))
    sc = np.asarray(sc)
    lab = np.asarray(lab)
    print(f"  可评 {len(sc)} 对（缺向量 {missing}）｜same {int((lab==1).sum())} / diff {int((lab==0).sum())}")
    if miss_rows:
        # 缺向量的原因：tracks 里有框但没进 emb 编码（n_tracks 1894 > 编码 1574）
        lv = collections.Counter(r.get("level") for r in miss_rows)
        in_tr = sum(1 for r in miss_rows for k in (r["a"], r["b"]) if k in H1)
        print(f"  缺向量的对按层级：{dict(lv)}；其中 {in_tr}/{2*len(miss_rows)} 个端点仍有框高（= 有轨迹但未编码）")
        print(f"  例：{[(r['a'], r['b'], r.get('level')) for r in miss_rows[:4]]}")

    # ---------------- T1 阈值迁移
    print("\n" + "-" * 78)
    print("T1 阈值迁移：8 段定的 τ 拿到 10-10，P/R 变成什么样")
    print(f"  {'τ':>5} {'P':>7} {'R':>7} {'TP':>4} {'FP':>4} {'预测数':>7}")
    for tau in (0.70, 0.75, 0.78, 0.80, 0.85, 0.90):
        p, r, tp, fp, n = pr_at_tau(sc, lab, tau)
        mark = "  ← 生产阈值" if abs(tau - 0.75) < 1e-9 else ("  ← 8 段聚类阈值" if abs(tau - 0.78) < 1e-9 else "")
        print(f"  {tau:>5.2f} {p:>7.3f} {r:>7.3f} {tp:>4d} {fp:>4d} {n:>7d}{mark}")
    r95, tau95 = recall_at_p(sc, lab, 0.95)
    p95, _, _, _, _ = pr_at_tau(sc, lab, tau95) if not np.isnan(tau95) else (float("nan"),) * 5
    print(f"\n  P≥0.95 时最大 R = {r95:.3f}（τ={tau95:.3f}，实测 P={p95:.3f}）")

    # ---------------- T2 判别力
    print("\n" + "-" * 78)
    print("T2 判别力：AUC（不受样本量影响）")
    pos = sc[lab == 1]
    neg = sc[lab == 0]
    a_all = auc_score(pos, neg)
    print(f"  long35 总体 AUC = {a_all:.3f}（pos {len(pos)} / neg {len(neg)}）")
    print(f"  同人 cos 中位 {np.median(pos):.3f}（四分位 {np.percentile(pos,25):.3f}–{np.percentile(pos,75):.3f}）")
    print(f"  异人 cos 中位 {np.median(neg):.3f}（四分位 {np.percentile(neg,25):.3f}–{np.percentile(neg,75):.3f}）")
    print(f"\n  {'机位对':>10} {'n_pos':>6} {'n_neg':>6} {'AUC':>7} {'同人中位':>9} {'异人中位':>9}")
    for pr in sorted(set(pairof)):
        m = np.array([x == pr for x in pairof])
        s, l = sc[m], lab[m]
        a = auc_score(s[l == 1], s[l == 0])
        mp = np.median(s[l == 1]) if (l == 1).any() else float("nan")
        mn = np.median(s[l == 0]) if (l == 0).any() else float("nan")
        print(f"  {pr:>10} {int((l==1).sum()):>6d} {int((l==0).sum()):>6d} {a:>7.3f} {mp:>9.3f} {mn:>9.3f}")

    # ---------------- T3 S5 前提
    print("\n" + "-" * 78)
    print("T3 S5 前提：同人 vs 异人的 |log 框高比|（S5 只在同人形态跨度大时才有空间）")
    dh_pos, dh_neg = [], []
    for s, l, it in zip(sc, lab, [x for x in gt1 if x["a"] in I1 and x["b"] in I1]):
        ha, hb = H1.get(it["a"]), H1.get(it["b"])
        if not ha or not hb:
            continue
        d = abs(np.log(ha / hb))
        (dh_pos if l == 1 else dh_neg).append(d)
    if dh_pos and dh_neg:
        print(f"  同人对 n={len(dh_pos)} 中位 {np.median(dh_pos):.2f}（= {np.exp(np.median(dh_pos)):.2f}× 框高差）")
        print(f"  异人对 n={len(dh_neg)} 中位 {np.median(dh_neg):.2f}（= {np.exp(np.median(dh_neg)):.2f}×）")
        print(f"  AUC(|log h比| 越小越同人) = {auc_score([-x for x in dh_pos], [-x for x in dh_neg]):.3f}")
    else:
        print("  框高缺失，跳过")

    # ---------------- 8 段对照
    print("\n" + "=" * 78)
    print("8 段同口径对照（pid 全展开：同 pid=1 / 异 pid=0）")
    E2, I2, H2, pid2, segof = load_s8()
    print(f"  [8 段] 轨向量 {E2.shape[0]}｜有 pid {len(pid2)}｜有框高 {len(H2)}")

    keys = [k for k in I2 if k in pid2]
    K = len(keys)
    print(f"  参与 {K} 轨，全组合 {K*(K-1)//2} 对")
    ii = np.array([I2[k] for k in keys])
    V = E2[ii]
    S = V @ V.T
    iu = np.triu_indices(K, 1)
    s8 = S[iu]
    p8 = np.array([pid2[keys[i]] for i in range(K)])
    l8 = (p8[iu[0]] == p8[iu[1]]).astype(int)
    same_cam = np.array([cam_of_seg(segof[keys[i]]) == cam_of_seg(segof[keys[j]])
                         for i, j in zip(*iu)])
    # 分层：跨机位 vs 同机位
    print(f"\n  {'分层':>14} {'n_pos':>7} {'n_neg':>8} {'AUC':>7} {'P@.75':>7} {'R@.75':>7}")
    for nm, m in (("全部", np.ones(len(s8), bool)),
                  ("同机位", same_cam),
                  ("跨机位", ~same_cam)):
        s, l = s8[m], l8[m]
        # 跨机位/同机位内 负例可能极大，下采样控制在 200k
        if len(s) > 200000:
            rs = np.random.RandomState(0)
            keep = rs.choice(len(s), 200000, replace=False)
            s, l = s[keep], l[keep]
        a = auc_score(s[l == 1], s[l == 0])
        p, r, _, _, _ = pr_at_tau(s, l, 0.75)
        print(f"  {nm:>14} {int((l==1).sum()):>7d} {int((l==0).sum()):>8d} {a:>7.3f} {p:>7.3f} {r:>7.3f}")
    r95_8, tau95_8 = recall_at_p(s8, l8, 0.95) if len(s8) <= 400000 else (float("nan"), float("nan"))
    print(f"\n  8 段 P≥0.95 时最大 R = {r95_8:.3f}（τ={tau95_8:.3f}）")

    # ---------------- 跨天逐机位对对照（唯一同总体的比法：同一对机位，两天各算一次）
    print("\n" + "-" * 78)
    print("跨天逐机位对对照：同一对机位在两天各算一次 AUC")
    print("  ★ 稳健性列：pid 全展开里，单个身份可能占掉绝大部分正例（如 female5 占 318-320 的 89%），")
    print("    此时机位对 AUC 量的是「那一个人的标注噪声」，不是机位对能力 → 必须看去掉后的值。")
    cam8 = np.array([cam_of_seg(segof[keys[i]]) for i in range(K)])
    pair8 = cam8[iu[0]] + "-" + cam8[iu[1]]
    l35_pairs = {}
    for pr in sorted(set(pairof)):
        m = np.array([x == pr for x in pairof])
        l35_pairs[pr] = (auc_score(sc[m][lab[m] == 1], sc[m][lab[m] == 0]),
                         int((lab[m] == 1).sum()), int((lab[m] == 0).sum()))
    print(f"\n  {'机位对':>9} {'l35AUC':>7} {'n+/n-':>8} | {'8段AUC':>7} {'n+/n-':>13} "
          f"{'支配身份':>9} {'占比':>5} {'去支配后':>8} {'去后n+':>6} {'差(l35-去)':>10}")
    rows_pair = {}
    for pr in sorted(set(pair8)):
        a1, p1, n1 = l35_pairs.get(pr, (float("nan"), 0, 0))
        m = pair8 == pr
        sel = np.where(m)[0]
        if len(sel) > 150000:
            sel = np.random.RandomState(1).choice(sel, 150000, replace=False)
        s, l = s8[sel], l8[sel]
        ii0 = iu[0][sel]
        a2 = auc_score(s[l == 1], s[l == 0])
        ids = np.array([pid2[keys[i]] for i in range(K)])
        pos_ids = list(ids[ii0[l == 1]])
        cnt = collections.Counter(pos_ids)
        top, topn = (cnt.most_common(1)[0] if cnt else ("-", 0))
        share = topn / max(len(pos_ids), 1)
        keep_pos = np.array([x != top for x in pos_ids])
        s_wo = np.concatenate([s[l == 1][keep_pos], s[l == 0]])
        l_wo = np.concatenate([np.ones(int(keep_pos.sum())), np.zeros(int((l == 0).sum()))])
        a2w = auc_score(s_wo[l_wo == 1], s_wo[l_wo == 0])
        d = (a1 - a2w) if (a1 == a1 and a2w == a2w) else float("nan")
        rows_pair[pr] = (a1, p1, n1, a2, top, share, a2w, int(keep_pos.sum()))
        has = "" if a1 == a1 else "  ←l35 无真值"
        print(f"  {pr:>9} {a1:>7.3f} {f'{p1}/{n1}':>8} | {a2:>7.3f} "
              f"{f'{int((l==1).sum())}/{int((l==0).sum())}':>13} {top:>9} {share:>5.0%} "
              f"{a2w:>8.3f} {int(keep_pos.sum()):>6d} {d:>10.3f}{has}")

    # ---------------- T3b 形态可分性：同机位 vs 跨机位
    print("\n" + "-" * 78)
    print("T3b 形态（框高）到底在哪一层可分？S5 的增益只在同机位有前提")
    hs8 = np.array([H2.get(keys[i], np.nan) for i in range(K)])
    ok = np.isfinite(hs8) & (hs8 > 0)
    print(f"  8 段有框高 {int(ok.sum())}/{K} 轨")
    for nm, m in (("8 段 同机位", same_cam), ("8 段 跨机位", ~same_cam)):
        dp, dn = [], []
        sel = np.where(m & ok[iu[0]] & ok[iu[1]])[0]
        if len(sel) > 300000:
            sel = np.random.RandomState(2).choice(sel, 300000, replace=False)
        for q in sel:
            i, j = iu[0][q], iu[1][q]
            dd = abs(np.log(hs8[i] / hs8[j]))
            (dp if l8[q] == 1 else dn).append(dd)
        if dp and dn:
            print(f"  {nm:<14} 同人 |log h比| 中位 {np.median(dp):.2f}（{np.exp(np.median(dp)):.2f}×）"
                  f"｜异人 {np.median(dn):.2f}（{np.exp(np.median(dn)):.2f}×）"
                  f"｜AUC {auc_score([-x for x in dp], [-x for x in dn]):.3f}")

    out = {
        "long35": {
            "n_pairs": int(len(sc)), "n_pos": int((lab == 1).sum()), "n_neg": int((lab == 0).sum()),
            "auc": a_all,
            "same_median": float(np.median(pos)) if len(pos) else None,
            "diff_median": float(np.median(neg)) if len(neg) else None,
            "recall_at_p95": r95, "tau_at_p95": tau95,
            "curve": {str(t): pr_at_tau(sc, lab, t)[:2] for t in (0.70, 0.75, 0.78, 0.80, 0.85, 0.90)},
            "dh_same_median": float(np.median(dh_pos)) if dh_pos else None,
            "dh_diff_median": float(np.median(dh_neg)) if dh_neg else None,
        },
        "seg8": {
            "n_tracks": K, "auc_all": auc_score(s8[l8 == 1], s8[l8 == 0]),
            "recall_at_p95": r95_8, "tau_at_p95": tau95_8,
        },
    }
    os.makedirs("tmp/full", exist_ok=True)
    json.dump(out, open("tmp/full/xday_long35_probe.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写 tmp/full/xday_long35_probe.json")


if __name__ == "__main__":
    main()
