#!/usr/bin/env python
"""外观嵌入的归一化打分评估：raw cos vs 行/列 z-score vs 互最近邻。

目的：raw cosine 在"提议生成"场景假阳性率高（选择偏差见 REPORT §6.2），
评估 per-track 归一化能否用同一嵌入得到更好的判别。

用法:
    python tools/crosscam_normalize.py --emb .../reid_emb.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

# 全部已人工判读的标注（2026-10-12 三轮核验 + 时空判据定案；8 正 / 22 负）
LABELED = {
    (1, 1): "P", (14, 3): "P", (69, 14): "P", (120, 18): "P", (120, 23): "P",
    (139, 24): "P", (146, 28): "P", (37, 11): "P",
    (14, 1): "N", (131, 21): "N", (61, 14): "N", (50, 11): "N",
    (132, 21): "N", (3, 1): "N", (7, 1): "N", (146, 21): "N",
    (127, 19): "N", (123, 19): "N", (57, 11): "N", (131, 23): "N",
    (124, 18): "N", (131, 24): "N", (3, 4): "N",
    (126, 18): "N", (43, 11): "N", (10, 1): "N", (131, 28): "N",
    (123, 18): "N", (111, 18): "N",
    (98, 26): "N", (142, 26): "N", (53, 11): "N",
}


def zscore(M, i, j, axis):
    if axis == "row":
        v = np.delete(M[i, :], j)
    else:
        v = np.delete(M[:, j], i)
    return (M[i, j] - v.mean()) / (v.std() + 1e-9)


def auc(pairs_pos, pairs_neg, key):
    c = 0.0
    for p in pairs_pos:
        for n in pairs_neg:
            if p[key] > n[key]:
                c += 1
            elif p[key] == n[key]:
                c += 0.5
    return c / (len(pairs_pos) * len(pairs_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    z = np.load(args.emb)
    A = sorted(int(k[1:]) for k in z.files if k.startswith("a"))
    B = sorted(int(k[1:]) for k in z.files if k.startswith("b"))
    Ea = {a: z[f"a{a}"] for a in A}
    Eb = {b: z[f"b{b}"] for b in B}

    S = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            S[i, j] = float(np.dot(Ea[a], Eb[b]))

    rows = []
    for (a, b), lab in LABELED.items():
        if a not in Ea or b not in Eb:
            continue
        i, j = A.index(a), B.index(b)
        raw = float(S[i, j])
        zr = zscore(S, i, j, "row")
        zc = zscore(S, i, j, "col")
        mnn = (A[int(np.argmax(S[:, j]))] == a) and (B[int(np.argmax(S[i, :]))] == b)
        rows.append({"lab": lab, "a": a, "b": b, "raw": raw,
                     "zr": float(zr), "zc": float(zc),
                     "sum": float(zr + zc), "mnn": bool(mnn)})

    P = [r for r in rows if r["lab"] == "P"]
    N = [r for r in rows if r["lab"] == "N"]
    print(f"标注: {len(P)} 正 / {len(N)} 负")
    for key in ("raw", "zr", "zc", "sum"):
        pv = [r[key] for r in P]
        nv = [r[key] for r in N]
        print(f"{key:>5}: AUC={auc(P, N, key):.3f}  "
              f"min(P)={min(pv):+.3f}  max(N)={max(nv):+.3f}")

    print("\n=== 按 row+col z 排序全部标注对 ===")
    for r in sorted(rows, key=lambda r: -r["sum"]):
        print(f"{r['lab']} A{r['a']:>4} x B{r['b']:<4} raw={r['raw']:.3f} "
              f"rz={r['zr']:+.2f} cz={r['zc']:+.2f} sum={r['sum']:+.2f} "
              f"mnn={r['mnn']}")

    # 互最近邻作为硬过滤器：查它能否零误伤地保留所有正样本
    pos_kept = sum(1 for r in P if r["mnn"])
    neg_kept = sum(1 for r in N if r["mnn"])
    print(f"\n互最近邻过滤器: 正样本保留 {pos_kept}/{len(P)}，"
          f"负样本放行 {neg_kept}/{len(N)}")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"labeled": rows, "n_pos": len(P), "n_neg": len(N)},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
