"""WILDTRACK · 阶段 F：合成像素扫描（P1 的剂量-反应曲线 + P1×P2 交互）。

问题：WILDTRACK 全部机位的人已有 141–341 px，AUC 普遍 0.7–0.9，没有方差可解释；
本站点人脸中位 52 px，AUC 0.45–0.62。两端之间没有数据点，P1 的拐点无法校准。

做法：**把同一批裁剪按目标人物高度降采样再升回**，制造中间的像素档位：
  对每个 crop，原始框高 h0（来自标注），目标高度 p，
  缩放因子 f = min(1, p/h0) → 把 320 高的存图缩到 320f 再升回 320（INTER_AREA / CUBIC）。
  有效人物像素 = min(p, h0)，因此 x 轴用**有效像素**而不是名义像素。

档位：native(320) / 160 / 112 / 80 / 56 / 40
配对与评分沿用阶段 C2：每对抽 15 个身份 × 200 次，消除身份构成偏差。

同时检验交互：ΔAUC(p=40 − native) 是否与共视率相关（若共视高的对退化更慢，
说明 P2 在低像素区才开始起作用）。

产物：tmp/wt/sweep_{p}.npz（每档特征）、tmp/wt/pixel_sweep.json
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

OUT = Path("tmp/wt")
CROPS = OUT / "crops_wt"
ANN = OUT / "data/Wildtrack_dataset/annotations_positions"
NV = 7
SCALES = [320, 160, 112, 80, 56, 40, 24, 16]
N_STEM = 10          # 每视角用 10 帧（20 帧的一半），控制总时长
K, REP = 15, 200


def box_heights():
    """(stem, pid, view) -> 原图框高 h0（px）。"""
    h = {}
    for f in sorted(ANN.glob("*.json")):
        for rec in json.loads(f.read_text(encoding="utf-8")):
            pid = rec.get("personID")
            for v in rec.get("views") or []:
                k = v.get("viewNum")
                if not isinstance(k, int) or not (0 <= k < NV):
                    continue
                y1, y2 = v.get("ymin", -1), v.get("ymax", -1)
                x1, x2 = v.get("xmin", -1), v.get("xmax", -1)
                if min(x1, y1, x2, y2) < 0 or x2 - x1 < 10 or y2 - y1 < 20:
                    continue
                h[(f.stem, pid, k)] = y2 - y1
    return h


def pick_stems(meta):
    stems = sorted({m["stem"] for m in meta})
    idx = np.linspace(0, len(stems) - 1, N_STEM).round().astype(int)
    return {stems[i] for i in sorted(set(idx))}


def degrade(im, factor):
    if factor >= 0.999:
        return im
    h, w = im.shape[:2]
    h2 = max(8, int(round(h * factor)))
    w2 = max(4, int(round(w * factor)))
    small = cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def auc_of(pos, neg):
    """Mann-Whitney：P(同人得分 > 异人得分)。秩从 1 起算，并列取平均秩（向量化）。

    2026-11-25 修正：原实现用 argsort().argsort() 得到 0 起秩，却代入 1 起公式，
    结果恒低 1/n_neg（8 正 8 负完全分离时返回 0.875 而非 1.0），且未处理并列。
    """
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))

def spear(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    sys.path.insert(0, "tmp/wt")
    from wt_fast_embed import Encoder

    meta = json.loads((CROPS / "meta.json").read_text(encoding="utf-8"))
    h0 = box_heights()
    keep = pick_stems(meta)
    meta = [m for m in meta if m["stem"] in keep]
    print(f"帧子集 {len(keep)} 帧，crop {len(meta)} 个", flush=True)

    H0 = np.array([h0.get((m["stem"], m["pid"], m["k"]), 0.0) for m in meta])
    # ★ 存图固定 320 高：原始框比 320 还大的人已经被缩过一次，
    #   所以「有效上限」是 min(h0, 320)，否则 native(320) 档会被二次缩小。
    HEFF = np.where(H0 > 0, np.minimum(H0, 320.0), 320.0)
    print("原始框高 h0：中位 %.0f，p10 %.0f，p90 %.0f（有效上限 min(h0,320) 中位 %.0f）"
          % (np.median(H0), np.percentile(H0, 10), np.percentile(H0, 90),
             np.median(HEFF)), flush=True)

    enc = None
    for p in SCALES:
        dst = OUT / f"sweep_{p}.npz"
        if dst.exists():
            print(f"[{p}px] 已有，跳过", flush=True)
            continue
        if enc is None:
            enc = Encoder(batch=64)
        imgs = []
        for m, hh in zip(meta, HEFF):
            im = cv2.imread(str(CROPS / m["file"]))
            if im is None:
                im = np.zeros((320, 160, 3), np.uint8)
            f = min(1.0, p / max(hh, 1.0))
            imgs.append(degrade(im, f))
        E = enc.encode(imgs)
        np.savez_compressed(dst, E=E)
        eff = np.minimum(p, HEFF)
        print(f"[{p}px] 有效像素中位 {np.median(eff):.0f}  完成 {E.shape}", flush=True)

    # 评分
    cov = {r["a"] * 10 + r["b"]: r["cov"]
           for r in json.loads((OUT / "wildtrack_p2.json").read_text(encoding="utf-8"))["pairs"]}
    byv = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(meta):
        byv[m["k"]][m["pid"]].append((i, m["stem"]))

    res = {}
    for p in SCALES:
        E = np.load(OUT / f"sweep_{p}.npz")["E"].astype(np.float32)
        rng = np.random.default_rng(7)
        rows = []
        for i, j in combinations(range(NV), 2):
            shared = sorted(set(byv[i]) & set(byv[j]))
            if len(shared) < K:
                continue
            pos_by, neg_pool = {}, {}
            for pid in shared:
                Ai, Bj = byv[i][pid], byv[j][pid]
                pos_by[pid] = np.array([float(E[a] @ E[b]) for a, sa in Ai
                                        for b, sb in Bj if sa != sb])
                neg_pool[pid] = [q for q in shared if q != pid]
            vals = []
            for _ in range(REP):
                pick = rng.choice(shared, K, replace=False)
                pos = np.concatenate([pos_by[x] for x in pick])
                neg = np.array([float(E[byv[i][x][rng.integers(len(byv[i][x]))][0]]
                                      @ E[byv[j][y][rng.integers(len(byv[j][y]))][0]])
                                for x in pick for y in neg_pool[x]])
                a = auc_of(pos, neg)
                if a is not None:
                    vals.append(a)
            if len(vals) < 30:
                continue
            rows.append(dict(a=i, b=j, cov=cov.get(i * 10 + j, 0.0),
                             n_id=len(shared),
                             auc=round(float(np.mean(vals)), 4),
                             lo=round(float(np.percentile(vals, 2.5)), 3),
                             hi=round(float(np.percentile(vals, 97.5)), 3)))
        res[p] = rows
        med = float(np.median([r["auc"] for r in rows])) if rows else float("nan")
        print(f"[{p}px] 可用对 {len(rows)}  中位 AUC {med:.3f}", flush=True)

    # 汇总表
    pairs = [f"C{i+1}-C{j+1}" for i, j in combinations(range(NV), 2)]
    print("\n各档中位 AUC（按对中位 / 全部对合并中位）")
    for p in SCALES:
        if p in res and res[p]:
            a = [r["auc"] for r in res[p]]
            print(f"  {p:>4}px  n={len(a):>2}  中位 {np.median(a):.3f}  "
                  f"范围 {min(a):.3f}–{max(a):.3f}")

    # 交互：ΔAUC(40 - native) vs 共视
    if 40 in res and 320 in res:
        m40 = {(r["a"], r["b"]): r["auc"] for r in res[40]}
        m320 = {(r["a"], r["b"]): r["auc"] for r in res[320]}
        both = [k for k in m40 if k in m320]
        d = [m40[k] - m320[k] for k in both]
        c = [next(r["cov"] for r in res[low] if (r["a"], r["b"]) == k) for k in both]
        print(f"\nΔAUC({low}px − native)：中位 {np.median(d):+.3f}，"
              f"范围 {min(d):+.3f}…{max(d):+.3f}，n={len(d)}")
        print(f"  ΔAUC vs 共视：r={np.corrcoef(c, d)[0,1]:+.3f}  ρ={spear(c, d):+.3f}")
        for k, dd, cc in sorted(zip(both, d, c), key=lambda z: z[2]):
            print(f"    C{k[0]+1}-C{k[1]+1}  共视 {cc:.2f}  native {m320[k]:.3f} → 40px {m40[k]:.3f}"
                  f"  Δ{dd:+.3f}")

    (OUT / "pixel_sweep.json").write_text(
        json.dumps(dict(scales=SCALES, n_stem=N_STEM, k=K, rep=REP,
                        pairs={str(p): v for p, v in res.items()}),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n-> tmp/wt/pixel_sweep.json")


if __name__ == "__main__":
    main()
