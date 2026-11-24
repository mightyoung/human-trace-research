"""WILDTRACK · 像素扫描（远端 GPU 版）。

与本地版同一口径，只是编码放 GPU（L20）：871 crops/s 量级，8 档一次跑完。
输入：
  --crops   crops_wt 目录（meta.json + jpg）
  --h0      crop_h0.json（"stem|pid|view" -> 原图框高 px）
  --cov     wildtrack_p2.json（取每对的实测共视率）
输出：
  --out     pixel_sweep.json（每档每对的等量重采样 AUC）+ sweep_{p}.npz
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchreid.reid.models import build_model

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NV = 7
SCALES = [320, 160, 112, 80, 56, 40, 24, 16]
K, REP, N_STEM = 15, 200, 10


def degrade(im, factor):
    if factor >= 0.999:
        return im
    h, w = im.shape[:2]
    h2 = max(8, int(round(h * factor)))
    w2 = max(4, int(round(w * factor)))
    small = cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def prep(im):
    x = cv2.resize(im, (128, 256), interpolation=cv2.INTER_LINEAR)
    x = x[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))


@torch.no_grad()
def encode(net, imgs, bs=256):
    out = []
    for i in range(0, len(imgs), bs):
        arr = np.stack([prep(im) for im in imgs[i:i + bs]])
        t = torch.from_numpy(arr).cuda()
        f = F.normalize(net(t), dim=1)
        out.append(f.half().float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def auc_of(pos, neg):
    if len(pos) < 5 or len(neg) < 5:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort().astype(float)
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def spear(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="/home/r4900g3/wt/crops_wt")
    ap.add_argument("--h0", default="/home/r4900g3/wt/crop_h0.json")
    ap.add_argument("--cov", default="/home/r4900g3/wt/wildtrack_p2.json")
    ap.add_argument("--weight", default="/home/r4900g3/r5/osnet_x1_0_msmt17.pth")
    ap.add_argument("--out", default="/home/r4900g3/wt/pixel_sweep.json")
    args = ap.parse_args()

    CROPS = Path(args.crops)
    meta = json.loads((CROPS / "meta.json").read_text(encoding="utf-8"))
    h0 = json.loads(Path(args.h0).read_text(encoding="utf-8"))
    stems = sorted({m["stem"] for m in meta})
    idx = np.linspace(0, len(stems) - 1, N_STEM).round().astype(int)
    keep = {stems[i] for i in sorted(set(idx))}
    meta = [m for m in meta if m["stem"] in keep]
    H0 = np.array([h0.get(f"{m['stem']}|{m['pid']}|{m['k']}", 0.0) for m in meta])
    HEFF = np.where(H0 > 0, np.minimum(H0, 320.0), 320.0)
    print(f"帧 {len(keep)}，crop {len(meta)}，h0 中位 {np.median(H0):.0f}", flush=True)

    net = build_model("osnet_x1_0", num_classes=1, pretrained=False)
    sd = torch.load(args.weight, map_location="cpu")
    net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("classifier")},
                        strict=False)
    net.eval().cuda()

    base = [cv2.imread(str(CROPS / m["file"])) for m in meta]
    base = [b if b is not None else np.zeros((320, 160, 3), np.uint8) for b in base]

    covmap = {}
    for r in json.loads(Path(args.cov).read_text(encoding="utf-8"))["pairs"]:
        covmap[(r["a"], r["b"])] = r["cov"]

    byv = defaultdict(lambda: defaultdict(list))
    for i, m in enumerate(meta):
        byv[m["k"]][m["pid"]].append((i, m["stem"]))

    res = {}
    for p in SCALES:
        E = encode(net, [degrade(im, min(1.0, p / max(hh, 1.0)))
                         for im, hh in zip(base, HEFF)])
        np.savez_compressed(Path(args.out).parent / f"sweep_{p}.npz", E=E)
        rng = np.random.default_rng(7)
        rows = []
        for i, j in combinations(range(NV), 2):
            shared = sorted(set(byv[i]) & set(byv[j]))
            if len(shared) < K:
                continue
            pos_by = {}
            neg_pool = {}
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
            rows.append(dict(a=i, b=j, cov=covmap.get((i, j), 0.0), n_id=len(shared),
                             auc=round(float(np.mean(vals)), 4),
                             lo=round(float(np.percentile(vals, 2.5)), 3),
                             hi=round(float(np.percentile(vals, 97.5)), 3)))
        res[p] = rows
        med = float(np.median([r["auc"] for r in rows])) if rows else float("nan")
        print(f"[{p}px] 有效 {np.median(np.minimum(p, HEFF)):.0f}px  n={len(rows)}  中位 AUC {med:.3f}",
              flush=True)

    print("\n各档中位 AUC")
    for p in SCALES:
        a = [r["auc"] for r in res[p]]
        if a:
            print(f"  {p:>4}px  n={len(a):>2}  中位 {np.median(a):.3f}  均值 {np.mean(a):.3f}"
                  f"  范围 {min(a):.3f}-{max(a):.3f}")

    low = min(SCALES)
    m_low = {(r["a"], r["b"]): r["auc"] for r in res[low]}
    m_nat = {(r["a"], r["b"]): r["auc"] for r in res[320]}
    both = [k for k in m_low if k in m_nat]
    d = [m_low[k] - m_nat[k] for k in both]
    c = [covmap.get(k, 0.0) for k in both]
    print(f"\nΔAUC({low}px − native)：中位 {np.median(d):+.3f}，"
          f"范围 {min(d):+.3f}…{max(d):+.3f}，n={len(d)}")
    print(f"  ΔAUC vs 共视：r={np.corrcoef(c, d)[0,1]:+.3f}  ρ={spear(c, d):+.3f}")

    Path(args.out).write_text(json.dumps(
        dict(scales=SCALES, n_stem=N_STEM, k=K, rep=REP, device="cuda",
             pairs={str(p): v for p, v in res.items()}), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
