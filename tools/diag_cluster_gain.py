#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""两种聚类配置的逐轨迹差分：新增/丢失的测试轨迹是什么货色。

回答的问题：覆盖率涨的那几个点，是真的人被认回来了，还是新放了噪声/碎片轨迹进来。
输出：分机位覆盖、新增/丢失轨迹的观测数分布、a/b 一致率。
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402
import eval_mask_variants as M  # noqa: E402
import cluster_camera_aware as CA  # noqa: E402

ROOT = S.ROOT


def obs_counts(det, segs):
    out = {}
    for s in segs:
        p = os.path.join(det, s, f"{s}_detections.jsonl")
        c = collections.Counter()
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                d = json.loads(line)
                for t in d.get("track_ids") or []:
                    c[(s, int(t))] += 1
        out.update(c)
    return out


def load(emb, segs):
    keys, As, Bs = [], [], []
    for s in segs:
        k, a, b = M.load_pair(emb, s)
        if len(k):
            keys += k
            As.append(a)
            Bs.append(b)
    return keys, np.concatenate(As), np.concatenate(Bs)


def solve(keys_tr, Atr, Btr, keys_te, Ate, Bte, cooc_te, mode, kw, same_min=0.90, min_eval=100):
    Vtr = _norm((Atr + Btr) / 2.0)
    if mode == "camera":
        cams = np.array([k[0].split("_")[0] for k in keys_tr])
        lab, kept = CA.camera_aware_labels(Vtr, cams, kw["thr_in"], kw["thr_out"], S.MIN_CLUSTER)
    else:
        lab, kept = S.hier_labels(Vtr, kw["hier_thr"])
    K = len(kept)
    mu_g = Vtr.mean(0)
    protos = np.asarray([_norm(Vtr[lab == c].mean(0, keepdims=True))[0] for c in kept], np.float32)
    protos = _norm(protos - mu_g)
    pcl = np.arange(K)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Ate + Bte) / 2.0).mean(0)
    Sa = S.score_rows(Ate, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Bte, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    rows = S.sweep(Sa, Sb, keys_te, cooc_te, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= same_min and r["evaluable"] >= min_eval]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    thr = best["thr"]
    acc = (Sa.max(1) >= thr) & (Sb.max(1) >= thr)
    return {"thr": thr, "best": best, "acc": dict(zip(keys_te, acc.tolist())),
            "ia": dict(zip(keys_te, Sa.argmax(1).tolist())),
            "ib": dict(zip(keys_te, Sb.argmax(1).tolist())),
            "clu_cams": [sorted(set(np.array([k[0].split("_")[0] for k in keys_tr])[lab == c].tolist()))
                         for c in kept],
            "clu_size": [int((lab == c).sum()) for c in kept],
            "K": K, "n_tr": len(keys_tr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb"))
    ap.add_argument("--det", default=os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd"))
    ap.add_argument("--out", default=os.path.join(ROOT, "output_crosscam_local/selector_osnet_1012/cluster_delta.json"))
    a = ap.parse_args()

    ktr, Atr, Btr = load(a.emb, S.TRAIN_SEGS)
    kte, Ate, Bte = load(a.emb, S.TEST_SEGS)
    cooc_te = []
    for s in S.TEST_SEGS:
        p = os.path.join(a.det, s, f"{s}_detections.jsonl")
        if os.path.exists(p):
            cooc_te += [((s, u), (s, v)) for u, v in cooc_pairs(p, set(kte), s, 5)]

    base = solve(ktr, Atr, Btr, kte, Ate, Bte, cooc_te, "global", dict(hier_thr=0.20))
    cand = solve(ktr, Atr, Btr, kte, Ate, Bte, cooc_te, "camera", dict(thr_in=0.15, thr_out=0.25))
    obs = obs_counts(a.det, S.TEST_SEGS)

    print(f"global 0.20: thr={base['thr']:.2f} cov={base['best']['coverage']:.3f} "
          f"same={base['best']['same']:.3f} fm={base['best']['false_merge']*100:.2f}%")
    print(f"cam 0.15/0.25: thr={cand['thr']:.2f} cov={cand['best']['coverage']:.3f} "
          f"same={cand['best']['same']:.3f} fm={cand['best']['false_merge']*100:.2f}%")

    print(f"\n{'机位':6s} {'轨迹':>4s} {'base接受':>8s} {'cam接受':>8s} {'毛新增':>5s} {'毛丢失':>5s}")
    per = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for k in kte:
        cam = k[0].split("_")[0]
        b, c = int(base["acc"][k]), int(cand["acc"][k])
        per[cam][0] += 1
        per[cam][1] += b
        per[cam][2] += c
        per[cam][3] += int(c and not b)
        per[cam][4] += int(b and not c)
    for cam in sorted(per):
        n, b, c, gn, gl = per[cam]
        print(f"{cam:6s} {n:4d} {b:8d} {c:8d} {gn:5d} {gl:5d}")

    new = [k for k in kte if cand["acc"][k] and not base["acc"][k]]
    lost = [k for k in kte if base["acc"][k] and not cand["acc"][k]]
    print(f"\n新增 {len(new)} 条 / 丢失 {len(lost)} 条")
    agree = sum(1 for k in new if cand["ia"][k] == cand["ib"][k])
    print(f"  新增轨迹的 a/b 一致: {agree}/{len(new)} = {agree/max(1,len(new))*100:.0f}%")
    for tag, ks in (("新增", new), ("丢失", lost)):
        if not ks:
            continue
        o = sorted(obs.get(k, 0) for k in ks)
        med = o[len(o) // 2]
        print(f"  {tag}: 观测数 中位 {med}  最小 {o[0]}  最大 {o[-1]}  "
              f"<20 观测占 {sum(1 for v in o if v < 20)/len(o)*100:.0f}%")
        cams = collections.Counter(k[0].split('_')[0] for k in ks)
        print(f"       机位分布 {dict(cams)}")

    # 新增轨迹被分到哪个簇：簇里有哪些机位、多少条 train 轨迹
    print("\n新增轨迹的落点（簇内机位构成）：")
    comp = collections.Counter()
    for k in new:
        c = cand["ia"][k]
        comp[tuple(cand["clu_cams"][c])] += 1
    for cc, n in comp.most_common():
        print(f"  {n:3d} 条 → 簇机位 {list(cc)}")
    sizes = sorted(cand["clu_size"][cand["ia"][k]] for k in new)
    print(f"  落点簇的 train 轨迹数：中位 {sizes[len(sizes)//2]}  最小 {sizes[0]}  最大 {sizes[-1]}")

    out = {"base": {"thr": base["thr"], **{k: v for k, v in base["best"].items()}, "K": base["K"]},
           "cand": {"thr": cand["thr"], **{k: v for k, v in cand["best"].items()}, "K": cand["K"]},
           "per_cam": {c: {"n": v[0], "base": v[1], "cand": v[2]} for c, v in per.items()},
           "new": [{"key": k, "obs": obs.get(k, 0)} for k in new],
           "lost": [{"key": k, "obs": obs.get(k, 0)} for k in lost]}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
