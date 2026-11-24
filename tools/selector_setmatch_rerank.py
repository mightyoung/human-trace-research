#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SetMatcher 作为**重排序器**的评估：候选集固定为余弦 top-8，只在候选中选一个。

为什么单独评：
  上一轮（study_v1）让模型在 125 个候选上端到端判定，coll 从 3.2% 升到 12%——
  联合判别在小样本上会塌向大簇。真正合理的用法是**不改召回，只改排序**：
  余弦先取 top-8，模型在这 8 个里重排。这样开放集问题退化为闭合集比较，
  错误上界被余弦候选集锁住。

对比口径（全判定，无弃权）：
  same_known = 同轨迹 a/b 两视图选出同一簇的比例（越高越好）
  另给"疑难子集"（余弦 top1-top2 间隔最小的 40%）上的 same，
  这才是重排序器真正要赢的地方。

用法：
  python tools/selector_setmatch_rerank.py --tag v1 --topk 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from camgroup_study import load_segs, TRAIN_SEGS  # noqa: E402
from selector_upgrade_study import build_variants, score_matrix  # noqa: E402
from selector_setmatch_study import (SetMatcher, cand_tokens, base_clusters,  # noqa: E402
                                     OUT, VAL_SEGS, TEST_SEGS, MAX_SUB)

ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    a = ap.parse_args()

    TR_keys, TR_Va, TR_Vb, TR_V, lab, keep_ids, _ = base_clusters(OUT)
    variants, C = build_variants(TR_V, lab, keep_ids)
    var = variants["multi8"]
    tok_t, msk_t = cand_tokens(var, keep_ids)
    K = len(keep_ids)

    model = SetMatcher(d=a.dim, layers=a.layers, heads=a.heads)
    model.load_state_dict(torch.load(os.path.join(OUT, f"model_{a.tag}.pt"), map_location="cpu"))
    model.eval()
    opt_emb = model.options(torch.from_numpy(tok_t))
    mt = torch.from_numpy(msk_t)

    TE_keys, TE_Va, TE_Vb, TE_V = load_segs(TEST_SEGS)
    sc_v = score_matrix(TE_V, var, C)          # 用均值向量选候选（与两视图无关，避免泄漏）
    sc_a = score_matrix(TE_Va, var, C)
    sc_b = score_matrix(TE_Vb, var, C)

    top = np.argsort(-sc_v, axis=1)[:, :a.topk]          # [N,k] 候选簇 id
    margin = np.sort(sc_v, axis=1)[:, -1] - np.sort(sc_v, axis=1)[:, -2]
    amb = margin <= np.quantile(margin, 0.4)             # 疑难 40%

    def model_scores(V, cand):
        """V:[N,D] cand:[N,k] → [N,k] 模型概率（softmax over k+1，取前 k）"""
        out = np.zeros((len(V), a.topk), np.float32)
        with torch.no_grad():
            for s in range(0, len(V), 8):
                q = V[s:s + 8]
                cd = torch.from_numpy(cand[s:s + 8])                 # [b,k]
                emb = opt_emb[cd]                                     # [b,k,S,d]
                mask = mt[cd]                                         # [b,k,S]
                logits = model(model.query(torch.from_numpy(q)), emb, mask)
                out[s:s + 8] = F.softmax(logits, -1)[:, :a.topk].numpy()
        return out

    p_a = model_scores(TE_Va, top)
    p_b = model_scores(TE_Vb, top)

    def pick(scores, cand):
        return cand[np.arange(len(cand)), scores.argmax(1)]

    cos_a = pick(np.take_along_axis(sc_a, top, 1), top)
    cos_b = pick(np.take_along_axis(sc_b, top, 1), top)
    mod_a, mod_b = pick(p_a, top), pick(p_b, top)

    def rep(pred_a, pred_b, subset=None):
        m = np.ones(len(pred_a), bool) if subset is None else subset
        pa, pb = pred_a[m], pred_b[m]
        rng = np.random.default_rng(7)
        L, R = rng.integers(0, len(pa), 40000), rng.integers(0, len(pa), 40000)
        kp = L != R
        # 塌缩诊断：唯一预测数 / 最大簇占比 / 随机对同簇率
        vals, cnts = np.unique(pa, return_counts=True)
        # Cohen's kappa：扣掉"两边都偏向大簇"带来的偶然一致（塌缩会抬高 same，也会抬高 p_e）
        ka = {v: c / len(pa) for v, c in zip(vals, cnts)}
        vb, cb = np.unique(pb, return_counts=True)
        kb = {v: c / len(pb) for v, c in zip(vb, cb)}
        pe = sum(ka.get(k, 0.0) * kb.get(k, 0.0) for k in set(ka) | set(kb))
        same = float((pa == pb).mean())
        return {"n": int(m.sum()),
                "same": round(same, 4),
                "kappa": round((same - pe) / max(1e-9, 1 - pe), 4),
                "n_unique": int(len(vals)),
                "top_share": round(float(cnts.max() / len(pa)), 4),
                "pair_same": round(float((pa[L[kp]] == pa[R[kp]]).mean()), 4)}

    def boot(pred_a, pred_b, pred_c, pred_d, subset, n=2000, seed=11):
        """cos vs setmatch 的配对 bootstrap：delta = same(model) - same(cos)。"""
        rng = np.random.default_rng(seed)
        idx = np.where(subset)[0]
        d = []
        for _ in range(n):
            b = rng.choice(idx, len(idx), replace=True)
            d.append((pred_c[b] == pred_d[b]).mean() - (pred_a[b] == pred_b[b]).mean())
        d = np.asarray(d)
        return {"delta": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(d, 2.5)), 4),
                         round(float(np.percentile(d, 97.5)), 4)],
                "p_gt0": round(float((d > 0).mean()), 4)}

    allm = np.ones(len(cos_a), bool)
    res = {"tag": a.tag, "topk": a.topk,
           "boot_all": boot(cos_a, cos_b, mod_a, mod_b, allm),
           "boot_amb40": boot(cos_a, cos_b, mod_a, mod_b, amb),
           "all": {"cos": rep(cos_a, cos_b), "setmatch": rep(mod_a, mod_b)},
           "ambiguous40": {"cos": rep(cos_a, cos_b, amb), "setmatch": rep(mod_a, mod_b, amb)},
           "agree_model_vs_cos": round(float((mod_a == cos_a).mean()), 4)}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    path = os.path.join(OUT, f"rerank_{a.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
