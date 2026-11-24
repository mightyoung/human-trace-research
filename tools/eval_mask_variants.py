#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""掩膜变体终局评估：基线 / maskfill / maskbbox 三组表征走同一判定管线。

判定管线与 eval_two_encodings.run() 完全一致（会话对齐 + 每身份 z 校准 +
self-cal 定 θ，覆盖优先、a/b 一致率 >=0.90），因此数字与 10-17/10-18 历史结果同口径。

关键控制：maskbbox 会因掩膜外接框过小（<24x48）丢掉一部分小目标轨迹，直接比
"各自全集"会把"丢难样本"混进收益里。故每种变体跑两遍：
  full   —— 该变体自己的全部轨迹（部署视角）
  common —— 三个变体共有的轨迹交集（表征质量视角，唯一可公平比较的口径）

用法：python tools/eval_mask_variants.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm  # noqa: E402
import session_selector_v1 as S  # noqa: E402
from eval_selector_1012test import cooc_pairs  # noqa: E402

ROOT = S.ROOT
LOCAL = os.path.join(ROOT, "output_crosscam_local")
VARIANTS = [
    ("baseline", os.path.join(LOCAL, "1012_osnet_emb")),
    ("maskfill", os.path.join(LOCAL, "1012_osnet_emb_maskfill")),
    ("maskbbox", os.path.join(LOCAL, "1012_osnet_emb_maskbbox")),
]


def load_pair(emb_dir, seg, allowed=None):
    za = np.load(os.path.join(emb_dir, f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(emb_dir, f"{seg}_emb_b.npz"))
    keys, va, vb = [], [], []
    for k in za.files:
        if k not in zb.files:
            continue
        key = (seg, int(str(k).rsplit("#", 1)[1]))
        if allowed is not None and key not in allowed:
            continue
        keys.append(key)
        va.append(za[k])
        vb.append(zb[k])
    return keys, np.asarray(va, np.float32), np.asarray(vb, np.float32)


def load_split(emb_dir, segs, allowed=None):
    keys, Va, Vb = [], [], []
    for s in segs:
        k, a, b = load_pair(emb_dir, s, allowed)
        keys += k
        if len(k):
            Va.append(a)
            Vb.append(b)
    if not Va:
        return [], np.zeros((0, 512), np.float32), np.zeros((0, 512), np.float32)
    return keys, np.concatenate(Va), np.concatenate(Vb)


def run(emb_dir, cooc_te, allowed=None):
    ktr, Va_tr, Vb_tr = load_split(emb_dir, S.TRAIN_SEGS, allowed)
    kte, Va_te, Vb_te = load_split(emb_dir, S.TEST_SEGS, allowed)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    lab, kept = S.hier_labels(Vtr)
    K = len(kept)
    mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        if m.sum() == 0:
            continue
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)

    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    grid = np.round(np.arange(-2.0, 8.01, 0.05), 3)
    keys_ok = set(kte)
    cooc = [(u, v) for u, v in cooc_te if u in keys_ok and v in keys_ok]
    rows = S.sweep(Sa, Sb, kte, cooc, grid)
    ok = [r for r in rows if np.isfinite(r["same"]) and r["same"] >= 0.90
          and r["evaluable"] >= 100]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    ref = {}
    for tgt in (0.30, 0.40, 0.50):
        cand = [r for r in rows if r["coverage"] >= tgt and r["evaluable"] >= 100
                and np.isfinite(r["false_merge"])]
        if cand:
            r = min(cand, key=lambda x: x["coverage"])
            ref[str(tgt)] = {"same": r["same"], "false_merge": r["false_merge"]}
    return {"n_train": len(ktr), "n_test": len(kte), "K": K, "n_cooc": len(cooc),
            "self_cal": best, "at_coverage": ref, "sweep": rows}


def gate_keys(min_used=0.5):
    """按"该轨迹多数采样帧能找到 YOLO 人体"筛轨迹（掩膜命中率代理真人率）。"""
    from collections import defaultdict
    allowed = set()
    for seg in S.TRAIN_SEGS + S.TEST_SEGS:
        p = os.path.join(LOCAL, "1012_osnet_crops_mask", f"{seg}_crops_maskfill.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=False)
        agg = defaultdict(list)
        for eid, used in zip(z["eids"].tolist(), z["q"][:, 2].tolist()):
            ab, tid, _k = str(eid).split("#")
            agg[(ab, int(tid))].append(float(used))
        for (_ab, tid), v in agg.items():
            if float(np.mean(v)) >= min_used:
                allowed.add((seg, tid))
    return allowed


def keyset(emb_dir):
    ks = set()
    for segs in (S.TRAIN_SEGS, S.TEST_SEGS):
        for s in segs:
            k, _a, _b = load_pair(emb_dir, s)
            ks |= set(k)
    return ks


def main():
    kte_guess = S.load(S.TEST_SEGS)[0]
    cooc_te = []
    for s in S.TEST_SEGS:
        for u, v in cooc_pairs(os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"),
                               set(kte_guess), s, 5):
            cooc_te.append(((s, u), (s, v)))
    print(f"[cooc] {len(cooc_te)} pairs", flush=True)

    avail = [(t, d) for t, d in VARIANTS if os.path.isdir(d)]
    sets = {}
    for t, d in avail:
        sets[t] = keyset(d)
        print(f"[keys] {t}: {len(sets[t])}", flush=True)
    common = set.intersection(*sets.values())
    print(f"[keys] common: {len(common)}", flush=True)

    gate = gate_keys(0.5)
    gate_test = {k for k in gate if k[0] in S.TEST_SEGS}
    print(f"[keys] gate>=0.5 used: {len(gate)} (test {len(gate_test)})", flush=True)

    out = {"cooc_pairs": len(cooc_te), "n_common": len(common),
           "n_gate": len(gate), "n_gate_test": len(gate_test), "runs": {}}
    scopes = [("full", None), ("common", common), ("gate", gate)]
    for scope, allowed in scopes:
        for tag, d in avail:
            if scope == "gate" and tag != "baseline":
                continue
            r = run(d, cooc_te, allowed)
            out["runs"][f"{tag}@{scope}"] = r
            b = r["self_cal"] or {}
            print(f"[{tag:9s}@{scope:6s}] tr={r['n_train']} te={r['n_test']} K={r['K']} "
                  f"pairs={r['n_cooc']} | cov={b.get('coverage', float('nan')):.3f} "
                  f"same={b.get('same', float('nan')):.3f} "
                  f"fm={b.get('false_merge', float('nan'))*100:.2f}%", flush=True)
            for t, v in r["at_coverage"].items():
                print(f"                 @cov>={t}: same={v['same']:.3f} "
                      f"fm={v['false_merge']*100:.2f}%", flush=True)

    dst = os.path.join(LOCAL, "selector_osnet_1012/eval_mask_variants.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {dst}")


if __name__ == "__main__":
    main()
