#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图数据集 × 聚类选择器关联：为每条轨迹/样本附上身份簇分配（pid）。

用途：验证"意图预测器与向量聚类结合"（身份条件化意图预测）。
- 输入：analysis/intent/1012_{split}.npz（meta 列 = [seg, cam, tid, frame]）
- 嵌入：output_crosscam/analysis/cradio_1012_emb/{seg}_emb_a/b.npz（键 {seg}#{tid}）
- 选择器：output_crosscam/analysis/selector_1012/selector.npz（125 原型）
- 输出：analysis/intent/1012_{split}_cc.npz（原字段 + pid [N]int32, -1=unknown + cam [N]str）

口径说明：train/val/test 一律用**同一个已交付选择器**分配（模拟部署时的选择器输出）。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector import IdentitySelector, _norm  # noqa: E402

ROOT = os.path.dirname(HERE)
EMB = os.path.join(ROOT, "output_crosscam", "analysis", "cradio_1012_emb")
SEL = os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012", "selector.npz")
INT = os.path.join(ROOT, "output_crosscam", "analysis", "intent")


def main():
    sel = IdentitySelector.load(SEL)
    print(f"[selector] protos={sel.protos.shape[0]} thr_unknown={sel.thr_unknown:.4f}")
    stats = {}
    for split in ("train", "val", "test"):
        d = np.load(os.path.join(INT, f"1012_{split}.npz"), allow_pickle=True)
        meta, X, Xf, y = d["meta"], d["X"], d["Xf"], d["y"]
        segs = sorted(set(str(m[0]) for m in meta))
        # 预载嵌入（缓存）
        cache = {}
        for s in segs:
            pa = os.path.join(EMB, f"{s}_emb_a.npz")
            pb = os.path.join(EMB, f"{s}_emb_b.npz")
            za = np.load(pa) if os.path.exists(pa) else None
            zb = np.load(pb) if os.path.exists(pb) else None
            cache[s] = (za, zb)
        # 唯一轨迹 → V
        tks, seen = [], set()
        for m in meta:
            key = (str(m[0]), int(m[2]))
            if key not in seen:
                seen.add(key)
                tks.append(key)
        Vs, missing = [], []
        for seg, tid in tks:
            za, zb = cache[seg]
            k = f"{seg}#{tid}"
            if za is not None and zb is not None and k in za.files and k in zb.files:
                Vs.append(_norm(((za[k] + zb[k]) / 2.0)[None, :])[0])
            else:
                Vs.append(None)
                missing.append(k)
        Vm = np.asarray([v for v in Vs if v is not None], dtype=np.float32)
        ids, sims = sel.predict(Vm) if len(Vm) else (np.array([]), np.array([]))
        pid_of, j = {}, 0
        for key, v in zip(tks, Vs):
            if v is None:
                pid_of[key] = -1
            else:
                pid_of[key] = int(ids[j])
                j += 1
        N = len(y)
        pid = np.array([pid_of.get((str(m[0]), int(m[2])), -1) for m in meta], dtype=np.int32)
        cam = np.array([str(m[1]) for m in meta])
        out = os.path.join(INT, f"1012_{split}_cc.npz")
        np.savez(out, X=X, Xf=Xf, y=y, meta=meta, pid=pid, cam=cam)
        n_known = int((pid >= 0).mean() * N)
        st = {"n_samples": int(N), "n_tracks": len(tks), "n_missing_emb": len(missing),
              "known_sample_ratio": float((pid >= 0).mean()),
              "n_unique_pid": int(len(set(pid[pid >= 0].tolist())))}
        stats[split] = st
        print(f"[{split}] samples={N} tracks={len(tks)} missing_emb={len(missing)} "
              f"known={st['known_sample_ratio']:.3f} unique_pid={st['n_unique_pid']}")
        if missing[:3]:
            print(f"    missing examples: {missing[:3]}")
    json.dump(stats, open(os.path.join(INT, "1012_cc_link_stats.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[done]")


if __name__ == "__main__":
    main()
