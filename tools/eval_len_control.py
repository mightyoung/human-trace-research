#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长度对齐敏感性：按"每条轨迹的观测帧数 >= N"过滤后再走同一条判定管线。

用途：检测器横比时，stride 不同会让轨迹长短不同，进而改变图库簇的粗细，
      导致"覆盖上涨"可能只是图库变粗的副作用。本脚本只加一个观测数下限过滤，
      指标定义、扫描网格、self-cal 规则全部沿用 eval_det_switch.run()。

用法：
  python tools/eval_len_control.py --variant baseline=output_crosscam_local/1012_osnet_emb \
      --variant yolo26s=output_crosscam_local/1012_yolo8_emb \
      --ab-map yolo26s=output_crosscam_local/1012_yolo8_ab \
      --cooc-map yolo26s=output_crosscam_local/1012_yolo8 \
      --min-obs 20 --min-eval 30
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import session_selector_v1 as S                       # noqa: E402
from eval_selector_1012test import cooc_pairs          # noqa: E402
from eval_mask_variants import load_pair               # noqa: E402
import eval_det_switch as D                            # noqa: E402

ROOT = S.ROOT
LOCAL = os.path.join(ROOT, "output_crosscam_local")
DEFAULT_AB = os.path.join(LOCAL, "1012_ab")


def obs_counts(ab_dir, segs):
    """每轨迹在 a/b 两半里的观测帧数合计：{(seg, tid): n}"""
    out = {}
    for s in segs:
        for half in ("a", "b"):
            p = os.path.join(ab_dir, f"{s}_{half}.jsonl")
            if not os.path.exists(p):
                continue
            for line in open(p, encoding="utf-8"):
                if not line.strip():
                    continue
                d = json.loads(line)
                for i in d.get("track_ids") or []:
                    k = (s, int(i))
                    out[k] = out.get(k, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True, help="NAME=EMB_DIR")
    ap.add_argument("--ab-map", action="append", default=[], help="NAME=AB_DIR（缺省=1012_ab）")
    ap.add_argument("--cooc-map", action="append", default=[], help="NAME=DET_DIR")
    ap.add_argument("--segs", default="")
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--same-min", type=float, default=0.90)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    variants = [tuple(spec.split("=", 1)) for spec in a.variant]
    want = [s for s in a.segs.split(",") if s] or (S.TRAIN_SEGS + S.TEST_SEGS)
    segs_tr = [s for s in want if s in S.TRAIN_SEGS]
    segs_te = [s for s in want if s in S.TEST_SEGS]
    abmap = dict(spec.split("=", 1) for spec in a.ab_map)
    cmap = dict(spec.split("=", 1) for spec in a.cooc_map)
    all_segs = segs_tr + segs_te

    res = {}
    print(f"[len-control] min_obs={a.min_obs} min_eval={a.min_eval} same_min={a.same_min}")
    for name, d in variants:
        ab = abmap.get(name, DEFAULT_AB)
        obs = obs_counts(ab, all_segs)
        allowed = {k for k, n in obs.items() if n >= a.min_obs}
        keys_te = set()
        for s in segs_te:
            k, _va, _vb = load_pair(d, s, allowed)
            keys_te |= set(k)
        cooc = []
        for s in segs_te:
            src = cmap.get(name)
            p = (os.path.join(src, s, f"{s}_detections.jsonl") if src
                 else os.path.join(ROOT, f"output_crosscam/1012_final/{s}.jsonl"))
            if os.path.exists(p):
                cooc += [((s, u), (s, v)) for u, v in cooc_pairs(p, keys_te, s, 5)]
        r = D.run(d, segs_tr, segs_te, cooc, allowed, a.min_eval, a.same_min)
        if r is None:
            print(f"[{name}] 过滤后无可用轨迹")
            continue
        b = r["self_cal"] or {}
        print(f"[{name:9s}] tr={r['n_train']:4d} te={r['n_test']:3d} K={r['K']:3d} "
              f"pairs={r['n_cooc']:5d} | cov={b.get('coverage', float('nan')):.3f} "
              f"same={b.get('same', float('nan')):.3f} "
              f"fm={b.get('false_merge', float('nan'))*100:.2f}% "
              f"eval={b.get('evaluable', 0)} thr={b.get('thr')}")
        res[name] = r
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"min_obs": a.min_obs, "runs": res}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
