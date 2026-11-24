#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1012 test（下午段）上的选择器评估——时间隔离、无标签口径。

背景/纪律：
  1012 train 段（08:03-09:35）与 synced-10min（08:55-09:05）时间重叠，
  因此 **不得** 用 1012 训练的模型在 synced-10min 上评估（frozen_eval_v1 / splits_v5 均属该时段）。
  本脚本只用 1012 test 段（14:22 / 15:03 起，各 15min）做评估，时间上与 train 隔离。

1012 test 无人工身份标注，故只输出**无标签代理指标**（不冒充精度）：
  ① 分片一致性 same：同一条轨迹的前半 _a 与后半 _b 是否落到同一身份簇（时序 ID 稳定性）
  ② 共现误合并率 coll：同帧同时出现（≥cooc-min 帧）的两条轨迹被判为同一簇的比例（假合并下界代理）
  ③ 未知率 unknown：开放集行为 + 域漂移的校准指示
  ④ 簇使用分布：跨机位覆盖了多少身份簇、是否塌陷到少数簇
对照：train 段同一套指标（cluster_report.json 里的 same/coll），衡量 train→test 的漂移。

用法：
  python tools/eval_selector_1012test.py \
    --selector output_crosscam_local/selector_osnet_1012/selector.npz \
    --emb-dir output_crosscam_local/1012_osnet_emb \
    --segs 305_test_544,318_test_534,320_test_518,323_test_527 \
    --train-report output_crosscam_local/selector_osnet_1012/cluster_report.json \
    --out output_crosscam_local/selector_osnet_1012/test_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector, _norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tail_digits(s):
    num = ""
    for ch in reversed(s):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_pair(emb_dir, seg):
    za = np.load(os.path.join(emb_dir, f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(emb_dir, f"{seg}_emb_b.npz"))
    keys, va, vb = [], [], []
    for k in za.files:
        if k not in zb.files:
            continue
        keys.append((seg, tail_digits(k)))
        va.append(za[k])
        vb.append(zb[k])
    if not keys:
        return keys, np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32)
    return keys, np.asarray(va, np.float32), np.asarray(vb, np.float32)


def cooc_pairs(jsonl_path, keys_set, seg, min_common=5):
    """同帧共现 ≥min_common 的轨迹对（同时在场 = 铁定不同人）。"""
    if not os.path.exists(jsonl_path):
        return []
    cnt = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ts = sorted(set(int(t) for t in (r.get("track_ids") or [])
                            if t is not None and t >= 0 and (seg, int(t)) in keys_set))
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    cnt[(ts[i], ts[j])] = cnt.get((ts[i], ts[j]), 0) + 1
    return [k for k, v in cnt.items() if v >= min_common]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--segs", required=True)
    ap.add_argument("--jsonl-dir", default=os.path.join(ROOT, "output_crosscam", "1012_final"))
    ap.add_argument("--cooc-min", type=int, default=5)
    ap.add_argument("--train-report", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sel = IdentitySelector.load(a.selector)
    segs = [s for s in a.segs.split(",") if s]

    keys, Va, Vb, seg_of = [], [], [], []
    for s in segs:
        ks, va, vb = load_pair(a.emb_dir, s)
        keys += ks
        Va.append(va)
        Vb.append(vb)
        seg_of += [s] * len(ks)
        print(f"[load] {s}: {len(ks)} tracks")
    Va = np.concatenate(Va)
    Vb = np.concatenate(Vb)
    cams = [s.split("_")[0] for s in seg_of]
    N = len(keys)
    print(f"[test] {N} tracks, K={sel.n_clusters} clusters, thr={sel.thr_unknown:.3f}")

    rep = {
        "n_test_tracks": N,
        "n_clusters": int(sel.n_clusters),
        "thr_unknown": float(sel.thr_unknown),
        "thr_loose": (float(sel.thr_loose) if sel.thr_loose is not None else None),
        "segs": segs,
    }

    # ---- ① 分片一致性 + ③ 未知率 ----
    def profile(loose):
        ia, sa = sel.predict(Va, cams=cams, loose=loose)
        ib, sb = sel.predict(Vb, cams=cams, loose=loose)
        ia = np.asarray(ia) if np.ndim(ia) else np.array([ia])
        ib = np.asarray(ib) if np.ndim(ib) else np.array([ib])
        acc_a = ia >= 0
        acc_b = ib >= 0
        both = acc_a & acc_b
        same_cond = float((ia[both] == ib[both]).mean()) if both.any() else None
        # 无条件口径（两侧都接受才算一致，另一侧 unknown 视为不一致）——更保守
        same_strict = float(((ia == ib) & both).sum() / max(1, N))
        per_cam = {}
        for cam in sorted(set(cams)):
            m = np.array([c == cam for c in cams])
            per_cam[cam] = {
                "n": int(m.sum()),
                "accept_a": float(acc_a[m].mean()),
                "accept_b": float(acc_b[m].mean()),
                "same_cond": (float((ia[m & both] == ib[m & both]).mean())
                              if (m & both).any() else None),
            }
        return {
            "loose": bool(loose),
            "accept_rate_a": float(acc_a.mean()),
            "accept_rate_b": float(acc_b.mean()),
            "accept_rate_both": float(both.mean()),
            "same_cond": same_cond,
            "same_strict": same_strict,
            "ids_a": ia, "ids_b": ib, "per_cam": per_cam,
        }

    p_cons = profile(False)
    rep["conservative"] = {k: v for k, v in p_cons.items() if not k.startswith("ids_")}
    if sel.thr_loose is not None:
        p_loose = profile(True)
        rep["loose"] = {k: v for k, v in p_loose.items() if not k.startswith("ids_")}
    else:
        p_loose = p_cons

    # ---- ② 共现误合并率 ----
    keys_set = set(keys)
    kidx = {k: i for i, k in enumerate(keys)}
    all_cooc = []
    for s in segs:
        for u, v in cooc_pairs(os.path.join(a.jsonl_dir, f"{s}.jsonl"), keys_set, s, a.cooc_min):
            all_cooc.append(((s, u), (s, v)))

    def merge_rate(p):
        ia, ib = p["ids_a"], p["ids_b"]
        pair_total = 0
        pair_both = 0
        merge = 0
        for u, v in all_cooc:
            iu, iv = kidx.get(u), kidx.get(v)
            if iu is None or iv is None:
                continue
            pair_total += 1
            # 交叉分片双向：a_u vs b_v、a_v vs b_u
            ok = (ia[iu] >= 0 and ib[iv] >= 0) or (ia[iv] >= 0 and ib[iu] >= 0)
            if not ok:
                continue
            pair_both += 1
            if (ia[iu] >= 0 and ib[iv] >= 0 and ia[iu] == ib[iv]) or \
               (ia[iv] >= 0 and ib[iu] >= 0 and ia[iv] == ib[iu]):
                merge += 1
        return {
            "n_cooc_pairs": pair_total,
            "n_evaluable": pair_both,
            "false_merge_rate": (float(merge / pair_both) if pair_both else None),
            "evaluable_frac": (float(pair_both / pair_total) if pair_total else None),
        }

    rep["conservative"]["cooc"] = merge_rate(p_cons)
    if sel.thr_loose is not None:
        rep["loose"]["cooc"] = merge_rate(p_loose)

    # ---- ④ 簇使用分布（塌陷检查） ----
    ia = p_cons["ids_a"]
    used = ia[ia >= 0]
    rep["cluster_usage"] = {
        "n_used": int(len(set(used.tolist()))) if len(used) else 0,
        "top1_share": (float(np.bincount(used).max() / len(used)) if len(used) else None),
        "n_used_over_clusters": (float(len(set(used.tolist())) / max(1, sel.n_clusters))
                                 if len(used) else 0.0),
    }
    # 跨机位复用：同一簇在几路出现
    cam_of = {}
    for i, c in enumerate(cams):
        if ia[i] >= 0:
            cam_of.setdefault(int(ia[i]), set()).add(c)
    multi = [c for c, s in cam_of.items() if len(s) >= 2]
    rep["cross_cam_cluster"] = {
        "n_multi_cam_clusters": len(multi),
        "frac_of_used": (float(len(multi) / max(1, len(cam_of))) if cam_of else None),
    }

    # ---- 对照 train ----
    if a.train_report and os.path.exists(a.train_report):
        tr = json.load(open(a.train_report, encoding="utf-8"))
        best = tr.get("best", {})
        rep["train_reference"] = {
            "n_train_tracks": tr.get("n_train_tracks"),
            "split_half_auc": tr.get("split_half", {}).get("auc"),
            "same": best.get("same"),
            "coll": best.get("coll"),
            "f1_cooc": best.get("f1"),
            "n_clusters": best.get("n_clusters"),
            "method": best.get("method"),
            "thr_unknown": tr.get("thr_calib", {}).get("thr_unknown"),
        }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 打印 ----
    print("\n== 1012 test（时间隔离）==")
    for tag in ("conservative", "loose"):
        if tag not in rep:
            continue
        p = rep[tag]
        print(f"[{tag}] accept a={p['accept_rate_a']:.3f} b={p['accept_rate_b']:.3f} "
              f"both={p['accept_rate_both']:.3f} | same_cond="
              f"{'n/a' if p['same_cond'] is None else round(p['same_cond'], 3)} "
              f"same_strict={p['same_strict']:.3f}")
        c = p["cooc"]
        print(f"          cooc pairs={c['n_cooc_pairs']} evaluable={c['n_evaluable']} "
              f"false_merge={'n/a' if c['false_merge_rate'] is None else round(c['false_merge_rate'], 4)}")
        for cam, v in p["per_cam"].items():
            print(f"          cam {cam}: n={v['n']} accept={v['accept_a']:.2f}/{v['accept_b']:.2f} "
                  f"same={'n/a' if v['same_cond'] is None else round(v['same_cond'], 3)}")
    print(f"[usage] clusters used={rep['cluster_usage']['n_used']}/{sel.n_clusters} "
          f"top1_share={rep['cluster_usage']['top1_share']}")
    print(f"[cross-cam] multi-cam clusters={rep['cross_cam_cluster']['n_multi_cam_clusters']}")
    if "train_reference" in rep:
        r = rep["train_reference"]
        print(f"[train ref] same={r['same']} coll={r['coll']} F1={r['f1_cooc']} K={r['n_clusters']}")
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
