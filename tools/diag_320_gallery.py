#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""320 图库段替换诊断：不动生产图库，只做离线 A/B。

问题：生产图库的 320 段取自 514 卷（08:03-09:03，90.5% 空帧、51 条轨迹），
      而 320 的 query 段（518，15:03）有 40 条轨迹。若换成 540 卷（10:22 起，有人），
      跨机位检索能否改善？

做法：同一批 294 条已标真值 query 不变，只换图库组合，比较
  1) 生产口径 top-1 身份命中率（含 camera-aware 聚类 + z 校准，与 76.8% 可比）
  2) 纯余弦 top-1 命中率（不经 z 校准，不受 K 变化影响）
  3) 320 机位分组的命中率（该段直接影响的目标）
  4) top-1 分数分布（查 z 校准漂移）

用法：
  PY tools/diag_320_gallery.py --emb-dir output_crosscam_local/1012_yolo8_dd_emb_diag
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm                      # noqa: E402
import session_selector_v1 as S                 # noqa: E402
import eval_mask_variants as M                  # noqa: E402
import cluster_camera_aware as CA               # noqa: E402
from case_control_modal import load_gt           # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_variant(emb_dir, train_segs, thr_in=0.15, thr_out=0.25, veto=0.30):
    ktr, Va, Vb = M.load_split(emb_dir, train_segs)
    kte, Qa, Qb = M.load_split(emb_dir, S.TEST_SEGS)
    Vtr = _norm((Va + Vb) / 2.0)
    cams = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams, thr_in, thr_out, S.MIN_CLUSTER)

    drop = []
    if veto is not None:
        for c in kept:
            idx = np.where(lab == c)[0]
            if idx.size >= 2:
                Xc = Vtr[idx]
                dmax = float(1.0 - (Xc @ Xc.T)[np.triu_indices(idx.size, 1)].min())
            else:
                dmax = 0.0
            if dmax >= veto:
                drop.append(c)
    kept = [c for c in kept if c not in set(drop)]

    mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        if not m.sum():
            continue
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)          # 必须是列下标，cluster_scores 用 pcl == k 填列
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    K = len(kept)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)

    Qv = _norm((Qa + Qb) / 2.0)
    mu_q = Qv.mean(0)
    Sa = S.score_rows(Qa, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Qb, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    # 纯余弦（不做中心化/z 校准）：Vq 与 未中心化质心 的余弦
    protos_raw = []
    for c in kept:
        m = lab == c
        if m.sum():
            protos_raw.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
    protos_raw = np.asarray(protos_raw, np.float32)
    Craw = Qv @ protos_raw.T
    return dict(ktr=ktr, kte=kte, lab=lab, kept=kept, pcl=pcl, Sa=Sa, Sb=Sb,
                Craw=Craw, n_tr=len(ktr), K=K, dropped=len(drop))


def evaluate(res, gal, qry, label, verbose=True):
    # 键必须是 protos 下标（= kept 顺序），不是簇标签；score_rows 的列按 protos 顺序
    cid_pid = {}
    for pi, c in enumerate(res["kept"]):
        votes = collections.Counter(gal[k] for k in
                                    [res["ktr"][i] for i in np.where(res["lab"] == c)[0]] if k in gal)
        if votes:
            cid_pid[pi] = votes.most_common(1)[0][0]

    rows = []
    for i, k in enumerate(res["kte"]):
        pid, ev = qry.get(k, ("", False))
        if not ev or not pid:
            continue
        a, b = int(np.argmax(res["Sa"][i])), int(np.argmax(res["Sb"][i]))
        pa, pb = cid_pid.get(a, ""), cid_pid.get(b, "")
        prod_ok = int(pa == pid and pb == pid)
        prod_any = int(pa == pid or pb == pid)
        ra = int(np.argmax(res["Craw"][i]))
        raw_pid = cid_pid.get(ra, "")
        rows.append(dict(key=k, seg=k[0], cam=k[0].split("_")[0], gt=pid,
                         prod_ok=prod_ok, prod_any=prod_any,
                         raw_ok=int(raw_pid == pid), raw_pid=raw_pid,
                         top1_score=float((res["Sa"][i] + res["Sb"][i]).max() / 2),
                         named=bool(pa and pb)))

    named = [r for r in rows if r["named"]]
    out = dict(label=label, K=res["K"], n_tr=res["n_tr"], dropped=res["dropped"],
               n_query=len(rows), n_named=len(named),
               prod_hit=sum(r["prod_ok"] for r in named), 
               prod_any=sum(r["prod_any"] for r in named),
               raw_hit=sum(r["raw_ok"] for r in rows),
               score_med=float(np.median([r["top1_score"] for r in named])) if named else 0.0)
    out["prod_rate"] = out["prod_hit"] / max(1, len(named))
    out["raw_rate"] = out["raw_hit"] / max(1, len(rows))
    out["by_cam"] = {}
    for cam in ("305", "318", "320", "323"):
        sub = [r for r in named if r["cam"] == cam]
        subraw = [r for r in rows if r["cam"] == cam]
        out["by_cam"][cam] = dict(
            n=len(sub), prod_hit=sum(r["prod_ok"] for r in sub),
            prod_rate=(sum(r["prod_ok"] for r in sub) / len(sub)) if sub else 0.0,
            n_raw=len(subraw), raw_hit=sum(r["raw_ok"] for r in subraw),
            raw_rate=(sum(r["raw_ok"] for r in subraw) / len(subraw)) if subraw else 0.0)
    if verbose:
        print(f"[{label}] 图库轨迹 {res['n_tr']} | K={res['K']} | veto 弃 {res['dropped']}")
        print(f"  生产口径 top-1 命中 {out['prod_hit']}/{len(named)} = {out['prod_rate']*100:.1f}%"
              f"  (任一侧命中 {out['prod_any']})  分数中位 {out['score_med']:.3f}")
        print(f"  纯余弦  top-1 命中 {out['raw_hit']}/{len(rows)} = {out['raw_rate']*100:.1f}%")
        for cam, v in out["by_cam"].items():
            print(f"    {cam}: 生产 {v['prod_hit']}/{v['n']}={v['prod_rate']*100:.1f}%"
                  f" | 余弦 {v['raw_hit']}/{v['n_raw']}={v['raw_rate']*100:.1f}%")
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=os.path.join(ROOT, "output_crosscam_local/1012_yolo8_dd_emb_diag"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "deliverable/data-request-v2/gt"))
    ap.add_argument("--new-seg", default="320_c540")
    ap.add_argument("--out", default=os.path.join(ROOT, "deliverable/data-request-v2/results/diag_320_gallery.json"))
    a = ap.parse_args()

    gal, qry = load_gt(a.gt)
    print(f"[gt] 图库已标 {len(gal)} / 查询已标 {sum(1 for v in qry.values() if v[0])}")

    base = list(S.TRAIN_SEGS)
    variants = [
        ("base_noveto", base, None),                        # 对上扩图库实验 76.8% 的口径（K=111）
        ("base", base, 0.30),                               # 生产配置
        ("add", base + [a.new_seg], 0.30),
        ("repl", [x if x != "320_train_514" else a.new_seg for x in base], 0.30),
        ("repl_noveto", [x if x != "320_train_514" else a.new_seg for x in base], None),
        ("repl_drop", [x for x in base if x != "320_train_514"], 0.30),
    ]
    results, all_rows = [], {}
    for label, segs, veto in variants:
        try:
            res = build_variant(a.emb_dir, segs, veto=veto)
        except Exception as e:
            print(f"[{label}] 跳过：{e}")
            continue
        o, rows = evaluate(res, gal, qry, label)
        results.append(o)
        all_rows[label] = rows

    print("\n" + "=" * 78)
    print(f"{'variant':<12}{'图库轨迹':>8}{'K':>5}{'生产命中':>10}{'生产%':>8}{'余弦命中':>10}{'余弦%':>8}")
    print("-" * 78)
    for o in results:
        print(f"{o['label']:<12}{o['n_tr']:>8}{o['K']:>5}"
              f"{str(o['prod_hit'])+'/'+str(o['n_named']):>10}{o['prod_rate']*100:>7.1f}%"
              f"{str(o['raw_hit'])+'/'+str(o['n_query']):>10}{o['raw_rate']*100:>7.1f}%")

    # ---- 固定分母：各变体都「两侧簇有身份」的共同 query 子集 ----
    if len(all_rows) >= 2:
        sets = [set(r["key"] for r in rows if r["named"]) for rows in all_rows.values()]
        common = set.intersection(*sets)
        print(f"\n=== 固定分母比较（共同子集 n={len(common)}）===")
        print(f"{'variant':<14}{'生产命中':>10}{'生产%':>9}{'320子集':>10}{'320%':>8}{'余弦命中':>10}{'余弦%':>8}")
        print("-" * 74)
        fixed = {}
        for label, rows in all_rows.items():
            idx = {r["key"]: r for r in rows}
            sub = [idx[k] for k in common if k in idx]
            c320 = [r for r in sub if r["cam"] == "320"]
            a320 = [idx[k] for k in common if k in idx and idx[k]["cam"] == "320"]
            h = sum(r["prod_ok"] for r in sub)
            h320 = sum(r["prod_ok"] for r in a320)
            rh = sum(r["raw_ok"] for r in sub)
            print(f"{label:<14}{str(h)+'/'+str(len(sub)):>10}{h/max(1,len(sub))*100:>8.1f}%"
                  f"{str(h320)+'/'+str(len(a320)):>10}{(h320/max(1,len(a320)))*100:>7.1f}%"
                  f"{str(rh)+'/'+str(len(sub)):>10}{rh/max(1,len(sub))*100:>7.1f}%")
            fixed[label] = dict(n=len(sub), prod_hit=h, prod_rate=h / max(1, len(sub)),
                                n320=len(a320), hit320=h320, raw_hit=rh)
    else:
        fixed = {}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": results, "fixed_denominator": fixed,
               "detail": {k: [{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in r.items()}
                              for r in v] for k, v in all_rows.items()}},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[DONE] -> {a.out}")


if __name__ == "__main__":
    main()
