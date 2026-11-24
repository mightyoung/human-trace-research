#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用版簇级真值派生：给 validation/<dataset> 生成 VALIDATION §5 要求的 gt/clusters_gt.csv。

与 `derive_clusters_gt.py`（1012 专用，依赖 ledger/clusters_geom.csv）的区别：
本脚本只用 **members + 逐成员真值 + 表征** 就能算，因此对 MARS / CHIRLA 这类外部包同样适用。

列与现场同义（缺的列留空，不编造）：
  cluster_id,n_members,n_labelled,n_unlabelled,n_ids,n_nonhuman,main_pid,main_n,main_share,
  mixed_members,n_head,n_body,n_partial,n_idswitch,d_max,cams,old_verdict,agree
- n_ids = 簇内不同 person_id 数（空 / no-human 不计）
- n_nonhuman = person_id ∈ {no-human, shadow*} 的成员数（沿用现场口径：影子/非人单列）
- d_max = 簇内最大成对余弦距离（1 - cos），单成员簇记 0
- n_idswitch / n_head / n_body / n_partial / old_verdict / agree：外部数据没有 → 留空

用法：
  PY tools/derive_clusters_gt_generic.py --pkg validation/mars --emb validation/mars/emb
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_mask_variants as M  # noqa: E402

NON_HUMAN = {"no-human", "nohuman", "no_human", "none", "无", "非人"}
SHADOW_PREFIX = "shadow"


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="validation/<dataset> 目录（需含 gt/ 与 results/）")
    ap.add_argument("--emb", default="", help="表征目录（默认 <pkg>/emb）")
    ap.add_argument("--config", default="A_baseline")
    ap.add_argument("--out", default="", help="默认 <pkg>/gt/clusters_gt.csv")
    a = ap.parse_args()

    pkg = os.path.abspath(a.pkg)
    emb = os.path.abspath(a.emb) if a.emb else os.path.join(pkg, "emb")
    out = os.path.abspath(a.out) if a.out else os.path.join(pkg, "gt", "clusters_gt.csv")

    mem_path = os.path.join(pkg, "results", "members", f"{a.config}.csv")
    gt_path = os.path.join(pkg, "gt", "gt_gallery_members_FILLED.csv")
    for p in (mem_path, gt_path):
        if not os.path.exists(p):
            raise SystemExit(f"缺少输入：{p}")

    pid_of = {r["track_id"]: (r.get("person_id") or "").strip()
              for r in csv.DictReader(open(gt_path, encoding="utf-8-sig"))}
    members = collections.defaultdict(list)
    for r in csv.DictReader(open(mem_path, encoding="utf-8-sig")):
        members[int(r["cluster_local"])].append(r["track_id"])

    segs = sorted({t.split("#")[0] for v in members.values() for t in v})
    keys, Va, Vb = M.load_split(emb, segs)
    vmax = {k: norm(((Va[i] + Vb[i]) / 2.0).reshape(1, -1))[0] for i, k in enumerate(keys)}

    def key(tid):
        s, n = tid.split("#")
        return (s, int(n))

    rows = []
    for cl in sorted(members):
        ts = members[cl]
        pids = [pid_of.get(t, "") for t in ts]
        named = [p for p in pids if p and p.lower() not in NON_HUMAN and not p.startswith(SHADOW_PREFIX)]
        nonhuman = sum(1 for p in pids if p.lower() in NON_HUMAN or p.startswith(SHADOW_PREFIX))
        ids = collections.Counter(named)
        mixed = [f"{p}({c})" for p, c in ids.most_common() if c and p != ids.most_common(1)[0][0]] if ids else []
        vecs = [vmax[key(t)] for t in ts if key(t) in vmax]
        if len(vecs) >= 2:
            V = np.stack(vecs)
            S = V @ V.T
            np.fill_diagonal(S, 1.0)
            dmax = round(float(1.0 - S.min()), 4)
        else:
            dmax = 0.0
        cams = sorted({t.split("_")[0] for t in ts})
        main_pid, main_n = ids.most_common(1)[0] if ids else ("", 0)
        rows.append(dict(
            cluster_id=cl, n_members=len(ts), n_labelled=sum(1 for p in pids if p),
            n_unlabelled=sum(1 for p in pids if not p), n_ids=len(ids), n_nonhuman=nonhuman,
            main_pid=main_pid, main_n=main_n,
            main_share=round(main_n / len(named), 3) if named else "",
            mixed_members="|".join(mixed), n_head="", n_body="", n_partial="",
            n_idswitch="", d_max=dmax, cams="|".join(cams), old_verdict="", agree=""))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    mixed_n = sum(1 for r in rows if r["n_ids"] >= 2)
    print(f"[out] {out}  簇 {len(rows)} 个（混人 {mixed_n}）｜config={a.config}｜表征 {len(vmax)} 条")


if __name__ == "__main__":
    main()
