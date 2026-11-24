#!/usr/bin/env python
"""病例对照实验（Stage A）：找出 OSNet 判错、但答案存在的查询轨迹。

不涉及图像，只做「谁错了、错在哪」的枚举。回答两个问题：
  1. 在有人工真值的查询里，OSNet 把正确簇排到第几；
  2. 哪些是"答案存在但没认出来"，可交给第二模态去救。

用法：
  PY tools/case_control_modal.py --emb output_crosscam_local/1012_yolo8_dd_emb \
      --gt deliverable/data-request-v2/gt \
      --out output_crosscam_local/selector_osnet_1012/case_control.csv
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm                      # noqa: E402
import session_selector_v1 as S                 # noqa: E402
import eval_mask_variants as M                  # noqa: E402
import cluster_camera_aware as CA               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gt(gt_dir):
    """返回 (gallery: {(seg,tid): pid}, query: {(seg,tid): (pid, evaluable)})."""
    gp = os.path.join(gt_dir, "gt_gallery_members_FILLED.csv")
    qp = os.path.join(gt_dir, "gt_query_gt_FILLED.csv")
    gal = {}
    for r in csv.DictReader(open(gp, encoding="utf-8-sig")):
        tid, pid = (r.get("track_id") or "").strip(), (r.get("person_id") or "").strip()
        if not tid or not pid or "unknown" in pid.lower():
            continue
        seg, _, t = tid.rpartition("#")
        gal[(seg, int(t))] = pid
    qry = {}
    for r in csv.DictReader(open(qp, encoding="utf-8-sig")):
        tid, pid = (r.get("query_id") or "").strip(), (r.get("person_id") or "").strip()
        if not tid:
            continue
        ev = (r.get("evaluable") or "").strip()
        seg, _, t = tid.rpartition("#")
        qry[(seg, int(t))] = (pid if pid and "unknown" not in pid.lower() else "",
                              ev == "1")
    return gal, qry


def build(emb_dir, thr_in=0.15, thr_out=0.25, veto=0.30):
    ktr, Va_tr, Vb_tr = M.load_split(emb_dir, S.TRAIN_SEGS)
    kte, Va_te, Vb_te = M.load_split(emb_dir, S.TEST_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    cams_tr = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams_tr, thr_in, thr_out, S.MIN_CLUSTER)

    drop = []
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
    K = len(kept)
    print(f"[cluster] K={K}（veto 弃用 {len(drop)}）")

    mu_g = Vtr.mean(0)
    protos, pcl = [], []
    for ci, c in enumerate(kept):
        m = lab == c
        if not m.sum():
            continue
        protos.append(_norm(Vtr[m].mean(0, keepdims=True))[0])
        pcl.append(ci)
    protos = _norm(np.asarray(protos, np.float32) - mu_g)
    pcl = np.asarray(pcl)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Va_te + Vb_te) / 2.0).mean(0)
    Sa = S.score_rows(Va_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Vb_te, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    return ktr, kte, lab, kept, Sa, Sb, mu_g, protos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="output_crosscam_local/1012_yolo8_dd_emb")
    ap.add_argument("--gt", default="deliverable/data-request-v2/gt")
    ap.add_argument("--thr-in", type=float, default=0.15)
    ap.add_argument("--thr-out", type=float, default=0.25)
    ap.add_argument("--veto", type=float, default=0.30)
    ap.add_argument("--out", default="output_crosscam_local/selector_osnet_1012/case_control.csv")
    a = ap.parse_args()

    gal, qry = load_gt(os.path.join(ROOT, a.gt) if not os.path.isabs(a.gt) else a.gt)
    print(f"[gt] 图库已标 {len(gal)} 条 / 查询已标 {len(qry)} 条（evaluable={sum(1 for v in qry.values() if v[1])}）")

    ktr, kte, lab, kept, Sa, Sb, mu_g, protos = build(
        os.path.join(ROOT, a.emb) if not os.path.isabs(a.emb) else a.emb,
        a.thr_in, a.thr_out, a.veto)

    # 簇 → 多数身份（用已标注图库成员投票）
    cid_pid = {}
    for ci, c in enumerate(kept):
        votes = collections.Counter(gal[k] for k in [ktr[i] for i in np.where(lab == c)[0]] if k in gal)
        if votes:
            cid_pid[ci] = votes.most_common(1)[0][0]
    named = [ci for ci in range(len(kept)) if ci in cid_pid]
    print(f"[cluster] 有身份的簇 {len(named)}/{len(kept)}")

    rows = []
    for i, k in enumerate(kte):
        pid, ev = qry.get(k, ("", False))
        s = np.maximum(Sa[i], 0) if False else None
        sc = (Sa[i] + Sb[i]) / 2.0
        order = np.argsort(-sc)
        top1 = int(order[0])
        # 正确簇：多数身份 == 查询身份
        correct = [ci for ci in named if cid_pid[ci] == pid] if pid else []
        rank = ""
        if correct:
            ranks = [int(np.where(order == ci)[0][0]) + 1 for ci in correct]
            rank = min(ranks)
        rows.append(dict(
            key=k, seg=k[0], tid=k[1], gt_pid=pid, evaluable=int(ev),
            top1_pid=cid_pid.get(top1, ""), top1_score=round(float(sc[top1]), 3),
            osnet_rank=rank,
            n_correct_clusters=len(correct),
            # a/b 是否同簇（当前口径的一致性判据）
            ab_same=int(int(np.argmax(Sa[i])) == int(np.argmax(Sb[i]))),
        ))

    ev_rows = [r for r in rows if r["evaluable"] == 1]
    with_pid = [r for r in ev_rows if r["gt_pid"]]
    has_correct = [r for r in with_pid if r["osnet_rank"] != ""]
    r1 = [r for r in has_correct if r["osnet_rank"] == 1]
    miss = [r for r in has_correct if r["osnet_rank"] != 1]
    print(f"\n=== 病例分布 ===")
    print(f"可评估查询            {len(ev_rows)}")
    print(f"其中有人工身份         {len(with_pid)}")
    print(f"该身份在图库里有簇     {len(has_correct)}  ← 理论上可答")
    print(f"  OSNet 排第 1         {len(r1)}  ({len(r1)/max(1,len(has_correct))*100:.1f}%)")
    print(f"  OSNet 未排第 1       {len(miss)}  ← 候选病例（第二模态可能救）")
    if miss:
        rk = collections.Counter(min(r["osnet_rank"], 6) for r in miss)
        print(f"  未排第1 的名次分布    {dict(sorted(rk.items()))}（6 表示 >=6）")
        by_cam = collections.Counter(r["seg"].split("_")[0] for r in miss)
        print(f"  按机位               {dict(by_cam)}")

    p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda x: (str(x["osnet_rank"] == ""), -x["evaluable"], str(x["key"]))):
            w.writerow(r)
    print(f"\n[out] {p}")


if __name__ == "__main__":
    main()
