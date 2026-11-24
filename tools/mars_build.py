#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MARS bbox_test → 本项目流水线可吃的"伪视频段"数据包（不跑检测/跟踪）。

为什么要这层适配
----------------
MARS 只给切好的轨迹裁图（12180 条 tracklet / 681089 张 jpg），没有原视频。
本项目的下游（d7_sweep 聚类扫描 + eval_precision 评价）只依赖三样东西：

  1. emb：`<seg>_emb_a.npz` / `<seg>_emb_b.npz`，键 `<seg>#<tid>`（轨迹前半/后半均值特征）
  2. det：`<det>/<seg>/<seg>_detections.jsonl`（只为算同帧共现对 cooc）
  3. gt ：`gt_gallery_members_FILLED.csv` / `gt_query_gt_FILLED.csv`

所以这里把每条 MARS tracklet 当作"一条轨迹"，塞进合成的段名 `<cam>_<g|q>`：
  - 机位 = `c1..c6`（d7_sweep 用 `seg.split("_")[0]` 取机位，正好命中机位感知聚类）
  - 图库段 `c{k}_g`、查询段 `c{k}_q`，两段轨迹互斥（MARS 无时间维度，
    "图库/查询隔离"用轨迹级互斥表达，而不是现场的时间切分）
  - 同段内每条轨迹占一段**不重叠的帧号**（每条间隔 FRAME_GAP），
    因此段内共现对恒为空 → fm 恒为 nan。这是 MARS 的固有限制（无同步），
    不是 bug：写进 run_info.deviations。

真值来源（官方，非人工标注）
--------------------------
  datasets/MARS_info/tracks_test_info.mat  (12180×4) = [图序起点, 图序终点, pid, cam]
  datasets/MARS_info/query_IDX.mat         (1×1980)  = 查询图序（每图序所属 tracklet 即查询轨迹）
  pid=-1 → 无效框（no-human），pid=0 → 名单外（junk）。二者在图库里作干扰项，不给身份标签。

用法：
  PY tools/mars_build.py --root datasets/bbox_test --info datasets/MARS_info \
     --out validation/mars --pids 150 --seed 42
"""
import argparse
import collections
import csv
import json
import os
import random
import re

import numpy as np
import scipy.io as sio

PID_JUNK = "0000"
PID_BAD = "00-1"
FRAME_GAP = 10000          # 同段内相邻轨迹的帧号间隔，保证段内零共现
NAME_RE = re.compile(r"^(.{4})C(\d)T(\d+)F(\d+)\.jpg$")


def scan(root):
    """→ {pid: {(cam, T): (n_img, sorted_frames)}}，pid 用官方 4 位串（-1 记为 '00-1'）"""
    out = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        got = collections.defaultdict(list)
        for f in os.listdir(p):
            m = NAME_RE.match(f)
            if not m:
                raise SystemExit(f"文件名不合规范: {d}/{f}")
            got[(int(m.group(2)), int(m.group(3)))].append(int(m.group(4)))
        out[d] = {k: (len(v), sorted(v)) for k, v in got.items()}
    return out


def match_tracklets(info_rows, img_groups):
    """把 mat 的行（起点/终点/pid/cam）对到目录里的 (cam,T) 组。

    依据：mat 的 col0/col1 是**全库图序**，而 bbox_test 的文件顺序就是
    「pid → cam → T」，所以同一 pid 内按 col0 排序的第 k 条 == 按 (cam,T) 排序的第 k 组。
    实测 636 个 pid / 12180 条：按此对齐后 cam 错 0 条、图数错 0 条。
    （不能用 (cam,图数) 当键：187 个身份存在重复键，会错配。）
    任一条对不上直接退出——宁可不跑，也不要静默错配。
    """
    by_pid = collections.defaultdict(list)
    for r in info_rows:
        pid = PID_BAD if r[2] == -1 else f"{r[2]:04d}"
        by_pid[pid].append((int(r[0]), int(r[1]), int(r[3])))
    mapping = {}          # (pid, idx_in_mat) -> (cam, T, n)
    for pid, rows in by_pid.items():
        rows.sort()
        if pid not in img_groups:
            raise SystemExit(f"mat 里的 pid {pid} 在 bbox_test 下不存在")
        groups = sorted(img_groups[pid].items())
        if len(groups) != len(rows):
            raise SystemExit(f"pid {pid}: mat {len(rows)} 条轨迹，目录 {len(groups)} 条")
        for i, ((cam, T), (n, _frs)) in enumerate(groups):
            s, e, mc = rows[i]
            if cam != mc or n != e - s + 1:
                raise SystemExit(f"pid {pid} 第 {i} 条对不上: 目录(cam={cam},n={n}) mat(cam={mc},n={e-s+1})")
            mapping[(pid, i)] = (cam, T, n)
    return by_pid, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/bbox_test")
    ap.add_argument("--info", default="datasets/MARS_info")
    ap.add_argument("--out", default="validation/mars")
    ap.add_argument("--pids", type=int, default=150, help="随机抽多少个本人身份（对应 VALIDATION §3）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--junk", type=int, default=768, help="pid=0 干扰轨迹条数")
    ap.add_argument("--bad", type=int, default=206, help="pid=-1 干扰轨迹条数")
    ap.add_argument("--min-frames", type=int, default=6)
    a = ap.parse_args()

    img = scan(a.root)
    frs_of = {(pid, cam, Tn): frs for pid, g in img.items() for (cam, Tn), (_n, frs) in g.items()}
    T = sio.loadmat(os.path.join(a.info, "tracks_test_info.mat"))["track_test_info"]
    qidx = np.asarray(sio.loadmat(os.path.join(a.info, "query_IDX.mat"))["query_IDX"]).ravel()
    by_pid, mapping = match_tracklets(T, img)
    pos_in_pid = {(pid, rows[i][0]): i for pid, rows in by_pid.items()
                  for i in range(len(rows))}

    # 图序 → 轨迹。query_IDX 是 **1 基的 tracks_test_info 行号**（uint16，max=12180，
    # 与 12180 条轨迹严格对应），不是全库图序——这点必须按官方 evaluate.py 口径走。
    qof = {}
    for q in qidx:
        r = T[int(q) - 1]
        pid = PID_BAD if r[2] == -1 else f"{r[2]:04d}"
        s = int(r[0])
        i = pos_in_pid[(pid, s)]
        qof[(pid, i)] = True
    print(f"[info] tracklet {T.shape[0]} 条｜查询轨迹 {len(qof)} 条｜pid 数 {len(by_pid)}")

    named = [p for p in by_pid if p not in (PID_JUNK, PID_BAD)]
    named.sort()
    cand = [p for p in named
            if len({mapping[(p, i)][0] for i in range(len(by_pid[p]))}) >= 2]
    rng = random.Random(a.seed)
    picked = sorted(rng.sample(cand, min(a.pids, len(cand))))
    print(f"[pick] 有 ≥2 机位的身份 {len(cand)}｜随机抽 {len(picked)}（seed={a.seed}）")

    # 组段：c{k}_{g|q}
    segs = collections.defaultdict(list)     # seg -> [(pid, cam, T, n, is_query)]
    for pid in picked:
        for i in range(len(by_pid[pid])):
            cam, Tn, n = mapping[(pid, i)]
            if n < a.min_frames:
                continue
            segs[f"c{cam}_{'q' if (pid, i) in qof else 'g'}"].append((pid, cam, Tn, n))
    for pid, tag in ((PID_JUNK, "junk"), (PID_BAD, "bad")):
        pool = []
        for i in range(len(by_pid[pid])):
            cam, Tn, n = mapping[(pid, i)]
            if n >= a.min_frames:
                pool.append((pid, cam, Tn, n))
        k = a.junk if tag == "junk" else a.bad
        take = sorted(rng.sample(pool, min(k, len(pool))))
        for rec in take:
            segs[f"c{rec[1]}_g"].append(rec)
        print(f"[distract] {tag}: 取 {len(take)}/{len(pool)} 条")

    out = os.path.abspath(a.out)
    for sub in ("gt", "det", "manifest"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    gt_gal, gt_qry, stats = [], [], {}
    for seg in sorted(segs):
        recs = sorted(segs[seg], key=lambda r: (r[0], r[1], r[2]))
        det_dir = os.path.join(out, "det", seg)
        os.makedirs(det_dir, exist_ok=True)
        is_q = seg.endswith("_q")
        with open(os.path.join(out, "manifest", f"{seg}.jsonl"), "w", encoding="utf-8") as fm, \
                open(os.path.join(det_dir, f"{seg}_detections.jsonl"), "w", encoding="utf-8") as fd:
            for j, (pid, cam, Tn, n) in enumerate(recs):
                tid = j
                prefix = f"{pid}C{cam}T{Tn:04d}F"
                frs = frs_of[(pid, cam, Tn)]
                rec = dict(track_id=f"{seg}#{tid}", tid=tid, pid=pid,
                           cam=f"c{cam}", split="query" if is_q else "gallery",
                           n_frames=n, dir=os.path.join(os.path.abspath(a.root), pid),
                           prefix=prefix)
                # 12180 条里有 16 条帧号不连续（个别图缺失/起点非 1），这些显式列帧号
                if frs == list(range(frs[0], frs[0] + len(frs))):
                    rec["f0"] = frs[0]
                else:
                    rec["frames"] = frs
                fm.write(json.dumps(rec, ensure_ascii=False) + "\n")
                base = j * FRAME_GAP
                for k in range(n):
                    fd.write(json.dumps({"frame": base + k,
                                         "track_ids": [tid],
                                         "boxes": [[0, 0, 128, 256]],
                                         "scores": [1.0]}) + "\n")
                if is_q:
                    gt_qry.append(dict(query_id=f"{seg}#{tid}", person_id=pid))
                else:
                    gt_gal.append(dict(track_id=f"{seg}#{tid}", seg=seg, cam=f"c{cam}",
                                       tid=tid, n_frames=n, box_type="body",
                                       person_id=pid if pid not in (PID_JUNK, PID_BAD) else "",
                                       note="", prompt_jpg=""))
        stats[seg] = len(recs)

    with open(os.path.join(out, "gt", "gt_gallery_members_FILLED.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["track_id", "seg", "cam", "tid", "n_frames",
                                           "box_type", "person_id", "note", "prompt_jpg"])
        w.writeheader()
        w.writerows(gt_gal)
    with open(os.path.join(out, "gt", "gt_query_gt_FILLED.csv"), "w",
              newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["query_id", "person_id"])
        w.writeheader()
        w.writerows(gt_qry)

    gal_lab = sum(1 for r in gt_gal if r["person_id"])
    run_info = dict(
        dataset="MARS",
        subset=dict(gallery=sorted(s for s in segs if s.endswith("_g")),
                    query=sorted(s for s in segs if s.endswith("_q")),
                    pids=len(picked), seed=a.seed),
        counts=dict(gallery_tracks=len(gt_gal), gallery_labeled=gal_lab,
                    gallery_unlabeled=len(gt_gal) - gal_lab,
                    query_tracks=len(gt_qry),
                    query_identities=len({r["person_id"] for r in gt_qry})),
        per_seg=stats,
        source=dict(bbox_test=os.path.abspath(a.root),
                    tracks_test_info="MARS-evaluation/info/tracks_test_info.mat",
                    query_IDX="MARS-evaluation/info/query_IDX.mat"),
        deviations=[
            "MARS 无时间维度：图库/查询用轨迹级互斥（官方 query_IDX），不是现场的时间切分",
            "MARS 无多机位同步：段内轨迹帧号互不重叠 → 共现对为空 → false_merge 不可计算（恒 nan）",
            "特征口径沿用历史 BGR（cv2 读图直接送 torchreid），与冻结基线一致，未开 --rgb",
            "pid=0（junk）与 pid=-1（invalid）在图库中作无标签干扰项，不映射为 shadow/no-human 身份",
            "轨迹内串人（id_switch）在 MARS 上不存在，purity 恒为 1",
        ],
    )
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, ensure_ascii=False, indent=2)
    print(f"[out] {out}")
    print(f"      图库 {len(gt_gal)} 条（有身份 {gal_lab}）｜查询 {len(gt_qry)} 条"
          f"｜段 {len(segs)} 个")


if __name__ == "__main__":
    main()
