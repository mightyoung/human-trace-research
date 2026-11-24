#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务二数据 v2：在 v1 基础上增加
  ① 社交邻域上下文（Social LSTM 借鉴）：检查帧 ±stride 内活跃的最近 K 个邻居的运动状态
  ② 行为特征 + 行为聚类标签（BP-SGCN "行为伪标签"借鉴）：样本历史窗的运动学描述子 → K-means

样本定义与 v1 完全一致（同一代码逻辑重建，n / 类别分布应对齐）。
邻居特征（每邻居 6 维，尺度=目标身高，跨机位可比）：
  [du, dv, ddu, ddv, v_mag, cooc_frac]
  du/dv: 邻居脚印 − 目标脚印（除以目标框高）
  ddu/ddv: 邻居速度 − 目标速度（每帧位移差，除以目标框高）
  v_mag: 邻居自身速度大小
  cooc_frac: 历史窗内邻居出现帧比例（社交共现强度）
行为特征（6 维，从历史窗自算，无泄漏）：
  [v_mean, v_std, |dθ|_mean, dθ_std, straightness, acc_std]

产物：{out}.npz（X, Xf, y, v_fut, theta, v_in, meta, Nc, Nmask, bfeat, bc）
      + stats json。行为聚类在 【train 上 fit】，val/test transform（防泄漏）。
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from intent_dataset import (load_tracks, foot, wrap_deg, classify, RES,  # noqa: E402
                            CLASS_NAMES, DEF_SEGS_1012, srcs_1012)

ROOT = os.path.dirname(HERE)
K_NEIGHBORS = 5


def build_v2(srcs, fps=25.0, stride=5, L_in=12, L_fut=5, win_stride=2,
             thr_stop=0.01, dt=None, thr_move=0.01):
    DT = dt or (stride / fps)
    min_len = L_in + L_fut + 2
    X, XF, Y, VF, TH, VI, META = [], [], [], [], [], [], []
    NC, NM, BF = [], [], []
    seg_stats = {}
    for seg, cam, jp in srcs:
        if not os.path.exists(jp):
            print(f"[skip] missing {jp}")
            continue
        W, H = RES[cam]
        tracks = load_tracks(jp)
        # 帧索引：frame -> list[(tid, box)]
        frame_idx = {}
        track_frames = {}
        for tid, seq in tracks.items():
            if len(seq) < 3:
                continue
            track_frames[tid] = set(f for f, _ in seq)
            for f, b in seq:
                frame_idx.setdefault(f, []).append((tid, b))
        n_samp, n_no_nb, n_nb_hist = 0, 0, 0
        for tid, seq in tracks.items():
            if len(seq) < min_len:
                continue
            frames = [f for f, _ in seq]
            t_ok = all((frames[i + 1] - frames[i]) <= stride * 1.5
                       for i in range(len(frames) - 1))
            if not t_ok:
                blocks, cur = [], [0]
                for i in range(len(frames) - 1):
                    if frames[i + 1] - frames[i] <= stride * 1.5:
                        cur.append(i + 1)
                    else:
                        blocks.append(cur)
                        cur = [i + 1]
                blocks.append(cur)
                blocks.sort(key=len, reverse=True)
                if not blocks or len(blocks[0]) < min_len:
                    continue
                seq = [seq[i] for i in blocks[0]]
            feats = []
            prev_uv = None
            for f, b in seq:
                u, v = foot(b)
                u, v = u / W, v / H
                w = (b[2] - b[0]) / W
                h = (b[3] - b[1]) / H
                if prev_uv is None:
                    du = dv = 0.0
                else:
                    du = u - prev_uv[0]
                    dv = v - prev_uv[1]
                feats.append([u, v, du, dv, w, h, DT])
                prev_uv = (u, v)
            feats = np.asarray(feats, dtype=np.float32)
            pts = feats[:, :2]
            n = len(feats)
            last_start = n - (L_in + L_fut)
            for s in range(0, last_start + 1, win_stride):
                xi = s + L_in - 1
                xf = s + L_in + L_fut - 1
                back = max(s, xi - 4)
                d_in = pts[xi] - pts[back]
                v_in = float(np.hypot(d_in[0], d_in[1]) / (max(1, xi - back) * DT))
                if thr_move > 0 and v_in < thr_move:
                    continue
                d = pts[xf] - pts[xi]
                v_f = float(np.hypot(d[0], d[1]) / (L_fut * DT))
                n_in = np.hypot(d_in[0], d_in[1])
                n_f = np.hypot(d[0], d[1])
                if n_in < 1e-8 or n_f < 1e-8:
                    theta = 0.0
                else:
                    theta = wrap_deg(math.degrees(math.atan2(d[1], d[0])
                                                  - math.atan2(d_in[1], d_in[0])))
                y = classify(v_f, theta, thr_stop)

                # ---- 社交邻域 ----
                xi_frame = seq[xi][0]
                h_t = max(float(feats[xi][5]), 1e-4)
                cand = {}   # tid2 -> (d_frame, box)
                for df in range(0, stride + 1):
                    for f2 in ({xi_frame - df, xi_frame + df}):
                        for tid2, b2 in frame_idx.get(f2, []):
                            if tid2 == tid or tid2 not in track_frames:
                                continue
                            if tid2 not in cand or df < cand[tid2][0]:
                                cand[tid2] = (df, b2)
                win_frames = set(int(seq[i][0]) for i in range(s, xi + 1))
                nb = []
                tu, tv = float(feats[xi][0]), float(feats[xi][1])
                for tid2, (df, b2) in cand.items():
                    u2, v2 = foot(b2)
                    u2, v2 = u2 / W, v2 / H
                    dist = math.hypot(u2 - tu, v2 - tv)
                    nb.append((dist, tid2, u2, v2))
                nb.sort(key=lambda z: z[0])
                nc_arr = np.zeros((K_NEIGHBORS, 6), np.float32)
                nm_arr = np.zeros((K_NEIGHBORS,), np.float32)
                for j, (dist, tid2, u2, v2) in enumerate(nb[:K_NEIGHBORS]):
                    # 邻居速度（其 xi_frame 前最近帧的位移；间隔过大则置 0）
                    seq2 = tracks[tid2]
                    f_prev, b_prev = None, None
                    for f2, b2 in seq2:
                        if f2 >= xi_frame:
                            break
                        f_prev, b_prev = f2, b2
                    if b_prev is not None and (xi_frame - f_prev) <= 2 * stride:
                        pu, pv = foot(b_prev)
                        du2 = (u2 - pu / W)
                        dv2 = (v2 - pv / H)
                    else:
                        du2 = dv2 = 0.0
                    du_t, dv_t = float(feats[xi][2]), float(feats[xi][3])
                    cooc = len(win_frames & track_frames[tid2]) / max(1, L_in)
                    nc_arr[j] = [(u2 - tu) / h_t, (v2 - tv) / h_t,
                                 (du2 - du_t) / h_t, (dv2 - dv_t) / h_t,
                                 math.hypot(du2, dv2) / h_t, cooc]
                    nm_arr[j] = 1.0
                NC.append(nc_arr)
                NM.append(nm_arr)
                if len(nb) == 0:
                    n_no_nb += 1
                else:
                    n_nb_hist += 1

                # ---- 行为特征（历史窗自算） ----
                seg_f = feats[s:xi + 1]
                vt = np.hypot(seg_f[1:, 2], seg_f[1:, 3])
                ang = np.arctan2(seg_f[:, 3], seg_f[:, 2])
                dth = np.diff(ang)
                dth = (dth + np.pi) % (2 * np.pi) - np.pi
                path = float(vt.sum())
                disp = float(np.hypot(*(pts[xi] - pts[s])))
                straight = disp / path if path > 1e-8 else 1.0
                acc = np.abs(np.diff(vt)) if len(vt) > 1 else np.zeros(1)
                BF.append([float(vt.mean()), float(vt.std()),
                           float(np.abs(dth).mean()), float(dth.std()),
                           float(straight), float(acc.mean())])

                X.append(feats[s:xi + 1].copy())
                XF.append(feats[xi + 1:xf + 1, :2].copy())
                Y.append(y)
                VF.append(v_f)
                TH.append(theta)
                VI.append(v_in)
                META.append((seg, cam, tid, int(seq[xi][0])))
                n_samp += 1
        seg_stats[seg] = {"tracks": len(tracks), "samples": n_samp,
                          "no_neighbor": n_no_nb, "with_neighbor": n_nb_hist}
        print(f"[seg] {seg}: samples={n_samp} no_nb={n_no_nb} ({n_no_nb / max(1, n_samp):.2%})")
    out = {"X": np.stack(X).astype(np.float32),
           "Xf": np.stack(XF).astype(np.float32),
           "y": np.asarray(Y, dtype=np.int64),
           "v_fut": np.asarray(VF, np.float32),
           "theta": np.asarray(TH, np.float32),
           "v_in": np.asarray(VI, np.float32),
           "meta": np.asarray(META, dtype=object),
           "Nc": np.stack(NC).astype(np.float32),
           "Nmask": np.stack(NM).astype(np.float32),
           "bfeat": np.asarray(BF, np.float32)}
    return out, seg_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output_crosscam",
                                                      "analysis", "intent"))
    ap.add_argument("--n-behavior", type=int, default=6)
    ap.add_argument("--fit-behavior-from", default="", help="train npz（行为聚类 fit 源）")
    a = ap.parse_args()
    segs = DEF_SEGS_1012[a.split]
    srcs = srcs_1012(segs)
    out, seg_stats = build_v2(srcs, fps=25.0, stride=5)
    npz_path = os.path.join(a.out_dir, f"1012_{a.split}_v2.npz")

    # 行为聚类：train fit / val|test transform
    from scipy.cluster.vq import kmeans2
    if a.split == "train":
        bf = out["bfeat"]
        mu, sd = bf.mean(0), bf.std(0) + 1e-6
        cen, bc = kmeans2((bf - mu) / sd, a.n_behavior, iter=60, minit="++", seed=42)
        bstats = {"mu": mu.tolist(), "sd": sd.tolist(), "centroids": cen.tolist()}
        json.dump(bstats, open(os.path.join(a.out_dir, "behavior_kmeans.json"), "w"),
                  ensure_ascii=False, indent=1)
        cnt = np.bincount(bc, minlength=a.n_behavior).tolist()
        print(f"[behavior] K={a.n_behavior} fit on train, counts={cnt}")
    else:
        bst = json.load(open(os.path.join(a.out_dir, "behavior_kmeans.json")))
        mu = np.asarray(bst["mu"], np.float32)
        sd = np.asarray(bst["sd"], np.float32)
        cen = np.asarray(bst["centroids"], np.float32)
        z = (out["bfeat"] - mu) / sd
        d2 = ((z[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
        bc = d2.argmin(1).astype(np.int64)
        cnt = np.bincount(bc, minlength=a.n_behavior).tolist()
        print(f"[behavior] transform on {a.split}, counts={cnt}")
    out["bc"] = bc.astype(np.int64)

    np.savez_compressed(npz_path, **out)
    stats = {"n": int(len(out["y"])), "n_neighbor_mean": float(out["Nmask"].sum(1).mean()),
             "no_neighbor_ratio": float((out["Nmask"].sum(1) == 0).mean()),
             "segs": seg_stats,
             "class_counts": {CLASS_NAMES[k]: int((out["y"] == k).sum()) for k in range(5)},
             "behavior_counts": cnt}
    json.dump(stats, open(npz_path.replace(".npz", "_stats.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"[done] {len(out['y'])} samples -> {os.path.basename(npz_path)} "
          f"(nb_mean={stats['n_neighbor_mean']:.2f}, no_nb={stats['no_neighbor_ratio']:.2%})")
    print(f"[dist] {stats['class_counts']}")


if __name__ == "__main__":
    main()
