#!/usr/bin/env python
"""P2 同机位碎片轨迹拼接（v2：边缘修正 + 静止对长间隔 + 自动分组）。

目标：把同一镜头、同一人的碎片轨迹（遮挡/出画/短暂丢失导致 track ID 切换）
合并，降低"一人多计"。判据（对应跨机位三道判据的机内版）：

  1) 时间门：A 结束 → B 开始，gap ∈ [-overlap_tol, gap_max] 帧
  2) 时空连续性：从 A 尾状态（脚点+速度）外推 gap 后的位置 vs B 首位置。
     若外推越出画面且该端点在画面边缘（出画情形），改用直接距离 dist_raw
     （人在画面外的走法不可知，速度外推失效——如"出门 19 秒后回来"）。
  3) 外观：DINOv3-B 余弦（若提供 embedding）
  4) 尺度：框高比（同一人不应突变；透视远近变化除外）

  静止对长间隔补充：A/B 两端几乎同位置且低速（久坐者离开又回座），
  gap 上限放宽到 --gap-long。

用法:
  python tools/track_stitching.py --jsonl cleaned.jsonl --wh 1920 1080 \
      --emb reid_emb_320_d3b.npz --prefix b --out-csv out/stitch_320.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load_tracks(jsonl_path):
    """tid -> list[(frame, box)]（仅保留 track_ids>=0 的）"""
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((r["frame"], b))
    for k in t:
        t[k].sort()
    return t


GRID_NX, GRID_NY = 12, 8   # 落点网格（与 STI 先验探针一致）


def fit_vel(obs, W, H):
    """脚点位置对帧号的线性拟合速度（归一化 px/frame）"""
    if len(obs) < 3:
        return (0.0, 0.0)
    fs = np.array([f for f, _ in obs], dtype=float)
    us = np.array([(b[0] + b[2]) / 2.0 / W for _, b in obs], dtype=float)
    vs = np.array([b[3] / H for _, b in obs], dtype=float)
    fm = fs.mean()
    A = np.vstack([fs - fm, np.ones_like(fs)]).T
    vx = np.linalg.lstsq(A, us, rcond=None)[0][0]
    vy = np.linalg.lstsq(A, vs, rcond=None)[0][0]
    return (float(vx), float(vy))


def edge_dist(p):
    """点到画面边界的最小归一化距离"""
    return min(p[0], 1.0 - p[0], p[1], 1.0 - p[1])


def out_of_frame(p, margin=0.03):
    return (p[0] < -margin or p[0] > 1 + margin
            or p[1] < -margin or p[1] > 1 + margin)


def track_summary(obs, W, H, win=15):
    f0, b0 = obs[0]
    f1, b1 = obs[-1]
    p0 = ((b0[0] + b0[2]) / 2.0 / W, b0[3] / H)
    p1 = ((b1[0] + b1[2]) / 2.0 / W, b1[3] / H)
    v0 = fit_vel(obs[:win], W, H)
    v1 = fit_vel(obs[-win:], W, H)
    hs = np.array([(b[3] - b[1]) / H for _, b in obs])
    # 驻留众数：落点网格直方图的最高密度格中心。间隔越长，尾点越失效、
    # 众数越可靠（实测 AUC 0.924 vs 尾点 0.814；gap>1200s 时 0.943 vs 0.827）
    gh = np.zeros((GRID_NY, GRID_NX), np.float64)
    for _, b in obs:
        x = (b[0] + b[2]) / 2.0 / W
        y = b[3] / H
        ix = min(max(int(x * GRID_NX), 0), GRID_NX - 1)
        iy = min(max(int(y * GRID_NY), 0), GRID_NY - 1)
        gh[iy, ix] += 1.0
    if gh.sum() > 0:
        iy, ix = np.unravel_index(int(np.argmax(gh)), gh.shape)
        p_mode = ((ix + 0.5) / GRID_NX, (iy + 0.5) / GRID_NY)
    else:
        p_mode = p1
    return {
        "f0": f0, "f1": f1, "p0": p0, "p1": p1, "v0": v0, "v1": v1,
        "p_mode": p_mode, "h": float(np.median(hs)), "n": len(obs),
    }


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--wh", nargs=2, type=int, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--emb", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--gap-max", type=int, default=750, help="常规最大间隔帧（30s）")
    ap.add_argument("--gap-long", type=int, default=3000,
                    help="静止对长间隔上限帧（120s）")
    ap.add_argument("--long-gap-frames", type=int, default=0,
                    help="长间隔分支：超过 gap_max 但不超过该帧数的对，改用两端位置距离判据"
                         "（0=关闭，保持既有行为）。实测 300–1200 s 的两端距离仍可分："
                         "同人中位 0.058 / 随机 0.331")
    ap.add_argument("--long-gap-dist", type=float, default=0.05,
                    help="长间隔分支的两端位置距离上限（归一化，默认 0.05）")
    ap.add_argument("--mode-weight", type=float, default=0.0,
                    help="打分里「驻留众数距离」的权重（默认 0 关闭）。"
                         "实测 cos-0.5*d_mode 在 4 段 train 上 AUC 0.961 vs cos 0.932")
    ap.add_argument("--long-gap-penalty", type=float, default=0.20,
                    help="长间隔边在 mutual-best 打分中的惩罚（默认 0.20）："
                         "短间隔边更可靠，必须优先，否则长边会抢走短边的名额")
    ap.add_argument("--still-dist", type=float, default=0.05,
                    help="静止对两端距离阈值（归一化）")
    ap.add_argument("--still-vel", type=float, default=0.002,
                    help="静止判定速度上限（归一化 px/frame）")
    ap.add_argument("--edge-thr", type=float, default=0.10,
                    help="边缘判定阈值（归一化）")
    ap.add_argument("--overlap-tol", type=int, default=10)
    ap.add_argument("--vel-win", type=int, default=15)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--link-cos", type=float, default=0.80)
    ap.add_argument("--link-pe", type=float, default=0.08)
    ap.add_argument("--out-groups", default="")
    ap.add_argument("--no-covis-guard", dest="covis_guard", action="store_false",
                    default=True,
                    help="关闭「同帧共现」硬约束（默认开：同一机位同一帧出现两个 track，必是两人）")
    ap.add_argument("--relax-gap", type=int, default=0,
                    help="短 gap 放宽：gap ≤ 该帧数的对可走放宽判据（0=关闭，推荐 20）")
    ap.add_argument("--relax-cos", type=float, default=0.90)
    ap.add_argument("--relax-pe", type=float, default=0.25)
    ap.add_argument("--relax-scale", nargs=2, type=float, default=[0.80, 1.25])
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    W, H = args.wh
    tracks = load_tracks(args.jsonl)
    tid2sum = {tid: track_summary(obs, W, H, args.vel_win)
               for tid, obs in tracks.items()}
    frames = ({tid: {f for f, _ in obs} for tid, obs in tracks.items()}
              if args.covis_guard else {})
    print(f"[stitch] {len(tracks)} tracks loaded from {args.jsonl}")

    emb = {}
    if args.emb:
        z = np.load(args.emb)
        for tid in tracks:
            k = f"{args.prefix}{tid}"
            if k in z.files:
                emb[tid] = z[k]
        print(f"[stitch] embeddings: {len(emb)}/{len(tracks)}")

    tids = sorted(tracks.keys())
    rows = []
    for ta in tids:
        sa = tid2sum[ta]
        for tb in tids:
            if tb == ta:
                continue
            sb = tid2sum[tb]
            gap = sb["f0"] - sa["f1"]
            if gap < -args.overlap_tol:
                continue
            # 判断是否为静止对（放宽 gap）
            v_ok = (abs(sa["v1"][0]) < args.still_vel
                    and abs(sa["v1"][1]) < args.still_vel
                    and abs(sb["v0"][0]) < args.still_vel
                    and abs(sb["v0"][1]) < args.still_vel)
            dr = dist(sa["p1"], sb["p0"])
            # 驻留众数距离：sa 的落点众数格 到 sb 的首点。
            # 长间隔下尾点无意义（人早已离开），众数是「他常待的地方」。
            d_mode = dist(sa["p_mode"], sb["p0"])
            still = v_ok and dr < args.still_dist
            gap_cap = args.gap_long if still else args.gap_max
            # ---- 长间隔分支（默认关闭，不影响既有行为）----
            # 速度外推在长 gap 下必然失效（人可能下楼/开会再回来）；
            # 实测「驻留众数」在 gap>1200 s 时判别力 AUC 0.943（尾点 0.827），
            # 所以长 gap 用众数距离，不用外推。
            long_gap = 0
            if args.long_gap_frames > 0 and gap > gap_cap:
                if gap > args.long_gap_frames or d_mode > args.long_gap_dist:
                    continue
                long_gap = 1
                g = max(gap, 0)
                edge_a = edge_dist(sa["p_mode"]) < args.edge_thr
                edge_b = edge_dist(sb["p0"]) < args.edge_thr
                fwd_out = out_of_frame(sa["p_mode"])
                bwd_out = out_of_frame(sb["p0"])
                pe_fwd = pe_bwd = pe_min = pe_adj = d_mode
            else:
                if gap > gap_cap:
                    continue
                g = max(gap, 0)
                # 前向外推
                predB = (sa["p1"][0] + sa["v1"][0] * g,
                         sa["p1"][1] + sa["v1"][1] * g)
                edge_a = edge_dist(sa["p1"]) < args.edge_thr
                edge_b = edge_dist(sb["p0"]) < args.edge_thr
                fwd_out = out_of_frame(predB)
                pe_fwd = dist(predB, sb["p0"]) if g > 0 else dr
                pe_fwd_adj = dr if (fwd_out and edge_a) else pe_fwd
                # 反向外推
                predA = (sb["p0"][0] - sb["v0"][0] * g,
                         sb["p0"][1] - sb["v0"][1] * g)
                bwd_out = out_of_frame(predA)
                pe_bwd = dist(predA, sa["p1"]) if g > 0 else dr
                pe_bwd_adj = dr if (bwd_out and edge_b) else pe_bwd
                pe_min = min(pe_fwd, pe_bwd)
                pe_adj = min(pe_fwd_adj, pe_bwd_adj)
            cos = None
            if ta in emb and tb in emb:
                cos = float(np.dot(emb[ta], emb[tb]))
            scale = (sb["h"] / sa["h"]) if sa["h"] > 1e-6 else None
            rows.append({
                "d_mode": round(d_mode, 4),
                "tid_a": ta, "tid_b": tb, "gap_frames": int(gap),
                "gap_s": round(gap / args.fps, 2),
                "pe_fwd": round(pe_fwd, 4), "pe_bwd": round(pe_bwd, 4),
                "pe_min": round(pe_min, 4), "pe_adj": round(pe_adj, 4),
                "dist_raw": round(dr, 4),
                "cos": None if cos is None else round(cos, 4),
                "scale": None if scale is None else round(scale, 3),
                "edge_a": int(edge_a), "edge_b": int(edge_b),
                "fwd_out": int(fwd_out), "bwd_out": int(bwd_out),
                "still": int(still), "long_gap": int(long_gap),
                "n_a": sa["n"], "n_b": sb["n"],
                "f0a": sa["f0"], "f1a": sa["f1"],
                "f0b": sb["f0"], "f1b": sb["f1"],
            })

    rows.sort(key=lambda r: r["pe_adj"])
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tid_a", "tid_b", "gap_frames", "gap_s", "pe_fwd", "pe_bwd",
            "pe_min", "pe_adj", "dist_raw", "d_mode", "cos", "scale",
            "edge_a", "edge_b", "fwd_out", "bwd_out", "still", "long_gap",
            "n_a", "n_b", "f0a", "f1a", "f0b", "f1b"]
    with open(out, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in cols)
                    + "\n")
    print(f"[stitch] {len(rows)} candidate pairs -> {out}")

    # ---- 连接构造：mutual-best 链（禁止传递闭包，防"同区域不同人"链式误连）----
    if args.out_groups:
        def accept(r):
            """基础判据，或「短 gap 放宽」判据（二者并集）。"""
            if r["cos"] is None:
                return False
            if r["cos"] >= args.link_cos and r["pe_adj"] <= args.link_pe:
                return True
            if args.relax_gap > 0 and r["gap_frames"] <= args.relax_gap:
                if r["cos"] >= args.relax_cos and r["pe_adj"] <= args.relax_pe:
                    sc = r["scale"]
                    if sc is None or (args.relax_scale[0] <= sc <= args.relax_scale[1]):
                        return True
            return False

        n_covis = 0
        cands = []
        for r in rows:
            if not accept(r):
                continue
            fa, fb = frames.get(r["tid_a"]), frames.get(r["tid_b"])
            if fa and fb and (fa & fb):
                n_covis += 1          # 同帧同场：必是两人，禁止连边
                continue
            cands.append(r)

        def score(r):
            s = r["cos"] - 2.0 * r["pe_adj"]
            if args.mode_weight > 0 and r.get("d_mode") is not None:
                s -= args.mode_weight * r["d_mode"]
            if r.get("long_gap"):
                s -= args.long_gap_penalty
            return s

        best_succ, best_pred = {}, {}
        for r in sorted(cands, key=score, reverse=True):
            a, b = r["tid_a"], r["tid_b"]
            best_succ.setdefault(a, r)
            best_pred.setdefault(b, r)
        links = []
        for a, r in best_succ.items():
            b = r["tid_b"]
            if b in best_pred and best_pred[b]["tid_a"] == a:
                links.append(r)
        succ = {r["tid_a"]: r["tid_b"] for r in links}
        pred = {r["tid_b"]: r["tid_a"] for r in links}
        starts = sorted(t for t in succ if t not in pred)
        chains = []
        for s in starts:
            ch = [s]
            cur = s
            while cur in succ:
                cur = succ[cur]
                ch.append(cur)
            if len(ch) > 1:
                chains.append(ch)
        chain_out = []
        for ch in chains:
            f0 = min(tid2sum[t]["f0"] for t in ch)
            f1 = max(tid2sum[t]["f1"] for t in ch)
            chain_out.append({"chain": ch, "f0": f0, "f1": f1,
                              "span_s": round((f1 - f0) / args.fps, 1)})
        Path(args.out_groups).write_text(
            json.dumps({
                "criteria": {"cos": args.link_cos, "pe_adj": args.link_pe,
                             "rule": "mutual_best, no transitive closure",
                             "covis_guard": bool(args.covis_guard),
                             "relax": ({"gap_max": args.relax_gap,
                                        "cos_min": args.relax_cos,
                                        "pe_max": args.relax_pe,
                                        "scale": list(args.relax_scale)}
                                       if args.relax_gap > 0 else None)},
                "n_candidates": len(cands), "n_links": len(links),
                "covis_dropped": n_covis,
                "links": links, "chains": chain_out,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stitch] {len(cands)} candidates -> {len(links)} mutual links"
              f" (covis_dropped={n_covis})"
              f" -> {len(chains)} chains -> {args.out_groups}")
        for c in chain_out:
            print(f"   chain: {c['chain']}  span {c['span_s']}s")

    print(f"\n=== top {args.top} by pe_adj ===")
    hdr = (f"{'A->B':>10} {'gap_s':>6} {'pe_adj':>7} {'pe_min':>7} "
           f"{'dist':>6} {'cos':>6} {'scale':>6} {'E/st':>5} {'n':>9}")
    print(hdr)
    for r in rows[:args.top]:
        cos = f"{r['cos']:.3f}" if r["cos"] is not None else "  n/a"
        est = f"{r['edge_a']}{r['edge_b']}/{r['still']}"
        print(f"{r['tid_a']:>4}->{r['tid_b']:<4} {r['gap_s']:>6.1f} "
              f"{r['pe_adj']:>7.3f} {r['pe_min']:>7.3f} {r['dist_raw']:>6.3f} "
              f"{cos:>6} "
              f"{'' if r['scale'] is None else format(r['scale'], '.2f'):>6} "
              f"{est:>5} {r['n_a']:>4}/{r['n_b']:<4}")


if __name__ == "__main__":
    main()
