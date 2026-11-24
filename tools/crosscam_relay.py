#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨机位接力研究（远景→近景）

用途：回答"远景机位 A 中的远处小目标，能否被近景机位 B 拍到并接管跟踪"。
第一步：映射覆盖分析 —— A(318) 画面的哪些区域映射进 B(320) 视野。

用法：
  python tools/crosscam_relay.py mapcover --a 318 --b 320
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosscam_spatiotemporal import (  # noqa: E402
    fit_map, apply_map, load_train_points, TRAIN_PAIRS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from det_view import local_det, net_dir  # noqa: E402
# 输出/实体目录可用环境变量切换：
#   CROSSCAM_RELAY_OUT → relay 产物目录（默认 analysis/relay）
#   CROSSCAM_NET_DIR   → net_tracks 实体目录（默认 = 规范目录 analysis/stitch_clean2，
#                        经 det_view.net_dir() 解析；2026-10-14 起不再默认无掩膜 stitch）
OUT_DIR = os.environ.get(
    "CROSSCAM_RELAY_OUT", os.path.join(ROOT, "output_crosscam", "analysis", "relay"))
NET_DIR = os.environ.get("CROSSCAM_NET_DIR") or net_dir()

RES = {"316": (2560, 1440), "305": (2560, 1440), "318": (1920, 1080),
       "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}

# 各路首帧墙钟（相对 08:55:00 的秒数，OSD 读取 2026-10-12；305 为推定±2s）
T0_OFFSET = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0


def load_foot_trajs(cam):
    """从检测 jsonl 提取每条 track 的脚点序列 -> {tid: (frames, u, v, h)}"""
    path = local_det(cam)
    W, H = RES[cam]
    acc = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for tid, box in zip(d["track_ids"], d["boxes"]):
                x1, y1, x2, y2 = box
                u = (x1 + x2) / 2 / W
                v = y2 / H
                h = (y2 - y1)
                acc.setdefault(tid, []).append((fr, u, v, h))
    out = {}
    for tid, rows in acc.items():
        rows.sort()
        fr = np.array([r[0] for r in rows], dtype=float)
        u = np.array([r[1] for r in rows])
        v = np.array([r[2] for r in rows])
        h = np.array([r[3] for r in rows])
        out[tid] = (fr, u, v, h)
    return out


def load_entities(cam):
    """读 net_tracks_{cam}.json -> 实体级轨迹 {eid: (frames, u, v, h, tid_first, tid_last)}"""
    path = os.path.join(NET_DIR, f"net_tracks_{cam}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    foot = load_foot_trajs(cam)
    ents = {}
    for e in data["entities"]:
        frs, us, vs, hs = [], [], [], []
        for tid in e["tracks"]:
            if tid not in foot:
                continue
            fr, u, v, h = foot[tid]
            frs.append(fr); us.append(u); vs.append(v); hs.append(h)
        if not frs:
            continue
        fr = np.concatenate(frs); u = np.concatenate(us)
        v = np.concatenate(vs); h = np.concatenate(hs)
        idx = np.argsort(fr)
        ents[e["eid"]] = (fr[idx], u[idx], v[idx], h[idx])
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mapcover", "events", "links", "score"])
    ap.add_argument("--sim-thr", type=float, default=0.65,
                    help="外观 sim 门槛（318×320 基准：正均值 0.707 / 负 p90 0.644）")
    ap.add_argument("--a", default="318")
    ap.add_argument("--b", default="320")
    ap.add_argument("--dt-min", type=float, default=-1.0,
                    help="最小间隔秒（负值允许时间重叠）")
    ap.add_argument("--dt-max", type=float, default=60.0)
    ap.add_argument("--only-moving", action="store_true", default=True)
    ap.add_argument("--all-entities", dest="only_moving", action="store_false")
    ap.add_argument("--pairs-csv", default=os.path.join(
        ROOT, "output_crosscam", "analysis", "confirmed_pairs.csv"))
    args = ap.parse_args()

    if args.cmd == "score":
        # 对时间门候选打外观分（末碎片 vs 首碎片 的 DINOv3-B cos）
        cams = ["316", "305", "318", "320", "323", "343"]
        prefix = {"318": "a", "320": "b"}
        for c in cams:
            prefix.setdefault(c, f"v{c}_")

        # 碎片时间跨度
        spans = {}
        for cam in cams:
            foot = load_foot_trajs(cam)
            spans[cam] = {tid: (float(fr[0]), float(fr[-1])) for tid, (fr, u, v, h) in foot.items()}

        # 实体首/末碎片
        ent_frag = {}
        for cam in cams:
            with open(os.path.join(NET_DIR, f"net_tracks_{cam}.json"),
                      encoding="utf-8") as f:
                d = json.load(f)
            for e in d["entities"]:
                tids = [t for t in e["tracks"] if t in spans[cam]]
                if not tids:
                    continue
                tids.sort(key=lambda t: spans[cam][t][0])
                ent_frag[f"{cam}:{e['eid']}"] = (tids[0], tids[-1])

        embs = {}
        for cam in cams:
            p = os.path.join(OUT_DIR, "remote_emb", f"reid_emb_{cam}_d3b.npz")
            npz = np.load(p)
            embs[cam] = {k: npz[k] for k in npz.files}

        def get_emb(cam, tid):
            k = f"{prefix[cam]}{tid}"
            return embs[cam].get(k)

        with open(os.path.join(OUT_DIR, "relay_candidates.json"), encoding="utf-8") as f:
            cand = json.load(f)["candidates"]

        n_missing = 0
        scored = []
        for c in cand:
            ca, ea = c["a"].split(":")
            cb, eb = c["b"].split(":")
            if c["a"] not in ent_frag or c["b"] not in ent_frag:
                n_missing += 1
                continue
            fa = ent_frag[c["a"]][1]  # a 的末碎片
            fb = ent_frag[c["b"]][0]  # b 的首碎片
            va = get_emb(ca, fa)
            vb = get_emb(cb, fb)
            if va is None or vb is None:
                n_missing += 1
                continue
            sim = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
            rec = dict(c)
            rec["sim"] = round(sim, 4)
            rec["a_frag_last"] = fa
            rec["b_frag_first"] = fb
            scored.append(rec)

        scored.sort(key=lambda r: -r["sim"])
        out_path = os.path.join(OUT_DIR, "relay_scored.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"sim_thr": args.sim_thr, "n_scored": len(scored),
                       "n_missing": n_missing, "candidates": scored},
                      f, ensure_ascii=False, indent=1)
        sims = np.array([r["sim"] for r in scored])
        print(f"[score] {len(scored)} 候选已打分（缺嵌入 {n_missing}）")
        print(f"  sim 分布: p50={np.percentile(sims,50):.3f} p90={np.percentile(sims,90):.3f} "
              f"p99={np.percentile(sims,99):.3f} max={sims.max():.3f}")
        n_ge = (sims >= args.sim_thr).sum()
        print(f"  sim >= {args.sim_thr}: {n_ge} 对")
        # 按机位对
        from collections import Counter
        cnt = Counter(r["dir"] for r in scored if r["sim"] >= args.sim_thr)
        print("\n  sim>=门槛 的机位对分布:")
        for d, n in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"    {d}: {n}")
        print(f"\n== Top 30 候选中转（sim 高）==")
        for r in scored[:30]:
            print(f"  {r['sim']:.3f}  {r['a']} (h{r['a_h']:.0f}) -{r['dt']:>5.1f}s-> {r['b']} "
                  f"(h{r['b_h']:.0f})  {r['dir']}")
        print(f"\n[out] {out_path}")
        return

    if args.cmd == "links":
        # 跨机位接力候选：A 实体消失/出现 vs B 实体出现/消失 的时间门
        cams = ["316", "305", "318", "320", "323", "343"]
        events = {}
        for cam in cams:
            with open(os.path.join(OUT_DIR, f"events_{cam}.json"), encoding="utf-8") as f:
                data = json.load(f)
            rows = data["events"]
            if args.only_moving:
                rows = [r for r in rows if r["moving"]]
            events[cam] = rows
        pairs_out = []
        for i, ca in enumerate(cams):
            for cb in cams[i + 1:]:
                for ra in events[ca]:
                    for rb in events[cb]:
                        # 方向1: a 消失 -> b 出现
                        dt1 = rb["t_first"] - ra["t_last"]
                        # 方向2: b 消失 -> a 出现
                        dt2 = ra["t_first"] - rb["t_last"]
                        if args.dt_min <= dt1 <= args.dt_max:
                            pairs_out.append({
                                "a": f"{ca}:{ra['eid']}", "b": f"{cb}:{rb['eid']}",
                                "dir": f"{ca}->{cb}", "dt": round(dt1, 1),
                                "a_h": ra["h_med"], "b_h": rb["h_med"],
                                "a_end": [ra["u1"], ra["v1"]],
                                "b_start": [rb["u0"], rb["v0"]],
                                "a_tlast": ra["t_last"], "b_tfirst": rb["t_first"],
                            })
                        if args.dt_min <= dt2 <= args.dt_max:
                            pairs_out.append({
                                "a": f"{cb}:{rb['eid']}", "b": f"{ca}:{ra['eid']}",
                                "dir": f"{cb}->{ca}", "dt": round(dt2, 1),
                                "a_h": rb["h_med"], "b_h": ra["h_med"],
                                "a_end": [rb["u1"], rb["v1"]],
                                "b_start": [ra["u0"], ra["v0"]],
                                "a_tlast": rb["t_last"], "b_tfirst": ra["t_first"],
                            })
        out_path = os.path.join(OUT_DIR, "relay_candidates.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"dt_window": [args.dt_min, args.dt_max],
                       "only_moving": args.only_moving,
                       "n": len(pairs_out), "candidates": pairs_out},
                      f, ensure_ascii=False, indent=1)
        print(f"[links] 时间门候选共 {len(pairs_out)} 对 -> {out_path}")
        # 按机位对统计
        from collections import Counter
        cnt = Counter(p["dir"] for p in pairs_out)
        for d, n in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"  {d}: {n}")
        # 每对机位抽样最近 dt 的 top5
        for d in sorted(cnt):
            sub = sorted([p for p in pairs_out if p["dir"] == d], key=lambda x: x["dt"])
            print(f"\n== {d} top8（dt 最近）")
            for p in sub[:8]:
                print(f"  {p['a']} (h{p['a_h']:.0f}) -{p['dt']:>5.1f}s-> {p['b']} (h{p['b_h']:.0f})"
                      f"  a_end=({p['a_end'][0]:.2f},{p['a_end'][1]:.2f}) b_start=({p['b_start'][0]:.2f},{p['b_start'][1]:.2f})")
        return

    if args.cmd == "events":
        os.makedirs(OUT_DIR, exist_ok=True)
        for cam in ["316", "305", "318", "320", "323", "343"]:
            ents = load_entities(cam)
            rows = []
            for eid, (fr, u, v, h) in ents.items():
                dur = (fr[-1] - fr[0]) / FPS
                # 位移：脚点序列的总范围（≥0.05 归一化单位视为移动目标）
                span = float(np.hypot(u.max() - u.min(), v.max() - v.min()))
                rec = {
                    "eid": eid,
                    "t_first": round(T0_OFFSET[cam] + fr[0] / FPS, 2),
                    "t_last": round(T0_OFFSET[cam] + fr[-1] / FPS, 2),
                    "f_first": int(fr[0]), "f_last": int(fr[-1]),
                    "u0": round(float(u[0]), 4), "v0": round(float(v[0]), 4),
                    "u1": round(float(u[-1]), 4), "v1": round(float(v[-1]), 4),
                    "h_med": round(float(np.median(h)), 1),
                    "h_first": round(float(h[0]), 1),
                    "h_last": round(float(h[-1]), 1),
                    "dur_s": round(dur, 1),
                    "span": round(span, 3),
                    "moving": bool(span >= 0.05 and dur >= 1.0),
                }
                rows.append(rec)
            rows.sort(key=lambda r: r["t_first"])
            out_path = os.path.join(OUT_DIR, f"events_{cam}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"cam": cam, "t0_offset": T0_OFFSET[cam],
                           "n": len(rows), "events": rows}, f, ensure_ascii=False, indent=1)
            n_small = sum(1 for r in rows if r["h_med"] < 70)
            n_mov = sum(1 for r in rows if r["moving"])
            print(f"[{cam}] {len(rows)} 实体（小目标 {n_small}，移动 {n_mov}）-> {out_path}")
        return

    if args.cmd == "mapcover":
        os.makedirs(OUT_DIR, exist_ok=True)
        pts = load_train_points(args.pairs_csv, set(TRAIN_PAIRS))
        print(f"[map] 训练点 {len(pts)} 个（{len(TRAIN_PAIRS)} 对确认）")
        coef = fit_map(pts)
        # 训练残差
        A = np.array([[p[0], p[1]] for p in pts])
        B = np.array([[p[2], p[3]] for p in pts])
        pred = apply_map(coef, A[:, 0], A[:, 1])
        res = np.hypot(pred[:, 0] - B[:, 0], pred[:, 1] - B[:, 1])
        print(f"[map] 训练残差: mean={res.mean():.4f}  p90={np.percentile(res,90):.4f}  max={res.max():.4f}")

        # --- 网格覆盖 ---
        us = np.arange(0.0, 1.001, 0.05)
        vs = np.arange(0.0, 1.001, 0.05)
        U, V = np.meshgrid(us, vs)
        Ub, Vb = apply_map(coef, U.ravel(), V.ravel()).T
        Ub = Ub.reshape(U.shape)
        Vb = Vb.reshape(V.shape)
        inb = (Ub >= 0) & (Ub <= 1) & (Vb >= 0) & (Vb <= 1)
        print(f"\n[cover] {args.a} 网格 {U.size} 点 -> {args.b} 视野内 "
              f"{inb.sum()} 点（{100*inb.mean():.1f}%）")
        # 按 v 分层看
        print(f"\n  v 行（{args.a} 纵向位置，0=顶部/远，1=底部/近） -> 映射在 {args.b} 视野内比例")
        for i, v in enumerate(vs):
            frac = inb[i].mean()
            bar = "#" * int(round(frac * 30))
            print(f"  v={v:.2f}  {frac*100:5.1f}%  {bar}")

        # --- 318 轨迹脚点分布 ---
        foot = load_foot_trajs(args.a)
        print(f"\n[foot] {args.a} 轨迹 {len(foot)} 条")
        cats = {"small(<70px)": [], "mid(70-150)": [], "large(>=150)": []}
        for tid, (fr, u, v, h) in foot.items():
            med_h = np.median(h)
            med_u = np.median(u)
            med_v = np.median(v)
            pb = apply_map(coef, np.atleast_1d(med_u), np.atleast_1d(med_v))
            inb1 = (0 <= pb[0, 0] <= 1) and (0 <= pb[0, 1] <= 1)
            rec = (tid, med_u, med_v, med_h, pb[0, 0], pb[0, 1], inb1)
            if med_h < 70:
                cats["small(<70px)"].append(rec)
            elif med_h < 150:
                cats["mid(70-150)"].append(rec)
            else:
                cats["large(>=150)"].append(rec)

        for name, recs in cats.items():
            if not recs:
                continue
            n_in = sum(1 for r in recs if r[6])
            print(f"\n  {name}: {len(recs)} 条轨迹，脚点映射进 {args.b} 视野: "
                  f"{n_in} ({100*n_in/len(recs):.1f}%)")
            # 抽样打印 10 条
            for r in recs[:10]:
                print(f"    tid={r[0]:>4}  foot=({r[1]:.2f},{r[2]:.2f}) h={r[3]:.0f}px"
                      f" -> b=({r[4]:.2f},{r[5]:.2f}) {'IN' if r[6] else 'OUT'}")

        # --- 可视化 ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                       "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
                if os.path.exists(fp):
                    font_manager.fontManager.addfont(fp)
                    plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
                    break
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            ax = axes[0]
            ax.set_title(f"{args.a} 画面网格（绿=映射进 {args.b} 视野）")
            ax.scatter(U[inb], V[inb], s=30, c="#2ca02c", marker="o", label="in view")
            ax.scatter(U[~inb], V[~inb], s=30, c="#d62728", marker="x", label="out of view")
            # 轨迹脚点
            for name, color, size in [("small(<70px)", "#1f77b4", 12),
                                      ("mid(70-150)", "#ff7f0e", 12),
                                      ("large(>=150)", "#9467bd", 12)]:
                if cats[name]:
                    uu = [r[1] for r in cats[name]]
                    vv = [r[2] for r in cats[name]]
                    ax.scatter(uu, vv, s=size, c=color, alpha=0.5, label=f"tracks {name}")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(1.02, -0.02)
            ax.set_xlabel("u (0=左)")
            ax.set_ylabel("v (0=上/远)")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)

            ax = axes[1]
            ax.set_title(f"{args.b} 坐标系（映射点分布）")
            ax.scatter(Ub[inb], Vb[inb], s=30, c="#2ca02c", marker="o")
            ax.scatter(np.clip(Ub[~inb], -0.2, 1.2), np.clip(Vb[~inb], -0.2, 1.2),
                       s=30, c="#d62728", marker="x")
            for name, color in [("small(<70px)", "#1f77b4"), ("mid(70-150)", "#ff7f0e"),
                                ("large(>=150)", "#9467bd")]:
                if cats[name]:
                    bb = [(r[4], r[5]) for r in cats[name] if r[6]]
                    if bb:
                        ax.scatter([b[0] for b in bb], [b[1] for b in bb], s=12,
                                   c=color, alpha=0.5)
            ax.set_xlim(-0.25, 1.25)
            ax.set_ylim(1.25, -0.25)
            ax.set_xlabel("u'")
            ax.set_ylabel("v'")
            ax.grid(alpha=0.3)

            out_png = os.path.join(OUT_DIR, f"mapcover_{args.a}_{args.b}.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=110)
            print(f"\n[fig] {out_png}")
        except Exception as e:
            print(f"[fig] 可视化失败: {e}")


if __name__ == "__main__":
    main()
