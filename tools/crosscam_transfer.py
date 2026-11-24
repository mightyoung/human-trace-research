#!/usr/bin/env python
"""走廊↔茶水间 初步转移观察（非重叠机位"离开-出现"事件匹配）

用途：在 316（行政走廊通道）× 323（茶水间）之间做**描述性**转移观察——
      提取"消失/出现"事件 → 时间门筛选 → DINOv2 外观匹配 → 输出候选与证据图。
注意：非重叠机位、无身份真值（GT），本工具产出=候选观察，不是确认结论；
      正式"转移拓扑学习"需更长记录（数小时~数天）。

时间对齐：两路切片各有 OSD 起点偏差（316=08:55:04、323=08:55:18）。
  dt(316消失→323出现) = (g2 - f1 + 350) / 25   秒   [350 = 14s×25]
  dt(323消失→316出现) = (f2 - g1 - 350) / 25   秒

用法:
    python tools/crosscam_transfer.py \
      --a-jsonl output_crosscam/316_detections.jsonl --a-video ".../316_085500-090500.mp4" --a-wh 2560 1440 \
      --b-jsonl output_crosscam/323_detections.jsonl --b-video ".../323_085500-090500.mp4" --b-wh 1920 1080 \
      --out-dir output_crosscam/analysis
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscam_reid import Embedder, prep_crop  # noqa: E402

FPS = 25.0
OFF = 350  # 323_frame = 316_frame - 350 对应同一墙钟时刻


def load_tracks(jsonl_path):
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
                    t.setdefault(int(tids[i]), []).append((int(r["frame"]), [float(v) for v in b]))
    for k in t:
        t[k].sort()
    return t


def norm_pos(box, W, H):
    return ((box[0] + box[2]) / 2 / W, box[3] / H)


def extract_events(tracks, W, H, tag, min_n=5):
    """每条轨迹提取 出现/消失 两个事件（含方向估计）。"""
    apps, disps = [], []
    for tid, lst in tracks.items():
        if len(lst) < min_n:
            continue
        # 出现
        fs0, b0 = lst[0]
        k = min(4, len(lst) - 1)
        fs1, b1 = lst[k]
        p0, p1 = norm_pos(b0, W, H), norm_pos(b1, W, H)
        apps.append({"tag": tag, "tid": tid, "f": fs0, "n": len(lst),
                     "u": round(p0[0], 3), "v": round(p0[1], 3),
                     "du": round(p1[0] - p0[0], 3), "dv": round(p1[1] - p0[1], 3)})
        # 消失
        fe1, b2 = lst[-1]
        fe0, b3 = lst[-1 - k]
        p2, p3 = norm_pos(b2, W, H), norm_pos(b3, W, H)
        disps.append({"tag": tag, "tid": tid, "f": fe1, "n": len(lst),
                      "u": round(p2[0], 3), "v": round(p2[1], 3),
                      "du": round(p2[0] - p3[0], 3), "dv": round(p2[1] - p3[1], 3)})
    return apps, disps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-jsonl", required=True)   # 316
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--a-wh", nargs=2, type=int, default=[2560, 1440])
    ap.add_argument("--b-jsonl", required=True)   # 323
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--b-wh", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--dt-min", type=float, default=0.5)
    ap.add_argument("--dt-max", type=float, default=45.0)
    ap.add_argument("--max-pairs-per-dir", type=int, default=40,
                    help="每方向保留的最大候选对数（控制外观编码算力）")
    ap.add_argument("--spread", action="store_true",
                    help="按 dt 均匀取样（默认取 dt 最小的前 N——这会把慢速/真实转移排除在外）")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ta = load_tracks(args.a_jsonl)
    tb = load_tracks(args.b_jsonl)
    A_W, A_H = args.a_wh
    B_W, B_H = args.b_wh
    apps_a, disps_a = extract_events(ta, A_W, A_H, "316")
    apps_b, disps_b = extract_events(tb, B_W, B_H, "323")
    print(f"[events] 316: 出现 {len(apps_a)} / 消失 {len(disps_a)}；"
          f"323: 出现 {len(apps_b)} / 消失 {len(disps_b)}")

    # 时间门：316 消失 -> 323 出现
    cand_ab = []
    for d in disps_a:
        for a in apps_b:
            dt = (a["f"] - d["f"] + OFF) / FPS
            if args.dt_min <= dt <= args.dt_max:
                cand_ab.append({"dirn": "316->323", "disp": d, "app": a, "dt": round(dt, 1)})
    # 323 消失 -> 316 出现
    cand_ba = []
    for d in disps_b:
        for a in apps_a:
            dt = (a["f"] - d["f"] - OFF) / FPS
            if args.dt_min <= dt <= args.dt_max:
                cand_ba.append({"dirn": "323->316", "disp": d, "app": a, "dt": round(dt, 1)})
    print(f"[time gate] 316->323 候选 {len(cand_ab)}；323->316 候选 {len(cand_ba)}")
    # 择优截断：默认取 dt 升序前 N（近距转移更可信）；--spread 则按 dt 均匀取样
    cand_ab.sort(key=lambda c: c["dt"])
    cand_ba.sort(key=lambda c: c["dt"])
    if args.spread:
        def _pick(lst):
            if len(lst) <= args.max_pairs_per_dir:
                return lst
            idx = np.linspace(0, len(lst) - 1, args.max_pairs_per_dir).astype(int)
            return [lst[i] for i in idx]
        cand_ab, cand_ba = _pick(cand_ab), _pick(cand_ba)
    else:
        cand_ab = cand_ab[: args.max_pairs_per_dir]
        cand_ba = cand_ba[: args.max_pairs_per_dir]
    print(f"[truncate] 保留 316->323 {len(cand_ab)}；323->316 {len(cand_ba)}"
          f"（{'dt 均匀取样' if args.spread else 'dt 升序前 N'}）")

    cand = cand_ab + cand_ba
    if not cand:
        print("no candidates")
        return
    # 时间门优先：完整保留，但外观编码只做每方向 dt 最小的 N 与全体（上限 120 对）
    cand_all = cand
    print(f"[appearance] 编码 {len(cand_all)} 个候选对的轨迹端点 ...")

    emb = Embedder()

    def end_embed(video, tracks, W, H, tag, tid, which, k=3):
        lst = tracks[tid]
        frames = [f for f, _ in lst]
        if which == "appear":
            idxs = np.linspace(0, max(0, min(len(lst) - 1, int(1.2 * FPS / 5))), k).astype(int)
        else:
            idxs = np.linspace(max(0, len(lst) - 1 - int(1.2 * FPS / 5)), len(lst) - 1, k).astype(int)
        cap = cv2.VideoCapture(video)
        vecs = []
        for i in idxs:
            fnum, box = lst[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = prep_crop(frame, box)
            if rgb is None:
                continue
            vecs.append(emb.embed_batch([rgb])[0])
        cap.release()
        if not vecs:
            return None
        m = np.mean(vecs, axis=0)
        return m / (np.linalg.norm(m) + 1e-9)

    cache = {}

    def get_emb(side, tid, which):
        key = (side, tid, which)
        if key not in cache:
            if side == "A":
                cache[key] = end_embed(args.a_video, ta, A_W, A_H, "316", tid, which)
            else:
                cache[key] = end_embed(args.b_video, tb, B_W, B_H, "323", tid, which)
        return cache[key]

    for c in cand_all:
        if c["dirn"] == "316->323":
            e1 = get_emb("A", c["disp"]["tid"], "disappear")
            e2 = get_emb("B", c["app"]["tid"], "appear")
        else:
            e1 = get_emb("B", c["disp"]["tid"], "disappear")
            e2 = get_emb("A", c["app"]["tid"], "appear")
        c["cos"] = None if (e1 is None or e2 is None) else round(float(np.dot(e1, e2)), 3)
    cand_all.sort(key=lambda c: -(c["cos"] if c["cos"] is not None else -9))
    print(f"[appearance] 完成，最高余弦: {cand_all[0]['cos']}")
    for c in cand_all[: args.top_k]:
        d_side = "316" if c["disp"]["tag"] == "316" else "323"
        a_side = "316" if c["app"]["tag"] == "316" else "323"
        print(f"  {c['dirn']}  {d_side}#{c['disp']['tid']} -> {a_side}#{c['app']['tid']}"
              f"  dt={c['dt']}s  cos={c['cos']}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, "transfer_events.json")
    Path(out_json).write_text(json.dumps(
        {"events": {"a_app": len(apps_a), "a_disp": len(disps_a),
                    "b_app": len(apps_b), "b_disp": len(disps_b)},
         "candidates": cand_all}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {out_json}")

    # ---- 时间线图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(12, 3.2), dpi=120)
        for d in disps_a:
            ax.plot(d["f"] / FPS / 60, 2, "v", color="#d62728", ms=4, alpha=0.6)
        for a in apps_b:
            ax.plot((a["f"] - OFF) / FPS / 60, 1.85, "^", color="#ff7f0e", ms=4, alpha=0.6)
        for d in disps_b:
            ax.plot((d["f"] + OFF) / FPS / 60, 1, "v", color="#9467bd", ms=4, alpha=0.6)
        for a in apps_a:
            ax.plot(a["f"] / FPS / 60, 0.85, "^", color="#1f77b4", ms=4, alpha=0.6)
        for c in cand_all[: args.top_k]:
            if c["cos"] is None:
                continue
            if c["dirn"] == "316->323":
                x1 = c["disp"]["f"] / FPS / 60
                x2 = (c["app"]["f"] - OFF) / FPS / 60
            else:
                x1 = (c["disp"]["f"] + OFF) / FPS / 60
                x2 = c["app"]["f"] / FPS / 60
            ax.plot([x1, x2], [2, 1.85] if c["dirn"] == "316->323" else [1, 0.85],
                    "-", color="#2ca02c", lw=1.0, alpha=min(0.9, max(0.25, (c["cos"] or 0))))
        ax.set_yticks([2, 1.85, 1, 0.85])
        ax.set_yticklabels(["316 消失", "323 出现", "323 消失", "316 出现"], fontsize=9)
        ax.set_xlabel("时间（分钟，统一到 316 时间轴）", fontsize=9)
        ax.set_ylim(0.6, 2.3)
        ax.set_title("走廊(316) ↔ 茶水间(323) 事件时间线（绿线=外观匹配 Top15 候选）", fontsize=10)
        ax.grid(alpha=0.25, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "transfer_timeline.png"), facecolor="white")
        print(f"[out] {os.path.join(args.out_dir, 'transfer_timeline.png')}")
    except Exception as e:  # noqa: BLE001
        print("[warn] timeline fig failed:", str(e)[:160])

    # ---- Top 候选裁剪证据图 ----
    try:
        def grab(video, fnum):
            cap = cv2.VideoCapture(video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fnum))
            ok, fr = cap.read()
            cap.release()
            return fr if ok else None

        rows = []
        for c in cand_all[:6]:
            if c["cos"] is None:
                continue
            if c["dirn"] == "316->323":
                vd, td, wd, hd = args.a_video, ta, A_W, A_H
                va, tt, wa, ha = args.b_video, tb, B_W, B_H
                d_ev, a_ev = c["disp"], c["app"]
            else:
                vd, td, wd, hd = args.b_video, tb, B_W, B_H
                va, tt, wa, ha = args.a_video, ta, A_W, A_H
                d_ev, a_ev = c["disp"], c["app"]
            fd, bd = td[d_ev["tid"]][-1]
            fa_, ba_ = tt[a_ev["tid"]][0]
            frd = grab(vd, fd)
            fra = grab(va, fa_)
            if frd is None or fra is None:
                continue
            cd = prep_crop(frd, bd)
            ca = prep_crop(fra, ba_)
            if cd is None or ca is None:
                continue

            def fit(rgb, H=260):
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                s = H / bgr.shape[0]
                return cv2.resize(bgr, (max(1, int(bgr.shape[1] * s)), H))

            cd2, ca2 = fit(cd), fit(ca)
            gap = np.full((260, 14, 3), 255, np.uint8)
            body = np.hstack([cd2, gap, ca2])
            bar = np.full((30, body.shape[1], 3), 252, np.uint8)
            txt = (f"{c['dirn']}  dt={c['dt']}s  cos={c['cos']}  "
                   f"[{d_ev['tid']}@{d_ev['f']}] -> [{a_ev['tid']}@{a_ev['f']}]")
            cv2.putText(bar, txt, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            rows.append(np.vstack([bar, body]))
        if rows:
            W = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
            vg = np.full((10, W, 3), 255, np.uint8)
            out = rows[0]
            for r in rows[1:]:
                out = np.vstack([out, vg, r])
            head = np.full((36, out.shape[1], 3), 252, np.uint8)
            cv2.putText(head, "transfer candidates: left=last frame of disappearing track, right=first frame of appearing track",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
            out = np.vstack([head, out])
            p = os.path.join(args.out_dir, "transfer_top_pairs.png")
            cv2.imwrite(p, out)
            print(f"[out] {p}")
    except Exception as e:  # noqa: BLE001
        print("[warn] pairs fig failed:", str(e)[:160])


if __name__ == "__main__":
    main()
