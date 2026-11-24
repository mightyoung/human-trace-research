#!/usr/bin/env python
"""深度方向信号验证实验（318×320 跨画面）—— 对应 docs/multicam-depth-spatial-survey.md 实验协议

思路：
  1) 用 MiDaS_small 对每帧算相对深度图（逆深度：值大 = 离该相机近）；
  2) 对每个已判读轨迹对，在其"共现窗口"内采样两路落脚点的深度序列（自底边中央 patch 取中值）；
  3) 318/320 为"对望"布局假设（人远离 A ⇔ 靠近 B）：
       预注册判据 score = -(slopeA * slopeB)      （归一化深度趋势的斜率积，越大越像同人）
       辅助判据 score_corr = -corr(dA, dB)
  4) 在 8 确认 / 22 否决 判读对上计算 AUC；含"运动量门限"筛除双方都静止的窗口。

用法:
    python tools/crosscam_depth.py \
        --a-video "media-resource/.../318_085500-090500.mp4" \
        --b-video "media-resource/.../320_085500-090500.mp4" \
        --a-jsonl output_crosscam/318_detections.jsonl \
        --b-jsonl output_crosscam/320_detections.jsonl \
        --out-dir output_crosscam/analysis
复算（不重跑视频/模型）:
    python tools/crosscam_depth.py --recompute-from output_crosscam/analysis/depth_series.json \
        --out-dir output_crosscam/analysis [--motion-min 0.05]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscam_spatiotemporal import POSITIVE, NEGATIVE, SHIFT  # noqa: E402

FPS = 25
A_W, A_H = 1920, 1080
B_W, B_H = 2560, 1440

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# ---------------------------------------------------------------- model
def _install_offline_hub_patch():
    """torch.hub 解析 GitHub 仓库名时优先使用本地缓存目录（网络中断时仍可加载）。

    背景：MiDaS 构建时内部会 torch.hub.load('rwightman/gen-efficientnet-pytorch', ...)，
    其仓库解析默认 `urlopen(github.com/...)` 判断分支；离线时抛 RemoteDisconnected，
    而 torch 自带的"查缓存回退"只覆盖 URLError，不触发 → 加载整体失败（2026-10-12 实测）。
    """
    import torch.hub as _th
    if getattr(_th, "_offline_cache_patch", False):
        return
    _orig = _th._parse_repo_info

    def _offline_parse(github):
        if ":" in github:  # 显式 ref，交给原逻辑（也会走缓存）
            return _orig(github)
        owner, name = github.split("/", 1)
        for ref in ("main", "master"):
            candidate = os.path.join(_th.get_dir(), f"{owner}_{name}_{ref}")
            if os.path.isdir(candidate):
                return owner, name, ref
        return _orig(github)

    _th._parse_repo_info = _offline_parse
    _th._offline_cache_patch = True


def load_midas():
    import torch
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    repo = os.path.join(hub_dir, "intel-isl_MiDaS_master")
    torch.hub.set_dir(hub_dir)
    _install_offline_hub_patch()
    if os.path.isdir(repo):
        try:
            m = torch.hub.load(repo, "MiDaS_small", source="local",
                               trust_repo=True, verbose=False)
            m.eval()
            print("[midas] loaded via local repo (no network)")
            return m
        except Exception as e:  # noqa: BLE001
            print("[midas] local hub load failed:", str(e)[:300])
    print("[midas] trying github spec (weights cache hit expected)...")
    m = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                       trust_repo=True, verbose=False)
    m.eval()
    return m


def depth_batch(model, frames_bgr):
    """批量：BGR 帧列表 -> 每帧原分辨率的相对深度图（np.float32）"""
    import torch
    xs = []
    for fr in frames_bgr:
        img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        xs.append(torch.from_numpy(x).permute(2, 0, 1))
    batch = torch.stack(xs)
    batch = (batch - torch.from_numpy(MEAN)) / torch.from_numpy(STD)
    with torch.no_grad():
        d = model(batch)
    if isinstance(d, dict):
        d = d.get("disp", list(d.values())[0])
    if d.dim() == 4:
        d = d.squeeze(1)
    h, w = frames_bgr[0].shape[:2]
    d = torch.nn.functional.interpolate(
        d.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze(1)
    return d.numpy()


def foot_depth(dmap, box):
    """落脚点深度：框底部中央 patch 的中值。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 10:
        return None
    cx = (x1 + x2) / 2.0
    px = max(3, int(w * 0.18))
    py = max(3, int(h * 0.10))
    xx1, xx2 = int(cx - px), int(cx + px)
    yy1, yy2 = int(y2 - 1.6 * py), int(y2)
    H, W = dmap.shape
    xx1, xx2 = max(0, xx1), min(W, max(1, xx2))
    yy1, yy2 = max(0, yy1), min(H, max(1, yy2))
    if xx2 - xx1 < 2 or yy2 - yy1 < 2:
        return None
    return float(np.median(dmap[yy1:yy2, xx1:xx2]))


# ---------------------------------------------------------------- data
def load_tracks_raw(path):
    t = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fr = int(r["frame"])
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((fr, [float(v) for v in b]))
    for k in t:
        t[k].sort()
    return t


def process_camera(video, need, model, batch=8):
    """need: {frame: [(key, box), ...]} -> {key: depth}"""
    result = {}
    frames_sorted = sorted(need.keys())
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    next_idx = 0
    buf_frames, buf_keys = [], []
    n_done, n_fail = 0, 0

    def flush():
        nonlocal n_done
        if not buf_frames:
            return
        try:
            dmaps = depth_batch(model, buf_frames)
        except Exception as e:  # noqa: BLE001
            print("  [warn] depth batch failed:", str(e)[:120])
            return
        for i, keys in enumerate(buf_keys):
            for key, box in keys:
                v = foot_depth(dmaps[i], box)
                if v is not None:
                    result[key] = v
        n_done += len(buf_frames)
        buf_frames.clear()
        buf_keys.clear()

    for fi in frames_sorted:
        while next_idx < fi:
            if not cap.grab():
                break
            next_idx += 1
        ret, frame = cap.read()
        if not ret:
            n_fail += 1
            continue
        next_idx += 1
        buf_frames.append(frame)
        buf_keys.append(need[fi])
        if len(buf_frames) >= batch:
            flush()
    flush()
    cap.release()
    print(f"  [video] {os.path.basename(video)}: {n_done} depth frames, {n_fail} read-fail")
    return result


# ---------------------------------------------------------------- stats
def auc_score(rows, key):
    P = [r[key] for r in rows if r["lab"] == "P" and r.get(key) is not None and r.get("usable")]
    N = [r[key] for r in rows if r["lab"] == "N" and r.get(key) is not None and r.get("usable")]
    if not P or not N:
        return None, P, N
    c = 0.0
    for p in P:
        for n in N:
            if p > n:
                c += 1
            elif p == n:
                c += 0.5
    return c / (len(P) * len(N)), P, N


def compute_rows(series, motion_min):
    rows = []
    for rec in series:
        lab, a, b = rec["lab"], rec["a"], rec["b"]
        t = np.asarray(rec["t"], dtype=float)
        dA = np.asarray(rec["dA"], dtype=float)
        dB = np.asarray(rec["dB"], dtype=float)
        row = {"lab": lab, "a": a, "b": b, "n": len(t),
               "slopeA": None, "slopeB": None, "score": None,
               "score_corr": None, "motion": None, "usable": False}
        if len(t) >= 6 and np.all(dA > 0) and np.all(dB > 0):
            nA = dA / np.mean(dA)
            nB = dB / np.mean(dB)
            mA = float(np.percentile(nA, 90) - np.percentile(nA, 10))
            mB = float(np.percentile(nB, 90) - np.percentile(nB, 10))
            row["motion"] = round(max(mA, mB), 4)
            row["motionA"], row["motionB"] = round(mA, 4), round(mB, 4)
            tt = t - t[0]
            sA = float(np.polyfit(tt, nA, 1)[0])
            sB = float(np.polyfit(tt, nB, 1)[0])
            row["slopeA"], row["slopeB"] = round(sA, 5), round(sB, 5)
            row["score"] = round(-(sA * sB), 6)
            if len(t) >= 8 and np.std(nA) > 1e-9 and np.std(nB) > 1e-9:
                row["score_corr"] = round(-float(np.corrcoef(nA, nB)[0, 1]), 4)
            row["usable"] = row["motion"] >= motion_min
        rows.append(row)
    return rows


def print_report(rows, motion_min):
    P = [r for r in rows if r["lab"] == "P"]
    N = [r for r in rows if r["lab"] == "N"]
    uP = [r for r in P if r["usable"]]
    uN = [r for r in N if r["usable"]]
    print(f"\n[motion gate >= {motion_min}] usable: {len(uP)}/{len(P)} P, {len(uN)}/{len(N)} N")
    print(f"\n{'pair':>10} {'n':>4} {'motion':>7} {'slopeA':>8} {'slopeB':>8} "
          f"{'score':>9} {'corr_s':>7} {'use':>3}  lab")
    for r in sorted(rows, key=lambda r: (r["lab"], -(r["score"] or -9))):
        f = lambda v, w=6: (f"{v:.5f}" if isinstance(v, float) else "  None").rjust(w)
        print(f"A{r['a']:>4}:{r['b']:<4} {r['n']:>4} {f(r['motion'], 7)} "
              f"{f(r['slopeA'], 8)} {f(r['slopeB'], 8)} {f(r['score'], 9)} "
              f"{f(r['score_corr'], 7)} {'Y' if r['usable'] else '.':>3}  {r['lab']}")
    for key in ("score", "score_corr"):
        auc, pv, nv = auc_score(rows, key)
        if auc is not None:
            print(f"\nAUC[{key}] (usable) = {auc:.3f}   "
                  f"min(P)={min(pv):.5f}  max(N)={max(nv):.5f}  "
                  f"P>0: {sum(1 for x in pv if x > 0)}/{len(pv)}  "
                  f"N>0: {sum(1 for x in nv if x > 0)}/{len(nv)}")

    # 双动子集（A/B 两侧运动幅度均 >= 0.05，排除单侧静止）——本实验中 corr 变体在此最优
    both = [r for r in rows if r["usable"] and (r.get("motionA") or 0) >= 0.05
            and (r.get("motionB") or 0) >= 0.05]

    def _auc_sub(sub, k):
        pv = [r[k] for r in sub if r["lab"] == "P" and r.get(k) is not None]
        nv = [r[k] for r in sub if r["lab"] == "N" and r.get(k) is not None]
        if not pv or not nv:
            return None
        c = 0.0
        for p in pv:
            for n in nv:
                c += 1 if p > n else (0.5 if p == n else 0)
        return c / (len(pv) * len(nv))

    if both:
        n_p = sum(1 for r in both if r["lab"] == "P")
        n_n = sum(1 for r in both if r["lab"] == "N")
        a1 = _auc_sub(both, "score")
        a2 = _auc_sub(both, "score_corr")
        print(f"both-move subset ({n_p}P/{n_n}N): "
              f"AUC[score]={a1 if a1 is None else round(a1, 3)}  "
              f"AUC[score_corr]={a2 if a2 is None else round(a2, 3)}")



def make_figs(rows, series, out_dir, motion_min):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- fig1: per-pair score
    use = [r for r in rows if r["usable"] and r["score"] is not None]
    use.sort(key=lambda r: -r["score"])
    names = [f"{r['a']}:{r['b']}" for r in use]
    vals = [r["score"] for r in use]
    cols = ["#2ca02c" if r["lab"] == "P" else "#d62728" for r in use]
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
    bars = ax.bar(range(len(use)), vals, color=cols, alpha=0.88)
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(range(len(use)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8.5)
    auc, _, _ = auc_score(rows, "score")
    ax.set_title(f"Depth-direction score  score=-(slopeA*slopeB)  "
                 f"green=confirmed same person / red=rejected  |  AUC={auc:.3f}", fontsize=11)
    ax.set_ylabel("score (facing-camera hypothesis)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_depth_direction.png"), facecolor="white")
    plt.close(fig)
    print("[out] fig_depth_direction.png")

    # --- fig2: example trends
    recs = {}
    for rec in series:
        recs[(rec["a"], rec["b"])] = rec
    examples = [r for r in rows if r["usable"] and r["lab"] == "P" and r["score"] is not None]
    examples.sort(key=lambda r: -r["score"])
    picks = [(r["a"], r["b"], "P", r["score"]) for r in examples[:3]]
    nneg = [r for r in rows if r["usable"] and r["lab"] == "N" and r["score"] is not None]
    nneg.sort(key=lambda r: -r["score"])
    picks += [(r["a"], r["b"], "N", r["score"]) for r in nneg[:1]]
    if picks:
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=120)
        for ax, (a, b, lab, sc) in zip(axes.ravel(), picks):
            rec = recs.get((a, b))
            if rec is None:
                continue
            t = np.asarray(rec["t"]) - rec["t"][0]
            dA = np.asarray(rec["dA"]); dB = np.asarray(rec["dB"])
            ax.plot(t, dA / dA.mean(), color="#1f77b4", lw=1.8, label="A(318) depth/mean")
            ax.plot(t, dB / dB.mean(), color="#ff7f0e", lw=1.8, label="B(320) depth/mean")
            ax.set_title(f"A{a} x B{b}  [{lab}]  score={sc:.4f}", fontsize=10)
            ax.set_xlabel("t (s, within co-visible window)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle("Foot-point relative depth trends (MiDaS) — facing cameras expect opposite trends",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_depth_trends.png"), facecolor="white")
        plt.close(fig)
        print("[out] fig_depth_trends.png")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-video")
    ap.add_argument("--b-video")
    ap.add_argument("--a-jsonl")
    ap.add_argument("--b-jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=60)
    ap.add_argument("--motion-min", type=float, default=0.05)
    ap.add_argument("--recompute-from", help="已有 depth_series.json，跳过视频处理")
    ap.add_argument("--only-pairs", help="逗号分隔 a:b，限抽烟测试")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("output_crosscam/analysis", exist_ok=True)

    if args.recompute_from:
        series = json.loads(Path(args.recompute_from).read_text(encoding="utf-8"))
        print(f"[load] {len(series)} pair series from {args.recompute_from}")
    else:
        for v in (args.a_video, args.b_video, args.a_jsonl, args.b_jsonl):
            if not v:
                raise SystemExit("need --a-video/--b-video/--a-jsonl/--b-jsonl")
        ta = load_tracks_raw(args.a_jsonl)
        tb = load_tracks_raw(args.b_jsonl)
        print(f"[data] A tracks={len(ta)}  B tracks={len(tb)}")

        pairs_all = [("P", a, b) for (a, b) in POSITIVE] + \
                    [("N", a, b) for (a, b) in NEGATIVE]
        if args.only_pairs:
            want = {tuple(int(x) for x in s.split(":")) for s in args.only_pairs.split(",")}
            pairs_all = [(l, a, b) for (l, a, b) in pairs_all if (a, b) in want]

        pair_frames, needA, needB, meta = {}, {}, {}, {}
        for pi, (lab, a, b) in enumerate(pairs_all):
            if a not in ta or b not in tb:
                print(f"  [skip] {a}:{b} missing track")
                continue
            bm_a = dict(ta[a])
            bm_b = dict(tb[b])
            frames = [f for f, _ in ta[a] if (f + SHIFT) in bm_b]
            if len(frames) < 6:
                print(f"  [skip] {a}:{b} co-visible frames {len(frames)}")
                continue
            if len(frames) > args.max_samples:
                sel = np.linspace(0, len(frames) - 1, args.max_samples).astype(int)
                frames = [frames[i] for i in sel]
            meta[(a, b)] = (lab, pi)
            pair_frames[(a, b)] = [(f, bm_a[f], f + SHIFT, bm_b[f + SHIFT]) for f in frames]
            for fa, ba, fb, bb in pair_frames[(a, b)]:
                needA.setdefault(fa, []).append(((a, b, "A", fa), ba))
                needB.setdefault(fb, []).append(((a, b, "B", fb), bb))

        print(f"[plan] {len(pair_frames)} pairs, depth frames A={len(needA)} B={len(needB)}")

        model = load_midas()
        t0 = time.time()
        print("[run] A video ...")
        resA = process_camera(args.a_video, needA, model)
        print("[run] B video ...")
        resB = process_camera(args.b_video, needB, model)
        print(f"[run] depth done in {time.time() - t0:.1f}s")

        series = []
        for (a, b), frames in pair_frames.items():
            lab, pi = meta[(a, b)]
            t, dA, dB = [], [], []
            for fa, ba, fb, bb in frames:
                va = resA.get((a, b, "A", fa))
                vb = resB.get((a, b, "B", fb))
                if va is None or vb is None:
                    continue
                t.append(fa / FPS)
                dA.append(va)
                dB.append(vb)
            if len(t) >= 6:
                series.append({"lab": lab, "a": a, "b": b,
                               "t": [round(x, 2) for x in t],
                               "dA": [round(x, 3) for x in dA],
                               "dB": [round(x, 3) for x in dB]})
        (Path(args.out_dir) / "depth_series.json").write_text(
            json.dumps(series, ensure_ascii=False), encoding="utf-8")
        print(f"[out] depth_series.json  ({len(series)} pairs)")

    rows = compute_rows(series, args.motion_min)
    print_report(rows, args.motion_min)
    (Path(args.out_dir) / "depth_scores.json").write_text(
        json.dumps({"motion_min": args.motion_min, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] depth_scores.json")
    try:
        make_figs(rows, series, args.out_dir, args.motion_min)
    except Exception as e:  # noqa: BLE001
        print("[warn] figure failed:", str(e)[:200])


if __name__ == "__main__":
    main()
