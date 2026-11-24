#!/usr/bin/env python
"""地板倒影的"深度签名"验证 + 深度门原型（带深度的动态掩膜核心实验）。

物理假设：
  真人站立/行走——框内各部位深度≈恒定（人体是一个等深直立物）；
  地板倒影——倒影"躺"在地板平面上，框内自上而下深度显著变近（越往下 = 地面上越靠近相机）。
  → 框内"下半/上半深度比" r2 或 "下1/3/上1/3 比" r3 应为：倒影>1（显著），真人≈1。

数据：spread 集经"兄弟轨迹审计"确认的 29 条倒影轨迹 vs 28 条源（真人）轨迹；
另附 5 条用户确认的墙玻璃镜像轨迹做参照（预期 r≈1——它们不在地板上，由 P1 脚点残差负责）。

用法：
  python tools/depth_gate_test.py \
    --labels output_crosscam/analysis/spread/depth_filter/_floorref_labels.json \
    --a-jsonl output_crosscam/316_detections.jsonl --b-jsonl output_crosscam/323_detections.jsonl \
    --a-video "media-resource/.../316_...mp4" --b-video "media-resource/.../323_...mp4" \
    --out-dir output_crosscam/analysis/depth_gate
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscam_depth import load_midas, depth_batch  # noqa: E402

WALL_MIRROR = {("316", 101), ("316", 74), ("316", 96), ("323", 164), ("323", 133)}
FLOOR_COEF = None  # main() 里从 P1 标定文件加载


def load_tracks(path):
    t = {}
    with open(path, encoding="utf-8") as f:
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


def sample_frames(lst, k=5):
    n = len(lst)
    if n == 0:
        return []
    idxs = np.linspace(0.15, 0.85, k)
    out = []
    seen = set()
    for x in idxs:
        i = int(round(x * (n - 1)))
        if i not in seen:
            seen.add(i)
            out.append(lst[i])
    return out


def box_depth_feats(dmap, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = dmap.shape
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x2 - x1 < 6 or y2 - y1 < 12:
        return None
    reg = dmap[y1:y2, x1:x2]
    hh = reg.shape[0]
    half = hh // 2
    top = float(np.median(reg[:half]))
    bot = float(np.median(reg[hh - half:]))
    t3 = max(2, hh // 3)
    top3 = float(np.median(reg[:t3]))
    bot3 = float(np.median(reg[hh - t3:]))
    colm = reg.mean(axis=1)
    rows = np.arange(hh) / max(1, hh - 1)
    a = np.polyfit(rows, colm, 1)
    # 同高度行对照比 r_h：框内深度 / 同排两侧（框外）深度
    # 倒影落在地板上 -> 与两侧地板深度连续（≈1）；真人（等深直立物）-> 比同排地板近（<1）
    margin = max(4, (x2 - x1) // 4)
    rowslice = dmap[y1:y2]
    left = rowslice[:, :max(0, x1 - margin)]
    right = rowslice[:, min(W, x2 + margin):]
    parts = [p for p in (left, right) if p.shape[1] >= 3]
    r_h = None
    if parts:
        side = np.concatenate(parts, axis=1)
        r_h = float(np.median(reg) / (float(np.median(side)) + 1e-9))
    # 脚点地板残差（P1 判据）：脚点深度 vs 地板曲线
    from crosscam_depth import foot_depth as _fd
    foot_resid = None
    d_foot_n = None
    try:
        d_foot = _fd(dmap, [x1, y1, x2, y2])
    except Exception:  # noqa: BLE001
        d_foot = None
    if d_foot is not None:
        fm = float(np.median(dmap)) + 1e-9
        d_foot_n = float(d_foot) / fm
        if FLOOR_COEF is not None:
            v_foot = float(y2) / H
            dp = float(np.polyval(FLOOR_COEF, v_foot))
            if dp > 1e-6:
                foot_resid = abs(d_foot_n - dp) / dp
    return {
        "r2": bot / (top + 1e-9),
        "r3": bot3 / (top3 + 1e-9),
        "slope_n": float(a[0] / (float(np.median(reg)) + 1e-6)),
        "d_med": float(np.median(reg)),
        "r_h": r_h,
        "foot_resid": foot_resid,
        "d_foot_n": d_foot_n,
    }


def process(video, need, model, batch=8):
    out = {}
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    next_idx = 0
    buf_fr, buf_keys = [], []

    def flush():
        if not buf_fr:
            return
        dmaps = depth_batch(model, buf_fr)
        for i, keys in enumerate(buf_keys):
            for key, box in keys:
                f = box_depth_feats(dmaps[i], box)
                if f:
                    out[key] = f
        buf_fr.clear()
        buf_keys.clear()

    for fi in sorted(need.keys()):
        while next_idx < fi:
            if not cap.grab():
                break
            next_idx += 1
        ok, fr = cap.read()
        if not ok:
            continue
        next_idx += 1
        buf_fr.append(fr)
        buf_keys.append(need[fi])
        if len(buf_fr) >= batch:
            flush()
    flush()
    cap.release()
    return out


def auc_of(pos, neg):
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    c = 0.0
    for p in pos:
        for n in neg:
            c += 1 if p > n else (0.5 if p == n else 0)
    return c / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lab = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    refl = [tuple(x) for x in lab["reflection"]]
    src = [tuple(x) for x in lab["source"]]

    # 加载 P1 地板曲线标定
    global FLOOR_COEF
    p1_path = Path("output_crosscam/analysis/reflection_p1/reflection_depth_scores.json")
    if p1_path.exists():
        FLOOR_COEF = json.loads(p1_path.read_text(encoding="utf-8"))["floor_coef_v_d"]
        print(f"[calib] floor_coef={[round(c,3) for c in FLOOR_COEF]}")

    ta = load_tracks(args.a_jsonl)
    tb = load_tracks(args.b_jsonl)

    groups = {"reflection": refl, "source": src,
              "wall_mirror": [x for x in WALL_MIRROR]}
    need_a, need_b = {}, {}
    keys = []  # (group, tag, tid, key)
    for grp, items in groups.items():
        for tag, tid in items:
            tracks = ta if tag == "316" else tb
            lst = tracks.get(int(tid), [])
            if len(lst) < 4:
                print(f"[skip] {grp} {tag}#{tid} n={len(lst)}")
                continue
            for f, box in sample_frames(lst, args.samples):
                key = (tag, int(tid), int(f))
                (need_a if tag == "316" else need_b).setdefault(int(f), []).append((key, box))
                keys.append((grp, tag, int(tid), key))

    print(f"[plan] instances={len(keys)}  frames A={len(need_a)} B={len(need_b)}", flush=True)
    model = load_midas()
    print("[run] depth A(316) ...", flush=True)
    fa = process(args.a_video, need_a, model)
    print("[run] depth B(323) ...", flush=True)
    fb = process(args.b_video, need_b, model)
    allf = {**fa, **fb}

    # 轨迹级聚合（各特征取中值）
    per_track = {}  # (grp, tag, tid) -> {feat: median}
    for grp, tag, tid, key in keys:
        f = allf.get(key)
        if not f:
            continue
        d = per_track.setdefault((grp, tag, tid), {})
        for k, v in f.items():
            d.setdefault(k, []).append(v)
    for k, d in per_track.items():
        for kk in list(d.keys()):
            arr = [x for x in d[kk] if x is not None]
            d[kk] = float(np.median(arr)) if arr else None

    # AUC
    def vals(grp, feat):
        return [d[feat] for (g, t, i), d in per_track.items() if g == grp]

    print(f"\n[data] tracks: reflection={sum(1 for g,_,_ in per_track if g=='reflection')}, "
          f"source={sum(1 for g,_,_ in per_track if g=='source')}, "
          f"wall_mirror={sum(1 for g,_,_ in per_track if g=='wall_mirror')}")
    print(f"\n{'feature':<10} {'AUC':>7}  refl_med  src_med  wall_med")
    results = {}
    for feat in ("r2", "r3", "slope_n", "d_med", "r_h", "foot_resid"):
        pv, nv, wv = vals("reflection", feat), vals("source", feat), vals("wall_mirror", feat)
        a = auc_of(pv, nv)
        med = lambda x: float(np.median(x)) if x else None
        results[feat] = {"auc": a, "refl_med": med(pv), "src_med": med(nv), "wall_med": med(wv)}
        print(f"{feat:<10} {a if a is None else round(a,3):>7}  "
              f"{med(pv) if med(pv) is None else round(med(pv),3):>8}  "
              f"{med(nv) if med(nv) is None else round(med(nv),3):>7}  "
              f"{med(wv) if med(wv) is None else round(med(wv),3):>8}")

    # 简单阈值报告
    for feat in ("r2", "r_h"):
        pv = vals("reflection", feat)
        nv = vals("source", feat)
        if not pv or not nv:
            continue
        lo = 0.5 if feat == "r_h" else 1.05
        hi = 1.5 if feat == "r_h" else 2.5
        best = None
        for t in np.arange(lo, hi, 0.05):
            if feat == "r_h":
                se = sum(1 for x in pv if x is not None and x < t) / len(pv)
                sp = sum(1 for x in nv if x is not None and x >= t) / len(nv)
            else:
                se = sum(1 for x in pv if x is not None and x > t) / len(pv)
                sp = sum(1 for x in nv if x is not None and x <= t) / len(nv)
            j = se + sp - 1
            if best is None or j > best[0]:
                best = (j, float(t), se, sp)
        print(f"Youden best({feat}): thr={best[1]:.2f}  refl detected={best[2]:.0%}  src passed={best[3]:.0%}")
        results[f"youden_{feat}"] = {"thr": best[1], "sens": best[2], "spec": best[3]}

    out = {"results": results,
           "per_track": {f"{g}:{t}#{i}": d for (g, t, i), d in per_track.items()}}
    Path(os.path.join(args.out_dir, "depth_gate_test.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out_dir}/depth_gate_test.json")

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
        groups_colors = {"reflection": "#d62728", "source": "#2ca02c", "wall_mirror": "#9467bd"}
        groups_names = {"reflection": f"floor reflection (n={sum(1 for g,_,_ in per_track if g=='reflection')})",
                        "source": f"real person / source (n={sum(1 for g,_,_ in per_track if g=='source')})",
                        "wall_mirror": f"wall mirror (n={sum(1 for g,_,_ in per_track if g=='wall_mirror')})"}
        for ax, feat, xlabel in ((axes[0], "r2", "r2 = median(lower half) / median(upper half)"),
                                 (axes[1], "r_h", "r_h = median(box) / median(same-row side strips)")):
            for gi, grp in enumerate(("reflection", "source", "wall_mirror")):
                xs = [d[feat] for (g, t, i), d in per_track.items() if g == grp]
                if not xs:
                    continue
                ys = np.random.default_rng(0).normal(gi, 0.06, len(xs))
                ax.scatter(xs, ys, c=groups_colors[grp], s=36, alpha=0.8, label=groups_names[grp])
            ax.axvline(1.0, color="#555", lw=1, ls=":")
            ax.set_xlabel(xlabel)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["refl", "real", "wall"], fontsize=9)
            ax.grid(alpha=0.2)
        axes[0].legend(loc="upper right", fontsize=8.5)
        fig.suptitle("In-box depth gradient: floor reflection vs real person (MiDaS)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_depth_gate.png"), facecolor="white")
        print(f"[out] {args.out_dir}/fig_depth_gate.png")
    except Exception as e:  # noqa: BLE001
        print("[warn] fig failed:", str(e)[:160])


if __name__ == "__main__":
    main()
