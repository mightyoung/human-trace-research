#!/usr/bin/env python
"""P1：反光镜像的深度一致性实验（MiDaS）

物理假设：
  H1 "地板不可能性"：真人落脚点贴合真实地面（足点深度与图像下方地面深度自洽）；
     镜像"人"位于玻璃面上，其足点深度与"该图像位置对应的地面深度"不符（用户提出的
     "实体地面不会有（这样的）人" → 用地面深度曲线量化）。
  H2 "玻璃面平坦性"：镜像贴在玻璃平面 → 框内深度较均匀；真人（斜视角）头脚有深度梯度。
  H3 上下邻域不一致：真实人脚下是地板、头顶上方是墙体/空间；镜像上下皆为玻璃反射内容。

特征（每实例，逐帧归一化 d/median(frame) 以消除 MiDaS 尺度不确定性）：
  d_foot      足点（框底边中央 patch）中值
  r_below     足点/下方 patch 之比
  r_above     足点/上方 patch 之比
  r_intra     框下半/上半 中值之比
  std_ratio   框内深度 std/mean
  floor_resid 足点深度 vs 地板曲线 d_floor(v)（由真人集合拟合，LOO）的相对残差

评估：AUC 分离 {镜像实例} vs {真人实例}。

用法：
  python tools/reflection_depth_test.py \
    --a-orig output_crosscam/316_detections.jsonl \
    --b-orig output_crosscam/323_detections.jsonl \
    --a-masked output_crosscam/glass_mask/316_masked.jsonl \
    --b-masked output_crosscam/glass_mask/323_masked.jsonl \
    --a-video "media-resource/.../316_...mp4" --b-video "media-resource/.../323_...mp4" \
    --phantom a:101,a:74,a:96,b:164,b:133 \
    --out-dir output_crosscam/analysis/reflection_p1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscam_depth import load_midas, depth_batch, foot_depth  # noqa: E402

A_WH = (2560, 1440)
B_WH = (1920, 1080)


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


def patch_med(dmap, x1, y1, x2, y2):
    H, W = dmap.shape
    x1, x2 = max(0, int(x1)), min(W, int(x2))
    y1, y2 = max(0, int(y1)), min(H, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return float(np.median(dmap[y1:y2, x1:x2]))


def features_for_box(dmap, box, frame_med):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    feat = {}
    d_foot = foot_depth(dmap, box)
    if d_foot is None:
        return None
    feat["d_foot"] = d_foot
    cx1, cx2 = x1 + 0.15 * w, x2 - 0.15 * w
    d_below = patch_med(dmap, cx1, y2 + 0.06 * h, cx2, y2 + 0.55 * h)
    d_above = patch_med(dmap, cx1, y1 - 0.40 * h, cx2, y1 - 0.06 * h)
    dU = patch_med(dmap, x1, y1, x2, y1 + 0.5 * h)
    dL = patch_med(dmap, x1, y1 + 0.5 * h, x2, y2)
    box_reg = dmap[max(0, int(y1)):min(dmap.shape[0], int(y2)),
                   max(0, int(x1)):min(dmap.shape[1], int(x2))]
    feat["r_below"] = d_foot / d_below if d_below else None
    feat["r_above"] = d_foot / d_above if d_above else None
    feat["r_intra"] = (dL / dU) if (dU and dL) else None
    if box_reg.size > 9:
        feat["std_ratio"] = float(box_reg.std() / (box_reg.mean() + 1e-9))
    else:
        feat["std_ratio"] = None
    # 归一化
    feat["d_foot_n"] = d_foot / frame_med
    feat["v_foot"] = float(y2) / dmap.shape[0]
    return feat


def process_side(video, need, model, batch=8):
    """need: {frame: [(key, box)]} -> {key: feat}"""
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
            fm = float(np.median(dmaps[i]))
            for key, box in keys:
                f = features_for_box(dmaps[i], box, fm)
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
    ap.add_argument("--a-orig", required=True)
    ap.add_argument("--b-orig", required=True)
    ap.add_argument("--a-masked", required=True)
    ap.add_argument("--b-masked", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--phantom", required=True, help="如 a:101,a:74,b:164")
    ap.add_argument("--real-tracks-per-cam", type=int, default=18)
    ap.add_argument("--real-inst-per-track", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ta_orig = load_tracks(args.a_orig)
    tb_orig = load_tracks(args.b_orig)
    ta_mask = load_tracks(args.a_masked)
    tb_mask = load_tracks(args.b_masked)

    # ---- 实例集合 ----
    phantom_spec = {}
    for s in args.phantom.split(","):
        c, t = s.split(":")
        phantom_spec.setdefault(c, []).append(int(t))

    need_a, need_b = {}, {}
    meta = []  # (key, label, cam, tid, frame)

    for cam, spec, orig in (("a", phantom_spec.get("a", []), ta_orig),
                            ("b", phantom_spec.get("b", []), tb_orig)):
        for tid in spec:
            for fr, box in orig.get(tid, []):
                key = (cam, "P", tid, fr)
                (need_a if cam == "a" else need_b).setdefault(fr, []).append((key, box))
                meta.append(key)

    rng = np.random.default_rng(0)
    for cam, mask in (("a", ta_mask), ("b", tb_mask)):
        cands = [tid for tid, lst in mask.items() if len(lst) >= 10]
        cands.sort(key=lambda t: -len(mask[t]))
        pick = cands[: args.real_tracks_per_cam]
        for tid in pick:
            lst = mask[tid]
            idxs = np.linspace(2, len(lst) - 3, args.real_inst_per_track).astype(int)
            for i in idxs:
                fr, box = lst[i]
                key = (cam, "R", tid, int(fr))
                (need_a if cam == "a" else need_b).setdefault(int(fr), []).append((key, box))
                meta.append(key)

    print(f"[plan] phantom instances={sum(1 for k in meta if k[1]=='P')}  "
          f"real instances={sum(1 for k in meta if k[1]=='R')}")

    model = load_midas()
    print("[run] depth on A(316) ...")
    fa = process_side(args.a_video, need_a, model)
    print("[run] depth on B(323) ...")
    fb = process_side(args.b_video, need_b, model)

    # ---- 特征统计与 AUC ----
    keys = [k for k in meta if k in fa or k in fb]
    feats = {k: (fa.get(k) or fb.get(k)) for k in keys}
    keys = [k for k in keys if feats.get(k)]
    P = [k for k in keys if k[1] == "P"]
    R = [k for k in keys if k[1] == "R"]
    print(f"[data] 有效实例: P={len(P)}  R={len(R)}")

    # 地板曲线（用真人拟合 LOO）
    vR = np.array([feats[k]["v_foot"] for k in R])
    dR = np.array([feats[k]["d_foot_n"] for k in R])
    coef = np.polyfit(vR, dR, 2)
    def floor_pred(v):
        return np.polyval(coef, v)
    for k in keys:
        f = feats[k]
        dp = floor_pred(f["v_foot"])
        f["floor_resid"] = abs(f["d_foot_n"] - dp) / (dp + 1e-9)

    # LOO 地板残差（真人留一轨迹）
    for k in R:
        tid_own = k[2]
        vR2 = np.array([feats[q]["v_foot"] for q in R if q[2] != tid_own])
        dR2 = np.array([feats[q]["d_foot_n"] for q in R if q[2] != tid_own])
        if len(vR2) >= 6:
            c2 = np.polyfit(vR2, dR2, 2)
            dp = np.polyval(c2, feats[k]["v_foot"])
            feats[k]["floor_resid_loo"] = abs(feats[k]["d_foot_n"] - dp) / (dp + 1e-9)
        else:
            feats[k]["floor_resid_loo"] = None
    # 公平版：镜像用全曲线残差（本就出样本），真人用 LOO 残差
    for k in keys:
        feats[k]["floor_resid_fair"] = (
            feats[k]["floor_resid"] if k[1] == "P" else feats[k].get("floor_resid_loo"))

    out = {"phantom_keys": [list(k) for k in P], "real_keys": [list(k) for k in R],
           "features": {str(k): feats[k] for k in keys},
           "floor_coef_v_d": list(coef)}
    os.makedirs(args.out_dir, exist_ok=True)
    Path(os.path.join(args.out_dir, "reflection_depth_scores.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    def fv(x, w=8):
        return ("None" if x is None else f"{x:.4f}").rjust(w)

    def fa_(x, w=9):
        return ("None" if x is None else f"{x:.3f}").rjust(w)

    print(f"\n{'feature':<18} {'AUC(P>R)':>9} {'AUC(反向)':>10}   P_med      R_med")
    results = {}
    for name in ["d_foot_n", "r_below", "r_above", "r_intra", "std_ratio",
                 "floor_resid", "floor_resid_loo", "floor_resid_fair"]:
        pv = [feats[k].get(name) for k in P]
        rv = [feats[k].get(name) for k in R]
        a1 = auc_of(pv, rv)
        a2 = auc_of([None if x is None else -x for x in pv],
                    [None if x is None else -x for x in rv])
        pm = np.nanmedian([x for x in pv if x is not None]) if any(x is not None for x in pv) else None
        rm = np.nanmedian([x for x in rv if x is not None]) if any(x is not None for x in rv) else None
        results[name] = {"auc": a1, "auc_neg": a2, "p_med": pm, "r_med": rm}
        print(f"{name:<18} {fa_(a1)} {fa_(a2)}  {fv(pm)} {fv(rm)}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {"phantom_keys": [list(k) for k in P], "real_keys": [list(k) for k in R],
           "features": {str(k): feats[k] for k in keys}, "auc": results,
           "floor_coef_v_d": list(coef)}
    Path(os.path.join(args.out_dir, "reflection_depth_scores.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out_dir}/reflection_depth_scores.json")

    # 图：v-d 散点 + 地板曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=120)
        ax = axes[0]
        ax.scatter([feats[k]["v_foot"] for k in P], [feats[k]["d_foot_n"] for k in P],
                   c="#d62728", s=28, label=f"镜像实例 (n={len(P)})", zorder=3)
        ax.scatter([feats[k]["v_foot"] for k in R], [feats[k]["d_foot_n"] for k in R],
                   c="#2ca02c", s=22, alpha=0.7, label=f"真人实例 (n={len(R)})", zorder=2)
        vv = np.linspace(0.05, 1.0, 100)
        ax.plot(vv, floor_pred(vv), "k--", lw=1.2, label="地板曲线(真人拟合)")
        ax.set_xlabel("足点图像纵坐标 v（0=上 1=下）")
        ax.set_ylabel("足点归一化深度 d_foot/median(frame)")
        ax.set_title("地板不可能性检验：镜像足点偏离地板曲线")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax2 = axes[1]
        fr_ = [feats[k].get("floor_resid_loo") for k in R]
        fp_ = [feats[k].get("floor_resid") for k in P]
        fr_ = [x for x in fr_ if x is not None]
        fp_ = [x for x in fp_ if x is not None]
        bp = ax2.boxplot([fp_, fr_], labels=["镜像(P)", "真人(LOO)"], patch_artist=True)
        for b, c in zip(bp["boxes"], ["#d62728", "#2ca02c"]):
            b.set_facecolor(c); b.set_alpha(0.6)
        ax2.set_ylabel("|d - d_floor(v)| / d_floor(v)")
        ax2.set_title("地板残差（越小=越贴合地面）")
        ax2.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_reflection_depth.png"), facecolor="white")
        print(f"[out] {args.out_dir}/fig_reflection_depth.png")
    except Exception as e:  # noqa: BLE001
        print("[warn] fig failed:", str(e)[:160])


if __name__ == "__main__":
    main()
