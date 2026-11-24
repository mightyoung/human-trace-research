#!/usr/bin/env python
"""翻转手性检验：倒影内容 = 源的镜像翻转；用 DINOv2 做"翻转相似度"正交验证。

物理预期：
  地板倒影：垂直翻转（flipud）倒影裁剪 -> 应更接近源外观（cos 提升）；
            水平翻转（fliplr）作为"错误翻转"对照（不应提升或更差）。
  对照组：自然非镜像共现对（来自 mirror_kinematics 控制池未通过者）。

用法：
  python tools/mirror_flip_test.py \
    --audit-csv output_crosscam/analysis/spread/depth_filter/shortlist_audited.csv \
    --ctrl-json output_crosscam/analysis/mirror_kinematics/mirror_kinematics.json \
    --a-jsonl output_crosscam/316_detections.jsonl --b-jsonl output_crosscam/323_detections.jsonl \
    --a-video "media-resource/.../316_...mp4" --b-video "media-resource/.../323_...mp4" \
    --out-dir output_crosscam/analysis/mirror_flip
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crosscam_depth import _install_offline_hub_patch  # noqa: E402
from crosscam_reid import Embedder, prep_crop  # noqa: E402


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


def build_true_pairs(audit_csv):
    pairs = set()
    with open(audit_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for side in ("disp", "app"):
                if row.get(f"{side}_sib") == "1":
                    tag, tid = row[side].split("#")
                    pairs.add((tag, int(tid), int(row[f"{side}_sib_tid"])))
    return sorted(pairs)


def build_ctrl_pairs(ctrl_json, rule_perr=0.10):
    d = json.load(open(ctrl_json, encoding="utf-8"))
    out = []
    for r in d.get("control_rows", []):
        if r.get("pred_err") is None or r["pred_err"] <= rule_perr:
            continue
        tag = r["pair"].split("#")[0]
        a, b = r["pair"].split("~")
        ta, ia = a.split("#")
        tb, ib = b.split("#")
        out.append((ta, int(ia), int(ib)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--ctrl-json", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--max-ctrl", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    _install_offline_hub_patch()
    true_pairs = build_true_pairs(args.audit_csv)
    ctrl_pairs_all = build_ctrl_pairs(args.ctrl_json)
    rng = np.random.default_rng(0)
    rng.shuffle(ctrl_pairs_all)
    ctrl_pairs = ctrl_pairs_all[: args.max_ctrl]
    print(f"[plan] true pairs={len(true_pairs)}  control pairs={len(ctrl_pairs)}")

    ta = load_tracks(args.a_jsonl)
    tb = load_tracks(args.b_jsonl)
    caps = {"316": cv2.VideoCapture(args.a_video), "323": cv2.VideoCapture(args.b_video)}
    frame_cache = {}

    def grab(tag, f):
        key = (tag, f)
        if key in frame_cache:
            return frame_cache[key]
        cap = caps[tag]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        frame_cache[key] = fr if ok else None
        return frame_cache[key]

    # 收集编码任务
    jobs = []  # (kind, label, tag, frame, role, tid, box)
    def add_pair(kind, tag, tid_r, tid_s):
        tracks = ta if tag == "316" else tb
        tr = dict(tracks.get(tid_r, []))
        ts = dict(tracks.get(tid_s, []))
        common = sorted(set(tr) & set(ts))
        if len(common) < 2:
            return
        idxs = np.linspace(0, len(common) - 1, args.frames).astype(int)
        for i in idxs:
            f = common[i]
            b_r = tr[f]
            b_s = ts[f]
            if (b_r[2] - b_r[0]) < 20 or (b_s[2] - b_s[0]) < 20:
                continue
            jobs.append((kind, f"{tag}#{tid_r}~{tag}#{tid_s}", tag, f, "R", tid_r, b_r))
            jobs.append((kind, f"{tag}#{tid_r}~{tag}#{tid_s}", tag, f, "S", tid_s, b_s))

    for tag, tid_r, tid_s in true_pairs:
        add_pair("true", tag, tid_r, tid_s)
    for tag, tid_r, tid_s in ctrl_pairs:
        add_pair("control", tag, tid_r, tid_s)
    print(f"[plan] encode jobs: {len(jobs)}")

    embedder = Embedder(device="cpu")
    # 批量编码：R / R_flipud / R_fliplr / S
    embs = {}  # (kind, pair, frame, role, variant) -> vec
    B = 16
    buf_rgb, buf_key = [], []

    def flush():
        if not buf_rgb:
            return
        vecs = embedder.embed_batch(buf_rgb)
        for k, v in zip(buf_key, vecs):
            embs[k] = v
        buf_rgb.clear()
        buf_key.clear()

    for kind, pair, tag, f, role, tid, box in jobs:
        fr = grab(tag, f)
        if fr is None:
            continue
        crop = prep_crop(fr, box)
        if crop is None:
            continue
        variants = {"raw": crop}
        if role == "R":
            variants["ud"] = np.flipud(crop).copy()
            variants["lr"] = np.fliplr(crop).copy()
        for name, im in variants.items():
            buf_rgb.append(np.ascontiguousarray(im))
            buf_key.append((kind, pair, f, role, name))
            if len(buf_rgb) >= B:
                flush()
    flush()
    caps["316"].release()
    caps["323"].release()
    print(f"[run] embeddings: {len(embs)}")

    # 配对聚合
    def cos(a, b):
        return float(np.dot(a, b))

    per = {}
    for (kind, pair, f, role, name), v in embs.items():
        d = per.setdefault((kind, pair, f), {})
        d[f"{role}_{name}"] = v
    rows = []
    for (kind, pair, f), d in per.items():
        if not all(k in d for k in ("R_raw", "R_ud", "R_lr", "S_raw")):
            continue
        base = cos(d["R_raw"], d["S_raw"])
        ud = cos(d["R_ud"], d["S_raw"])
        lr = cos(d["R_lr"], d["S_raw"])
        rows.append({"kind": kind, "pair": pair, "frame": f,
                     "cos_base": base, "cos_ud": ud, "cos_lr": lr,
                     "boost_ud": ud - base, "boost_lr": lr - base})
    # 按对聚合
    pairs_agg = {}
    for r in rows:
        pairs_agg.setdefault((r["kind"], r["pair"]), []).append(r)
    agg = {}
    for (kind, pair), lst in pairs_agg.items():
        agg[(kind, pair)] = {
            "cos_base": float(np.median([x["cos_base"] for x in lst])),
            "cos_ud": float(np.median([x["cos_ud"] for x in lst])),
            "cos_lr": float(np.median([x["cos_lr"] for x in lst])),
            "boost_ud": float(np.median([x["boost_ud"] for x in lst])),
            "boost_lr": float(np.median([x["boost_lr"] for x in lst])),
            "n": len(lst),
        }

    def vals(kind, key):
        return [v[key] for (k, p), v in agg.items() if k == kind]

    print(f"\n[data] aggregated pairs: true={sum(1 for k,_ in agg if k=='true')}  "
          f"control={sum(1 for k,_ in agg if k=='control')}")
    print(f"\n{'metric':<10} {'true_med':>9} {'ctrl_med':>9}")
    for key in ("cos_base", "cos_ud", "cos_lr", "boost_ud", "boost_lr"):
        tv, cv_ = vals("true", key), vals("control", key)
        if not tv:
            continue
        print(f"{key:<10} {np.median(tv):>9.3f} {(np.median(cv_) if cv_ else float('nan')):>9.3f}")

    tv, cv_ = vals("true", "boost_ud"), vals("control", "boost_ud")
    if tv and cv_:
        c = 0.0
        for x in tv:
            for y in cv_:
                c += 1 if x > y else (0.5 if x == y else 0)
        auc = c / (len(tv) * len(cv_))
        print(f"\n[AUC] boost_ud 分离 true vs control = {auc:.3f}")

    out = {"aggregated": {f"{k}:{p}": v for (k, p), v in agg.items()},
           "rows": rows}
    Path(os.path.join(args.out_dir, "mirror_flip.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out_dir}/mirror_flip.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
        for kind, color, name in (("true", "#2ca02c", "true mirror pairs"),
                                  ("control", "#d62728", "non-mirror control")):
            xs = [v["boost_ud"] for (k, p), v in agg.items() if k == kind]
            ys = [v["boost_lr"] for (k, p), v in agg.items() if k == kind]
            ax.scatter(xs, ys, c=color, s=40, alpha=0.8, label=f"{name} (n={len(xs)})", edgecolors="none")
        ax.axvline(0, color="#555", ls="--", lw=1)
        ax.set_xlabel("boost_flipud = cos(flipud(R), S) - cos(R, S)")
        ax.set_ylabel("boost_fliplr = cos(fliplr(R), S) - cos(R, S)")
        ax.set_title("Flip-chirality test: vertical flip should boost true floor reflections")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_mirror_flip.png"), facecolor="white")
        print(f"[out] {args.out_dir}/fig_mirror_flip.png")
    except Exception as e:  # noqa: BLE001
        print("[warn] fig failed:", str(e)[:160])


if __name__ == "__main__":
    main()
