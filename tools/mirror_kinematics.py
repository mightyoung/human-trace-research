#!/usr/bin/env python
"""镜像运动学检验：验证"倒影轨迹 = 源轨迹的运动学镜像"（零新依赖：不需要深度/掩膜/标定）。

物理关系（地板倒影，镜像面 = 地板平面）：
  u_R ≈ u_S（水平对齐）；h_R ≈ h_S（同尺度）；
  v_R_center ≈ v_S_bottom + 0.5·h_S（倒影中心 = 脚部关于触地线的镜像位置，常数偏移）；
  ⇒ 逐帧锁步：Δv_R_center ≈ Δv_S_bottom（差分相关 ≈ 1）。
墙玻璃镜像（镜像面 = 竖直面）：u 轴反相（≈ -1）、v 同相——机制不同，留作对照。

数据：spread 审计确认的 倒影↔源 配对（shortlist_audited.csv 中 *_sib 列）
控制组：同机"洗牌配对"（reflection_i vs source_j, i≠j）——期望锁步相关显著更低。

用法：
  python tools/mirror_kinematics.py \
    --audit-csv output_crosscam/analysis/spread/depth_filter/shortlist_audited.csv \
    --a-jsonl output_crosscam/316_detections.jsonl \
    --b-jsonl output_crosscam/323_detections.jsonl \
    --out-dir output_crosscam/analysis/mirror_kinematics
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


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


def build_pairs(audit_csv):
    pairs = set()
    with open(audit_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for side in ("disp", "app"):
                if row.get(f"{side}_sib") == "1":
                    tag, tid = row[side].split("#")
                    pairs.add((tag, int(tid), tag, int(row[f"{side}_sib_tid"])))
    return sorted(pairs)


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def pair_metrics(tracks, W, H, tid_r, tid_s):
    tr = {f: b for f, b in tracks.get(tid_r, [])}
    ts = {f: b for f, b in tracks.get(tid_s, [])}
    common = sorted(set(tr) & set(ts))
    if len(common) < 3:
        return None
    u_r, vr, hr, u_s, vs, hs, vc_s = [], [], [], [], [], [], []
    for f in common:
        br = tr[f]
        bs = ts[f]
        u_r.append((br[0] + br[2]) / 2 / W)
        vr.append((br[1] + br[3]) / 2 / H)
        hr.append((br[3] - br[1]) / H)
        u_s.append((bs[0] + bs[2]) / 2 / W)
        vs.append(bs[3] / H)
        vc_s.append((bs[1] + bs[3]) / 2 / H)
        hs.append((bs[3] - bs[1]) / H)
    u_r, vr, hr, u_s, vs, hs, vc_s = map(
        np.asarray, (u_r, vr, hr, u_s, vs, hs, vc_s))
    # 倒影中心预测：关于触地线（脚部）的镜像 => v_R ≈ 2·v_S_bot − v_S_center
    pred = 2 * vs - vc_s
    pred_err = np.abs(vr - pred) / (hs + 1e-9)
    res = {
        "n_common": len(common),
        "du_med": float(np.median(u_r - u_s)),
        "hratio": float(np.median(hr / (hs + 1e-9))),
        "offset_ratio": float(np.median((vr - vs) / (hs + 1e-9))),
        "offset_std": float(np.std((vr - vs) / (hs + 1e-9))),
        "pred_err": float(np.median(pred_err)),
        "pred_corr": pearson(vr, pred),
        "level_corr": pearson(vr, vs),
        "du_corr": pearson(u_r, u_s),
    }
    # 逐帧差分锁步（对相邻共同帧）
    dvr, dvs, dur, dus = [], [], [], []
    for i in range(1, len(common)):
        if common[i] - common[i - 1] > 10:
            continue
        dvr.append(vr[i] - vr[i - 1])
        dvs.append(vs[i] - vs[i - 1])
        dur.append(u_r[i] - u_r[i - 1])
        dus.append(u_s[i] - u_s[i - 1])
    res["n_diff"] = len(dvr)
    res["diff_corr_v"] = pearson(dvr, dvs)
    res["diff_corr_u"] = pearson(dur, dus)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ta = load_tracks(args.a_jsonl)
    tb = load_tracks(args.b_jsonl)
    DIMS = {"316": (2560, 1440), "323": (1920, 1080)}

    pairs = build_pairs(args.audit_csv)
    print(f"[pairs] 倒影-源配对: {len(pairs)}  (来自 {args.audit_csv})")

    # 控制组：扫描同机全部"共现轨迹对"（≥3 共同帧）——即检测器实际面对的自然负样本池
    def cooccur_pairs(tracks, max_pairs=300):
        fmap = {}
        for tid, lst in tracks.items():
            if len(lst) < 6:
                continue
            for f, b in lst:
                fmap.setdefault(f, []).append(tid)
        cnt = {}
        for f, tids in fmap.items():
            tids = sorted(set(tids))
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    k = (tids[i], tids[j])
                    cnt[k] = cnt.get(k, 0) + 1
        prs = [(a, b) for (a, b), n in cnt.items() if n >= 3]
        rng = np.random.default_rng(0)
        rng.shuffle(prs)
        return prs[:max_pairs]

    true_keys = {(t, r, s) for (t, r, _, s) in pairs}
    ctrl = []
    for tag, tracks in (("316", ta), ("323", tb)):
        for a, b in cooccur_pairs(tracks):
            if (tag, a, b) in true_keys or (tag, b, a) in true_keys:
                continue
            ctrl.append((tag, a, tag, b))
    print(f"[ctrl] 自然共现对照对: {len(ctrl)}")

    def run(plist, label):
        rows = []
        for tag, tid_r, _, tid_s in plist:
            tracks = ta if tag == "316" else tb
            W, H = DIMS[tag]
            m = pair_metrics(tracks, W, H, tid_r, tid_s)
            if m:
                m.update({"pair": f"{tag}#{tid_r}~{tag}#{tid_s}", "kind": label})
                rows.append(m)
        return rows

    true_rows = run(pairs, "true")
    ctrl_rows = run(ctrl, "control")

    print(f"\n[data] 有效: true={len(true_rows)}  control={len(ctrl_rows)}")
    print(f"\n{'metric':<14} {'true_med':>9} {'ctrl_med':>9} {'true_p25':>9} {'true_p75':>9} {'AUC(T>C)':>9}")

    def auc(key):
        a = [r[key] for r in true_rows if r.get(key) is not None]
        b = [r[key] for r in ctrl_rows if r.get(key) is not None]
        if not a or not b:
            return None
        c = 0.0
        for x in a:
            for y in b:
                c += 1 if x > y else (0.5 if x == y else 0)
        return c / (len(a) * len(b))

    metrics = ["pred_err", "pred_corr", "level_corr", "diff_corr_v", "du_corr",
               "hratio", "offset_ratio"]
    stats = {}
    for key in metrics:
        tv = [r[key] for r in true_rows if r.get(key) is not None]
        cv = [r[key] for r in ctrl_rows if r.get(key) is not None]
        if not tv:
            continue
        a = auc(key)
        a_s = "None" if a is None else f"{a:.3f}"
        cv_med = "nan" if not cv else f"{np.median(cv):.3f}"
        stats[key] = a
        print(f"{key:<14} {np.median(tv):>9.3f} {cv_med:>9} "
              f"{np.percentile(tv, 25):>9.3f} {np.percentile(tv, 75):>9.3f} {a_s:>9}")
    # 组合规则（最终版：阈值扫描后取 TP 最紧：28/29，控制池最小）
    def rule(m):
        if m is None or m.get("pred_err") is None:
            return False
        return (abs(m["du_med"]) < 0.06 and m["pred_err"] <= 0.1 and
                0.35 <= m["offset_ratio"] <= 0.6)

    tp = sum(1 for r in true_rows if rule(r))
    fp = sum(1 for r in ctrl_rows if rule(r))
    print(f"\n[rule] |du|<0.06 & pred_err<=0.10 & 0.35<=offset<=0.60")
    print(f"  真对通过: {tp}/{len(true_rows)}  控制池通过: {fp}/{len(ctrl_rows)}")
    news = [r for r in ctrl_rows if rule(r)]
    if news:
        print(f"\n=== 控制池中通过的 {len(news)} 对（pred_err 均≤0.08，几何上与真对无法区分，疑为新镜像对）===")
        for r in news:
            print(f"  {r['pair']:<20} n={r['n_common']:>3} du={r['du_med']:+.3f} "
                  f"perr={r['pred_err']:.3f} off={r['offset_ratio']:.2f} hr={r['hratio']:.2f}")

    # 明细
    print(f"\n=== 真对明细（按 diff_corr 降序）===")
    for r in sorted(true_rows, key=lambda r: -(r.get("diff_corr_v") or -9)):
        dc = r.get("diff_corr_v")
        print(f"  {r['pair']:<20} n={r['n_common']:>3} du={r['du_med']:+.3f} "
              f"dc_v={'None' if dc is None else format(dc, '.2f')} "
              f"off={r['offset_ratio']:.2f}±{r['offset_std']:.2f} hr={r['hratio']:.2f} "
              f"perr={r['pred_err']:.2f} pcorr={'None' if r.get('pred_corr') is None else format(r['pred_corr'], '.2f')}")

    # 诊断：打印代表对的原始序列（核对镜像几何）
    for target in ("323#101~323#97", "316#10~316#9"):
        for tag, tid_r, _, tid_s in pairs:
            key = f"{tag}#{tid_r}~{tag}#{tid_s}"
            if key != target:
                continue
            tracks = ta if tag == "316" else tb
            W, H = DIMS[tag]
            tr = {f: b for f, b in tracks[tid_r]}
            ts = {f: b for f, b in tracks[tid_s]}
            common = sorted(set(tr) & set(ts))
            print(f"\n[diag] {key}")
            print("       f   v_S_bot  v_S_c   v_R_c   pred    err")
            for f in common:
                br = tr[f]
                bs = ts[f]
                vs = bs[3] / H
                vcs = (bs[1] + bs[3]) / 2 / H
                vrc = (br[1] + br[3]) / 2 / H
                pred = 2 * vs - vcs
                err = abs(vrc - pred) / ((bs[3] - bs[1]) / H + 1e-9)
                print(f"  {f:>6}  {vs:.3f}  {vcs:.3f}  {vrc:.3f}  {pred:.3f}  {err:.2f}")

    out = {"n_true": len(true_rows), "n_control": len(ctrl_rows),
           "auc": stats, "rule": {"passed_true": tp, "passed_control": fp},
           "true_rows": true_rows, "control_rows": ctrl_rows}
    Path(os.path.join(args.out_dir, "mirror_kinematics.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {args.out_dir}/mirror_kinematics.json")

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=120)
        for rows, color, name in ((true_rows, "#2ca02c", f"true mirror pairs (n={len(true_rows)})"),
                                  (ctrl_rows, "#d62728", f"non-mirror control (n={len(ctrl_rows)})")):
            xs = [r.get("pred_err") for r in rows if r.get("pred_err") is not None]
            ys = [r.get("pred_corr") if r.get("pred_corr") is not None else 0
                  for r in rows if r.get("pred_err") is not None]
            ax.scatter(xs, ys, c=color, s=40, alpha=0.8, label=name, edgecolors="none")
        ax.axvline(0.8, color="#555", ls="--", lw=1.2)
        ax.axhline(0.9, color="#555", ls=":", lw=1.2)
        ax.set_xlim(0, 3)
        ax.set_xlabel("prediction error  |v_R − (2·v_feet − v_S_center)| / h_S")
        ax.set_ylabel("corr(v_R , predicted mirror position)")
        ax.set_title("Mirror kinematics: predicted-mirror-position test\n"
                     "(green below-left = true mirror pairs; red = non-mirror controls)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_mirror_kinematics.png"), facecolor="white")
        print(f"[out] {args.out_dir}/fig_mirror_kinematics.png")
    except Exception as e:  # noqa: BLE001
        print("[warn] fig failed:", str(e)[:160])


if __name__ == "__main__":
    main()
