#!/usr/bin/env python
"""批量深度过滤：为跨机转移候选计算"双端点地板一致性"分数 -> 真转移候选短名单。

原理（承接 P1 实验 output_crosscam/analysis/reflection_p1/reflection_depth_scores.json）：
  真人足点深度贴合"地板曲线" d_floor(v)（由真人集合拟合）；
  镜面幻影的足点位于玻璃反光面上，深度显著偏离地板 -> floor_resid 大。
  P1 实测：逐实例 AUC=0.931（镜像残差中值 1.22 / 真人 0.29）。

本工具：
  1) 对每个候选的两个端点各取 K 帧（消失端=轨迹末尾，出现端=轨迹开头）
  2) MiDaS-small 相对深度 -> 每帧 floor_resid（归一化 d_foot/median(frame)）
  3) 端点分=K 帧残差中值；候选分=max(消失端, 出现端) -> verdict PASS/REVIEW/REJECT
  4) 联合分层（PASS 基础上）：
       FAST_CHECK  深度达标但 dt<2s（镜像时间规则嫌疑，须复核）
       TIER1       PASS & 非镜像嫌疑 & cos>=tier1_cos  -> 主短名单
       TIER2       PASS & 非镜像嫌疑 & cos>=tier2_cos  -> 备选
       TIER3       其余 PASS（低外观相似，仅留档）
  5) 输出 JSON + CSV + 汇总图 + 短名单全帧证据图
  6) 附玻璃区标注（草稿掩膜，信息性）：disp_glass / app_glass

用法：
  python tools/transfer_depth_filter.py \
    --events  output_crosscam/analysis/spread/transfer_events.json \
    --a-jsonl output_crosscam/316_detections.jsonl \
    --b-jsonl output_crosscam/323_detections.jsonl \
    --a-video "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4" \
    --b-video "media-resource/multi-cemra/synced-10min/05_茶水间_323/323_085500-090500.mp4" \
    --p1-scores output_crosscam/analysis/reflection_p1/reflection_depth_scores.json \
    --out-dir output_crosscam/analysis/spread/depth_filter
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
from crosscam_depth import load_midas  # noqa: E402
from reflection_depth_test import load_tracks, process_side  # noqa: E402

# P1 幻影轨迹集（用户确认的玻璃镜像，用于验证过滤器）
KNOWN_PHANTOM = {("316", 101), ("316", 74), ("316", 96), ("323", 164), ("323", 133)}


def pick_indices(n, side, k, gap):
    """disp: 轨迹末尾起的 k 帧；app: 轨迹开头起的 k 帧（gap 帧间隔）。"""
    if n <= 0:
        return []
    if side == "disp":
        idxs = [n - 1 - gap * i for i in range(k)]
    else:
        idxs = [gap * i for i in range(k)]
    idxs = sorted({i for i in idxs if 0 <= i < n})
    return idxs


def point_in_regions(u, v, polys):
    """多边形内点判定（even-odd rule）。polys: [[[u,v],...], ...]"""
    for poly in polys:
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > v) != (y2 > v):
                xin = (x2 - x1) * (v - y1) / (y2 - y1 + 1e-12) + x1
                if u < xin:
                    inside = not inside
        if inside:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--p1-scores", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--t-pass", type=float, default=0.8, help="PASS 阈值（floor_resid）")
    ap.add_argument("--t-reject", type=float, default=1.1, help="REJECT 阈值")
    ap.add_argument("--tier1-cos", type=float, default=0.6, help="TIER1 外观阈值")
    ap.add_argument("--tier2-cos", type=float, default=0.5, help="TIER2 外观阈值")
    ap.add_argument("--k", type=int, default=3, help="每端点采样帧数")
    ap.add_argument("--gap", type=int, default=6, help="端点采样帧间隔")
    ap.add_argument("--evidence-top", type=int, default=6)
    ap.add_argument("--glass-masks", default="output_crosscam/glass_mask/glass_masks.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ev = json.loads(Path(args.events).read_text(encoding="utf-8"))
    cands = ev["candidates"]
    ta = load_tracks(args.a_jsonl)
    tb = load_tracks(args.b_jsonl)
    p1 = json.loads(Path(args.p1_scores).read_text(encoding="utf-8"))
    floor_coef = p1["floor_coef_v_d"]

    # 玻璃区草稿（信息性标注）
    glass = {}
    gm_path = Path(args.glass_masks)
    if gm_path.exists():
        gm = json.loads(gm_path.read_text(encoding="utf-8"))
        for cam, spec in gm.get("cameras", {}).items():
            if spec.get("enabled"):
                glass[cam] = spec.get("regions", [])

    # ---- 构建采样计划 ----
    need_a, need_b = {}, {}
    plans = []
    miss_disp = miss_app = 0
    for ci, c in enumerate(cands):
        rec = {
            "ci": ci, "dirn": c["dirn"], "dt": c["dt"], "cos": c["cos"],
            "disp_tag": c["disp"]["tag"], "disp_tid": c["disp"]["tid"],
            "app_tag": c["app"]["tag"], "app_tid": c["app"]["tid"],
            "mirror_suspect": bool(c["dt"] < 2.0 and (c["cos"] or 0) >= 0.6),
            "samples": {"disp": [], "app": []},
            "endpoints": {},
        }
        # 玻璃区标注（端点的落脚点归一化坐标）
        rec["disp_glass"] = point_in_regions(c["disp"]["u"], c["disp"]["v"],
                                             glass.get(c["disp"]["tag"], []))
        rec["app_glass"] = point_in_regions(c["app"]["u"], c["app"]["v"],
                                            glass.get(c["app"]["tag"], []))
        for side in ("disp", "app"):
            e = c[side]
            cam = "a" if e["tag"] == "316" else "b"
            tracks = ta if cam == "a" else tb
            lst = tracks.get(int(e["tid"]), [])
            if not lst:
                if side == "disp":
                    miss_disp += 1
                else:
                    miss_app += 1
                continue
            endpos = lst[-1] if side == "disp" else lst[0]
            rec["endpoints"][side] = {"cam": cam, "f": int(endpos[0]),
                                      "box": [float(v) for v in endpos[1]]}
            rec[f"{side}_f_match"] = (int(endpos[0]) == int(e["f"]))
            for i in pick_indices(len(lst), side, args.k, args.gap):
                f, box = lst[i]
                key = (cam, side, int(e["tid"]), int(f))
                (need_a if cam == "a" else need_b).setdefault(int(f), []).append((key, box))
                rec["samples"][side].append(key)
        plans.append(rec)

    n_disp_match = sum(1 for r in plans if r.get("disp_f_match"))
    n_app_match = sum(1 for r in plans if r.get("app_f_match"))
    print(f"[plan] candidates={len(cands)}  A frames={len(need_a)}  B frames={len(need_b)}")
    print(f"[plan] 端点帧一致性: disp {n_disp_match}/{len(plans)}, app {n_app_match}/{len(plans)}"
          f"  (missing: disp {miss_disp}, app {miss_app})", flush=True)

    model = load_midas()
    print("[run] depth on A(316) ...", flush=True)
    fa = process_side(args.a_video, need_a, model)
    print("[run] depth on B(323) ...", flush=True)
    fb = process_side(args.b_video, need_b, model)
    allf = {**fa, **fb}
    print(f"[run] depth features: {len(allf)} instances", flush=True)

    # ---- 打分 ----
    def resid_of(key):
        f = allf.get(key)
        if not f:
            return None
        dp = float(np.polyval(floor_coef, f["v_foot"]))
        return abs(f["d_foot_n"] - dp) / (dp + 1e-9)

    for rec in plans:
        for side in ("disp", "app"):
            vals = [resid_of(k) for k in rec["samples"][side]]
            vals = [v for v in vals if v is not None]
            rec[f"{side}_resid"] = float(np.median(vals)) if vals else None
            rec[f"{side}_n"] = len(vals)
        parts = [x for x in (rec["disp_resid"], rec["app_resid"]) if x is not None]
        rec["score"] = float(max(parts)) if parts else None
        if rec["score"] is None:
            rec["verdict"] = "INCOMPLETE"
        elif rec["score"] <= args.t_pass:
            rec["verdict"] = "PASS"
        elif rec["score"] <= args.t_reject:
            rec["verdict"] = "REVIEW"
        else:
            rec["verdict"] = "REJECT"
        rec["known_phantom_hit"] = (
            (rec["disp_tag"], rec["disp_tid"]) in KNOWN_PHANTOM
            or (rec["app_tag"], rec["app_tid"]) in KNOWN_PHANTOM)
        # 联合分层
        if rec["verdict"] != "PASS":
            rec["tier"] = None
        elif rec["mirror_suspect"]:
            rec["tier"] = "FAST_CHECK"
        elif (rec["cos"] or 0) >= args.tier1_cos:
            rec["tier"] = "TIER1"
        elif (rec["cos"] or 0) >= args.tier2_cos:
            rec["tier"] = "TIER2"
        else:
            rec["tier"] = "TIER3"

    # ---- 汇总 ----
    from collections import Counter
    cnt = Counter(r["verdict"] for r in plans)
    tier_cnt = Counter(r["tier"] for r in plans if r["tier"])
    print(f"\n[verdict] {dict(cnt)}  (t_pass={args.t_pass}, t_reject={args.t_reject})")
    print(f"[tier] {dict(tier_cnt)}")

    def _f(x):
        return "None" if x is None else f"{x:.3f}"

    fasts = sorted([r for r in plans if r["tier"] == "FAST_CHECK"], key=lambda r: -(r["cos"] or 0))
    t1 = sorted([r for r in plans if r["tier"] == "TIER1"], key=lambda r: -(r["cos"] or 0))
    t2 = sorted([r for r in plans if r["tier"] == "TIER2"], key=lambda r: -(r["cos"] or 0))
    t3 = [r for r in plans if r["tier"] == "TIER3"]

    print(f"\n=== FAST_CHECK（深度达标但 dt<2s，镜像时间规则嫌疑，须复核）===")
    for r in fasts:
        print(f"  {r['dirn']}  {r['disp_tag']}#{r['disp_tid']} -> {r['app_tag']}#{r['app_tid']}  "
              f"dt={r['dt']}s cos={r['cos']}  resid=[{_f(r['disp_resid'])}/{_f(r['app_resid'])}]"
              + ("  [glass]" if (r["disp_glass"] or r["app_glass"]) else ""))
    if not fasts:
        print("  (none)")

    print(f"\n=== TIER1 主短名单（PASS & 非镜像嫌疑 & cos>={args.tier1_cos}）===")
    for r in t1:
        print(f"  {r['dirn']}  {r['disp_tag']}#{r['disp_tid']} -> {r['app_tag']}#{r['app_tid']}  "
              f"dt={r['dt']}s cos={r['cos']}  resid=[{_f(r['disp_resid'])}/{_f(r['app_resid'])}]"
              + ("  [glass]" if (r["disp_glass"] or r["app_glass"]) else ""))
    if not t1:
        print("  (none)")

    print(f"\n=== TIER2 备选（cos {args.tier2_cos}~{args.tier1_cos}）：{len(t2)} 条；"
          f"TIER3 低外观留档：{len(t3)} 条（详见 CSV/JSON）===")

    print(f"\n=== REJECT 顶部（镜像/幻影嫌疑）===")
    rej = sorted([r for r in plans if r["verdict"] == "REJECT"], key=lambda r: -(r["score"] or 0))
    for r in rej[:10]:
        flag = " [known-phantom]" if r["known_phantom_hit"] else ""
        print(f"  {r['dirn']}  {r['disp_tag']}#{r['disp_tid']} -> {r['app_tag']}#{r['app_tid']}  "
              f"dt={r['dt']}s cos={r['cos']}  score={r['score']:.3f}{flag}")

    # 验证：已知幻影 / 镜像嫌疑的判定
    print(f"\n=== 验证 ===")
    kp_rows = [r for r in plans if r["known_phantom_hit"]]
    print(f"已知幻影轨迹参与的候选: {len(kp_rows)} 条 -> verdict {dict(Counter(r['verdict'] for r in kp_rows))}")
    ms_rows = [r for r in plans if r["mirror_suspect"]]
    print(f"镜像嫌疑(dt<2s)候选: {len(ms_rows)} 条 -> verdict "
          f"{dict(Counter(r['verdict'] for r in ms_rows))}")

    # ---- 输出 ----
    out = {
        "t_pass": args.t_pass, "t_reject": args.t_reject,
        "tier1_cos": args.tier1_cos, "tier2_cos": args.tier2_cos,
        "floor_coef_v_d": floor_coef,
        "counts": dict(cnt), "tier_counts": dict(tier_cnt),
        "rows": plans,
    }
    Path(os.path.join(args.out_dir, "depth_filter.json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out_dir}/depth_filter.json")

    # CSV
    def fmt_r(x):
        return "" if x is None else round(x, 3)

    all_csv = os.path.join(args.out_dir, "depth_filter_all.csv")
    with open(all_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tier", "verdict", "dirn", "disp", "app", "dt", "cos",
                    "disp_resid", "app_resid", "score", "mirror_suspect",
                    "known_phantom", "disp_glass", "app_glass"])
        for r in sorted(plans, key=lambda r: (r["verdict"] != "PASS", -(r["cos"] or 0))):
            w.writerow([r["tier"] or "", r["verdict"], r["dirn"],
                        f"{r['disp_tag']}#{r['disp_tid']}", f"{r['app_tag']}#{r['app_tid']}",
                        r["dt"], r["cos"], fmt_r(r["disp_resid"]), fmt_r(r["app_resid"]),
                        fmt_r(r["score"]), r["mirror_suspect"], r["known_phantom_hit"],
                        r["disp_glass"], r["app_glass"]])
    sl_csv = os.path.join(args.out_dir, "transfer_shortlist.csv")
    tier_order = {"FAST_CHECK": 0, "TIER1": 1, "TIER2": 2}
    sl_rows = fasts + t1 + t2
    with open(sl_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tier", "dirn", "disp", "app", "dt", "cos",
                    "disp_resid", "app_resid", "score", "disp_glass", "app_glass"])
        for r in sorted(sl_rows, key=lambda r: (tier_order[r["tier"]], -(r["cos"] or 0))):
            w.writerow([r["tier"], r["dirn"], f"{r['disp_tag']}#{r['disp_tid']}",
                        f"{r['app_tag']}#{r['app_tid']}", r["dt"], r["cos"],
                        fmt_r(r["disp_resid"]), fmt_r(r["app_resid"]), fmt_r(r["score"]),
                        r["disp_glass"], r["app_glass"]])
    print(f"[out] {all_csv}")
    print(f"[out] {sl_csv}")

    # ---- 汇总图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        COL = {"REVIEW": "#ff9900", "REJECT": "#d62728", "INCOMPLETE": "#888888"}
        fig, ax = plt.subplots(figsize=(10, 5.8), dpi=120)
        for v in ("REJECT", "REVIEW", "INCOMPLETE"):
            sub = [r for r in plans if r["verdict"] == v and r["score"] is not None]
            if not sub:
                continue
            ax.scatter([r["cos"] for r in sub],
                       [min(r["score"], 2.2) for r in sub],
                       c=COL[v], s=24, alpha=0.85, label=f"{v} (n={len(sub)})", zorder=2)
        sub = [r for r in plans if r["verdict"] == "PASS" and r["score"] is not None]
        ax.scatter([r["cos"] for r in sub], [min(r["score"], 2.2) for r in sub],
                   c="#2ca02c", s=30, alpha=0.8, label=f"PASS depth (n={len(sub)})",
                   zorder=3, edgecolors="none")
        if fasts:
            ax.scatter([r["cos"] for r in fasts], [min(r["score"], 2.2) for r in fasts],
                       c="#9467bd", s=60, marker="*", label=f"FAST_CHECK (n={len(fasts)})", zorder=4)
        ax.axhline(args.t_pass, color="#555", ls="--", lw=1.2)
        ax.axhline(args.t_reject, color="#555", ls=":", lw=1.2)
        ax.text(0.40, args.t_pass + 0.03, f"t_pass={args.t_pass}", fontsize=9)
        ax.text(0.40, args.t_reject + 0.03, f"t_reject={args.t_reject}", fontsize=9)
        for r in t1[:6]:
            ax.annotate(f"{r['disp_tag']}#{r['disp_tid']}->{r['app_tag']}#{r['app_tid']}",
                        (r["cos"], min(r["score"], 2.2)), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("appearance cosine (DINOv2)")
        ax.set_ylabel("depth floor residual  score = max(both endpoints)")
        ax.set_title("Depth floor consistency of transfer candidates\n"
                     "(PASS=green, FAST_CHECK star=dt<2s mirror-suspect, red=REJECT)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "depth_filter.png"), facecolor="white")
        print(f"[out] {args.out_dir}/depth_filter.png")
    except Exception as e:  # noqa: BLE001
        print("[warn] fig failed:", str(e)[:160])

    # ---- 证据图：FAST_CHECK 全部 + TIER1 顶部 ----
    ev_rows = fasts + t1[: args.evidence_top]
    if ev_rows:
        try:
            va = cv2.VideoCapture(args.a_video)
            vb = cv2.VideoCapture(args.b_video)

            def grab(cap, idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, fr = cap.read()
                return fr if ok else None

            def draw(fr, box, W=620):
                fr = fr.copy()
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 0, 255), 4)
                s = W / fr.shape[1]
                return cv2.resize(fr, (W, int(fr.shape[0] * s)))

            rows_img = []
            for r in ev_rows:
                ed, ea = r["endpoints"].get("disp"), r["endpoints"].get("app")
                if not ed or not ea:
                    continue
                fd, bd = ed["f"], ed["box"]
                fa_, ba_ = ea["f"], ea["box"]
                capd = va if ed["cam"] == "a" else vb
                capa = va if ea["cam"] == "a" else vb
                imd = draw(grab(capd, fd), bd)
                ima = draw(grab(capa, fa_), ba_)
                H = max(imd.shape[0], ima.shape[0])
                imd = np.pad(imd, ((0, H - imd.shape[0]), (0, 0), (0, 0)), constant_values=255)
                ima = np.pad(ima, ((0, H - ima.shape[0]), (0, 0), (0, 0)), constant_values=255)
                gap = np.full((H, 12, 3), 235, np.uint8)
                body = np.hstack([imd, gap, ima])
                bar = np.full((46, body.shape[1], 3), 252, np.uint8)
                cv2.putText(bar, f"[{r['tier']}] {r['dirn']}  dt={r['dt']}s  cos={r['cos']}  "
                                 f"resid={_f(r['disp_resid'])}/{_f(r['app_resid'])}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
                cv2.putText(bar, f"{r['disp_tag']}#{r['disp_tid']} last f={fd}",
                            (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
                cv2.putText(bar, f"{r['app_tag']}#{r['app_tid']} first f={fa_}",
                            (body.shape[1] // 2 + 6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (60, 60, 60), 1, cv2.LINE_AA)
                rows_img.append(np.vstack([bar, body]))
            va.release()
            vb.release()
            if rows_img:
                W = max(r.shape[1] for r in rows_img)
                rows_img = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255)
                            for r in rows_img]
                vg = np.full((14, W, 3), 255, np.uint8)
                out_img = rows_img[0]
                for r in rows_img[1:]:
                    out_img = np.vstack([out_img, vg, r])
                cv2.imwrite(os.path.join(args.out_dir, "shortlist_evidence.png"), out_img)
                print(f"[out] {args.out_dir}/shortlist_evidence.png")
        except Exception as e:  # noqa: BLE001
            print("[warn] evidence failed:", str(e)[:160])


if __name__ == "__main__":
    main()
