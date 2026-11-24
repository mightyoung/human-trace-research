#!/usr/bin/env python
"""倒影几何审计：对候选短名单做"真人-倒影"配对检查，产出最终分级短名单。

背景（2026-10-12 实测）：
  本场景多个机位存在**地板/玻璃倒影轨迹**——倒影在画面中位于真人框的**正下方**且尺寸相近、
  底边与真人框顶边精确衔接（ByteTrack 会把倒影单独立轨）。跨机转移候选若其任一端点是
  倒影轨迹，则该候选为"倒影伪对"（含 FAST_CHECK 集与部分 TIER 候选）。

判定（启发式，保守）：
  对候选的两个端点轨迹，在端点帧（±10 帧窗口）内查找"正上方衔接的兄弟轨迹"：
    - 尺寸比 >= 0.5；水平对齐 |Δcx| <= 0.35*max(w)；
    - 兄弟框底边与自身框顶边间距 <= max(0.5*h, 40px)。
  命中 → 该端点标记 reflection_suspect。

用法：
  python tools/reflection_audit.py \
    --filter-json output_crosscam/analysis/spread/depth_filter/depth_filter.json \
    --a-jsonl output_crosscam/316_detections.jsonl \
    --b-jsonl output_crosscam/323_detections.jsonl \
    --out-csv output_crosscam/analysis/spread/depth_filter/shortlist_audited.csv
"""
import argparse
import csv
import json
from pathlib import Path


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


def frames_map(tracks):
    m = {}
    for tid, lst in tracks.items():
        for f, b in lst:
            m.setdefault(f, []).append((tid, b))
    return m


def find_sibling(tracks, fmap, tid, end_frame, window=(0, -5, 5, -10, 10)):
    """在端点帧±窗口内查找倒影几何的兄弟轨迹；返回 (frame, sibling_tid, sibling_box) 或 None"""
    lst = tracks.get(tid)
    if not lst:
        return None
    box_at = {f: b for f, b in lst}
    for w in window:
        f = end_frame + w
        b = box_at.get(f)
        if b is None:
            continue
        x1, y1, x2, y2 = b
        w0, h0 = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        for t2, b2 in fmap.get(f, []):
            if t2 == tid:
                continue
            bx1, by1, bx2, by2 = b2
            bw, bh = bx2 - bx1, by2 - by1
            if bh < 0.5 * h0:
                continue
            if abs((bx1 + bx2) / 2 - cx) > 0.35 * max(w0, bw):
                continue
            gap = y1 - by2  # 兄弟底边到自身顶边
            if -0.25 * h0 <= gap <= max(0.5 * h0, 40):
                return (f, t2, b2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-json", required=True)
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.filter_json).read_text(encoding="utf-8"))
    ta = load_tracks(args.a_jsonl)
    tb = load_tracks(args.b_jsonl)
    fma, fmb = frames_map(ta), frames_map(tb)

    rows = []
    for r in d["rows"]:
        if not r.get("tier"):
            continue
        rec = dict(r)
        for side in ("disp", "app"):
            tag = r[f"{side}_tag"]
            tid = r[f"{side}_tid"]
            end = r["endpoints"].get(side)
            tracks = ta if tag == "316" else tb
            fmap = fma if tag == "316" else fmb
            sib = find_sibling(tracks, fmap, tid, end["f"]) if end else None
            rec[f"{side}_sib"] = bool(sib)
            rec[f"{side}_sib_tid"] = sib[1] if sib else None
            rec[f"{side}_sib_gap"] = round(end["box"][1] - sib[2][3], 1) if sib and end else None
        rec["refl_any"] = bool(rec["disp_sib"] or rec["app_sib"])
        rec["refl_both"] = bool(rec["disp_sib"] and rec["app_sib"])
        rows.append(rec)

    tier_order = {"FAST_CHECK": 0, "TIER1": 1, "TIER2": 2}
    rows.sort(key=lambda r: (tier_order.get(r["tier"], 9), r["refl_any"], -(r["cos"] or 0)))

    out_csv = Path(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tier", "dirn", "disp", "app", "dt", "cos", "disp_resid", "app_resid",
                    "disp_sib", "disp_sib_tid", "app_sib", "app_sib_tid", "refl_any", "refl_both",
                    "disp_glass", "app_glass"])
        for r in rows:
            w.writerow([
                r["tier"], r["dirn"],
                f"{r['disp_tag']}#{r['disp_tid']}", f"{r['app_tag']}#{r['app_tid']}",
                r["dt"], r["cos"],
                round(r["disp_resid"], 3) if r["disp_resid"] is not None else "",
                round(r["app_resid"], 3) if r["app_resid"] is not None else "",
                int(r["disp_sib"]), r["disp_sib_tid"] or "",
                int(r["app_sib"]), r["app_sib_tid"] or "",
                int(r["refl_any"]), int(r["refl_both"]),
                int(r["disp_glass"]), int(r["app_glass"]),
            ])
    print(f"[out] {out_csv}")

    # 汇总打印
    from collections import Counter
    print(f"\n=== 审计结果（{Path(args.filter_json).name}）===")
    for tier in ("FAST_CHECK", "TIER1", "TIER2"):
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        c = Counter(("both" if r["refl_both"] else ("any" if r["refl_any"] else "clean")) for r in sub)
        print(f"[{tier}] n={len(sub)}  双端倒影={c.get('both',0)} 单端倒影={c.get('any',0)} 单端干净={c.get('clean',0)}")

    print(f"\n=== 各 tier 明细（clean 在前）===")
    for r in rows:
        flag = "REFL" if r["refl_both"] else ("refl?" if r["refl_any"] else "clean")
        sib_d = f"~{r['disp_sib_tid']}" if r["disp_sib"] else ""
        sib_a = f"~{r['app_sib_tid']}" if r["app_sib"] else ""
        print(f"  [{r['tier']:>10}] [{flag:>5}] {r['dirn']}  "
              f"{r['disp_tag']}#{r['disp_tid']}{sib_d} -> {r['app_tag']}#{r['app_tid']}{sib_a}  "
              f"dt={r['dt']}s cos={r['cos']}")


if __name__ == "__main__":
    main()
