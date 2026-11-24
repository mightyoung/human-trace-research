"""M3-B 计数基线：按 60 秒窗口比「在场人数」。

真值来源（关键前提）：test 段的**检测轨迹数与查询数完全相同**（305 102/102、318 78/78、
320 40/40、323 75/75），即查询集 = 该段全部轨迹。因此每条轨迹都有跨度（起止帧）与
尽量标注的身份，可以重建「某窗口内有哪些人在场」。

口径：
  GT_hi   = 窗口内出现过的身份数（已标身份去重；UNK 轨迹各自算 1 人）
  GT_lo   = 只算已标身份（UNK 全不计）—— 323 的 UNK 占 41%，必须同时报上下界
  系统计数 = 窗口内**已接受**查询所属簇的去重数（一个簇 = 系统认为的一个人）

指标：MAE（对 GT_hi / GT_lo）、偏差（系统−GT）、漏计归因（图库外 / 被拒 / 非人）。

用法：PY tools/m3_counting.py [--window 60] [--run results] [--cfg A_baseline]
输出：results/m3_counting.json、results/m3_counting_windows.csv
"""
import argparse
import collections
import csv
import json
import os

import eval_query_fusion as F

DET_REL = "output_crosscam_local/1012_yolo8_dd"
FPS = 25.0


def is_nonperson(pid):
    p = (pid or "").strip().lower()
    return p == "no-human" or p.startswith("shadow")


def load_spans(det_root, seg):
    span = {}
    for line in open(os.path.join(det_root, seg, "%s_detections.jsonl" % seg),
                     encoding="utf-8"):
        r = json.loads(line)
        f = r["frame"]
        for t in r["track_ids"]:
            t = int(t)
            a, b = span.get(t, (f, f))
            span[t] = (min(a, f), max(b, f))
    return span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results")
    ap.add_argument("--deliv", default="")
    ap.add_argument("--cfg", default="A_baseline")
    ap.add_argument("--window", type=int, default=60, help="窗口秒数")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    root = os.path.dirname(os.path.dirname(deliv))
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)
    det_root = os.path.join(root, DET_REL)

    gtruth = F.load_gallery_truth(gt)
    ident = F.build_ident(run, a.cfg, gtruth)
    qrows = F.rd(os.path.join(gt, "gt_query_gt_FILLED.csv"))
    sysrows = {r["query_id"]: r for r in F.rd(os.path.join(run, "queries", a.cfg + ".csv"))}

    segs = sorted({r["seg"] for r in qrows})
    W = a.window * FPS
    out_rows = []
    agg = collections.Counter()

    for seg in segs:
        span = load_spans(det_root, seg)
        qs = [r for r in qrows if r["seg"] == seg]
        lo_f = min(v[0] for v in span.values())
        hi_f = max(v[1] for v in span.values())
        nwin = int((hi_f - lo_f) // W) + 1
        for wi in range(nwin):
            w0, w1 = lo_f + wi * W, lo_f + (wi + 1) * W
            present = []      # (track_id, pid, accepted, cluster)
            for r in qs:
                tid = int(r["query_id"].split("#")[1])
                sp = span.get(tid)
                if not sp or sp[1] < w0 or sp[0] > w1:
                    continue
                pid = (r["person_id"] or "").strip()
                if is_nonperson(pid):
                    continue
                s = sysrows.get(r["query_id"], {})
                acc = s.get("final_status") == "accepted"
                cl = int(s["final_cluster_local"]) if acc else None
                present.append((tid, pid, acc, cl))

            gt_keys = {(pid if pid else "unk:%d" % tid) for tid, pid, _, _ in present}
            gt_lo = {pid for _, pid, _, _ in present if pid}
            sys_keys = {cl for _, _, acc, cl in present if acc and cl is not None}
            # 漏计归因（按人）
            miss_go = miss_rej = 0
            for key in gt_keys:
                if not key.startswith("unk:"):
                    covered = any(acc for tid, pid, acc, _ in present
                                  if (pid or "unk:%d" % tid) == key)
                    if covered:
                        continue
                    if key not in gtruth.values():
                        miss_go += 1
                    else:
                        miss_rej += 1
            out_rows.append(dict(seg=seg, window=wi, t0=round(w0 / FPS, 1),
                                 t1=round(w1 / FPS, 1), n_tracks=len(present),
                                 gt_hi=len(gt_keys), gt_lo=len(gt_lo), sys=len(sys_keys),
                                 err_hi=len(sys_keys) - len(gt_keys),
                                 err_lo=len(sys_keys) - len(gt_lo),
                                 miss_gal_out=miss_go, miss_rejected=miss_rej))
            agg["hi"] += abs(len(sys_keys) - len(gt_keys))
            agg["lo"] += abs(len(sys_keys) - len(gt_lo))
            agg["bias_hi"] += len(sys_keys) - len(gt_keys)
            agg["bias_lo"] += len(sys_keys) - len(gt_lo)
            agg["n"] += 1
            agg["gt_hi"] += len(gt_keys)
            agg["gt_lo"] += len(gt_lo)
            agg["sys"] += len(sys_keys)

    n = max(1, agg["n"])
    print("窗口 %d 个（%ds/窗）｜GT_hi 合计 %d｜GT_lo 合计 %d｜系统合计 %d"
          % (agg["n"], a.window, agg["gt_hi"], agg["gt_lo"], agg["sys"]))
    print("MAE vs GT_hi = %.2f 人/窗｜MAE vs GT_lo = %.2f 人/窗"
          % (agg["hi"] / n, agg["lo"] / n))
    print("偏差（系统−GT）：vs hi %+.2f｜vs lo %+.2f 人/窗"
          % (agg["bias_hi"] / n, agg["bias_lo"] / n))
    print("漏计归因：图库外 %d 人次｜被拒 %d 人次"
          % (sum(r["miss_gal_out"] for r in out_rows),
             sum(r["miss_rejected"] for r in out_rows)))

    print("\n分机位：")
    print("%-16s %-5s %-7s %-7s %-7s %-6s %-6s"
          % ("seg", "窗数", "GT_hi", "GT_lo", "系统", "MAE", "偏差"))
    by_seg = collections.defaultdict(list)
    for r in out_rows:
        by_seg[r["seg"]].append(r)
    for seg in segs:
        sub = by_seg[seg]
        mae = sum(abs(x["err_hi"]) for x in sub) / len(sub)
        bias = sum(x["err_hi"] for x in sub) / len(sub)
        print("%-16s %-5d %-7.1f %-7.1f %-7.1f %-6.2f %+.2f"
              % (seg, len(sub), sum(x["gt_hi"] for x in sub) / len(sub),
                 sum(x["gt_lo"] for x in sub) / len(sub),
                 sum(x["sys"] for x in sub) / len(sub), mae, bias))

    json.dump(dict(window_s=a.window, n=agg["n"], mae_hi=round(agg["hi"] / n, 3),
                   mae_lo=round(agg["lo"] / n, 3),
                   bias_hi=round(agg["bias_hi"] / n, 3),
                   bias_lo=round(agg["bias_lo"] / n, 3),
                   gt_hi=agg["gt_hi"], gt_lo=agg["gt_lo"], sys=agg["sys"],
                   miss_gal_out=sum(r["miss_gal_out"] for r in out_rows),
                   miss_rejected=sum(r["miss_rejected"] for r in out_rows)),
              open(os.path.join(run, "m3_counting.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(run, "m3_counting_windows.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print("\n→ results/m3_counting.{json,_windows.csv}")


if __name__ == "__main__":
    main()
