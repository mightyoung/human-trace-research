"""诊断「同一个人被切成多条轨迹 / 落进多个簇」的规模——计数多算的根因。

三块：
  A 图库侧：每个身份占几个簇（碎片度）；每个簇含几个身份（混人度）
  B 查询侧：已接受的轨迹里，同一身份被分到几个簇；同一簇里出现几个身份
  C 计数口径：窗口内「去重到身份」的人数 vs 现在「按簇计」的人数，看多算能收回多少

用法：PY tools/diag_fragmentation.py [--run results_gated] [--cfg A_baseline] [--window 60]
输出：results/fragmentation.json、results/fragmentation_identity.csv
"""
import argparse
import collections
import csv
import json
import os

import eval_query_fusion as F
import m3_counting as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results_gated")
    ap.add_argument("--deliv", default="")
    ap.add_argument("--cfg", default="A_baseline")
    ap.add_argument("--window", type=int, default=60)
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    root = os.path.dirname(os.path.dirname(deliv))
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)
    det_root = os.path.join(root, "output_crosscam_local/1012_yolo8_dd")

    gtruth = F.load_gallery_truth(gt)
    ident = F.build_ident(run, a.cfg, gtruth)
    mem = collections.defaultdict(list)
    for r in F.rd(os.path.join(run, "members", a.cfg + ".csv")):
        mem[int(r["cluster_local"])].append(r["track_id"])

    # ---- A 图库侧 ----
    id2cl = collections.defaultdict(set)
    for cl, ts in mem.items():
        for t in ts:
            p = gtruth.get(t)
            if p:
                id2cl[p].add(cl)
    frag = {p: len(cs) for p, cs in id2cl.items()}
    hist = collections.Counter(frag.values())
    print("=== A 图库侧（%d 簇 / %d 身份） ===" % (len(mem), len(id2cl)))
    print("同一身份落在几个簇：", dict(sorted(hist.items())))
    multi = {p: n for p, n in frag.items() if n >= 2}
    print("被拆成 ≥2 个簇的身份：%d / %d（%.0f%%）"
          % (len(multi), len(frag), 100 * len(multi) / max(1, len(frag))))
    top = sorted(frag.items(), key=lambda x: -x[1])[:8]
    print("拆分最多的身份：", top)

    # ---- B 查询侧 ----
    qtruth = F.load_query_truth(gt)
    sysrows = {r["query_id"]: r for r in F.rd(os.path.join(run, "queries", a.cfg + ".csv"))}
    acc = [r for r in sysrows.values() if r["final_status"] == "accepted"]
    qid2identity = collections.defaultdict(set)   # 身份 -> 被分到的簇
    cl2qid = collections.defaultdict(set)         # 簇 -> 出现的身份
    rowscsv = []
    for r in acc:
        cl = int(r["final_cluster_local"])
        pid = qtruth.get(r["query_id"], "")
        rowscsv.append((r["query_id"], r["cam"], cl, pid, ident.get(cl, "")))
        if pid:
            qid2identity[pid].add(cl)
        if ident.get(cl):
            cl2qid[cl].add(pid or "")
    print("\n=== B 查询侧（已接受 %d 条） ===" % len(acc))
    frag_q = {p: len(cs) for p, cs in qid2identity.items()}
    print("同一身份在查询里被分到几个簇：", dict(sorted(collections.Counter(frag_q.values()).items())))
    print("含 ≥2 个身份的簇：%d / %d"
          % (sum(1 for v in cl2qid.values() if len(v) >= 2), len(cl2qid)))

    # ---- C 计数口径 ----
    W = a.window * 25
    print("\n=== C 计数：按簇计 vs 去重到身份（窗口 %ds） ===" % a.window)
    print("%-16s %-6s %-8s %-9s %-9s %-8s"
          % ("seg", "窗数", "GT_hi", "按簇(现)", "去重身份", "去重后MAE"))
    tot = collections.Counter()
    for seg in sorted({r["seg"] for r in F.rd(os.path.join(gt, "gt_query_gt_FILLED.csv"))}):
        span = C.load_spans(det_root, seg)
        qs = [r for r in F.rd(os.path.join(gt, "gt_query_gt_FILLED.csv")) if r["seg"] == seg]
        lo_f = min(v[0] for v in span.values())
        hi_f = max(v[1] for v in span.values())
        nwin = int((hi_f - lo_f) // W) + 1
        s_cl = s_id = s_gt = 0
        mae_id = 0.0
        for wi in range(nwin):
            w0, w1 = lo_f + wi * W, lo_f + (wi + 1) * W
            present = []
            for r in qs:
                tid = int(r["query_id"].split("#")[1])
                sp = span.get(tid)
                if not sp or sp[1] < w0 or sp[0] > w1:
                    continue
                pid = (r["person_id"] or "").strip()
                if C.is_nonperson(pid):
                    continue
                s = sysrows.get(r["query_id"], {})
                ok = s.get("final_status") == "accepted"
                present.append((tid, pid, ok, int(s["final_cluster_local"]) if ok else None))
            gt_keys = {(pid if pid else "unk:%d" % tid) for tid, pid, _, _ in present}
            clusters = {cl for _, _, ok, cl in present if ok and cl is not None}
            ids = set()
            n_noid = 0
            for cl in clusters:
                cid = ident.get(cl, "")
                if cid:
                    ids.add(cid)
                else:
                    n_noid += 1
            dedup = len(ids) + n_noid
            s_cl += len(clusters)
            s_id += dedup
            s_gt += len(gt_keys)
            mae_id += abs(dedup - len(gt_keys))
        n = max(1, nwin)
        print("%-16s %-6d %-8.1f %-9.1f %-9.1f %-8.2f"
              % (seg, nwin, s_gt / n, s_cl / n, s_id / n, mae_id / n))
        tot["gt"] += s_gt
        tot["cl"] += s_cl
        tot["id"] += s_id
        tot["mae_id"] += mae_id
        tot["n"] += nwin
    n = max(1, tot["n"])
    print("%-16s %-6d %-8.1f %-9.1f %-9.1f %-8.2f"
          % ("合计", tot["n"], tot["gt"] / n, tot["cl"] / n, tot["id"] / n,
             tot["mae_id"] / n))
    print("\n→ 去重到身份把多算收回 %.1f 人/窗（按簇 %.2f → 去重 %.2f）"
          % ((tot["cl"] - tot["id"]) / n, tot["cl"] / n, tot["id"] / n))

    json.dump(dict(cfg=a.cfg, run=run,
                   gallery_identity_split=dict(hist.items()),
                   gallery_multi=multi,
                   query_identity_split=dict(collections.Counter(frag_q.values()).items()),
                   windows=dict(n=tot["n"], gt=round(tot["gt"] / n, 3),
                                clusters=round(tot["cl"] / n, 3),
                                dedup=round(tot["id"] / n, 3),
                                mae_dedup=round(tot["mae_id"] / n, 3))),
              open(os.path.join(run, "fragmentation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(os.path.join(run, "fragmentation_identity.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "cam", "cluster", "q_pid", "c_pid"])
        w.writerows(rowscsv)
    print("→ results/fragmentation.{json,_identity.csv}")


if __name__ == "__main__":
    main()
