"""在全 42 个配置上校验「嵌套 + 外观」合并的安全性（不动聚类，纯后处理）。

判据：
  - 轨迹级：合并对里 GT 同人 / 异人的比例（配置无关，只算一次）
  - 簇级（逐配置）：某次合并把两条轨迹所在的**不同簇**连起来时，
    这两个簇的主体身份是否相同。不同 → 记一次「错误并簇」。
  - 输出每个配置的：簇并合次数 / 其中错并次数 / 错并率。

用法：PY tools/validate_nested_merge.py [--appear 0.70] [--run results_gated]
输出：results_gated/nested_merge_validate.json
"""
import argparse
import collections
import json
import os
import sys

import eval_query_fusion as F
import fix_nested_tracks as N

DET_REL = "output_crosscam_local/1012_yolo8_dd"
EMB_REL = "output_crosscam_local/1012_yolo8_dd_emb"


def build_pairs(root, seg, appear):
    frames = N.load_frames(os.path.join(root, DET_REL), seg)
    pairs, _ = N.merge_pairs(frames)
    E = N.load_emb(os.path.join(root, EMB_REL), seg)
    out = []
    for t1, t2, nc, sh in pairs:
        k1, k2 = "%s#%d" % (seg, t1), "%s#%d" % (seg, t2)
        if k1 not in E or k2 not in E:
            continue
        c = float(E[k1] @ E[k2])
        if c >= appear:
            out.append((t1, t2, round(c, 3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--appear", type=float, default=0.70)
    ap.add_argument("--run", default="results_gated")
    ap.add_argument("--deliv", default="")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    root = os.path.dirname(os.path.dirname(deliv))
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)
    gtruth = F.load_gallery_truth(gt)
    qtruth = F.load_query_truth(gt)

    qrows = F.rd(os.path.join(gt, "gt_query_gt_FILLED.csv"))
    segs = sorted({r["seg"] for r in qrows})
    pairs = {s: build_pairs(root, s, a.appear) for s in segs}

    # 轨迹级精度
    print("=== 轨迹级（配置无关，τ=%.2f）===" % a.appear)
    tot_same = tot_diff = 0
    for s in segs:
        same = diff = unk = 0
        for t1, t2, c in pairs[s]:
            p1 = (qtruth.get("%s#%d" % (s, t1), "") or "").strip()
            p2 = (qtruth.get("%s#%d" % (s, t2), "") or "").strip()
            if p1 and p2 and p1 == p2:
                same += 1
            elif p1 and p2:
                diff += 1
            else:
                unk += 1
        tot_same += same
        tot_diff += diff
        print("  %-16s 合并 %2d：同人 %2d｜异人 %d｜含 UNK %d" % (s, len(pairs[s]), same, diff, unk))
    pr = tot_same / max(1, tot_same + tot_diff)
    print("  合计同人 %d / 异人 %d → 精度 %.3f" % (tot_same, tot_diff, pr))

    # 簇级（逐配置）
    print("\n=== 簇级：每个配置有多少次「并簇」，其中把不同身份的簇并起来的有几次 ===")
    print("%-14s %-8s %-8s %-8s" % ("配置", "并簇", "错并", "错并率"))
    rows = []
    for fn in sorted(os.listdir(os.path.join(run, "queries"))):
        if not fn.endswith(".csv"):
            continue
        cfg = fn[:-4]
        ident = F.build_ident(run, cfg, gtruth)
        sysrows = {r["query_id"]: r for r in F.rd(os.path.join(run, "queries", fn))}
        n_merge = n_bad = 0
        for s in segs:
            for t1, t2, c in pairs[s]:
                q1, q2 = "%s#%d" % (s, t1), "%s#%d" % (s, t2)
                r1, r2 = sysrows.get(q1), sysrows.get(q2)
                if not r1 or not r2:
                    continue
                if r1["final_status"] != "accepted" or r2["final_status"] != "accepted":
                    continue
                c1, c2 = int(r1["final_cluster_local"]), int(r2["final_cluster_local"])
                if c1 == c2:
                    continue
                n_merge += 1
                i1, i2 = ident.get(c1, ""), ident.get(c2, "")
                if i1 and i2 and i1 != i2:
                    n_bad += 1
        rate = n_bad / n_merge if n_merge else 0.0
        rows.append(dict(cfg=cfg, merges=n_merge, bad=n_bad, bad_rate=round(rate, 3)))
        print("%-14s %-8d %-8d %-8.2f" % (cfg, n_merge, n_bad, rate))

    base = [r for r in rows if r["cfg"] == "A_baseline"]
    print("\nA_baseline：并簇 %d，错并 %d" % (base[0]["merges"], base[0]["bad"]) if base else "")
    worst = sorted(rows, key=lambda r: -r["bad_rate"])[:5]
    print("错并率最高的 5 个配置：", [(r["cfg"], r["bad"], r["merges"]) for r in worst])
    json.dump(dict(appear=a.appear, track_precision=round(pr, 3),
                   track_same=tot_same, track_diff=tot_diff, configs=rows),
              open(os.path.join(run, "nested_merge_validate.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ results/nested_merge_validate.json")


if __name__ == "__main__":
    main()
