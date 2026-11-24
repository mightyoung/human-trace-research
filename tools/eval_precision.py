"""把 D3（图库逐成员真值）+ D4（查询逐条真值）接到 D7 各配置的判决上，算精确度。

口径（与 data-request-v2 D4 派生规则一致）：
  correct = (final_cluster 的主体身份 == query 的 person_id)

  主体身份：该簇已标注成员按 person_id 投票的多数派。用**逐轨迹**真值而非簇级真值，
  因此聚类配置改变（切开/否决）后投票自动重算，不需要把簇号在配置间映射。
  query 无 person_id（看不清）或簇无任何已标成员 → 记 U，不进 C/W。

输出：results/precision.csv
用法：PY tools/eval_precision.py [--run results|results/robust]
"""
import argparse
import collections
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(ROOT, "deliverable", "data-request-v2")
GT = os.path.join(DELIV, "gt")


def load_gallery_truth():
    """track_id -> person_id（D3 逐成员真值；空与 UNK 不计）"""
    p = os.path.join(GT, "gt_gallery_members_FILLED.csv")
    truth = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        pid = r["person_id"].strip()
        if pid and pid.upper() != "UNK":
            truth[r["track_id"]] = pid
    return truth


def load_query_truth():
    p = os.path.join(GT, "gt_query_gt_FILLED.csv")
    truth = {}
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        pid = r["person_id"].strip()
        truth[r["query_id"]] = ("" if pid.upper() == "UNK" else pid)
    return truth


def is_nonperson(pid):
    """no-human / shadow* = 影子·镜像·物件（323 远场、318 挂帽椅子）。"""
    p = (pid or "").strip().lower()
    return p == "no-human" or p.startswith("shadow")


def load_members(run):
    """config -> {cluster_local: [track_id,...]}"""
    d = os.path.join(run, "members")
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".csv"):
            continue
        cfg = fn[:-4]
        m = collections.defaultdict(list)
        for r in csv.DictReader(open(os.path.join(d, fn), encoding="utf-8-sig")):
            m[int(r["cluster_local"])].append(r["track_id"])
        out[cfg] = m
    return out


def cluster_identity(members, gtruth, strict=False):
    """返回 (主体身份, 已标数, 身份数, 主体占比, 是否混人)

    strict=True 时启用「平票记 U」：若最高票存在并列（第二名票数等于第一名），
    该簇没有唯一主体身份，返回空身份 → 落到它的查询记 U。
    """
    votes = collections.Counter()
    for t in members:
        p = gtruth.get(t)
        if p:
            votes[p] += 1
    n_lab = sum(votes.values())
    if not votes:
        return "", 0, 0, 0.0, False
    ranked = votes.most_common()
    pid, n_top = ranked[0]
    if strict and len(ranked) >= 2 and ranked[1][1] == n_top:
        return "", n_lab, len(votes), n_top / n_lab, len(votes) >= 2
    return pid, n_lab, len(votes), n_top / n_lab, len(votes) >= 2


def eval_cfg(cfg, members, gtruth, qtruth, per_q_rows, strict=False):
    """返回 (C, W, U, accepted, 明细)

    strict=True 启用论文严格口径（两条，其余不变）：
      1) 平票簇记 U —— 簇内最高票并列时不认定主体身份；
      2) 真身份不在图库者剔除 —— 该查询连「接受」都不计（不是 U，是移出评估面）。
    默认 False = 历史宽口径（平票取 Counter 首个、图库外查询照常计入），用于复现旧数。
    """
    gal_ids = set(gtruth.values()) if strict else None
    ident = {}
    for cl, mem in members.items():
        ident[cl] = cluster_identity(mem, gtruth, strict=strict)
    C = W = U = acc = 0
    detail = []
    for r in per_q_rows:
        if r["final_status"] != "accepted":
            continue
        qp = qtruth.get(r["query_id"], "")
        if is_nonperson(qp):
            continue          # 影子/镜像/物件：不判对错、不计接受
        if strict and qp and qp not in gal_ids:
            continue          # 真身份不在图库：移出评估面（不计接受、不计 U）
        acc += 1
        cl = int(r["final_cluster_local"])
        cp, n_lab, n_ids, share, mixed = ident.get(cl, ("", 0, 0, 0.0, False))
        if not qp or not cp:
            U += 1
            verdict = "U"
        elif qp == cp:
            C += 1
            verdict = "C"
        else:
            W += 1
            verdict = "W"
        detail.append(dict(query_id=r["query_id"], cam=r["cam"], cluster_local=cl,
                           q_pid=qp, c_pid=cp, verdict=verdict, cluster_n_lab=n_lab,
                           cluster_n_ids=n_ids, top_share=round(share, 3),
                           cluster_mixed=int(mixed)))
    return C, W, U, acc, detail


def main():
    global DELIV, GT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results")
    ap.add_argument("--deliv", default="", help="数据包根目录（默认 deliverable/data-request-v2），换跟踪器后的新运行可指向别处")
    ap.add_argument("--gt-dir", default="", help="真值目录（默认 <deliv>/gt）")
    ap.add_argument("--c-denom", type=float, default=229.0,
                    help="C/denom 的分母。默认 229；换跟踪器后查询总体变化时按新可答数传（列名保持不变）")
    ap.add_argument("--out-dir", default="",
                    help="结果写到这个目录（默认 <deliv>/<run>）。做真值变体实验时必须指定，避免覆盖冻结结果")
    ap.add_argument("--strict", action="store_true",
                    help="论文严格口径：平票簇记 U + 真身份不在图库者剔除。"
                         "默认关闭=历史宽口径（复现旧数用）。"
                         "两者在 A_baseline 上的差：宽 124/118/6/0 vs 严格 123/117/4/2")
    a = ap.parse_args()
    if a.deliv:
        DELIV = os.path.abspath(a.deliv)
        GT = os.path.abspath(a.gt_dir) if a.gt_dir else os.path.join(DELIV, "gt")
    run = os.path.join(DELIV, a.run)
    outdir = os.path.abspath(a.out_dir) if a.out_dir else run
    # 防呆：换了真值目录却没换输出目录，等于拿变体真值覆盖交付包的冻结结果（2026-10-24 踩过一次）
    if a.gt_dir and not a.out_dir and (os.sep + "deliverable" + os.sep) in (outdir + os.sep):
        raise SystemExit(
            f"拒绝写回冻结交付包：{outdir}\n"
            f"  用了 --gt-dir {a.gt_dir}，输出仍指向 deliverable/。做变体实验请加 --out-dir <临时目录>。")
    gtruth = load_gallery_truth()
    qtruth = load_query_truth()
    allmem = load_members(run)
    summ = {r["config"]: r for r in
            csv.DictReader(open(os.path.join(run, "summary.csv"), encoding="utf-8-sig"))}

    print(f"图库真值 {len(gtruth)} 条轨迹｜查询 {sum(1 for v in qtruth.values() if v)} 条有身份"
          f"（{sum(1 for v in qtruth.values() if not v)} 条看不清）｜C 分母 = {int(a.c_denom)}")

    out = []
    for cfg in sorted(allmem):
        qpath = os.path.join(run, "queries", f"{cfg}.csv")
        rows = list(csv.DictReader(open(qpath, encoding="utf-8-sig")))
        C, W, U, acc, detail = eval_cfg(cfg, allmem[cfg], gtruth, qtruth, rows,
                                        strict=a.strict)
        s = summ.get(cfg, {})
        rec = dict(config=cfg, K=s.get("K", ""), theta=s.get("theta", ""),
                   coverage=s.get("both_accept_rate", ""),
                   final_output_rate=s.get("final_output_rate", ""),
                   accepted=acc, C=C, W=W, U=U,
                   precision=round(C / (C + W), 4) if C + W else "",
                   recall_229=round(C / a.c_denom, 4), C_over_229=C)
        out.append(rec)
        # 明细单独存，便于逐条复核
        dp = os.path.join(outdir, "verdicts", f"{cfg}.csv")
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        with open(dp, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(detail[0].keys()) if detail else
                               ["query_id", "cam", "cluster_local", "q_pid", "c_pid",
                                "verdict", "cluster_n_lab", "cluster_n_ids",
                                "top_share", "cluster_mixed"])
            w.writeheader()
            w.writerows(detail)

    op = os.path.join(outdir, "precision.csv")
    with open(op, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"[out] {op}  {len(out)} 配置\n")

    print(f"{'配置':<30}{'K':>4}{'θ':>6}{'cov':>7}{'acc':>5}{'C':>4}{'W':>4}{'U':>4}{'精度':>7}{'C/den':>7}")
    for r in out:
        prec = f"{r['precision']:.3f}" if r["precision"] != "" else "—"
        print(f"{r['config']:<30}{r['K']:>4}{float(r['theta']):>6.2f}"
              f"{float(r['coverage']):>7.3f}{r['accepted']:>5}{r['C']:>4}{r['W']:>4}{r['U']:>4}"
              f"{prec:>7}{r['recall_229']:>7.3f}")


if __name__ == "__main__":
    main()
