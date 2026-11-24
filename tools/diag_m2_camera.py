"""M2 追加：按机位校准能不能带来「精度不降」的增益（交叉验证）。

做法：规则族 = 基线接受 ∪「取两侧高分片，只要分数 ≥ θ + δ[cam]」。
      每个机位单独扫 δ；在机位内按查询序号奇偶切成两半，
      一半调 δ（取 C 最大且整体精度 ≥ 基线精度的 δ），另一半评效果，再交换。
      两次留出增益若符号不一致，就说明是过拟合。

输出：results/m2_camera_cv.json
"""
import argparse
import json
import os

import eval_query_fusion as F

GRID = [round(-0.6 + 0.1 * i, 1) for i in range(13)]


def apply_rule(rows, ident, qtruth, delta, base_only=False):
    """基线（可选：附加「取高分片，分数 ≥ θ + δ[cam]」）；返回 C/W/U 与逐条 verdict。"""
    C = W = U = 0
    det = {}
    for r in rows:
        qid = r["query_id"]
        ca, cb = r["cluster_a_local"], r["cluster_b_local"]
        sa, sb = F.fnum(r["score_a"]), F.fnum(r["score_b"])
        aa, ab = r["accept_a"] == "1", r["accept_b"] == "1"
        same = ca == cb and ca != ""
        qp = qtruth.get(qid, "")
        if aa and ab and same:
            cl, tag = int(ca), "both"
        elif base_only:
            continue
        else:
            th = 1.85 + delta.get(r["cam"], 0.0)
            if max(sa, sb) < th:
                continue
            cl = int(ca) if sa >= sb else int(cb)
            tag = "add"
        cp = ident.get(cl, "")
        if not qp or not cp:
            U += 1
            v = "U"
        elif qp == cp:
            C += 1
            v = "C"
        else:
            W += 1
            v = "W"
        det[qid] = v
    return C, W, U, det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results")
    ap.add_argument("--cfg", default="A_baseline")
    a = ap.parse_args()
    deliv = F.DELIV
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)
    gtruth = F.load_gallery_truth(gt)
    qtruth = F.load_query_truth(gt)
    ident, _ = F.build_ident(run, a.cfg, gtruth, with_share=True)
    rows = F.rd(os.path.join(run, "queries", a.cfg + ".csv"))

    bC, bW, bU, base_det = apply_rule(rows, ident, qtruth, {}, base_only=True)
    base_prec = bC / (bC + bW)
    print("基线 C %d / W %d / U %d｜精度 %.4f" % (bC, bW, bU, base_prec))

    cams = sorted({r["cam"] for r in rows})
    # 机位内奇偶分半
    halves = {}
    for cam in cams:
        sub = [r for r in rows if r["cam"] == cam]
        halves[cam] = (sub[0::2], sub[1::2])

    out = {"base": dict(C=bC, W=bW, U=bU, precision=round(base_prec, 4)), "cams": {}}
    for cam in cams:
        h1, h2 = halves[cam]
        print("\n机位 %s（分半 n=%d/%d）" % (cam, len(h1), len(h2)))
        res = {}
        for name, tune, test in (("奇数调→偶数评", h1, h2), ("偶数调→奇数评", h2, h1)):
            # tune 段：只在该段内挑 δ，使该段 C 最大且该段精度 ≥ 基线精度
            rows_t = [r for r in rows if r in tune]
            best = None
            bestd = None
            for d in GRID:
                C, W, U, _ = apply_rule(rows_t, ident, qtruth, {cam: d}, None)
                if C + W == 0:
                    continue
                pr = C / (C + W)
                if pr + 1e-9 < base_prec:
                    continue
                if best is None or C > best[0]:
                    best, bestd = (C, W, pr), d
            if best is None:
                print("  %s：调参段无可行 δ（任何 δ 都过不了精度约束）" % name)
                res[name] = dict(delta=None, holdout_dC=0, holdout_dW=0,
                                 full_dC=0, full_dW=0, infeasible=True)
                continue
            dsel = 0.0 if bestd is None else bestd
            # 全量用该 δ 评估该机位贡献
            C0, W0, _, det0 = apply_rule(rows, ident, qtruth, {}, base_only=True)
            C1, W1, _, det1 = apply_rule(rows, ident, qtruth, {cam: dsel})
            dc, dw = C1 - C0, W1 - W0
            tested = [r for r in rows if r in test]
            # 只在留出段上算增量
            dc_test = sum(1 for r in tested
                          if det1.get(r["query_id"]) == "C" and det0.get(r["query_id"]) != "C")
            dw_test = sum(1 for r in tested
                          if det1.get(r["query_id"]) == "W" and det0.get(r["query_id"]) != "W")
            print("  %s：tune 选 δ=%+.1f（该段 C %d W %d 精度 %.3f）｜留出段 ΔC %+d ΔW %+d"
                  % (name, dsel, best[0], best[1], best[2], dc_test, dw_test))
            res[name] = dict(delta=dsel, holdout_dC=dc_test, holdout_dW=dw_test,
                             full_dC=dc, full_dW=dw)
        out["cams"][cam] = res

    print("\n=== 判读 ===")
    stable = 0
    for cam, r in out["cams"].items():
        ds = [r["奇数调→偶数评"]["holdout_dC"], r["偶数调→奇数评"]["holdout_dC"]]
        ws = [r["奇数调→偶数评"]["holdout_dW"], r["偶数调→奇数评"]["holdout_dW"]]
        good = all(d > 0 for d in ds) and all(w <= 0 for w in ws)
        stable += good
        print("  %s 留出增益两次分别为 C %+d/%+d、W %+d/%+d → %s"
              % (cam, ds[0], ds[1], ws[0], ws[1], "稳定" if good else "不稳定（过拟合）"))
    print("  稳定机位 %d/%d → %s" % (stable, len(cams),
                                 "分机位校准可用" if stable >= 3 else "分机位校准不成立"))
    json.dump(out, open(os.path.join(run, "m2_camera_cv.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ results/m2_camera_cv.json")


if __name__ == "__main__":
    main()
