#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 79.3% 里的「人脸识别到谁」从真值换成**真实识别**，量实际折损。

背景
----
`exp_pose_pool.py`（8 段 / 445 具名轨 / 26 身份）报出：库均值 56.4% → 形态加权 68.2%
→ 库+实时池 **79.3%**（τ=0.80，recall，分母 179）。但它顶部口径写明：

    「人脸『识别到谁』用 GT 身份 → 这是**上限**；
      『这条轨能否写入池』用真实可用脸（face_vecs 有 |full）→ 这部分现实」

本脚本把这两处都换成实测：

1. **可用门**：`face8/face_track_*.csv` 的 `usable≥1`（pd≥20 且 fw≥32 且 lap≥50 且
   |yaw|≤0.35 且 0.25≤pitch≤0.95）→ 8 段 1178 轨里只有 **77 条 = 6.5%**，
   而原口径用「有 |full 向量」= 141/445 = **31.7%**。
2. **识别**：人脸库 = 上午(train) 每「机位×身份」的可用脸均值原型；
   查询轨的人脸对同机位原型取最近邻，需 ≥ τ_face 且领先次优 ≥ margin，
   否则视为「没认出」→ 不写池。

四种写池（同一份因果主循环，只有写入规则不同）
---------------------------------------------
  gt         写 = 有任意脸向量，身份 = GT          ← 原口径（79.3%）
  gt_usable  写 = 有可用脸，    身份 = GT          ← 只补覆盖率这一半
  real       写 = 有可用脸 + 认出，身份 = 识别结果  ← 完整现实
  real_oracle_cov  写 = 有可用脸 + 认出，身份 = GT  ← 分离「认错」与「漏认」

用法
----
  python tools/exp_pose_pool_real.py \
    --emb-dir tmp/full/emb12 --face tmp/full/face8 --det output_crosscam_local/1012_yolo8_dd \
    --gt deliverable/paper-package-2026-11-10/20-data/e4-full-corpus/gt_8seg.json \
    --segs "305_test_544 305_train_529 318_test_534 318_train_525 \
            320_test_518 320_train_514 323_test_527 323_train_510"
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp", "full"))
from cluster_pos import load_emb                          # noqa: E402
from exp_pose_pool import load_det_meta, topk_mean, cam_of  # noqa: E402


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def load_face_proto(face_dir, proto="full"):
    """返回 (vecs{key: vec}, usable{key})。

    vecs 只收 tag==proto；usable 来自 face_track_*.csv 的 usable>=1（逐帧质量门通过数）。
    """
    vecs = {}
    for p in sorted(glob.glob(os.path.join(face_dir, "face_vecs_*.npz"))):
        z = np.load(p, allow_pickle=True)
        for k, v in zip(z["keys"], z["vecs"]):
            k = str(k)
            if "|" not in k:
                continue
            key, tag = k.rsplit("|", 1)
            if tag != proto:
                continue
            v = np.asarray(v, dtype=np.float64)
            n = float(np.linalg.norm(v))
            if n > 0:
                vecs[key] = v / n
    usable = set()
    for p in sorted(glob.glob(os.path.join(face_dir, "face_track_*.csv"))):
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    if float(r.get("usable") or 0) >= 1:
                        usable.add("%s#%d" % (r["vol"], int(r["tid"])))
                except (KeyError, ValueError):
                    continue
    return vecs, usable


def build_gallery(rows, fvec, usable, eidx, min_id_tracks=1):
    """上午(train) 每「机位×身份」的人脸原型 = 该组可用脸轨的均值。"""
    acc = collections.defaultdict(list)
    for r in rows:
        if not r["tr"]:
            continue
        if r["k"] not in usable or r["k"] not in fvec:
            continue
        acc[(r["cam"], r["pid"])].append(fvec[r["k"]])
    gal = {}
    for key, vs in acc.items():
        if len(vs) < min_id_tracks:
            continue
        v = np.mean(np.stack(vs), 0)
        gal[key] = v / max(np.linalg.norm(v), 1e-9)
    return gal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--segs", required=True)
    ap.add_argument("--proto", default="full")
    ap.add_argument("--pool-k", type=int, default=3)
    ap.add_argument("--pool-max", type=int, default=8)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85])
    ap.add_argument("--main-tau", type=float, default=0.80)
    ap.add_argument("--sigma", type=float, default=0.6, help="形态加权带宽（S5）")
    ap.add_argument("--tau-face", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40, 0.45])
    ap.add_argument("--face-margin", type=float, default=0.02)
    ap.add_argument("--gallery-mode", default="train", choices=["train", "loo"],
                    help="train=只用上午可用脸建原型（生产冷启动）；"
                         "loo=用该身份其余全部可用脸（含下午，留一，测「库补齐后」的识别精度）")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-queries", default="", help="导出逐查询命中明细 CSV（179 行）")
    ap.add_argument("--dump-pools", default="", help="导出实测人脸写入池 JSON")
    ap.add_argument("--placebo", type=int, default=0,
                    help="安慰剂核：置换查询框高后重算 S5 的次数（0=关）")
    ap.add_argument("--decomp", default="", help="把核拆解结果写到此 JSON")
    args = ap.parse_args()

    segs = args.segs.split()
    E64, keys = load_emb(args.emb_dir, segs)
    E = l2n(E64)
    eidx = {k: i for i, k in enumerate(keys)}
    ikey = {i: k for k, i in eidx.items()}
    gt = json.load(open(args.gt, encoding="utf-8"))
    fvec, usable = load_face_proto(args.face, args.proto)
    start, shape = load_det_meta(args.det, segs, set(eidx))

    rows = []
    for k in keys:
        if k not in gt:
            continue
        rows.append(dict(k=k, i=eidx[k], pid=gt[k], cam=cam_of(k),
                         tr=("_train_" in k), t=start.get(k, 0),
                         face=(k in fvec), use=(k in usable and k in fvec)))
    n_face = sum(r["face"] for r in rows)
    n_use = sum(r["use"] for r in rows)
    print("评估轨 %d ｜ 有任意脸 %d（%.1f%%）｜ **可用脸 %d（%.1f%%）** ｜ 机位 %d"
          % (len(rows), n_face, 100 * n_face / len(rows), n_use,
             100 * n_use / max(1, len(rows)), len({r["cam"] for r in rows})))

    # 历史库（上午，按机位）
    lib = collections.defaultdict(list)
    for r in rows:
        if r["tr"]:
            lib[(r["cam"], r["pid"])].append(r)
    libv = {}
    for key, rs in lib.items():
        v = E[[x["i"] for x in rs]].mean(0)
        libv[key] = v / max(np.linalg.norm(v), 1e-9)

    gal = build_gallery(rows, fvec, usable, eidx)
    cand_ids = collections.defaultdict(set)
    for (c, p) in libv:
        cand_ids[c].add(p)
    print("历史库 %d 个「机位×身份」｜ **人脸库覆盖 %d 个（%.1f%%）**"
          % (len(libv), len(gal), 100 * len(gal) / max(1, len(libv))))
    for c in sorted(cand_ids):
        n_c = len(cand_ids[c])
        n_g = sum(1 for (cc, pp) in gal if cc == c)
        print("  %-4s 候选身份 %2d ｜ 有上午人脸原型 %2d" % (c, n_c, n_g))

    queries = sorted([r for r in rows if not r["tr"]], key=lambda r: (r["cam"], r["t"], r["k"]))
    print("查询轨（下午）%d" % len(queries))

    # ---- 识别：查询脸 → 同机位原型 ----
    acc_all = collections.defaultdict(list)
    for r in rows:
        if r["use"]:
            acc_all[(r["cam"], r["pid"])].append((r["k"], fvec[r["k"]]))

    def gallery_loo(q):
        """留一原型：该身份同机位其余可用脸均值（除本条轨自己）。"""
        out = {}
        for (c, p), items in acc_all.items():
            if c != q["cam"]:
                continue
            vs = [v for k, v in items if k != q["k"]]
            if not vs:
                continue
            v = np.mean(np.stack(vs), 0)
            out[p] = v / max(np.linalg.norm(v), 1e-9)
        return out

    def recognize(q, tau_face, margin):
        if not q["use"]:
            return None, -1.0, -1.0
        f = fvec[q["k"]]
        g = gallery_loo(q) if args.gallery_mode == "loo" else \
            {p: v for (c, p), v in gal.items() if c == q["cam"]}
        sims = [(float(f @ v), p) for p, v in g.items()]
        if not sims:
            return None, -1.0, -1.0
        sims.sort(reverse=True)
        best = sims[0]
        second = sims[1][0] if len(sims) > 1 else -1.0
        if best[0] < tau_face or (best[0] - second) < margin:
            return None, best[0], best[0] - second
        return best[1], best[0], best[0] - second

    def has_gallery(q):
        if args.gallery_mode == "loo":
            return q["use"] and any(k != q["k"] for k, _ in acc_all.get((q["cam"], q["pid"]), []))
        return (q["cam"], q["pid"]) in gal and q["use"]

    # 识别质量（用 GT 评，机制本身不看 GT）
    print("\n===== 人脸识别质量（查询=下午可用脸轨，库=上午同机位原型）=====")
    print("  %-8s %-6s %-8s %-8s %-10s %-10s %s"
          % ("τ_face", "有库", "认出", "认对", "精度", "召回(对全部)", "漏认(有库未认出)"))
    face_eval = {}
    for tf in args.tau_face:
        n_gal = n_out = n_ok = 0
        for q in queries:
            if not q["use"]:
                continue
            if not has_gallery(q):
                continue
            n_gal += 1
            p, s, m = recognize(q, tf, args.face_margin)
            if p is None:
                continue
            n_out += 1
            n_ok += (p == q["pid"])
        face_eval[tf] = (n_gal, n_out, n_ok)
        print("  %-8.2f %-6d %-8d %-8d %-10s %-10s %d"
              % (tf, n_gal, n_out, n_ok,
                 "%.1f%%" % (100.0 * n_ok / n_out) if n_out else "-",
                 "%.1f%%" % (100.0 * n_ok / n_gal) if n_gal else "-",
                 n_gal - n_out))

    # ---- 池 + 主循环 ----
    def run(tau_face):
        pools = {k: collections.defaultdict(list) for k in
                 ("gt", "gt_usable", "gt_gallery", "real", "real_oracle_cov")}
        rec = []
        stat = collections.Counter()
        for q in queries:
            cands = sorted(cand_ids[q["cam"]])
            if not cands:
                continue
            libsc = {p: float(E[q["i"]] @ libv[(q["cam"], p)]) for p in cands}
            # S5 形态加权
            hq = q_h.get(q["k"], -1.0)
            s5 = {}
            for p in cands:
                rs = lib[(q["cam"], p)]
                hs = np.array([q_h.get(x["k"], -1.0) for x in rs])
                ws = np.where((hs > 0) & (hq > 0), np.exp(-np.abs(np.log(np.maximum(hs, 1e-6) / hq)) / args.sigma), 1.0)
                ws = np.maximum(ws, 1e-6)
                v = (E[[x["i"] for x in rs]] * ws[:, None]).sum(0) / ws.sum()
                s5[p] = float(E[q["i"]] @ (v / max(np.linalg.norm(v), 1e-9)))
            poolsc = {}
            for name, pool in pools.items():
                poolsc[name] = {p: topk_mean(E, q["i"], pool[(q["cam"], p)], args.pool_k)[0]
                                for p in cands}
            pid, fs, fm = recognize(q, tau_face, args.face_margin)
            sc = {"s0": libsc, "s5": s5}
            for name in poolsc:
                sc["s0+" + name] = {p: max(libsc[p], poolsc[name][p]) for p in cands}
                sc["s5+" + name] = {p: max(s5[p], poolsc[name][p]) for p in cands}
            rec.append(dict(q=q, sc=sc, fid=pid, face_s=fs, face_m=fm))
            # 因果写入
            def push(pool, ent):
                if ent is None:
                    return
                lst = pool[(q["cam"], ent)]
                lst.append(q["i"])
                if len(lst) > args.pool_max:
                    del lst[0]
            if q["face"] and q["pid"] in cands:
                push(pools["gt"], q["pid"]); stat["w_gt"] += 1
            if q["use"] and q["pid"] in cands:
                push(pools["gt_usable"], q["pid"]); stat["w_gt_usable"] += 1
                if has_gallery(q):
                    # 只保留「该身份本来有人脸原型、识别器有能力叫出名字」的那些
                    push(pools["gt_gallery"], q["pid"]); stat["w_gt_gallery"] += 1
            if pid is not None:
                push(pools["real"], pid)
                stat["w_real"] += 1
                stat["w_real_ok" if pid == q["pid"] else "w_real_bad"] += 1
                if q["pid"] in cands:
                    push(pools["real_oracle_cov"], q["pid"]); stat["w_oracle"] += 1
        return rec, stat

    q_h = {k: shape[k][0] for k in shape}
    tau_face_main = args.tau_face[len(args.tau_face) // 2]
    rec, stat = run(tau_face_main)

    def recall(sub, key, tau):
        n = 0
        for r in sub:
            v = r["sc"][key]
            if not v:
                continue
            best = max(v, key=v.get)
            if v[best] >= tau and best == r["q"]["pid"]:
                n += 1
        return n

    ev = [r for r in rec if r["q"]["pid"] in cand_ids[r["q"]["cam"]]]
    names = ["gt", "gt_usable", "gt_gallery", "real", "real_oracle_cov"]
    print("\n===== 主表（recall，分母 %d = 本人上午在该机位有库）=====" % len(ev))
    print("  gt=原口径(有任意脸+真值身份)  gt_usable=可用脸+真值身份")
    print("  gt_gallery=可用脸+该身份有人脸原型+真值身份  real=可用脸+真实识别")
    print("  %-5s %-8s %-8s" % ("τ", "S0 库均值", "S5 形态加权")
          + "".join("%-22s" % ("S0+" + n) for n in names)
          + "".join("%-22s" % ("S5+" + n) for n in names))
    table = {}
    for tau in args.tau:
        cells = []
        row = {}
        for base in ("s0", "s5"):
            k0 = recall(ev, base, tau)
            row[base] = (k0, len(ev))
            cells.append("%5.1f%%(%3d)" % (100.0 * k0 / len(ev), k0))
        for base in ("s0", "s5"):
            for n in names:
                k = recall(ev, base + "+" + n, tau)
                row[base + "+" + n] = (k, len(ev))
                cells.append("%5.1f%%(%3d)" % (100.0 * k / len(ev), k))
        print("  %-5.2f " % tau + " ".join("%-17s" % c for c in cells))
        table[tau] = row
    print()
    print("  写入统计：%s" % dict(stat))
    print("  τ_face=%.2f margin=%.2f" % (tau_face_main, args.face_margin))

    # 按机位拆（主阈值）
    mt = args.main_tau
    print("\n  ---- 按机位（τ=%.2f）----" % mt)
    print("  %-4s %4s %-9s %-9s %-9s %-9s" %
          ("机位", "n", "S0", "S5", "S0+real", "S5+real"))
    for c in sorted({r["q"]["cam"] for r in ev}):
        sub = [r for r in ev if r["q"]["cam"] == c]
        vals = [100.0 * recall(sub, k, mt) / max(1, len(sub)) for k in
                ("s0", "s5", "s0+real", "s5+real")]
        print("  %-4s %4d %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%%"
              % (c, len(sub), vals[0], vals[1], vals[2], vals[3]))

    # ---- EXP-003a：核拆解（分层 / 安慰剂 / 同机位归一化）----
    if args.decomp:
        def s5_scores(q, hmap):
            hq = hmap.get(q["k"], -1.0)
            out = {}
            for p in sorted(cand_ids[q["cam"]]):
                rs = lib.get((q["cam"], p), [])
                if not rs:
                    continue
                hs = np.array([hmap.get(x["k"], -1.0) for x in rs])
                ws = np.where((hs > 0) & (hq > 0),
                              np.exp(-np.abs(np.log(np.maximum(hs, 1e-6) / hq)) / args.sigma),
                              1.0)
                ws = np.maximum(ws, 1e-6)
                v = (E[[x["i"] for x in rs]] * ws[:, None]).sum(0) / ws.sum()
                out[p] = float(E[q["i"]] @ (v / max(np.linalg.norm(v), 1e-9)))
            return out

        def hit(scmap, q):
            if not scmap:
                return 0
            b = max(scmap, key=scmap.get)
            return int(scmap[b] >= args.main_tau and b == q["pid"])

        result = dict(n=len(ev), tau=args.main_tau, sigma=args.sigma)

        # 1) 分层：|log(h_lib_median / h_query)|
        strata = [("<0.15", 0.0, 0.15), ("0.15-0.4", 0.15, 0.4),
                  ("0.4-0.8", 0.4, 0.8), (">=0.8", 0.8, 99.0)]
        buckets = {lab: dict(n=0, s0=0, s5=0) for lab, _, _ in strata}
        skipped = 0
        for r in ev:
            q = r["q"]
            hq = q_h.get(q["k"], -1.0)
            hs = [q_h.get(x["k"], -1.0) for x in lib.get((q["cam"], q["pid"]), [])]
            hs = [h for h in hs if h > 0]
            if hq <= 0 or not hs:
                skipped += 1
                continue
            d = abs(float(np.log(np.median(hs) / hq)))
            for lab, lo, hi in strata:
                if lo <= d < hi:
                    buckets[lab]["n"] += 1
                    buckets[lab]["s0"] += hit(r["sc"]["s0"], q)
                    buckets[lab]["s5"] += hit(r["sc"]["s5"], q)
                    break
        print("\n===== 分层：|log(h_lib_med / h_q)|（跳过无框高 %d）=====" % skipped)
        for lab, _, _ in strata:
            b = buckets[lab]
            print("  %-9s n=%3d  S0=%3d  S5=%3d  delta=%+d"
                  % (lab, b["n"], b["s0"], b["s5"], b["s5"] - b["s0"]))
        result["strata"] = {lab: buckets[lab] for lab, _, _ in strata}
        result["strata_skipped"] = skipped

        # 2) 安慰剂：置换查询框高（保持每个身份库内 h 不变）
        n_s0 = sum(hit(r["sc"]["s0"], r["q"]) for r in ev)
        n_s5 = sum(hit(r["sc"]["s5"], r["q"]) for r in ev)
        result["baseline"] = dict(s0=n_s0, s5=n_s5)
        placebo = []
        if args.placebo > 0:
            rng = np.random.default_rng(20261124)
            keys = [r["q"]["k"] for r in ev]
            vals = np.array([q_h.get(k, -1.0) for k in keys])
            for _ in range(args.placebo):
                perm = rng.permutation(len(keys))
                hmap = dict(q_h)
                for i, j in enumerate(perm):
                    hmap[keys[i]] = float(vals[j])
                placebo.append(sum(hit(s5_scores(r["q"], hmap), r["q"]) for r in ev))
            placebo = np.array(placebo, dtype=float)
            print("\n===== 安慰剂核：置换查询框高 %d 次 =====" % args.placebo)
            print("  均值 %.1f  中位 %.1f  P5 %.1f  P95 %.1f  最小 %d 最大 %d"
                  % (placebo.mean(), np.median(placebo), np.percentile(placebo, 5),
                     np.percentile(placebo, 95), placebo.min(), placebo.max()))
            print("  对照：S0=%d（无核），真核=%d" % (n_s0, n_s5))
            result["placebo"] = dict(reps=args.placebo, mean=float(placebo.mean()),
                                     median=float(np.median(placebo)),
                                     p05=float(np.percentile(placebo, 5)),
                                     p95=float(np.percentile(placebo, 95)),
                                     min=int(placebo.min()), max=int(placebo.max()),
                                     frac_ge_true=float((placebo >= n_s5).mean()),
                                     frac_gt_s0=float((placebo > n_s0).mean()))

        # 3) 同机位归一化框高
        med_cam = {}
        for cam in sorted({r["cam"] for r in rows}):
            hs = [q_h.get(r["k"], -1.0) for r in rows if r["cam"] == cam and r["tr"]]
            hs = [h for h in hs if h > 0]
            if hs:
                med_cam[cam] = float(np.median(hs))
        norm_h = {}
        for k, v in q_h.items():
            c = cam_of(k)
            norm_h[k] = (v / med_cam[c]) if (v > 0 and c in med_cam) else v
        n_norm = sum(hit(s5_scores(r["q"], norm_h), r["q"]) for r in ev)
        print("\n===== 同机位归一化框高后 =====")
        print("  S0=%d  S5_norm=%d  (n=%d, 机位中位框高 %s)"
              % (n_s0, n_norm, len(ev),
                 {c: round(v, 1) for c, v in sorted(med_cam.items())}))
        result["normalized"] = dict(s0=n_s0, s5=n_norm,
                                    cam_median={c: round(v, 2) for c, v in med_cam.items()})

        with open(args.decomp, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        print("  decomp -> %s" % args.decomp)

    # ---- 派生结果导出（不影响默认输出）----
    if args.dump_queries:
        import csv as _csv
        scen = ["s0", "s5", "s0+real", "s5+real", "s0+gt", "s5+gt"]
        head = (["query_key", "camera", "gt_identity", "gt_has_gallery",
                 "query_height_px", "gallery_tracks",
                 "face_similarity", "face_margin", "face_pick", "face_correct"]
                + ["%s_%s" % (s, x) for s in scen for x in ("top1", "score", "hit")])
        with open(args.dump_queries, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(head)
            for r in ev:
                q = r["q"]
                gl = len(lib.get((q["cam"], q["pid"]), []))
                line = [q["k"], q["cam"], q["pid"], int(gl > 0),
                        "%.1f" % q_h.get(q["k"], -1.0), gl,
                        "%.4f" % r["face_s"], "%.4f" % r["face_m"],
                        r["fid"] or "",
                        (int(r["fid"] == q["pid"]) if r["fid"] else 0)]
                for s in scen:
                    v = r["sc"][s]
                    if v:
                        b = max(v, key=v.get)
                        line += [b, "%.4f" % v[b],
                                 int(v[b] >= args.main_tau and b == q["pid"])]
                    else:
                        line += ["", "", 0]
                w.writerow(line)
        print("  dumped %d queries -> %s" % (len(ev), args.dump_queries))

    if args.dump_pools:
        i2k = {v: k for k, v in eidx.items()}
        pools = collections.defaultdict(list)
        evicted = 0
        for r in rec:
            if r["fid"] is None:
                continue
            q = r["q"]
            lst = pools[(q["cam"], r["fid"])]
            lst.append(q["i"])
            if len(lst) > args.pool_max:
                lst.pop(0)
                evicted += 1
        payload = dict(pool_k=args.pool_k, pool_max=args.pool_max,
                       tau_face=tau_face_main, face_margin=args.face_margin,
                       n_writes=int(stat["w_real"]),
                       n_writes_correct=int(stat["w_real_ok"]),
                       n_evicted=evicted,
                       entries=[dict(camera=c, identity=p, n_tracks=len(v),
                                     tracks=[i2k[i] for i in v])
                                for (c, p), v in sorted(pools.items())])
        with open(args.dump_pools, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("  dumped %d pool entries -> %s" % (len(pools), args.dump_pools))
    if args.out:
        json.dump(dict(params={k: v for k, v in vars(args).items()},
                       stat=dict(stat),
                       table={str(k): {kk: list(vv) for kk, vv in v.items()}
                              for k, v in table.items()}),
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
