#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建选择器 v2：multi8 多原型 + per-camera 阈值（两档）。

依 2026-10-14 升级实验（tools/selector_upgrade_study.py）结论：
  - multi4/8/16 全面优于单原型（曲线/跨段/公平子集），multi8 综合最优 → 主方案
  - thr 两档：selfcal 中点（保守档，默认）/ legacy 中点（宽松覆盖档）

产物：
  selector_v2.npz        主方案（multi8 子原型 412 个）
  selector_v2_single.npz 对照/回退（125 单原型，校验应与旧 selector.npz 一致）
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from selector import IdentitySelector, _norm  # noqa: E402
from selector_upgrade_study import (build_variants, score_matrix, wp_thr,  # noqa: E402
                                    legacy_thr_vec, MIN_CLUSTER)
from camgroup_study import (load_segs, select_best, cam_of, CAMS,  # noqa: E402
                            TRAIN_SEGS, JDL)
from train_selector_cluster import cooc_pairs, pairwise_f1_cooc  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "selector_1012")
# ★ S7 倒影判别器（收缩 LDA 方向）：把 obs.flags.reflection_score 的口径搬到任意 768d 向量
REFL = os.path.join(ROOT, "output_crosscam", "analysis", "gil", "reflection_cluster_v1.json")


def load_reflection_probe():
    """读倒影判别器（L2 归一化方向 w + score 归一化区间）→ 可直接作用于 768d 向量。"""
    if not os.path.exists(REFL):
        return None
    p = json.load(open(REFL, encoding="utf-8"))["probe"]
    w = np.asarray(p["weights"], np.float32)
    w = w / (np.linalg.norm(w) + 1e-9)
    return dict(w=w, lo=float(p["score_norm"]["lo"]), hi=float(p["score_norm"]["hi"]),
                source=os.path.basename(REFL),
                loo_auc_mean=float((p.get("eval", {}) or {}).get("leave_one_camera_mean", 0.0)))


def reflection_score(V, probe):
    """任意 [N,D] 向量 → 倒影分 [0,1]（越大越像倒影）。"""
    s = (np.asarray(V, np.float32) @ probe["w"] - probe["lo"]) / max(probe["hi"] - probe["lo"], 1e-9)
    return np.clip(s, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reflection-gate", action="store_true",
                    help="把 S7 倒影判别器接入选择器：计算训练向量倒影分并写入 npz meta"
                         "（供运行时 predict(refl=…) 用）；默认关闭，行为与旧版一致")
    ap.add_argument("--reflection-tau", type=float, default=0.5,
                    help="硬门阈值（写进 meta）。建议在 val 段校准，不要用 test")
    ap.add_argument("--reflection-base", type=float, default=0.5,
                    help="软门基线（写进 meta）：低于基线的样本不受罚")
    ap.add_argument("--reflection-alpha", type=float, default=1.0,
                    help="软门强度（写进 meta）：eff = best − α·max(0, refl − base)")
    ap.add_argument("--reflection-drop", action="store_true",
                    help="★ 危险：把倒影分 >= tau 的训练向量从聚类输入中剔除后再建库"
                         "（默认只报数不删；1012 无倒影标签，开启前请先看命中数）")
    args = ap.parse_args()

    TR_keys, TR_Va, TR_Vb, TR_V = load_segs(TRAIN_SEGS)

    # ---- ★ S7 倒影门（默认关闭 → 与旧版完全一致；开启后把判别器配置写进 npz meta）----
    refl_probe = load_reflection_probe() if args.reflection_gate else None
    gate_info = None
    if refl_probe is not None:
        s_tr = reflection_score(TR_V, refl_probe)
        n_flag = int((s_tr >= args.reflection_tau).sum())
        print(f"[refl] probe={refl_probe['source']} loo_auc={refl_probe['loo_auc_mean']:.3f} | "
              f"train 倒影分 mean={s_tr.mean():.3f} p95={np.percentile(s_tr, 95):.3f} "
              f"max={s_tr.max():.3f} | >=tau({args.reflection_tau}) 命中 {n_flag}/{len(s_tr)}")
        if args.reflection_drop and n_flag:
            keep = s_tr < args.reflection_tau
            TR_keys = [k for k, m in zip(TR_keys, keep) if m]
            TR_Va, TR_Vb, TR_V = TR_Va[keep], TR_Vb[keep], TR_V[keep]
            print(f"[refl] 已从聚类输入剔除 {n_flag} 条 → 剩 {len(TR_keys)}")
        gate_info = dict(enabled=True, mode="soft", tau=args.reflection_tau,
                         base=args.reflection_base, alpha=args.reflection_alpha,
                         source=refl_probe["source"],
                         loo_auc_mean=round(refl_probe["loo_auc_mean"], 4),
                         n_train=int(len(s_tr)), n_flagged=n_flag,
                         dropped=(n_flag if args.reflection_drop else 0),
                         calib_note="操作点(τ/base/α)须在 val 段按覆盖损失预算校准（禁 test-oracle）；"
                                    "soft: eff=best−α·max(0,refl−base) 仅改接受判定、不改簇归属；"
                                    "hard: refl≥τ→unknown")

    tr_cam = np.array([cam_of(k[0]) for k in TR_keys])
    keys_set = set(TR_keys)
    cooc_all = []
    for s in TRAIN_SEGS:
        for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), keys_set, s, 5):
            cooc_all.append(((s, u), (s, v)))
    kidx = {k: i for i, k in enumerate(TR_keys)}

    _, results = select_best(TR_V, TR_Va, TR_Vb, TR_keys, cooc_all)
    best = [d for d in results if d["method"] == "mutual-knn"][0]
    lab = best["_lab"]
    print(f"[base] {best['method']} k={best.get('k')} thr={best.get('thr')} "
          f"ncl={best['n_clusters']} F1={best['f1']:.4f}")
    ids = sorted(set(lab))
    sizes = np.array([(lab == c).sum() for c in ids])
    keep_ids = [c for c, s in zip(ids, sizes) if s >= MIN_CLUSTER]
    print(f"[keep] {len(keep_ids)} clusters")

    variants, C = build_variants(TR_V, lab, keep_ids)

    def calc(variant, name):
        v = variant
        sc_a = score_matrix(TR_Va, v, C)
        sc_b = score_matrix(TR_Vb, v, C)
        sa, sb = sc_a.max(1), sc_b.max(1)
        g_self = wp_thr(sa, sb, cooc_all, TR_keys, kidx)
        pc_self = {c: wp_thr(sa, sb, cooc_all, TR_keys, kidx, cam_mask=(tr_cam == c))
                   for c in CAMS}
        g_loose = legacy_thr_vec(TR_Va, TR_Vb, TR_V, TR_keys, cooc_all, kidx)
        pc_loose = {c: legacy_thr_vec(TR_Va, TR_Vb, TR_V, TR_keys, cooc_all, kidx,
                                      cam_mask=(tr_cam == c)) for c in CAMS}
        # train 侧验证（共现负对 F1 + 一致性）
        ia = sc_a.argmax(1)
        ib = sc_b.argmax(1)
        ia_map = {k: int(ia[i]) for i, k in enumerate(TR_keys)}
        ib_map = {k: int(ib[i]) for i, k in enumerate(TR_keys)}
        pf = pairwise_f1_cooc(ia_map, ib_map, cooc_all, TR_keys)
        # margin per-cam 分位校准（ret=0.7：保留同人分片 min-margin 的 70%）
        srt_a = np.sort(sc_a, 1)
        srt_b = np.sort(sc_b, 1)
        pos_m = np.minimum(srt_a[:, -1] - srt_a[:, -2], srt_b[:, -1] - srt_b[:, -2])
        margin_pc = {}
        for c in CAMS:
            m = tr_cam == c
            margin_pc[c] = float(np.percentile(pos_m[m], 30))
        print(f"[{name}] F1={pf['f1']:.4f} P={pf['prec']:.3f} R={pf['rec']:.3f} | "
              f"selfcal thr: g={g_self['thr']:.4f} "
              + " ".join(f"{c}:{pc_self[c]['thr']:.4f}" for c in CAMS))
        print(f"[{name}] loose  thr: g={g_loose['thr']:.4f} "
              + " ".join(f"{c}:{pc_loose[c]['thr']:.4f}" for c in CAMS))
        print(f"[{name}] margin ret0.7: " + " ".join(f"{c}:{margin_pc[c]:.4f}" for c in CAMS))
        info = {"method": best["method"], "params": {"k": best.get("k"),
                                                     "thr": best.get("thr")},
                "n_clusters": C, "n_subprotos": int(v["subprotos"].shape[0]),
                "topk": v["topk"], "f1_cooc_train": pf["f1"],
                "thr_rule": "selfcal: mid-point(min(sa,sb)_p5, cooc_p99); "
                            "loose: mid-point(Va.Vb_p5, cooc_VV_p99)",
                "selfcal": {"global": g_self["thr"], "per_cam":
                            {c: pc_self[c]["thr"] for c in CAMS}},
                "loose": {"global": g_loose["thr"], "per_cam":
                          {c: pc_loose[c]["thr"] for c in CAMS}},
                "margin_prec_per_cam": {c: round(margin_pc[c], 6) for c in CAMS},
                "margin_rule": "per-cam 分位: p30(min(margin_a,margin_b) 同人池) ≈ 保留70%",
                "version": "v2-multi8-percam"}
        return info, v

    info_m, vm = calc(variants["multi8"], "multi8")
    info_s, vs = calc(variants["single"], "single")

    # ★ 倒影门配置写进两库 meta（默认 None → 不落 meta，行为不变）
    if gate_info is not None:
        info_m["reflection_gate"] = gate_info
        info_s["reflection_gate"] = gate_info
        print(f"[refl] 已写入 npz meta: {json.dumps(gate_info, ensure_ascii=False)}")

    # ===== 与旧 selector.npz 对齐（簇 ID / 原型行序兼容） =====
    old_npz = os.path.join(OUT, "selector.npz")
    aligned = False
    if os.path.exists(old_npz):
        old_z = np.load(old_npz, allow_pickle=True)
        old_p = old_z["protos"]                       # [125, D]
        sp = vs["subprotos"]                          # [125, D] 当前 slot 顺序
        S = sp @ old_p.T
        slot = S.argmax(1)                            # 当前 slot k → 旧槽位
        d = float(np.abs(S[np.arange(len(slot)), slot] - 1.0).max())
        assert len(set(slot.tolist())) == len(slot), "对齐失败：非双射"
        # multi8 子原型重编号为旧槽位
        vm["sub2cluster"] = slot[vm["sub2cluster"]].astype(np.int64)
        # single 原型重排为旧槽位顺序（行 i = 旧槽位 i）
        order = np.argsort(slot)
        vs["subprotos"] = sp[order]
        co = float((vs["subprotos"] * old_p).sum(1).min())
        print(f"[align] slot 双射 OK; single 行序对齐后逐行 cos min={co:.6f}")
        info_s["aligned_to_old"] = True
        info_m["aligned_to_old"] = True
        aligned = True
        # 兼容性自检：新 single loose 档（= 旧 legacy 口径）vs 旧行为（手工复现）
        thr_old = float(old_z["thr_unknown"])
        S_old = TR_V @ old_p.T
        out_old = np.where(S_old.max(1) >= thr_old, S_old.argmax(1), -1)
        sel_tmp = IdentitySelector(vs["subprotos"], thr_unknown=info_s["selfcal"]["global"],
                                   thr_loose=info_s["loose"]["global"])
        out_new, _ = sel_tmp.predict(TR_V, cams=None, loose=True)
        n_diff = int((out_old != out_new).sum())
        thr_diff = abs(info_s["loose"]["global"] - thr_old)
        print(f"[regress] new-single(loose) vs old behavior: thr_diff={thr_diff:.2e} "
              f"pred_diff={n_diff}/1395 ({'OK' if n_diff == 0 else 'CHECK'})")
        # multi8 冒烟：对同样向量分配，报告簇级一致性（与 single 分配做比较统计）
        sel_m8 = IdentitySelector(vm["subprotos"], thr_unknown=info_m["selfcal"]["global"],
                                  proto_cluster=vm["sub2cluster"])
        out_m8, _ = sel_m8.predict(TR_V, cams=None)
        agree = float((out_m8 == out_new).mean())
        print(f"[smoke] multi8 vs single(loose) train agree={agree:.3f} "
              f"(multi8 known={float((out_m8 >= 0).mean()):.3f}, "
              f"single known={float((out_new >= 0).mean()):.3f})")

    for name, info, v in (("selector_v2", info_m, vm), ("selector_v2_single", info_s, vs)):
        mg = None
        mpm = None
        mret = None
        if name == "selector_v2":
            # 精度档 margin：全局参考值 0.005（兼容）+ per-cam 分位校准（ret=0.7 主用）
            mg = 0.005
            mpm = info["margin_prec_per_cam"]
            mret = 0.7
            info["margin_note"] = ("top1-top2 簇分差下限；'prec' 模式用 per-cam 分位值"
                                   "（train p30 ≈ 保留 70%）；全局参考值 0.005 为兼容保留")
        sel = IdentitySelector(
            v["subprotos"], thr_unknown=info["selfcal"]["global"],
            proto_cluster=v["sub2cluster"],
            thr_unknown_per_cam=info["selfcal"]["per_cam"],
            thr_loose=info["loose"]["global"],
            thr_loose_per_cam=info["loose"]["per_cam"],
            margin_prec=mg, margin_prec_per_cam=mpm, margin_ret=mret, meta=info)
        path = os.path.join(OUT, f"{name}.npz")
        sel.save(path)
        print(f"[save] {path}")

    # 快速健全性: 新 npz 加载 + 单向量 predict
    sel = IdentitySelector.load(os.path.join(OUT, "selector_v2.npz"))
    ids_, sims_ = sel.predict(TR_V[:5], cams=[cam_of(k[0]) for k in TR_keys[:5]])
    print(f"[smoke] multi8 predict 5 vecs: ids={ids_.tolist()} sims={np.round(sims_,3).tolist()}")
    del ids_, sims_
    print("[done]")


if __name__ == "__main__":
    main()
