#!/usr/bin/env python
"""S5 / Q4：倒影指派 —— 不用掩膜，把倒影判定写成「真人 ← 倒影」的全局指派问题。

设计（docs/reid-identity-ownership-redesign.md §3.5）：
  倒影 = 另一个观测的镜像副本。对候选倒影 r，必须存在一个**真人**观测 s 使其被"解释"：
      cost(s, r) = 镜像残差(du_med, pred_err, offset_ratio) + 尺度不一致(hratio)
  求全局指派（真人 ← 倒影，一对至多一），**只有被解释的观测才可能是倒影**。

与既有 `track_cleaning.py`（逐轨规则）的三个升级：
  ① **唯一性**：一个真人至多解释一个（时间不重叠的）倒影段；
  ② **切断倒影链**：已被判为倒影的轨迹**不得再充当源**（逐轨规则会出现 194→193、207→193、
     208→193、212→194 这种"倒影解释倒影"的链）；
  ③ **保守兜底**：没有任何源能解释的观测一律判为真人（第一约束 = 不误删真人）。

硬纪律：**不逐帧删框，只打 flag**（写入 GIL obs.flags；判定真相同时写入 annotation 表，视图无关）。

用法：
  python tools/gil_reflection_assign.py                      # 只读：指标 + JSON
  python tools/gil_reflection_assign.py --apply              # 另写 GIL obs.flags + annotation
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOT = os.path.dirname(HERE)
from mirror_kinematics import load_tracks, pair_metrics  # noqa: E402
from track_cleaning import cooccur  # noqa: E402

CAMS = ["305", "316", "318", "320", "323", "343"]
DIMS = {"305": (2560, 1440), "316": (2560, 1440), "318": (1920, 1080),
        "320": (2560, 1440), "323": (1920, 1080), "343": (2560, 1440)}
FPS = 25.0

# 镜像几何规则（沿用 mirror_kinematics 验证过的组合规则，val 上真对 28/29、对照池仅 7/408）
# 尺度门 hr（倒影/源 框高比）= 本次新增。地板倒影是镜像，尺度应≈1；人工真值 28 条实测 hr∈[0.83,1.06]。
#   hr_lo=0.60：保住真倒影 316#81(hr=0.657)；
#   hr_hi=1.10：排除全部已被人工审计判 FP 的条目（318 全部 1.29~2.44、305#12=2.61、323#193=2.08、323#115=2.40）。
# v_min（静止对运动门）：v2 审计建议的"双轨中位速度<0.5px/f 拒收"，但实测会误伤真倒影
#   316#79/81、323#68/89（其源速度 0.08~0.26px/f 与 318 静坐对无法区分）→ 默认禁用(=0.0)，
#   改用 hr_hi 达到同样效果且零误伤（见报告 A/B）。
RULE = dict(du_max=0.06, perr_max=0.10, off_lo=0.35, off_hi=0.60,
            hr_lo=0.60, hr_hi=1.10, n_min=4, v_min=0.0)


def rule_pass(m):
    if m is None or m.get("pred_err") is None:
        return False
    if m["n_common"] < RULE["n_min"]:
        return False
    return (abs(m["du_med"]) < RULE["du_max"]
            and m["pred_err"] <= RULE["perr_max"]
            and RULE["off_lo"] <= m["offset_ratio"] <= RULE["off_hi"]
            and RULE["hr_lo"] <= m["hratio"] <= RULE["hr_hi"])


def span(tr):
    return tr[0][0] / FPS, tr[-1][0] / FPS


def med_speed_px(tr):
    """轨迹中位速度（px/帧）。v2 人工审计认定的第一类假阳性机制是「静止对」：
    双轨近似静止时 v_R≈2·v_F−v_S 恒等式平凡成立（318 全部 FP、305#12/22）。"""
    if len(tr) < 2:
        return 0.0
    sp = []
    for i in range(1, len(tr)):
        df = tr[i][0] - tr[i - 1][0]
        if df <= 0 or df > 5:
            continue
        b0, b1 = tr[i - 1][1], tr[i][1]
        du = (b1[0] + b1[2]) / 2 - (b0[0] + b0[2]) / 2
        dv = (b1[1] + b1[3]) / 2 - (b0[1] + b0[3]) / 2
        sp.append(float(np.hypot(du, dv)) / df)
    return float(np.median(sp)) if sp else 0.0


def overlap(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0])


def build_edges(tracks, W, H):
    """枚举同机位共现对 → 有向镜像边 (reflection, source, weight, meta)。

    返回 (edges, hr_excluded)：后者是「镜像几何全过、仅尺度门 hr 不过」的候选（存疑，供审计）。
    """
    edges, hr_excl, static_excl = [], [], []
    for (a, b), n in cooccur(tracks, n_min=RULE["n_min"]).items():
        for r, s in ((a, b), (b, a)):
            m = pair_metrics(tracks, W, H, r, s)
            if not m or m.get("pred_err") is None or m["n_common"] < RULE["n_min"]:
                continue
            if not (abs(m["du_med"]) < RULE["du_max"] and m["pred_err"] <= RULE["perr_max"]
                    and RULE["off_lo"] <= m["offset_ratio"] <= RULE["off_hi"]):
                continue
            r_t0, r_t1 = span(tracks[r])
            s_t0, s_t1 = span(tracks[s])
            vs, vr = med_speed_px(tracks[s]), med_speed_px(tracks[r])
            rec = dict(
                r=r, s=s, w=1.0 - m["pred_err"],
                perr=round(m["pred_err"], 4), off=round(m["offset_ratio"], 4),
                du=round(m["du_med"], 4), hr=round(m["hratio"], 3),
                v_s=round(vs, 2), v_r=round(vr, 2),
                n=int(m["n_common"]), diff_corr=(None if m.get("diff_corr_v") is None
                                                 else round(m["diff_corr_v"], 3)),
                r_t0=round(r_t0, 2), r_t1=round(r_t1, 2),
                s_t0=round(s_t0, 2), s_t1=round(s_t1, 2))
            if max(vs, vr) < RULE["v_min"]:      # ★ 静止对：恒等式平凡成立 → 拒收
                static_excl.append(rec)
            elif RULE["hr_lo"] <= m["hratio"] <= RULE["hr_hi"]:
                edges.append(rec)
            else:
                hr_excl.append(rec)
    return edges, hr_excl, static_excl


def assign(edges):
    """全局指派：唯一性 + 时间不重叠 + 切断倒影链。返回 (role, paired, rejected, amb)。

    ★ 两轮，关键在先验「源必须是真人」：
      第 1 轮只用「自身不被任何观测解释」（无作为 r 的边）的节点作源 —— 它们物理上必然是真人；
      第 2 轮放开，但已被判为倒影的节点**永不作源**（切断"倒影解释倒影"的链）。
    这与设计文档 §3.5「真人 ← 倒影 一对至多一」一致，且比"纯按 perr 贪心"更保守
    （纯贪心会让 212→194(perr .021) 压过 194→193(perr .048)，把真正的倒影 194 误判成真人）。
    """
    has_r_edge = {e["r"] for e in edges}
    role, paired, used_s = {}, {}, {}
    rejected, amb = [], collections.defaultdict(list)
    for e in edges:
        amb[e["r"]].append(e)

    def try_pair(e):
        r, s = e["r"], e["s"]
        if paired.get(r) is not None:
            return "r_assigned"
        if role.get(r) == "person":
            return "r_is_person"
        if role.get(s) == "reflection":          # ★ 切断倒影链
            return "s_is_reflection"
        iv = (e["r_t0"], e["r_t1"])
        if any(overlap(iv, x) for x in used_s.get(s, [])):
            return "s_occupied_interval"
        paired[r] = s
        role[r] = "reflection"
        role.setdefault(s, "person")
        used_s.setdefault(s, []).append(iv)
        return None

    clean = sorted([e for e in edges if e["s"] not in has_r_edge], key=lambda x: -x["w"])
    rest = sorted([e for e in edges if e["s"] in has_r_edge], key=lambda x: -x["w"])
    for e in clean + rest:
        why = try_pair(e)
        if why:
            rejected.append((e, why))
    # 歧义清单：同一 r 有多个候选源（≥2），列出并标注最终归宿
    ambiguity = []
    for r, es in amb.items():
        if len(es) < 2:
            continue
        es = sorted(es, key=lambda x: x["perr"])
        ambiguity.append(dict(r=r, final=("reflection<-" + str(paired[r])) if r in paired
                              else ("person" if role.get(r) == "person" else "unset"),
                              candidates=[dict(source=x["s"], perr=x["perr"], off=x["off"],
                                               hr=x["hr"], n=x["n"], dc=x["diff_corr"],
                                               t_r=[x["r_t0"], x["r_t1"]],
                                               t_s=[x["s_t0"], x["s_t1"]])
                                          for x in es]))
    return role, paired, rejected, ambiguity


def eval_truth(by_cam, truth):
    """与用户真值对账。返回 per-cam 与汇总指标。"""
    out = {}
    tot = dict(tp=0, fp_labeled_person=0, fp_unlabeled=0, fn=0, n_truth=0)
    for cam, rows in by_cam.items():
        pred = {x["tid"] for x in rows}
        t_refl = truth["reflection"].get(cam, set())
        t_src = truth["source"].get(cam, set())
        tp = pred & t_refl
        fp_src = pred & t_src                      # ★ 判为倒影但用户标为真人 = 误删真人
        fp_un = pred - t_refl - t_src              # 未标注（用户没标，需人工复核）
        fn = t_refl - pred
        out[cam] = dict(n_pred=len(pred), n_truth=len(t_refl),
                        tp=sorted(tp), fp_against_source=sorted(fp_src),
                        fp_unlabeled=sorted(fp_un), fn=sorted(fn),
                        precision_recall_denominator_note="仅在已标注集合上可算")
        tot["tp"] += len(tp)
        tot["fp_labeled_person"] += len(fp_src)
        tot["fp_unlabeled"] += len(fp_un)
        tot["fn"] += len(fn)
        tot["n_truth"] += len(t_refl)
    # 保守可算口径：把「已标 source 的命中」算硬 FP；未标注的不计入 P 分母
    labeled_pred = tot["tp"] + tot["fp_labeled_person"]
    p_lab = tot["tp"] / labeled_pred if labeled_pred else None
    r_all = tot["tp"] / tot["n_truth"] if tot["n_truth"] else None
    out["_summary"] = dict(
        tp=tot["tp"], fp_against_source=tot["fp_labeled_person"],
        fp_unlabeled=tot["fp_unlabeled"], fn=tot["fn"], n_truth=tot["n_truth"],
        precision_on_labeled=p_lab, recall_on_truth=r_all,
        hard_metric="误判为倒影的已标注真人（必须≈0）= %d" % tot["fp_labeled_person"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default=",".join(CAMS))
    ap.add_argument("--raw-dir", default="output_crosscam")
    ap.add_argument("--truth", default="webapp/backend/data/labels.json")
    ap.add_argument("--status-json", default="webapp/backend/data/cams.json")
    ap.add_argument("--out", default="output_crosscam/analysis/gil/reflection_assign_v1.json")
    ap.add_argument("--store", default="output_crosscam/analysis/gil/store_v1")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    cams = [c.strip() for c in a.cams.split(",") if c.strip()]
    tracks = {}
    for c in cams:
        p = os.path.join(ROOT, a.raw_dir, f"{c}_detections.jsonl")
        tracks[c] = load_tracks(p)
        print(f"[load] {c}: {len(tracks[c])} tracks ({p})")

    # 用户真值
    truth = {"reflection": {}, "source": {}}
    tp_path = os.path.join(ROOT, a.truth)
    if os.path.exists(tp_path):
        md = json.load(open(tp_path, encoding="utf-8")).get("mirror_decisions", {})
        for k, v in md.items():
            cam, tid = k.split(":")
            if v in ("reflection", "source"):
                truth[v].setdefault(cam, set()).add(int(tid))
    # 掩膜状态（track_status）
    status = {}
    sp = os.path.join(ROOT, a.status_json)
    if os.path.exists(sp):
        status = json.load(open(sp, encoding="utf-8")).get("track_status", {})

    by_cam, diag, amb_all, hr_excl_all, static_excl_all = {}, {}, {}, {}, {}
    for c in cams:
        W, H = DIMS[c]
        edges, hr_excl, static_excl = build_edges(tracks[c], W, H)
        hr_excl_all[c] = hr_excl
        static_excl_all[c] = static_excl
        role, paired, rejected, ambiguity = assign(edges)
        refl = sorted(paired)
        by_cam[c] = [dict(tid=r, source=paired[r],
                          **{k: e[k] for k in ("perr", "off", "du", "hr", "n",
                                               "diff_corr", "r_t0", "r_t1",
                                               "s_t0", "s_t1")})
                     for r in refl
                     for e in [next(x for x in edges if x["r"] == r and x["s"] == paired[r])]]
        amb_all[c] = ambiguity
        n_all = sum(1 for t, l in tracks[c].items() if len(l) >= RULE["n_min"])
        cut = sum(1 for e, why in rejected if why == "s_is_reflection")
        diag[c] = dict(n_tracks_ge_nmin=n_all, n_edges=len(edges), n_reflection=len(refl),
                       n_person=n_all - len(refl),
                       n_cut_chain=cut,
                       n_rejected_assigned=sum(1 for _, w in rejected if w == "r_assigned"),
                       n_rejected_s_occupied=sum(1 for _, w in rejected if w == "s_occupied_interval"),
                       n_ambiguous=len(ambiguity), n_hr_excluded=len(hr_excl),
                       n_static_excluded=len(static_excl))
        print(f"[assign] {c}: 轨(≥{RULE['n_min']}) {n_all}  镜像边 {len(edges)}  "
              f"→ 判倒影 {len(refl)} / 真人 {n_all-len(refl)}  "
              f"（切链 {cut}，歧义 {len(ambiguity)}，尺度门排除 {len(hr_excl)}，"
              f"静止对排除 {len(static_excl)}）")

    # ---- 真值对账 ----
    ev = eval_truth(by_cam, truth)
    print("\n=== 与用户真值对账（仅 316/323 有标注）===")
    for c in cams:
        if c not in truth["reflection"] and c not in truth["source"]:
            continue
        e = ev[c]
        print(f"  {c}: 判倒影 {e['n_pred']}  真值 {e['n_truth']}  TP {len(e['tp'])}  "
              f"★误判已标真人 {len(e['fp_against_source'])}  未标注额外 {len(e['fp_unlabeled'])}  FN {len(e['fn'])}")
        if e["fp_against_source"]:
            print(f"      ★★ 误判为倒影的真标注真人: {[f'{c}#{t}' for t in e['fp_against_source']]}")
        if e["fn"]:
            print(f"      漏判真值倒影: {[f'{c}#{t}' for t in e['fn']]}")
        if e["fp_unlabeled"]:
            print(f"      未标注额外命中: {[f'{c}#{t}' for t in e['fp_unlabeled']]}")
    print(f"  汇总: TP {ev['_summary']['tp']} / 真值 {ev['_summary']['n_truth']}，"
          f"★误判已标真人 {ev['_summary']['fp_against_source']}，"
          f"未标注额外 {ev['_summary']['fp_unlabeled']}，漏判 {ev['_summary']['fn']}")

    # ---- 掩膜交叉审计 ----
    mask_cross = {}
    print("\n=== 掩膜(track_status=masked) 交叉审计 ===")
    for c in cams:
        st = status.get(c, {})
        if not st:
            continue
        alg = {x["tid"] for x in by_cam[c]}
        masked = {int(t) for t, v in st.items() if v == "masked"}
        keep = {int(t) for t, v in st.items() if v == "keep"}
        refl_t = {int(t) for t, v in st.items() if v == "reflection"}
        mask_cross[c] = dict(
            mask_total=len(masked),
            mask_agree_reflection=len(masked & alg),          # 掩膜∩几何 一致
            mask_only=len(masked - alg - refl_t),             # 掩膜单独判（几何不认）
            mask_only_in_keepzone=len(masked & keep),         # 掩膜剔了但 status=keep?? (信息性)
            geom_only=len(alg - masked - refl_t),             # 几何判但掩膜没拦
            keep_flagged_reflection=len(alg & keep),          # ★ 几何判倒影但用户标 keep
        )
        print(f"  {c}: 掩膜 {len(masked)}  与几何一致 {len(masked & alg)}  "
              f"掩膜独有(几何不认) {len(masked - alg - refl_t)}  "
              f"几何独有(掩膜漏网) {len(alg - masked - refl_t)}  "
              f"★几何判倒影但标 keep {len(alg & keep)}")

    # ---- 歧义清单（同一 r 有 ≥2 候选源）----
    print("\n=== 歧义清单（同一观测有多个候选源，需人工终审）===")
    n_amb = 0
    for c in cams:
        for x in amb_all[c]:
            n_amb += 1
            cands = ", ".join(f"源{s}(perr={p['perr']:.3f})" for s, p in
                              [(y["source"], y) for y in x["candidates"]])
            print(f"  {c}#{x['r']} → 归宿 {x['final']}；候选 {cands}")
    if not n_amb:
        print("  （无）")

    # ---- 尺度门排除清单（镜像几何全过、只有框高比不符）----
    print("\n=== 尺度门 hr 排除（镜像几何全过，仅倒影/源 框高比越界 → 判非倒影）===")
    n_ex = 0
    for c in cams:
        for x in hr_excl_all[c]:
            n_ex += 1
            print(f"  {c}#{x['r']} → 源{x['s']}  hr={x['hr']:.2f} perr={x['perr']:.4f} "
                  f"off={x['off']:.3f} n={x['n']} 速度 {x['v_s']}/{x['v_r']} px/f")
    if not n_ex:
        print("  （无）")

    # ---- 静止对排除清单（v2 审计认定的第一类假阳性机制）----
    print("\n=== 静止对排除（双轨中位速度 <%.1f px/f，恒等式平凡成立）===" % RULE["v_min"])
    n_st = 0
    for c in cams:
        for x in static_excl_all[c]:
            n_st += 1
            print(f"  {c}#{x['r']} → 源{x['s']}  速度 {x['v_s']}/{x['v_r']} px/f  "
                  f"perr={x['perr']:.4f} hr={x['hr']:.2f} n={x['n']}")
    if not n_st:
        print("  （无）")

    # ---- 与 v2 全量人工审计对账（金标准：track_cleaning v1 清单 − restored_fp）----
    gold = {}
    gold_src = {"track_cleaning": "output_crosscam/analysis/track_cleaning/track_cleaning.json",
                "audit": "output_crosscam/analysis/reflection_gate/audit_full_2026-10-13.json",
                "manifest": "output_crosscam/cleanset_v2/clean_manifest_v2.json"}
    tc_p = os.path.join(ROOT, gold_src["track_cleaning"])
    man_p = os.path.join(ROOT, gold_src["manifest"])
    if os.path.exists(tc_p) and os.path.exists(man_p):
        tc = json.load(open(tc_p, encoding="utf-8"))["summary"]
        man = json.load(open(man_p, encoding="utf-8"))["cameras"]
        for c in cams:
            v1 = {int(t) for t in tc.get(c, {}).get("flags", {})}
            rest = set(int(x) for x in man.get(c, {}).get("restored_fp", []))
            gold[c] = v1 - rest
        ge = {}
        gt = dict(tp=0, fp=0, fn=0, n_gold=0)
        for c in cams:
            pred = {x["tid"] for x in by_cam[c]}
            g = gold[c]
            ge[c] = dict(gold=len(g), pred=len(pred),
                         tp=sorted(pred & g), fp=sorted(pred - g), fn=sorted(g - pred))
            gt["tp"] += len(pred & g); gt["fp"] += len(pred - g)
            gt["fn"] += len(g - pred); gt["n_gold"] += len(g)
        p = gt["tp"] / (gt["tp"] + gt["fp"]) if gt["tp"] + gt["fp"] else None
        r = gt["tp"] / gt["n_gold"] if gt["n_gold"] else None
        ge["_summary"] = dict(**gt, precision=(round(p, 4) if p else None),
                              recall=(round(r, 4) if r else None),
                              f1=(round(2 * p * r / (p + r), 4) if p and r else None))
        print("\n=== 与 v2 全量人工审计对账（金标准）===")
        for c in cams:
            e = ge[c]
            if not e["gold"] and not e["pred"]:
                continue
            print(f"  {c}: 金标准 {e['gold']}  预测 {e['pred']}  TP {len(e['tp'])}  "
                  f"FP {len(e['fp'])}{e['fp'] if e['fp'] else ''}  FN {len(e['fn'])}{e['fn'] if e['fn'] else ''}")
        s = ge["_summary"]
        print(f"  汇总: P {s['precision']} R {s['recall']} F1 {s['f1']}  "
              f"(TP {s['tp']} FP {s['fp']} FN {s['fn']} / 金标准 {s['n_gold']})")
    else:
        ge = None

    out = dict(version="reflection_assign_v1", rule=RULE, fps=FPS,
               cams=cams, diag=diag, assignments=by_cam, eval=ev,
               eval_v2_audit=ge,
               mask_cross=mask_cross, ambiguity=amb_all, hr_excluded=hr_excl_all,
               static_excluded=static_excl_all,
               design="docs/reid-identity-ownership-redesign.md §3.5；真人←倒影 一对至多一 + 切链 + 保守兜底")
    outp = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {a.out}")

    # ---- 落库 GIL ----
    if a.apply:
        from gil import GILStore
        st = GILStore(os.path.join(ROOT, a.store), create=False)
        n_flag, n_ann, n_missing = 0, 0, 0
        for c in cams:
            rows = {int(r["tid"]): int(r["id"]) for r in
                    st.con.execute("SELECT id, tid FROM obs WHERE cam=? AND tombstone=0", (c,))}
            pred = {x["tid"]: x for x in by_cam[c]}
            for tid, oid in rows.items():
                x = pred.get(tid)
                if x is not None:
                    flags = dict(reflection=round(1 - x["perr"], 4),
                                 explained_by=int(x["source"]), src="reflection_assign_v1")
                else:
                    flags = dict(reflection=0.0, explained_by=None, src="reflection_assign_v1")
                st.con.execute("UPDATE obs SET flags=? WHERE id=?",
                               (json.dumps(flags, ensure_ascii=False), oid))
                n_flag += 1
            # annotation（视图无关：记录 raw 全量判定）
            for tid in sorted(set(pred) | set(rows)):
                val = "reflection" if tid in pred else "person"
                st.con.execute(
                    "INSERT OR REPLACE INTO annotation(target_kind,target_id,value,confidence,author,ts) "
                    "VALUES('obs',?,?,?,'reflection_assign_v1',datetime('now'))",
                    (f"{c}|{tid}", val,
                     ("geom_perr=%.4f_src=%d" % (pred[tid]["perr"], pred[tid]["source"]))
                     if tid in pred else "explained_by_none(conservative)"))
                n_ann += 1
            n_missing += len(set(pred) - set(rows))
        st.con.commit()
        print(f"[gil] obs.flags 写入 {n_flag}；annotation 写入 {n_ann}"
              f"（其中 {n_missing} 条判定在 clean2 视图中不存在——已被掩膜剔除）")
        print("[gil] stats:", json.dumps(st.stats(), ensure_ascii=False))
        st.close()


if __name__ == "__main__":
    main()
