#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D7 统一扫描引擎（清单第二批）。

覆盖清单的 A / C / E / D / G 五组（B 组已证与 C 等价，跳过）：

  A  查询拒识       机位感知图库、无否决、只调 θ（1.40–4.50 步长 0.05）
  C  簇否决·删簇重搜 d_max 0.22–0.40 步长 0.01
  E  簇否决 × θ     与 C 合并：C 的每一档都扫全 θ，工作点按 self-cal 另选
  D  跨机位边切开    split_thr 0.15–0.25，另跑一组叠加 veto 0.30
  G  消融            全局聚类 0.20 + veto；机位感知改成员级 complete-linkage

产出（results/）：
  summary.csv        每个配置一行，列对齐清单 D7 §165
  curve.csv          (config, theta) 长表，画图 2 用
  queries/<cfg>.csv  逐查询输出，D2 格式
  best.json          各配置的 self-cal 工作点

口径约定（写进 D0 §1.2b）：
  final_output_rate       主口径：两片均过 θ 且同簇  / 229
  final_output_rate_loose 辅口径：至少一片过 θ       / 229
  ab_split_rate           两片均过但异簇 / 两片均过

相对 baseline 的 added/dropped/relabeled **按簇成员集合比较**，不按簇号——
各配置的 kept 下标互不可比（deliverable/.../README_rules.md §2.6）。

运行：
  PY="C:/Users/muyizWindows/.workbuddy/binaries/python/envs/dml_torch/Scripts/python.exe"
  PYTHONPATH=D:/human-count/pylibs $PY tools/d7_sweep.py
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from selector import _norm                                    # noqa: E402
import session_selector_v1 as S                               # noqa: E402
import eval_mask_variants as M                                # noqa: E402
import cluster_camera_aware as CA                             # noqa: E402
from eval_selector_1012test import cooc_pairs                 # noqa: E402

EMB = os.path.join(S.ROOT, "output_crosscam_local", "1012_yolo8_dd_emb")
DET = os.path.join(S.ROOT, "output_crosscam_local", "1012_yolo8_dd")
OUT = os.path.join(S.ROOT, "deliverable/data-request-v2/results")

THR_IN, THR_OUT = 0.15, 0.25          # 生产基线
SAME_MIN, MIN_EVAL = 0.90, 100        # self-cal 判据（只约束 same，不约束 fm）
GRID = np.round(np.arange(1.40, 4.501, 0.05), 3)


# ---------------------------------------------------------------- 数据

CAM_ALIAS = {"走廊通道": "行政走廊通道"}   # 10-23 段名简写，OSD 实测同名


def cam_of(seg):
    """从段名取机位标识。

    生产 6cam：   '318_6cam_tr'          -> '318'
    E21 三天：    '2026-10-22_办公区1_tr' -> '办公区1'

    以首段是否为 'YYYY-MM-DD' 形式判断，避免把日期当年机位。

    ★ 别名归一：10-23 的段名写作「走廊通道」，但 OSD 实测为
    「58所-成都-14楼-行政走廊通道」，与 10-22/10-26 同一物理机位
    （docs/e21-results-2026-11-09.md §5）。不归一会让跨天同机位对照错配。
    当日独立聚类时该归一不改变任何结果（每天只有一种走廊）。
    """
    p = seg.split("_")
    if len(p) >= 2 and p[0][:4].isdigit() and "-" in p[0]:
        return CAM_ALIAS.get(p[1], p[1])
    return CAM_ALIAS.get(p[0], p[0])


def load_data(emb_dir=EMB, det_dir=DET, segs_tr=None, segs_te=None):
    segs_tr = segs_tr or S.TRAIN_SEGS
    segs_te = segs_te or S.TEST_SEGS
    ktr, Atr, Btr = M.load_split(emb_dir, segs_tr)
    kte, Ate, Bte = M.load_split(emb_dir, segs_te)
    Vtr = _norm((Atr + Btr) / 2.0)
    cams = np.array([cam_of(k[0]) for k in ktr])
    keys_ok = set(kte)
    cooc = []
    for s in segs_te:
        p = os.path.join(det_dir, s, f"{s}_detections.jsonl")
        if os.path.exists(p):
            cooc += [((s, u), (s, v)) for u, v in cooc_pairs(p, keys_ok, s, 5)]
    cooc = [(u, v) for u, v in cooc if u in keys_ok and v in keys_ok]
    return ktr, kte, Ate, Bte, Vtr, cams, cooc


# ---------------------------------------------------------------- 聚类

def make_cluster(cfg, Vtr, cams):
    kind = cfg.get("cluster", "camera")
    if kind == "camera":
        return CA.camera_aware_labels(Vtr, cams, cfg.get("thr_in", THR_IN),
                                      cfg.get("thr_out", THR_OUT), S.MIN_CLUSTER,
                                      member_level=cfg.get("member_level", False))
    if kind == "global":
        return S.hier_labels(Vtr, cfg.get("hier_thr", 0.20))
    if kind == "split":
        if cfg.get("base", "camera") == "global":
            lab, kept = S.hier_labels(Vtr, cfg.get("hier_thr", 0.20))
        else:
            lab, kept = CA.camera_aware_labels(Vtr, cams, cfg.get("thr_in", THR_IN),
                                               cfg.get("thr_out", THR_OUT), S.MIN_CLUSTER)
        return CA.split_cross_camera(lab, kept, Vtr, cams, cfg["thr_split"])
    raise ValueError(f"unknown cluster kind: {kind}")


def apply_veto(lab, kept, Vtr, veto):
    if veto is None:
        return list(kept), set()
    drop = set()
    for c in kept:
        idx = np.where(lab == c)[0]
        if idx.size >= 2:
            Xc = Vtr[idx]
            dmax = float(1.0 - (Xc @ Xc.T)[np.triu_indices(idx.size, 1)].min())
        else:
            dmax = 0.0
        if dmax >= veto:
            drop.add(c)
    return [c for c in kept if c not in drop], drop


def score(lab, kept, Vtr, Ate, Bte):
    K = len(kept)
    mu_g = Vtr.mean(0)
    protos = _norm(np.asarray([_norm(Vtr[lab == c].mean(0, keepdims=True))[0]
                               for c in kept], np.float32) - mu_g)
    pcl = np.arange(K)
    m_hat, s_hat, _ = S.fit_ident_stats(S.score_rows(Vtr, protos, pcl, K, mu=mu_g), lab, kept)
    mu_q = _norm((Ate + Bte) / 2.0).mean(0)
    Sa = S.score_rows(Ate, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    Sb = S.score_rows(Bte, protos, pcl, K, mu=mu_q, calib=(m_hat, s_hat))
    return Sa, Sb, K


# ---------------------------------------------------------------- 评估

def eval_fingerprint(cooc, ev_pairs=None):
    """评估集指纹：样本数 + 排序后的 sha1 前 16 位。

    用途：离线对齐「两次比较的分母是不是同一批样本」。任何跨配置/跨日的 fm
    比较，都必须先比对这两个值（见 docs/e21-results-2026-11-09.md §13）。
    """
    def dig(pairs):
        h = hashlib.sha1()
        for u, v in sorted(pairs):
            h.update(f"{u}\x1f{v}\x1e".encode("utf-8"))
        return h.hexdigest()[:16]

    return {"cooc_n": len(cooc), "cooc_sha1_16": dig(cooc),
            "fm_eval_n": len(ev_pairs) if ev_pairs is not None else 0,
            "fm_eval_sha1_16": dig(ev_pairs) if ev_pairs is not None else ""}


def evaluable_pairs(Sa, Sb, keys, cooc, thr):
    """某个 θ 下「可评估」的共现对（历史口径的分母样本）。

    用于把该集合固化成共同评估集，供其它配置在 `--fm-eval file` 下复用。
    """
    sa, sb = Sa.max(1), Sb.max(1)
    acca, accb = sa >= thr, sb >= thr
    kidx = {k: i for i, k in enumerate(keys)}
    out = []
    for u, v in cooc:
        iu, iv = kidx[u], kidx[v]
        if (acca[iu] and accb[iv]) or (acca[iv] and accb[iu]):
            out.append((u, v))
    return out


def eval_grid(Sa, Sb, kte, cooc, grid, eval_pairs=None):
    """逐 θ 评估：三指标 + 最终输出函数的三列。"""
    ia, sa = Sa.argmax(1), Sa.max(1)
    ib, sb = Sb.argmax(1), Sb.max(1)
    same_mask = ia == ib
    rows = []
    for t in grid:
        r = S.eval_point(Sa, Sb, kte, cooc, float(t), eval_pairs=eval_pairs)
        acca, accb = sa >= t, sb >= t
        both = acca & accb
        strict = both & same_mask
        loose = acca | accb
        r.update({
            "final_output_rate": float(strict.mean()),
            "final_output_rate_loose": float(loose.mean()),
            "ab_split_rate": (float((both & ~same_mask).sum() / both.sum())
                              if both.any() else float("nan")),
            "n_strict": int(strict.sum()),
            "n_loose": int(loose.sum()),
        })
        rows.append(r)
    return rows


def pick(rows, same_min=SAME_MIN, min_eval=MIN_EVAL, margin=0.0, fm_budget=-1.0):
    """self-cal 选点。

    margin>0 时启用邻域稳健性约束：要求该点 θ±margin 内的所有网格点也满足判据，
    否则视为边界解不采用。动机见 results/D7-report.md §二.4 ——
    split0.20+veto0.30 的 cov 0.699 来自 θ=1.75 这个孤岛（1.70/1.80 均破线），
    原规则「可行点里取覆盖最大」必然挑中它。

    fm_budget>0 时把假合并率并入可行域；<=0（默认）不约束，保持历史行为。

    ★ 为什么需要 fm 预算：只看 {same, evaluable} 时，「可行点里取覆盖最大」必然
      选中最松的 θ —— coverage 最大、fm 也最大。跨配置比较时各配置独立校准到
      自己的最松点，工作点不在同一条质量线上，coverage 之差会把「用了更松的阈值」
      误读成「算法更好」。实测见 docs/e21-results-2026-11-09.md §3：
      E21 三天 A_baseline 的 fm 分别为 10.1% / 28.4% / 7.4%，而收紧到同一预算后
      10-23 的 +0.056「增益」直接消失。
    """
    budget_on = fm_budget is not None and fm_budget > 0

    def ok(r):
        base = (np.isfinite(r["same"]) and r["same"] >= same_min
                and r["evaluable"] >= min_eval)
        if not base:
            return False
        if budget_on:
            fm = r["false_merge"]
            if not np.isfinite(fm) or fm > fm_budget:
                return False
        return True

    feas = [r for r in rows if ok(r)]
    if not feas:
        return None
    if margin <= 0:
        return max(feas, key=lambda r: r["coverage"])

    idx = {round(r["thr"], 4): r for r in rows}
    stable = []
    for r in feas:
        t = r["thr"]
        nb = [idx[k] for k in idx if abs(k - t) <= margin + 1e-9]
        if nb and all(ok(x) for x in nb):
            stable.append(r)
    pool = stable if stable else feas
    return max(pool, key=lambda r: r["coverage"])


def infeasible_reason(rows, same_min=SAME_MIN, min_eval=MIN_EVAL, fm_budget=-1.0):
    """self-cal 无可行点时，给出是「same/evaluable 不满足」还是「fm 预算不满足」。

    区分这两者很重要：前者说明该配置在该日根本没跑出可用工作点（数据问题），
    后者说明质量预算太严、该配置达不到（配置能力问题）。混为一谈会误判。
    """
    base = [r for r in rows
            if np.isfinite(r["same"]) and r["same"] >= same_min
            and r["evaluable"] >= min_eval]
    if not base:
        return "same/evaluable 无可行点"
    if fm_budget is not None and fm_budget > 0:
        fms = [r["false_merge"] for r in base if np.isfinite(r["false_merge"])]
        fmin = min(fms) if fms else float("nan")
        if np.isfinite(fmin):
            return (f"fm 预算 {fm_budget*100:.1f}% 不可行"
                    f"（该配置在 same≥{same_min:.2f} 下最低 fm={fmin*100:.1f}%）")
        return "fm 预算不可行（无有效 fm 值）"
    return "unknown"


def per_query(cfgname, Sa, Sb, kte, lab, kept, theta):
    """D2 格式的逐查询输出。簇号是本配置内的 kept 下标，跨配置不可比。"""
    pos = {c: i for i, c in enumerate(kept)}
    ia, sa = Sa.argmax(1), Sa.max(1)
    ib, sb = Sb.argmax(1), Sb.max(1)
    out = []
    for i, k in enumerate(kte):
        acca = bool(sa[i] >= theta)
        accb = bool(sb[i] >= theta)
        ca, cb = int(ia[i]), int(ib[i])
        if acca and accb and ca == cb:
            st, fin = "accepted", pos[kept[ca]]
        elif not (acca and accb):
            st, fin = "rejected_low_score", ""
        else:
            st, fin = "rejected_ab_split", ""
        out.append({"config": cfgname, "query_id": f"{k[0]}#{k[1]}",
                    "cam": k[0].split("_")[0],
                    "score_a": round(float(sa[i]), 4), "score_b": round(float(sb[i]), 4),
                    "accept_a": int(acca), "accept_b": int(accb),
                    "cluster_a_local": ca, "cluster_b_local": cb,
                    "final_status": st, "final_cluster_local": fin})
    return out


def members_of(lab, kept, c, ktr):
    """簇 c 的成员集合（'seg#tid' 字符串集合）——跨配置可比的唯一标识。"""
    idx = np.where(lab == c)[0]
    return frozenset(f"{ktr[i][0]}#{ktr[i][1]}" for i in idx)


# ---------------------------------------------------------------- 配置表

def build_cfgs():
    cfgs = [dict(name="A_baseline", group="A", cluster="camera",
                 note="机位感知图库、无否决、只调 θ")]

    for v in np.round(np.arange(0.15, 0.401, 0.01), 2):
        cfgs.append(dict(name=f"C_dmax{v:.2f}", group="C", cluster="camera",
                         veto=float(v), note="删簇重搜（=清单 B/C 合并，B 已证等价）"))

    for t in (0.15, 0.17, 0.18, 0.20, 0.22, 0.25):
        cfgs.append(dict(name=f"D_split{t:.2f}", group="D", cluster="split",
                         thr_split=t, note="跨机位 MST 长边切开"))
        cfgs.append(dict(name=f"D_split{t:.2f}_veto0.30", group="D", cluster="split",
                         thr_split=t, veto=0.30, note="切开 + 簇否决"))

    cfgs.append(dict(name="G_global0.20_veto0.30", group="G", cluster="global",
                     hier_thr=0.20, veto=0.30, note="全局聚类 0.20 + 否决"))
    cfgs.append(dict(name="G_member_complete", group="G", cluster="camera",
                     member_level=True, note="跨机位改成员级 complete-linkage"))
    cfgs.append(dict(name="G_member_complete_veto0.30", group="G", cluster="camera",
                     member_level=True, veto=0.30, note="成员级 complete + 否决"))
    return cfgs


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑名称含这些子串的配置（逗号分隔）")
    ap.add_argument("--emb", default=EMB,
                    help="表征目录（默认生产图库；扩图库/换编码时指向新目录）")
    ap.add_argument("--det", default=DET, help="检测目录（供共现对计算，默认生产）")
    ap.add_argument("--train-segs", default="", help="图库段，逗号分隔（默认 1012 四机位）")
    ap.add_argument("--test-segs", default="", help="查询段，逗号分隔（默认 1012 四机位）")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--grid", default="", help="形如 1.40,4.50,0.05；默认用内置 GRID")
    ap.add_argument("--robust-margin", type=float, default=0.0,
                    help="self-cal 邻域稳健性：要求 θ±margin 内均可行（默认 0=关，沿用历史口径）")
    ap.add_argument("--min-eval", type=int, default=MIN_EVAL,
                    help="self-cal 的最少可评估查询数（默认 100；小数据集按 VALIDATION §4.1 降到可评估数的一半）")
    ap.add_argument("--same-min", type=float, default=SAME_MIN,
                    help="self-cal 的 a/b 同簇率下限（默认 0.90）。仅用于诊断性放宽，须在报告里注明")
    ap.add_argument("--fm-budget", type=float, default=-1.0,
                    help="self-cal 的假合并率上限（默认 -1 = 不约束，保持历史行为）。"
                         "给定后与 same/evaluable 一并构成可行域；"
                         "若该配置达不到预算则跳过并说明原因，不做静默退化")
    ap.add_argument("--fm-eval", default="adaptive",
                    choices=["adaptive", "all", "baseline", "file"],
                    help="fm 的评估集口径。adaptive=历史口径，分母随 θ 变化，**不可跨配置比较**；"
                         "all=分母固定为全部共现对（**判帕累托/优劣时用这个**，"
                         "它在所有配置上同一分母，等价于比较绝对误合并对数）；"
                         "baseline=分母取本批第一个配置工作点上的可评估对，"
                         "属「条件于 A 的决定」，会**偏袒接受更宽的配置**（A 未决定的对不计 C 的错），"
                         "仅用于回答「在 A 已决定的那些对上谁更准」；"
                         "file=用 --fm-eval-pairs 指定的固定集合（复现某次比较时用）")
    ap.add_argument("--fm-eval-pairs", default="",
                    help="--fm-eval file 时的固定共现对文件（JSON 数组 [[u,v],...]）；"
                         "每次运行都会把各配置的评估集导出到 <out>/pairs/<cfg>.json")
    ap.add_argument("--thr-in", type=float, default=THR_IN,
                    help=f"机位内合并阈值（默认 {THR_IN}，即生产值）。改它会同时改变 fm_floor")
    ap.add_argument("--thr-out", type=float, default=THR_OUT,
                    help=f"跨机位子簇质心 complete-linkage 阈值（默认 {THR_OUT}，即生产值）。"
                         "E21 §14 实测 0.20 时覆盖持平、污染降 28%%")
    a = ap.parse_args()

    if a.grid:
        lo, hi, step = [float(x) for x in a.grid.split(",")]
        grid = np.round(np.arange(lo, hi + 1e-9, step), 3)
    else:
        grid = GRID

    os.makedirs(os.path.join(a.out, "queries"), exist_ok=True)
    t_all = time.time()
    ktr, kte, Ate, Bte, Vtr, cams, cooc = load_data(
        a.emb, a.det,
        [x.strip() for x in a.train_segs.split(",") if x.strip()] or None,
        [x.strip() for x in a.test_segs.split(",") if x.strip()] or None)
    print(f"[data] 图库 {len(ktr)} / 查询 {len(kte)} / 共现对 {len(cooc)} | "
          f"θ 网格 {len(grid)} 点 [{grid[0]}, {grid[-1]}] | emb={os.path.basename(a.emb)}")

    cfgs = build_cfgs()
    # 阈值透传到每个配置（配置表里不写死，便于整批换生产参数后重跑对照）
    for c in cfgs:
        c.setdefault("thr_in", a.thr_in)
        c.setdefault("thr_out", a.thr_out)
    if (a.thr_in, a.thr_out) != (THR_IN, THR_OUT):
        print(f"[cfg] 覆盖聚类阈值：thr_in={a.thr_in:.2f} thr_out={a.thr_out:.2f}"
              f"（生产默认 {THR_IN:.2f}/{THR_OUT:.2f}）")
    if a.only:
        keys = [x.strip() for x in a.only.split(",") if x.strip()]
        cfgs = [c for c in cfgs if any(k in c["name"] for k in keys)]
    print(f"[cfg] 待跑 {len(cfgs)} 个配置\n")

    # ── fm 评估集：None=历史口径；否则固定集合，分母与 θ 无关 ──
    ev_pairs = None
    if a.fm_eval == "all":
        ev_pairs = list(cooc)
    elif a.fm_eval == "file":
        with open(a.fm_eval_pairs, encoding="utf-8") as fh:
            raw = json.load(fh)
        kset = set(kte)
        ev_pairs = []
        for u, v in raw:
            u = tuple(u) if isinstance(u, list) else u   # JSON 把 tuple 存成 list
            v = tuple(v) if isinstance(v, list) else v
            if u in kset and v in kset:
                ev_pairs.append((u, v))
        if not ev_pairs:
            raise SystemExit(f"--fm-eval-pairs 与当前查询集无交集：{a.fm_eval_pairs}")
    elif a.fm_eval == "baseline":
        if not cfgs:
            raise SystemExit("--fm-eval baseline 需要至少一个配置")
        ref = cfgs[0]
        lab_r, kept_r0 = make_cluster(ref, Vtr, cams)
        kept_r, _ = apply_veto(lab_r, kept_r0, Vtr, ref.get("veto"))
        if not kept_r:
            raise SystemExit(f"--fm-eval baseline：参考配置 {ref['name']} 无簇可用")
        Sa_r, Sb_r, _Kr = score(lab_r, kept_r, Vtr, Ate, Bte)
        rows_r = eval_grid(Sa_r, Sb_r, kte, cooc, grid)
        b_r = pick(rows_r, margin=a.robust_margin, min_eval=a.min_eval,
                   same_min=a.same_min)          # 参考点不施加 fm 预算（否则循环依赖）
        if b_r is None:
            raise SystemExit(f"--fm-eval baseline：参考配置 {ref['name']} 无可行点")
        ev_pairs = evaluable_pairs(Sa_r, Sb_r, kte, cooc, b_r["thr"])
        print(f"[fm] 参考 {ref['name']}：θ={b_r['thr']:.2f} → 共同评估集 {len(ev_pairs)} 对"
              f"（占全部共现对 {len(ev_pairs)/max(1,len(cooc))*100:.1f}%）")
    fp = eval_fingerprint(cooc, ev_pairs)
    if ev_pairs is None:
        print(f"[fm] 口径 adaptive（历史）｜cooc={fp['cooc_n']} sha1={fp['cooc_sha1_16']}"
              f"  ⚠ 分母随 θ 变化，fm 不可跨配置比较")
    else:
        print(f"[fm] 口径 {a.fm_eval}（固定）｜cooc={fp['cooc_n']} sha1={fp['cooc_sha1_16']}"
              f"｜评估集 n={fp['fm_eval_n']} sha1={fp['fm_eval_sha1_16']}")

    if a.fm_budget and a.fm_budget > 0:
        print(f"[self-cal] 质量预算：fm ≤ {a.fm_budget*100:.1f}%"
              f"（达不到的配置跳过，不做静默退化）")

    summ, curves, bests = [], [], {}
    ref_members = None     # baseline 的每条查询 → 簇成员集合

    for n, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        lab, kept0 = make_cluster(cfg, Vtr, cams)
        kept, vetoed = apply_veto(lab, kept0, Vtr, cfg.get("veto"))
        if not kept:
            print(f"[{cfg['name']}] 无簇可用，跳过")
            continue
        Sa, Sb, K = score(lab, kept, Vtr, Ate, Bte)
        rows = eval_grid(Sa, Sb, kte, cooc, grid, eval_pairs=ev_pairs)
        best = pick(rows, margin=a.robust_margin, min_eval=a.min_eval,
                    same_min=a.same_min, fm_budget=a.fm_budget)
        if best is None:
            why = infeasible_reason(rows, same_min=a.same_min,
                                    min_eval=a.min_eval, fm_budget=a.fm_budget)
            print(f"[{cfg['name']}] K={K} self-cal 无可行点（{why}），跳过")
            continue

        # 逐查询输出（self-cal 工作点）
        pq = per_query(cfg["name"], Sa, Sb, kte, lab, kept, best["thr"])
        with open(os.path.join(a.out, "queries", f"{cfg['name']}.csv"), "w",
                  newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pq[0].keys()))
            w.writeheader()
            w.writerows(pq)

        # 导出该配置工作点上的可评估对 —— 供 --fm-eval file 复现同口径比较
        os.makedirs(os.path.join(a.out, "pairs"), exist_ok=True)
        with open(os.path.join(a.out, "pairs", f"{cfg['name']}.json"), "w",
                  encoding="utf-8") as fh:
            json.dump([list(p) for p in
                       evaluable_pairs(Sa, Sb, kte, cooc, best["thr"])], fh)

        # 簇 → 成员（供离线接真值：本地簇号每配置不同，必须靠成员集合对应）
        os.makedirs(os.path.join(a.out, "members"), exist_ok=True)
        mrows = []
        for cl, c in enumerate(kept):
            for t in sorted(members_of(lab, kept, c, ktr)):
                mrows.append({"config": cfg["name"], "cluster_local": cl,
                              "cluster_lab": int(c), "track_id": t})
        with open(os.path.join(a.out, "members", f"{cfg['name']}.csv"), "w",
                  newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["config", "cluster_local",
                                               "cluster_lab", "track_id"])
            w.writeheader()
            w.writerows(mrows)

        # 相对 baseline 的转移：按簇成员集合比较，不按簇号
        mem = {}
        pos = {c: i for i, c in enumerate(kept)}
        for r in pq:
            if r["final_status"] == "accepted":
                mem[r["query_id"]] = members_of(lab, kept, kept[int(r["final_cluster_local"])], ktr)
        if cfg["name"] == "A_baseline":
            ref_members = mem
            added = dropped = relabeled = ""
        else:
            a_ = d_ = r_ = 0
            for q in {x["query_id"] for x in pq}:
                inb, inv = q in ref_members, q in mem
                if inv and not inb:
                    a_ += 1
                elif inb and not inv:
                    d_ += 1
                elif inb and inv and ref_members[q] != mem[q]:
                    r_ += 1
            added, dropped, relabeled = a_, d_, r_

        row = {
            "config": cfg["name"], "group": cfg["group"], "K": K,
            "thr_in": cfg["thr_in"], "thr_out": cfg["thr_out"],
            "theta": round(float(best["thr"]), 2),
            "n_gallery": len(ktr), "n_query": len(kte),
            "both_accept_rate": round(best["coverage"], 4),
            "ab_same_rate": round(best["same"], 4),
            "final_output_rate": round(best["final_output_rate"], 4),
            "final_output_rate_loose": round(best["final_output_rate_loose"], 4),
            "ab_split_rate": round(best["ab_split_rate"], 4),
            "C": "", "W": "", "U": "", "precision": "",
            "C_over_229": "",
            "fm_raw": round(best["false_merge"], 4) if np.isfinite(best["false_merge"]) else "",
            "fm_eval": a.fm_eval,
            "fm_eval_n": fp["fm_eval_n"],
            "fm_adapt": (round(best["false_merge_adaptive"], 4)
                         if np.isfinite(best.get("false_merge_adaptive", float("nan")))
                         else round(best["false_merge"], 4) if np.isfinite(best["false_merge"]) else ""),
            "fm_clean": "",
            "merged_identities": "",
            "added": added, "dropped": dropped, "relabeled": relabeled,
            "note": cfg.get("note", "") + "｜C/W/U 待 D4 真值",
        }
        summ.append(row)
        bests[cfg["name"]] = {**best, "K": K,
                              "thr_in": cfg["thr_in"], "thr_out": cfg["thr_out"],
                              "n_vetoed": len(vetoed),
                              "n_kept_before_veto": len(kept0),
                              "fm_budget": a.fm_budget,
                              "fm_eval": a.fm_eval,
                              **fp}
        for r in rows:
            fm_adapt = r.get("false_merge_adaptive", r["false_merge"])
            curves.append({"config": cfg["name"], "group": cfg["group"], "K": K,
                           "theta": r["thr"],
                           "coverage": round(r["coverage"], 4),
                           "same": round(r["same"], 4) if np.isfinite(r["same"]) else "",
                           "fm": round(r["false_merge"], 5) if np.isfinite(r["false_merge"]) else "",
                           "fm_adapt": round(fm_adapt, 5) if np.isfinite(fm_adapt) else "",
                           "evaluable": r["evaluable"],
                           "fm_eval_n": fp["fm_eval_n"],
                           "final_output_rate": round(r["final_output_rate"], 4),
                           "final_output_rate_loose": round(r["final_output_rate_loose"], 4),
                           "ab_split_rate": round(r["ab_split_rate"], 4) if np.isfinite(r["ab_split_rate"]) else ""})

        print(f"[{n}/{len(cfgs)}] {cfg['name']:<28} K={K:<4} θ={best['thr']:.2f} "
              f"cov={best['coverage']:.3f} same={best['same']:.3f} "
              f"fm={best['false_merge']*100:5.2f}% out={best['final_output_rate']:.3f} "
              f"({time.time()-t0:.1f}s)")

    # 落盘
    if summ:
        with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
            w.writeheader()
            w.writerows(summ)
    if curves:
        with open(os.path.join(a.out, "curve.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(curves[0].keys()))
            w.writeheader()
            w.writerows(curves)
    # best.json 是唯一会被「全跳过」清空的产物 —— 预算过严时不能覆盖已有结果
    if bests:
        with open(os.path.join(a.out, "best.json"), "w", encoding="utf-8") as fh:
            json.dump(bests, fh, ensure_ascii=False, indent=1)
    else:
        print("[warn] 无任何配置通过 self-cal，未写 best.json（避免覆盖已有结果）")

    print(f"\n[done] {len(summ)} 配置 / {len(curves)} 曲线点 | "
          f"{time.time()-t_all:.0f}s | {a.out}")


if __name__ == "__main__":
    main()
