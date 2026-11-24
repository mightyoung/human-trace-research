#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E21 跨天身份绑定：用高置信度人脸把「最新一天的人」绑到图库日的身份簇上。

问题
----
跨天（A 日建库、B 日查询）时，外观会把陌生人硬塞进最近的旧簇（§8.2）。本实验
问的是：**能不能用 B 日自己的高置信度人脸，把 B 日的人员信息绑到 A 日的身份簇
上**，从而让簇内信息保持"最新的那个人"，而不是不断膨胀的旧簇。

三条链路（同一份代码，只换图库与查询）：
  ① 同日对照   G = A.tr      Q = A.te
  ② 跨天       G = A.tr      Q = B.te
  ③ 跨天大库   G = A.tr+A.te Q = B.te

度量（全部免标注，故可跨天比较）
--------------------------------
- **可绑定率**：B 日轨里同时满足「有可用脸」且「同机位图库簇有脸原型」的比例。
  这是「这件事能不能做」的分母，直接受 P1 的像素预算约束。
- **人脸—外观一致率**：两边都给出裁定时，指的是不是同一个簇。两条独立通道一致
  是双方都对的正向证据；不一致则是复核队列的入口。
- **共现冲突（fm）**：同机位同帧出现 ≥10 帧的两条 B 日轨，被判定成同一个身份簇
  的比例。同时在场是物理事实，与脸无关，所以这个指标**不循环**。
- **同人合并率（人脸伪真值）**：B 日内两条轨的人脸余弦 ≥ 0.6（且不共现）视作
  同一人，量它们是否落进同一个簇。★ 这条对人脸绑定是半循环的：绑定本身也用人脸。
  所以它只用来读「碎片是否被收拢」，不与 fm 并列为结论。

口径提醒
--------
分数是**裸余弦**（0–1 量纲），不是 e21 那套 mean/std 校准分（θ 到 4.5）。
所以本文件的 θ 与 `d7_sweep` 的 θ 不可互换；跨天衰减只在本文件内部比较。

用法
----
    python tools/e21_xday_bind.py --faces tmp/e21_face            # 人脸 JSON 目录
    python tools/e21_xday_bind.py --sweep                          # 扫 τ_face / margin
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import collections

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB = os.path.join(ROOT, "tmp", "e21_local", "emb")
DET = os.path.join(ROOT, "tmp", "e21_local", "det_gpu")

DAYS = {"A": "2026-10-22", "B": "2026-10-26"}
CAMS = {
    "2026-10-22": ["办公区1", "办公区2", "办公区球机", "茶水间", "行政走廊通道", "电梯间"],
    "2026-10-26": ["办公区1", "办公区2", "办公区球机", "茶水间", "行政走廊通道", "电梯间"],
}
# 两天同名机位才可比
SHARED = [c for c in CAMS["2026-10-22"] if c in CAMS["2026-10-26"]]

HIER_THR = 0.15        # 机位内聚类阈值（与部署流水线一致）
MIN_TRACK_H = 60.0     # 图库模板质量门：框高中位低于此不入模板
COOC_MIN_FRAMES = 10   # 同帧 >= 10 帧才算"同时在场"
FACE_SAME = 0.60       # 同人伪真值的脸余弦门


# ----------------------------------------------------------------- 数据加载

def _norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def seg_name(day, cam, split):
    return f"{day}_{cam}_{split}"


def load_emb(day, cam, split):
    """返回 (keys, V)；V 为 a/b 两视图均值再归一化。"""
    pa = os.path.join(EMB, f"{seg_name(day,cam,split)}_emb_a.npz")
    pb = os.path.join(EMB, f"{seg_name(day,cam,split)}_emb_b.npz")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return [], np.zeros((0, 512), np.float32)
    za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
    keys = [k for k in za.files if k in set(zb.files)]
    if not keys:
        return [], np.zeros((0, 512), np.float32)
    A = np.asarray([za[k] for k in keys], np.float32)
    B = np.asarray([zb[k] for k in keys], np.float32)
    return keys, _norm(A + B)


def load_face(path):
    """返回 {seg#tid: (emb, n_used, h_med)}；只收 n_used>0。"""
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))["tracks"]
    out = {}
    for k, v in d.items():
        if v.get("n_used") and v.get("emb"):
            out[k] = (np.asarray(v["emb"], np.float32), int(v["n_used"]),
                      float(v.get("h_med") or 0.0))
    return out


def load_det(seg):
    """返回 {tid: (frames set, h_med)}。"""
    p = os.path.join(DET, seg, f"{seg}_detections.jsonl")
    if not os.path.exists(p):
        return None
    per = collections.defaultdict(list)
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for t, b in zip(d["track_ids"], d["boxes"]):
                per[int(t)].append((fr, b[3] - b[1]))
    return {t: (set(fr for fr, _ in xs), float(np.median([h for _, h in xs])))
            for t, xs in per.items()}


# ----------------------------------------------------------------- 图库

def hier_clusters(V, thr):
    """机位内 average-linkage 层次聚类，余弦距离阈值 thr。"""
    n = len(V)
    if n == 0:
        return []
    if n == 1:
        return [0]
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    D = pdist(V, metric="cosine")
    Z = linkage(D, method="average")
    return fcluster(Z, t=thr, criterion="distance").tolist()


def build_gallery(day, cams, splits, faces):
    """按机位建图库簇；返回 {cam: dict(template, face, h_meds, members, key_of)}"""
    gal = {}
    for cam in cams:
        keys, V = [], np.zeros((0, 512), np.float32)
        for sp in splits:
            k2, v2 = load_emb(day, cam, sp)
            if not k2:
                continue
            keys += k2
            V = np.vstack([V, v2]) if len(V) else v2
        if not keys:
            continue
        det = {}
        for sp in splits:
            d = load_det(seg_name(day, cam, sp))
            if d:
                det.update({f"{seg_name(day,cam,sp)}#{t}": v for t, v in d.items()})
        hm = np.asarray([det.get(k, (None, 0.0))[1] for k in keys], np.float32)
        lab = np.asarray(hier_clusters(V, HIER_THR))
        clusters = {}
        for c in sorted(set(lab.tolist())):
            idx = np.where(lab == c)[0]
            good = [i for i in idx if hm[i] >= MIN_TRACK_H] or list(idx)
            T = _norm(V[good].mean(0))
            fv = [faces[keys[i]][0] for i in idx if keys[i] in faces]
            fproto = _norm(np.mean(fv, axis=0)) if fv else None
            clusters[c] = dict(template=T, face=fproto,
                               n=int(len(idx)), n_face=len(fv),
                               h=np.asarray([hm[i] for i in good], np.float32),
                               embs=V[good])
        gal[cam] = dict(clusters=clusters, keys=keys)
    return gal


# ----------------------------------------------------------------- 打分

def appearance(qv, qh, clusters, use_kernel, sigma=0.6):
    """返回 (best_cluster, best_score, second_score)。"""
    ids, sc = [], []
    for c, d in clusters.items():
        T = d["template"]
        if use_kernel and len(d["h"]):
            w = np.exp(-np.abs(np.log(max(qh, 1e-3) / np.maximum(d["h"], 1e-3))) / sigma)
            T = _norm((w[:, None] * d["embs"]).sum(0))
        ids.append(c)
        sc.append(float(T @ qv))
    sc = np.asarray(sc, np.float32)
    if not len(sc):
        return None, -1.0, -1.0
    o = np.argsort(-sc)
    return ids[o[0]], float(sc[o[0]]), float(sc[o[1]]) if len(sc) > 1 else -1.0


def face_best(qf, clusters):
    ids, sc = [], []
    for c, d in clusters.items():
        if d["face"] is None:
            continue
        ids.append(c)
        sc.append(float(d["face"] @ qf))
    if not sc:
        return None, -1.0, -1.0
    sc = np.asarray(sc, np.float32)
    o = np.argsort(-sc)
    return ids[o[0]], float(sc[o[0]]), float(sc[o[1]]) if len(sc) > 1 else -1.0


# ----------------------------------------------------------------- 主实验

def run_cell(name, gal_day, gal_splits, q_day, q_split, faces_dir, args):
    cams = [c for c in SHARED if c in CAMS[gal_day] and c in CAMS[q_day]]
    faces_g, faces_q = {}, {}
    for cam in cams:
        for sp in gal_splits:
            f = load_face(os.path.join(faces_dir, f"{seg_name(gal_day,cam,sp)}.json"))
            if f:
                faces_g.update(f)
        f = load_face(os.path.join(faces_dir, f"{seg_name(q_day,cam,q_split)}.json"))
        if f:
            faces_q.update(f)

    gal = build_gallery(gal_day, cams, gal_splits, faces_g)

    rows = []
    for cam in cams:
        if cam not in gal:
            continue
        keys, V = load_emb(q_day, cam, q_split)
        if not keys:
            continue
        seg = seg_name(q_day, cam, q_split)
        det = load_det(seg) or {}
        hq = np.asarray([det.get(int(k.split("#")[1]), (None, 0.0))[1] for k in keys],
                        np.float32)
        for i, k in enumerate(keys):
            f = faces_q.get(k)
            qf = f[0] if f else None
            for use_kernel, tag in ((False, "S0"), (True, "S1")):
                c_id, s1, s2 = appearance(V[i], float(hq[i]), gal[cam]["clusters"], use_kernel)
                rows.append(dict(cell=name, cam=cam, key=k, cfg=tag, app=c_id,
                                 app_score=s1, app_margin=s1 - s2,
                                 has_face=bool(qf is not None),
                                 face_id=None, face_score=None, face_margin=None,
                                 h=float(hq[i])))
            if qf is not None:
                fid, fs1, fs2 = face_best(qf, gal[cam]["clusters"])
                for r in rows[-2:]:
                    r.update(face_id=fid, face_score=fs1, face_margin=fs1 - fs2)

    # 共现对（负例）
    cooc = collections.defaultdict(set)
    for cam in cams:
        seg = seg_name(q_day, cam, q_split)
        det = load_det(seg) or {}
        items = [(f"{seg}#{t}", v[0]) for t, v in det.items()]
        for a in range(len(items)):
            ka, fa = items[a]
            for b in range(a + 1, len(items)):
                kb, fb = items[b]
                if len(fa & fb) >= COOC_MIN_FRAMES:
                    cooc[cam].add((min(ka, kb), max(ka, kb)))

    # 同人伪真值对（人脸）
    same = collections.defaultdict(set)
    for cam in cams:
        seg = seg_name(q_day, cam, q_split)
        ks = [f"{seg}#{t}" for t in (load_det(seg) or {})]
        ks = [k for k in ks if k in faces_q]
        F = _norm(np.asarray([faces_q[k][0] for k in ks])) if ks else np.zeros((0, 512))
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                if float(F[a] @ F[b]) >= FACE_SAME:
                    pair = (min(ks[a], ks[b]), max(ks[a], ks[b]))
                    same[cam].add(pair)

    # 当天同机位人脸分组（连通分量，余弦 >= FACE_GROUP）
    groups = {}
    for cam in cams:
        seg = seg_name(q_day, cam, q_split)
        ks = [k for k in faces_q if k.startswith(seg + "#")]
        if len(ks) < 2:
            continue
        F = _norm(np.asarray([faces_q[k][0] for k in ks]))
        seen, comp = set(), []
        for i in range(len(ks)):
            if i in seen:
                continue
            stack, comp_i = [i], []
            seen.add(i)
            while stack:
                u = stack.pop()
                comp_i.append(ks[u])
                for v in range(len(ks)):
                    if v not in seen and float(F[u] @ F[v]) >= args.group_tau:
                        seen.add(v)
                        stack.append(v)
            if len(comp_i) > 1:
                comp.append(comp_i)
        if comp:
            groups[cam] = comp

    res = dict(cell=name, gal=f"{gal_day}[{'+'.join(gal_splits)}]", q=f"{q_day}.{q_split}")
    for cfg in ("S0", "S1"):
        sub = [r for r in rows if r["cfg"] == cfg]
        res[cfg] = metrics(sub, cooc, same, args)
    if args.propagate:
        res["prop"] = propagate_groups([r for r in rows if r["cfg"] == "S1"], cooc, groups, args)
        res["S1_prop"] = metrics([r for r in rows if r["cfg"] == "S1"], cooc, same, args,
                                 recompute=False)
    res["groups"] = {c: len(v) for c, v in groups.items()}
    res["by_cam"] = {cam: dict(
        n=len([r for r in rows if r["cfg"] == "S1" and r["cam"] == cam]),
        n_face=len([r for r in rows if r["cfg"] == "S1" and r["cam"] == cam and r["has_face"]]),
        n_proto=len([c for c, d in gal.get(cam, {}).get("clusters", {}).items() if d["face"] is not None]),
        K=len(gal.get(cam, {}).get("clusters", {})),
    ) for cam in gal}
    return res, rows, gal


def metrics(rows, cooc, same, args, recompute=True):
    th = args.theta
    n = len(rows)
    if not n:
        return {}
    for r in rows if recompute else []:
        app_ok = r["app_score"] is not None and r["app_score"] >= th
        face_ok = (r["face_id"] is not None and r["face_score"] >= args.tau_face
                   and r["face_margin"] >= args.margin)
        r["app_dec"] = bool(app_ok)
        r["face_dec"] = bool(face_ok)
        r["final"] = r["face_id"] if face_ok else (r["app"] if app_ok else None)
        r["final_app"] = r["app"] if app_ok else None

    n_face = sum(1 for r in rows if r["has_face"])
    n_face_dec = sum(1 for r in rows if r["face_dec"])
    n_app_dec = sum(1 for r in rows if r["app_dec"])
    both = [r for r in rows if r["face_dec"] and r["app_dec"]]
    agree = sum(1 for r in both if r["face_id"] == r["app"])
    disagree = [r for r in both if r["face_id"] != r["app"]]

    # 共现冲突（不循环）
    def fm(key):
        ev = mg = 0
        for cam, pairs in cooc.items():
            idx = {r["key"]: r for r in rows if r["cam"] == cam}
            for u, v in pairs:
                ru, rv = idx.get(u), idx.get(v)
                if not ru or not rv:
                    continue
                a, b = ru[key], rv[key]
                if a is None or b is None:
                    continue
                ev += 1
                mg += int(a == b)
        return ev, mg

    ev_app, mg_app = fm("final_app")
    ev_fin, mg_fin = fm("final")

    # 同人合并（半循环，只读碎片收拢）
    def merge_rate(key):
        ev = mg = 0
        for cam, pairs in same.items():
            idx = {r["key"]: r for r in rows if r["cam"] == cam}
            for u, v in pairs:
                ru, rv = idx.get(u), idx.get(v)
                if not ru or not rv:
                    continue
                a, b = ru[key], rv[key]
                if a is None or b is None:
                    continue
                ev += 1
                mg += int(a == b)
        return ev, mg

    ev_s, mg_s_app = merge_rate("final_app")
    _, mg_s_fin = merge_rate("final")

    n_rescue = sum(1 for r in rows if r["face_dec"] and not r["app_dec"])
    n_override = sum(1 for r in disagree)
    # 外观自己也不确定的子集：这是人脸最该起作用的地方
    low = [r for r in rows if r["app_dec"] and (r["app_margin"] or 0) < 0.05]
    low_face = [r for r in low if r["face_dec"]]
    low_agree = sum(1 for r in low_face if r["face_id"] == r["app"])

    return dict(
        n=n, n_face=n_face, face_rate=n_face / n,
        n_rescue=n_rescue, n_override=n_override,
        n_low=len(low), n_low_face=len(low_face),
        low_face_rate=(len(low_face) / len(low)) if low else float("nan"),
        low_agree_rate=(low_agree / len(low_face)) if low_face else float("nan"),
        n_face_dec=n_face_dec, face_dec_rate=n_face_dec / n,
        n_app_dec=n_app_dec, app_dec_rate=n_app_dec / n,
        n_both=len(both), agree=agree,
        agree_rate=(agree / len(both)) if both else float("nan"),
        n_disagree=len(disagree),
        out_rate=(sum(1 for r in rows if r["final"] is not None) / n),
        fm_app=(mg_app / ev_app) if ev_app else float("nan"), fm_app_n=ev_app,
        fm_fin=(mg_fin / ev_fin) if ev_fin else float("nan"), fm_fin_n=ev_fin,
        same_n=ev_s,
        same_app=(mg_s_app / ev_s) if ev_s else float("nan"),
        same_fin=(mg_s_fin / ev_s) if ev_s else float("nan"),
    )

def propagate_groups(rows, cooc, groups, args):
    """机制二：不靠建库日的脸，只用**当天**的人脸分组做锚点传播。

    对每个（机位内）人脸分组：
      - 成员里若有人被外观以高间隔裁定 → 整组绑到该簇（锚点）
      - 否则整组保持各自的外观裁定
    返回 (新 decisions, 统计)。
    与机制一不同，它不需要建库日有脸，因此绕开「建库日没收进这个人」的乘法项。
    """
    by_key = {}
    for r in rows:
        by_key[(r["cam"], r["key"])] = r
    changed = rescued = overridden = 0
    for cam, comps in groups.items():
      for members in comps:
        ms = [by_key.get((cam, k)) for k in members]
        ms = [m for m in ms if m]
        if len(ms) < 2:
            continue
        anchors = [m for m in ms if m["app_dec"] and (m["app_margin"] or 0) >= args.anchor_margin]
        if not anchors:
            continue
        anchors.sort(key=lambda m: -(m["app_margin"] or 0))
        tgt = anchors[0]["app"]
        for m in ms:
            if m["final"] == tgt:
                continue
            if m["final"] is None:
                rescued += 1
            else:
                overridden += 1
            m["final"] = tgt
            changed += 1
    return dict(group_n=len(groups), changed=changed, rescued=rescued, overridden=overridden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default=os.path.join(ROOT, "tmp", "e21_face"))
    ap.add_argument("--theta", type=float, default=0.60, help="外观接受阈值（裸余弦）")
    ap.add_argument("--tau-face", type=float, default=0.45)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--propagate", action="store_true", help="启用当天人脸分组锚点传播")
    ap.add_argument("--group-tau", type=float, default=0.55, help="当天人脸分组阈值")
    ap.add_argument("--anchor-margin", type=float, default=0.05, help="锚点要求的外观间隔")
    ap.add_argument("--out", default=os.path.join(ROOT, "tmp", "e21_xday_bind.json"))
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    A, B = DAYS["A"], DAYS["B"]
    cells = [("同日 A.tr→A.te", A, ["tr"], A, "te"),
             ("跨天 A.tr→B.te", A, ["tr"], B, "te"),
             ("跨天大库 A.tr+te→B.te", A, ["tr", "te"], B, "te")]

    if args.sweep:
        grid = [(tf, m) for tf in (0.35, 0.40, 0.45, 0.50, 0.55) for m in (0.0, 0.05, 0.10)]
        print(f"{'τ_face':>7} {'margin':>7} | {'跨天 cell':<24} {'脸可裁':>7} {'脸裁定':>7} "
              f"{'一致':>6} {'fm_app':>7} {'fm_fin':>7}")
        for tf, m in grid:
            args.tau_face, args.margin = tf, m
            res, _, _ = run_cell(cells[1][0], cells[1][1], cells[1][2], cells[1][3],
                                 cells[1][4], args.faces, args)
            d = res["S1"]
            print(f"{tf:>7.2f} {m:>7.2f} | {res['cell']:<24} {d['face_rate']:>7.3f} "
                  f"{d['face_dec_rate']:>7.3f} {d['agree_rate']:>6.3f} "
                  f"{d['fm_app']:>7.3f} {d['fm_fin']:>7.3f}")
        return

    out = []
    for name, gd, gs, qd, qs in cells:
        res, rows, gal = run_cell(name, gd, gs, qd, qs, args.faces, args)
        out.append(res)
        print(f"\n=== {res['cell']}  图库 {res['gal']} → 查询 {res['q']}")
        print(f"  {'':>4} {'n':>5} {'有脸':>6} {'脸裁定':>7} {'外观裁定':>8} "
              f"{'双裁定':>6} {'一致':>6} {'输出':>6} {'fm_app':>7} {'fm_fin':>7} "
              f"{'同人_app':>8} {'同人_fin':>8}")
        for cfg in ("S0", "S1"):
            d = res[cfg]
            print(f"  {cfg:>4} {d['n']:>5} {d['face_rate']:>6.3f} {d['face_dec_rate']:>7.3f} "
                  f"{d['app_dec_rate']:>8.3f} {d['n_both']:>6} "
                  f"{(d['agree_rate'] if d['agree_rate']==d['agree_rate'] else -1):>6.3f} "
                  f"{d['out_rate']:>6.3f} {d['fm_app']:>7.3f} {d['fm_fin']:>7.3f} "
                  f"{d['same_app']:>8.3f} {d['same_fin']:>8.3f}")
        if "S1_prop" in res:
            d0, d1 = res["S1"], res["S1_prop"]
            print(f"  +当天人脸分组传播: 分组 {res['prop']['group_n']} 改判 {res['prop']['changed']} "
                  f"(救回 {res['prop']['rescued']} / 覆盖 {res['prop']['overridden']})")
            print(f"     输出 {d0['out_rate']:.3f}→{d1['out_rate']:.3f}  "
                  f"fm {d0['fm_fin']:.3f}→{d1['fm_fin']:.3f}  "
                  f"同人合并 {d0['same_fin']:.3f}→{d1['same_fin']:.3f}")
        for cam, v in res["by_cam"].items():
            print(f"     {cam:<10} K={v['K']:<3} 有脸原型簇={v['n_proto']:<3} "
                  f"查询 {v['n']:>4} 有脸 {v['n_face']:>4}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→", args.out)


if __name__ == "__main__":
    main()
