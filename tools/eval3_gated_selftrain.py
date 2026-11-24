# -*- coding: utf-8 -*-
"""EVAL-3：门控自训练环（H1 首个可执行实验）——三臂对照。

臂：
  A1 冻结检索（现状基线，selector_v2 不动）
  A2 仅原型库更新（朴素）：val 段被分配且 sim≥0.95 的向量 → 每簇均值原型入库
  A3 伪标签自训练 + 质量门控：val 向量先过三重门——
      (a) sim≥0.95；(b) **per-cam margin 精度门**（margin="prec"，r80 保留规则）；
      (c) **时间互斥过滤**（同簇成员中跨机位时间段重叠>2s 者整对剔除；318×320 豁免）

评测（test 段，标签无关指标；固定对集，跨臂可比）：
  ① 覆盖率：test 已分配率（1 − unknown）
  ② 跨段保持：test↔train 互为最近邻且 sim≥0.90 的对 → 同簇率
  ③ 分片一致：同轨 a/b 半向量 → 同簇率
  ④ 误接代理（无标签）：同簇且**时间不可共存的 test 轨对**计数（跨机位重叠>2s，318×320 豁免）
     → 同时报告"公共子集"（各臂皆已分配的轨）上的对照，实现"同覆盖约束"比较

预注册失败条件（评测协议 §6 精神）：若 A3 在同一覆盖率下误接代理不优于 A2 → H1 不成立。
只读既有产物；不写任何生产文件。输出 analysis/eval3/self_train_arms.json。
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
SEL = os.path.join(ANA, "selector_1012", "selector_v2.npz")
EMB = os.path.join(ANA, "cradio_1012_emb")
DET = os.path.join(ROOT, "output_crosscam", "1012_final")
OUT = os.path.join(ANA, "eval3")
SEGS = {
    "train": ["305_train_529", "318_train_525", "320_train_514", "323_train_510"],
    "val": ["305_val_517", "318_val_525", "320_val_519", "323_val_538"],
    "test": ["305_test_544", "318_test_534", "320_test_518", "323_test_527"],
}
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0
EXEMPT = {("318", "320")}


def load_emb(seg):
    za = np.load(os.path.join(EMB, f"{seg}_emb_a.npz"))
    zb = np.load(os.path.join(EMB, f"{seg}_emb_b.npz"))
    ks = [k for k in za.files if k in zb.files]
    va = np.asarray([za[k] for k in ks], np.float32)
    vb = np.asarray([zb[k] for k in ks], np.float32)
    v = va + vb
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return ks, va, vb, v


def load_intervals(seg):
    p = os.path.join(DET, f"{seg}.jsonl")
    acc = defaultdict(lambda: [10 ** 9, -1])
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            fr = d["frame"]
            for t in d.get("track_ids", []):
                if t is None or int(t) < 0:
                    continue
                a = acc[int(t)]
                a[0] = min(a[0], fr)
                a[1] = max(a[1], fr)
    cam = seg[:3]
    return {t: (T0[cam] + r[0] / FPS, T0[cam] + r[1] / FPS) for t, r in acc.items()}


def overlap_pairs(tracks_meta):
    """tracks_meta: list[(key, cam, t0, t1, cluster)] -> 同簇且不可共存的对数（明细键对）。"""
    viol = []
    n = len(tracks_meta)
    for i in range(n):
        ki, ci, a0, a1, gi = tracks_meta[i]
        if gi < 0:
            continue
        for j in range(i + 1, n):
            kj, cj, b0, b1, gj = tracks_meta[j]
            if gj < 0 or gi != gj or ci == cj:
                continue
            if tuple(sorted((ci, cj))) in EXEMPT:
                continue
            ov = min(a1, b1) - max(a0, b0)
            if ov > 2.0:
                viol.append((ki, kj))
    return viol


def clone_sel(sel, protos, proto_cluster):
    return IdentitySelector(
        protos=protos, thr_unknown=sel.thr_unknown,
        proto_cluster=proto_cluster,
        thr_unknown_per_cam=dict(sel.thr_unknown_per_cam or {}),
        thr_loose=sel.thr_loose, thr_loose_per_cam=dict(sel.thr_loose_per_cam or {}),
        margin_prec=sel.margin_prec, margin_prec_per_cam=dict(sel.margin_prec_per_cam or {}),
        margin_ret=sel.margin_ret, meta=dict(sel.meta))


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = IdentitySelector.load(SEL)
    print(f"[selector] protos={sel.protos.shape[0]} clusters={sel.n_clusters} "
          f"margin_prec_per_cam={sel.margin_prec_per_cam}", flush=True)

    # ---- 载入数据 ----
    data = {}
    for split, segs in SEGS.items():
        keys, cams, vs, vas, vbs, bestim = [], [], [], [], [], []
        for seg in segs:
            ks, va, vb, v = load_emb(seg)
            iv = load_intervals(seg)
            cam = seg[:3]
            for k, a, b, vm in zip(ks, va, vb, v):
                tid = int(k.split("#")[-1])
                if tid not in iv:
                    continue
                keys.append(k)
                cams.append(cam)
                vas.append(a)
                vbs.append(b)
                vs.append(vm)
                bestim.append(iv[tid])
        data[split] = dict(keys=keys, cams=cams, V=np.asarray(vs, np.float32),
                           Va=np.asarray(vas, np.float32), Vb=np.asarray(vbs, np.float32),
                           t=bestim)
        print(f"[{split}] tracks={len(keys)}", flush=True)

    # ---- 固定对集：test↔train 互近邻（sim≥0.90） ----
    Vt, Vr = data["test"]["V"], data["train"]["V"]
    S = Vt @ Vr.T
    best_train = S.argmax(1)
    best_test = S.argmax(0)
    pairs, both_known_arm1 = [], 0
    for i in range(len(Vt)):
        j = int(best_train[i])
        if int(best_test[j]) == i and S[i, j] >= 0.90:
            pairs.append((i, j, float(S[i, j])))
    print(f"[retention-pairs] n={len(pairs)}", flush=True)

    # ---- 评测函数 ----
    def evaluate(tag, sel_arm):
        ids_t, sims_t = sel_arm.predict(data["test"]["V"], cams=data["test"]["cams"])
        ids_r, _ = sel_arm.predict(Vr, cams=data["train"]["cams"])
        ids_a, _ = sel_arm.predict(data["test"]["Va"], cams=data["test"]["cams"])
        ids_b, _ = sel_arm.predict(data["test"]["Vb"], cams=data["test"]["cams"])
        n = len(ids_t)
        unknown = float((ids_t == -1).mean())
        ab_both = (ids_a != -1) & (ids_b != -1)
        ab_same = float(((ids_a == ids_b) & ab_both).sum() / max(ab_both.sum(), 1))
        ab_unknown = float(((ids_a == -1) & (ids_b == -1)).mean())
        ret_same = sum(1 for i, j, _ in pairs if ids_t[i] != -1 and ids_t[i] == ids_r[j])
        ret_both_unk = sum(1 for i, j, _ in pairs if ids_t[i] == -1 and ids_r[j] == -1)
        meta = [(data["test"]["keys"][i], data["test"]["cams"][i],
                 data["test"]["t"][i][0], data["test"]["t"][i][1], int(ids_t[i]))
                for i in range(n)]
        viol = overlap_pairs(meta)
        return dict(tag=tag, unknown=round(unknown, 4),
                    ab_same=round(ab_same, 4), ab_unknown=round(ab_unknown, 4),
                    ret_n=len(pairs), ret_same=ret_same, ret_both_unknown=ret_both_unk,
                    ret_same_rate=round(ret_same / max(len(pairs), 1), 4),
                    viol=len(viol), viol_keys=viol[:20],
                    n_test=n, ids=ids_t.tolist())

    arms = {}
    arms["A1_frozen"] = evaluate("A1_frozen", sel)

    # ---- A2 朴素更新 ----
    ids_v, sims_v = sel.predict(data["val"]["V"], cams=data["val"]["cams"])
    members = defaultdict(list)
    for i, (cid, s) in enumerate(zip(ids_v, sims_v)):
        if cid >= 0 and s >= 0.95:
            members[int(cid)].append(i)
    new_protos, new_cluster = [sel.protos], [sel.proto_cluster]
    a2_stats = dict(n_val=len(ids_v), n_confident=int(sum(len(v) for v in members.values())),
                    clusters_updated=0, min_members=3)
    for c, idxs in members.items():
        if len(idxs) < 3:
            continue
        mu = data["val"]["V"][idxs].mean(0)
        mu = mu / (np.linalg.norm(mu) + 1e-9)
        new_protos.append(mu[None, :])
        new_cluster.append(np.array([c]))
        a2_stats["clusters_updated"] += 1
    P2 = np.concatenate(new_protos, 0)
    C2 = np.concatenate(new_cluster, 0)
    sel2 = clone_sel(sel, P2, C2)
    arms["A2_naive_update"] = evaluate("A2_naive_update", sel2)

    # ---- A3 门控更新 ----
    ids_v3, sims_v3 = sel.predict(data["val"]["V"], cams=data["val"]["cams"], margin="prec")
    gated = defaultdict(list)
    for i, (cid, s) in enumerate(zip(ids_v3, sims_v3)):
        if cid >= 0 and s >= 0.95:
            gated[int(cid)].append(i)
    # (c) 时间互斥过滤
    n_dropped = 0
    val_t = data["val"]["t"]
    val_cam = data["val"]["cams"]
    for c, idxs in list(gated.items()):
        drop = set()
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                i, j = idxs[ii], idxs[jj]
                if val_cam[i] == val_cam[j]:
                    continue
                if tuple(sorted((val_cam[i], val_cam[j]))) in EXEMPT:
                    continue
                ov = min(val_t[i][1], val_t[j][1]) - max(val_t[i][0], val_t[j][0])
                if ov > 2.0:
                    drop.add(i)
                    drop.add(j)
        if drop:
            gated[c] = [i for i in idxs if i not in drop]
            n_dropped += len(drop)
    new_protos, new_cluster = [sel.protos], [sel.proto_cluster]
    a3_stats = dict(n_val=len(ids_v3), n_gated=int(sum(len(v) for v in gated.values())),
                    n_mutex_dropped=n_dropped, clusters_updated=0, min_members=3)
    for c, idxs in gated.items():
        if len(idxs) < 3:
            continue
        mu = data["val"]["V"][idxs].mean(0)
        mu = mu / (np.linalg.norm(mu) + 1e-9)
        new_protos.append(mu[None, :])
        new_cluster.append(np.array([c]))
        a3_stats["clusters_updated"] += 1
    P3 = np.concatenate(new_protos, 0)
    C3 = np.concatenate(new_cluster, 0)
    sel3 = clone_sel(sel, P3, C3)
    arms["A3_gated_update"] = evaluate("A3_gated_update", sel3)

    # ---- 公共子集对照（同覆盖约束） ----
    common = np.ones(len(data["test"]["keys"]), bool)
    for a in arms.values():
        common &= np.asarray(a["ids"]) != -1
    common_viol = {}
    for tag, a in arms.items():
        ids = np.asarray(a["ids"])
        meta = [(data["test"]["keys"][i], data["test"]["cams"][i],
                 data["test"]["t"][i][0], data["test"]["t"][i][1],
                 int(ids[i]) if common[i] else -1) for i in range(len(ids))]
        common_viol[tag] = len(overlap_pairs(meta))
    a2_meta = dict(a2_stats)
    out = dict(version="eval3/1",
               selector=SEL, pairs_n=len(pairs),
               arms={k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in arms.items()},
               common_subset=dict(n=int(common.sum()), viol=common_viol),
               update_stats=dict(A2=a2_meta, A3=a3_stats),
               note="只读实验；A3 预注册失败条件=同覆盖下误接代理不优于 A2")
    with open(os.path.join(OUT, "self_train_arms.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"{'arm':>18} {'unknown':>8} {'ab_same':>8} {'ret_same':>9} {'ret_unk':>8} {'viol':>5}")
    for tag, a in arms.items():
        print(f"{tag:>18} {a['unknown']:>8} {a['ab_same']:>8} {a['ret_same']:>4}/{a['ret_n']:<4} "
              f"{a['ret_both_unknown']:>8} {a['viol']:>5}")
    print("common subset viol:", common_viol, "| n:", int(common.sum()))
    print("update stats:", a2_meta, a3_stats)
    print("saved", os.path.join(OUT, "self_train_arms.json"))


if __name__ == "__main__":
    main()
