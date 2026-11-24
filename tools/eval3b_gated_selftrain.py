#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EVAL-3b：H1 门控自训练「增强版」——流式多轮 + 遗忘/漂移监控 + margin 门扫描。

承接 eval3_gated_selftrain.py（EVAL-3 首实验：单次 val 更新，效应量小、建议增强）。
三项增强（对其 §3「下一步增强」的直接执行）：
  ① 更新流扩大：val 按时间四等分（跨机位同步），逐轮累积更新（4 轮），
     轮内分类使用**上一轮更新后的选择器**（真·自训练环，检验错误放大）；
  ② 遗忘/漂移监控：每轮记录 retention（跨段保持）、簇原型位移、test 指派 churn；
  ③ margin 门保留率扫描：r70(p30,生产)/r80(p20)/r85(p15)/r90(p10) 四档对照。

臂（全部在同一 test 上评测）：
  A1_frozen            冻结基线（= eval3 A1，复现校验）
  A2_naive / A3_gated  单次全 val 更新（= eval3 A2/A3，复现校验）
  R2_naive / R2_gated  val 四时段逐轮流式更新（本轮特征）
  S_r80 / S_r85 / S_r90 单次 gated 更新在更严 margin 门下的扫描（r70 即 A3）

评测口径与 eval3 完全一致（unknown / ab_same / 跨段保持 / 误接代理 viol + 公共子集）。
只读实验；不写任何生产文件；输出 analysis/eval3/self_train_v2.json。
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector, _norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
SEL = os.path.join(ANA, "selector_1012", "selector_v2.npz")
EMB = os.path.join(ANA, "cradio_1012_emb")
DET = os.path.join(ROOT, "output_crosscam", "1012_final")
OUT = os.path.join(ANA, "eval3")
EVAL3_JSON = os.path.join(OUT, "self_train_arms.json")
SEGS = {
    "train": ["305_train_529", "318_train_525", "320_train_514", "323_train_510"],
    "val": ["305_val_517", "318_val_525", "320_val_519", "323_val_538"],
    "test": ["305_test_544", "318_test_534", "320_test_518", "323_test_527"],
}
T0 = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0, "323": 18.0, "343": 19.0}
FPS = 25.0
EXEMPT = {("318", "320")}
N_ROUNDS = 4          # val 时段四等分
CHUNK = 22500 / N_ROUNDS
MIN_MEMBERS = 3
SIM_ACCEPT = 0.95


# ---------------- 数据加载（与 eval3 一致） ----------------
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


def load_split(split):
    keys, cams, vs, vas, vbs, bestim, midfr = [], [], [], [], [], [], []
    for seg in SEGS[split]:
        ks, va, vb, v = load_emb(seg)
        iv = load_intervals(seg)
        cam = seg[:3]
        # 帧号中点（供轮次切分；从原始帧区间取）
        fr_acc = defaultdict(lambda: [10 ** 9, -1])
        with open(os.path.join(DET, f"{seg}.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for t in d.get("track_ids", []):
                    if t is None or int(t) < 0:
                        continue
                    a = fr_acc[int(t)]
                    a[0] = min(a[0], d["frame"])
                    a[1] = max(a[1], d["frame"])
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
            f0, f1 = fr_acc[tid]
            midfr.append((f0 + f1) / 2.0)
    return dict(keys=keys, cams=cams, V=np.asarray(vs, np.float32),
                Va=np.asarray(vas, np.float32), Vb=np.asarray(vbs, np.float32),
                t=bestim, midf=np.asarray(midfr))


def overlap_pairs(tracks_meta):
    """同簇且时间不可共存的对数（明细键对）。"""
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


def calc_margins(train, sel):
    """复刻 build_selector_v2 的 margin 分位校准（p30 已验证与生产值逐位一致）。"""
    var_p, var_c = sel.protos, sel.proto_cluster
    def sc(V):
        S = V.astype(np.float32) @ var_p.T
        out = np.full((S.shape[0], sel.n_clusters), -np.inf, np.float32)
        for c in range(sel.n_clusters):
            cols = np.where(var_c == c)[0]
            if cols.size:
                out[:, c] = S[:, cols].max(1)
        return out
    sa = np.sort(sc(train["Va"]), 1)
    sb = np.sort(sc(train["Vb"]), 1)
    pos_m = np.minimum(sa[:, -1] - sa[:, -2], sb[:, -1] - sb[:, -2])
    cams = np.array(train["cams"])
    out = {}
    for c in ["305", "318", "320", "323"]:
        pm = pos_m[cams == c]
        out[c] = {f"p{p}": float(np.percentile(pm, p)) for p in (30, 20, 15, 10)}
    return out


def mutex_filter(idx_accept, chunk, acc):
    """接受集内跨机位时间重叠>2s 者整对剔除（318×320 豁免）。"""
    drop = set()
    for ii in range(len(idx_accept)):
        for jj in range(ii + 1, len(idx_accept)):
            i, j = idx_accept[ii], idx_accept[jj]
            if chunk["cams"][i] == chunk["cams"][j]:
                continue
            if tuple(sorted((chunk["cams"][i], chunk["cams"][j]))) in EXEMPT:
                continue
            ov = min(chunk["t"][i][1], chunk["t"][j][1]) - max(chunk["t"][i][0], chunk["t"][j][0])
            if ov > 2.0:
                drop.add(i)
                drop.add(j)
    return [i for i in idx_accept if i not in drop], len(drop)


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = IdentitySelector.load(SEL)
    print(f"[selector] protos={sel.protos.shape[0]} clusters={sel.n_clusters}", flush=True)

    data = {}
    for split in SEGS:
        data[split] = load_split(split)
        print(f"[{split}] tracks={len(data[split]['keys'])}", flush=True)

    margins = calc_margins(data["train"], sel)
    print("[margin] 校准:", {c: {k: round(v, 5) for k, v in d.items()} for c, d in margins.items()},
          flush=True)

    # ---- 固定对集：test↔train 互近邻（sim≥0.90） ----
    Vt, Vr = data["test"]["V"], data["train"]["V"]
    S = Vt @ Vr.T
    best_train = S.argmax(1)
    best_test = S.argmax(0)
    pairs = [(i, int(best_train[i]), float(S[i, int(best_train[i])]))
             for i in range(len(Vt)) if int(best_test[int(best_train[i])]) == i and S[i, int(best_train[i])] >= 0.90]
    print(f"[retention-pairs] n={len(pairs)}", flush=True)

    def evaluate(tag, sel_arm, extra=None):
        ids_t, _ = sel_arm.predict(data["test"]["V"], cams=data["test"]["cams"])
        ids_r, _ = sel_arm.predict(Vr, cams=data["train"]["cams"])
        ids_a, _ = sel_arm.predict(data["test"]["Va"], cams=data["test"]["cams"])
        ids_b, _ = sel_arm.predict(data["test"]["Vb"], cams=data["test"]["cams"])
        n = len(ids_t)
        unknown = float((ids_t == -1).mean())
        ab_both = (ids_a != -1) & (ids_b != -1)
        ab_same = float(((ids_a == ids_b) & ab_both).sum() / max(ab_both.sum(), 1))
        ret_same = sum(1 for i, j, _ in pairs if ids_t[i] != -1 and ids_t[i] == ids_r[j])
        ret_both_unk = sum(1 for i, j, _ in pairs if ids_t[i] == -1 and ids_r[j] == -1)
        meta = [(data["test"]["keys"][i], data["test"]["cams"][i],
                 data["test"]["t"][i][0], data["test"]["t"][i][1], int(ids_t[i])) for i in range(n)]
        viol = overlap_pairs(meta)
        r = dict(tag=tag, unknown=round(unknown, 4), ab_same=round(ab_same, 4),
                 ret_same=ret_same, ret_n=len(pairs), ret_both_unknown=ret_both_unk,
                 ret_same_rate=round(ret_same / max(len(pairs), 1), 4),
                 viol=len(viol), n_test=n, ids=ids_t.tolist())
        if extra:
            r.update(extra)
        return r

    # ---- 复现臂 A1/A2/A3 ----
    arms = {}
    arms["A1_frozen"] = evaluate("A1_frozen", sel)

    def run_oneshot(tag, gated, margin_key="p30"):
        ids_v, sims_v = sel.predict(data["val"]["V"], cams=data["val"]["cams"],
                                    margin="prec" if gated else None,
                                    )
        # margin="prec" 用库自带 p30；扫描档手动指定
        if gated and margin_key != "p30":
            m_arr = np.asarray([margins[c][margin_key] for c in data["val"]["cams"]],
                               np.float32)
            # 用显式 margin 重算（数值路径：predict(margin=数组) 不被支持 → 手算）
            ids_v, sims_v = predict_with_margin(sel, data["val"]["V"], data["val"]["cams"], m_arr)
        members = defaultdict(list)
        for i, (cid, s) in enumerate(zip(ids_v, sims_v)):
            if cid >= 0 and s >= SIM_ACCEPT:
                members[int(cid)].append(i)
        n_dropped = 0
        if gated:
            for c, idxs in list(members.items()):
                kept, nd = mutex_filter(idxs, {"cams": data["val"]["cams"], "t": data["val"]["t"]}, None)
                members[c] = kept
                n_dropped += nd
        stats = dict(n_val=len(ids_v), n_confident=int(sum(len(v) for v in members.values())),
                     clusters_updated=0, mutex_dropped=n_dropped, margins=margin_key)
        P, C = [sel.protos], [sel.proto_cluster]
        for c, idxs in sorted(members.items()):
            if len(idxs) < MIN_MEMBERS:
                continue
            mu = data["val"]["V"][idxs].mean(0)
            mu = mu / (np.linalg.norm(mu) + 1e-9)
            P.append(mu[None, :])
            C.append(np.array([c]))
            stats["clusters_updated"] += 1
        sel_arm = clone_sel(sel, np.concatenate(P, 0), np.concatenate(C, 0))
        return evaluate(tag, sel_arm, extra={"update": stats}), sel_arm

    def predict_with_margin(sel_arm, V, cams, m_arr):
        """selector.predict 的 margin 数组变体（复用其打分逻辑，逐位一致）。"""
        v = _norm(np.atleast_2d(V.astype(np.float32)), axis=1)
        S = v @ sel_arm.protos.T
        sc = sel_arm._cluster_scores(S)
        idx = sc.argmax(1)
        best = sc[np.arange(len(idx)), idx]
        thr = sel_arm._thr_for(cams, False)
        srt = np.sort(sc, axis=1)
        mg = srt[:, -1] - srt[:, -2]
        out = np.where((best >= thr) & (mg >= m_arr), idx, -1)
        return out, best

    r_a2, _ = run_oneshot("A2_naive", gated=False)
    arms["A2_naive"] = r_a2
    r_a3, _ = run_oneshot("A3_gated", gated=True)
    arms["A3_gated"] = r_a3

    # ---- 复现校验（对 eval3 产物） ----
    repro = {}
    if os.path.exists(EVAL3_JSON):
        old = json.load(open(EVAL3_JSON, encoding="utf-8"))
        for tag, key in (("A1_frozen", "A1_frozen"), ("A2_naive", "A2_naive_update"),
                         ("A3_gated", "A3_gated_update")):
            o = old["arms"][key]
            n = arms[tag]
            repro[tag] = dict(
                unknown_old=o["unknown"], unknown_new=n["unknown"], unknown_delta=round(n["unknown"] - o["unknown"], 6),
                ab_old=o["ab_same"], ab_new=n["ab_same"], ab_delta=round(n["ab_same"] - o["ab_same"], 6),
                ret_old=f"{o['ret_same']}/{o['ret_n']}", ret_new=f"{n['ret_same']}/{n['ret_n']}",
                viol_old=o["viol"], viol_new=n["viol"])
        print("[repro] vs eval3:", json.dumps(repro, ensure_ascii=False))

    # ---- 流式多轮（val 四时段） ----
    def run_rounds(tag, gated, base_ids, margin_key="p30"):
        chunk_bounds = [(q * CHUNK, (q + 1) * CHUNK + (100 if q == N_ROUNDS - 1 else 0))
                        for q in range(N_ROUNDS)]
        cur = sel                      # 当前选择器（逐轮更新）
        cur_protos = [sel.protos]
        cur_cluster = [sel.proto_cluster]
        traj = []
        base = np.asarray(base_ids)
        for r in range(N_ROUNDS):
            lo, hi = chunk_bounds[r]
            idx = [i for i, mf in enumerate(data["val"]["midf"]) if lo <= mf < hi]
            if not idx:
                traj.append(dict(round=r + 1, n_chunk=0, skipped=True))
                continue
            Vc = data["val"]["V"][idx]
            ccam = [data["val"]["cams"][i] for i in idx]
            ct = [data["val"]["t"][i] for i in idx]
            if gated:
                if margin_key == "p30":
                    ids_v, sims_v = cur.predict(Vc, cams=ccam, margin="prec")
                else:
                    m_arr = np.asarray([margins[c][margin_key] for c in ccam], np.float32)
                    ids_v, sims_v = predict_with_margin(cur, Vc, ccam, m_arr)
            else:
                ids_v, sims_v = cur.predict(Vc, cams=ccam)
            members = defaultdict(list)
            for k, (cid, s) in enumerate(zip(ids_v, sims_v)):
                if cid >= 0 and s >= SIM_ACCEPT:
                    members[int(cid)].append(k)
            n_drop = 0
            if gated:
                for c, idxs in list(members.items()):
                    kept, nd = mutex_filter(idxs, {"cams": ccam, "t": ct}, None)
                    members[c] = kept
                    n_drop += nd
            n_acc = int(sum(len(v) for v in members.values()))
            updated, drifts = [], []
            for c, lidx in sorted(members.items()):
                if len(lidx) < MIN_MEMBERS:
                    continue
                mu = Vc[lidx].mean(0)
                mu = mu / (np.linalg.norm(mu) + 1e-9)
                cols = np.where(cur.proto_cluster == c)[0]
                centroid = _norm(cur.protos[cols].mean(0, keepdims=True))[0]
                drifts.append(float(np.dot(mu, centroid)))
                cur_protos.append(mu[None, :])
                cur_cluster.append(np.array([c]))
                updated.append(c)
            cur = clone_sel(sel, np.concatenate(cur_protos, 0), np.concatenate(cur_cluster, 0))
            ev = evaluate(f"{tag}_r{r+1}", cur)
            ids = np.asarray(ev["ids"])
            churn = dict(
                switched=int(((base != -1) & (ids != -1) & (base != ids)).sum()),
                gained_known=int(((base == -1) & (ids != -1)).sum()),
                lost_known=int(((base != -1) & (ids == -1)).sum()))
            traj.append(dict(round=r + 1, n_chunk=len(idx), n_confident=n_acc,
                             mutex_dropped=n_drop, clusters_updated=len(updated),
                             drift_cos_med=round(float(np.median(drifts)), 4) if drifts else None,
                             drift_cos_min=round(float(np.min(drifts)), 4) if drifts else None,
                             unknown=ev["unknown"], ab_same=ev["ab_same"],
                             ret_same=ev["ret_same"], ret_n=ev["ret_n"],
                             ret_both_unknown=ev["ret_both_unknown"], viol=ev["viol"],
                             churn=churn))
            print(f"[{tag}] r{r+1}: chunk={len(idx)} conf={n_acc} upd={len(updated)} "
                  f"unk={ev['unknown']} ab={ev['ab_same']} ret={ev['ret_same']}/{ev['ret_n']} "
                  f"viol={ev['viol']} drift_med={traj[-1]['drift_cos_med']} churn={churn}", flush=True)
        final = evaluate(tag, cur)
        return final, traj, cur

    base_ids = arms["A1_frozen"]["ids"]
    r_naive, traj_naive, sel_rn = run_rounds("R2_naive", gated=False, base_ids=base_ids)
    arms["R2_naive"] = r_naive
    r_gated, traj_gated, sel_rg = run_rounds("R2_gated", gated=True, base_ids=base_ids)
    arms["R2_gated"] = r_gated

    # ---- 流式多轮 × margin 扫描（更松的门在流式下是否退化） ----
    rounds_scan = {}
    for mk in ["p20", "p15", "p10"]:
        r_s, traj_s, _ = run_rounds(f"RR_{mk}", gated=True, base_ids=base_ids, margin_key=mk)
        arms[f"RR_{mk}"] = r_s
        rounds_scan[mk] = traj_s

    # ---- margin 扫描（单次 gated） ----
    scan = {}
    for mk in ["p30", "p20", "p15", "p10"]:
        r, _ = run_oneshot(f"S_{mk}", gated=True, margin_key=mk)
        scan[mk] = {k: v for k, v in r.items() if k != "ids"}
        print(f"[scan {mk}] unk={r['unknown']} ab={r['ab_same']} ret={r['ret_same']}/{r['ret_n']} "
              f"viol={r['viol']} upd={r['update']['clusters_updated']} "
              f"conf={r['update']['n_confident']}", flush=True)

    # ---- 公共子集（同覆盖约束） ----
    all_arms = dict(arms)
    all_arms.update({f"S_{mk}": dict(tag=f"S_{mk}", ids=None) for mk in scan})
    common = np.ones(len(data["test"]["keys"]), bool)
    for a in list(arms.values()):
        common &= np.asarray(a["ids"]) != -1
    common_viol = {}
    for tag, a in arms.items():
        ids = np.asarray(a["ids"])
        meta = [(data["test"]["keys"][i], data["test"]["cams"][i],
                 data["test"]["t"][i][0], data["test"]["t"][i][1],
                 int(ids[i]) if common[i] else -1) for i in range(len(ids))]
        common_viol[tag] = len(overlap_pairs(meta))

    out = dict(version="eval3b/1",
               selector=SEL, pairs_n=len(pairs), n_rounds=N_ROUNDS,
               margins_calibrated={c: {k: round(v, 6) for k, v in d.items()} for c, d in margins.items()},
               arms={k: {kk: vv for kk, vv in v.items() if kk != "ids"} for k, v in arms.items()},
               scan=scan,
               rounds=dict(naive=traj_naive, gated=traj_gated),
               rounds_scan=rounds_scan,
               common_subset=dict(n=int(common.sum()), viol=common_viol),
               repro_vs_eval3=repro,
               note="只读实验；流式多轮=val 四等分逐轮累积，轮内分类用上轮更新后选择器")
    with open(os.path.join(OUT, "self_train_v2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n===== 汇总 =====")
    print(f"{'arm':>14} {'unknown':>8} {'ab_same':>8} {'ret':>9} {'ret_unk':>8} {'viol':>5}")
    for tag, a in arms.items():
        print(f"{tag:>14} {a['unknown']:>8} {a['ab_same']:>8} {a['ret_same']:>4}/{a['ret_n']:<4} "
              f"{a['ret_both_unknown']:>8} {a['viol']:>5}")
    print("common subset viol:", common_viol)
    print("saved", os.path.join(OUT, "self_train_v2.json"))


if __name__ == "__main__":
    main()
