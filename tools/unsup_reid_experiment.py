# -*- coding: utf-8 -*-
"""无监督 ReID 增强研究实验（v3：严格 训练/验证/测试 三集）。

数据纪律（splits_v3.json，实体粒度）：
  - test   ：全部人工判读对涉及的实体（跨机位 30 对 + labels_v2 同机位/跨机位）→ 只读报告
  - val    ：多片段实体抽样（305/316/323/343），用于 ① 门限校准 ② 度量超参选择
  - train  ：其余实体 → 伪标签（实体链接正对 + 硬负对）训练度量
  test 与 val 实体均不得进入训练。

实验1 无监督门限聚类一致性：
  - 在 test 与 val 上分别报告分布；用 val 校准操作门限（t_strict/t_mid），再到 test 评估
实验2 轨迹图自训练（度量学习）：
  - 训练对仅来自 train 实体；低秩维度在 val 上择优；test 报告前后 AUC

用法:
  python tools/unsup_reid_experiment.py --model d3b
  python tools/unsup_reid_experiment.py --model cradio
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CAMS = ["318", "320", "305", "316", "323", "343"]
EMB = {
    "d3b": lambda cam: os.path.join(ROOT, "output_crosscam", "analysis", "relay",
                                    "remote_emb", f"reid_emb_{cam}_d3b.npz"),
    "cradio": lambda cam: os.path.join(ROOT, "output_crosscam", "analysis",
                                       "cradio_scale_v3", f"{cam}_emb_scale.npz"),
    "d3bridge": lambda cam: os.path.join(ROOT, "output_crosscam", "analysis",
                                         "dinov3_scale_v3", f"{cam}_emb_d3b.npz"),
}
SPLITS_P = os.path.join(ROOT, "output_crosscam", "analysis", "splits_v3.json")


def l2m(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def auc(pv, nv):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pv for n in nv]))


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_all(model):
    per_cam = {}
    for cam in CAMS:
        p = EMB[model](cam)
        if not os.path.exists(p):
            print(f"[load] 缺 {p}，跳过该模型")
            return None
        d = np.load(p)
        per_cam[cam] = {key_to_tid(k): d[k] for k in d.files}
    return per_cam


def load_splits():
    if not os.path.exists(SPLITS_P):
        print("[splits] 未找到 splits_v3.json（将退化为 test-only 模式）")
        return None
    return json.load(open(SPLITS_P, encoding="utf-8"))


def load_labels():
    lv = json.load(open(os.path.join(ROOT, "output_crosscam", "analysis",
                                     "stitch", "labels_v2.json"), encoding="utf-8"))
    pairs_pos, pairs_neg = [], []
    for p in lv["same_camera"]["positives"]:
        pairs_pos.append((str(p["cam"]), int(p["a"]), str(p["cam"]), int(p["b"]), "same"))
    for p in lv["same_camera"]["negatives"]:
        pairs_neg.append((str(p["cam"]), int(p["a"]), str(p["cam"]), int(p["b"]), "same"))
    for a, b in lv["cross_camera"]["positives"]:
        pairs_pos.append(("318", a, "320", b, "cross"))
    for a, b in lv["cross_camera"]["negatives"]:
        pairs_neg.append(("318", a, "320", b, "cross"))
    return pairs_pos, pairs_neg


def sim(per_cam, c1, t1, c2, t2):
    if t1 not in per_cam.get(c1, {}) or t2 not in per_cam.get(c2, {}):
        return None
    a = per_cam[c1][t1]
    b = per_cam[c2][t2]
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(a @ b)


def exp1_threshold(per_cam, pos_pairs, neg_pairs, tag, splits=None):
    print(f"\n{'='*66}\n[实验1] 无监督门限聚类一致性 —— {tag}\n{'='*66}")
    pv = [sim(per_cam, c1, t1, c2, t2) for c1, t1, c2, t2, _ in pos_pairs]
    nv = [sim(per_cam, c1, t1, c2, t2) for c1, t1, c2, t2, _ in neg_pairs]
    pv = [v for v in pv if v is not None]
    nv = [v for v in nv if v is not None]
    a = auc(pv, nv)
    t_oracle = (min(pv) + max(nv)) / 2
    print(f"[test] 正对 n={len(pv)}: min={min(pv):.3f} mean={np.mean(pv):.3f} max={max(pv):.3f}")
    print(f"[test] 负对 n={len(nv)}: min={min(nv):.3f} mean={np.mean(nv):.3f} max={max(nv):.3f}")
    print(f"[test] AUC={a:.4f} | t_oracle(事后参考，禁止用于选择)={t_oracle:.3f}")

    t_op = None
    val_info = {}
    if splits is not None:
        vp = [sim(per_cam, c, a_, c, b_) for c, a_, b_ in splits["val"]["pos_pairs"]]
        vn = [sim(per_cam, c, a_, c, b_) for c, a_, b_ in splits["val"]["neg_pairs"]]
        vp = [v for v in vp if v is not None]
        vn = [v for v in vn if v is not None]
        vauc = auc(vp, vn)
        p5p = float(np.percentile(vp, 5))
        p95n = float(np.percentile(vn, 95))
        p99n = float(np.percentile(vn, 99))
        # F1 最优门限（val 网格搜索；负对含少量噪声，用 F1/分位数而非 min/max）
        grid = np.linspace(min(vp + vn), max(vp + vn), 400)
        best_f1, t_f1 = -1.0, float(np.median(vp))
        for t in grid:
            tp = float(np.mean([v >= t for v in vp]))
            fp = float(np.mean([v >= t for v in vn]))
            f1 = 2 * tp * (1 - fp) / (tp + (1 - fp) + 1e-9)
            if f1 > best_f1:
                best_f1, t_f1 = f1, float(t)
        print(f"[val ] 正对 n={len(vp)}: min={min(vp):.3f} p5={p5p:.3f} mean={np.mean(vp):.3f}")
        print(f"[val ] 负对 n={len(vn)}: p95={p95n:.3f} p99={p99n:.3f} max={max(vn):.3f}")
        print(f"[val ] AUC={vauc:.4f} | t_f1={t_f1:.3f} (F1={best_f1:.3f}) | "
              f"t_p5pos={p5p:.3f} | t_p99neg={p99n:.3f}")
        for label, tv in [("F1", t_f1), ("p5pos", p5p), ("p99neg", p99n)]:
            pr = np.mean([v >= tv for v in pv]) * 100
            nr = np.mean([v >= tv for v in nv]) * 100
            print(f"[test@val门限 {label} t={tv:.3f}] 正合并率 {pr:.0f}% | 负合并率 {nr:.0f}%")
        t_op = t_f1
        val_info = {"val_auc": round(vauc, 4), "t_f1": round(t_f1, 4),
                    "t_p5pos": round(p5p, 4), "t_p99neg": round(p99n, 4),
                    "val_pos_n": len(vp), "val_neg_n": len(vn)}

    print(f"门限扫描（test，仅供诊断）: t | 正合并率 | 负合并率")
    for t in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94]:
        pr = np.mean([v >= t for v in pv])
        nr = np.mean([v >= t for v in nv])
        print(f"  {t:.2f} | {pr*100:5.1f}% | {nr*100:5.1f}%")
    out = {"tag": tag, "auc_test": round(a, 4), "t_oracle": round(t_oracle, 4),
           "t_op_val": round(t_op, 4) if t_op is not None else None,
           "pos": [round(v, 4) for v in pv], "neg": [round(v, 4) for v in nv]}
    out.update(val_info)
    return out


def build_entity_map():
    """{cam: {tid -> entity_id}} 与实体成员表（实体图 = 规范目录 stitch_clean2）"""
    from det_view import net_tracks_path
    ent_of = {}
    members = {}
    for cam in CAMS:
        p = net_tracks_path(cam)
        d = json.load(open(p))
        for e in d["entities"]:
            eid = f"{cam}:{e['eid']}"
            members[eid] = [int(t) for t in e["tracks"]]
            for t in e["tracks"]:
                ent_of[(cam, int(t))] = eid
    return ent_of, members


def exp2_selftrain(per_cam, pos_pairs, neg_pairs, model, tag, splits=None):
    print(f"\n{'='*66}\n[实验2] 轨迹图自训练度量学习 —— {tag}（严格三集）\n{'='*66}")
    import torch
    import torch.nn as nn

    ent_of, members = build_entity_map()
    # 留出：测试对实体 ∪ val 实体（均不得进入伪标签训练）
    hold_eids = set()
    eval_keys = set()
    for c1, t1, c2, t2, _ in pos_pairs + neg_pairs:
        eval_keys.add((c1, t1))
        eval_keys.add((c2, t2))
    for (cam, tid) in eval_keys:
        if (cam, tid) in ent_of:
            hold_eids.add(ent_of[(cam, tid)])
    val_eids = set()
    if splits is not None:
        for cam, eids in splits["val"]["entities"].items():
            for e in eids:
                val_eids.add(f"{cam}:{e}")
    train_exclude = hold_eids | val_eids
    print(f"[挖掘] 测试实体 {len(hold_eids)}（{len(eval_keys)} 轨）+ val 实体 {len(val_eids)} → 留出")

    def have(cam, tid):
        return cam in per_cam and tid in per_cam[cam]

    # 正对：train 实体内部片段对
    pos_pairs_m = []
    for eid, tids in members.items():
        if eid in train_exclude or len(tids) < 2:
            continue
        cam = eid.split(":")[0]
        tids = [t for t in tids if have(cam, t)]
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                pos_pairs_m.append((cam, tids[i], cam, tids[j]))
    # 负对：train 池内跨实体硬负 + 随机负
    rng = np.random.RandomState(0)
    all_tr = [(cam, tid) for (cam, tid) in ent_of
              if ent_of[(cam, tid)] not in train_exclude and have(cam, tid)]
    keys = sorted(set(all_tr))
    E = l2m(np.stack([per_cam[c][t] for c, t in keys]).astype(np.float32))
    S = E @ E.T
    np.fill_diagonal(S, -1)
    neg_pairs_m = []
    seen = set()
    for i, (c1, t1) in enumerate(keys):
        e1 = ent_of[(c1, t1)]
        order = np.argsort(-S[i])
        cnt = 0
        for j in order:
            c2, t2 = keys[j]
            if ent_of[(c2, t2)] == e1:
                continue
            pair = tuple(sorted([(c1, t1), (c2, t2)]))
            if pair in seen:
                continue
            seen.add(pair)
            neg_pairs_m.append((c1, t1, c2, t2))
            cnt += 1
            if cnt >= 6:
                break
    for _ in range(len(pos_pairs_m) * 2):
        i, j = rng.randint(0, len(keys)), rng.randint(0, len(keys))
        c1, t1 = keys[i]
        c2, t2 = keys[j]
        if ent_of[(c1, t1)] == ent_of[(c2, t2)]:
            continue
        neg_pairs_m.append((c1, t1, c2, t2))
    print(f"[挖掘] 训练对: 正 {len(pos_pairs_m)} / 负 {len(neg_pairs_m)}")

    def to_tensor(pairs):
        A = np.stack([per_cam[c1][t1] for c1, t1, _, _ in pairs]).astype(np.float32)
        B = np.stack([per_cam[c2][t2] for _, _, c2, t2 in pairs]).astype(np.float32)
        return torch.from_numpy(l2m(A)), torch.from_numpy(l2m(B))

    Atr, Btr = to_tensor(pos_pairs_m)
    Anr, Bnr = to_tensor(neg_pairs_m)
    alpha = 12.0
    n_steps = 400
    import time as _t

    def train_metric(D_OUT):
        torch.manual_seed(0)
        M_ = nn.Linear(768, D_OUT, bias=False)
        nn.init.orthogonal_(M_.weight)
        opt = torch.optim.Adam(M_.parameters(), lr=1e-3, weight_decay=2e-4)

        def f_(x):
            return torch.nn.functional.normalize(M_(x), dim=-1)

        t0 = _t.time()
        loss = None
        for step in range(n_steps):
            sa = (f_(Atr) * f_(Btr)).sum(dim=-1)
            sn = (f_(Anr) * f_(Bnr)).sum(dim=-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                alpha * (sa - 0.5), torch.ones_like(sa)) + \
                torch.nn.functional.binary_cross_entropy_with_logits(
                    alpha * (sn - 0.5), torch.zeros_like(sn))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  [train] dim={D_OUT} {n_steps}步 {_t.time()-t0:.1f}s "
              f"loss={loss.item():.4f}")
        return M_

    _emb_cache = {}

    def metric_emb(M_, cam, tid):
        key = (id(M_), cam, tid)
        if key not in _emb_cache:
            with torch.no_grad():
                e = torch.nn.functional.normalize(
                    M_(torch.from_numpy(np.asarray(per_cam[cam][tid])[None].astype(np.float32))),
                    dim=-1)
            _emb_cache[key] = e.numpy()[0]
        return _emb_cache[key]

    def sims_of(M_, pairs_c):
        out = []
        for c1, t1, c2, t2 in pairs_c:
            if not (have(c1, t1) and have(c2, t2)):
                continue
            out.append(float(metric_emb(M_, c1, t1) @ metric_emb(M_, c2, t2)))
        return out

    test_pos_c = [(c1, t1, c2, t2) for c1, t1, c2, t2, _ in pos_pairs]
    test_neg_c = [(c1, t1, c2, t2) for c1, t1, c2, t2, _ in neg_pairs]
    val_pos_c, val_neg_c = [], []
    if splits is not None:
        val_pos_c = [(c, a_, c, b_) for c, a_, b_ in splits["val"]["pos_pairs"]]
        val_neg_c = [(c, a_, c, b_) for c, a_, b_ in splits["val"]["neg_pairs"]]

    # 基线（原始嵌入，test）
    pv0 = [sim(per_cam, c1, t1, c2, t2) for c1, t1, c2, t2, _ in pos_pairs]
    nv0 = [sim(per_cam, c1, t1, c2, t2) for c1, t1, c2, t2, _ in neg_pairs]
    pv0 = [v for v in pv0 if v is not None]
    nv0 = [v for v in nv0 if v is not None]
    a0 = auc(pv0, nv0)
    print(f"[评估] 训练前 test AUC = {a0:.4f} (margin {min(pv0)-max(nv0):+.4f})")

    # ---- 超参选择（val）----
    best = None
    for D in [64, 128, 256]:
        M_ = train_metric(D)
        if val_pos_c:
            vp = sims_of(M_, val_pos_c)
            vn = sims_of(M_, val_neg_c)
            va = auc(vp, vn) if vp and vn else 0.0
            print(f"  [val ] dim={D} AUC={va:.4f} (pos {len(vp)}/neg {len(vn)})")
        else:
            va = 0.0
        if best is None or va > best[0]:
            best = (va, D, M_)
    D_sel = best[1]
    M = best[2]
    print(f"[选择] dim={D_sel}（由验证集择优；候选 64/128/256）")

    # ---- test 报告（只读）----
    pv = sims_of(M, test_pos_c)
    nv = sims_of(M, test_neg_c)
    a1 = auc(pv, nv)
    print(f"[评估] 训练后 test AUC = {a1:.4f} (margin {min(pv)-max(nv):+.4f})"
          f" | ΔAUC = {a1-a0:+.4f}")
    val_auc_final = None
    if val_pos_c:
        vp = sims_of(M, val_pos_c)
        vn = sims_of(M, val_neg_c)
        val_auc_final = auc(vp, vn)
        print(f"[评估] val AUC（同配置） = {val_auc_final:.4f}")
    for dom in ["same", "cross"]:
        tp = [(c1, t1, c2, t2) for c1, t1, c2, t2, d in pos_pairs if d == dom]
        tn = [(c1, t1, c2, t2) for c1, t1, c2, t2, d in neg_pairs if d == dom]
        pp = sims_of(M, tp)
        nn_ = sims_of(M, tn)
        if pp and nn_:
            print(f"  [{dom}] test AUC {auc(pp, nn_):.4f} (pos n={len(pp)} neg n={len(nn_)})")
    return {"tag": tag, "auc_before": round(a0, 4), "auc_after": round(a1, 4),
            "delta": round(a1 - a0, 4), "dim_selected": D_sel,
            "val_auc": round(val_auc_final, 4) if val_auc_final is not None else None,
            "n_pos_train": len(pos_pairs_m), "n_neg_train": len(neg_pairs_m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="d3b",
                    choices=["d3b", "cradio", "d3bridge"])
    ap.add_argument("--skip-exp2", action="store_true")
    args = ap.parse_args()

    per_cam = load_all(args.model)
    if per_cam is None:
        return
    splits = load_splits()
    pos_pairs, neg_pairs = load_labels()
    print(f"[load] {args.model}: {sum(len(v) for v in per_cam.values())} 轨 | "
          f"标注对: 正 {len(pos_pairs)} / 负 {len(neg_pairs)} | "
          f"test 实体已冻结（sha {splits['meta']['frozen_sha1_16'] if splits else 'n/a'}）")

    r1 = exp1_threshold(per_cam, pos_pairs, neg_pairs, args.model, splits)
    r2 = None
    if not args.skip_exp2:
        r2 = exp2_selftrain(per_cam, pos_pairs, neg_pairs, args.model, args.model, splits)
    out = {"model": args.model, "exp1": r1, "exp2": r2}
    out_p = os.path.join(ROOT, "output_crosscam", "analysis", "unsup_reid",
                         f"exp_{args.model}_v3.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[out] -> {out_p}")


if __name__ == "__main__":
    main()
