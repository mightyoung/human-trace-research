#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器 v3 原型：集合匹配器（SetMatcher）——借鉴 StartLux-Decision 的判定结构，
不引入语言模型。

借鉴的四点（对应 startlux_decide/prompt.py + readout.py + modelcfg.py）：
  1. 选项即 token：候选身份不是分类头的一行权重，而是序列里的 token（子原型向量投影），
     增删身份只改候选列表，不改输出层；
  2. 双向注意力：查询 token 与全部候选 token 同处一个序列，判定时能看到"有哪些选项"，
     而不是逐个做 1v1 余弦；
  3. 显式 none：第 K+1 个选项是学出来的 none 向量（对应 none_probability），
     开放集拒识从"阈值"变成"比较"；
  4. 温度校准 + 覆盖率标定：温度/阈值在 val 上定，test 只用于报数（项目数据纪律）。

对照口径（同 selector_upgrade_study）：same_known / coll_known / coverage。
唯一差别：阈值在 **val** 上按目标覆盖率标定，test 只评估。

用法：
  python tools/selector_setmatch_study.py --epochs 40 --tag v1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from camgroup_study import load_segs, select_best, CAMS, TRAIN_SEGS, JDL  # noqa: E402
from selector_upgrade_study import build_variants, score_matrix, MIN_CLUSTER  # noqa: E402
from train_selector_cluster import cooc_pairs  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "selector_setmatch")
VAL_SEGS = ["305_val_517", "318_val_525", "320_val_519", "323_val_538"]
TEST_SEGS = ["305_test_544", "318_test_534", "320_test_518", "323_test_527"]
MAX_SUB = 4          # 每簇最多 4 个子原型作为 token
N_OPT = 26           # 训练时选项数（与 StartLux 的 26 字母上限同量级）


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


# ---------------------------------------------------------------- 数据与基聚类
def base_clusters(cache):
    """复现 selector_upgrade_study 的基聚类（同一 select_best 代码路径），带缓存。"""
    path = os.path.join(cache, "base_lab.npz")
    TR_keys, TR_Va, TR_Vb, TR_V = load_segs(TRAIN_SEGS)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if len(z["lab"]) == len(TR_keys):
            print(f"[base] cache hit: {path}")
            return (TR_keys, TR_Va, TR_Vb, TR_V,
                    z["lab"], [int(x) for x in z["keep_ids"]], json.loads(str(z["params"])))
    cooc = []
    keys_set = set(TR_keys)
    for s in TRAIN_SEGS:
        for u, v in cooc_pairs(os.path.join(JDL, f"{s}.jsonl"), keys_set, s, 5):
            cooc.append(((s, u), (s, v)))
    _, results = select_best(TR_V, TR_Va, TR_Vb, TR_keys, cooc)
    best = [d for d in results if d["method"] == "mutual-knn"][0]
    lab = best["_lab"]
    ids = sorted(set(lab))
    sizes = np.array([(lab == c).sum() for c in ids])
    keep_ids = [int(c) for c, s in zip(ids, sizes) if s >= MIN_CLUSTER]
    params = {"method": best["method"], "k": best.get("k"), "thr": best.get("thr"),
              "n_clusters": int(best["n_clusters"]), "f1": float(best["f1"]),
              "n_keep": len(keep_ids)}
    print(f"[base] k={params['k']} thr={params['thr']} ncl={params['n_clusters']} "
          f"keep={len(keep_ids)} F1={params['f1']:.4f}")
    os.makedirs(cache, exist_ok=True)
    np.savez(path, lab=lab, keep_ids=np.asarray(keep_ids), params=np.array(json.dumps(params)))
    return TR_keys, TR_Va, TR_Vb, TR_V, lab, keep_ids, params


def cand_tokens(var, keep_ids):
    """[K, MAX_SUB, D] 候选 token + 有效掩码；簇顺序 = keep_ids 顺序。"""
    sub, s2c = var["subprotos"], var["sub2cluster"]
    K, D = len(keep_ids), sub.shape[1]
    tok = np.zeros((K, MAX_SUB, D), np.float32)
    msk = np.zeros((K, MAX_SUB), bool)
    for i, c in enumerate(keep_ids):
        idx = np.where(s2c == c)[0][:MAX_SUB]
        tok[i, :len(idx)] = sub[idx]
        msk[i, :len(idx)] = True
    return tok, msk


# ---------------------------------------------------------------- 模型
class SetMatcher(nn.Module):
    """查询 token + 候选原型 token 同序列双向编码，选项概率点积读出（无分类头）。"""

    def __init__(self, dim=768, d=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(dim, d)
        self.type_emb = nn.Parameter(torch.zeros(2, d))
        nn.init.normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.q_head = nn.Linear(d, d, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(1))       # 读出温度（可学）
        self.none = nn.Parameter(torch.randn(d) * 0.02)     # 第 K+1 个选项
        self.bias = nn.Parameter(torch.zeros(1))

    def options(self, tok):
        """[K,S,D] → [K,S,d]（投影 + 选项类型嵌入；每个 epoch 只需算一次）。"""
        K, S, _ = tok.shape
        return self.proj(tok.reshape(K * S, -1)).reshape(K, S, -1) + self.type_emb[1]

    def query(self, q):
        return self.proj(q) + self.type_emb[0]

    def forward(self, qt, opt, msk):
        """qt:[B,d] opt:[B,N,S,d] msk:[B,N,S] → logits[B,N+1]（末列 none）。"""
        B, N, S, _ = opt.shape
        seq = torch.cat([qt.unsqueeze(1), opt.reshape(B, N * S, -1)], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=qt.device),
                         ~msk.reshape(B, N * S)], 1)
        h = self.enc(seq, src_key_padding_mask=pad)
        hq = self.q_head(h[:, 0])
        hp = h[:, 1:].reshape(B, N, S, -1)
        w = msk.unsqueeze(-1).float()
        hp = (hp * w).sum(2) / w.sum(2).clamp_min(1.0)
        scale = self.log_scale.exp().clamp(0.01, 100.0) * (hq.shape[-1] ** -0.5)
        logits = scale * torch.einsum("bd,bnd->bn", hq, hp) + self.bias
        none = scale * (hq * self.none).sum(-1, keepdim=True)
        return torch.cat([logits, none], 1)


# ---------------------------------------------------------------- 训练
def sample_batch(sel, rows, views, K, rng, p_none):
    """返回 (q[B,D], oi[B,N] 簇索引, y[B])；y=K 表示 none。"""
    B = len(sel)
    q = np.empty((B, views.shape[-1]), np.float32)
    oi = np.zeros((B, N_OPT), np.int64)
    y = np.empty(B, np.int64)
    for j, r in enumerate(sel):
        i, pos = rows[r]
        q[j] = views[i, rng.integers(0, views.shape[1])]
        pool = rng.permutation(K)
        pool = pool[pool != pos]
        if rng.random() < p_none:
            picks = pool[:N_OPT]
            y[j] = N_OPT                     # 训练时选项数 = N_OPT，none 排在末位
        else:
            picks = np.concatenate([[pos], pool[:N_OPT - 1]])
            y[j] = -1
        rng.shuffle(picks)
        if y[j] < 0:
            y[j] = int(np.where(picks == pos)[0][0])
        oi[j] = picks
    return q, oi, y


def train_model(views, lab, keep_ids, slot, tok_t, msk_t, args, hold_pack):
    K = len(keep_ids)
    rows = [(i, slot[c]) for i, c in enumerate(lab) if c in slot]
    print(f"[train] {len(rows)} 条训练轨 → {K} 个候选簇", flush=True)
    model = SetMatcher(d=args.dim, layers=args.layers, heads=args.heads, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    steps = max(1, len(rows) // args.batch)

    tok_ids = None
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps)
    best = {"nll": 1e9, "state": None, "epoch": -1}
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(rows))
        tot, cnt = 0.0, 0
        for b in range(steps):
            sel = perm[b * args.batch:(b + 1) * args.batch]
            if len(sel) < 2:
                continue
            opt_emb = model.options(torch.from_numpy(tok_t))        # 每步重算，避免复用计算图
            q, oi, y = sample_batch(sel, rows, views, K, rng, args.p_none)
            emb = opt_emb[torch.from_numpy(oi)]                    # [B,N,S,d]
            mk = torch.from_numpy(msk_t[oi])                       # [B,N,S]
            out = model(model.query(torch.from_numpy(q)), emb, mk)
            loss = F.cross_entropy(out, torch.from_numpy(y), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            cnt += 1
        msg = f"[ep {ep + 1:02d}] loss={tot / max(cnt, 1):.4f}"
        if hold_pack is not None and (ep + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                oemb = model.options(torch.from_numpy(tok_t))
                lp = model(model.query(hold_pack["q"]),
                           oemb.unsqueeze(0).expand(len(hold_pack["q"]), -1, -1, -1),
                           torch.from_numpy(msk_t).unsqueeze(0).expand(len(hold_pack["q"]), -1, -1))
                nll = float(F.cross_entropy(lp, hold_pack["y"]))
            msg += f" hold_nll={nll:.4f}"
            if nll < best["nll"]:
                best = {"nll": nll, "epoch": ep + 1,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        print(msg, flush=True)
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"[best] epoch={best['epoch']} hold_nll={best['nll']:.4f}")
    return model


def probs(model, tok_t, msk_t, V, K, bs=8):
    """全量候选下的 [N,K+1] 概率（末列 none）。"""
    model.eval()
    out = []
    with torch.no_grad():
        oemb = model.options(torch.from_numpy(tok_t))
        mt = torch.from_numpy(msk_t).unsqueeze(0)
        for s in range(0, len(V), bs):
            n = min(bs, len(V) - s)
            qt = model.query(torch.from_numpy(V[s:s + n]))
            logits = model(qt, oemb.unsqueeze(0).expand(n, -1, -1, -1),
                           mt.expand(n, -1, -1))
            out.append(F.softmax(logits, -1).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- 评估
def eval_at(p_a, p_b, thr, cam, K):
    conf_a, conf_b = p_a[:, :K].max(1), p_b[:, :K].max(1)
    ia = np.where((conf_a >= thr) & (conf_a >= p_a[:, K]), p_a[:, :K].argmax(1), -1)
    ib = np.where((conf_b >= thr) & (conf_b >= p_b[:, K]), p_b[:, :K].argmax(1), -1)
    kn = (ia >= 0) & (ib >= 0)
    rng = np.random.default_rng(7)
    L = rng.integers(0, len(ia), 40000)
    R = rng.integers(0, len(ia), 40000)
    kp = L != R
    il, ir = ia[L[kp]], ib[R[kp]]
    k2 = (il >= 0) & (ir >= 0)
    per = {}
    for c in CAMS:
        m = cam == c
        if m.any():
            knc = kn & m
            per[c] = {"cov": round(float(knc.sum() / max(m.sum(), 1)), 4),
                      "same": round(float((ia[knc] == ib[knc]).mean()), 4) if knc.sum() else None}
    return {"coverage": float(kn.mean()),
            "same_known": float((ia[kn] == ib[kn]).mean()) if kn.sum() else None,
            "coll_known": float((il[k2] == ir[k2]).mean()) if k2.sum() else None,
            "per_cam": per}


def thr_for_coverage(conf, target):
    return float(np.quantile(conf, 1.0 - target))


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def cosine_eval_at(base_a, base_b, thr, cam):
    ka = base_a.max(1) >= thr
    kb = base_b.max(1) >= thr
    kk = ka & kb
    ia = np.where(ka, base_a.argmax(1), -1)
    ib = np.where(kb, base_b.argmax(1), -1)
    rng = np.random.default_rng(7)
    L = rng.integers(0, len(ia), 40000)
    R = rng.integers(0, len(ia), 40000)
    kp = L != R
    il, ir = ia[L[kp]], ib[R[kp]]
    k2 = (il >= 0) & (ir >= 0)
    per = {}
    for c in CAMS:
        m = cam == c
        if m.any():
            knc = kk & m
            per[c] = {"cov": round(float(knc.sum() / max(m.sum(), 1)), 4),
                      "same": round(float((base_a[knc].argmax(1) == base_b[knc].argmax(1)).mean()), 4)
                      if knc.sum() else None}
    return {"coverage": float(kk.mean()),
            "same_known": float((ia[kk] == ib[kk]).mean()) if kk.sum() else None,
            "coll_known": float((il[k2] == ir[k2]).mean()) if k2.sum() else None,
            "per_cam": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-none", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--targets", type=str, default="0.2,0.4,0.6")
    ap.add_argument("--tag", type=str, default="v1")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    TR_keys, TR_Va, TR_Vb, TR_V, lab, keep_ids, base_params = base_clusters(OUT)
    variants, C = build_variants(TR_V, lab, keep_ids)
    var = variants["multi8"]
    tok_t, msk_t = cand_tokens(var, keep_ids)
    K = len(keep_ids)
    slot = {c: i for i, c in enumerate(keep_ids)}
    print(f"[variant] multi8 subprotos={var['subprotos'].shape[0]} | K={K}", flush=True)

    VA_keys, VA_Va, VA_Vb, VA_V = load_segs(VAL_SEGS)
    TE_keys, TE_Va, TE_Vb, TE_V = load_segs(TEST_SEGS)
    base_val_a, base_val_b = score_matrix(VA_Va, var, C), score_matrix(VA_Vb, var, C)
    base_te_a, base_te_b = score_matrix(TE_Va, var, C), score_matrix(TE_Vb, var, C)

    tr_idx = np.where([lab[i] in slot for i in range(len(lab))])[0]
    rng = np.random.default_rng(args.seed + 1)
    hold = rng.choice(len(tr_idx), max(1, len(tr_idx) // 10), replace=False)
    hold_mask = np.zeros(len(tr_idx), bool)
    hold_mask[hold] = True
    fit = tr_idx[~hold_mask]
    views = np.stack([TR_Va, TR_Vb], 1)                    # [N,2,D]
    hold_pack = {"q": torch.from_numpy(TR_V[tr_idx][hold].astype(np.float32)),
                 "y": torch.from_numpy(np.array([slot[lab[tr_idx][i]] for i in hold]))}

    model = train_model(views[fit], lab[fit], keep_ids, slot, tok_t, msk_t, args, hold_pack)

    p_val_a = probs(model, tok_t, msk_t, VA_Va, K)
    p_val_b = probs(model, tok_t, msk_t, VA_Vb, K)
    p_te_a = probs(model, tok_t, msk_t, TE_Va, K)
    p_te_b = probs(model, tok_t, msk_t, TE_Vb, K)

    ca, cb = p_val_a[:, :K].argmax(1), p_val_b[:, :K].argmax(1)
    conf = np.maximum(p_val_a[:, :K].max(1), p_val_b[:, :K].max(1))
    val_ece = ece(conf, (ca == cb).astype(float))
    print(f"[cal] val ECE(proxy)={val_ece:.4f} | P(none) mean={p_val_a[:, K].mean():.3f}")

    te_cam = np.array([k[0].split("_")[0] for k in TE_keys])
    res = {"tag": args.tag, "base": base_params, "args": vars(args),
           "n_train": int(len(fit)), "n_val": int(len(VA_keys)), "n_test": int(len(TE_keys)),
           "val_ece_proxy": val_ece, "p_none_val_mean": float(p_val_a[:, K].mean()),
           "rows": []}
    for t in [float(x) for x in args.targets.split(",")]:
        thr_cos = thr_for_coverage(base_val_a.max(1), t)
        r_cos = cosine_eval_at(base_te_a, base_te_b, thr_cos, te_cam)
        thr_sm = thr_for_coverage(np.maximum(p_val_a[:, :K].max(1), p_val_b[:, :K].max(1)), t)
        r_sm = eval_at(p_te_a, p_te_b, thr_sm, te_cam, K)
        res["rows"].append({"target_cov": t, "cos": r_cos, "setmatch": r_sm,
                            "thr_cos": thr_cos, "thr_sm": thr_sm})
        def _f(x):
            return "n/a" if x is None else f"{x:.3f}"
        print(f"[cov {t:.1f}] cos      cov={r_cos['coverage']:.3f} "
              f"same={_f(r_cos['same_known'])} coll={_f(r_cos['coll_known'])}")
        print(f"[cov {t:.1f}] setmatch cov={r_sm['coverage']:.3f} "
              f"same={_f(r_sm['same_known'])} coll={_f(r_sm['coll_known'])}", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    path = os.path.join(OUT, f"study_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), os.path.join(OUT, f"model_{args.tag}.pt"))
    print(f"[done] {path} ({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
