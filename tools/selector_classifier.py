#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""选择器变体：聚类伪标签 → 训练轻量分类头（无监督训练），与原型匹配法对比。

输入：selector.npz（含原型）+ 训练嵌入（_a/_b 分片）+ 测试嵌入
流程：
  1) 训练库向量 → 最近原型分配 → 伪标签
  2) 训练 MLP（D→512→K，交叉熵），保存
  3) 对比评估（测试集）：分片一致率 / 碰撞率 / cross-set top1 sim —— 原型法 vs MLP 法
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import IdentitySelector, _norm  # noqa: E402


def tail_digits(s):
    num = ""
    for ch in reversed(s):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_split(emb_dir, segs):
    keys, va, vb = [], [], []
    for seg in segs:
        pa = os.path.join(emb_dir, f"{seg}_emb_a.npz")
        pb = os.path.join(emb_dir, f"{seg}_emb_b.npz")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        za, zb = np.load(pa), np.load(pb)
        for k in za.files:
            if k not in zb.files:
                continue
            keys.append((seg, tail_digits(k)))
            va.append(za[k])
            vb.append(zb[k])
    return keys, np.asarray(va, np.float32), np.asarray(vb, np.float32)


class MLPHead(nn.Module):
    def __init__(self, d_in, d_hid, n_cls):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_hid, n_cls))

    def forward(self, x):
        return self.net(x)


def _reporter():
    """台账上报口；未走 ops_run 包装器或 ops_metric 缺失时返回 None。"""
    try:
        from ops_metric import Reporter
    except Exception:
        return None
    return Reporter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--train-segs", required=True)
    ap.add_argument("--eval-segs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    sel = IdentitySelector.load(a.selector)
    K = len(sel.protos)
    D = sel.protos.shape[1]
    tkeys, Tva, Tvb = load_split(a.emb_dir, [s for s in a.train_segs.split(",") if s])
    ekeys, Eva, Evb = load_split(a.emb_dir, [s for s in a.eval_segs.split(",") if s])
    print(f"[data] train={len(tkeys)} eval={len(ekeys)} K={K} D={D}")

    # 1) 伪标签（用 merged 向量分配的簇）
    Tm = _norm((Tva + Tvb) / 2.0, axis=1)
    pseudo, _ = sel.predict(Tm)
    # 丢弃 unknown（-1）样本用于训练
    mask = pseudo >= 0
    Xtr = torch.from_numpy(Tm[mask].astype(np.float32))
    ytr = torch.from_numpy(pseudo[mask].astype(np.int64))
    print(f"[pseudo] labeled={mask.sum()}/{len(mask)} (unknown {int((~mask).sum())})")

    # 2) 训练 MLP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPHead(D, 512, K).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()
    n = len(Xtr)
    rep = _reporter()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if (ep + 1) % 10 == 0:
            # 训练集拟合度
            model.eval()
            with torch.no_grad():
                acc = (model(Xtr.to(dev)).argmax(1).cpu() == ytr).float().mean().item()
            print(f"[ep {ep+1}] loss={tot/n:.4f} fit_acc={acc:.4f}")
            if rep:
                rep.metric(ep + 1, loss=tot / max(1, n), fit_acc=acc)
    if rep:
        rep.close()

    model.eval()

    # 3) 对比评估
    def eval_both(Va, Vb, tag):
        # 原型法
        ia, sa = sel.predict(Va)
        ib, sb = sel.predict(Vb)
        same_p = float(np.mean(ia == ib))
        # MLP 法
        with torch.no_grad():
            pa = model(torch.from_numpy(_norm(Va)).to(dev)).argmax(1).cpu().numpy()
            pb = model(torch.from_numpy(_norm(Vb)).to(dev)).argmax(1).cpu().numpy()
        same_m = float(np.mean(pa == pb))
        # 碰撞率（随机对）
        rng = np.random.default_rng(0)
        l = rng.integers(0, len(Va), 20000)
        r = rng.integers(0, len(Va), 20000)
        keep = l != r
        coll_p = float(np.mean(ia[l[keep]] == ib[r[keep]]))
        coll_m = float(np.mean(pa[l[keep]] == pb[r[keep]]))
        print(f"[{tag}] proto: same={same_p:.3f} coll={coll_p:.3f} | "
              f"mlp: same={same_m:.3f} coll={coll_m:.3f}")
        return {"proto": {"same": same_p, "coll": coll_p},
                "mlp": {"same": same_m, "coll": coll_m}}

    res = {"K": K, "n_train": int(mask.sum())}
    res["train_set"] = eval_both(Tva, Tvb, "train")
    res["eval_set"] = eval_both(Eva, Evb, "eval")
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()
