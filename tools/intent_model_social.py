#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图模型 v2：社交邻域上下文（Social LSTM 借鉴）+ 行为聚类条件化（BP-SGCN 借鉴）。

在 intent_model_cc.py 的 BERT 架构上新增两个正交开关：
  --social    检查帧邻域 K=5 他人的运动状态 → MLP + 可学习 query 注意力 → 上下文 token
  --behavior  行为聚类标签（train 上 K-means 伪标签）→ embedding token

评估：test 全量 + per-camera + per-neighbor-count + per-behavior-cluster 分层，
写 metrics_full.json（结构兼容 cc 汇总口径）。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["stop", "forward", "backward", "turn_left", "turn_right"]


class IntentBERTSocial(nn.Module):
    def __init__(self, n_feat=7, d_model=256, n_layers=4, n_heads=4, dim_ff=1024,
                 dropout=0.1, n_classes=5, max_len=64, n_behavior=6,
                 use_social=False, use_behavior=False, n_nb_feat=6):
        super().__init__()
        self.use_social = use_social
        self.use_behavior = use_behavior
        self.proj = nn.Linear(n_feat, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        n_extra = 1 + (1 if use_social else 0) + (1 if use_behavior else 0)
        self.pos = nn.Parameter(torch.zeros(1, max_len + 2 + n_extra, d_model))
        nn.init.normal_(self.pos, std=0.02)
        if use_behavior:
            self.bc_emb = nn.Embedding(n_behavior + 1, d_model)
            nn.init.normal_(self.bc_emb.weight, std=0.02)
        if use_social:
            self.nb_mlp = nn.Sequential(
                nn.Linear(n_nb_feat, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, d_model))
            self.nb_q = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.nb_q, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, nc=None, nmask=None, bc=None):
        h = self.proj(x)                      # [B, L, d]
        b = x.shape[0]
        toks = [self.cls.expand(b, -1, -1)]
        if self.use_behavior and bc is not None:
            toks.append(self.bc_emb(bc.clamp(min=0)).unsqueeze(1))
        toks.append(h)
        if self.use_social and nc is not None:
            hn = self.nb_mlp(nc)              # [B, K, d]
            att = (self.nb_q.expand(b, -1, -1) @ hn.transpose(1, 2)).squeeze(1)
            att = att / math.sqrt(hn.shape[-1])
            if nmask is not None:
                att = att.masked_fill(nmask < 0.5, -1e4)
            w = att.softmax(-1)               # [B, K]
            ctx = (w.unsqueeze(-1) * hn).sum(1)
            if nmask is not None:
                ctx = ctx * (nmask.sum(1, keepdim=True) > 0).float()
            toks.append(ctx.unsqueeze(1))
        h = torch.cat(toks, dim=1)
        l = h.shape[1]
        h = h + self.pos[:, :l]
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))


def load_ds(path, stats=None):
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    nc = z["Nc"].astype(np.float32) if "Nc" in z.files else np.zeros((len(y), 5, 6), np.float32)
    nm = z["Nmask"].astype(np.float32) if "Nmask" in z.files else np.zeros((len(y), 5), np.float32)
    bc = z["bc"].astype(np.int64) if "bc" in z.files else np.zeros(len(y), np.int64)
    cam = np.array([str(m[1]) for m in z["meta"]]) if "meta" in z.files else np.array(["?"] * len(y))
    if stats is None:
        mu = X.reshape(-1, X.shape[-1]).mean(0)
        sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
        nc_mu = nc.reshape(-1, nc.shape[-1]).mean(0)
        nc_sd = nc.reshape(-1, nc.shape[-1]).std(0) + 1e-6
        stats = {"mu": mu, "sd": sd, "nc_mu": nc_mu, "nc_sd": nc_sd}
    X = (X - stats["mu"]) / stats["sd"]
    nc = (nc - stats["nc_mu"]) / stats["nc_sd"]
    nc = nc * nm[..., None]     # 无效邻居置零
    return X, y, nc, nm, bc, cam, stats


def conf_metrics(y, p, n_classes=5):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, q in zip(y, p):
        cm[t, q] += 1
    out = {"acc": float(np.trace(cm) / max(1, cm.sum())), "n": int(cm.sum())}
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        pr = tp / max(1e-9, tp + fp)
        rc = tp / max(1e-9, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        f1s.append(f1)
        out[CLASS_NAMES[c]] = {"precision": float(pr), "recall": float(rc),
                               "f1": float(f1), "support": int(cm[c].sum())}
    out["macro_f1"] = float(np.mean(f1s))
    out["confusion"] = cm.tolist()
    return out


@torch.no_grad()
def predict_all(model, X, nc, nm, bc, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        ncb = torch.from_numpy(nc[i:i + bs]).to(device)
        nmb = torch.from_numpy(nm[i:i + bs]).to(device)
        bcb = torch.from_numpy(bc[i:i + bs]).to(device)
        out.append(model(xb, ncb, nmb, bcb).argmax(-1).cpu().numpy())
    return np.concatenate(out)


def _reporter():
    """台账上报口；未走 ops_run 包装器或 ops_metric 缺失时返回 None。"""
    try:
        from ops_metric import Reporter
    except Exception:
        return None
    return Reporter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--social", action="store_true")
    ap.add_argument("--behavior", action="store_true")
    ap.add_argument("--n-behavior", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[env] torch={torch.__version__} device={device} social={a.social} behavior={a.behavior}")
    rep = _reporter()

    Xtr, ytr, nctr, nmtr, bctr, ctr, stats = load_ds(a.train)
    Xva, yva, ncva, nmva, bcva, cva, _ = load_ds(a.val, stats)
    Xte, yte, ncte, nmte, bcte, cte, _ = load_ds(a.test, stats)
    print(f"[data] train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
          f"nb_mean_train={nmtr.sum(1).mean():.2f} no_nb={float((nmtr.sum(1)==0).mean()):.2%}")

    bs_eff = min(a.batch, max(8, len(Xtr)))
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr),
                                     torch.from_numpy(nctr), torch.from_numpy(nmtr),
                                     torch.from_numpy(bctr)),
                       batch_size=bs_eff, shuffle=True, drop_last=len(Xtr) >= 2 * bs_eff)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva),
                                     torch.from_numpy(ncva), torch.from_numpy(nmva),
                                     torch.from_numpy(bcva)), batch_size=512)
    cnt = np.bincount(ytr, minlength=5).astype(np.float64)
    w = np.sqrt(cnt.sum() / (cnt + 1e-9))
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[weights] {np.round(w, 3).tolist()}")

    model = IntentBERTSocial(n_feat=Xtr.shape[2], d_model=a.d_model, n_layers=a.layers,
                             n_heads=a.heads, dim_ff=a.ff, dropout=a.dropout,
                             max_len=Xtr.shape[1], n_behavior=a.n_behavior,
                             use_social=a.social, use_behavior=a.behavior).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(weight=wt)

    best_f1, best_ep, patience, bad = -1.0, -1, 8, 0
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb, ncb, nmb, bcb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            ncb, nmb, bcb = ncb.to(device), nmb.to(device), bcb.to(device)
            opt.zero_grad()
            loss = crit(model(xb, ncb, nmb, bcb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for xb, yb, ncb, nmb, bcb in dl_va:
                pv.append(model(xb.to(device), ncb.to(device), nmb.to(device),
                                bcb.to(device)).argmax(-1).cpu().numpy())
        pv = np.concatenate(pv)
        mv = conf_metrics(yva, pv)
        hist.append({"ep": ep, "loss": tot / max(1, n), "val_acc": mv["acc"],
                     "val_macro_f1": mv["macro_f1"]})
        print(f"[ep {ep:02d}] loss={tot/max(1,n):.4f} val_acc={mv['acc']:.4f} "
              f"val_f1={mv['macro_f1']:.4f} ({time.time()-t0:.0f}s)")
        loss_v = tot / max(1, n)
        if rep:
            rep.metric(ep, loss=loss_v, val_acc=mv['acc'], val_f1=mv['macro_f1'])
        if mv["macro_f1"] > best_f1:
            best_f1, best_ep, bad = mv["macro_f1"], ep, 0
            torch.save({"sd": model.state_dict(),
                        "cfg": {"d_model": a.d_model, "n_layers": a.layers,
                                "n_heads": a.heads, "dim_ff": a.ff, "dropout": a.dropout,
                                "n_feat": int(Xtr.shape[2]), "max_len": int(Xtr.shape[1]),
                                "n_behavior": a.n_behavior, "use_social": a.social,
                                "use_behavior": a.behavior},
                        "stats": {k: v.tolist() for k, v in stats.items()}},
                       os.path.join(a.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print(f"[early] stop at ep {ep} (best ep {best_ep})")
                break

    if rep:
        rep.close()

    ck = torch.load(os.path.join(a.out_dir, "best.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = IntentBERTSocial(n_feat=cfg["n_feat"], d_model=cfg["d_model"],
                             n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                             dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
                             max_len=cfg["max_len"], n_behavior=cfg["n_behavior"],
                             use_social=cfg["use_social"],
                             use_behavior=cfg["use_behavior"]).to(device)
    model.load_state_dict(ck["sd"])
    pred = predict_all(model, Xte, ncte, nmte, bcte, device)
    nb_cnt = nmte.sum(1).astype(int)
    rep = {"config": {"social": a.social, "behavior": a.behavior,
                      "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
                      "best_ep": best_ep, "best_val_f1": best_f1, "seed": a.seed},
           "overall": conf_metrics(yte, pred),
           "per_cam": {}, "per_nb": {}, "per_bc": {}}
    for c in ("305", "318", "320", "323"):
        m = cte == c
        if m.any():
            rep["per_cam"][c] = conf_metrics(yte[m], pred[m])
    rep["per_nb"]["has_neighbor"] = conf_metrics(yte[nb_cnt > 0], pred[nb_cnt > 0])
    if (nb_cnt == 0).any():
        rep["per_nb"]["no_neighbor"] = conf_metrics(yte[nb_cnt == 0], pred[nb_cnt == 0])
    for b in range(a.n_behavior):
        m = bcte == b
        if m.sum() >= 10:
            rep["per_bc"][f"bc{b}"] = conf_metrics(yte[m], pred[m])
    print("\n===== TEST (overall) =====")
    print(f"acc={rep['overall']['acc']:.4f} macro_f1={rep['overall']['macro_f1']:.4f}")
    for c in ("305", "318", "320", "323"):
        if c in rep["per_cam"]:
            d = rep["per_cam"][c]
            print(f"  cam {c}: acc={d['acc']:.4f} f1={d['macro_f1']:.4f} n={d['n']}")
    for tag, d in rep["per_nb"].items():
        print(f"  nb-{tag}: acc={d['acc']:.4f} f1={d['macro_f1']:.4f} n={d['n']}")
    json.dump({"rep": rep, "hist": hist, "args": vars(a)},
              open(os.path.join(a.out_dir, "metrics_full.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[done] -> {os.path.join(a.out_dir, 'metrics_full.json')}")


if __name__ == "__main__":
    main()
