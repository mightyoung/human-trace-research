#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""意图模型（摄像头分组 + 聚类身份条件化）—— 面向"分摄像头、结合向量聚类"的验证。

扩展自 intent_model.py，新增两个正交维度：
  A. --cams "305,318"   只用指定摄像头的样本训练/早停（空=全部）→ 分组预测器
  B. --id-mode proto    [CLS] 后插入身份 token（聚类簇 pid 的 embedding；-1→unknown 槽）
                        —— 模拟"选择器输出"作为条件输入（部署口径）

评估：test 全量 + per-camera 拆分 + known/unknown(pid) 分层，写 metrics_full.json。
注意：每个模型使用自己训练子集的 mu/sd；test 评估用同一 mu/sd（跨模型对比时按子集口径）。
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = ["stop", "forward", "backward", "turn_left", "turn_right"]


class IntentBERTCC(nn.Module):
    def __init__(self, n_feat=7, d_model=256, n_layers=4, n_heads=4,
                 dim_ff=1024, dropout=0.1, n_classes=5, max_len=64,
                 n_proto=125, id_mode="none"):
        super().__init__()
        self.id_mode = id_mode
        self.proj = nn.Linear(n_feat, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_len + 2, d_model))
        nn.init.normal_(self.pos, std=0.02)
        if id_mode == "proto":
            self.id_emb = nn.Embedding(n_proto + 1, d_model)   # index 0 = unknown
            nn.init.normal_(self.id_emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, pid=None):
        h = self.proj(x)                      # [B, L, d]
        b = x.shape[0]
        toks = [self.cls.expand(b, -1, -1)]
        if self.id_mode == "proto":
            idx = (pid + 1).clamp_(0, self.id_emb.num_embeddings - 1)
            toks.append(self.id_emb(idx).unsqueeze(1))   # [B, 1, d]
        toks.append(h)
        h = torch.cat(toks, dim=1)
        l = h.shape[1]
        h = h + self.pos[:, :l]
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))


def load_ds(path, mu=None, sd=None):
    z = np.load(path, allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(np.int64)
    pid = z["pid"].astype(np.int64) if "pid" in z.files else np.full(len(y), -1, np.int64)
    cam = np.array([str(c) for c in z["cam"]]) if "cam" in z.files else np.array(["?"] * len(y))
    if mu is None:
        mu = X.reshape(-1, X.shape[-1]).mean(0)
        sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    return (X - mu) / sd, y, pid, cam, mu, sd


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
def predict_all(model, X, pid, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        pb = torch.from_numpy(pid[i:i + bs]).to(device)
        out.append(model(xb, pb).argmax(-1).cpu().numpy())
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
    ap.add_argument("--cams", default="", help="逗号分隔，空=全部")
    ap.add_argument("--id-mode", default="none", choices=["none", "proto"])
    ap.add_argument("--n-proto", type=int, default=125)
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
    cams = [c for c in a.cams.split(",") if c] or None
    print(f"[env] torch={torch.__version__} device={device} task=cams:{cams or 'ALL'} id_mode={a.id_mode}")
    rep = _reporter()

    Xtr, ytr, ptr, ctr, mu, sd = load_ds(a.train)
    Xva, yva, pva, cva, _, _ = load_ds(a.val, mu, sd)
    Xte, yte, pte, cte, _, _ = load_ds(a.test, mu, sd)
    if cams:
        mtr, mva = np.isin(ctr, cams), np.isin(cva, cams)
        Xtr, ytr, ptr = Xtr[mtr], ytr[mtr], ptr[mtr]
        Xva, yva, pva = Xva[mva], yva[mva], pva[mva]
    print(f"[data] train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    print(f"[dist] train classes={np.bincount(ytr, minlength=5).tolist()}")
    print(f"[pid ] train known={float((ptr>=0).mean()):.3f} test known={float((pte>=0).mean()):.3f}")

    bs_eff = min(a.batch, max(8, len(Xtr)))
    dl_tr = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr),
                                     torch.from_numpy(ptr)),
                       batch_size=bs_eff, shuffle=True, drop_last=len(Xtr) >= 2 * bs_eff)
    dl_va = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva),
                                     torch.from_numpy(pva)), batch_size=512)
    cnt = np.bincount(ytr, minlength=5).astype(np.float64)
    w = np.sqrt(cnt.sum() / (cnt + 1e-9))
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[weights] {np.round(w, 3).tolist()}")

    model = IntentBERTCC(n_feat=Xtr.shape[2], d_model=a.d_model, n_layers=a.layers,
                         n_heads=a.heads, dim_ff=a.ff, dropout=a.dropout,
                         max_len=Xtr.shape[1], n_proto=a.n_proto,
                         id_mode=a.id_mode).to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(weight=wt)

    best_f1, best_ep, patience, bad = -1.0, -1, 8, 0
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb, pb in dl_tr:
            xb, yb, pb = xb.to(device), yb.to(device), pb.to(device)
            opt.zero_grad()
            loss = crit(model(xb, pb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for xb, yb, pb in dl_va:
                pv.append(model(xb.to(device), pb.to(device)).argmax(-1).cpu().numpy())
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
                                "n_heads": a.heads, "dim_ff": a.ff,
                                "dropout": a.dropout, "n_feat": int(Xtr.shape[2]),
                                "max_len": int(Xtr.shape[1]), "n_proto": a.n_proto,
                                "id_mode": a.id_mode},
                        "mu": mu, "sd_norm": sd},
                       os.path.join(a.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print(f"[early] stop at ep {ep} (best ep {best_ep})")
                break

    # ===== 评估（best checkpoint；test 全量拆分） =====
    if rep:
        rep.close()

    ck = torch.load(os.path.join(a.out_dir, "best.pt"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = IntentBERTCC(n_feat=cfg["n_feat"], d_model=cfg["d_model"],
                         n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                         dim_ff=cfg["dim_ff"], dropout=cfg["dropout"],
                         max_len=cfg["max_len"], n_proto=cfg["n_proto"],
                         id_mode=cfg["id_mode"]).to(device)
    model.load_state_dict(ck["sd"])
    pred = predict_all(model, Xte, pte, device)
    rep = {"config": {"cams": cams or "ALL", "id_mode": a.id_mode,
                      "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
                      "best_ep": best_ep, "best_val_f1": best_f1},
           "overall": conf_metrics(yte, pred),
           "per_cam": {}, "per_known": {}}
    for c in ("305", "318", "320", "323"):
        m = cte == c
        if m.any():
            rep["per_cam"][c] = conf_metrics(yte[m], pred[m])
    for tag, m in (("known", pte >= 0), ("unknown", pte < 0)):
        if m.any():
            rep["per_known"][tag] = conf_metrics(yte[m], pred[m])
    print("\n===== TEST (overall) =====")
    print(f"acc={rep['overall']['acc']:.4f} macro_f1={rep['overall']['macro_f1']:.4f}")
    for c in ("305", "318", "320", "323"):
        if c in rep["per_cam"]:
            d = rep["per_cam"][c]
            print(f"  cam {c}: acc={d['acc']:.4f} f1={d['macro_f1']:.4f} n={d['n']}")
    for tag in ("known", "unknown"):
        if tag in rep["per_known"]:
            d = rep["per_known"][tag]
            print(f"  pid-{tag}: acc={d['acc']:.4f} f1={d['macro_f1']:.4f} n={d['n']}")
    json.dump({"rep": rep, "hist": hist, "args": vars(a)},
              open(os.path.join(a.out_dir, "metrics_full.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[done] -> {os.path.join(a.out_dir, 'metrics_full.json')}")


if __name__ == "__main__":
    main()
