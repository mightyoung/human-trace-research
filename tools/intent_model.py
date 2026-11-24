#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务二模型：BERT 架构轨迹运动意图预测器（Transformer Encoder + CLS 分类头）。

架构（参考 BERT 范式，规模适配轨迹数据）：
  token 投影(7→d) → [CLS] + 位置编码 → TransformerEncoder(层数/头/FFN 可配)
  → 取 CLS 表征 → LayerNorm → 线性分类头(5类: stop/forward/backward/turn_left/turn_right)

数据：intent_dataset.py 产出的 npz（train/val/test）
评估：test acc / macro-F1 / 每类 P-R-F1 / 混淆矩阵（PNG+json）
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASS_NAMES = ["stop", "forward", "backward", "turn_left", "turn_right"]


class IntentBERT(nn.Module):
    """BERT 范式：投影 + CLS + 位置编码 + TransformerEncoder + 分类头。"""

    def __init__(self, n_feat=7, d_model=256, n_layers=4, n_heads=4,
                 dim_ff=1024, dropout=0.1, n_classes=5, max_len=64):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: [B, L, F]
        h = self.proj(x)                       # [B, L, d]
        b, l, _ = h.shape
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)  # [B, L+1, d]
        h = h + self.pos[:, :l + 1]
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0]))   # CLS


def load_ds(path, mu=None, sd=None):
    z = np.load(path, allow_pickle=True)
    X, y = z["X"].astype(np.float32), z["y"].astype(np.int64)
    if mu is None:
        mu = X.reshape(-1, X.shape[-1]).mean(0)
        sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xn = (X - mu) / sd
    return Xn, y, mu, sd


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        p = model(xb.to(device)).argmax(-1).cpu().numpy()
        preds.append(p)
        gts.append(yb.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return preds, gts


def metrics_from_conf(cm):
    """cm: [C, C]，行=真值，列=预测"""
    out = {}
    n = cm.sum()
    out["acc"] = float(np.trace(cm) / max(1, n))
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / max(1e-9, tp + fp)
        r = tp / max(1e-9, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        f1s.append(f1)
        out[CLASS_NAMES[c]] = {"precision": float(p), "recall": float(r),
                               "f1": float(f1), "support": int(cm[c].sum())}
    out["macro_f1"] = float(np.mean(f1s))
    out["confusion"] = cm.tolist()
    return out


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
    print(f"[env] torch={torch.__version__} device={device}")
    rep = _reporter()

    Xtr, ytr, mu, sd = load_ds(a.train)
    Xva, yva, _, _ = load_ds(a.val, mu, sd)
    Xte, yte, _, _ = load_ds(a.test, mu, sd)
    print(f"[data] train={len(Xtr)} val={len(Xva)} test={len(Xte)}  L={Xtr.shape[1]} F={Xtr.shape[2]}")
    print(f"[dist] train classes={np.bincount(ytr, minlength=5).tolist()}")

    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    te = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    ltr = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True)
    lva = DataLoader(va, batch_size=512)
    lte = DataLoader(te, batch_size=512)

    # 类别权重（inverse freq，温和开方）
    cnt = np.bincount(ytr, minlength=5).astype(np.float64)
    w = np.sqrt(cnt.sum() / (cnt + 1e-9))
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[weights] {np.round(w, 3).tolist()}")

    model = IntentBERT(n_feat=Xtr.shape[2], d_model=a.d_model, n_layers=a.layers,
                       n_heads=a.heads, dim_ff=a.ff, dropout=a.dropout,
                       max_len=Xtr.shape[1]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_par/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss(weight=wt)

    best_f1, best_ep, patience, bad = -1.0, -1, 8, 0
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in ltr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        pv, gv = evaluate(model, lva, device)
        cmv = np.zeros((5, 5), dtype=np.int64)
        for t, p in zip(gv, pv):
            cmv[t, p] += 1
        mv = metrics_from_conf(cmv)
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
                                "max_len": int(Xtr.shape[1])},
                        "mu": mu, "sd_norm": sd},
                       os.path.join(a.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= patience:
                print(f"[early] stop at ep {ep} (best ep {best_ep})")
                break

    if rep:
        rep.close()

    # ===== 测试集评估（best checkpoint）=====
    ck = torch.load(os.path.join(a.out_dir, "best.pt"), map_location=device, weights_only=False)
    model = IntentBERT(n_feat=ck["cfg"]["n_feat"], d_model=ck["cfg"]["d_model"],
                       n_layers=ck["cfg"]["n_layers"], n_heads=ck["cfg"]["n_heads"],
                       dim_ff=ck["cfg"]["dim_ff"], dropout=ck["cfg"]["dropout"],
                       max_len=ck["cfg"]["max_len"]).to(device)
    model.load_state_dict(ck["sd"])
    pt, gt = evaluate(model, lte, device)
    cmt = np.zeros((5, 5), dtype=np.int64)
    for t, p in zip(gt, pt):
        cmt[t, p] += 1
    mtest = metrics_from_conf(cmt)
    print("\n===== TEST =====")
    print(f"acc={mtest['acc']:.4f} macro_f1={mtest['macro_f1']:.4f}")
    for c, name in enumerate(CLASS_NAMES):
        d = mtest[name]
        print(f"  {name:10s} P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} n={d['support']}")
    print("confusion (row=true, col=pred):")
    print(cmt)

    # 混淆矩阵图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        cmn = cmt / np.maximum(1, cmt.sum(1, keepdims=True))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5), CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(5), CLASS_NAMES)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xlabel("pred"); ax.set_ylabel("true")
        ax.set_title(f"Intent model (test acc={mtest['acc']:.3f}, macroF1={mtest['macro_f1']:.3f})")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, "confusion_test.png"), dpi=150)
        print(f"[fig] {os.path.join(a.out_dir, 'confusion_test.png')}")
    except Exception as e:
        print("[fig] skip:", e)

    json.dump({"test": mtest, "val_best": {"ep": best_ep, "macro_f1": best_f1},
               "hist": hist, "config": vars(a)},
              open(os.path.join(a.out_dir, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[done] metrics -> {os.path.join(a.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
