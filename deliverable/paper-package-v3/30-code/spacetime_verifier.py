"""在 27 条难例上找「跨机位合并验证器」的可行阈值，并做留一交叉验证。

前提：这 27 条都是**已通过 .75 + attr8 且被时空矛盾标出**的对，因此是最难的一批。
在难例上能分开 ≠ 在全量上能用；但反过来，难例上分不开就一定不能用。

两件事：
  ① 单信号阈值扫描：去均值余弦 / 框高比，各自能救回多少真错、误杀多少真对
  ② 三特征逻辑回归的 LOO 预测（防止 n=27 上过拟合），报 LOO AUC 与工作点
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROWS = json.loads(open("tmp/full/spacetime_signal.json", encoding="utf-8").read())


def sweep(feat, higher_is_same=True, name=""):
    rows = [r for r in ROWS if not np.isnan(r[feat])]
    print(f"\n[{name or feat}] 方向：{'同人更高' if higher_is_same else '同人更低'}")
    print(f"  {'阈值':>6s}{'判对(保留)':>11s}{'漏真对(误杀)':>13s}{'挡掉真错':>10s}")
    for t in np.arange(0.05, 0.95, 0.05) if higher_is_same else np.arange(0.30, 1.01, 0.05):
        keep = [(r["label"] == 1) if (r[feat] >= t if higher_is_same else r[feat] <= t) else False
                for r in rows]
        pos = [r for r in rows if r["label"] == 1]
        neg = [r for r in rows if r["label"] == 0]
        kept_same = sum(1 for r in pos if (r[feat] >= t if higher_is_same else r[feat] <= t))
        kept_diff = sum(1 for r in neg if (r[feat] >= t if higher_is_same else r[feat] <= t))
        print(f"  {t:6.2f}{kept_same:7d}/{len(pos):<4d}{len(pos) - kept_same:10d}"
              f"{len(neg) - kept_diff:10d}/{len(neg)}")


sweep("osnet_off", True, "OSNet 去均值余弦")
sweep("eg2_off", True, "EG2 去均值余弦")
sweep("h_ratio", False, "两轨中位框高比（同人更低）")

# ---- LOO 逻辑回归 ----
FEATS = ["osnet_off", "eg2_off", "h_ratio"]
rows = [r for r in ROWS if all(not np.isnan(r[f]) for f in FEATS)]
print(f"\n[LOO 逻辑回归] 可用样本 {len(rows)}，特征 {FEATS}")
X = np.array([[r[f] for f in FEATS] for r in rows], dtype=float)
X[:, 2] = 1.0 - X[:, 2]          # 翻转成「同人更高」
y = np.array([r["label"] for r in rows])
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd
pred = np.zeros(len(y))
coefs = []
for i in range(len(y)):
    m = np.ones(len(y), bool)
    m[i] = False
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xs[m], y[m])
    pred[i] = lr.predict_proba(Xs[i:i + 1])[0, 1]
    coefs.append(lr.coef_[0])
a = roc_auc_score(y, pred)
full = LogisticRegression(C=1.0, max_iter=1000).fit(Xs, y)
print(f"  LOO AUC = {a:.3f}（n={len(y)}，真对 {int(y.sum())} / 真错 {int(len(y) - y.sum())}）")
print(f"  全量拟合系数（标准化后）：{dict(zip([f.replace('_off','') + ('_inv' if f=='h_ratio' else '') for f in FEATS], np.round(full.coef_[0], 3)))}")
print(f"  {'阈值':>6s}{'真对保留':>10s}{'真错挡掉':>10s}")
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    kept_same = int(sum(1 for i in range(len(y)) if y[i] == 1 and pred[i] >= t))
    cut_diff = int(sum(1 for i in range(len(y)) if y[i] == 0 and pred[i] < t))
    print(f"  {t:6.2f}{kept_same:6d}/{int(y.sum()):<4d}{cut_diff:6d}/{int(len(y) - y.sum())}")

print("\n逐条（LOO 概率）")
for r, p in sorted(zip(rows, pred), key=lambda t: t[1]):
    print(f"  {'错' if r['label'] == 0 else '对'}  p={p:.2f}  osnet_off {r['osnet_off']:6.3f}"
          f"  eg2_off {r['eg2_off']:6.3f}  h比 {r['h_ratio']:.2f}  {r['item']} {r['pair']}")
