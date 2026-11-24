"""生成合成 intent 数据集（仅用于打通 ops_run → run_metric 链路；不是真实数据）。

不是真实数据：7 维轨迹特征 + 5 类，按类中心加噪，训练/验证/测试分片。
输出与 intent_dataset.py 相同结构：X [N, L, F]、y [N]。
"""
import argparse
import os

import numpy as np

CLASS_NAMES = ["stop", "forward", "backward", "turn_left", "turn_right"]


def make(n, L, F, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, len(CLASS_NAMES), size=n)
    X = rng.normal(0, 0.35, size=(n, L, F)).astype(np.float32)
    for c in range(len(CLASS_NAMES)):
        m = y == c
        if not m.any():
            continue
        # 每类给一条缓慢变化的趋势，使任务可学但不至于太简单
        slope = (c - 2) * 0.05
        base = rng.normal(0, 0.15, size=(m.sum(), 1, F))
        trend = np.cumsum(np.full((m.sum(), L, F), slope), axis=1)
        X[m] += base + trend
    return X, y.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--len", type=int, default=16)
    ap.add_argument("--feat", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, n, seed in (("train", a.n_train, 1), ("val", a.n_val, 2), ("test", a.n_test, 3)):
        X, y = make(n, a.len, a.feat, seed)
        p = os.path.join(a.out, f"{name}.npz")
        np.savez_compressed(p, X=X, y=y)
        print(f"{name}: X={X.shape} y={np.bincount(y, minlength=5).tolist()} -> {p}")


if __name__ == "__main__":
    main()
