# -*- coding: utf-8 -*-
"""C-RADIOv4 spatial 特征对「玻璃区域」的可分性实验（掩膜自动化可行性依据）。

背景（2026-10-13 查证）：C-RADIOv4 的 SAM3 只是蒸馏教师（config: sam3 use_summary=False），
公开实现**不暴露分割解码器**（README/auto_map 均为特征提取器：summary + spatial）。
因此"掩膜自动化"若走 C-RADIOv4 路线，只能基于 spatial 特征做区域判别。本实验回答：
  玻璃区 vs 非玻璃区的 patch 特征是否可分？→ 决定"特征自动出掩膜草稿"是否可行。

方法：
  - 帧：316 抽 4 帧（掩膜已经目视验证，作为正样本机位）+ 343 抽 4 帧（草稿机位）
  - 每帧 512 输入 → spatial [32,32,1152]（32×32 patch 网格，网格与全图归一化坐标线性对应）
  - 标签：316 区域 u>=0.60 全高；343 区域 右 u>=0.62 全高 ∪ 左 u<=0.12×v∈[0.05,0.75]
  - 分类器：最近质心（余弦）+ ridge 线性探针（numpy 闭式解，无 sklearn 依赖）
  - 评估：留一帧（LOFO）：训练=其余帧 patch，测试=留出帧 patch → 平衡准确率/玻璃类召回

用法（本地 CPU）：
  python tools/cradio_glass_separability.py
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RADIO_DIR = os.path.join(ROOT, "models", "C-RADIOv4-SO400M")
sys.path.insert(0, RADIO_DIR)

VID = {
    "316": "media-resource/multi-cemra/synced-10min/01_行政走廊通道_316/316_085500-090500.mp4",
    "343": "media-resource/multi-cemra/synced-10min/06_电梯间_343/343_085500-090500.mp4",
}
PICK_FRAMES = {"316": [2000, 5000, 9000, 13500], "343": [2000, 5000, 9000, 14000]}


def glass_labels(u, v, cam):
    """归一化坐标 → 玻璃区标签（布尔数组）。"""
    if cam == "316":
        return u >= 0.60                       # 右侧整面玻璃墙（已验证）
    if cam == "343":
        right = u >= 0.62                      # 右侧疑似电梯玻璃
        left = (u <= 0.12) & (v >= 0.05) & (v <= 0.75)   # 左侧暗面
        return right | left
    raise ValueError(cam)


def extract_frame_feats(m, proc, frame):
    """全帧 → 拉伸 512×512 → RADIO → [32,32,1152]。
    说明：RADIO 要求输入为 min_resolution_step 的整数倍；全帧保持长宽比会
    产生 910 等非整数倍尺寸，故直接拉伸为 512 方图（patch 网格与 u/v 线性对应）。"""
    img = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pix = torch.from_numpy(rgb).permute(2, 0, 1)[None]   # [1,3,512,512] in [0,1]
    with torch.no_grad():
        out = m(pix)
    sp = out[1][0]                              # [1024, 1152]
    g = sp.reshape(32, 32, -1).float().numpy()  # [32, 32, 1152]
    return g


def plot_scatter(X, y, title, path):
    """简单 PCA(2) 散点（手写协方差特征分解）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Xc = X - X.mean(0, keepdims=True)
        C = Xc.T @ Xc / len(Xc)
        w, V = np.linalg.eigh(C.astype(np.float64))
        P = Xc @ V[:, -2:]
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(6.4, 5), dpi=120)
        ax.scatter(P[~y, 0], P[~y, 1], s=6, c="#9aa0a6", label="非玻璃 patch", alpha=.5)
        ax.scatter(P[y, 0], P[y, 1], s=6, c="#e53935", label="玻璃 patch", alpha=.5)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        print(f"  [fig] {path}")
    except Exception as ex:
        print(f"  [fig-skip] {ex}")


def main():
    print("[load] C-RADIOv4 (local CPU) ...", flush=True)
    from transformers import AutoModel, CLIPImageProcessor
    m = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True).eval()
    proc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    print("[load] ok", flush=True)

    US, VS = np.meshgrid((np.arange(32) + 0.5) / 32, (np.arange(32) + 0.5) / 32)
    # 注意：patch (i,j) -> u=j 方向, v=i 方向
    U = ((np.arange(32) + 0.5) / 32)[None, :].repeat(32, 0)   # [i, j] -> u (j)
    V = ((np.arange(32) + 0.5) / 32)[:, None].repeat(32, 1)   # [i, j] -> v (i)

    out = {"model": "C-RADIOv4-SO400M", "frames": {}}
    data = {}   # cam -> list of (feats [1024,1152], labels [1024])
    for cam, picks in PICK_FRAMES.items():
        cap = cv2.VideoCapture(VID[cam])
        feats, labels = [], []
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            assert ok, (cam, f)
            t0 = time.time()
            g = extract_frame_feats(m, proc, frame)
            lab = glass_labels(U, V, cam).reshape(-1)
            feats.append(g.reshape(1024, -1))
            labels.append(lab)
            print(f"  [{cam} f{f}] feats {g.shape} glass {lab.sum()}/1024 "
                  f"({time.time()-t0:.1f}s)", flush=True)
        cap.release()
        data[cam] = (feats, labels)

    # ---- 分类评估（留一帧 LOFO）----
    rng = np.random.default_rng(0)
    for cam in data:
        feats, labels = data[cam]
        n_f = len(feats)
        accs, bal_accs, aucs = [], [], []
        for hold in range(n_f):
            Xtr = np.concatenate([feats[i] for i in range(n_f) if i != hold])
            ytr = np.concatenate([labels[i] for i in range(n_f) if i != hold])
            Xte, yte = feats[hold], labels[hold]
            # 标准化
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
            # 最近质心（余弦）
            cg = Xtr_n[ytr].mean(0); cn = Xtr_n[~ytr].mean(0)
            def cos(a, b):
                return a @ b / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)
            sg, sn = cos(Xte_n, cg), cos(Xte_n, cn)
            score_cent = sg - sn                       # >0 判玻璃
            pred = score_cent > 0
            # ridge 线性探针
            lam = 10.0
            ypm = np.where(ytr, 1.0, -1.0)
            w = np.linalg.solve(Xtr_n.T @ Xtr_n + lam * np.eye(Xtr_n.shape[1]),
                                Xtr_n.T @ ypm)
            score_ridge = Xte_n @ w
            # 指标
            def bal(pred, yte):
                tpr = pred[yte].mean() if yte.sum() else 0
                tnr = (~pred[~yte]).mean() if (~yte).sum() else 0
                return 0.5 * (tpr + tnr)
            def auc(sc, yte):
                p, n = sc[yte], sc[~yte]
                return float(np.mean([(a > b) + .5 * (a == b) for a in p for b in n]))
            accs.append((pred == yte).mean())
            bal_accs.append(bal(pred, yte))
            aucs.append(auc(score_cent, yte))
        say = lambda x: f"{np.mean(x):.3f}±{np.std(x):.3f}"
        print(f"[{cam}] LOFO: acc {say(accs)} | balanced-acc {say(bal_accs)} | "
              f"AUC {say(aucs)}", flush=True)
        out["frames"][cam] = {"n_frames": n_f, "picks": PICK_FRAMES[cam],
                              "acc": [round(float(x), 4) for x in accs],
                              "bal_acc": [round(float(x), 4) for x in bal_accs],
                              "auc": [round(float(x), 4) for x in aucs]}
        # 可视化（全 patch）
        X = np.concatenate(feats); y = np.concatenate(labels)
        plot_scatter(X, y, f"{cam} 玻璃区 vs 非玻璃区（patch 特征 PCA）",
                     os.path.join(ROOT, "output_crosscam", "analysis", "clean_review",
                                  f"featsep_{cam}.png"))

    out_p = os.path.join(ROOT, "output_crosscam", "analysis", "clean_review",
                         "glass_feature_separability.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump(out, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {out_p}")


if __name__ == "__main__":
    main()
