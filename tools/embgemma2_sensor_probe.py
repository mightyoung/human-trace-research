#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EmbeddingGemma 2 能不能吃传感器信号？——三条路线的最小实测。

路线 A（音频塔）：把高频振动类信号当 16kHz 波形喂 audio_tower（12 层 Conformer）。
   问：嵌入能否区分不同"工况"？能否被文本查询命中？
路线 B（视觉塔）：把多通道时序画成图，走 vision_tower（16 层 ViT）。
   问：文本能否检索图表（这决定"用一句话查历史时段"是否成立）？
路线 C（慢信号对照）：1Hz 温度曲线升采样到 16kHz 后走音频塔。
   问：它是否退化成"都一样"——若是，慢通道根本不该走音频塔。

全部为合成数据（项目里没有传感器日志），只回答"机制是否成立"，不代表真实场景精度。
产物：output_crosscam/analysis/embgemma2_sensor/probe.json + charts/

用法：
  python tools/embgemma2_sensor_probe.py [--skip-charts] [--n 6]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP = os.path.join(ROOT, "models", "google--embeddinggemma-2", "snapshots", "master")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "embgemma2_sensor")
SR = 16000
QP = "task: search result | query: "   # 模型卡：非对称检索的 query 前缀（文档/音视频不加前缀）


# ---------------------------------------------------------------- 信号合成
def _chirp(t, f0, f1, dur):
    k = (f1 - f0) / dur
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def gen_audio(kind, rng, dur=8.0):
    t = np.arange(int(SR * dur)) / SR
    n = rng.uniform(0.03, 0.06)
    if kind == "motor":                       # 稳态电机：基频 + 谐波
        f = rng.uniform(45, 55)
        x = sum(a * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28))
                for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)))
        x *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.3 * t)
    elif kind == "knock":                     # 轴承类周期性冲击 + 共振
        x = np.zeros_like(t)
        for onset in np.arange(0, dur, 1.0 / rng.uniform(10, 14)):
            i = int(onset * SR)
            L = min(int(0.02 * SR), len(t) - i)
            if L > 0:
                tt = np.arange(L) / SR
                x[i:i + L] += np.exp(-tt / 0.004) * np.sin(2 * np.pi * 3000 * tt)
    elif kind == "ramp":                      # 升速：频率扫掠
        x = _chirp(t, rng.uniform(25, 35), rng.uniform(130, 170), dur)
    else:                                     # 对照：白噪声
        x = np.zeros_like(t)
    x = x + n * rng.standard_normal(len(t)) * (1.0 if kind == "noise" else 0.15)
    return (0.3 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def gen_slow(kind, rng, dur=10.0):
    """1Hz 温度类信号 → 升采样到 16k（零阶保持），模拟"直接喂音频塔"的朴素做法。"""
    n = int(dur)
    if kind == "ramp":
        v = np.linspace(20, 30, n)
    elif kind == "step":
        v = np.where(np.arange(n) < n // 2, 22.0, 28.0)
    else:
        v = np.full(n, 25.0)
    v = v + 0.05 * rng.standard_normal(n)
    z = np.repeat(v, SR)                      # 1Hz → 16kHz
    return (0.3 * z / (np.abs(z).max() + 1e-9)).astype(np.float32)


def gen_chart(kind, rng, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 240
    t = np.arange(n)
    if kind == "flat":
        a = 25 + 0.2 * rng.standard_normal(n)
        b = 0.5 + 0.05 * rng.standard_normal(n)
    elif kind == "ramp":
        a = 22 + np.linspace(0, 8, n) + 0.2 * rng.standard_normal(n)
        b = 0.5 + 0.3 * (a - a.min()) / 9 + 0.05 * rng.standard_normal(n)
    elif kind == "step":
        a = np.where(t < n // 2, 23.0, 29.0) + 0.2 * rng.standard_normal(n)
        b = np.where(t < n // 2, 0.4, 1.2) + 0.05 * rng.standard_normal(n)
    else:                                     # periodic
        a = 25 + 2.0 * np.sin(2 * np.pi * t / 60) + 0.2 * rng.standard_normal(n)
        b = 0.5 + 0.4 * (np.sin(2 * np.pi * t / 60) > 0) + 0.05 * rng.standard_normal(n)
    fig, ax = plt.subplots(2, 1, figsize=(5.12, 3.84), dpi=100, sharex=True)
    ax[0].plot(t, a, lw=1.2)
    ax[0].set_ylabel("temperature (C)")
    ax[1].plot(t, b, lw=1.2, color="tab:orange")
    ax[1].set_ylabel("vibration rms")
    ax[1].set_xlabel("minutes")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 模型
def load():
    from transformers import AutoProcessor, AutoModel
    pr = AutoProcessor.from_pretrained(MP)
    m = AutoModel.from_pretrained(MP, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    return pr, m


def embed(pr, m, text, *, audio=None, images=None):
    kw = {"text": text, "return_tensors": "pt"}
    if audio is not None:
        kw.update(audio=audio, sampling_rate=SR)
    if images is not None:
        kw.update(images=images)
    inp = pr(**kw)
    with torch.no_grad():
        out = m(**inp)
    h = out.last_hidden_state
    mask = inp.get("attention_mask")
    if mask is None:
        v = h.mean(1)
    else:
        w = mask.unsqueeze(-1).float()
        v = (h * w).sum(1) / w.sum(1).clamp_min(1)
    v = torch.nn.functional.normalize(v, dim=-1)
    return v[0].numpy()


def knn_loo(X, y):
    """留一法 1-NN 余弦分类准确率（合成数据上衡量"嵌入是否按类别聚"）。"""
    S = X @ X.T
    np.fill_diagonal(S, -2)
    pred = y[S.argmax(1)]
    return float((pred == y).mean()), pred


def top1_by_text(Q, X, y):
    """每条文本查询 → 24 个样本里 top-1 命中同类比例。"""
    S = Q @ X.T
    idx = S.argmax(1)
    return float((y[idx] == np.arange(len(Q))).mean()), idx.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--skip-charts", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    pr, m = load()
    res = {"model": "google/embeddinggemma-2 (local)", "n_per_class": a.n}

    # ---------- 路线 A：振动类信号走音频塔 ----------
    kinds = ["motor", "knock", "ramp", "noise"]
    X, y, tA = [], [], []
    for ci, k in enumerate(kinds):
        for _ in range(a.n):
            w = gen_audio(k, rng)
            s = time.time()
            X.append(embed(pr, m, [pr.audio_token], audio=[w]))
            tA.append(time.time() - s)
            y.append(ci)
    X, y = np.stack(X), np.array(y)
    acc, pred = knn_loo(X, y)
    qa = ["a machine humming steadily", "periodic knocking from a bearing",
          "a motor spinning up faster and faster", "steady background noise"]
    Q = np.stack([embed(pr, m, [QP + q]) for q in qa])
    acc_q, idx = top1_by_text(Q, X, y)
    res["audio_route"] = {"knn_loo_acc": round(acc, 3), "text_top1_acc": round(acc_q, 3),
                          "mean_ms": round(1000 * float(np.mean(tA)), 0),
                          "pred": pred.tolist(), "text_top1_idx": idx}

    # ---------- 路线 C：慢信号（1Hz 温度）走音频塔 ----------
    slow = {k: embed(pr, m, [pr.audio_token], audio=[gen_slow(k, rng)])
            for k in ("ramp", "step", "flat")}
    def cos(u, v):
        return round(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))), 4)
    res["slow_route"] = {"cos_ramp_flat": cos(slow["ramp"], slow["flat"]),
                         "cos_step_flat": cos(slow["step"], slow["flat"]),
                         "cos_ramp_step": cos(slow["ramp"], slow["step"])}

    # ---------- 路线 B：多通道图表走视觉塔 ----------
    if not a.skip_charts:
        from PIL import Image
        cdir = os.path.join(OUT, "charts")
        os.makedirs(cdir, exist_ok=True)
        ck = ["flat", "ramp", "step", "periodic"]
        XI, yi, tB = [], [], []
        for ci, k in enumerate(ck):
            for j in range(a.n):
                p = gen_chart(k, rng, os.path.join(cdir, f"{k}_{j}.png"))
                s = time.time()
                XI.append(embed(pr, m, ["<|image|>"], images=[Image.open(p).convert("RGB")]))
                tB.append(time.time() - s)
                yi.append(ci)
        XI, yi = np.stack(XI), np.array(yi)
        accI, predI = knn_loo(XI, yi)
        qc = ["temperature is flat and stable", "temperature rising steadily",
              "a sudden step change in temperature", "temperature oscillating periodically"]
        QI = np.stack([embed(pr, m, [QP + q]) for q in qc])
        accQ, idxI = top1_by_text(QI, XI, yi)
        res["chart_route"] = {"knn_loo_acc": round(accI, 3), "text_top1_acc": round(accQ, 3),
                              "mean_ms": round(1000 * float(np.mean(tB)), 0),
                              "pred": predI.tolist(), "text_top1_idx": idxI}

    res["elapsed_s"] = round(time.time() - t0, 1)
    np.savez(os.path.join(OUT, "embeddings.npz"), X=X, y=y,
             **({"XI": XI, "yi": yi} if not a.skip_charts else {}))
    p = os.path.join(OUT, "probe.json")
    json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
