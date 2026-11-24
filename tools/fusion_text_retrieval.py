#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文本检索能力三方对比：原版 Gemma2 / T1 直连 / T3 桥接。

文本塔在所有配置下均未改动；差异只在图像侧嵌入（原版 vision vs DINOv3 融合）。
对 8 个文本 query 检索 35 轨图像嵌入，统计目标命中（top-1 / top-2 / top-5）。

运行（embgemma venv）:
  HF_HUB_OFFLINE=1 python tools/fusion_text_retrieval.py
"""
import os
import json
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "google--embeddinggemma-2", "snapshots", "master")
RC = os.path.join(ROOT, "output_crosscam", "analysis", "reid_compare")
FU = os.path.join(ROOT, "output_crosscam", "analysis", "fusion")

# (query, 目标轨迹 keys)  —— 目标为人工确认的人物
TARGETS = [
    ("a person carrying a backpack", {"a69", "b14"}),
    ("a woman wearing a hat", {"a146", "b28"}),
    ("a person wearing a white shirt", {"a3"}),
    ("a person mopping the floor cleaning", {"a1", "b1"}),
    ("a man wearing glasses and dark t-shirt", {"a14", "b3"}),
    ("a person wearing a navy blue jacket", {"a37", "b11"}),
    ("a person wearing a dark hoodie", {"a139", "b24"}),
    ("a woman with a fur-collar coat", {"a120", "b18", "b23"}),
]

SETS = [
    ("原版 Gemma2 视觉", os.path.join(RC, "embgemma2_cache.npz")),
    ("T1 直连（零训练）", os.path.join(FU, "fused_emb_cache.npz")),
    ("T3 桥接（DINOv3）", os.path.join(FU, "bridge_emb_cache.npz")),
    ("C-RADIOv4 桥接", os.path.join(FU, "cradio_emb_cache.npz")),
]


def l2(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    from sentence_transformers import SentenceTransformer
    print("[load] model ...", flush=True)
    model = SentenceTransformer(MODEL_DIR, model_kwargs={"torch_dtype": "float32"})

    # 编码全部 query
    qvs = {}
    for q, _ in TARGETS:
        qv = np.asarray(model.encode(q, prompt_name="SearchQuery",
                                     normalize_embeddings=True)).reshape(-1)
        qvs[q] = l2(qv)
    print(f"[query] {len(qvs)} 编码完成", flush=True)

    summary = {}
    for name, path in SETS:
        d = np.load(path)
        keys = list(d.files)
        E = np.stack([l2(d[k]) for k in keys])
        hit1 = hit2 = hit5 = 0
        detail = []
        for q, targets in TARGETS:
            sims = E @ qvs[q]
            order = np.argsort(-sims)
            top = [keys[i] for i in order[:5]]
            h1 = top[0] in targets
            h2 = any(k in targets for k in top[:2])
            h5 = any(k in targets for k in top[:5])
            hit1 += h1; hit2 += h2; hit5 += h5
            detail.append({"query": q, "top5": top, "hit1": h1, "hit2": h2, "hit5": h5})
        n = len(TARGETS)
        print()
        print(f"=== {name} ===")
        print(f"命中: top-1 {hit1}/{n} | top-2 {hit2}/{n} | top-5 {hit5}/{n}")
        for d_ in detail:
            mark = "✓" if d_["hit5"] else "✗"
            tt = " ".join(d_["top5"][:3])
            print(f"  {mark} \"{d_['query'][:38]:<38}\" top3: {tt}")
        summary[name] = {"hit1": hit1, "hit2": hit2, "hit5": hit5, "n": n, "detail": detail}

    out = os.path.join(FU, "text_retrieval_compare.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
