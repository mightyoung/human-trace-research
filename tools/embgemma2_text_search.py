#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmbeddingGemma 2 文本→图像检索 demo（"以文搜人"）

用文本 query 在 35 条轨迹的裁剪嵌入（crops_s8 子集，GPU 版嵌入）中检索。
展示多模态统一空间能力：文本与图像在同一 768d 空间可直接比对。

运行（embgemma venv）:
  HF_HUB_OFFLINE=1 C:/Users/might/.workbuddy/binaries/python/envs/embgemma/Scripts/python.exe tools/embgemma2_text_search.py
"""
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "google--embeddinggemma-2", "snapshots", "master")
EMB = os.path.join(ROOT, "output_crosscam", "analysis", "reid_compare", "embgemma2_emb_gpu.npz")
OUT = os.path.join(ROOT, "output_crosscam", "analysis", "reid_compare", "embgemma2_text_search.json")

# 检索 query（针对 crops 子集里的已知人物特征）
QUERIES = [
    ("a person carrying a backpack", ["69:14 背包男"]),
    ("a woman wearing a hat", ["146:28 帽女士"]),
    ("a person wearing a white shirt", ["白衣人(318)"]),
    ("a person mopping the floor cleaning", ["1:1 保洁员"]),
    ("a man wearing glasses and dark t-shirt", ["14:3 黑T恤眼镜男"]),
    ("a person wearing a navy blue jacket", ["37:11 藏青外套男"]),
    ("a person wearing a dark hoodie", ["139:24 黑卫衣男"]),
    ("a woman with a fur-collar coat", ["120:18/120:23 皮领女士"]),
]

# 已知正对成员（用于命中标注）
POS_MEMBERS = {1, 14, 37, 69, 120, 139, 146}  # 318 侧正对轨迹
POS_MEMBERS_B = {1, 3, 11, 14, 18, 23, 24, 28}  # 320 侧正对轨迹


def l2(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    from sentence_transformers import SentenceTransformer

    print("[load] model ...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_DIR, model_kwargs={"torch_dtype": "float32"})
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    d = np.load(EMB)
    keys = list(d.files)
    E = np.stack([l2(d[k]) for k in keys])  # (n,768)
    print(f"[emb] {len(keys)} tracks: {keys[:8]}...", flush=True)

    results = []
    for q, targets in QUERIES:
        t0 = time.time()
        qv = np.asarray(model.encode(q, prompt_name="SearchQuery",
                                     normalize_embeddings=True))
        sims = E @ l2(qv)
        order = np.argsort(-sims)
        top = [(keys[i], float(sims[i])) for i in order[:5]]
        dt = time.time() - t0
        print(f"\nQ: \"{q}\"  ({dt:.2f}s)  预期: {targets}")
        for rank, (k, s) in enumerate(top, 1):
            cam = "318" if k.startswith("a") else "320"
            tid = int(k[1:])
            is_pos = (tid in POS_MEMBERS) if k.startswith("a") else (tid in POS_MEMBERS_B)
            tag = " [正对成员]" if is_pos else ""
            print(f"  {rank}. {cam}#{tid:<4} cos={s:.3f}{tag}")
        results.append({"query": q, "targets": targets, "top5": top})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
