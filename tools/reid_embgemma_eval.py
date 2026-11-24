#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmbeddingGemma 2 图像嵌入评估（vs DINOv3-B，同协议对比）

评估集：crosscam_reid_compare.py 的 30 对（8 POS + 22 NEG），同批 crops_s8.pkl（280 张 224×224）。
聚合：逐图嵌入 L2 归一化 → 轨迹均值 → 再 L2 归一化 → 余弦 → AUC。

运行（embgemma venv）:
  C:/Users/might/.workbuddy/binaries/python/envs/embgemma/Scripts/python.exe tools/reid_embgemma_eval.py
"""
import json
import os
import pickle
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "google--embeddinggemma-2", "snapshots", "master")
CROPS = os.path.join(ROOT, "output_crosscam", "analysis", "reid_compare", "crops_s8.pkl")
OUT_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "reid_compare")

POS_PAIRS = [(14, 3), (120, 18), (37, 11), (120, 23), (69, 14), (139, 24),
             (1, 1), (146, 28)]
NEG_PAIRS = [(14, 1), (3, 1), (50, 11), (7, 1), (10, 1), (126, 18), (57, 11),
             (124, 18), (43, 11), (3, 4), (131, 24), (53, 11), (131, 28),
             (131, 23), (142, 26), (98, 26), (61, 14), (123, 19), (127, 19),
             (146, 21), (132, 21), (131, 21)]


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def l2(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    from sentence_transformers import SentenceTransformer

    print(f"[load] {MODEL_DIR}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_DIR, model_kwargs={"torch_dtype": "float32"})
    print(f"[load] done in {time.time()-t0:.1f}s | dim={model.get_sentence_embedding_dimension()}")

    with open(CROPS, "rb") as f:
        ca, cb = pickle.load(f)
    print(f"[crops] A={len(ca)} tracks, B={len(cb)} tracks")

    # 探测单图编码 API（一张图 -> 一个 embedding）
    probe = ca[list(ca.keys())[0]][0]
    print(f"[probe] crop shape={probe.shape} dtype={probe.dtype}")

    def embed_images(imgs):
        """list of (H,W,3) uint8 RGB -> (n,768) normalized embeddings（批量）"""
        E = model.encode([{"image": im} for im in imgs],
                         normalize_embeddings=True, batch_size=8,
                         show_progress_bar=False)
        E = np.asarray(E)
        if E.ndim == 1:
            E = E[None]
        return np.stack([l2(e) for e in E])

    # ---- 渐进缓存（断点续跑）----
    cache_path = os.path.join(OUT_DIR, "embgemma2_cache.npz")
    cache = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        cache = {k: z[k] for k in z.files}
        print(f"[cache] loaded {len(cache)} track embeddings")

    def encode_track(cam_tag, tid, crops):
        key = f"{cam_tag}{tid}"
        if key in cache:
            return cache[key]
        emb = l2(embed_images(crops).mean(axis=0))
        cache[key] = emb
        np.savez(cache_path, **cache)
        return emb

    emb_a, emb_b = {}, {}
    n_img = 0
    t0 = time.time()
    for i, (tid, crops) in enumerate(ca.items()):
        emb_a[tid] = encode_track("a", tid, crops)
        n_img += len(crops)
        print(f"  [A {i+1}/{len(ca)}] tid={tid} ({time.time()-t0:.0f}s)", flush=True)
    for i, (tid, crops) in enumerate(cb.items()):
        emb_b[tid] = encode_track("b", tid, crops)
        n_img += len(crops)
        print(f"  [B {i+1}/{len(cb)}] tid={tid} ({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    print(f"[encode] {n_img} crops in {dt:.1f}s ({1000*dt/max(n_img,1):.0f} ms/img)")

    # 打分
    def cos(a, b):
        return float(emb_a[a] @ emb_b[b])

    pv = [cos(a, b) for a, b in POS_PAIRS]
    nv = [cos(a, b) for a, b in NEG_PAIRS]
    A = auc(pv, nv)
    print(f"\n=== EmbeddingGemma 2（图像, 768d, CPU fp32）===")
    print(f"AUC = {A:.3f}")
    print(f"POS: min={min(pv):.3f} mean={np.mean(pv):.3f} max={max(pv):.3f}")
    print(f"NEG: min={min(nv):.3f} mean={np.mean(nv):.3f} max={max(nv):.3f}")
    print(f"gap = minPOS - maxNEG = {min(pv)-max(nv):+.3f}")
    print(f"\n逐对：")
    for (a, b), v in zip(POS_PAIRS, pv):
        print(f"  POS {a:>3}:{b:<3} {v:.3f}")
    for (a, b), v in zip(NEG_PAIRS, nv):
        print(f"  NEG {a:>3}:{b:<3} {v:.3f}")

    # 与 DINOv3-B 对照（读历史 scores.csv）
    import csv
    dinov3b = {}
    csv_path = os.path.join(OUT_DIR, "scores.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dinov3b[r["pair"]] = float(r["dinov3_vitb16"])
        dp = [dinov3b[f"{a}:{b}"] for a, b in POS_PAIRS if f"{a}:{b}" in dinov3b]
        dn = [dinov3b[f"{a}:{b}"] for a, b in NEG_PAIRS if f"{a}:{b}" in dinov3b]
        if dp and dn:
            Ad = auc(dp, dn)
            print(f"\n=== DINOv3-B/16（对照, 历史）=== AUC = {Ad:.3f}")
            print(f"对照 POS min={min(dp):.3f} / NEG max={max(dn):.3f} / gap={min(dp)-max(dn):+.3f}")

    result = {
        "model": "google/embeddinggemma-2 (local)",
        "dtype": "float32/cpu", "dim": 768,
        "ms_per_img": round(1000 * dt / n_img, 1),
        "auc": round(A, 4),
        "pos": [round(v, 4) for v in pv],
        "neg": [round(v, 4) for v in nv],
        "pos_pairs": POS_PAIRS, "neg_pairs": NEG_PAIRS,
        "min_pos": round(min(pv), 4), "max_neg": round(max(nv), 4),
    }
    out = os.path.join(OUT_DIR, "embgemma2_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
