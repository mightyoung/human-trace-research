#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmbeddingGemma 2 图像嵌入评估（服务器 GPU 版）

路径约定（服务器）:
  WORK = /home/r4900g3/yhh/human-count/crosscam_run/embgemma/
运行:
  /home/r4900g3/yhh/human-count/embed_venv/bin/python reid_embgemma_eval_remote.py
"""
import json
import os
import pickle
import time

import numpy as np

WORK = "/home/r4900g3/yhh/human-count/crosscam_run/embgemma"
MODEL_DIR = os.path.join(WORK, "model", "master")
CROPS = os.path.join(WORK, "crops_s8.pkl")
OUT_DIR = WORK

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
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0"
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} -> device={dev}", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_DIR, device=dev,
                                model_kwargs={"torch_dtype": "float32"})
    print(f"[load] {time.time()-t0:.1f}s, dim={model.get_embedding_dimension()}", flush=True)
    # 记录 GPU 显存
    if torch.cuda.is_available():
        print(f"[gpu] mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    with open(CROPS, "rb") as f:
        ca, cb = pickle.load(f)
    print(f"[crops] A={len(ca)}, B={len(cb)}", flush=True)

    def embed_images(imgs):
        E = model.encode([{"image": im} for im in imgs],
                         normalize_embeddings=True, batch_size=16,
                         show_progress_bar=False)
        E = np.asarray(E)
        if E.ndim == 1:
            E = E[None]
        return np.stack([l2(e) for e in E])

    cache_path = os.path.join(OUT_DIR, "embgemma2_cache_gpu.csv")
    emb_a, emb_b = {}, {}
    n_img = 0
    t0 = time.time()
    for tag, coll, store in [("A", ca, emb_a), ("B", cb, emb_b)]:
        for i, (tid, crops) in enumerate(coll.items()):
            store[tid] = l2(embed_images(crops).mean(axis=0))
            n_img += len(crops)
            print(f"  [{tag} {i+1}/{len(coll)}] tid={tid} elapsed={time.time()-t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"[encode] {n_img} crops in {dt:.1f}s ({1000*dt/n_img:.0f} ms/img)", flush=True)

    # 保存轨迹级嵌入（供 MRL 截断/二次分析）
    emb_path = os.path.join(OUT_DIR, "embgemma2_emb_gpu.npz")
    np.savez(emb_path,
             **{f"a{tid}": v for tid, v in emb_a.items()},
             **{f"b{tid}": v for tid, v in emb_b.items()})
    print(f"[emb] saved {emb_path}", flush=True)

    def cos(a, b):
        return float(emb_a[a] @ emb_b[b])

    pv = [cos(a, b) for a, b in POS_PAIRS]
    nv = [cos(a, b) for a, b in NEG_PAIRS]
    A = auc(pv, nv)
    print(f"\n=== EmbeddingGemma 2（图像, 768d, GPU {dev}, fp32）===")
    print(f"AUC = {A:.4f}")
    print(f"POS: min={min(pv):.4f} mean={np.mean(pv):.4f} max={max(pv):.4f}")
    print(f"NEG: min={min(nv):.4f} mean={np.mean(nv):.4f} max={max(nv):.4f}")
    print(f"gap = minPOS - maxNEG = {min(pv)-max(nv):+.4f}")
    for (a, b), v in zip(POS_PAIRS, pv):
        print(f"  POS {a:>3}:{b:<3} {v:.4f}")
    for (a, b), v in zip(NEG_PAIRS, nv):
        print(f"  NEG {a:>3}:{b:<3} {v:.4f}")

    result = {
        "model": "google/embeddinggemma-2 (local)",
        "device": f"gpu/{dev}", "dtype": "float32", "dim": 768,
        "backend": "transformers-5.19.0 + sentence-transformers-6.1.0 (venv reuse rfdetr env)",
        "ms_per_img": round(1000 * dt / n_img, 2),
        "auc": round(A, 4),
        "pos": [round(v, 4) for v in pv],
        "neg": [round(v, 4) for v in nv],
        "pos_pairs": POS_PAIRS, "neg_pairs": NEG_PAIRS,
        "min_pos": round(min(pv), 4), "max_neg": round(max(nv), 4),
    }
    out = os.path.join(OUT_DIR, "embgemma2_result_gpu.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
