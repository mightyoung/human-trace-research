# -*- coding: utf-8 -*-
"""MRL 截断 / int8 量化 —— cradio_v3 嵌入的判别力保持验证（缓存压缩实验）。

背景：EmbeddingGemma 2 官方支持 MRL（Matryoshka Representation Learning，
前缀子向量即可用）。全局 ReID 缓存计划做「256d 粗筛 + 768d 精排」两级检索、
并把存储压到 1/3。但我们的嵌入经过 W 桥接 + 语言塔，**前缀截断是否保持
判别力必须以本场景数据实测**。本脚本同时给出无损对照（int8 量化）。

评估口径（与历史严格可比）：
  1) 协议A：318×320 判读 30 对（8 正 + 22 负）→ AUC / margin / pos_min / neg_max
  2) 全库检索：8 个 318 侧查询 → gallery=其余 5 机位全库 + 仅 320 库 → rank-1 / 平均 rank
  3) 存储：768d-f16 基准 vs 各压缩档 bytes/轨

用法：
  python tools/mrl_validate.py
  python tools/mrl_validate.py --emb-dir output_crosscam/analysis/cradio_scale_v3
"""
import argparse
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["318", "320", "305", "316", "323", "343"]

POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]

DIMS = [768, 512, 384, 256, 192, 128]


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_embs(emb_dir, pat="{cam}_emb_scale.npz"):
    embs = {}
    for cam in CAMS:
        p = os.path.join(emb_dir, pat.format(cam=cam))
        d = np.load(p)
        embs[cam] = {key_to_tid(k): d[k].astype(np.float32) for k in d.files}
    return embs


def trunc_norm(v, dim):
    """MRL 截断 + 重新 L2 归一化（官方要求截后必须重归一化）"""
    x = np.asarray(v, dtype=np.float32)[:dim]
    return x / (np.linalg.norm(x) + 1e-9)


def quant_int8(v):
    """对称 per-vector int8 量化（存 int8 + scale）→ 还原 f32"""
    x = np.asarray(v, dtype=np.float32)
    s = np.max(np.abs(x)) + 1e-12
    q = np.round(x / s * 127.0).clip(-127, 127).astype(np.int8)
    return q.astype(np.float32) * (s / 127.0)


def sim_fn(transform):
    def s(e, c1, t1, c2, t2):
        a, b = transform(e[c1][t1]), transform(e[c2][t2])
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return s


def auc(pv, nv):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pv for n in nv]))


def proto_a(embs, transform):
    s = sim_fn(transform)
    pv = [s(embs, "318", a, "320", b) for a, b in POS30]
    nv = [s(embs, "318", a, "320", b) for a, b in NEG30]
    return {"auc": round(auc(pv, nv), 4),
            "pos_min": round(min(pv), 4), "pos_mean": round(float(np.mean(pv)), 4),
            "neg_max": round(max(nv), 4), "neg_mean": round(float(np.mean(nv)), 4),
            "margin": round(min(pv) - max(nv), 4)}


def retrieval(embs, transform, query_cam="318", thr_rank=600):
    """8 个已知配对查询：全库（其余 5 路）+ 仅 320 库 rank"""
    # 库矩阵
    gal_cams = [c for c in CAMS if c != query_cam]
    keys, vecs = [], []
    for c in gal_cams:
        for tid, v in embs[c].items():
            keys.append(f"{c}|{tid}")
            vecs.append(transform(v))
    E = np.stack(vecs)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    keys = np.array(keys)
    # 320 库子集
    is320 = np.array([k.startswith("320|") for k in keys])
    E320, keys320 = E[is320], keys[is320]

    ranks, ranks320 = [], []
    for a, b in POS30:
        q = transform(embs[query_cam][a])
        q = q / (np.linalg.norm(q) + 1e-9)
        target = f"320|{b}"
        s = E @ q
        order = np.argsort(-s)[:thr_rank]
        hit = np.where(keys[order] == target)[0]
        ranks.append(int(hit[0]) + 1 if len(hit) else None)
        s3 = E320 @ q
        o3 = np.argsort(-s3)
        hit3 = np.where(keys320[o3] == target)[0]
        ranks320.append(int(hit3[0]) + 1 if len(hit3) else None)
    return {
        "all": {"rank1": sum(1 for r in ranks if r == 1), "n": len(ranks),
                "avg": round(float(np.mean([r for r in ranks if r])), 1) if ranks else None,
                "ranks": ranks},
        "only320": {"rank1": sum(1 for r in ranks320 if r == 1), "n": len(ranks320),
                    "avg": round(float(np.mean([r for r in ranks320 if r])), 2) if ranks320 else None,
                    "ranks": ranks320},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir",
                    default=os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v3"))
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "output_crosscam", "analysis", "mrl",
                                         "mrl_validation.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    embs = load_embs(args.emb_dir)
    n_tracks = sum(len(v) for v in embs.values())
    print(f"[load] {args.emb_dir}: {n_tracks} 轨（{ {c: len(v) for c, v in embs.items()} }）")

    variants = []
    for d in DIMS:
        variants.append((f"mrl_{d}", (lambda dd: (lambda v: trunc_norm(v, dd)))(d), d, "f16"))
    variants.append(("int8_768", quant_int8, 768, "int8+s"))

    results = {"n_tracks": n_tracks, "variants": {}}
    for name, tf, dim, store in variants:
        t0 = time.time()
        pa = proto_a(embs, tf)
        rt = retrieval(embs, tf)
        if store == "f16":
            bytes_per = dim * 2 + 24          # f16 向量 + 元数据估
        else:
            bytes_per = dim * 1 + 4 + 24      # int8 + scale
        results["variants"][name] = {
            "dim": dim, "store": store, "bytes_per_track": bytes_per,
            "protoA": pa, "retrieval": rt,
            "elapsed_s": round(time.time() - t0, 1),
        }
        r320 = rt["only320"]
        print(f"[{name:>9}] AUC={pa['auc']:.4f} margin={pa['margin']:+.4f} | "
              f"全库 rank1 {rt['all']['rank1']}/8 avg {rt['all']['avg']} | "
              f"320库 rank1 {r320['rank1']}/8 avg {r320['avg']} | "
              f"{bytes_per}B/轨（×{1 - bytes_per / (768 * 2 + 24):.0%} 省）")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
