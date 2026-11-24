#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语义查人（以文搜人）—— 文本属性词表离线编码。

远程服务器无 EmbeddingGemma 模型，故文本塔向量必须本地预计算后下发。
做法：把"颜色 x 服饰 + 独立概念"展开成英文短语，用 EmbeddingGemma2 编码成
768d（与全局 ReID 桥接库同空间），运行时按中文关键词解析后组合命中短语向量。

产物：
  webapp/backend/data/reid/text_vocab.npz     {phrases: [...], emb: (N,768)}
  webapp/backend/data/reid/text_eval.json     已知人物检索命中自评

运行（embgemma venv）：
  HF_HUB_OFFLINE=1 <embgemma_py> webapp/dataprep/build_text_vocab.py
"""
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models", "google--embeddinggemma-2", "snapshots", "master")
DATA = os.path.join(ROOT, "webapp", "backend", "data")

# ---- 概念体系 ----
COLORS = {"white": "白", "black": "黑", "gray": "灰", "red": "红", "blue": "蓝",
          "navy": "藏青", "green": "绿", "yellow": "黄", "brown": "棕",
          "beige": "米白", "pink": "粉", "orange": "橙", "purple": "紫"}
# garment key -> (英文名, 中文别名)
GARMENTS = {
    "top": ("top", "上衣 衬衫 衬衣 衣服 衫 T恤 短袖"),
    "jacket": ("jacket", "外套 夹克 西装"),
    "hoodie": ("hoodie", "卫衣 连帽"),
    "pants": ("pants", "裤 裤子 长裤"),
    "dress": ("dress", "裙 裙子 连衣裙"),
    "hat": ("hat", "帽 帽子"),
}
STANDALONE = {
    "backpack": ("a person carrying a backpack", "背包 书包 双肩包"),
    "glasses": ("a person wearing glasses", "眼镜 戴眼镜"),
    "mask": ("a person wearing a face mask", "口罩"),
    "scarf": ("a person wearing a scarf", "围巾"),
    "umbrella": ("a person holding an umbrella", "伞 打伞"),
    "hair_long": ("a person with long hair", "长发"),
}
GENDER = {"man": ("a man", "男 男士 男人"), "woman": ("a woman", "女 女士 女人")}
ACTION = {
    "clean": ("a person mopping the floor cleaning", "保洁 拖地 打扫 清洁"),
    "backpack": ("a person carrying a backpack", "背书包"),
}
SPECIAL = {
    "fur_coat": ("a woman with a fur-collar coat", "毛皮领 皮领 毛领"),
}
GENERIC = "a person walking in an office building"


def build_phrases():
    phrases = {}  # key -> english phrase
    for ckey in list(COLORS) + ["any"]:
        for gkey, (gen, _) in GARMENTS.items():
            phrases[f"{ckey}|{gkey}"] = (
                f"a person wearing a {gen}" if ckey == "any" else f"a person wearing a {ckey} {gen}")
        phrases[f"{ckey}|color"] = (
            "a person" if ckey == "any" else f"a person wearing {ckey} clothing")
    for skey, (ph, _) in STANDALONE.items():
        phrases[f"-|{skey}"] = ph
    for gkey, (ph, _) in GENDER.items():
        phrases[f"-|{gkey}"] = ph
    for akey, (ph, _) in ACTION.items():
        phrases[f"-|{akey}"] = ph
    for skey, (ph, _) in SPECIAL.items():
        phrases[f"-|{skey}"] = ph
    phrases["-|generic"] = GENERIC
    return phrases


def main():
    from sentence_transformers import SentenceTransformer
    phrases = build_phrases()
    keys = list(phrases.keys())
    texts = [phrases[k] for k in keys]
    print(f"[vocab] {len(texts)} phrases", flush=True)

    t0 = time.time()
    model = SentenceTransformer(MODEL_DIR, model_kwargs={"torch_dtype": "float32"})
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    E = np.asarray(model.encode(texts, prompt_name="SearchQuery", normalize_embeddings=True,
                                batch_size=32, show_progress_bar=False), dtype=np.float32)
    print(f"[enc] {time.time()-t0:.1f}s -> {E.shape}", flush=True)

    out = os.path.join(DATA, "reid", "text_vocab.npz")
    np.savez_compressed(out, keys=np.array(keys), phrases=np.array(texts), emb=E)
    print(f"[out] {out} ({os.path.getsize(out)//1024} KB)")

    # ---- 自评：已知人物在全局库中的检索命中 ----
    store = np.load(os.path.join(DATA, "reid", "store_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(DATA, "reid", "store_meta.json"), encoding="utf-8"))
    idx = {(m["cam"], m["tid"]): m["i"] for m in meta["tracks"]}
    KEY2I = {k: i for i, k in enumerate(keys)}

    def qvec(parts):
        v = np.mean([E[KEY2I[p]] for p in parts if p in KEY2I], axis=0)
        return v / (np.linalg.norm(v) + 1e-9)

    # 正例 = 人工确认的同一人在两侧机位的轨迹（318/320）
    CASES = [
        ("白衣服灰帽子的人", ["white|top", "gray|hat"], [("318", 3), ("318", 146), ("320", 28)]),
        ("背书包的人", ["-|backpack"], [("318", 69), ("320", 14)]),
        ("戴眼镜穿黑T恤的男人", ["black|top", "-|glasses", "-|man"], [("318", 14), ("320", 3)]),
        ("藏青外套的男士", ["navy|jacket"], [("318", 37), ("320", 11)]),
        ("穿黑卫衣的人", ["black|hoodie"], [("318", 139), ("320", 24)]),
        ("保洁员在拖地", ["-|clean"], [("318", 1), ("320", 1)]),
        ("戴帽子的女士", ["gray|hat", "-|woman"], [("318", 146), ("320", 28)]),
        ("毛皮领外套的女士", ["-|fur_coat"], [("318", 120), ("320", 18), ("320", 23)]),
    ]
    ev = []
    for q, parts, pos in CASES:
        qv = qvec(parts)
        sim = store @ qv
        order = np.argsort(-sim)[:8]
        top = [(meta["tracks"][int(i)]["cam"], meta["tracks"][int(i)]["tid"], round(float(sim[i]), 3))
               for i in order]
        # 最佳正例排名
        full = np.argsort(-sim)
        best = None
        for r, i in enumerate(full):
            m = meta["tracks"][int(i)]
            if (m["cam"], m["tid"]) in pos:
                best = (r + 1, m["cam"], m["tid"]); break
        hit = bool(best and best[0] <= 5)
        print(f'\nQ "{q}"  parts={parts}  最佳正例排名={best}  top5命中={hit}')
        print("  top8:", top)
        ev.append({"query": q, "parts": parts, "pos": pos, "top8": top,
                   "best_rank": best[0] if best else None, "hit5": hit})
    n_hit = sum(1 for e in ev if e["hit5"])
    print(f"\n[eval] top-5 命中 {n_hit}/{len(ev)}")
    json.dump({"n_query": len(ev), "hit5": n_hit, "cases": ev},
              open(os.path.join(DATA, "reid", "text_eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
