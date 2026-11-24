# -*- coding: utf-8 -*-
"""DINOv3 x EmbeddingGemma 2 融合实验：用 DINOv3-B 替换 Gemma 视觉塔（零训练直连）。

融合管线（官方模块化接口，见 modeling_embedding_gemma2.py）:
    原图 -> Gemma4ImageProcessor -> pixel_values [2520,768] + position_ids
    -> [替换] DINOv3-B 前向 -> patch tokens [2304,768] -> 3x3 平均池化 -> [256,768]
    -> embed_vision (原装 RMSNorm+Linear 768->512) -> 语言塔 -> mean pool -> 嵌入

用法:
    python tools/fusion_dinogemma.py smoke   # 2 张图全流程 smoke + 重建视觉检查
    python tools/fusion_dinogemma.py eval    # 30 对（280 图）评估 → AUC
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

EMB_PATH = os.path.join(ROOT, "models", "google--embeddinggemma-2", "snapshots", "master")
DINO_W = os.path.join(ROOT, "models", "dinov3_vitb16_timm", "model.safetensors")
OUT_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "fusion")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------- #
# 1. DINOv3 视觉塔适配器：Gemma patch 序列 -> DINOv3 -> [N_pool, 768]
# ----------------------------------------------------------------------------- #
class DinoV3Adapter:
    """把 Gemma processor 的 pixel_values/position_ids 送进 DINOv3-B/16。

    - patch 序列重建为图像（[0,1] 值域，HWC 顺序已验证）
    - ImageNet 归一化（DINOv3 训练分布）
    - DINOv3 forward_features -> patch tokens -> 3x3 平均池化 -> 乘 sqrt(768)
    """

    def __init__(self, device="cpu"):
        import torch
        import timm
        self.torch = torch
        self.device = device
        m = timm.create_model("vit_base_patch16_dinov3", pretrained=False,
                              num_classes=0, global_pool="token")
        timm.models.load_checkpoint(m, DINO_W)
        self.model = m.to(device).eval()
        self.num_prefix = m.num_prefix_tokens  # 5 (cls + registers)
        self.pool_k = None  # 从数据推断

    def _rebuild_image(self, pixel_values, position_ids):
        """pixel_values [1, S, 768], position_ids [1, S, 2] -> img [H, W, 3] float [0,1]"""
        torch = self.torch
        pv = pixel_values[0]           # [S, 768]
        ids = position_ids[0]          # [S, 2]
        valid = (ids != -1).all(dim=-1)
        pv = pv[valid]                 # [N, 768]
        ids = ids[valid]               # [N, 2]
        xs, ys = ids[:, 0], ids[:, 1]
        grid_w = int(xs.max()) + 1
        grid_h = int(ys.max()) + 1
        n = grid_w * grid_h
        assert pv.shape[0] == n, f"patch 数 {pv.shape[0]} != 网格 {grid_w}x{grid_h}"
        order = torch.argsort(ys * grid_w + xs)          # row-major 排序
        ph = pw = 16
        patches = pv[order].reshape(grid_h, grid_w, ph, pw, 3)
        img = patches.permute(0, 2, 1, 3, 4).reshape(grid_h * ph, grid_w * pw, 3)
        return img, grid_h, grid_w

    def features(self, pixel_values, position_ids):
        """-> [N_pool, 768]（与 Gemma 池化后的 token 排布一致：row-major 16x16）"""
        torch = self.torch
        img, gh, gw = self._rebuild_image(pixel_values, position_ids)
        # ImageNet 归一化
        mean = torch.tensor(IMAGENET_MEAN, device=img.device)
        std = torch.tensor(IMAGENET_STD, device=img.device)
        x = ((img - mean) / std).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        with torch.no_grad():
            feats = self.model.forward_features(x)
        patch = feats[:, self.num_prefix:, :]         # [1, N, 768]
        # 3x3 平均池化（对应 Gemma pooling_kernel_size=3）
        k = 3
        gh2, gw2 = gh // k, gw // k
        g = patch[0].reshape(gh, gw, -1)[:gh2 * k, :gw2 * k]
        p = g.reshape(gh2, k, gw2, k, -1).mean(dim=(1, 3))   # [gh2, gw2, 768]
        z = p.reshape(-1, p.shape[-1])                 # [N_pool, 768]
        z = z * (z.shape[-1] ** 0.5)                   # 复刻 Gemma sqrt(hidden) 缩放
        return z


# ----------------------------------------------------------------------------- #
# 2. Monkey-patch：替换 EmbeddingGemma2Model.get_image_features
# ----------------------------------------------------------------------------- #
def patch_model(st_model, adapter):
    """把底层 EmbeddingGemma2Model 的视觉塔换成 DINOv3。

    返回 (auto_model, 原函数) 便于恢复。
    """
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    torch = adapter.torch

    wrapper = st_model[0]
    am = getattr(wrapper, "auto_model", None)
    if am is None:
        am = getattr(wrapper, "model", None)
    assert am is not None, f"找不到底层模型（wrapper={type(wrapper)}）"
    assert type(am).__name__ == "EmbeddingGemma2Model", type(am).__name__

    orig = am.get_image_features

    def patched(pixel_values, image_position_ids=None, **kwargs):
        z = adapter.features(pixel_values, image_position_ids)     # [N_pool, 768]
        z = z.to(next(am.embed_vision.parameters()).device)
        pooler_output = am.embed_vision(inputs_embeds=z)           # [N_pool, 512]
        output_length = pixel_values.shape[-2] // (am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z,
            pooler_output=torch.split(pooler_output, split_sizes),
        )

    am.get_image_features = patched
    return am, orig


# ----------------------------------------------------------------------------- #
# 3. 数据与评估
# ----------------------------------------------------------------------------- #
# 主评估协议：22 项负对（与 DINOv3-B/Gemma2 历史评估严格一致，crops_s8 覆盖）
POS = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG = [(14, 1), (3, 1), (50, 11), (7, 1), (10, 1), (126, 18), (57, 11),
       (124, 18), (43, 11), (3, 4), (131, 24), (53, 11), (131, 28),
       (131, 23), (142, 26), (98, 26), (61, 14), (123, 19), (127, 19),
       (146, 21), (132, 21), (131, 21)]
# labels_v2 扩展负对：(123,18) crop 可用；(111,18) 无 crop 不评
EXTRA_NEG = [(123, 18)]


def load_crops():
    p = os.path.join(ROOT, "output_crosscam", "analysis", "reid_compare", "crops_s8.pkl")
    with open(p, "rb") as f:
        ca, cb = pickle.load(f)
    return ca, cb


def load_models(patch=True):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_PATH, model_kwargs={"torch_dtype": "float32"})
    adapter = DinoV3Adapter(device="cpu")
    wrapper = st[0]
    am = getattr(wrapper, "auto_model", None)
    if am is None:
        am = getattr(wrapper, "model", None)
    assert am is not None, f"找不到底层模型 {type(wrapper)}"
    if patch:
        patch_model(st, adapter)
    return st, adapter, am, None


def encode_track(st_model, crops):
    """crops: list of HWC uint8 RGB。返回轨迹级 768d（均值归一化，与评估协议一致）"""
    embs = []
    for im in crops:
        e = st_model.encode({"image": im}, normalize_embeddings=True,
                            show_progress_bar=False)
        e = np.asarray(e).reshape(-1)
        embs.append(e / (np.linalg.norm(e) + 1e-12))
    v = np.mean(embs, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def auc(pv, nv):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pv for n in nv]))


def cmd_smoke(args):
    """2 张图：全流程 + 重建视觉检查"""
    from PIL import Image
    st, adapter, am, orig = load_models(patch=False)
    ca, cb = load_crops()
    ka = sorted(ca.keys())
    tid = ka[0]
    img = ca[tid][0]

    # ① 原版嵌入（未替换）
    e_orig = np.asarray(st.encode({"image": img}, normalize_embeddings=True)).reshape(-1)
    print("[smoke] 原版嵌入 shape:", e_orig.shape)

    # ② 替换视觉塔
    am, orig_fn = patch_model(st, adapter)

    # ③ 融合嵌入
    e_fused = np.asarray(st.encode({"image": img}, normalize_embeddings=True)).reshape(-1)
    print("[smoke] 融合嵌入 shape:", e_fused.shape)
    cos = float(np.dot(e_orig, e_fused))
    print(f"[smoke] cos(原版, 融合) = {cos:.4f}（应显著 < 1，表明视觉塔已替换）")

    # 重建图像视觉检查
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(EMB_PATH)
    o = proc(images=img, return_tensors="pt")
    with adapter.torch.no_grad():
        rimg, gh, gw = adapter._rebuild_image(o["pixel_values"], o["image_position_ids"])
    rimg = (rimg.numpy() * 255).astype(np.uint8)
    rimg_small = np.array(Image.fromarray(rimg).resize((224, 224), Image.LANCZOS))
    # 左=原图，右=重建
    side = np.full((224, 224 * 2 + 6, 3), 255, np.uint8)
    side[:, :224] = img
    side[:, 224 + 6:] = rimg_small
    Image.fromarray(side).save(os.path.join(OUT_DIR, "rebuild_check.png"))
    print(f"[smoke] 重建图 {gh}x{gw} patch -> 已存 rebuild_check.png（左=原图，右=重建）")

    # DINOv3 特征形状
    with adapter.torch.no_grad():
        z = adapter.features(o["pixel_values"], o["image_position_ids"])
    print("[smoke] DINOv3 池化特征 shape:", tuple(z.shape), "（期望 [256, 768]）")

    # 是否与 Gemma 视觉塔输出同形状
    with adapter.torch.no_grad():
        vo = am.vision_tower(pixel_values=o["pixel_values"],
                             pixel_position_ids=o["image_position_ids"])
    print("[smoke] Gemma 视觉塔输出 shape:", tuple(vo.last_hidden_state.shape))
    print("[smoke] PASS" if tuple(z.shape) == tuple(vo.last_hidden_state.shape) else "[smoke] SHAPE MISMATCH!")


def cmd_eval(args):
    """全量 30 对评估（280 图）。结果 json + 缓存 npz。"""
    st, adapter, am, orig = load_models()
    ca, cb = load_crops()
    cache_p = os.path.join(OUT_DIR, "fused_emb_cache.npz")
    cache = {}
    if os.path.exists(cache_p):
        d = np.load(cache_p)
        cache = {k: d[k] for k in d.files}
        print(f"[eval] 缓存载入 {len(cache)} 条")

    t0 = time.time()
    n_new = 0
    for prefix, coll in (("a", ca), ("b", cb)):
        keys = sorted(coll.keys())
        for i, k in enumerate(keys):
            ck = f"{prefix}{k}"
            if ck in cache:
                continue
            e = encode_track(st, coll[k])
            cache[ck] = e.astype(np.float32)
            n_new += 1
            dt = time.time() - t0
            print(f"[eval] {ck} ({n_new} new, {dt/n_new:.1f}s/img est)", flush=True)
            if n_new % 5 == 0:
                np.savez(cache_p, **cache)
    np.savez(cache_p, **cache)
    print(f"[eval] 编码完成，共 {len(cache)} 轨迹")

    pv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in POS]
    nv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in NEG]
    print()
    print("=== 融合模型（DINOv3 视觉塔 + Gemma 语言塔, 零训练）===")
    print(f"AUC = {auc(pv, nv):.4f}")
    print(f"POS: min={min(pv):.4f} mean={np.mean(pv):.4f} max={max(pv):.4f}")
    print(f"NEG: min={min(nv):.4f} mean={np.mean(nv):.4f} max={max(nv):.4f}")
    print(f"gap = minPOS - maxNEG = {min(pv) - max(nv):+.4f}")
    res = {"pos": pv, "neg": nv, "auc": auc(pv, nv), "model": "dino-vision+gemma-lm (zero-shot)"}
    with open(os.path.join(OUT_DIR, "fused_result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[eval] ->", os.path.join(OUT_DIR, "fused_result.json"))


def cmd_bridge(args):
    """闭式解桥接实验（三阶段）：
    collect: 采集 280 图的 (z_dino, z_gemma) token 特征对 -> npz
    solve  : ridge 闭式解 W: z_dino @ W ≈ z_gemma
    eval   : 用 W 桥接后跑 30 对评估
    """
    import numpy as np
    torch = None
    if args.stage == "collect":
        st, adapter, am, orig = load_models(patch=False)
        torch = adapter.torch
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(EMB_PATH)
        ca, cb = load_crops()
        zd, zg = [], []
        t0 = time.time()
        n = 0
        for prefix, coll in (("a", ca), ("b", cb)):
            for k in sorted(coll.keys()):
                for im in coll[k]:   # 每轨全部 8 帧（280 图，样本充分）
                    o = proc(images=im, return_tensors="pt")
                    with torch.no_grad():
                        zgd = am.vision_tower(pixel_values=o["pixel_values"],
                                              pixel_position_ids=o["image_position_ids"]).last_hidden_state
                        zdd = adapter.features(o["pixel_values"], o["image_position_ids"])
                    assert zgd.shape == zdd.shape
                    zd.append(zdd.numpy().astype(np.float32))
                    zg.append(zgd.numpy().astype(np.float32))
                    n += 1
                    if n % 50 == 0:
                        print(f"[collect] {n} imgs, {time.time()-t0:.0f}s", flush=True)
        np.savez(os.path.join(OUT_DIR, "bridge_pairs.npz"),
                 zd=np.stack(zd), zg=np.stack(zg))
        print(f"[collect] 完成 {n} 图 -> bridge_pairs.npz  shapes={np.stack(zd).shape}")

    elif args.stage == "solve":
        d = np.load(os.path.join(OUT_DIR, "bridge_pairs.npz"))
        zd_all = d["zd"]   # [N, 256, 768]
        zg_all = d["zg"]
        n_img = zd_all.shape[0]
        # ---- 分析1：token 级对应 cos（同网格位置）----
        zdf = zd_all.reshape(-1, 768).astype(np.float64)
        zgf = zg_all.reshape(-1, 768).astype(np.float64)
        tok_cos = np.sum(zdf * zgf, axis=1) / (
            np.linalg.norm(zdf, axis=1) * np.linalg.norm(zgf, axis=1) + 1e-9)
        print(f"[analyze] token 级对应 cos（{n_img}图×256tok）: mean={tok_cos.mean():.4f} "
              f"p10={np.percentile(tok_cos,10):.4f} p90={np.percentile(tok_cos,90):.4f}")
        # ---- 分析2：池化特征（256 token 均值）的跨图相似结构 ----
        pd_ = zd_all.mean(axis=1)   # [N, 768]
        pg_ = zg_all.mean(axis=1)
        pd_n = pd_ / (np.linalg.norm(pd_, axis=1, keepdims=True) + 1e-9)
        pg_n = pg_ / (np.linalg.norm(pg_, axis=1, keepdims=True) + 1e-9)
        Sd = pd_n @ pd_n.T
        Sg = pg_n @ pg_n.T
        iu = np.triu_indices(n_img, k=1)
        rep_corr = float(np.corrcoef(Sd[iu], Sg[iu])[0, 1])
        print(f"[analyze] 池化特征跨图相似度矩阵相关（{n_img}图 pairwise）: r={rep_corr:.4f}")
        # ---- 分析3：线性 CKA ----
        def center(X):
            return X - X.mean(axis=0, keepdims=True)
        Xc, Yc = center(zdf), center(zgf)
        # 抽样降低计算（CKA 用全部也行，768x768 矩阵很快）
        K = Xc.T @ Xc
        L = Yc.T @ Yc
        cka = float(np.sum((Xc.T @ Yc) ** 2) / (
            np.sqrt(np.sum(K ** 2)) * np.sqrt(np.sum(L ** 2)) + 1e-12))
        print(f"[analyze] 线性 CKA（token 级）: {cka:.4f}")

        # ---- ridge 闭式解 ----
        M = zdf.shape[0]
        print(f"[solve] pairs M={M}")
        lam = 1.0
        A = zdf.T @ zdf + lam * np.eye(768)
        B = zdf.T @ zgf
        W = np.linalg.solve(A, B)          # [768, 768]
        pred = zdf @ W
        cos_fit = (np.sum(pred * zgf, axis=1) /
                   (np.linalg.norm(pred, axis=1) * np.linalg.norm(zgf, axis=1) + 1e-9))
        print(f"[solve] ridge W 拟合: mean cos={cos_fit.mean():.4f} p10={np.percentile(cos_fit,10):.4f}")
        np.save(os.path.join(OUT_DIR, "bridge_W.npy"), W.astype(np.float32))
        print("[solve] -> bridge_W.npy")

    elif args.stage == "eval":
        st, adapter, am, orig = load_models(patch=False)
        torch = adapter.torch
        W = np.load(os.path.join(OUT_DIR, "bridge_W.npy")).astype(np.float32)
        Wt = torch.from_numpy(W)

        def patched_features(pixel_values, image_position_ids):
            z = adapter.features(pixel_values, image_position_ids)
            return z @ Wt

        # 复用 patch_model 但注入桥接
        import types
        am2, _ = patch_model(st, adapter)
        # 重新实现 patched：包一层
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        def patched(pixel_values, image_position_ids=None, **kwargs):
            z = patched_features(pixel_values, image_position_ids)
            z = z.to(next(am2.embed_vision.parameters()).device)
            pooler_output = am2.embed_vision(inputs_embeds=z)
            output_length = pixel_values.shape[-2] // (am2.config.vision_config.pooling_kernel_size ** 2)
            k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
            non_pad_mask = (image_position_ids != -1).all(dim=-1)
            split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
            return BaseModelOutputWithPooling(
                last_hidden_state=z, pooler_output=torch.split(pooler_output, split_sizes))

        am2.get_image_features = patched
        ca, cb = load_crops()
        cache_p = os.path.join(OUT_DIR, "bridge_emb_cache.npz")
        cache = {}
        if os.path.exists(cache_p):
            dd = np.load(cache_p)
            cache = {k: dd[k] for k in dd.files}
        t0 = time.time()
        n_new = 0
        for prefix, coll in (("a", ca), ("b", cb)):
            for k in sorted(coll.keys()):
                ck = f"{prefix}{k}"
                if ck in cache:
                    continue
                e = encode_track(st, coll[k])
                cache[ck] = e.astype(np.float32)
                n_new += 1
                if n_new % 10 == 0:
                    print(f"[bridge-eval] {ck} ({n_new} new, {(time.time()-t0)/n_new:.1f}s/img)", flush=True)
                    np.savez(cache_p, **cache)
        np.savez(cache_p, **cache)
        pv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in POS]
        nv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in NEG]
        print()
        print("=== 桥接模型（DINOv3 + ridge-W -> Gemma 空间 + Gemma 语言塔）===")
        print(f"AUC = {auc(pv, nv):.4f}")
        print(f"POS: min={min(pv):.4f} mean={np.mean(pv):.4f} max={max(pv):.4f}")
        print(f"NEG: min={min(nv):.4f} mean={np.mean(nv):.4f} max={max(nv):.4f}")
        print(f"gap = {min(pv) - max(nv):+.4f}")
        with open(os.path.join(OUT_DIR, "bridge_result.json"), "w", encoding="utf-8") as f:
            json.dump({"pos": pv, "neg": nv, "auc": auc(pv, nv),
                       "model": "dino+ridgeW+gemma-lm"}, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "eval", "bridge"])
    ap.add_argument("--stage", choices=["collect", "solve", "eval"], default="collect")
    args = ap.parse_args()
    {"smoke": cmd_smoke, "eval": cmd_eval, "bridge": cmd_bridge}[args.cmd](args)


if __name__ == "__main__":
    main()
