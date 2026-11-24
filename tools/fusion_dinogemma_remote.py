# -*- coding: utf-8 -*-
"""服务器版融合实验（GPU）：DINOv3 视觉塔 -> EmbeddingGemma 2 语言塔。

与本地 tools/fusion_dinogemma.py 同逻辑，路径与设备适配服务器：
    python fusion_dinogemma_remote.py eval                    # T1 零训练直连
    python fusion_dinogemma_remote.py bridge --stage collect  # T3 采集
    python fusion_dinogemma_remote.py bridge --stage solve
    python fusion_dinogemma_remote.py bridge --stage eval
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
EMB_PATH = os.path.join(ROOT, "embgemma", "model", "master")
DINO_W = os.path.join(ROOT, "models", "dinov3_vitb16_timm", "model.safetensors")
CROPS_P = os.path.join(ROOT, "embgemma", "crops_s8.pkl")
OUT_DIR = os.path.join(ROOT, "fusion")
os.makedirs(OUT_DIR, exist_ok=True)
DEV = os.environ.get("FUSION_DEV", "cuda:1")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DinoV3Adapter:
    def __init__(self, device="cuda:1"):
        import torch
        import timm
        self.torch = torch
        self.device = device
        m = timm.create_model("vit_base_patch16_dinov3", pretrained=False,
                              num_classes=0, global_pool="token")
        timm.models.load_checkpoint(m, DINO_W)
        self.model = m.to(device).eval()
        self.num_prefix = m.num_prefix_tokens

    def _rebuild_image(self, pixel_values, position_ids):
        torch = self.torch
        pv = pixel_values[0]
        ids = position_ids[0]
        valid = (ids != -1).all(dim=-1)
        pv = pv[valid]
        ids = ids[valid]
        xs, ys = ids[:, 0], ids[:, 1]
        grid_w = int(xs.max()) + 1
        grid_h = int(ys.max()) + 1
        n = grid_w * grid_h
        assert pv.shape[0] == n
        order = torch.argsort(ys * grid_w + xs)
        ph = pw = 16
        patches = pv[order].reshape(grid_h, grid_w, ph, pw, 3)
        img = patches.permute(0, 2, 1, 3, 4).reshape(grid_h * ph, grid_w * pw, 3)
        return img, grid_h, grid_w

    def features(self, pixel_values, position_ids):
        torch = self.torch
        img, gh, gw = self._rebuild_image(pixel_values, position_ids)
        mean = torch.tensor(IMAGENET_MEAN, device=img.device)
        std = torch.tensor(IMAGENET_STD, device=img.device)
        x = ((img - mean) / std).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.forward_features(x)
        patch = feats[:, self.num_prefix:, :]
        k = 3
        gh2, gw2 = gh // k, gw // k
        g = patch[0].reshape(gh, gw, -1)[:gh2 * k, :gw2 * k]
        p = g.reshape(gh2, k, gw2, k, -1).mean(dim=(1, 3))
        z = p.reshape(-1, p.shape[-1])
        z = z * (z.shape[-1] ** 0.5)
        return z


def patch_model(st_model, adapter):
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    torch = adapter.torch
    wrapper = st_model[0]
    am = getattr(wrapper, "auto_model", None)
    if am is None:
        am = getattr(wrapper, "model", None)
    assert am is not None
    assert type(am).__name__ == "EmbeddingGemma2Model", type(am).__name__

    def patched(pixel_values, image_position_ids=None, **kwargs):
        z = adapter.features(pixel_values, image_position_ids)
        z = z.to(next(am.embed_vision.parameters()).device)
        pooler_output = am.embed_vision(inputs_embeds=z)
        output_length = pixel_values.shape[-2] // (am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z,
            pooler_output=torch.split(pooler_output, split_sizes),
        )

    am.get_image_features = patched
    return am


# 主评估协议：22 项负对（与 DINOv3-B/Gemma2 历史评估严格一致，crops_s8 覆盖）
POS = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG = [(14, 1), (3, 1), (50, 11), (7, 1), (10, 1), (126, 18), (57, 11),
       (124, 18), (43, 11), (3, 4), (131, 24), (53, 11), (131, 28),
       (131, 23), (142, 26), (98, 26), (61, 14), (123, 19), (127, 19),
       (146, 21), (132, 21), (131, 21)]
# labels_v2 扩展负对：(123,18) crop 可用；(111,18) 无 crop 不评
EXTRA_NEG = [(123, 18)]


def load_crops():
    with open(CROPS_P, "rb") as f:
        ca, cb = pickle.load(f)
    return ca, cb


def load_models(patch=True):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_PATH, device=DEV,
                             model_kwargs={"torch_dtype": "float32"})
    adapter = DinoV3Adapter(device=DEV)
    wrapper = st[0]
    am = getattr(wrapper, "auto_model", None)
    if am is None:
        am = getattr(wrapper, "model", None)
    assert am is not None, f"找不到底层模型 {type(wrapper)}"
    if patch:
        patch_model(st, adapter)
    return st, adapter, am


def encode_track(st_model, crops):
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


def cmd_eval(args):
    """T1: 零训练直连全量评估"""
    st, adapter, am = load_models(patch=True)
    ca, cb = load_crops()
    cache_p = os.path.join(OUT_DIR, "fused_emb_cache.npz")
    cache = {}
    if os.path.exists(cache_p):
        d = np.load(cache_p)
        cache = {k: d[k] for k in d.files}
        print(f"[eval] 缓存载入 {len(cache)}", flush=True)
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
                print(f"[eval] {ck} ({n_new} new, {(time.time()-t0)/n_new:.2f}s/img)", flush=True)
                np.savez(cache_p, **cache)
    np.savez(cache_p, **cache)
    pv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in POS]
    nv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in NEG]
    print()
    print("=== T1 融合模型（DINOv3 视觉塔 + Gemma 语言塔, 零训练）===")
    print(f"AUC = {auc(pv, nv):.4f}")
    print(f"POS: min={min(pv):.4f} mean={np.mean(pv):.4f} max={max(pv):.4f}")
    print(f"NEG: min={min(nv):.4f} mean={np.mean(nv):.4f} max={max(nv):.4f}")
    print(f"gap = {min(pv) - max(nv):+.4f}")
    with open(os.path.join(OUT_DIR, "fused_result.json"), "w", encoding="utf-8") as f:
        json.dump({"pos": pv, "neg": nv, "auc": auc(pv, nv),
                   "model": "dino-vision+gemma-lm (zero-shot)"}, f, ensure_ascii=False, indent=1)
    print("[eval] -> fused_result.json")


def cmd_bridge(args):
    import numpy as np
    if args.stage == "collect":
        st, adapter, am = load_models(patch=False)
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
                    pv_ = o["pixel_values"].to(DEV)
                    ids_ = o["image_position_ids"].to(DEV)
                    with torch.no_grad():
                        zgd = am.vision_tower(pixel_values=pv_, pixel_position_ids=ids_).last_hidden_state
                        zdd = adapter.features(o["pixel_values"], o["image_position_ids"])
                    zgd = zgd.cpu().numpy().astype(np.float32)
                    zdd = zdd.cpu().numpy().astype(np.float32)
                    assert zgd.shape == zdd.shape, (zgd.shape, zdd.shape)
                    zd.append(zdd)
                    zg.append(zgd)
                    n += 1
                    if n % 50 == 0:
                        print(f"[collect] {n} imgs, {time.time()-t0:.0f}s", flush=True)
        np.savez(os.path.join(OUT_DIR, "bridge_pairs.npz"),
                 zd=np.stack(zd), zg=np.stack(zg))
        print(f"[collect] 完成 {n} 图 shapes={np.stack(zd).shape}")

    elif args.stage == "solve":
        d = np.load(os.path.join(OUT_DIR, "bridge_pairs.npz"))
        zd_all, zg_all = d["zd"], d["zg"]
        n_img = zd_all.shape[0]
        zdf = zd_all.reshape(-1, 768).astype(np.float64)
        zgf = zg_all.reshape(-1, 768).astype(np.float64)
        tok_cos = np.sum(zdf * zgf, axis=1) / (
            np.linalg.norm(zdf, axis=1) * np.linalg.norm(zgf, axis=1) + 1e-9)
        print(f"[analyze] token 级对应 cos（{n_img}图×{zdf.shape[0]//n_img}tok）: "
              f"mean={tok_cos.mean():.4f} p10={np.percentile(tok_cos,10):.4f} "
              f"p90={np.percentile(tok_cos,90):.4f}")
        pd_ = zd_all.mean(axis=1)
        pg_ = zg_all.mean(axis=1)
        pd_n = pd_ / (np.linalg.norm(pd_, axis=1, keepdims=True) + 1e-9)
        pg_n = pg_ / (np.linalg.norm(pg_, axis=1, keepdims=True) + 1e-9)
        iu = np.triu_indices(n_img, k=1)
        rep_corr = float(np.corrcoef((pd_n @ pd_n.T)[iu], (pg_n @ pg_n.T)[iu])[0, 1])
        print(f"[analyze] 池化特征跨图相似度矩阵相关: r={rep_corr:.4f}")
        Xc = zdf - zdf.mean(axis=0, keepdims=True)
        Yc = zgf - zgf.mean(axis=0, keepdims=True)
        K = Xc.T @ Xc
        L = Yc.T @ Yc
        cka = float(np.sum((Xc.T @ Yc) ** 2) / (
            np.sqrt(np.sum(K ** 2)) * np.sqrt(np.sum(L ** 2)) + 1e-12))
        print(f"[analyze] 线性 CKA（token 级）: {cka:.4f}")
        lam = 1.0
        A = zdf.T @ zdf + lam * np.eye(768)
        B = zdf.T @ zgf
        W = np.linalg.solve(A, B)
        pred = zdf @ W
        cos_fit = (np.sum(pred * zgf, axis=1) /
                   (np.linalg.norm(pred, axis=1) * np.linalg.norm(zgf, axis=1) + 1e-9))
        print(f"[solve] ridge W 拟合: mean cos={cos_fit.mean():.4f} "
              f"p10={np.percentile(cos_fit,10):.4f}")
        np.save(os.path.join(OUT_DIR, "bridge_W.npy"), W.astype(np.float32))
        print("[solve] -> bridge_W.npy")

    elif args.stage == "eval":
        st, adapter, am = load_models(patch=False)
        torch = adapter.torch
        W = np.load(os.path.join(OUT_DIR, "bridge_W.npy")).astype(np.float32)
        Wt = torch.from_numpy(W)
        patch_model(st, adapter)  # 先打上基础 patch，再覆盖
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        dev = next(am.embed_vision.parameters()).device

        def patched(pixel_values, image_position_ids=None, **kwargs):
            z = adapter.features(pixel_values, image_position_ids)
            z = (z @ Wt.to(dev)).to(dev)
            pooler_output = am.embed_vision(inputs_embeds=z)
            output_length = pixel_values.shape[-2] // (am.config.vision_config.pooling_kernel_size ** 2)
            k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
            non_pad_mask = (image_position_ids != -1).all(dim=-1)
            split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
            return BaseModelOutputWithPooling(
                last_hidden_state=z, pooler_output=torch.split(pooler_output, split_sizes))

        am.get_image_features = patched
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
                    print(f"[bridge-eval] {ck} ({n_new}, {(time.time()-t0)/n_new:.2f}s/img)", flush=True)
                    np.savez(cache_p, **cache)
        np.savez(cache_p, **cache)
        pv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in POS]
        nv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in NEG]
        print()
        print("=== T3 桥接模型（DINOv3 + ridge-W -> Gemma 空间 + Gemma 语言塔）===")
        print(f"AUC = {auc(pv, nv):.4f}")
        print(f"POS: min={min(pv):.4f} mean={np.mean(pv):.4f} max={max(pv):.4f}")
        print(f"NEG: min={min(nv):.4f} mean={np.mean(nv):.4f} max={max(nv):.4f}")
        print(f"gap = {min(pv) - max(nv):+.4f}")
        with open(os.path.join(OUT_DIR, "bridge_result.json"), "w", encoding="utf-8") as f:
            json.dump({"pos": pv, "neg": nv, "auc": auc(pv, nv),
                       "model": "dino+ridgeW+gemma-lm"}, f, ensure_ascii=False, indent=1)
        print("[bridge-eval] -> bridge_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "bridge"])
    ap.add_argument("--stage", choices=["collect", "solve", "eval"], default="collect")
    args = ap.parse_args()
    if args.cmd == "eval":
        cmd_eval(args)
    else:
        cmd_bridge(args)


if __name__ == "__main__":
    main()
