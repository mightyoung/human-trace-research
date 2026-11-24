# -*- coding: utf-8 -*-
"""C-RADIOv4 × EmbeddingGemma 2 桥接实验（服务器 GPU）。

桥接链：512 图 -> C-RADIOv4 spatial [1024, 1152] -> 2x2 池化 -> [256, 1152]
        -> ridge W (1152->768) -> Gemma embed_vision -> 语言塔 -> 嵌入

命令:
    python cradio_fusion_remote.py load      # 加载测试
    python cradio_fusion_remote.py perf      # 速度/显存
    python cradio_fusion_remote.py collect   # 采集 280 图 RADIO 特征
    python cradio_fusion_remote.py solve     # ridge W 求解 + 分析
    python cradio_fusion_remote.py eval      # 30 对评估
"""
import os
import sys
import json
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/r4900g3/yhh/human-count/crosscam_run")
from fusion_dinogemma_remote import (  # noqa: E402
    load_crops, POS, NEG, auc, encode_track, EMB_PATH, DEV)

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
RADIO_DIR = os.path.join(ROOT, "cradio", "C-RADIOv4-SO400M")
OUT = os.path.join(ROOT, "cradio")
G = int(os.environ.get("CUDA_DEV", "1"))


def load_radio(device=DEV):
    from transformers import AutoModel, CLIPImageProcessor
    m = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True)
    m = m.to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    return m, proc


def load_gemma():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_PATH, device=DEV, model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    return st, am


def radio_spatial_512(m, proc, img_np, device=DEV):
    """224/768 uint8 RGB -> [256, 1152]（512 输入，2x2 池化到 16x16 网格）"""
    pix = proc(images=[Image.fromarray(img_np)], return_tensors="pt",
               do_resize=True, do_center_crop=False,
               do_normalize=False).pixel_values.to(device)
    with torch.no_grad():
        out = m(pix)
    sp = out[1][0]                       # [1024, 1152]
    g = sp.reshape(32, 32, -1)
    p = g.reshape(16, 2, 16, 2, -1).mean(dim=(1, 3))
    return p.reshape(256, -1)            # [256, 1152]


def rebuild_image(pixel_values, position_ids):
    """Gemma processor 的 patch 序列 -> [H, W, 3] float [0,1]（CPU）"""
    pv = pixel_values[0]
    ids = position_ids[0]
    valid = (ids != -1).all(dim=-1)
    pv = pv[valid]
    ids = ids[valid]
    xs, ys = ids[:, 0], ids[:, 1]
    gw = int(xs.max()) + 1
    gh = int(ys.max()) + 1
    order = torch.argsort(ys * gw + xs)
    patches = pv[order].reshape(gh, gw, 16, 16, 3)
    img = patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)
    return img


def memd(tag):
    a = torch.cuda.memory_allocated(G) / 2**20
    r = torch.cuda.memory_reserved(G) / 2**20
    p = torch.cuda.max_memory_allocated(G) / 2**20
    print(f"  [{tag}] allocated={a:.0f} MB | reserved={r:.0f} MB | peak={p:.0f} MB")


def smi():
    import subprocess
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], text=True)
        for l in out.strip().splitlines():
            p, mm = [x.strip() for x in l.split(",")]
            if p == str(pid):
                print(f"  [nvidia-smi] 本进程实测 = {mm}")
    except Exception as ex:
        print("  smi 查询失败:", ex)


def cmd_load(args):
    t0 = time.time()
    m, proc = load_radio()
    print(f"[load] RADIO {time.time()-t0:.0f}s | params={sum(p.numel() for p in m.parameters())/1e6:.0f}M")
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    z = radio_spatial_512(m, proc, img)
    print("[load] 512 模式特征:", tuple(z.shape))


def cmd_perf(args):
    m, proc = load_radio()
    ca, cb = load_crops()
    imgs = [ca[k][i] for k in sorted(ca.keys())[:3] for i in (0, 3)]
    print(f"[perf] model params={sum(p.numel() for p in m.parameters())/1e6:.0f}M")
    memd("RADIO 加载后")
    # warmup
    for im in imgs[:2]:
        radio_spatial_512(m, proc, im)
    torch.cuda.synchronize()
    # E2E 512 模式（含 processor）
    t0 = time.perf_counter()
    for _ in range(20):
        radio_spatial_512(m, proc, imgs[0])
    torch.cuda.synchronize()
    e2e = (time.perf_counter() - t0) / 20 * 1000
    # 纯前向（GPU 事件）
    pix = proc(images=[Image.fromarray(imgs[0])], return_tensors="pt",
               do_resize=True, do_center_crop=False,
               do_normalize=False).pixel_values.to(DEV)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        with torch.no_grad():
            out = m(pix)
    e.record(); torch.cuda.synchronize()
    fwd = s.elapsed_time(e) / 20
    # processor 计时（CPU）
    t0 = time.perf_counter()
    for _ in range(20):
        proc(images=[Image.fromarray(imgs[0])], return_tensors="pt",
             do_resize=True, do_center_crop=False, do_normalize=False)
    proc_ms = (time.perf_counter() - t0) / 20 * 1000
    # 峰值
    torch.cuda.reset_peak_memory_stats(G)
    radio_spatial_512(m, proc, imgs[0])
    print()
    print(f"  [速度] RADIO 512 模式 E2E={e2e:.0f} ms/图 | processor={proc_ms:.0f} ms | 纯前向(GPU)={fwd:.0f} ms")
    memd("单图前向峰值")
    smi()
    # 224 模式对照
    t0 = time.perf_counter()
    for _ in range(20):
        pix224 = proc(images=[Image.fromarray(imgs[0])], return_tensors="pt").pixel_values.to(DEV)
        with torch.no_grad():
            out = m(pix224)
    torch.cuda.synchronize()
    e2e224 = (time.perf_counter() - t0) / 20 * 1000
    print(f"  [速度] 224 default 模式 E2E={e2e224:.0f} ms/图（对照）")


def cmd_collect(args):
    m, proc = load_radio()
    ca, cb = load_crops()
    zr = []
    t0 = time.time()
    n = 0
    for prefix, coll in (("a", ca), ("b", cb)):
        for k in sorted(coll.keys()):
            for im in coll[k]:
                z = radio_spatial_512(m, proc, im).cpu().numpy().astype(np.float32)
                zr.append(z)
                n += 1
    np.savez(os.path.join(OUT, "radio_feats.npz"), zr=np.stack(zr))
    print(f"[collect] {n} 图 shapes={np.stack(zr).shape} in {time.time()-t0:.0f}s")


def cmd_solve(args):
    d = np.load(os.path.join(OUT, "radio_feats.npz"))
    zr = d["zr"]                                     # [280, 256, 1152]
    dg = np.load(os.path.join(ROOT, "fusion", "bridge_pairs.npz"))
    zg = dg["zg"]                                    # [280, 256, 768]（Gemma 视觉塔输出）
    assert zr.shape[0] == zg.shape[0], (zr.shape, zg.shape)
    n_img = zr.shape[0]
    X = zr.reshape(-1, 1152).astype(np.float64)
    Y = zg.reshape(-1, 768).astype(np.float64)
    # 注：两空间维度不同，token 级逐元素 cos 不适用；用维度无关指标。
    # 分析1：池化特征（256 token 均值）的跨图相似结构相关
    pr = zr.mean(axis=1)
    pg = zg.mean(axis=1)
    pr_n = pr / (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
    pg_n = pg / (np.linalg.norm(pg, axis=1, keepdims=True) + 1e-9)
    iu = np.triu_indices(n_img, k=1)
    rep_corr = float(np.corrcoef((pr_n @ pr_n.T)[iu], (pg_n @ pg_n.T)[iu])[0, 1])
    print(f"[analyze] 池化特征跨图相似度结构相关（{n_img}图）: r={rep_corr:.4f}")
    # 分析2：线性 CKA（跨维度可行）
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    K = Xc.T @ Xc
    L = Yc.T @ Yc
    cka = float(np.sum((Xc.T @ Yc) ** 2) / (
        np.sqrt(np.sum(K ** 2)) * np.sqrt(np.sum(L ** 2)) + 1e-12))
    print(f"[analyze] 线性 CKA（token 级, 1152 vs 768）: {cka:.4f}")
    # ridge W: 1152 -> 768
    lam = 1.0
    A = X.T @ X + lam * np.eye(1152)
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    pred = X @ W
    cos_fit = np.sum(pred * Y, axis=1) / (
        np.linalg.norm(pred, axis=1) * np.linalg.norm(Y, axis=1) + 1e-9)
    print(f"[solve] ridge W (1152->768) 拟合: mean cos={cos_fit.mean():.4f} "
          f"p10={np.percentile(cos_fit,10):.4f}")
    np.save(os.path.join(OUT, "radio_W.npy"), W.astype(np.float32))
    print("[solve] -> radio_W.npy")


def cmd_eval(args):
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    st, am = load_gemma()
    m_radio, proc = load_radio()
    Wt = torch.from_numpy(np.load(os.path.join(OUT, "radio_W.npy")).astype(np.float32)).to(DEV)

    def patched(pixel_values, image_position_ids=None, **kwargs):
        img = rebuild_image(pixel_values, image_position_ids)          # [768,768,3]
        img_np = (img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        z = radio_spatial_512(m_radio, proc, img_np)                   # [256,1152] GPU
        z = (z @ Wt).to(DEV)
        pooler_output = am.embed_vision(inputs_embeds=z)
        output_length = pixel_values.shape[-2] // (am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z, pooler_output=torch.split(pooler_output, split_sizes))

    am.get_image_features = patched
    ca, cb = load_crops()
    cache_p = os.path.join(OUT, "cradio_emb_cache.npz")
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
                print(f"[eval] {ck} ({n_new}, {(time.time()-t0)/n_new:.2f}s/track)", flush=True)
                np.savez(cache_p, **cache)
    np.savez(cache_p, **cache)
    pv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in POS]
    nv = [float(np.dot(cache[f"a{a}"], cache[f"b{b}"])) for a, b in NEG]
    print()
    print("=== C-RADIOv4 桥接模型（RADIO + ridge-W + Gemma 语言塔）===")
    print(f"AUC = {auc(pv, nv):.4f}")
    print(f"POS: min={min(pv):.4f} mean={np.mean(pv):.4f} max={max(pv):.4f}")
    print(f"NEG: min={min(nv):.4f} mean={np.mean(nv):.4f} max={max(nv):.4f}")
    print(f"gap = {min(pv) - max(nv):+.4f}")
    with open(os.path.join(OUT, "cradio_result.json"), "w", encoding="utf-8") as f:
        json.dump({"pos": pv, "neg": nv, "auc": auc(pv, nv),
                   "model": "cradio+ridgeW+gemma-lm"}, f, ensure_ascii=False, indent=1)
    print("[eval] -> cradio_result.json")


def cmd_perf2(args):
    """组合模型（RADIO+Gemma 语言塔）的速度与显存实测"""
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    t0 = time.time()
    st, am = load_gemma()
    m_radio, proc = load_radio()
    print(f"[perf2] 双模型加载 {time.time()-t0:.0f}s")
    memd("加载后（未 patch）")
    Wt = torch.from_numpy(np.load(os.path.join(OUT, "radio_W.npy")).astype(np.float32)).to(DEV)

    def patched(pixel_values, image_position_ids=None, **kwargs):
        img = rebuild_image(pixel_values, image_position_ids)
        img_np = (img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        z = radio_spatial_512(m_radio, proc, img_np)
        z = (z @ Wt).to(DEV)
        pooler_output = am.embed_vision(inputs_embeds=z)
        output_length = pixel_values.shape[-2] // (am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z, pooler_output=torch.split(pooler_output, split_sizes))

    am.get_image_features = patched
    ca, cb = load_crops()
    imgs = [ca[k][i] for k in sorted(ca.keys())[:3] for i in (0, 3)]
    for im in imgs[:2]:
        st.encode({"image": im}, show_progress_bar=False)
    torch.cuda.reset_peak_memory_stats(G)
    t0 = time.perf_counter()
    for _ in range(20):
        st.encode({"image": imgs[0]}, show_progress_bar=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    memd("单图前向后（峰值）")
    smi()
    print(f"  [速度] C-RADIOv4 组合模型 E2E = {ms:.0f} ms/图（20 次重复）")
    print(f"  [参数量] RADIO {sum(p.numel() for p in m_radio.parameters())/1e6:.0f}M + Gemma 744M")


if __name__ == "__main__":
    {"load": cmd_load, "perf": cmd_perf, "collect": cmd_collect,
     "solve": cmd_solve, "eval": cmd_eval, "perf2": cmd_perf2}[sys.argv[1]](sys.argv[1:])
