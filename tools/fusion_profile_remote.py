# -*- coding: utf-8 -*-
"""T3 融合模型性能剖析（服务器 GPU）：速度分环节 + 显存占用 + 对比原版。

用法: python fusion_profile_remote.py
"""
import os
import sys
import time
import subprocess

import numpy as np
import torch

sys.path.insert(0, "/home/r4900g3/yhh/human-count/crosscam_run")
from fusion_dinogemma_remote import load_models, load_crops, DEV, EMB_PATH  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

N_REP = 20


def gpu_clock_ms(fn, n=N_REP):
    """GPU 事件精确计时（ms/次）"""
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def wall_ms(fn, n=N_REP):
    """含 CPU 环节的墙钟计时（ms/次）"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def mem_report(tag):
    alloc = torch.cuda.memory_allocated() / 2**20
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  [{tag}] 当前分配={alloc:.0f} MB | 峰值={peak:.0f} MB")
    return alloc, peak


def smi_mem():
    """nvidia-smi 查本进程实际显存"""
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], text=True)
        for line in out.strip().splitlines():
            p, m = [x.strip() for x in line.split(",")]
            if p == str(pid):
                return m
    except Exception as ex:
        return f"(查询失败 {ex})"
    return "(未找到)"


def main():
    ca, cb = load_crops()
    imgs = [ca[k][i] for k in sorted(ca.keys())[:5] for i in (0, 3, 7)]  # 15 张
    proc = AutoProcessor.from_pretrained(EMB_PATH)

    # ---------------- 阶段 A：原版模型 ----------------
    print("=" * 64)
    print("阶段 A：原版 EmbeddingGemma 2（对照）")
    from sentence_transformers import SentenceTransformer
    st_o = SentenceTransformer(EMB_PATH, device=DEV,
                               model_kwargs={"torch_dtype": "float32"})
    am_o = getattr(st_o[0], "auto_model", None) or getattr(st_o[0], "model", None)
    n_gemma = sum(p.numel() for p in am_o.parameters())
    print(f"  参数量: {n_gemma/1e6:.0f} M (全部塔)")
    mem_report("原版模型加载后")
    # warmup
    for im in imgs[:2]:
        st_o.encode({"image": im}, show_progress_bar=False)
    e2e_o = wall_ms(lambda: st_o.encode({"image": imgs[0]}, show_progress_bar=False), n=N_REP)
    proc_o = wall_ms(lambda: proc(images=imgs[0], return_tensors="pt"), n=N_REP)
    torch.cuda.reset_peak_memory_stats()
    st_o.encode({"image": imgs[0]}, show_progress_bar=False)
    peak_o = torch.cuda.max_memory_allocated() / 2**20
    print(f"  [速度] E2E={e2e_o:.0f} ms/图 | processor={proc_o:.0f} ms | 其余(全塔前向)={e2e_o-proc_o:.0f} ms")
    print(f"  [显存] 单图前向峰值={peak_o:.0f} MB")

    # 卸载
    del st_o, am_o
    torch.cuda.empty_cache()

    # ---------------- 阶段 B：融合模型（T3 桥接） ----------------
    print("=" * 64)
    print("阶段 B：T3 融合模型（DINOv3 + ridge-W + Gemma 语言塔）")
    W = np.load("/home/r4900g3/yhh/human-count/crosscam_run/fusion/bridge_W.npy").astype(np.float32)
    Wt = torch.from_numpy(W)

    st, adapter, am = load_models(patch=False)
    # 手动 patch（带桥接 W）
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    dev = DEV
    am_orig_fn = am.get_image_features

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

    n_dino = sum(p.numel() for p in adapter.model.parameters())
    print(f"  参数量: DINOv3-B={n_dino/1e6:.0f} M + Gemma(全塔)={n_gemma/1e6:.0f} M = {(n_dino+n_gemma)/1e6:.0f} M")

    am.get_image_features = patched
    mem_report("融合模型加载后（patch 前）")

    # warmup（触发 DINOv3 在 GPU 上的 kernel 编译/缓存）
    for im in imgs[:2]:
        st.encode({"image": im}, show_progress_bar=False)
    mem_report("warmup 后")

    torch.cuda.reset_peak_memory_stats()
    e2e_f = wall_ms(lambda: st.encode({"image": imgs[0]}, show_progress_bar=False), n=N_REP)
    peak_f = torch.cuda.max_memory_allocated() / 2**20

    # 分环节
    o = proc(images=imgs[0], return_tensors="pt")
    proc_f = wall_ms(lambda: proc(images=imgs[0], return_tensors="pt"), n=N_REP)

    def dino_step():
        z = adapter.features(o["pixel_values"], o["image_position_ids"])
        return z @ Wt.to(dev)
    dino_ms = gpu_clock_ms(dino_step)

    def embed_step():
        z = adapter.features(o["pixel_values"], o["image_position_ids"]) @ Wt.to(dev)
        return am.embed_vision(inputs_embeds=z)
    embed_ms = gpu_clock_ms(embed_step) - dino_ms

    print()
    print("  [速度拆解] (每图, 15 张样本)"
          f"\n    E2E                 = {e2e_f:7.0f} ms"
          f"\n    ├─ processor        = {proc_f:7.0f} ms  (resize+patchify, CPU)"
          f"\n    ├─ DINOv3 2304tok前向 = {dino_ms:7.0f} ms  (含重建+归一化+池化+W)"
          f"\n    ├─ embed_vision     = {embed_ms:7.0f} ms"
          f"\n    └─ 语言塔+池化+回归  ≈ {e2e_f - proc_f - dino_ms - embed_ms:6.0f} ms")
    print(f"  [显存] 单图前向峰值={peak_f:.0f} MB | 模型权重≈{(n_dino+n_gemma)*4/2**20:.0f} MB(fp32)")
    print(f"  [显存] nvidia-smi 本进程实测 = {smi_mem()}")

    # 对比汇总
    print("=" * 64)
    print("汇总（服务器 L20，fp32，单图串行）:")
    print(f"  原版 Gemma2      : {e2e_o:6.0f} ms/图")
    print(f"  融合模型（T3）    : {e2e_f:6.0f} ms/图  (×{e2e_f/e2e_o:.1f})")
    print(f"  推算 1 万图耗时    : 原版 {10000*e2e_o/3600/1000:.1f} h | 融合 {10000*e2e_f/3600/1000:.1f} h")


if __name__ == "__main__":
    main()
