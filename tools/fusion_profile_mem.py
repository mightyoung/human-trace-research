# -*- coding: utf-8 -*-
"""精测 T3 融合模型显存与速度（服务器）: 修正设备索引 + nvidia-smi 全量。

用法: python fusion_profile_mem.py
"""
import os
import sys
import time
import subprocess

import numpy as np
import torch

sys.path.insert(0, "/home/r4900g3/yhh/human-count/crosscam_run")
from fusion_dinogemma_remote import load_models, load_crops, DEV  # noqa: E402

DEV_IDX = 1


def memd(tag):
    a = torch.cuda.memory_allocated(DEV_IDX) / 2**20
    r = torch.cuda.memory_reserved(DEV_IDX) / 2**20
    p = torch.cuda.max_memory_allocated(DEV_IDX) / 2**20
    print(f"  [{tag}] allocated={a:.0f} MB | reserved={r:.0f} MB | peak={p:.0f} MB")
    return a, r, p


def smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader"], text=True)
        lines = [l for l in out.strip().splitlines() if l.strip()]
        print("  [nvidia-smi 计算进程]")
        for l in lines:
            print("    ", l)
        print(f"  [本进程 pid={os.getpid()}]")
    except Exception as ex:
        print("  查询失败:", ex)


def main():
    # 基线
    print("=== 环境基线 ===")
    memd("基线")
    smi()

    print("=== 加载原版 Gemma（对照） ===")
    from sentence_transformers import SentenceTransformer
    from fusion_dinogemma_remote import EMB_PATH
    st = SentenceTransformer(EMB_PATH, device=DEV,
                             model_kwargs={"torch_dtype": "float32"})
    memd("原版加载后")
    ca, cb = load_crops()
    imgs = [ca[k][i] for k in sorted(ca.keys())[:3] for i in (0, 3)]
    for im in imgs[:2]:
        st.encode({"image": im}, show_progress_bar=False)
    torch.cuda.reset_peak_memory_stats(DEV_IDX)
    st.encode({"image": imgs[0]}, show_progress_bar=False)
    memd("原版单图前向后")

    print("=== 卸载原版 + 加载融合模型（T3） ===")
    del st
    torch.cuda.empty_cache()
    memd("卸载后")

    W = np.load("/home/r4900g3/yhh/human-count/crosscam_run/fusion/bridge_W.npy").astype(np.float32)
    st2, adapter, am = load_models(patch=True)
    Wt = torch.from_numpy(W).to(DEV)
    # 桥接版本 patch（同 profile 脚本）
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    def patched(pixel_values, image_position_ids=None, **kwargs):
        z = adapter.features(pixel_values, image_position_ids)
        z = (z @ Wt).to(DEV)
        pooler_output = am.embed_vision(inputs_embeds=z)
        output_length = pixel_values.shape[-2] // (am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z, pooler_output=torch.split(pooler_output, split_sizes))

    am.get_image_features = patched
    print("  (patch 完成, 未 warmup)")
    memd("融合加载后")

    for im in imgs[:2]:
        st2.encode({"image": im}, show_progress_bar=False)
    memd("融合 warmup 后")

    torch.cuda.reset_peak_memory_stats(DEV_IDX)
    t0 = time.perf_counter()
    for _ in range(20):
        st2.encode({"image": imgs[0]}, show_progress_bar=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"  [速度] 融合 E2E = {ms:.0f} ms/图（20 次重复均值）")
    memd("融合单图前向后（峰值）")
    smi()

    # 本地 CPU 内存
    try:
        import psutil
        p = psutil.Process(os.getpid())
        print(f"  [宿主内存] RSS = {p.memory_info().rss/2**30:.1f} GB")
    except Exception:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if "VmRSS" in line:
                    print(f"  [宿主内存] {line.strip()}")


if __name__ == "__main__":
    main()
