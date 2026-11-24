# -*- coding: utf-8 -*-
"""C-RADIOv4 模块级微基准：完整 vs spatial 路径、fp32/fp16、512/224 模式、参数分布。"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
RADIO_DIR = os.path.join(ROOT, "cradio", "C-RADIOv4-SO400M")
DEV = os.environ.get("FUSION_DEV", "cuda:0")


def load_radio():
    from transformers import AutoModel, CLIPImageProcessor
    m = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True)
    m = m.to(DEV).eval()
    proc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    return m, proc


def main():
    m, proc = load_radio()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[parts] 总参数 {n_params/1e6:.0f}M | device {DEV}", flush=True)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pix512 = proc(images=[Image.fromarray(img)], return_tensors="pt",
                  do_resize=True, do_center_crop=False,
                  do_normalize=False).pixel_values.to(DEV)
    pix224 = proc(images=[Image.fromarray(img)], return_tensors="pt").pixel_values.to(DEV)
    print(f"[parts] 512 输入 {tuple(pix512.shape)} | 224 输入 {tuple(pix224.shape)}")

    def bench(fn, n=20, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    ms32 = bench(lambda: m(pix512))
    ms_sp = bench(lambda: m(pix512)[1])
    ms224 = bench(lambda: m(pix224))
    print(f"[parts] fp32-512 full={ms32:.1f}ms | spatial 取用={ms_sp:.1f}ms（与 full 同量级=无额外计算）")
    print(f"[parts] fp32-224 full={ms224:.1f}ms", flush=True)

    # fp16：朴素 .half() 与内部 float32 算子不兼容，改用 autocast
    ok16 = False
    try:
        def f16():
            with torch.autocast("cuda", dtype=torch.float16):
                return m(pix512)
        ms16 = bench(f16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            o16 = m(pix512)
        s32 = m(pix512)[1][0].float()
        s16 = o16[1][0].float()

        def pool(x):
            g = x.reshape(32, 32, -1)
            return g.reshape(16, 2, 16, 2, -1).mean(dim=(1, 3)).reshape(256, -1)

        cos = torch.nn.functional.cosine_similarity(pool(s32), pool(s16), dim=-1).mean().item()
        print(f"[parts] fp16(autocast)-512 full={ms16:.1f}ms (加速 {ms32/ms16:.2f}x) | "
              f"fp16 vs fp32 池化特征 cos={cos:.5f}")
        ok16 = True
    except Exception as e:
        print(f"[parts] fp16-autocast 失败: {type(e).__name__}: {str(e)[:150]}")

    with torch.no_grad():
        o = m(pix512)
    print(f"[parts] summary {tuple(o[0].shape)}（2×CLS token 拼接，无独立参数） | "
          f"spatial {tuple(o[1].shape)}（全部 patch token，共享 431M 骨干）")
    print("[parts] DONE", flush=True)


if __name__ == "__main__":
    main()
