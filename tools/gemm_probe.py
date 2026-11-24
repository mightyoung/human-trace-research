# -*- coding: utf-8 -*-
"""gemm_probe.py — 语言塔典型 GEMM 形状的 fp32/fp16/fp8 微基准（量化收益先导探测）。

回答的问题：EmbeddingGemma2 语言塔的矩阵乘法（hidden 512 / inter 2048 / seq 260）
在 fp8 量化下能否比 fp16 更快？若不能（小形状 = launch/访存受限），则"量化替换"
对吞吐无实质收益。

用法（远程）：
  ~/yhh/human-count/embed_venv/bin/python gemm_probe.py --out gemm_probe.json
"""
import json
import time

import torch


def bench(fn, rounds=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rounds):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rounds * 1e6  # us


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gemm_probe.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    print(f"[env] torch {torch.__version__} | {torch.cuda.get_device_name(dev)}", flush=True)

    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev), "rows": []}

    # ---- fp8 API 探测 ----
    has_scaled_mm = hasattr(torch, "_scaled_mm")
    print(f"[api] torch._scaled_mm present: {has_scaled_mm}", flush=True)
    fp8_ok = False
    if has_scaled_mm:
        try:
            a = torch.randn(64, 512, device=dev).to(torch.float8_e4m3fn)
            b = torch.randn(512, 512, device=dev).to(torch.float8_e4m3fn)
            sa = torch.tensor([1.0], device=dev)
            sb = torch.tensor([1.0], device=dev)
            out = torch._scaled_mm(a, b, sa, sb, out_dtype=torch.float16)
            torch.cuda.synchronize()
            print(f"[api] fp8 mm OK, out shape {tuple(out.shape)} dtype {out.dtype}", flush=True)
            fp8_ok = True
        except Exception as e:
            print(f"[api] fp8 mm FAILED: {type(e).__name__}: {e}", flush=True)
            res["fp8_error"] = f"{type(e).__name__}: {str(e)[:300]}"

    # ---- 形状列表：batch=1（单图 260 token）与 batch=16（4160 token）----
    shapes = [
        ("attn_qkv", [(260, 512, 512), (4160, 512, 512)]),
        ("attn_o",   [(260, 512, 512), (4160, 512, 512)]),
        ("ffn_up",   [(260, 512, 2048), (4160, 512, 2048)]),
        ("ffn_dn",   [(260, 2048, 512), (4160, 2048, 512)]),
        ("embed_vision", [(256, 768, 512), (4096, 768, 512)]),
    ]

    for name, pairs in shapes:
        for (M, K, N) in pairs:
            a = torch.randn(M, K, device=dev)
            b = torch.randn(K, N, device=dev)
            row = {"shape": name, "M": M, "K": K, "N": N}

            row["fp32_us"] = round(bench(lambda: a @ b), 1)
            ah, bh = a.half(), b.half()
            row["fp16_us"] = round(bench(lambda: ah @ bh), 1)

            if fp8_ok:
                try:
                    a8 = a.to(torch.float8_e4m3fn)
                    b8 = b.to(torch.float8_e4m3fn)
                    bt = b8.t().contiguous().t()  # 保证行主序 [K,N]
                    sa = torch.tensor([1.0], device=dev)
                    sb = torch.tensor([1.0], device=dev)
                    row["fp8_us"] = round(bench(
                        lambda: torch._scaled_mm(a8, bt, sa, sb, out_dtype=torch.float16)), 1)
                    row["fp16_vs_fp8_x"] = round(row["fp16_us"] / max(row["fp8_us"], 0.1), 2)
                except Exception as e:
                    row["fp8_error"] = f"{type(e).__name__}: {str(e)[:150]}"

            res["rows"].append(row)
            print(f"[gemm] {row}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
