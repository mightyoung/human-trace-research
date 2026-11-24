# -*- coding: utf-8 -*-
"""emb_quant_probe.py — EmbeddingGemma2（语言塔）量化可行性实测。

Parts:
  A. 模型解剖：各子塔参数量/显存分布；audio/vision tower 是否死重
  B. 基线分段计时：RADIO 桥 / embed_vision / 语言塔 forward 各占多少
     （由此推导：量化语言塔的收益上限）
  C. fp16 autocast 复验：速度/显存/嵌入 cos
  D. 死重剥离：把未使用的 audio/vision tower 移出显存 → 省多少 GB + 功能不受影响验证

用法（远程 GPU0）：
  ~/yhh/human-count/embed_venv/bin/python emb_quant_probe.py --out emb_quant_probe.json
"""
import argparse
import gc
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/r4900g3/yhh/human-count/crosscam_run")
from embed_fast import FastEncoder, _make_crops  # noqa: E402


def cos_rows(a, b):
    """a/b 均已 L2 归一 → 行点积即 cos。"""
    return float((a * b).sum(1).mean()), float((a * b).sum(1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="emb_quant_probe.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--w-dir", default="cradio_scale_v4")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    res = {"device": args.device, "torch": torch.__version__}

    print("[load] FastEncoder ...", flush=True)
    t0 = time.perf_counter()
    enc = FastEncoder(device=args.device, w_dir=args.w_dir)
    torch.cuda.synchronize(dev)
    res["load_sec"] = round(time.perf_counter() - t0, 1)
    am = enc.am
    print(f"[load] done in {res['load_sec']}s", flush=True)

    # ---------- A. 模型解剖 ----------
    print("[A] anatomy ...", flush=True)
    children = [n for n, _ in am.named_children()]
    print(f"[A] top children: {children}", flush=True)
    stats = {"children": children, "params_M": {}, "mem_gb_loaded": round(
        torch.cuda.memory_allocated(dev) / 1e9, 2)}
    used = 0
    for cname in ["language_model", "vision_tower", "audio_tower",
                  "embed_vision", "embed_audio", "embed_text"]:
        sub = getattr(am, cname, None)
        if sub is None:
            continue
        n = sum(p.numel() for p in sub.parameters())
        stats["params_M"][cname] = round(n / 1e6, 1)
    stats["params_M"]["TOTAL_am"] = round(sum(p.numel() for p in am.parameters()) / 1e6, 1)
    stats["params_M"]["m_radio"] = round(sum(p.numel() for p in enc.m_radio.parameters()) / 1e6, 1)
    print(f"[A] {json.dumps(stats['params_M'])}", flush=True)
    res["anatomy"] = stats

    # ---------- crops ----------
    crops = _make_crops("316", n_tracks=10, per_track=8)
    crops += _make_crops("318", n_tracks=6, per_track=8)
    print(f"[data] crops={len(crops)}", flush=True)
    res["n_crops"] = len(crops)

    def timeit(fn, rounds=20, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(rounds):
            fn()
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / rounds * 1000.0  # ms

    # ---------- B. 基线 + 分段计时 ----------
    print("[B] baseline ...", flush=True)
    t0 = time.perf_counter()
    embs_base = enc.encode(crops, batch=16)
    torch.cuda.synchronize(dev)
    res["e2e_ms_per_img"] = round((time.perf_counter() - t0) / len(crops) * 1000, 2)
    res["e2e_npz_mean"] = round(float(np.abs(embs_base).mean()), 4)
    print(f"[B] e2e {res['e2e_ms_per_img']} ms/img", flush=True)

    B = 16
    pix = enc._prep(crops[:B], "rproc").to(dev)
    enc._pix = pix
    t_radio = timeit(lambda: enc._radio_bridge(pix), rounds=20)
    # 重算 z2（与桥接内部一致的中间量）
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = enc.m_radio(pix)
        sp = out[1]
        b, l, dim = sp.shape
        g = sp.reshape(b, 32, 32, dim)
        pp = g.reshape(b, 16, 2, 16, 2, dim).mean(dim=(2, 4)).reshape(b, 256, dim)
    z2 = (pp.reshape(b * 256, dim).float() @ enc.Wt)
    t_ev = timeit(lambda: enc.am.embed_vision(inputs_embeds=z2), rounds=50)

    ids = enc.ids.expand(B, -1)
    mask = enc.mask.expand(B, -1)

    def tower_fwd():
        with torch.no_grad():
            return enc.am(input_ids=ids, attention_mask=mask,
                          pixel_values=torch.zeros(B, 1, 1, device=dev),
                          image_position_ids=None, return_dict=True)

    t_tower = timeit(tower_fwd, rounds=20)
    res["breakdown_ms_per_batch16"] = {
        "radio_bridge": round(t_radio, 2),
        "embed_vision": round(t_ev, 2),
        "language_tower_fwd": round(t_tower, 2),
        "sum_of_three": round(t_radio + t_ev + t_tower, 2),
    }
    print(f"[B] {res['breakdown_ms_per_batch16']}", flush=True)
    res["tower_share_of_e2e"] = round(t_tower / (res["e2e_ms_per_img"] * B), 3)

    # ---------- C. fp16 autocast 复验 ----------
    print("[C] fp16 autocast ...", flush=True)
    try:
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            embs_fp16 = enc.encode(crops, batch=16)
        torch.cuda.synchronize(dev)
        ms = (time.perf_counter() - t0) / len(crops) * 1000
        mean_c, min_c = cos_rows(embs_fp16, embs_base)
        res["fp16"] = {"ms_per_img": round(ms, 2), "cos_mean": round(mean_c, 6),
                       "cos_min": round(min_c, 6)}
        print(f"[C] {res['fp16']}", flush=True)
    except Exception as e:
        res["fp16"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"[C] FAILED: {res['fp16']['error']}", flush=True)

    # ---------- D. 死重剥离 ----------
    print("[D] strip dead towers ...", flush=True)
    mem_before = torch.cuda.memory_allocated(dev) / 1e9
    moved = []
    for cname in ["audio_tower", "vision_tower"]:
        sub = getattr(am, cname, None)
        if sub is not None:
            sub.to("meta")
            moved.append(cname)
    gc.collect()
    torch.cuda.empty_cache()
    mem_after = torch.cuda.memory_allocated(dev) / 1e9
    res["strip"] = {"moved": moved, "mem_before_gb": round(mem_before, 2),
                    "mem_after_gb": round(mem_after, 2),
                    "saved_gb": round(mem_before - mem_after, 2)}
    print(f"[D] saved {res['strip']['saved_gb']} GB", flush=True)

    # 功能验证：剥离后重编码 32 张，与基线 cos 必须 ≈1.0
    try:
        embs_strip = enc.encode(crops[:32], batch=16)
        mean_c, min_c = cos_rows(embs_strip, embs_base[:32])
        res["strip"]["cos_mean"] = round(mean_c, 6)
        res["strip"]["cos_min"] = round(min_c, 6)
        print(f"[D] cos mean {mean_c:.6f} min {min_c:.6f}", flush=True)
    except Exception as e:
        res["strip"]["verify_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[D] verify FAILED: {res['strip']['verify_error']}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
