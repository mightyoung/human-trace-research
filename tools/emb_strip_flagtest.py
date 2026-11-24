# -*- coding: utf-8 -*-
"""emb_strip_flagtest.py — 死重剥离补丁（strip_dead 开关）副本验证。

验证目标（bench 副本 embed_fast_stripflag.py，不动生产 crosscam_run/embed_fast.py）：
  1. 默认路径（不剥离）→ 编码正常
  2. strip_dead_towers() 方法剥离 → 嵌入 vs 默认**逐位一致**
  3. FastEncoder(strip_dead=True) 构造期剥离 → 编码 vs 方法版**逐位一致**、
     且 audio_tower/vision_tower 为 None、embed_vision 保留
产物：bench_emb_strip_flagtest.json + emb_strip_flagtest_vectors.npy（16 条留档向量）
用法：~/yhh/human-count/embed_venv/bin/python emb_strip_flagtest.py --device cuda:1
"""
import argparse
import gc
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/r4900g3/yhh/human-count/bench")
from embed_fast_stripflag import FastEncoder, _make_crops  # noqa: E402

VEC_OUT = "/share/data4/yhh/data/human-count/emb_strip_flagtest_vectors.npy"


def bitstats(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return {"bitwise_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
            "max_abs": float(np.abs(a - b).max()),
            "n_diff": int((a != b).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/share/data4/yhh/data/human-count/bench_emb_strip_flagtest.json")
    args = ap.parse_args()
    dev = args.device
    res = {"start": time.strftime("%F %T")}

    crops = _make_crops(cam="318", n_tracks=8, per_track=8)
    res["n_crops"] = len(crops)
    print(f"[flagtest] crops={len(crops)}", flush=True)

    # 1) 默认（不剥离）
    enc = FastEncoder(device=dev, w_dir="cradio_scale_v4")
    e_off = enc.encode(crops)
    res["mem_after_default"] = round(torch.cuda.memory_allocated() / 1e9, 3)

    # 2) 方法剥离
    removed = enc.strip_dead_towers()
    e_on = enc.encode(crops)
    res["helper_strip"] = {"removed": removed,
                           "mem_after_strip": round(torch.cuda.memory_allocated() / 1e9, 3),
                           "crops": bitstats(e_off, e_on)}
    print(f"[flagtest] helper: removed={removed} equal={res['helper_strip']['crops']['bitwise_equal']}",
          flush=True)
    np.save(VEC_OUT, e_off[:16])
    res["vectors_saved"] = VEC_OUT

    # 3) 构造期剥离（新进程级路径：重新加载）
    del enc
    gc.collect()
    torch.cuda.empty_cache()
    enc2 = FastEncoder(device=dev, w_dir="cradio_scale_v4", strip_dead=True)
    e_init = enc2.encode(crops)
    res["init_strip"] = {
        "audio_tower_is_none": getattr(enc2.am, "audio_tower", "absent") is None,
        "vision_tower_is_none": getattr(enc2.am, "vision_tower", "absent") is None,
        "embed_vision_present": getattr(enc2.am, "embed_vision", None) is not None,
        "crops_vs_helper": bitstats(e_on, e_init),
        "mem": round(torch.cuda.memory_allocated() / 1e9, 3),
    }
    print(f"[flagtest] init: equal_vs_helper={res['init_strip']['crops_vs_helper']['bitwise_equal']}",
          flush=True)

    res["end"] = time.strftime("%F %T")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("[out]", args.out, flush=True)
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
