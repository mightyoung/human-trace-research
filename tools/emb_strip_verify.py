# -*- coding: utf-8 -*-
"""emb_strip_verify.py — EmbeddingGemma2 死重剥离：逐位级验证 + 收益测量 + 机制对比。

背景：FastEncoder 桥接路径只用 [language_model + embed_vision]，不用
      [vision_tower + audio_tower + embed_audio]（472M 死重）。此前仅做过
      cos≈1.0（6 位小数、32 样本）的后置 meta 移动验证。本脚本做逐位级验证。

验证三机制（在生产 FastEncoder 路径上）：
  full    : 全量加载（对照）
  A_post  : 后置剥离（audio_tower/embed_audio/vision_tower → None）
  B_audio : 载入期剥离（config.audio_config=None，官方支持）＋后置剥离 vision_tower
  C_kwarg : 结构测试——model_kwargs 直传 audio_config=None 是否等价可行

对比样本：256 crops（316×128 + 318×128 真实轨迹裁剪）
          文本 = 生产词表（webapp/dataprep/build_text_vocab.py 逐字复制，~110 条）
                 + embgemma2_text_search.py 的 8 条 demo query
指标：逐位一致（uint32 视图）/ 精确相等 / max_abs / 差异计数 / cos_min；
      显存（allocated+reserved）、加载时间、编码速度；
      持久化：stripped ckpt 落盘（/share/data4）→ 重载逐位一致 + 磁盘/加载收益。

用法（远程 GPU1，与 GPU0 训练并行）：
  ~/yhh/human-count/embed_venv/bin/python emb_strip_verify.py \
      --device cuda:1 --out /share/data4/yhh/data/human-count/bench_emb_strip.json
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/r4900g3/yhh/human-count/crosscam_run")
from embed_fast import FastEncoder, _make_crops  # noqa: E402

EMB_PATH = "/home/r4900g3/yhh/human-count/crosscam_run/embgemma/model/master"
STRIP_DIR = "/share/data4/yhh/data/human-count/embgemma_stripped"

# ---------------- 生产文本短语（取自 webapp/dataprep/build_text_vocab.py，逐字复制） ----------------
COLORS = {"white": "白", "black": "黑", "gray": "灰", "red": "红", "blue": "蓝",
          "navy": "藏青", "green": "绿", "yellow": "黄", "brown": "棕",
          "beige": "米白", "pink": "粉", "orange": "橙", "purple": "紫"}
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
SPECIAL = {"fur_coat": ("a woman with a fur-collar coat", "毛皮领 皮领 毛领")}
GENERIC = "a person walking in an office building"

DEMO_QUERIES = [
    "a person carrying a backpack",
    "a woman wearing a hat",
    "a person wearing a white shirt",
    "a person mopping the floor cleaning",
    "a man wearing glasses and dark t-shirt",
    "a person wearing a navy blue jacket",
    "a person wearing a dark hoodie",
    "a woman with a fur-collar coat",
]


def build_phrases():
    phrases = {}
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


# ---------------- 工具 ----------------
def bitstats(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = np.abs(a - b)
    row = (a * b).sum(1) if a.ndim == 2 else (a * b).sum()
    return {
        "bitwise_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
        "exact_equal": bool(np.array_equal(a, b)),
        "max_abs": float(d.max()),
        "n_diff_elems": int((d != 0).sum()),
        "cos_min": float(np.min(row)),
        "cos_mean": float(np.mean(row)),
    }


def mem(dev):
    return {"alloc_gb": round(torch.cuda.memory_allocated(dev) / 1e9, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(dev) / 1e9, 3)}


def nparam(m):
    return int(sum(p.numel() for p in m.parameters()))


def strip_modules(am, names):
    removed = []
    for n in names:
        sub = getattr(am, n, None)
        if sub is not None:
            removed.append({"name": n, "params_M": round(nparam(sub) / 1e6, 1)})
            setattr(am, n, None)
            del sub
    gc.collect()
    torch.cuda.empty_cache()
    return removed


def text_encode(st, texts):
    return np.asarray(st.encode(texts, prompt_name="SearchQuery",
                                normalize_embeddings=True, batch_size=32,
                                show_progress_bar=False), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--w-dir", default="cradio_scale_v4")
    ap.add_argument("--out", default="emb_strip_verify.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    res = {"device": args.device, "w_dir": args.w_dir,
           "torch": torch.__version__, "start": time.strftime("%F %T")}

    # ---- 样本 ----
    print("[prep] crops ...", flush=True)
    crops = []
    for cam, n in [("316", 128), ("318", 128)]:
        try:
            c = _make_crops(cam=cam, n_tracks=n // 8, per_track=8)
            crops.extend(c)
            print(f"    cam{cam}: {len(c)}", flush=True)
        except Exception as e:
            print(f"    cam{cam} FAILED: {e}", flush=True)
    crops = crops[:256]
    res["n_crops"] = len(crops)
    phrases = build_phrases()
    texts = [phrases[k] for k in phrases] + DEMO_QUERIES
    res["n_texts"] = len(texts)
    print(f"[prep] crops={len(crops)} texts={len(texts)}", flush=True)
    assert len(crops) >= 64, "crops too few"

    # ================= 1. 全量（对照） =================
    print("[1] full load ...", flush=True)
    t0 = time.perf_counter()
    enc1 = FastEncoder(device=args.device, w_dir=args.w_dir)
    torch.cuda.synchronize(dev)
    res["full"] = {"load_sec": round(time.perf_counter() - t0, 1),
                   "mem_after_load": mem(dev)}
    am1 = enc1.am
    res["full"]["params_M"] = {n: round(nparam(getattr(am1, n)) / 1e6, 1)
                               for n in ["language_model", "vision_tower", "audio_tower",
                                         "embed_vision", "embed_audio"]
                               if getattr(am1, n, None) is not None}
    print(f"    loaded {res['full']['load_sec']}s params={res['full']['params_M']}", flush=True)

    print("[1] encode x2 (determinism) ...", flush=True)
    e1a = enc1.encode(crops)
    e1b = enc1.encode(crops)
    res["determinism_same_instance"] = bitstats(e1a, e1b)
    t0 = time.perf_counter()
    e1 = enc1.encode(crops)
    torch.cuda.synchronize(dev)
    res["full"]["ms_per_img"] = round((time.perf_counter() - t0) / len(crops) * 1000, 2)
    t1 = text_encode(enc1.st, texts)
    res["full"]["mem_after_encode"] = mem(dev)
    res["full"]["texts_sha"] = hashlib.sha1(t1.tobytes()).hexdigest()[:16]
    print(f"    {res['full']['ms_per_img']} ms/img  det={res['determinism_same_instance']}", flush=True)

    # ================= 2. A：后置剥离 =================
    print("[2] A_post strip ...", flush=True)
    removed = strip_modules(am1, ["audio_tower", "embed_audio", "vision_tower"])
    res["A_post"] = {"removed": removed, "mem_after_strip": mem(dev)}
    eA = enc1.encode(crops)
    tA = text_encode(enc1.st, texts)
    res["A_post"]["crops"] = bitstats(e1, eA)
    res["A_post"]["texts"] = bitstats(t1, tA)
    t0 = time.perf_counter()
    enc1.encode(crops)
    torch.cuda.synchronize(dev)
    res["A_post"]["ms_per_img"] = round((time.perf_counter() - t0) / len(crops) * 1000, 2)
    print(f"    strip {removed}; equal={res['A_post']['crops']['bitwise_equal']} "
          f"texts={res['A_post']['texts']['bitwise_equal']}", flush=True)

    del enc1
    del am1
    gc.collect()
    torch.cuda.empty_cache()
    res["mem_freed_after_del1"] = mem(dev)

    # ================= 3. B：载入期剥离 audio（config）+ 后置 vision =================
    print("[3] B_audio load-time strip ...", flush=True)
    import transformers
    import embed_fast as ef
    cfgB = transformers.AutoConfig.from_pretrained(EMB_PATH)
    cfgB.audio_config = None
    res["B_audio"] = {"mechanism": "model_kwargs={'config': cfg(audio_config=None)}"}
    _origST = ef.SentenceTransformer

    def _st_patched(path, **kw):
        mk = dict(kw.get("model_kwargs") or {})
        mk["config"] = cfgB
        kw["model_kwargs"] = mk
        return _origST(path, **kw)

    try:
        ef.SentenceTransformer = _st_patched
        t0 = time.perf_counter()
        enc2 = ef.FastEncoder(device=args.device, w_dir=args.w_dir)
        torch.cuda.synchronize(dev)
        res["B_audio"]["load_sec"] = round(time.perf_counter() - t0, 1)
        ef.SentenceTransformer = _origST
        am2 = enc2.am
        res["B_audio"]["audio_tower_is_none"] = getattr(am2, "audio_tower", "absent") is None
        res["B_audio"]["embed_vision_present"] = getattr(am2, "embed_vision", None) is not None
        res["B_audio"]["vision_tower_loaded"] = getattr(am2, "vision_tower", None) is not None
        strip_modules(am2, ["vision_tower"])
        res["B_audio"]["mem_after_strip"] = mem(dev)
        eB = enc2.encode(crops)
        tB = text_encode(enc2.st, texts)
        res["B_audio"]["crops_vs_full"] = bitstats(e1, eB)
        res["B_audio"]["texts_vs_full"] = bitstats(t1, tB)
        print(f"    audio_off={res['B_audio']['audio_tower_is_none']} "
              f"equal={res['B_audio']['crops_vs_full']['bitwise_equal']}", flush=True)
    except Exception as e:
        ef.SentenceTransformer = _origST
        res["B_audio"]["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        enc2 = None
        print(f"    B FAILED: {res['B_audio']['error']}", flush=True)

    # ================= 4. C：kwarg 直传结构测试（audio_config=None 直传） =================
    print("[4] C_kwarg structural test ...", flush=True)
    try:
        stC = _origST(EMB_PATH, device=str(dev),
                      model_kwargs={"torch_dtype": "float32", "audio_config": None})
        amC = getattr(stC[0], "auto_model", None) or getattr(stC[0], "model", None)
        res["C_kwarg"] = {
            "mechanism": "model_kwargs={'audio_config': None} 直传",
            "audio_tower_is_none": getattr(amC, "audio_tower", "absent") is None,
            "embed_vision_present": getattr(amC, "embed_vision", None) is not None,
            "texts_probe": bitstats(t1[:4], text_encode(stC, texts[:4])),
        }
        del stC, amC
        gc.collect()
        torch.cuda.empty_cache()
        print(f"    C audio_off={res['C_kwarg']['audio_tower_is_none']}", flush=True)
    except Exception as e:
        res["C_kwarg"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"    C FAILED: {res['C_kwarg']['error']}", flush=True)

    # ================= 5. 持久化：stripped ckpt 落盘 + 重载 =================
    print("[5] persistence ...", flush=True)
    try:
        assert enc2 is not None
        am2 = enc2.am
        os.makedirs(STRIP_DIR, exist_ok=True)
        # 复制非权重文件
        for fn in os.listdir(EMB_PATH):
            if fn.endswith((".safetensors", ".bin")):
                continue
            src = os.path.join(EMB_PATH, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(STRIP_DIR, fn))
        # config：audio_config=null
        cfgS = transformers.AutoConfig.from_pretrained(EMB_PATH)
        cfgS.audio_config = None
        cfgS.save_pretrained(STRIP_DIR)
        # 权重：剔除 vision_tower./audio_tower./embed_audio.
        sd = am2.state_dict()
        drop_pref = ("vision_tower.", "audio_tower.", "embed_audio.")
        kept = {k: v for k, v in sd.items() if not k.startswith(drop_pref)}
        dropped = len(sd) - len(kept)
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in kept.items()},
                  os.path.join(STRIP_DIR, "model.safetensors"))
        sz_dir = sum(os.path.getsize(os.path.join(STRIP_DIR, f))
                     for f in os.listdir(STRIP_DIR))
        sz_src = sum(os.path.getsize(os.path.join(EMB_PATH, f))
                     for f in os.listdir(EMB_PATH))
        res["persist"] = {"dir": STRIP_DIR, "dropped_tensors": dropped,
                          "kept_tensors": len(kept),
                          "size_MB": round(sz_dir / 1e6, 1),
                          "src_size_MB": round(sz_src / 1e6, 1)}
        # 重载验证
        del enc2, am2
        gc.collect()
        torch.cuda.empty_cache()
        stR = _origST(STRIP_DIR, device=str(dev), model_kwargs={"torch_dtype": "float32"})
        amR = getattr(stR[0], "auto_model", None) or getattr(stR[0], "model", None)
        res["persist"]["reload_audio_tower_is_none"] = getattr(amR, "audio_tower", "absent") is None
        res["persist"]["reload_vision_tower_present"] = getattr(amR, "vision_tower", None) is not None
        strip_modules(amR, ["vision_tower"])
        tR = text_encode(stR, texts)
        res["persist"]["texts_vs_full"] = bitstats(t1, tR)
        print(f"    size {res['persist']['size_MB']}MB (src {res['persist']['src_size_MB']}MB) "
              f"texts_equal={res['persist']['texts_vs_full']['bitwise_equal']}", flush=True)
        del stR, amR
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        res["persist"] = {**res.get("persist", {}),
                          "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"    persist FAILED: {res['persist']['error']}", flush=True)

    res["end"] = time.strftime("%F %T")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[out] {args.out}", flush=True)
    print(json.dumps({k: res[k] for k in res if k not in ("device", "w_dir", "torch")},
                     ensure_ascii=False, indent=1)[:1200], flush=True)


if __name__ == "__main__":
    main()
