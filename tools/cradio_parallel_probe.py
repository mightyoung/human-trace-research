# -*- coding: utf-8 -*-
"""cradio_v3 链路并行化探针 —— 分解计时 + 批处理/精度扩展性基准。

测量：
  A. CPU 前处理：rebuild_image（现状链路中 Gemma processor→重建 的往返开销）
  B. RADIO 桥接段批处理扩展：CLIPImageProcessor → RADIO(512) → 2x2池化 → W → embed_vision
     batch ∈ {1,8,16,32,64} × {fp32, fp16}；含 GPU 利用率采样
  C. 现状 E2E 复现：逐图 st.encode（patched 链路）→ 现状 s/图基线
  差值 C-B(bs=1) ≈ 重建往返 + 语言塔 + ST 封装
输出：cradio_parallel_probe.json
"""
import os
import json
import time
import threading
import subprocess

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from sentence_transformers import SentenceTransformer

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
RADIO_DIR = os.path.join(ROOT, "cradio", "C-RADIOv4-SO400M")
EMB_PATH = os.path.join(ROOT, "embgemma", "model", "master")
W_P = os.path.join(ROOT, "cradio_scale_v3", "radio_W_scale.npy")
DATA = "/share/data4/yhh/data/human-count"
DEV = os.environ.get("FUSION_DEV", "cuda:1")
OUT_P = os.path.join(ROOT, "cradio_parallel_probe.json")


def log(*a):
    print(*a, flush=True)


# ---------------- 数据 ----------------
def prep_crop(frame, box, out_h=224, pad=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad)); Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad)); Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
        return None
    s = out_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, out_h))
    if nw >= out_h:
        off = (nw - out_h) // 2
        c = c[:, off:off + out_h]
    else:
        padw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def load_tracks(jsonl_path):
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((int(r["frame"]), b))
    return t


def make_crops(cam="316", n_tracks=8, per_track=8):
    tracks = load_tracks(f"{DATA}/inference-out/{cam}_detections.jsonl")
    cap = cv2.VideoCapture(f"{DATA}/video-6way/{cam}.mp4")
    crops = []
    for tid in sorted(tracks)[:n_tracks * 2]:
        if len(crops) >= n_tracks * per_track:
            break
        rows = tracks[tid]
        if len(rows) < 4:
            continue
        idx = np.linspace(0, len(rows) - 1, min(per_track, len(rows))).astype(int)
        for j in idx:
            fr, box = rows[j]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, frame = cap.read()
            if not ok:
                continue
            c = prep_crop(frame, box)
            if c is not None:
                crops.append(c)
    cap.release()
    return crops[:n_tracks * per_track]


# ---------------- 模型 ----------------
def load_radio():
    m = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True).to(DEV).eval()
    proc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    return m, proc


def rebuild_image(pixel_values, position_ids):
    pv = pixel_values[0]
    ids = position_ids[0]
    valid = (ids != -1).all(dim=-1)
    pv = pv[valid]; ids = ids[valid]
    xs, ys = ids[:, 0], ids[:, 1]
    gw = int(xs.max()) + 1; gh = int(ys.max()) + 1
    order = torch.argsort(ys * gw + xs)
    patches = pv[order].reshape(gh, gw, 16, 16, 3)
    img = patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)
    return img


def gpu_util_loop(stop_evt, out_list, dev_idx):
    while not stop_evt.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader",
                 "-i", str(dev_idx)], capture_output=True, text=True, timeout=5)
            out_list.append(int(r.stdout.strip().split()[0]))
        except Exception:
            pass
        time.sleep(0.4)


def main():
    res = {}
    log("[1] 采真实 crop ...")
    crops = make_crops()
    log(f"    crops: {len(crops)} 张 ({crops[0].shape})")
    res["n_crops"] = len(crops)

    log("[2] 加载模型 ...")
    m_radio, rproc = load_radio()
    st = SentenceTransformer(EMB_PATH, device=DEV, model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    Wt = torch.from_numpy(np.load(W_P).astype(np.float32)).to(DEV)
    log("    模型就绪")

    # --- 捕获现状链路的 Gemma processor 产物（供 rebuild 计时） ---
    CAP = {}

    def patched(pixel_values, image_position_ids=None, **kwargs):
        if "pv" not in CAP:
            CAP["pv"] = pixel_values.detach().cpu().clone()
            CAP["ids"] = image_position_ids.detach().cpu().clone()
        img = rebuild_image(pixel_values.detach().cpu(), image_position_ids.detach().cpu())
        img_np = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
        pix = rproc(images=[Image.fromarray(img_np)], return_tensors="pt",
                    do_resize=True, do_center_crop=False, do_normalize=False).pixel_values.to(DEV)
        with torch.no_grad():
            out = m_radio(pix)
        sp = out[1][0]
        g = sp.reshape(32, 32, -1)
        p = g.reshape(16, 2, 16, 2, -1).mean(dim=(1, 3)).reshape(256, -1)
        z = (p @ Wt).to(DEV)
        pooler_output = am.embed_vision(inputs_embeds=z)
        output_length = image_position_ids.shape[1] // (am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z, pooler_output=torch.split(pooler_output, split_sizes))

    am.get_image_features = patched

    # --- C: 现状 E2E（逐图 st.encode）---
    log("[3] 现状 E2E：逐图 st.encode x16 ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for im in crops[:16]:
        st.encode({"image": im}, normalize_embeddings=True, show_progress_bar=False)
    torch.cuda.synchronize()
    e2e_s = (time.perf_counter() - t0) / 16
    log(f"    现状 E2E: {e2e_s*1000:.1f} ms/图")
    res["current_e2e_ms_per_img"] = round(e2e_s * 1000, 1)

    # --- A: rebuild_image CPU 计时 ---
    if "pv" in CAP:
        pv, ids = CAP["pv"], CAP["ids"]
        log(f"[4] rebuild_image: 输入 pv {tuple(pv.shape)} ids {tuple(ids.shape)}")
        for _ in range(3):
            rebuild_image(pv, ids)
        t0 = time.perf_counter()
        for _ in range(20):
            rebuild_image(pv, ids)
        rb_ms = (time.perf_counter() - t0) / 20 * 1000
        log(f"    rebuild_image: {rb_ms:.1f} ms/图")
        res["rebuild_ms_per_img"] = round(rb_ms, 1)
    else:
        log("[4] 未捕获到 processor 产物，跳过 rebuild 计时")

    # --- B: 批处理段（proc→RADIO→池化→W→embed_vision）---
    log("[5] 批处理段扩展性基准 ...")
    imgs = crops

    def run_batch(bs, fp16):
        t_batch = []
        for i in range(0, len(imgs), bs):
            chunk = [Image.fromarray(x) for x in imgs[i:i + bs]]
            ta = time.perf_counter()
            pix = rproc(images=chunk, return_tensors="pt", do_resize=True,
                        do_center_crop=False, do_normalize=False).pixel_values.to(DEV)
            tb = time.perf_counter()
            with torch.no_grad():
                if fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = m_radio(pix)
                        sp = out[1]
                        B, L, D = sp.shape
                        g = sp.reshape(B, 32, 32, D)
                        pp = g.reshape(B, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B, 256, D)
                    z = (pp.reshape(B * 256, D) @ Wt.half()).float().reshape(B, 256, -1)
                    po = am.embed_vision(inputs_embeds=z)
                else:
                    out = m_radio(pix)
                    sp = out[1]
                    B, L, D = sp.shape
                    g = sp.reshape(B, 32, 32, D)
                    pp = g.reshape(B, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B, 256, D)
                    z = (pp.reshape(B * 256, D) @ Wt).reshape(B, 256, -1)
                    po = am.embed_vision(inputs_embeds=z)
            torch.cuda.synchronize()
            tc = time.perf_counter()
            t_batch.append((tb - ta, tc - tb))
        return t_batch

    table = []
    for fp16 in [False, True]:
        for bs in [1, 8, 16, 32, 64]:
            # warmup 一遍
            run_batch(bs, fp16)
            torch.cuda.reset_peak_memory_stats(DEV)
            util = []
            stop_evt = threading.Event()
            dev_idx = int(DEV.split(":")[1]) if ":" in DEV else 0
            th = threading.Thread(target=gpu_util_loop, args=(stop_evt, util, dev_idx), daemon=True)
            th.start()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tb = run_batch(bs, fp16)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            stop_evt.set(); th.join(timeout=2)
            per_img = dt / len(imgs) * 1000
            thr = len(imgs) / dt
            peak = torch.cuda.max_memory_allocated(DEV) / 1e9
            cpu_t = sum(x[0] for x in tb) / len(imgs) * 1000
            gpu_t = sum(x[1] for x in tb) / len(imgs) * 1000
            row = dict(batch=bs, fp16=fp16, ms_per_img=round(per_img, 2),
                       throughput=round(thr, 1), peak_mem_gb=round(peak, 2),
                       cpu_proc_ms=round(cpu_t, 2), gpu_ms=round(gpu_t, 2),
                       gpu_util_mean=round(float(np.mean(util)), 1) if util else None,
                       gpu_util_max=int(np.max(util)) if util else None,
                       speedup_vs_current=round((e2e_s * 1000) / per_img, 1))
            table.append(row)
            log(f"    bs={bs:<3} fp16={int(fp16)}  {per_img:6.1f} ms/图  {thr:5.1f} 图/s  "
                f"mem={peak:.2f}GB  cpu={cpu_t:.1f}ms gpu={gpu_t:.1f}ms  "
                f"util={row['gpu_util_mean']}%  vs现状 {row['speedup_vs_current']}x")

    res["batch_table"] = table
    res["device"] = DEV
    with open(OUT_P, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"[DONE] -> {OUT_P}")


if __name__ == "__main__":
    main()
