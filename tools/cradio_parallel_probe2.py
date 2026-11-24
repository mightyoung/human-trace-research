# -*- coding: utf-8 -*-
"""cradio_v3 并行化探针 v2：fp16 补测 + 快速预处理 + 路径一致性 + embed_vision 必要性。

输出：
  [B] fp16 批处理段（修复 matmul dtype）
  [D] cv2 快速预处理 vs CLIPImageProcessor（耗时 + 数值对齐）
  [E] 一致性：Path-A（现状 st.encode） vs Path-B（跳过 Gemma processor/rebuild 的直通)
  [F] embed_vision 是否被消费（去调用后 st.encode 输出是否变化）
  [G] st.encode 输出维度与耗时
"""
import os
import json
import time

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
OUT_P = os.path.join(ROOT, "cradio_parallel_probe2.json")


def log(*a):
    print(*a, flush=True)


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


def rebuild_image(pixel_values, position_ids):
    pv = pixel_values[0]
    ids = position_ids[0]
    valid = (ids != -1).all(dim=-1)
    pv = pv[valid]; ids = ids[valid]
    xs, ys = ids[:, 0], ids[:, 1]
    gw = int(xs.max()) + 1; gh = int(ys.max()) + 1
    order = torch.argsort(ys * gw + xs)
    patches = pv[order].reshape(gh, gw, 16, 16, 3)
    return patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)


def fast_preproc(img, size=512):
    """cv2 快速预处理：HxWx3 uint8 -> [3,size,size] float32（对齐 CLIP 语义待验）"""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous().float()


def main():
    res = {}
    log("[1] crops ...")
    crops = make_crops()
    log(f"    {len(crops)} crops")
    m_radio, rproc = None, None
    log("[2] load models ...")
    m_radio = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True).to(DEV).eval()
    rproc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    st = SentenceTransformer(EMB_PATH, device=DEV, model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    Wt = torch.from_numpy(np.load(W_P).astype(np.float32)).to(DEV)
    log("    就绪")

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

    # [G] st.encode 输出维度 + 基线
    log("[G] st.encode 基线（8 图）...")
    embs_st = []
    t0 = time.perf_counter()
    for im in crops[:8]:
        e = np.asarray(st.encode({"image": im}, normalize_embeddings=True, show_progress_bar=False)).reshape(-1)
        embs_st.append(e)
    dt = (time.perf_counter() - t0) / 8
    log(f"    dim={embs_st[0].shape[0]}  {dt*1000:.1f} ms/图")
    res["st_dim"] = int(embs_st[0].shape[0])
    res["st_ms_per_img"] = round(dt * 1000, 1)

    # [F] embed_vision 是否被消费：换成不调 embed_vision 的版本
    def patched_noev(pixel_values, image_position_ids=None, **kwargs):
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
        return BaseModelOutputWithPooling(last_hidden_state=z, pooler_output=None)

    log("[F] embed_vision 必要性探测 ...")
    try:
        am.get_image_features = patched_noev
        e2 = np.asarray(st.encode({"image": crops[0]}, normalize_embeddings=True,
                                  show_progress_bar=False)).reshape(-1)
        cos = float(np.dot(e2, embs_st[0]) / (np.linalg.norm(e2) * np.linalg.norm(embs_st[0]) + 1e-12))
        log(f"    noEV 输出 cos={cos:.6f}  (dim={e2.shape[0]})")
        res["embed_vision_consumed"] = bool(cos < 0.999)
        res["noEV_cos"] = round(cos, 6)
    except Exception as ex:
        log(f"    noEV 失败: {type(ex).__name__}: {str(ex)[:120]}")
        res["embed_vision_consumed"] = True
        res["noEV_error"] = str(ex)[:200]

    # 恢复
    am.get_image_features = patched

    # [D] 预处理对比
    log("[D] 预处理对比 ...")
    ref = rproc(images=[Image.fromarray(crops[0])], return_tensors="pt",
                do_resize=True, do_center_crop=False, do_normalize=False).pixel_values[0]
    fast0 = fast_preproc(crops[0])
    log(f"    ref: {tuple(ref.shape)} min={ref.min():.2f} max={ref.max():.2f} mean={ref.mean():.2f}")
    log(f"    fast maxdiff={float((ref-fast0).abs().max()):.2f}")

    for _ in range(3):
        fast_preproc(crops[0])
    t0 = time.perf_counter()
    for im in crops[:32]:
        fast_preproc(im)
    fast_ms = (time.perf_counter() - t0) / 32 * 1000
    t0 = time.perf_counter()
    for im in crops[:8]:
        rproc(images=[Image.fromarray(im)], return_tensors="pt",
              do_resize=True, do_center_crop=False, do_normalize=False)
    slow_ms = (time.perf_counter() - t0) / 8 * 1000
    log(f"    cv2-fast: {fast_ms:.2f} ms/图 | CLIPProcessor 单图: {slow_ms:.1f} ms/图")
    res["fast_preproc_ms"] = round(fast_ms, 2)
    res["clip_proc_single_ms"] = round(slow_ms, 1)

    # [E] 路径一致性：Path-B（跳过 Gemma processor/rebuild 直通）
    log("[E] 一致性验证（Path-B 直通 vs Path-A 现状）...")
    cos_list = []
    for i, im in enumerate(crops[:8]):
        pix = fast_preproc(im).unsqueeze(0).to(DEV)
        with torch.no_grad():
            out = m_radio(pix)
        sp = out[1][0]
        g = sp.reshape(32, 32, -1)
        p = g.reshape(16, 2, 16, 2, -1).mean(dim=(1, 3)).reshape(256, -1)
        z = (p @ Wt).float()
        e = z.mean(dim=0).cpu().numpy()
        e = e / (np.linalg.norm(e) + 1e-12)
        c = float(np.dot(e, embs_st[i]))
        cos_list.append(c)
        log(f"    img{i}: cos={c:.4f}")
    res["pathB_cos_mean"] = round(float(np.mean(cos_list)), 4)
    res["pathB_cos_min"] = round(float(np.min(cos_list)), 4)

    # [B] fp16 批处理段（修复版）
    log("[B] 批处理段 fp16/混合精度补测 ...")
    imgs = crops

    def run_batch(bs, mode):
        for i in range(0, len(imgs), bs):
            chunk = [Image.fromarray(x) for x in imgs[i:i + bs]]
            pix = rproc(images=chunk, return_tensors="pt", do_resize=True,
                        do_center_crop=False, do_normalize=False).pixel_values.to(DEV)
            with torch.no_grad():
                if mode == "fp32":
                    out = m_radio(pix)
                    sp = out[1]
                else:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = m_radio(pix)
                    sp = out[1]
                B, L, D = sp.shape
                g = sp.reshape(B, 32, 32, D).float()
                pp = g.reshape(B, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B, 256, D)
                z = (pp.reshape(B * 256, D) @ Wt).reshape(B, 256, -1)
            torch.cuda.synchronize()
        return

    table = []
    for mode in ["fp32", "fp16"]:
        for bs in [8, 16, 32, 64]:
            run_batch(bs, mode)
            torch.cuda.reset_peak_memory_stats(DEV)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_batch(bs, mode)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            per_img = dt / len(imgs) * 1000
            row = dict(mode=mode, batch=bs, ms_per_img=round(per_img, 2),
                       throughput=round(len(imgs) / dt, 1),
                       peak_mem_gb=round(torch.cuda.max_memory_allocated(DEV) / 1e9, 2))
            table.append(row)
            log(f"    {mode} bs={bs:<3} {per_img:6.2f} ms/图  {row['throughput']:5.1f} 图/s  mem={row['peak_mem_gb']:.2f}GB")
    res["batch_table_v2"] = table

    with open(OUT_P, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"[DONE] -> {OUT_P}")


if __name__ == "__main__":
    main()
