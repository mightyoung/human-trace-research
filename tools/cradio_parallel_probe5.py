# -*- coding: utf-8 -*-
"""probe5: 批量改造原型验证 —— patched 支持 batch → st.encode(list, batch_size) 全链路 batch。

测量：
  1) 现状口径：16 图逐图 st.encode → ms/图
  2) batch 口径：64 图 st.encode(list, batch_size ∈ {8,16,32,64}) → ms/图
  3) 一致性：单图嵌入 vs batch 嵌入 的 cos
  4) 打印 batch 下 patched 收到的 pixel_values 形状（确认组装方式）
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
OUT_P = os.path.join(ROOT, "cradio_parallel_probe5.json")


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
    pv = pixel_values[0]; ids = position_ids[0]
    valid = (ids != -1).all(dim=-1)
    pv = pv[valid]; ids = ids[valid]
    xs, ys = ids[:, 0], ids[:, 1]
    gw = int(xs.max()) + 1; gh = int(ys.max()) + 1
    order = torch.argsort(ys * gw + xs)
    patches = pv[order].reshape(gh, gw, 16, 16, 3)
    return patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)


def main():
    res = {}
    log("[1] crops ...")
    crops = make_crops()
    log(f"    {len(crops)}")
    log("[2] load ...")
    m_radio = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True).to(DEV).eval()
    rproc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    st = SentenceTransformer(EMB_PATH, device=DEV, model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    Wt = torch.from_numpy(np.load(W_P).astype(np.float32)).to(DEV)
    log("    就绪")

    SHAPES = []

    def patched(pixel_values, image_position_ids=None, **kwargs):
        pv = pixel_values.detach().cpu()
        ids = image_position_ids.detach().cpu()
        SHAPES.append((tuple(pv.shape), tuple(ids.shape)))
        B = pv.shape[0]
        imgs = []
        for b in range(B):
            img = rebuild_image(pv[b:b + 1], ids[b:b + 1])
            imgs.append((img.numpy() * 255).clip(0, 255).astype(np.uint8))
        pix = rproc(images=[Image.fromarray(x) for x in imgs], return_tensors="pt",
                    do_resize=True, do_center_crop=False, do_normalize=False).pixel_values.to(DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = m_radio(pix)
            sp = out[1]
            B2, L, D = sp.shape
            g = sp.reshape(B2, 32, 32, D)
            pp = g.reshape(B2, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B2, 256, D)
        # 平铺格式（与单图版行为一致）：[B2*256, 768]
        z2 = (pp.reshape(B2 * 256, D).float() @ Wt)
        po = am.embed_vision(inputs_embeds=z2)          # [B2*256, 512]
        per = po.shape[0] // B2
        parts = list(torch.split(po, [per] * B2))
        return BaseModelOutputWithPooling(
            last_hidden_state=z2, pooler_output=parts)

    am.get_image_features = patched

    # 一致性锚点：单图逐图（前 8）
    log("[3] 单图口径 x8 ...")
    e_single = []
    t0 = time.perf_counter()
    for im in crops[:8]:
        e = np.asarray(st.encode({"image": im}, normalize_embeddings=True,
                                 show_progress_bar=False)).reshape(-1)
        e_single.append(e)
    t_single = (time.perf_counter() - t0) / 8 * 1000
    log(f"    单图: {t_single:.1f} ms/图   shapes(head)={SHAPES[:1]}")
    res["single_ms_per_img"] = round(t_single, 1)
    res["single_shapes"] = [list(s) for s in SHAPES[:2]]
    SHAPES.clear()

    # batch 口径
    for bs in [8, 16, 32, 64]:
        try:
            st.encode(crops[:bs], normalize_embeddings=True, batch_size=bs,
                      show_progress_bar=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = st.encode(crops[:bs], normalize_embeddings=True, batch_size=bs,
                            show_progress_bar=False)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            per_img = dt / bs * 1000
            log(f"    batch={bs:<3} {per_img:7.2f} ms/图  ({bs/dt:5.1f} 图/s)  "
                f"shapes={SHAPES[-1] if SHAPES else None}")
            res[f"batch{bs}_ms_per_img"] = round(per_img, 2)
            res[f"batch{bs}_shapes"] = [list(s) for s in SHAPES[-2:]] if SHAPES else []
            if bs == 8:
                out8 = np.asarray(out)
                coss = [float(np.dot(out8[i], e_single[i])) for i in range(8)]
                log(f"    batch8 vs single cos: mean={np.mean(coss):.4f} min={np.min(coss):.4f}")
                res["batch8_cos_mean"] = round(float(np.mean(coss)), 4)
                res["batch8_cos_min"] = round(float(np.min(coss)), 4)
        except Exception as ex:
            import traceback
            log(f"    batch={bs} 失败: {type(ex).__name__}: {str(ex)[:200]}")
            log(traceback.format_exc()[:800])
            res[f"batch{bs}_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"

    with open(OUT_P, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"[DONE] -> {OUT_P}")
    log(json.dumps(res, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
