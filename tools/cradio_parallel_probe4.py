# -*- coding: utf-8 -*-
"""probe4: hook 追踪 st.encode 模块级 IO，找出 768 维嵌入的真实构成。"""
import os
import json

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


def log(*a):
    print(*a, flush=True)


def prep_crop(frame, box, out_h=224, pad=0.08):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad)); Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad)); Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
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


def rebuild_image(pixel_values, position_ids):
    pv = pixel_values[0]; ids = position_ids[0]
    valid = (ids != -1).all(dim=-1)
    pv = pv[valid]; ids = ids[valid]
    xs, ys = ids[:, 0], ids[:, 1]
    gw = int(xs.max()) + 1; gh = int(ys.max()) + 1
    order = torch.argsort(ys * gw + xs)
    patches = pv[order].reshape(gh, gw, 16, 16, 3)
    return patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)


def desc(x, d=0):
    if d > 4:
        return "..."
    if torch.is_tensor(x):
        xf = x.float()
        return (f"T{tuple(x.shape)} m={xf.mean():.3f} s={xf.std():.3f} "
                f"n={float(xf.norm()):.1f} dt={x.dtype}")
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {desc(v, d + 1)}" for k, v in list(x.items())[:6]) + "}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(str(desc(v, d + 1)) for v in x[:2]) + f" |n={len(x)}]"
    return f"{type(x).__name__}({str(x)[:50]})"


def main():
    log("[1] crop ...")
    tracks = {}
    with open(f"{DATA}/inference-out/316_detections.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for i, b in enumerate(r.get("boxes", []) or []):
                tid = (r.get("track_ids") or [None])[i]
                if tid is not None and tid >= 0:
                    tracks.setdefault(int(tid), []).append((int(r["frame"]), b))
    tid = sorted(tracks)[0]
    rows = tracks[tid]
    cap = cv2.VideoCapture(f"{DATA}/video-6way/316.mp4")
    fr, box = rows[len(rows) // 2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
    ok, frame = cap.read()
    cap.release()
    im = prep_crop(frame, box)
    log(f"    crop ok {im.shape}")

    log("[2] load ...")
    m_radio = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True).to(DEV).eval()
    rproc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    st = SentenceTransformer(EMB_PATH, device=DEV, model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    Wt = torch.from_numpy(np.load(W_P).astype(np.float32)).to(DEV)
    log(f"    st modules: {[type(m).__name__ for m in st]}")

    CAP = {}

    def patched(pixel_values, image_position_ids=None, **kwargs):
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
        CAP["z"] = z.detach()
        CAP["po"] = pooler_output.detach()
        CAP["split_sizes"] = split_sizes
        return BaseModelOutputWithPooling(
            last_hidden_state=z, pooler_output=torch.split(pooler_output, split_sizes))

    am.get_image_features = patched

    # hooks
    for i, m in enumerate(st):
        def mk(i):
            def f(module, inp, out):
                log(f"[st[{i}] {type(module).__name__}]")
                log(f"  IN : {desc(inp)}")
                log(f"  OUT: {desc(out)}")
            return f
        m.register_forward_hook(mk(i))

    log("[3] encode ...")
    e = np.asarray(st.encode({"image": im}, normalize_embeddings=True,
                             show_progress_bar=False)).reshape(-1)
    log(f"    e_A dim={e.shape} norm={np.linalg.norm(e):.4f}")

    z = CAP["z"]
    et = torch.tensor(e, device=z.device, dtype=torch.float32)
    cands = {
        "z.mean(0)": z.float().mean(0),
        "z[0]": z.float()[0],
        "z[1]": z.float()[1],
        "z[-1]": z.float()[-1],
        "z.sum(0)": z.float().sum(0),
        "z.max(0)": z.float().max(0).values,
    }
    for k, v in cands.items():
        c = float((v @ et) / (v.norm() + 1e-12) / (et.norm() + 1e-12))
        log(f"    cos(e_A, {k}) = {c:.4f}")
    log(f"    split_sizes={CAP['split_sizes']} z={tuple(z.shape)} po={tuple(CAP['po'].shape)}")
    log("[DONE]")


if __name__ == "__main__":
    main()
