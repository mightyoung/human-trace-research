# -*- coding: utf-8 -*-
"""快速探测：C-RADIOv4 batch 输入的输出结构与 embed_vision 行为。"""
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
RADIO_DIR = os.path.join(ROOT, "cradio", "C-RADIOv4-SO400M")
EMB_PATH = os.path.join(ROOT, "embgemma", "model", "master")
DEV = os.environ.get("FUSION_DEV", "cuda:1")

print("[1] load RADIO ...", flush=True)
m = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True).to(DEV).eval()
proc = CLIPImageProcessor.from_pretrained(RADIO_DIR)

imgs = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
pix = proc(images=[Image.fromarray(x) for x in imgs], return_tensors="pt",
           do_resize=True, do_center_crop=False, do_normalize=False).pixel_values
print("[2] pix shape:", tuple(pix.shape), flush=True)

with torch.no_grad():
    out = m(pix.to(DEV))
print("[3] out type:", type(out), "len:", len(out), flush=True)
for i, o in enumerate(out):
    if torch.is_tensor(o):
        print(f"    out[{i}]: tensor {tuple(o.shape)} {o.dtype}", flush=True)
    else:
        print(f"    out[{i}]: {type(o)} -> {getattr(o, 'shape', '?')}", flush=True)

# spatial 池化 batch 版本预览
sp = out[1]
if isinstance(sp, (list, tuple)):
    print("[4] out[1] is list/tuple len", len(sp), "elem0", tuple(sp[0].shape), flush=True)
    sp = torch.stack(list(sp), dim=0)
print("[4] spatial stacked:", tuple(sp.shape), flush=True)
B, L, D = sp.shape
g = sp.reshape(B, 32, 32, D)
p = g.reshape(B, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B, 256, D)
print("[5] pooled:", tuple(p.shape), flush=True)

# embed_vision batch 探测（注意：需先乘桥接 W 到 768 维）
print("[6] load Gemma ...", flush=True)
from sentence_transformers import SentenceTransformer
st = SentenceTransformer(EMB_PATH, device=DEV, model_kwargs={"torch_dtype": "float32"})
am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
print("[6] am:", type(am).__name__, flush=True)
Wt = torch.from_numpy(np.load(os.path.join(
    ROOT, "cradio_scale_v3", "radio_W_scale.npy")).astype(np.float32)).to(DEV)
print("[6.5] W:", tuple(Wt.shape), flush=True)
z = p.reshape(B * 256, D) @ Wt          # [B*256, 768]
z = z.reshape(B, 256, -1)
with torch.no_grad():
    e = am.embed_vision(inputs_embeds=z)
print("[7] embed_vision out:", type(e), getattr(e, "shape", None), flush=True)
if torch.is_tensor(e):
    print("    e:", tuple(e.shape), e.dtype, flush=True)
print("[DONE]", flush=True)
