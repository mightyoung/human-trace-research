# -*- coding: utf-8 -*-
"""cradio_v3 探针 v3：路径一致性验证（修正值域）。

Path-A: 现状 st.encode（Gemma processor → rebuild → CLIP → RADIO → W → meanpool）
Path-B1: 跳过 Gemma processor/rebuild（crop 直接 CLIP → RADIO → W → meanpool）
Path-B2: 再叠加 cv2 快速预处理（/255 归一）
比较 cos（8-16 图）→ 判定改造路径的嵌入一致性。
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
OUT_P = os.path.join(ROOT, "cradio_parallel_probe3.json")


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
    """cv2 快速预处理 → [3,size,size] float32 [0,1]（对齐 CLIP 值域）"""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous().float() / 255.0


def z_of(pix_batch, m_radio, Wt):
    """pix [1,3,512,512] -> z [256,768]"""
    with torch.no_grad():
        out = m_radio(pix_batch)
    sp = out[1][0]
    g = sp.reshape(32, 32, -1)
    p = g.reshape(16, 2, 16, 2, -1).mean(dim=(1, 3)).reshape(256, -1)
    return (p @ Wt)


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

    # Path-A: 现状
    log("[A] st.encode x8 ...")
    t0 = time.perf_counter()
    embs_a = []
    for im in crops[:8]:
        e = np.asarray(st.encode({"image": im}, normalize_embeddings=True,
                                 show_progress_bar=False)).reshape(-1)
        embs_a.append(e)
    log(f"    {len(embs_a)} embs, dim={embs_a[0].shape[0]}, {(time.perf_counter()-t0)/8*1000:.1f} ms/img")

    # Path-B1/B2
    log("[B] 直通路径 ...")
    coss_b1, coss_b2 = [], []
    for i, im in enumerate(crops[:8]):
        # B1: CLIP processor 直接
        pix = rproc(images=[Image.fromarray(im)], return_tensors="pt", do_resize=True,
                    do_center_crop=False, do_normalize=False).pixel_values.to(DEV)
        z1 = z_of(pix, m_radio, Wt)
        e1 = z1.mean(dim=0).cpu().numpy()
        e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        # B2: cv2 fast
        pix2 = fast_preproc(im).unsqueeze(0).to(DEV)
        z2 = z_of(pix2, m_radio, Wt)
        e2 = z2.mean(dim=0).cpu().numpy()
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        c1 = float(np.dot(e1, embs_a[i]))
        c2 = float(np.dot(e2, embs_a[i]))
        coss_b1.append(c1); coss_b2.append(c2)
        log(f"    img{i}: A-B1 cos={c1:.4f} | A-B2 cos={c2:.4f}")

    res["cos_A_B1"] = [round(c, 4) for c in coss_b1]
    res["cos_A_B2"] = [round(c, 4) for c in coss_b2]
    res["cos_A_B1_mean"] = round(float(np.mean(coss_b1)), 4)
    res["cos_A_B1_min"] = round(float(np.min(coss_b1)), 4)
    res["cos_A_B2_mean"] = round(float(np.mean(coss_b2)), 4)
    res["cos_A_B2_min"] = round(float(np.min(coss_b2)), 4)

    # 自相似矩阵健全性检查（A 路径）
    simA = np.array(embs_a) @ np.array(embs_a).T
    log(f"[C] A 自相似矩阵 mean={simA.mean():.3f} diag={np.diag(simA).mean():.3f}")

    with open(OUT_P, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    log(f"[DONE] -> {OUT_P}")
    log(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
