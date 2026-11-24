# -*- coding: utf-8 -*-
"""C-RADIOv4 × EmbeddingGemma 2 桥接 —— 规模化实验（服务器 GPU）。

六个机位全轨迹（~668 轨 / ~5300 图，vs 首轮 35 轨 / 280 图）：
  collect: 逐图计算 RADIO spatial [256,1152] + Gemma 视觉塔 [256,768]
           → 流式累积 ridge 的 A/B 矩阵（float64；排除跨机位评估留出轨防泄漏）
  solve:   W (1152→768) 闭式解 → radio_W_scale.npy（含结构相关/SSE 诊断）
  encode:  全轨迹桥接嵌入（RADIO→W→embed_vision→语言塔；每轨 8 帧均值）
           → {cam}_emb_scale.npz（键前缀与历史一致：a/b/v{cam}_）

用法（先上传 holdout.json）:
  python cradio_scale_remote.py collect
  python cradio_scale_remote.py solve
  python cradio_scale_remote.py encode
"""
import os
import sys
import json
import time

import numpy as np
import torch
import cv2
from PIL import Image

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
RADIO_DIR = os.path.join(ROOT, "cradio", "C-RADIOv4-SO400M")
EMB_PATH = os.path.join(ROOT, "embgemma", "model", "master")
OUT = os.path.join(ROOT, os.environ.get("SCALE_OUT", "cradio_scale"))
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("FUSION_DEV", "cuda:1")

DATA = "/share/data4/yhh/data/human-count"
CAMS = ["318", "320", "305", "316", "323", "343"]
PREFIX = {"318": "a", "320": "b", "305": "v305_", "316": "v316_",
          "323": "v323_", "343": "v343_"}
HOLDOUT_P = os.path.join(OUT, "holdout.json")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_view import remote_det, describe  # noqa: E402


# ---------------- 与生产线完全一致的裁剪 ----------------
def prep_crop(frame, box, out_h=224, pad=0.08):
    """与 crosscam_reid.prep_crop 完全一致（BGR in → RGB out）"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad))
    Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad))
    Y2 = min(frame.shape[0], int(y2 + h * pad))
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
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2,
                               cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def load_tracks(jsonl_path):
    """与 encode_tracks.load_tracks 完全一致"""
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
    for k in t:
        t[k].sort()
    return t


def sample_crops(cap, tf, samples=8):
    idxs = np.linspace(0, len(tf) - 1, min(samples, len(tf))).astype(int)
    crops = []
    for k in idxs:
        fnum, box = tf[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = prep_crop(frame, box)
        if rgb is not None:
            crops.append(rgb)
    return crops


# ---------------- 模型（与 cradio_fusion_remote.py 一致） ----------------
def load_radio(device=DEV):
    from transformers import AutoModel, CLIPImageProcessor
    m = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True)
    m = m.to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    return m, proc


def load_gemma():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_PATH, device=DEV,
                             model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    assert am is not None
    return st, am


def radio_spatial_512(m, proc, img_np, device=DEV):
    """224/768 uint8 RGB -> [256, 1152]（512 输入，2x2 池化）"""
    pix = proc(images=[Image.fromarray(img_np)], return_tensors="pt",
               do_resize=True, do_center_crop=False,
               do_normalize=False).pixel_values.to(device)
    with torch.no_grad():
        out = m(pix)
    sp = out[1][0]
    g = sp.reshape(32, 32, -1)
    p = g.reshape(16, 2, 16, 2, -1).mean(dim=(1, 3))
    return p.reshape(256, -1)


def rebuild_image(pixel_values, position_ids):
    """Gemma processor patch 序列 -> [H, W, 3] float [0,1]（CPU）"""
    pv = pixel_values[0]
    ids = position_ids[0]
    valid = (ids != -1).all(dim=-1)
    pv = pv[valid]
    ids = ids[valid]
    xs, ys = ids[:, 0], ids[:, 1]
    gw = int(xs.max()) + 1
    gh = int(ys.max()) + 1
    order = torch.argsort(ys * gw + xs)
    patches = pv[order].reshape(gh, gw, 16, 16, 3)
    img = patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)
    return img


def encode_track(st_model, crops):
    embs = []
    for im in crops:
        e = st_model.encode({"image": im}, normalize_embeddings=True,
                            show_progress_bar=False)
        e = np.asarray(e).reshape(-1)
        embs.append(e / (np.linalg.norm(e) + 1e-12))
    v = np.mean(embs, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------- collect ----------------
def cmd_collect(args):
    hold = {}
    if os.path.exists(HOLDOUT_P):
        hj = json.load(open(HOLDOUT_P))
        hold = {int(c): set(int(x) for x in v) for c, v in hj.items()}
    print(f"[collect] holdout: {[(c, len(v)) for c, v in hold.items()]}", flush=True)

    m_radio, rproc = load_radio()
    st, am = load_gemma()
    from transformers import AutoProcessor
    gproc = AutoProcessor.from_pretrained(EMB_PATH)
    print("[collect] 三模型就绪", flush=True)

    A = np.zeros((1152, 1152), np.float64)
    B = np.zeros((1152, 768), np.float64)
    C = 0.0
    zr_means, zg_means, meta = [], [], []
    n = n_train = n_skip = 0
    t0 = time.time()
    for cam in CAMS:
        tracks = load_tracks(remote_det(cam, DATA))
        cap = cv2.VideoCapture(f"{DATA}/video-6way/{cam}.mp4")
        assert cap.isOpened(), cam
        for tid in sorted(tracks):
            crops = sample_crops(cap, tracks[tid])
            if not crops:
                n_skip += 1
                continue
            is_h = int(tid) in hold.get(int(cam), set())
            for im in crops:
                o = gproc(images=im, return_tensors="pt")
                pv = o["pixel_values"].to(DEV)
                ids = o["image_position_ids"].to(DEV)
                with torch.no_grad():
                    zg = am.vision_tower(pixel_values=pv,
                                         pixel_position_ids=ids).last_hidden_state
                while zg.ndim > 2:      # 无 batch 维时直接为 [256,768]
                    zg = zg[0]
                assert zg.shape == (256, 768), zg.shape
                img = rebuild_image(o["pixel_values"], o["image_position_ids"])
                img_np = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
                zr = radio_spatial_512(m_radio, rproc, img_np)
                zrg = zr.cpu().numpy().astype(np.float64)
                zgg = zg.cpu().numpy().astype(np.float64)
                if not is_h:
                    A += zrg.T @ zrg
                    B += zrg.T @ zgg
                    C += float((zgg ** 2).sum())
                    n_train += 1
                zr_means.append(zrg.mean(axis=0).astype(np.float32))
                zg_means.append(zgg.mean(axis=0).astype(np.float32))
                meta.append({"cam": cam, "tid": int(tid),
                             "holdout": bool(is_h)})
                n += 1
                if n % 200 == 0:
                    print(f"[collect] {n} imgs (train {n_train}) "
                          f"{time.time()-t0:.0f}s", flush=True)
        cap.release()
        print(f"[collect] cam {cam} done | total {n} imgs | "
              f"{time.time()-t0:.0f}s", flush=True)

    np.savez(os.path.join(OUT, "scale_collect.npz"),
             A=A, B=B, C=np.array([C]),
             zr_mean=np.stack(zr_means), zg_mean=np.stack(zg_means))
    json.dump(meta, open(os.path.join(OUT, "scale_meta.json"), "w"))
    print(f"[collect] DONE {n} imgs (train {n_train}, skip {n_skip}), "
          f"{time.time()-t0:.0f}s", flush=True)


# ---------------- solve ----------------
def cmd_solve(args):
    d = np.load(os.path.join(OUT, "scale_collect.npz"))
    A, B = d["A"], d["B"]
    C = float(d["C"][0])
    zr_mean, zg_mean = d["zr_mean"], d["zg_mean"]
    meta = json.load(open(os.path.join(OUT, "scale_meta.json")))
    n_img = len(meta)
    n_train = int(sum(1 for m in meta if not m["holdout"]))
    print(f"[solve] imgs={n_img} train={n_train} A_scale={np.abs(A).max():.1e}")

    # 分析：池化特征的跨图相似结构相关（train / holdout 分别）
    def struct_corr(idx):
        pr = zr_mean[idx].astype(np.float64)
        pg = zg_mean[idx].astype(np.float64)
        pr /= (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
        pg /= (np.linalg.norm(pg, axis=1, keepdims=True) + 1e-9)
        iu = np.triu_indices(len(idx), k=1)
        return float(np.corrcoef((pr @ pr.T)[iu], (pg @ pg.T)[iu])[0, 1])

    idx_tr = [i for i, m in enumerate(meta) if not m["holdout"]]
    idx_h = [i for i, m in enumerate(meta) if m["holdout"]]
    # 大子集抽样控制计算量
    if len(idx_tr) > 2000:
        rng = np.random.RandomState(0)
        idx_tr = list(rng.choice(idx_tr, 2000, replace=False))
    r_tr = struct_corr(idx_tr)
    r_h = struct_corr(idx_h) if len(idx_h) >= 8 else float("nan")
    print(f"[analyze] 池化特征跨图相似结构相关: train={r_tr:.4f} "
          f"holdout={r_h:.4f} (n_h={len(idx_h)})", flush=True)

    # ridge W：与首轮同 λ=1.0
    lam = 1.0
    W = np.linalg.solve(A + lam * np.eye(1152), B)
    sse = float(np.trace(W.T @ A @ W) - 2 * np.trace(W.T @ B) + C)
    per_tok = sse / (n_train * 256)
    np.save(os.path.join(OUT, "radio_W_scale.npy"), W.astype(np.float32))
    stats = {"n_imgs": n_img, "n_train": n_train, "n_holdout": len(idx_h),
             "struct_corr_train": r_tr, "struct_corr_holdout": r_h,
             "sse_train": sse, "sse_per_token": per_tok,
             "W_absmax": float(np.abs(W).max())}
    json.dump(stats, open(os.path.join(OUT, "scale_solve_stats.json"), "w"),
              indent=1)
    print(f"[solve] SSE_train={sse:.1f} per-token={per_tok:.4f} | "
          f"W -> radio_W_scale.npy", flush=True)


# ---------------- encode ----------------
def cmd_encode(args):
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    st, am = load_gemma()
    m_radio, rproc = load_radio()
    Wt = torch.from_numpy(
        np.load(os.path.join(OUT, "radio_W_scale.npy")).astype(np.float32)).to(DEV)
    print("[encode] 模型就绪", flush=True)

    def patched(pixel_values, image_position_ids=None, **kwargs):
        img = rebuild_image(pixel_values.detach().cpu(),
                            image_position_ids.detach().cpu())
        img_np = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
        z = radio_spatial_512(m_radio, rproc, img_np)
        z = (z @ Wt).to(DEV)
        pooler_output = am.embed_vision(inputs_embeds=z)
        output_length = pixel_values.shape[-2] // (am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z,
            pooler_output=torch.split(pooler_output, split_sizes))

    am.get_image_features = patched

    t00 = time.time()
    for cam in CAMS:
        out_p = os.path.join(OUT, f"{cam}_emb_scale.npz")
        if os.path.exists(out_p):
            d = np.load(out_p)
            if len(d.files) > 0:
                print(f"[encode] {cam} 已完成 {len(d.files)} 轨，跳过", flush=True)
                continue
        tracks = load_tracks(remote_det(cam, DATA))
        cap = cv2.VideoCapture(f"{DATA}/video-6way/{cam}.mp4")
        out = {}
        t0 = time.time()
        n_skip = 0
        for i, tid in enumerate(sorted(tracks)):
            crops = sample_crops(cap, tracks[tid])
            if not crops:
                n_skip += 1
                continue
            e = encode_track(st, crops)
            out[f"{PREFIX[cam]}{tid}"] = e.astype(np.float32)
            if (i + 1) % 40 == 0:
                print(f"[encode] {cam} [{i+1}/{len(tracks)}] "
                      f"{(time.time()-t0)/(i+1):.2f}s/track", flush=True)
                np.savez(out_p, **out)
        cap.release()
        np.savez(out_p, **out)
        print(f"[encode] {cam} DONE {len(out)} tracks (skip {n_skip}) "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[encode] ALL DONE {time.time()-t00:.0f}s", flush=True)


# ---------------- smoke ----------------
def cmd_smoke(args):
    """单图全链路冒烟：processor → vision tower → rebuild → RADIO → A/B 累积"""
    m_radio, rproc = load_radio()
    st, am = load_gemma()
    from transformers import AutoProcessor
    gproc = AutoProcessor.from_pretrained(EMB_PATH)
    tracks = load_tracks(remote_det("318", DATA))
    cap = cv2.VideoCapture(f"{DATA}/video-6way/318.mp4")
    tid = sorted(tracks)[0]
    crops = sample_crops(cap, tracks[tid])
    print(f"[smoke] tid={tid} crops={len(crops)} shape={crops[0].shape}")
    o = gproc(images=crops[0], return_tensors="pt")
    pv = o["pixel_values"].to(DEV)
    ids = o["image_position_ids"].to(DEV)
    with torch.no_grad():
        zg = am.vision_tower(pixel_values=pv, pixel_position_ids=ids).last_hidden_state
    print(f"[smoke] last_hidden_state: {tuple(zg.shape)}")
    while zg.ndim > 2:
        zg = zg[0]
    zgg = zg.double().cpu().numpy()
    img = rebuild_image(o["pixel_values"], o["image_position_ids"])
    img_np = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
    zr = radio_spatial_512(m_radio, rproc, img_np)
    zrg = zr.double().cpu().numpy()
    print(f"[smoke] zg={zgg.shape} zr={zrg.shape}")
    A = zrg.T @ zrg
    B = zrg.T @ zgg
    print(f"[smoke] A={A.shape} B={B.shape}")
    assert zgg.shape == (256, 768) and zrg.shape == (256, 1152)
    assert A.shape == (1152, 1152) and B.shape == (1152, 768)
    print("[smoke] SMOKE_OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "collect", "solve", "encode"])
    args = ap.parse_args()
    print(describe(), flush=True)
    {"smoke": cmd_smoke, "collect": cmd_collect, "solve": cmd_solve,
     "encode": cmd_encode}[args.cmd](sys.argv[1:])
