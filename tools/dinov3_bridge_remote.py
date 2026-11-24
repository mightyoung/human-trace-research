# -*- coding: utf-8 -*-
"""DINOv3-B × EmbeddingGemma 2 桥接 —— 严格三集版（服务器）。

与首轮 T3（280 图，W 在评估轨上拟合=泄漏）不同：本版按 splits_v3 划分，
W 仅在 train 轨上拟合（test ∪ val 全部留出）；并附：
  - solve:  随机正交 W 的 train SSE 对照
  - control: 仅 val 轨上 ridge-W vs 随机 W 的干净 AUC 对照

阶段（与 cradio_scale_remote.py 完全同构）:
  collect → solve → encode ；命令 smoke/control 另用
用法:
  export FUSION_DEV=cuda:0
  python dinov3_bridge_remote.py collect / solve / encode / control / smoke
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
EMB_PATH = os.path.join(ROOT, "embgemma", "model", "master")
DINO_W = os.path.join(ROOT, "models", "dinov3_vitb16_timm", "model.safetensors")
OUT = os.path.join(ROOT, os.environ.get("D3B_OUT", "dinov3_scale_v3"))
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("FUSION_DEV", "cuda:0")

DATA = "/share/data4/yhh/data/human-count"
CAMS = ["318", "320", "305", "316", "323", "343"]
PREFIX = {"318": "a", "320": "b", "305": "v305_", "316": "v316_",
          "323": "v323_", "343": "v343_"}
HOLDOUT_P = os.path.join(OUT, "holdout.json")
SPLITS_P = os.path.join(OUT, "splits_v3.json")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_view import remote_det, describe  # noqa: E402


# ---------------- 与生产线一致的裁剪/加载 ----------------
def prep_crop(frame, box, out_h=224, pad=0.08):
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


# ---------------- 模型 ----------------
def load_d3b():
    import timm
    m = timm.create_model("vit_base_patch16_dinov3", pretrained=False,
                          num_classes=0, global_pool="token")
    timm.models.load_checkpoint(m, DINO_W)
    m = m.to(DEV).eval()
    return m


def load_gemma():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMB_PATH, device=DEV,
                             model_kwargs={"torch_dtype": "float32"})
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    assert am is not None
    return st, am


def rebuild_image(pixel_values, position_ids):
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


def d3_pool_32x32(m, x):
    """DINOv3 forward -> patch tokens -> 3x3 池化 -> ×sqrt(768) -> [256,768]"""
    with torch.no_grad():
        feats = m.forward_features(x)
    patch = feats[:, m.num_prefix_tokens:, :]
    gh = gw = int(round(patch.shape[1] ** 0.5))
    k = 3
    gh2, gw2 = gh // k, gw // k
    g = patch[0].reshape(gh, gw, -1)[:gh2 * k, :gw2 * k]
    p = g.reshape(gh2, k, gw2, k, -1).mean(dim=(1, 3))
    z = p.reshape(-1, p.shape[-1])
    z = z * (z.shape[-1] ** 0.5)
    return z


def d3_features(m, gproc, img_np_or_tensor, device=DEV):
    """crop(uint8 RGB) -> [256,768]；或 (pixel_values, position_ids) tensor 对"""
    if torch.is_tensor(img_np_or_tensor):
        img = img_np_or_tensor
    else:
        o = gproc(images=img_np_or_tensor, return_tensors="pt")
        img = rebuild_image(o["pixel_values"], o["image_position_ids"])
    mean = torch.tensor(IMAGENET_MEAN, device=device)
    std = torch.tensor(IMAGENET_STD, device=device)
    x = ((img.to(device) - mean) / std).permute(2, 0, 1).unsqueeze(0)
    return d3_pool_32x32(m, x)


# ---------------- collect ----------------
def cmd_collect(args):
    hold = {}
    if os.path.exists(HOLDOUT_P):
        hj = json.load(open(HOLDOUT_P))
        hold = {int(c): set(int(x) for x in v) for c, v in hj.items()}
    print(f"[collect] holdout: {[(c, len(v)) for c, v in hold.items()]}", flush=True)

    m_d3 = load_d3b()
    st, am = load_gemma()
    from transformers import AutoProcessor
    gproc = AutoProcessor.from_pretrained(EMB_PATH)
    print("[collect] 两模型就绪", flush=True)

    A = np.zeros((768, 768), np.float64)
    B = np.zeros((768, 768), np.float64)
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
                while zg.ndim > 2:
                    zg = zg[0]
                assert zg.shape == (256, 768), zg.shape
                z = d3_features(m_d3, gproc, rebuild_image(o["pixel_values"],
                                                           o["image_position_ids"]))
                zrg = z.cpu().numpy().astype(np.float64)
                zgg = zg.cpu().numpy().astype(np.float64)
                if not is_h:
                    A += zrg.T @ zrg
                    B += zrg.T @ zgg
                    C += float((zgg ** 2).sum())
                    n_train += 1
                zr_means.append(zrg.mean(axis=0).astype(np.float32))
                zg_means.append(zgg.mean(axis=0).astype(np.float32))
                meta.append({"cam": cam, "tid": int(tid), "holdout": bool(is_h)})
                n += 1
                if n % 200 == 0:
                    print(f"[collect] {n} imgs (train {n_train}) "
                          f"{time.time()-t0:.0f}s", flush=True)
        cap.release()
        print(f"[collect] cam {cam} done | total {n} imgs | {time.time()-t0:.0f}s",
              flush=True)

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

    def struct_corr(idx):
        pr = zr_mean[idx].astype(np.float64)
        pg = zg_mean[idx].astype(np.float64)
        pr /= (np.linalg.norm(pr, axis=1, keepdims=True) + 1e-9)
        pg /= (np.linalg.norm(pg, axis=1, keepdims=True) + 1e-9)
        iu = np.triu_indices(len(idx), k=1)
        return float(np.corrcoef((pr @ pr.T)[iu], (pg @ pg.T)[iu])[0, 1])

    idx_tr = [i for i, m in enumerate(meta) if not m["holdout"]]
    idx_h = [i for i, m in enumerate(meta) if m["holdout"]]
    if len(idx_tr) > 2000:
        rng = np.random.RandomState(0)
        idx_tr = list(rng.choice(idx_tr, 2000, replace=False))
    r_tr = struct_corr(idx_tr)
    r_h = struct_corr(idx_h) if len(idx_h) >= 8 else float("nan")
    print(f"[analyze] 池化特征跨图相似结构相关: train={r_tr:.4f} "
          f"holdout={r_h:.4f} (n_h={len(idx_h)})", flush=True)

    lam = 1.0
    W = np.linalg.solve(A + lam * np.eye(768), B)
    sse = float(np.trace(W.T @ A @ W) - 2 * np.trace(W.T @ B) + C)
    # 随机正交对照（train SSE）
    rng = np.random.RandomState(0)
    Q, _ = np.linalg.qr(rng.randn(768, 768))
    sse_rand = float(np.trace(Q.T @ A @ Q) - 2 * np.trace(Q.T @ B) + C)
    np.save(os.path.join(OUT, "d3bW_scale.npy"), W.astype(np.float32))
    np.save(os.path.join(OUT, "d3bW_rand.npy"), Q.astype(np.float32))
    stats = {"n_imgs": n_img, "n_train": n_train, "n_holdout": len(idx_h),
             "struct_corr_train": r_tr, "struct_corr_holdout": r_h,
             "sse_train": sse, "sse_train_rand": sse_rand,
             "sse_per_token": sse / (n_train * 256)}
    json.dump(stats, open(os.path.join(OUT, "scale_solve_stats.json"), "w"),
              indent=1)
    print(f"[solve] SSE_train={sse:.3e} per-token={stats['sse_per_token']:.4f} | "
          f"随机W SSE={sse_rand:.3e}（比值 {sse_rand/sse:.4f}）")
    print(f"[solve] W -> d3bW_scale.npy", flush=True)


# ---------------- encode ----------------
def _make_patched(am, Wt, m_d3, gproc):
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    def patched(pixel_values, image_position_ids=None, **kwargs):
        img = rebuild_image(pixel_values.detach().cpu(),
                            image_position_ids.detach().cpu())
        z = d3_features(m_d3, gproc, img)
        z = (z @ Wt).to(DEV)
        pooler_output = am.embed_vision(inputs_embeds=z)
        output_length = pixel_values.shape[-2] // (
            am.config.vision_config.pooling_kernel_size ** 2)
        k_squared = int((image_position_ids.shape[1] // output_length) ** 0.5) ** 2
        non_pad_mask = (image_position_ids != -1).all(dim=-1)
        split_sizes = (non_pad_mask.sum(dim=-1) // k_squared).tolist()
        return BaseModelOutputWithPooling(
            last_hidden_state=z,
            pooler_output=torch.split(pooler_output, split_sizes))

    return patched


def cmd_encode(args):
    st, am = load_gemma()
    m_d3 = load_d3b()
    from transformers import AutoProcessor
    gproc = AutoProcessor.from_pretrained(EMB_PATH)
    Wt = torch.from_numpy(
        np.load(os.path.join(OUT, "d3bW_scale.npy")).astype(np.float32)).to(DEV)
    am.get_image_features = _make_patched(am, Wt, m_d3, gproc)
    print("[encode] 模型就绪", flush=True)

    def encode_track(cap, tf):
        crops = sample_crops(cap, tf)
        if not crops:
            return None
        embs = []
        for im in crops:
            e = st.encode({"image": im}, normalize_embeddings=True,
                          show_progress_bar=False)
            e = np.asarray(e).reshape(-1)
            embs.append(e / (np.linalg.norm(e) + 1e-12))
        v = np.mean(embs, axis=0)
        return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    t00 = time.time()
    for cam in CAMS:
        out_p = os.path.join(OUT, f"{cam}_emb_d3b.npz")
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
            e = encode_track(cap, tracks[tid])
            if e is None:
                n_skip += 1
                continue
            out[f"{PREFIX[cam]}{tid}"] = e
            if (i + 1) % 40 == 0:
                print(f"[encode] {cam} [{i+1}/{len(tracks)}] "
                      f"{(time.time()-t0)/(i+1):.2f}s/track", flush=True)
                np.savez(out_p, **out)
        cap.release()
        np.savez(out_p, **out)
        print(f"[encode] {cam} DONE {len(out)} tracks (skip {n_skip}) "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"[encode] ALL DONE {time.time()-t00:.0f}s", flush=True)


# ---------------- control（val 上的干净对照） ----------------
def cmd_control(args):
    """仅对 val 轨做 ridge-W vs 随机-W 的编码 AUC 对比（干净对照实验）。"""
    sp = json.load(open(SPLITS_P))
    val_tracks = {c: set(int(x) for x in v) for c, v in sp["val"]["tracks"].items()}
    st, am = load_gemma()
    m_d3 = load_d3b()
    from transformers import AutoProcessor
    gproc = AutoProcessor.from_pretrained(EMB_PATH)
    mean = torch.tensor(IMAGENET_MEAN, device=DEV)
    std = torch.tensor(IMAGENET_STD, device=DEV)

    def encode_tracks(Wt):
        am.get_image_features = _make_patched(am, Wt, m_d3, gproc)
        res = {}
        for cam in CAMS:
            vt = sorted(val_tracks[cam])
            if not vt:
                continue
            tracks = load_tracks(remote_det(cam, DATA))
            cap = cv2.VideoCapture(f"{DATA}/video-6way/{cam}.mp4")
            for tid in vt:
                if tid not in tracks:
                    continue
                embs = []
                for im in sample_crops(cap, tracks[tid]):
                    e = st.encode({"image": im}, normalize_embeddings=True,
                                  show_progress_bar=False)
                    e = np.asarray(e).reshape(-1)
                    embs.append(e / (np.linalg.norm(e) + 1e-12))
                if embs:
                    v = np.mean(embs, axis=0)
                    res[(cam, tid)] = v / (np.linalg.norm(v) + 1e-12)
            cap.release()
            print(f"[control] {cam} val 轨编码完 {len(vt)}", flush=True)
        return res

    def eval_val(emb):
        vp, vn = [], []
        for c, a, b in sp["val"]["pos_pairs"]:
            if (c, a) in emb and (c, b) in emb:
                vp.append(float(emb[(c, a)] @ emb[(c, b)]))
        for c, a, b in sp["val"]["neg_pairs"]:
            if (c, a) in emb and (c, b) in emb:
                vn.append(float(emb[(c, a)] @ emb[(c, b)]))
        vp_, vn_ = np.array(vp), np.array(vn)
        auc = float(np.mean([(p > n) + 0.5 * (p == n) for p in vp_ for n in vn_]))
        return auc, len(vp_), len(vn_)

    Wr = torch.from_numpy(
        np.load(os.path.join(OUT, "d3bW_scale.npy")).astype(np.float32)).to(DEV)
    Wq = torch.from_numpy(
        np.load(os.path.join(OUT, "d3bW_rand.npy")).astype(np.float32)).to(DEV)
    print("[control] 编码 val（ridge W）...", flush=True)
    emb_ridge = encode_tracks(Wr)
    a1, np1, nn1 = eval_val(emb_ridge)
    print(f"[control] ridge-W  val AUC={a1:.4f} (pos {np1}/neg {nn1})", flush=True)
    print("[control] 编码 val（随机 W）...", flush=True)
    emb_rand = encode_tracks(Wq)
    a2, np2, nn2 = eval_val(emb_rand)
    print(f"[control] 随机-W   val AUC={a2:.4f} (pos {np2}/neg {nn2})", flush=True)
    json.dump({"auc_ridge": a1, "auc_rand": a2, "n_pos": np1, "n_neg": nn1},
              open(os.path.join(OUT, "control_val.json"), "w"), indent=1)
    print(f"[control] 差值 {a1-a2:+.4f} -> control_val.json", flush=True)


# ---------------- smoke ----------------
def cmd_smoke(args):
    m_d3 = load_d3b()
    st, am = load_gemma()
    from transformers import AutoProcessor
    gproc = AutoProcessor.from_pretrained(EMB_PATH)
    tracks = load_tracks(remote_det("318", DATA))
    cap = cv2.VideoCapture(f"{DATA}/video-6way/318.mp4")
    tid = sorted(tracks)[0]
    crops = sample_crops(cap, tracks[tid])
    print(f"[smoke] tid={tid} crops={len(crops)}")
    o = gproc(images=crops[0], return_tensors="pt")
    img = rebuild_image(o["pixel_values"], o["image_position_ids"])
    z = d3_features(m_d3, gproc, img)
    print(f"[smoke] d3 features: {tuple(z.shape)}（期望 [256,768]）")
    with torch.no_grad():
        zg = am.vision_tower(pixel_values=o["pixel_values"].to(DEV),
                             pixel_position_ids=o["image_position_ids"].to(DEV)).last_hidden_state
    while zg.ndim > 2:
        zg = zg[0]
    print(f"[smoke] gemma vision: {tuple(zg.shape)}")
    assert tuple(z.shape) == tuple(zg.shape)
    print("[smoke] SMOKE_OK")


if __name__ == "__main__":
    ap = ["smoke", "collect", "solve", "encode", "control"]
    cmd = sys.argv[1]
    assert cmd in ap, ap
    print(describe(), flush=True)
    {"smoke": cmd_smoke, "collect": cmd_collect, "solve": cmd_solve,
     "encode": cmd_encode, "control": cmd_control}[cmd](sys.argv[2:])
