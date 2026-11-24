#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本机版 C-RADIOv4×EmbeddingGemma2 编码器（CPU/DML，免远程）。

定位：把远程 `embed_server.py --w-dir cradio_scale_v4` 的 cradio 编码链
（C-RADIOv4-SO400M → 2×2 池化 256 token → 桥接 W → EmbeddingGemma2 embed_vision
→ mean-pool）在本机复刻，用于给 long35 等"只有 d3b 嵌入"的数据补 cradio 空间向量。

数值口径（与远程 embed_server.py 的 legacy 路径逐项对齐）：
  - 裁剪：`_prep_crop`（pad=0.08，按高度缩放到 224、宽度居中裁/补边、BGR→RGB）
  - 采样：`np.linspace(0, n-1, min(samples, n))`（与 _sample_crops 一致）
  - 预处理：CLIPImageProcessor(do_resize=True, do_center_crop=False, do_normalize=False)
  - 桥接：RADIO spatial [b,1024,1152] → reshape 32×32 → 2×2 均值 → [b,256,1152] → @ W
  - 聚合：逐 crop L2 归一 → 轨迹内均值 → 再 L2 归一（与 _run_video_job 一致）

与远程的已知差异（如实标注）：
  - 远程 CUDA autocast fp16；本机 CPU/DML 走 fp32（数值差异应 ≤1e-3 量级，自检见 --selftest）
  - 无 gpu_lock / 多 worker：本脚本单进程串行

用法:
  # 自检（对远程 cradio 产物算 cos）
  python tools/cradio_local.py --selftest --cam 305 --tracks 4
  # 批量编码 long35（a/b 两个 crop 子集，供自一致性指标 ab_same）
  python tools/cradio_local.py --jsonl output_crosscam/long35/305_long35_detections.jsonl \
      --video media-resource/multi-cemra/long-35min/305_long35.mp4 --prefix v305_ \
      --out output_crosscam/long35/embs_cradio/305_emb_long35.npz --ab
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _prep_crop(frame, box, out_h=224, pad=0.08):
    """与 crosscam_reid / cradio_scale_remote / embed_server 完全一致（BGR in → RGB out）"""
    import cv2
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
    """与 encode_tracks.load_tracks / embed_server._load_tracks 一致。"""
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
    """与 embed_server._sample_crops 一致。返回 (crops, 采样下标)。"""
    import cv2
    idxs = np.linspace(0, len(tf) - 1, min(samples, len(tf))).astype(int)
    crops, kept = [], []
    for j, k in enumerate(idxs):
        fnum, box = tf[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = _prep_crop(frame, box)
        if rgb is not None:
            crops.append(rgb)
            kept.append(j)
    return crops, kept


class LocalCradioEncoder:
    """C-RADIOv4 → 桥接 W → EmbeddingGemma2 的本地编码器（CPU/DML、fp32）。"""

    def __init__(self, radio_dir=None, gemma_dir=None, w_path=None,
                 device="cpu", batch=8, verbose=True):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.batch = batch
        radio_dir = radio_dir or str(ROOT / "models" / "C-RADIOv4-SO400M")
        gemma_dir = gemma_dir or str(ROOT / "models" / "google--embeddinggemma-2"
                                     / "snapshots" / "master")
        w_path = w_path or str(ROOT / "output_crosscam" / "analysis"
                               / "cradio_scale_v4" / "radio_W_scale.npy")
        t0 = time.time()
        from transformers import AutoModel, CLIPImageProcessor
        from sentence_transformers import SentenceTransformer
        self.m_radio = AutoModel.from_pretrained(radio_dir, trust_remote_code=True)
        self.m_radio.to(self.device).eval()
        self.rproc = CLIPImageProcessor.from_pretrained(radio_dir)
        self.st = SentenceTransformer(gemma_dir, device=str(self.device),
                                      model_kwargs={"torch_dtype": "float32"})
        self.am = getattr(self.st[0], "auto_model", None) or getattr(self.st[0], "model", None)
        assert self.am is not None, "未找到 auto_model"
        self.Wt = torch.from_numpy(np.load(w_path).astype(np.float32)).to(self.device)
        self.ids, self.mask = self._capture_template()
        self.am.get_image_features = self._patched
        self._pix = None
        if verbose:
            print(f"[cradio-local] loaded in {time.time() - t0:.1f}s "
                  f"(template L={self.template_len}, device={self.device})", flush=True)

    # ---------------- 模板捕获 ----------------
    def _capture_template(self):
        cap = {}
        f0f, f0g = self.am.forward, self.am.get_image_features

        def hook(*a, **k):
            ids = k.get("input_ids")
            if ids is not None and ids.shape[0] == 1 and "ids" not in cap:
                cap["ids"] = ids.detach().clone()
                m = k.get("attention_mask")
                cap["mask"] = m.detach().clone() if m is not None else self.torch.ones_like(ids)
            return f0f(*a, **k)

        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self._pix = self._prep([dummy]).to(self.device)
        self.am.forward = hook
        self.am.get_image_features = self._patched
        try:
            self.st.encode({"image": dummy}, normalize_embeddings=True,
                           show_progress_bar=False)
        finally:
            self.am.forward = f0f
            self.am.get_image_features = f0g
            self._pix = None
        assert "ids" in cap, "模板捕获失败"
        self.template_len = int(cap["ids"].shape[1])
        img_tok = int((cap["ids"][0] == self.am.config.image_token_id).sum())
        assert img_tok == 256, f"模板图像占位异常: {img_tok}（期望 256）"
        return cap["ids"].to(self.device), cap["mask"].to(self.device)

    # ---------------- 图像侧 ----------------
    def _patched(self, pixel_values, image_position_ids=None, **kw):
        return self._radio_bridge(self._pix)

    def _radio_bridge(self, pix):
        torch = self.torch
        with torch.no_grad():
            out = self.m_radio(pix)
            sp = out[1]
            b, l, dim = sp.shape
            g = sp.reshape(b, 32, 32, dim)
            pp = g.reshape(b, 16, 2, 16, 2, dim).mean(dim=(2, 4)).reshape(b, 256, dim)
            z2 = pp.reshape(b * 256, dim).float() @ self.Wt
            po = self.am.embed_vision(inputs_embeds=z2)
            per = po.shape[0] // b
            parts = list(torch.split(po, [per] * b))
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(last_hidden_state=z2, pooler_output=parts)

    def _prep(self, chunk):
        from PIL import Image
        out = self.rproc(images=[Image.fromarray(im) for im in chunk],
                         return_tensors="pt", do_resize=True,
                         do_center_crop=False, do_normalize=False)
        return out.pixel_values

    @property
    def torch_mod(self):
        return self.torch

    # ---------------- 主接口 ----------------
    def encode(self, crops, batch=None):
        torch = self.torch
        bs = batch or self.batch
        embs = []
        with torch.no_grad():
            for i in range(0, len(crops), bs):
                ch = crops[i:i + bs]
                b = len(ch)
                self._pix = self._prep(ch).to(self.device)
                ids = self.ids.expand(b, -1)
                mask = self.mask.expand(b, -1)
                out = self.am(input_ids=ids, attention_mask=mask,
                              pixel_values=torch.zeros(b, 1, 1, device=self.device),
                              image_position_ids=None, return_dict=True)
                lh = out.last_hidden_state.float()
                m = mask.unsqueeze(-1).float()
                embs.append(torch.nn.functional.normalize((lh * m).sum(1) / m.sum(1), dim=-1))
        return torch.cat(embs, 0).float().cpu().numpy()


def encode_tracks(enc, video, tracks, samples=8, min_frames=3, limit=None,
                  ab=False, batch=8):
    """逐轨迹编码。ab=True 时同时返回 a/b 两路子集均值（偶/奇采样下标）。

    聚合口径与远程一致：逐 crop L2 归一 → 轨迹内均值 → 再 L2 归一。
    a/b 子集仅用于自一致性指标（crop 层扰动），全量 V=归一(mean(all))。
    """
    import cv2
    cap = cv2.VideoCapture(video)
    out, out_a, out_b, n_crops, skipped = {}, {}, {}, 0, 0
    tids = sorted(tracks)
    if limit:
        tids = tids[:limit]
    t0 = time.time()
    for n, tid in enumerate(tids, 1):
        tf = tracks[tid]
        if len(tf) < 1:
            skipped += 1
            continue
        crops, kept = sample_crops(cap, tf, samples)
        if len(crops) < max(1, min(min_frames, len(tf))):
            skipped += 1
            continue
        e = enc.encode(crops, batch=batch)
        n_crops += len(e)
        m = e.mean(0)
        out[f"{tid}"] = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)
        if ab and len(e) >= 2:
            ka = [i for i, j in enumerate(kept) if j % 2 == 0]
            kb = [i for i, j in enumerate(kept) if j % 2 == 1]
            if ka and kb:
                ma, mb = e[ka].mean(0), e[kb].mean(0)
                out_a[f"{tid}"] = (ma / (np.linalg.norm(ma) + 1e-9)).astype(np.float32)
                out_b[f"{tid}"] = (mb / (np.linalg.norm(mb) + 1e-9)).astype(np.float32)
        if n % 20 == 0 or n == len(tids):
            el = time.time() - t0
            print(f"[encode] {n}/{len(tids)} tracks, {n_crops} crops, "
                  f"{n_crops / max(el, 1e-9):.2f} img/s, eta "
                  f"{(len(tids) - n) * el / n / 60:.1f} min", flush=True)
    cap.release()
    return out, out_a, out_b, dict(n_tracks=len(out), n_crops=n_crops, skipped=skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ab", action="store_true", help="同时输出 a/b 子集均值（自一致性用）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--selftest", action="store_true",
                    help="在 6 路 10min 305 上前 N 条轨迹自检（对 cradio_scale_v4 参照算 cos）")
    args = ap.parse_args()

    if args.selftest:
        jsonl = str(ROOT / "output_crosscam" / "305_detections.jsonl")
        video = str(ROOT / "media-resource" / "multi-cemra" / "synced-10min"
                    / "02_办公区1_305" / "305_085500-090500.mp4")
        ref = np.load(str(ROOT / "output_crosscam" / "analysis" / "cradio_scale_v4"
                          / "305_emb_scale.npz"))
        tracks = load_tracks(jsonl)
        # 取与参照共有的、帧数足够的轨迹
        cand = [t for t in sorted(tracks) if f"v305_{t}" in ref.files and len(tracks[t]) >= 6]
        limit = args.limit or 4
        enc = LocalCradioEncoder(device=args.device, batch=args.batch)
        t0 = time.time()
        out, _, _, stats = encode_tracks(enc, video, {t: tracks[t] for t in cand[:limit]},
                                        samples=args.samples, limit=limit, batch=args.batch)
        el = time.time() - t0
        coss = []
        for t in sorted(out):
            c = float(np.dot(out[t], ref[f"v305_{t}"]))
            coss.append(c)
            print(f"  track {t}: cos(local, remote) = {c:.5f}")
        print(f"[selftest] n={len(coss)} cos mean={np.mean(coss):.5f} min={np.min(coss):.5f} "
              f"| {stats['n_crops']} crops in {el:.1f}s = {stats['n_crops'] / el:.2f} img/s")
        return

    tracks = load_tracks(args.jsonl)
    enc = LocalCradioEncoder(device=args.device, batch=args.batch)
    out, out_a, out_b, stats = encode_tracks(
        enc, args.video, tracks, samples=args.samples, min_frames=args.min_frames,
        limit=args.limit or None, ab=args.ab, batch=args.batch)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {f"{args.prefix}{k}": v for k, v in out.items()}
    if args.ab:
        payload.update({f"{args.prefix}{k}#a": v for k, v in out_a.items()})
        payload.update({f"{args.prefix}{k}#b": v for k, v in out_b.items()})
    np.savez(args.out, **payload)
    print(f"[done] {args.out} n={len(out)} stats={json.dumps(stats)}")


if __name__ == "__main__":
    main()
