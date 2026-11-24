# -*- coding: utf-8 -*-
"""embed_fast.py — cradio_v3 Phase 2：模板法极速批量编码（全绕开 ST 封装）。

原理（probe6c 实测、源码级确认）：
  - ST 的 encode 对每张图做 Gemma processor patchify（~9.4ms/图）→ 我们再 rebuild 回图（~8ms/图）
    → RADIO processor（~7ms/图）——两趟"处理器往返"纯属浪费。
  - EmbeddingGemma2Model.forward 的官方装配只需三件套：
      input_ids（256 个图像占位 token 的模板）+ attention_mask + pixel_values(非 None 即可)
    图像特征由 masked_scatter 填入（get_image_features 被替换为我们的批处理 RADIO 桥接）。
  - 模板与图像内容和尺寸无关（224² 方图恒为 L=260、256 占位）→ 捕获一次，后续 expand 复用。
  - 实测（L20, batch16）：21.3 ms/图（46.9 图/s）= 8.7× vs 单图逐图 185ms/图；默认路径（prefetch=False）
    19.7–20.1 ms/图（~50 图/s）= 9.2–9.4×；与 ST 原链路嵌入 cos=0.99995（min 0.99962，差异仅来自图像缩放插值路径）。
  - prefetch 默认 False（实测：PyTorch 异步提交已天然形成背靠背流水，显式预取线程反而略慢 22.0 vs 20.1）。

用法（服务器）：
    from embed_fast import FastEncoder
    enc = FastEncoder()                          # 加载模型 + 捕获模板（一次性）
    embs = enc.encode(list_of_224sq_rgb_uint8)   # [N,768]，L2 归一，与 ST 口径一致
    # 可选：enc.encode(crops, prep="cv2")        # cv2 预处理替代 rproc（见 smoke 复验）

前提：输入 crop 为方形（224²，与生产 prep_crop 一致）。
      （非方图在 ST 原链路里会直接抛 min_resolution_step 错误——此路径无额外限制。）
"""
import os
import gc
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling
from sentence_transformers import SentenceTransformer

DEFAULTS = dict(
    root="/home/r4900g3/yhh/human-count/crosscam_run",
    device="cuda:1",
    batch=16,
    prefetch=False,   # 实测：PyTorch 异步提交已自带"隐式流水"，显式预取线程反而略慢（21.3 vs 19.7）
    w_dir="cradio_scale_v3",
    w_name="radio_W_scale.npy",
    strip_dead=False,  # True=剥离死重（audio_tower/embed_audio/vision_tower，嵌入逐位不变，省 ~1.85GB）
)


class FastEncoder:
    """模板法快速编码器：RADIO 桥接（批处理 fp16）+ 直喂 EmbeddingGemma2 forward + mean-pool。"""

    def __init__(self, root=None, device=None, batch=None, prefetch=None,
                 w_dir=None, w_name=None, strip_dead=None):
        d = dict(DEFAULTS)
        root = root or d["root"]
        device = device or d["device"]
        batch = batch or d["batch"]
        prefetch = d["prefetch"] if prefetch is None else prefetch
        w_dir = w_dir or d["w_dir"]
        w_name = w_name or d["w_name"]
        strip_dead = d["strip_dead"] if strip_dead is None else strip_dead

        self.device = torch.device(device)
        self.batch = batch
        self.prefetch = prefetch

        radio_dir = os.path.join(root, "cradio", "C-RADIOv4-SO400M")
        emb_path = os.path.join(root, "embgemma", "model", "master")
        w_path = os.path.join(root, w_dir, w_name)

        self.m_radio = (AutoModel.from_pretrained(radio_dir, trust_remote_code=True)
                        .to(self.device).eval())
        self.rproc = CLIPImageProcessor.from_pretrained(radio_dir)
        self.st = SentenceTransformer(emb_path, device=str(self.device),
                                      model_kwargs={"torch_dtype": "float32"})
        self.am = getattr(self.st[0], "auto_model", None) or getattr(self.st[0], "model", None)
        assert self.am is not None, "未找到 auto_model"
        self.Wt = torch.from_numpy(np.load(w_path).astype(np.float32)).to(self.device)

        self.ids, self.mask = self._capture_template()
        self.am.get_image_features = self._patched
        self._pix = None
        if strip_dead:
            self.strip_dead_towers()

    # ---------------- 死重剥离 ----------------
    def strip_dead_towers(self):
        """剥离推理期死重塔（audio_tower / embed_audio / vision_tower）。

        实测（emb_strip_verify.py，256 crops + 117 texts，2026-10-14）：
          - 嵌入逐位不变（bitwise=0 diff，max_abs=0.0），文本查询路径同样不变；
          - 省 1.845 GB 显存（4.744→2.899 GB）；编码耗时持平（22.3→21.2 ms/图）。
        注意：embed_vision 是 RADIO 桥接的依赖，必须保留（只剥离上面三组）。
        另注：transformers 5.19 + ST 6.1 下"载入期"config 传参方式与 ST 冲突
        （TypeError），故采用本方法（加载后释放），语义不变。
        """
        removed = []
        for name in ("audio_tower", "embed_audio", "vision_tower"):
            sub = getattr(self.am, name, None)
            if sub is not None:
                setattr(self.am, name, None)
                del sub
                removed.append(name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return removed

    # ---------------- 模板捕获 ----------------
    def _capture_template(self):
        """跑一次 st.encode 单图，hook am.forward 捕获 (input_ids, attention_mask)。

        捕获时即安装快速路径并真跑一次（RADIO 前向 ~50ms）——同时作为端到端自检，
        且不依赖 EmbeddingGemma2 自带的原始 vision_tower。
        """
        cap = {}
        f0f = self.am.forward
        f0g = self.am.get_image_features

        def hook(*a, **k):
            ids = k.get("input_ids")
            if ids is not None and ids.shape[0] == 1 and "ids" not in cap:
                cap["ids"] = ids.detach().clone()
                m = k.get("attention_mask")
                cap["mask"] = m.detach().clone() if m is not None else torch.ones_like(ids)
            return f0f(*a, **k)

        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self._pix = self._prep([dummy], "rproc").to(self.device)
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

    # ---------------- 图像侧（RADIO 桥接） ----------------
    def _patched(self, pixel_values, image_position_ids=None, **kw):
        """am.get_image_features 的替换：忽略 ST 传入的 patch 数据，直用本批 pix。"""
        return self._radio_bridge(self._pix)

    def _radio_bridge(self, pix):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = self.m_radio(pix)
            sp = out[1]
            b, l, dim = sp.shape
            g = sp.reshape(b, 32, 32, dim)
            pp = g.reshape(b, 16, 2, 16, 2, dim).mean(dim=(2, 4)).reshape(b, 256, dim)
        z2 = pp.reshape(b * 256, dim).float() @ self.Wt          # [b*256, 768]
        po = self.am.embed_vision(inputs_embeds=z2)               # [b*256, 512]
        per = po.shape[0] // b
        parts = list(torch.split(po, [per] * b))
        return BaseModelOutputWithPooling(last_hidden_state=z2, pooler_output=parts)

    # ---------------- 预处理 ----------------
    def _prep(self, chunk, prep):
        if prep == "cv2":
            xs = [cv2.resize(im, (512, 512), interpolation=cv2.INTER_LINEAR)
                  .astype(np.float32) / 255.0 for im in chunk]
            return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).contiguous()
        out = self.rproc(images=[Image.fromarray(im) for im in chunk], return_tensors="pt",
                         do_resize=True, do_center_crop=False, do_normalize=False)
        return out.pixel_values                                   # CPU tensor（b,3,512,512）

    # ---------------- 主接口 ----------------
    @torch.no_grad()
    def encode(self, crops, batch=None, prep="rproc"):
        """crops: list/array of HWC uint8 RGB（方形 224²）。返回 [N,768] L2 归一 numpy。"""
        bs = batch or self.batch
        chunks = [crops[i:i + bs] for i in range(0, len(crops), bs)]
        embs = []
        if self.prefetch and len(chunks) > 1:
            ex = ThreadPoolExecutor(2)
            try:
                futs = [ex.submit(self._prep, ch, prep) for ch in chunks]
                for ch, fu in zip(chunks, futs):
                    embs.append(self._gpu_batch(ch, fu.result()))
            finally:
                ex.shutdown()
        else:
            for ch in chunks:
                embs.append(self._gpu_batch(ch, self._prep(ch, prep)))
        return torch.cat(embs, 0).float().cpu().numpy()

    def _gpu_batch(self, chunk, pix_cpu):
        b = len(chunk)
        self._pix = pix_cpu.to(self.device)
        ids = self.ids.expand(b, -1)
        mask = self.mask.expand(b, -1)
        out = self.am(input_ids=ids, attention_mask=mask,
                      pixel_values=torch.zeros(b, 1, 1, device=self.device),
                      image_position_ids=None, return_dict=True)
        lh = out.last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        return F.normalize((lh * m).sum(1) / m.sum(1), dim=-1)


# ==================== smoke / 复验（在服务器运行） ====================

def _make_crops(cam="316", n_tracks=4, per_track=8):
    import json as _json
    DATA = "/share/data4/yhh/data/human-count"
    tracks = {}
    with open(f"{DATA}/inference-out/{cam}_detections.jsonl", encoding="utf-8") as f:
        for line in f:
            r = _json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, bx in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    tracks.setdefault(int(tids[i]), []).append((int(r["frame"]), bx))
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
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            X1 = max(0, int(x1 - w * 0.08)); Y1 = max(0, int(y1 - h * 0.08))
            X2 = min(frame.shape[1], int(x2 + w * 0.08)); Y2 = min(frame.shape[0], int(y2 + h * 0.08))
            c = frame[Y1:Y2, X1:X2]
            if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
                continue
            s = 224 / c.shape[0]
            nw = max(1, int(round(c.shape[1] * s)))
            c = cv2.resize(c, (nw, 224))
            if nw >= 224:
                off = (nw - 224) // 2
                c = c[:, off:off + 224]
            else:
                padw = 224 - nw
                c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2, cv2.BORDER_REPLICATE)
            crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    cap.release()
    return crops[:n_tracks * per_track]


if __name__ == "__main__":
    import json

    res = {}
    print("[smoke] crops ...", flush=True)
    crops = _make_crops()
    print(f"    {len(crops)}", flush=True)

    print("[smoke] load FastEncoder ...", flush=True)
    t0 = time.perf_counter()
    enc = FastEncoder()
    torch.cuda.synchronize(enc.device)
    res["load_sec"] = round(time.perf_counter() - t0, 1)
    res["template_len"] = enc.template_len
    print(f"    loaded in {res['load_sec']}s, template L={enc.template_len}", flush=True)

    # 1) legacy 对照（内联 rebuild 版 patched，跑 st.encode）
    from transformers.modeling_outputs import BaseModelOutputWithPooling as _B

    def _rebuild(pv, ip):
        pv = pv[0]; ids = ip[0]
        v = (ids != -1).all(-1); pv = pv[v]; ids = ids[v]
        xs, ys = ids[:, 0], ids[:, 1]
        gw = int(xs.max()) + 1; gh = int(ys.max()) + 1
        order = torch.argsort(ys * gw + xs)
        pt = pv[order].reshape(gh, gw, 16, 16, 3)
        return pt.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)

    def _patched_legacy(pixel_values, image_position_ids=None, **kw):
        pv = pixel_values.detach().cpu(); ip = image_position_ids.detach().cpu()
        imgs = []
        for b in range(pv.shape[0]):
            img = _rebuild(pv[b:b + 1], ip[b:b + 1])
            imgs.append((img.numpy() * 255).clip(0, 255).astype(np.uint8))
        pix = enc.rproc(images=[Image.fromarray(x) for x in imgs], return_tensors="pt",
                        do_resize=True, do_center_crop=False, do_normalize=False).pixel_values
        pix = pix.to(enc.device)
        return enc._radio_bridge(pix)

    enc.am.get_image_features = _patched_legacy
    try:
        enc.st.encode(crops[:16], batch_size=16, show_progress_bar=False)
        torch.cuda.synchronize(enc.device)
        t0 = time.perf_counter()
        e_legacy = np.asarray(enc.st.encode(crops, batch_size=16, show_progress_bar=False))
        torch.cuda.synchronize(enc.device)
        t_legacy = (time.perf_counter() - t0) / len(crops) * 1000
        res["legacy_ms_per_img"] = round(t_legacy, 2)
        print(f"[smoke] legacy: {t_legacy:.1f} ms/图", flush=True)
    finally:
        enc.am.get_image_features = enc._patched

    # 2) fast 编码（默认路径，prefetch=False）+ cos 对照
    enc.encode(crops[:16])
    torch.cuda.synchronize(enc.device)
    t0 = time.perf_counter()
    e_fast = enc.encode(crops)
    torch.cuda.synchronize(enc.device)
    t_fast = (time.perf_counter() - t0) / len(crops) * 1000
    cc = [float(np.dot(e_fast[i], e_legacy[i])) for i in range(len(crops))]
    res["fast_ms_per_img"] = round(t_fast, 2)
    res["cos_vs_legacy"] = {"mean": round(float(np.mean(cc)), 5), "min": round(float(np.min(cc)), 5)}
    print(f"[smoke] fast: {t_fast:.1f} ms/图  cos={np.mean(cc):.5f} (min {np.min(cc):.5f})", flush=True)

    # 3) prefetch=True 对照
    enc.prefetch = True
    enc.encode(crops[:16])
    torch.cuda.synchronize(enc.device)
    t0 = time.perf_counter()
    enc.encode(crops)
    torch.cuda.synchronize(enc.device)
    res["fast_prefetch_ms_per_img"] = round((time.perf_counter() - t0) / len(crops) * 1000, 2)
    enc.prefetch = False
    print(f"[smoke] prefetch=True: {res['fast_prefetch_ms_per_img']:.1f} ms/图", flush=True)

    # 4) cv2 预处理复验（归一化对照）
    enc.encode(crops[:16], prep="cv2")
    torch.cuda.synchronize(enc.device)
    t0 = time.perf_counter()
    e_cv2 = enc.encode(crops, prep="cv2")
    torch.cuda.synchronize(enc.device)
    res["cv2_ms_per_img"] = round((time.perf_counter() - t0) / len(crops) * 1000, 2)
    cc2 = [float(np.dot(e_cv2[i], e_fast[i])) for i in range(len(crops))]
    res["cv2_cos_vs_rproc"] = {"mean": round(float(np.mean(cc2)), 5), "min": round(float(np.min(cc2)), 5)}
    print(f"[smoke] cv2: {res['cv2_ms_per_img']:.1f} ms/图  cos vs rproc={np.mean(cc2):.5f}", flush=True)

    out_p = os.path.join(DEFAULTS["root"], "embed_fast_smoke.json")
    with open(out_p, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"[DONE] -> {out_p}", flush=True)
    print(json.dumps(res, ensure_ascii=False), flush=True)
