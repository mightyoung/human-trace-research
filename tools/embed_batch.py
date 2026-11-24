# -*- coding: utf-8 -*-
"""cradio_v3 批处理编码模块（原型，探针 v5 验证通过的正式化封装）。

在服务器上使用（路径按运行环境调整）。用法：把 `install_batch_patch()` 装好之后，
直接调用 SentenceTransformer 的批量接口即可获得全链路批处理：

    st, am = install_batch_patch(m_radio, rproc, st, Wt, device="cuda:1")
    embs = st.encode(list_of_rgb_arrays, normalize_embeddings=True, batch_size=16)
    # embs: [N, 768]，与单图逐图的嵌入完全一致（cos=1.0000，探针 v5 实测）

实测性能（L20，8-64 图/批）：46.6-49.0 ms/图（≈4× vs 单图逐图 185 ms/图）。
注意：本模块只替换 get_image_features 为批处理版，其余（语言塔/池化）由
SentenceTransformer 原生批处理——不改 ST 内部代码。
"""
import numpy as np
import torch
import cv2
from PIL import Image
from transformers.modeling_outputs import BaseModelOutputWithPooling


def rebuild_image(pixel_values, position_ids):
    """Gemma patch 序列 → 图像（单样本）"""
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
    return patches.permute(0, 2, 1, 3, 4).reshape(gh * 16, gw * 16, 3)


def install_batch_patch(m_radio, rproc, st, Wt, device="cuda:0", use_fp16=True):
    """把 st 的视觉注入点替换为「批处理版 RADIO 桥接」。

    参数：
      m_radio  — C-RADIOv4 模型（eval、已在 device）
      rproc    — RADIO 的 CLIPImageProcessor
      st       — SentenceTransformer（EmbeddingGemma2）
      Wt       — 桥接矩阵 [1152, 768]（torch tensor，已在 device）
      device   — cuda 设备串
      use_fp16 — RADIO 前向使用 fp16 autocast（已验证无损）

    返回 (st, am)。此后 st.encode(list_of_images, batch_size=N) 即为批处理全链路。
    """
    am = getattr(st[0], "auto_model", None) or getattr(st[0], "model", None)
    assert am is not None, "未找到 auto_model"

    def patched(pixel_values, image_position_ids=None, **kwargs):
        pv = pixel_values.detach().cpu()
        ids = image_position_ids.detach().cpu()
        B = pv.shape[0]  # 注意：ST 批模式下为 [B, N, 3]（每样本等长）

        # 1) 逐样本重建图像（CPU，1.3ms/图）
        imgs = []
        for b in range(B):
            img = rebuild_image(pv[b:b + 1], ids[b:b + 1])
            imgs.append((img.numpy() * 255).clip(0, 255).astype(np.uint8))

        # 2) 批量 RADIO 前向（fp16）
        pix = rproc(images=[Image.fromarray(x) for x in imgs], return_tensors="pt",
                    do_resize=True, do_center_crop=False,
                    do_normalize=False).pixel_values.to(device)
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = m_radio(pix)
                    sp = out[1]
                    B2, L, D = sp.shape
                    g = sp.reshape(B2, 32, 32, D)
                    pp = g.reshape(B2, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B2, 256, D)
                z2 = (pp.reshape(B2 * 256, D).float() @ Wt)          # [B*256, 768]
            else:
                out = m_radio(pix)
                sp = out[1]
                B2, L, D = sp.shape
                g = sp.reshape(B2, 32, 32, D)
                pp = g.reshape(B2, 16, 2, 16, 2, D).mean(dim=(2, 4)).reshape(B2, 256, D)
                z2 = (pp.reshape(B2 * 256, D) @ Wt)

        # 3) embed_vision（被 EmbeddingGemma2Model.forward 消费，不可省）
        po = am.embed_vision(inputs_embeds=z2)                        # [B*256, 512]
        per = po.shape[0] // B2
        parts = list(torch.split(po, [per] * B2))
        return BaseModelOutputWithPooling(last_hidden_state=z2, pooler_output=parts)

    am.get_image_features = patched
    return st, am


def load_cradio_bundle(root, device="cuda:1", w_name="radio_W_scale.npy",
                       w_dir="cradio_scale_v3"):
    """便捷加载：RADIO + EmbeddingGemma2(ST) + 桥接 W + 安装批处理补丁。

    root 例：/home/r4900g3/yhh/human-count/crosscam_run
    返回 (st, m_radio, rproc, Wt, am)
    """
    import os
    from transformers import AutoModel, CLIPImageProcessor
    from sentence_transformers import SentenceTransformer

    radio_dir = os.path.join(root, "cradio", "C-RADIOv4-SO400M")
    emb_path = os.path.join(root, "embgemma", "model", "master")
    w_path = os.path.join(root, w_dir, w_name)

    m_radio = AutoModel.from_pretrained(radio_dir, trust_remote_code=True).to(device).eval()
    rproc = CLIPImageProcessor.from_pretrained(radio_dir)
    st = SentenceTransformer(emb_path, device=device,
                             model_kwargs={"torch_dtype": "float32"})
    Wt = torch.from_numpy(np.load(w_path).astype(np.float32)).to(device)
    st, am = install_batch_patch(m_radio, rproc, st, Wt, device=device)
    return st, m_radio, rproc, Wt, am


if __name__ == "__main__":
    # 冒烟：加载 + 预热 + 8 图 batch 编码稳态计时（需在服务器运行）
    import os
    import time
    root = "/home/r4900g3/yhh/human-count/crosscam_run"
    dev = os.environ.get("FUSION_DEV", "cuda:1")
    st, *_ = load_cradio_bundle(root, device=dev)
    imgs = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
    # 预热（首批含 cudnn/cublas 自动调优与内存分配，不代表稳态）
    st.encode(imgs, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(2):
        t0 = time.perf_counter()
        out = st.encode(imgs, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / 8 * 1000)
    print(f"[embed_batch] 8 图 batch 稳态: {min(times):.1f} ms/图 (两轮: "
          f"{times[0]:.1f}/{times[1]:.1f}), emb {np.asarray(out).shape}")
