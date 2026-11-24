#!/usr/bin/env python
"""HF 格式 GroundingDINO → ONNX（本机 DirectML 推理链路用）。

背景（2026-10-17 实锤）：
  远程生产检测模型 = IDEA-Research/grounding-dino-base（HF 格式），
  model.safetensors sha1 = f9ce46b65eda3420734c…（与远程 dino-model 逐字节一致）。
  本机无 CUDA/ROCm；torch-directml 直跑 HF 模型触发 DML 内核崩溃（实测），
  已知可行路径 = ONNX + onnxruntime-directml（tools/local_detect.py）。

形式（全固定形状；16:9 源预处理恒为 750×1333，Swin 常量折叠无影响）：
  inputs : pixel_values   f32  [1, 3, 750, 1333]
           input_ids      i64  [1, 4]              （prompt "person." 的 BERT 编码）
           attention_mask i64  [1, 4]              （全 1）
           pixel_mask     bool [1, 750, 1333]      （全 True）
  outputs: pred_logits    f32  [1, 900, 256]        （与 HF 输出同构）
           pred_boxes     f32  [1, 900, 4]          （归一化 cxcywh）

用法:
  python tools/export_onnx_hf.py \
      --model models/IDEA-Research--grounding-dino-base \
      --out models/onnx/groundingdino_base.onnx
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def _install_text_patches(input_ids):
    """文本侧导出补丁（legacy trace 导出器的算子限制规避）。

    背景：`generate_masks_with_special_tokens_and_transfer_map`（文本 self-attention
    mask / position ids 生成）内部使用 torch.isin / torch.cummax / torch.cummin；
    opset 17 无 cummax/cummin 的导出分解。因 prompt 固定（input_ids 恒定），
    该函数输出为常量——预计算后替换为常量返回（一次绕过全部三者）。

    另对 torch.isin 做防御性等价替换：(x[...,None]==y).any(-1)。
    """
    import transformers.models.grounding_dino.modeling_grounding_dino as gd

    with torch.no_grad():
        masks, pos_ids = gd.generate_masks_with_special_tokens_and_transfer_map(input_ids)
    masks_c, pos_ids_c = masks.clone(), pos_ids.clone()
    print(f"    [patch] 文本 mask 预计算: mask={tuple(masks_c.shape)} "
          f"pos_ids={pos_ids_c.tolist()}", flush=True)

    def _text_mask_const(_input_ids):
        return masks_c, pos_ids_c

    gd.generate_masks_with_special_tokens_and_transfer_map = _text_mask_const

    def _isin_compat(elements, test_elements, *, assume_unique=False, invert=False):
        res = (elements.unsqueeze(-1) == test_elements).any(dim=-1)
        return ~res if invert else res

    torch.isin = _isin_compat


def main():
    ap = argparse.ArgumentParser(description="HF GroundingDINO -> ONNX")
    ap.add_argument("--model", default="models/IDEA-Research--grounding-dino-base")
    ap.add_argument("--out", default="models/onnx/groundingdino_base.onnx")
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--legacy", action="store_true",
                    help="使用旧版 trace 导出器（dynamo=False）；默认 dynamo 新导出器")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    print(f"[1/4] 加载 HF 模型: {args.model}", flush=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).eval()

    text = args.prompt + "."
    tok = proc.tokenizer(text, return_tensors="pt")
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    print(f"    text={text!r}  input_ids={input_ids.tolist()}", flush=True)

    H, W = 750, 1333
    g = torch.Generator().manual_seed(0)
    pixel_values = torch.randn(1, 3, H, W, generator=g)
    pixel_mask = torch.ones(1, H, W, dtype=torch.bool)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values, input_ids, attention_mask, pixel_mask):
            out = self.m(pixel_values=pixel_values, input_ids=input_ids,
                         attention_mask=attention_mask, pixel_mask=pixel_mask)
            return out.logits, out.pred_boxes

    wrap = Wrap(model).eval()
    _install_text_patches(input_ids)  # 文本侧补丁（isin/cummax/cummin -> 常量）
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"[2/4] 导出 ONNX（opset {args.opset}，固定形状，"
          f"{'legacy-trace' if args.legacy else 'dynamo'} 导出器）...", flush=True)
    t0 = time.time()
    export_kwargs = dict(
        opset_version=args.opset,
        input_names=["pixel_values", "input_ids", "attention_mask", "pixel_mask"],
        output_names=["pred_logits", "pred_boxes"],
        do_constant_folding=True,
    )
    if args.legacy:
        export_kwargs["dynamo"] = False
    torch.onnx.export(
        wrap,
        (pixel_values, input_ids, attention_mask, pixel_mask),
        args.out,
        **export_kwargs,
    )
    print(f"    done in {time.time() - t0:.1f}s -> {args.out}", flush=True)

    if args.no_verify:
        return

    print("[3/4] 数值验证（ORT CPU vs PyTorch，同输入）...", flush=True)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.out, so, providers=["CPUExecutionProvider"])
    feed = {
        "pixel_values": pixel_values.numpy(),
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
        "pixel_mask": pixel_mask.numpy(),
    }
    t0 = time.time()
    ol, ob = sess.run(None, feed)
    print(f"    ORT CPU: {time.time() - t0:.1f}s", flush=True)
    with torch.no_grad():
        rl, rb = wrap(pixel_values, input_ids, attention_mask, pixel_mask)
    rl, rb = rl.numpy(), rb.numpy()

    def scores(x):
        z = np.clip(x[0], -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-z))).max(axis=1)

    os_, rs_ = scores(ol), scores(rl)
    ko = set(np.where(os_ > 0.30)[0].tolist())
    kr = set(np.where(rs_ > 0.30)[0].tolist())
    common = sorted(ko & kr)
    union = ko | kr
    print(f"    检出集合: ref={len(kr)} ort={len(ko)} 交集={len(common)} "
          f"一致率={len(common)/len(union):.1%}", flush=True)
    if common:
        sd = float(np.abs(os_[common] - rs_[common]).max())
        bd = float(np.abs(ob[0][common] - rb[0][common]).max())
        print(f"    匹配 query: conf max|Δ|={sd:.2e}  box max|Δ|={bd:.2e}", flush=True)

    print("[4/4] 完成", flush=True)


if __name__ == "__main__":
    main()
