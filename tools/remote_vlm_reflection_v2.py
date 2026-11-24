#!/usr/bin/env python
"""P4b：VLM 鉴镜实验·提示词变体（仅 6 张镜像图）

目的：区分"0.5B 模型无此能力" vs "提示词引导不足"。
三种提示词：直接选择 / 链式观察 / 落地检测（双脚是否踩地）。
输出：vlm_results_v2.json
"""
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = "/share/data4/yhh/models/LLaVA-OneVision 0.5B"
D = "/home/r4900g3/yhh/human-count/crosscam/reflection_test"

PROMPTS = {
    "direct": ("图中用红色矩形框标注的区域里有人物目标。请判断：这个目标是场景中真实站立/行走的人，"
               "还是玻璃或镜面表面反射形成的虚影（倒影）？只回答两个词之一：真人 或 反光。"),
    "chain": ("请仔细观察图片。先用一句话描述红框区域周围是否存在玻璃、镜面或反光现象；"
              "然后判断红框中的目标是：A 真实存在的人；B 玻璃/镜面反射形成的虚影。"
              "最后一行必须输出：答案=A 或 答案=B。"),
    "ground": ("红框里的目标，其双脚是否踩在真实的地板上、身体处于真实三维空间中？"
               "如果它更像是出现在玻璃或镜面上的反光影像，输出 答案=B；"
               "如果是真实存在的人，输出 答案=A。只输出一行：答案=A 或 答案=B。"),
}

try:
    from transformers import LlavaOnevisionForConditionalGeneration as M
except Exception:
    from transformers import AutoModelForImageTextToText as M

print("[vlm2] loading ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL)
model = M.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:1")
model.eval()
print("[vlm2] ready", flush=True)

man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
phantoms = [m for m in man if m["label"] == "phantom"]
results = []
for m in phantoms:
    img = Image.open(os.path.join(D, m["file"])).convert("RGB")
    for pname, ptxt in PROMPTS.items():
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ptxt}]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        ans = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({**m, "prompt": pname, "answer": ans})
        print(f"{m['file']} [{pname}] -> {ans[:120]}", flush=True)

json.dump(results, open(os.path.join(D, "vlm_results_v2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n[vlm2] done", flush=True)
