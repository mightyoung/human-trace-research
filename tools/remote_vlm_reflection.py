#!/usr/bin/env python
"""P4：VLM 鉴镜实验（在远程 GPU 上运行）

输入：crosscam/reflection_test/ 下的 14 张测试图（6 镜像 + 8 真人，红框标注）
输出：vlm_results.json + 控制台汇总
模型：/share/data4/yhh/models/LLaVA-OneVision 0.5B（HF 格式）

用法（远程）：
  cd /home/r4900g3/yhh/human-count && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ~/miniconda3/envs/rfdetr/bin/python crosscam/reflection_test/remote_vlm_reflection.py
"""
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = "/share/data4/yhh/models/LLaVA-OneVision 0.5B"
D = "/home/r4900g3/yhh/human-count/crosscam/reflection_test"

PROMPT = ("图中用红色矩形框标注的区域里有人物目标。请判断：这个目标是场景中真实站立/行走的人，"
          "还是玻璃或镜面表面反射形成的虚影（倒影）？只回答两个词之一：真人 或 反光。")

try:
    from transformers import LlavaOnevisionForConditionalGeneration as M
except Exception:
    from transformers import AutoModelForImageTextToText as M

print("[vlm] loading processor ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL)
print("[vlm] loading model ...", flush=True)
model = M.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:1")
model.eval()
print("[vlm] ready", flush=True)

man = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
results = []
for m in man:
    img = Image.open(os.path.join(D, m["file"])).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    ans = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    results.append({**m, "answer": ans})
    print(m["file"], m["label"], "->", ans, flush=True)

json.dump(results, open(os.path.join(D, "vlm_results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

def cls(ans):
    a = ans.replace(" ", "")
    neg = any(k in a for k in ["反光", "镜像", "虚影", "倒影", "反射"])
    pos = any(k in a for k in ["真人", "真实", "实体"])
    if neg and not pos:
        return "phantom"
    if pos and not neg:
        return "real"
    return "unclear"

tp = sum(1 for r in results if r["label"] == "phantom" and cls(r["answer"]) == "phantom")
fn = sum(1 for r in results if r["label"] == "phantom" and cls(r["answer"]) != "phantom")
tn = sum(1 for r in results if r["label"] == "real" and cls(r["answer"]) == "real")
fp = sum(1 for r in results if r["label"] == "real" and cls(r["answer"]) != "real")
print(f"\n[SUMMARY] 镜像识别 {tp}/{tp+fn}  |  真人识别 {tn}/{tn+fp}  (n={len(results)})")
