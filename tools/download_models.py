#!/usr/bin/env python
"""下载视觉底座模型权重（走 hf-mirror，非 gated 渠道）。

2026-10-12 查证结论：
- DINOv3 官方 facebook/* 均为 gated（403）→ 用 timm 官方重新托管的权重（非 gated，等价权重）
- C-RADIOv4-SO400M 非 gated，可直接下载（含 trust_remote_code 所需 py 文件）
"""
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOBS = [
    # (repo, local_dir, allow_patterns)
    ("nvidia/C-RADIOv4-SO400M", os.path.join(ROOT, "models", "C-RADIOv4-SO400M"),
     ["*.json", "*.py", "*.safetensors", "*.md", "*.txt", "*.tar"]),
    ("timm/vit_small_patch16_dinov3.lvd1689m", os.path.join(ROOT, "models", "dinov3_vits16_timm"),
     None),
    ("timm/vit_base_patch16_dinov3.lvd1689m", os.path.join(ROOT, "models", "dinov3_vitb16_timm"),
     None),
]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for repo, dst, allow in JOBS:
        if only and only not in repo:
            continue
        print(f"\n{'='*60}\n[{time.strftime('%H:%M:%S')}] download {repo}\n  -> {dst}", flush=True)
        t0 = time.time()
        try:
            p = snapshot_download(repo, local_dir=dst, allow_patterns=allow,
                                  max_workers=4)
            print(f"[done] {repo} in {time.time()-t0:.0f}s -> {p}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {repo}: {type(e).__name__} {str(e)[:300]}", flush=True)
    print("\n[all done]", flush=True)


if __name__ == "__main__":
    main()
