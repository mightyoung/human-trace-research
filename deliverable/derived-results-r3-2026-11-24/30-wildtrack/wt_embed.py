"""WILDTRACK · 阶段 B：OSNet 提特征，带断点续跑。

产物（每 320 个 crop 覆盖写一次）：
  tmp/wt/wt_emb.npy     (N,512) L2 归一化
  tmp/wt/wt_emb_done.txt  已完成条数
重跑本脚本会从 done 处继续。
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

OUT = Path("tmp/wt")
CROPS = OUT / "crops_wt"
OSNET = ("models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
         "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
BS = 32


def main():
    meta = json.loads((CROPS / "meta.json").read_text(encoding="utf-8"))
    n = len(meta)
    done = 0
    dp = OUT / "wt_emb_done.txt"
    ep = OUT / "wt_emb.npy"
    if dp.exists() and ep.exists():
        done = int(dp.read_text().strip() or 0)
        E = np.load(ep)
        if E.shape[0] != done:
            print(f"⚠ 缓存不一致 E={E.shape} done={done} → 从头开始")
            done, E = 0, None
        else:
            E = E.astype(np.float32)
    else:
        E = None
    print(f"总计 {n}，已完成 {done}", flush=True)
    if done >= n:
        print("全部完成")
        return

    from torchreid.reid.utils import FeatureExtractor
    ext = FeatureExtractor(model_name="osnet_x1_0", model_path=OSNET, device="cpu")
    torch.set_num_threads(12)
    print("extractor ready", flush=True)

    t0 = time.time()
    buf, idx = [], []
    newE = [] if E is None else [E]

    def flush():
        nonlocal buf, idx
        if not buf:
            return
        f = ext(buf)
        arr = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
        arr = arr.astype(np.float32)
        arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)
        newE.append(arr)
        idx.clear()
        buf = []

    for i in range(done, n):
        p = CROPS / meta[i]["file"]
        im = cv2.imread(str(p))
        if im is None:
            im = np.zeros((320, 160, 3), dtype=np.uint8)
        buf.append(im[:, :, ::-1])          # BGR -> RGB
        if len(buf) >= BS:
            flush()
            cur = i + 1
            if (cur - done) % 320 < BS:
                Cat = np.concatenate(newE).astype(np.float32)
                np.save(ep, Cat)
                dp.write_text(str(cur))
                sp = (time.time() - t0) / max(cur - done, 1)
                print(f"  {cur}/{n}  {sp*1000:.0f} ms/crop  剩余 {(n-cur)*sp/60:.1f} min",
                      flush=True)
    flush()
    Cat = np.concatenate(newE).astype(np.float32)
    np.save(ep, Cat)
    dp.write_text(str(n))
    print(f"完成 {Cat.shape}，用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
