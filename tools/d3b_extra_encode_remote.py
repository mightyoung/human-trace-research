# -*- coding: utf-8 -*-
"""d3bridge 补编码（4+1 轨）：318 {53,82,98,135} + 305 {12}。

背景：v4 修正评估需要恢复被误剔的评估轨。cradio 侧已补（j4/j6），
d3bridge 侧用同一 patched 桥接链路补编码，W 取 dinov3_scale_v4/d3bW_scale.npy。

用法（远程 crosscam_run/）:
  FUSION_DEV=cuda:0 D3B_OUT=dinov3_scale_v4 python d3b_extra_encode_remote.py
"""
import os
import sys
import time

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dinov3_bridge_remote as d3b  # noqa: E402

DATA = d3b.DATA

JOBS = [
    ("318", f"{DATA}/inference-out-extra/318_subset_4tracks.jsonl", "a",
     f"{DATA}/inference-out-extra/318_extra4_emb_d3b.npz"),
    ("305", f"{DATA}/inference-out-extra/305_subset_1track.jsonl", "v305_",
     f"{DATA}/inference-out-extra/305_extra1_emb_d3b.npz"),
]


def main():
    st, am = d3b.load_gemma()
    m_d3 = d3b.load_d3b()
    from transformers import AutoProcessor
    gproc = AutoProcessor.from_pretrained(d3b.EMB_PATH)
    wp = os.path.join(d3b.OUT, "d3bW_scale.npy")
    Wt = torch.from_numpy(np.load(wp).astype(np.float32)).to(d3b.DEV)
    am.get_image_features = d3b._make_patched(am, Wt, m_d3, gproc)
    print(f"[extra] 模型就绪 DEV={d3b.DEV} W={wp}", flush=True)

    def encode_track(cap, tf):
        crops = d3b.sample_crops(cap, tf)
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

    for cam, jsonl, prefix, out_p in JOBS:
        tracks = d3b.load_tracks(jsonl)
        cap = cv2.VideoCapture(f"{DATA}/video-6way/{cam}.mp4")
        out = {}
        t0 = time.time()
        for tid in sorted(tracks):
            e = encode_track(cap, tracks[tid])
            if e is None:
                print(f"[extra] {cam}#{tid} skip (no crops)", flush=True)
                continue
            out[f"{prefix}{tid}"] = e
        cap.release()
        np.savez(out_p, **out)
        print(f"[extra] {cam} DONE {len(out)} tracks {time.time()-t0:.0f}s -> {out_p}",
              flush=True)
    print("[extra] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
