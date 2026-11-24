# -*- coding: utf-8 -*-
"""C-RADIOv4 稠密（SAM 对齐）能力演示：
  1) 帧级 token KMeans 聚类分割（看稠密特征能否自动分出人/玻璃/地面/家具等区域）
  2) 人物相似度图（以某检测框为种子 token，余弦相似度全图热力）
"""
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
RADIO_DIR = os.path.join(ROOT, "cradio", "C-RADIOv4-SO400M")
OUT = os.path.join(ROOT, "cradio_scale")
DATA = "/share/data4/yhh/data/human-count"
DEV = os.environ.get("FUSION_DEV", "cuda:0")
os.makedirs(OUT, exist_ok=True)


def load_radio():
    from transformers import AutoModel, CLIPImageProcessor
    m = AutoModel.from_pretrained(RADIO_DIR, trust_remote_code=True)
    m = m.to(DEV).eval()
    proc = CLIPImageProcessor.from_pretrained(RADIO_DIR)
    return m, proc


def main():
    m, proc = load_radio()
    print("[dense] RADIO 就绪", flush=True)
    jobs = [("316", 6000), ("318", 3000)]
    for cam, fnum in jobs:
        cap = cv2.VideoCapture(f"{DATA}/video-6way/{cam}.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        cap.release()
        assert ok, (cam, fnum)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img512 = cv2.resize(img, (512, 512))
        pix = proc(images=[Image.fromarray(img512)], return_tensors="pt",
                   do_resize=False, do_center_crop=False,
                   do_normalize=False).pixel_values.to(DEV)
        with torch.no_grad():
            out = m(pix)
        sp = out[1][0]                                  # [1024, 1152]
        T = torch.nn.functional.normalize(sp, dim=-1)

        # --- 1) KMeans 分割 ---
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=6, n_init=4, random_state=0).fit(
            T.cpu().numpy())
        lab = km.labels_.reshape(32, 32)
        palette = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25],
                            [0, 130, 200], [245, 130, 48], [145, 30, 180]],
                           np.uint8)
        big = cv2.resize(palette[lab], (frame.shape[1], frame.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        overlay = (0.55 * frame + 0.45 * cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
                   ).astype(np.uint8)
        h = 540
        w1 = int(frame.shape[1] * h / frame.shape[0])
        left = cv2.resize(frame, (w1, h))
        right = cv2.resize(overlay, (w1, h))
        cv2.imwrite(f"{OUT}/dense_cluster_{cam}_f{fnum}.jpg",
                    np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 90])

        # --- 2) 人物相似度图 ---
        det_p = f"{DATA}/inference-out/{cam}_detections.jsonl"
        box = None
        with open(det_p) as f:
            for line in f:
                r = json.loads(line)
                if int(r["frame"]) == fnum and r.get("boxes"):
                    bs = [b for b in r["boxes"] if (b[2] - b[0]) * (b[3] - b[1]) > 500]
                    if bs:
                        box = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    break
        if box is not None:
            W, H = frame.shape[1], frame.shape[0]
            cx = int(min(31, max(0, ((box[0] + box[2]) / 2) / W * 32)))
            cy = int(min(31, max(0, ((box[1] + box[3]) / 2) / H * 32)))
            ti = cy * 32 + cx
            q = T[ti]
            smap = (T @ q).reshape(32, 32).cpu().numpy()
            smap = ((smap - smap.min()) / (smap.max() - smap.min() + 1e-9) * 255
                    ).astype(np.uint8)
            heat = cv2.applyColorMap(
                cv2.resize(smap, (frame.shape[1], frame.shape[0])),
                cv2.COLORMAP_JET)
            overlay2 = (0.5 * frame + 0.5 * heat).astype(np.uint8)
            cv2.rectangle(overlay2, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 255, 0), 3)
            cv2.imwrite(f"{OUT}/dense_personsim_{cam}_f{fnum}.jpg",
                        np.hstack([left, cv2.resize(overlay2, (w1, h))]),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[dense] {cam} f{fnum} done", flush=True)
    print("[dense] DONE", flush=True)


if __name__ == "__main__":
    main()
