# -*- coding: utf-8 -*-
"""人脸裁剪 → C-RADIOv4→EmbeddingGemma 桥接编码（远程 GPU）。

目的：把「人脸线」的向量落到**与主链身体向量同一个 768d 空间** ——
复用 FastEncoder（C-RADIOv4 spatial → ridge W → EmbeddingGemma2 → mean-pool）。
这样身体/人脸不再是两个不可比的空间（身体=Gemma 768d，人脸=ArcFace 512d），
而是同一空间的两个模态 → 融合变成「同空间证据合并」。

用法（远程，embed_venv）：
  cd /home/r4900g3/yhh/human-count/crosscam_run
  ~/yhh/human-count/embed_venv/bin/python face_gemma_encode_remote.py \
      --manifest /home/r4900g3/face_manifest_p0c.json \
      --video-dir /home/r4900g3/yhh/human-count/crosscam/in \
      --out /home/r4900g3/face_emb_gemma_p0c.npz --device cuda:1

输出 npz：keys[str] / vecs[N,768]（L2 归一）/ n_used[N] / pad / out_size
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

CROSS = "/home/r4900g3/yhh/human-count/crosscam_run"
sys.path.insert(0, CROSS)


def square_crop(img, box, pad=0.25, out_size=224):
    """人脸框 → 正方形扩展裁切 → 224² RGB（与身体裁剪同口径的方形输入）"""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + pad)
    side = max(side, 8.0)
    a = int(round(cx - side / 2)); b = int(round(cy - side / 2))
    c = int(round(cx + side / 2)); d = int(round(cy + side / 2))
    a2, b2 = max(0, a), max(0, b)
    c2, d2 = min(W, c), min(H, d)
    if c2 - a2 < 4 or d2 - b2 < 4:
        return None
    crop = img[b2:d2, a2:c2]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def read_frames(cam, frames, video_dir):
    """顺序 grab，只 retrieve 需要的帧。返回 {frame: img}"""
    vp = os.path.join(video_dir, f"{cam}.mp4")
    need = set(int(f) for f in frames)
    if not need:
        return {}
    hi = max(need)
    cap = cv2.VideoCapture(vp)
    out, fr = {}, 0
    try:
        while fr <= hi:
            if not cap.grab():
                break
            if fr in need:
                ok, img = cap.read()
                if ok:
                    out[fr] = img
            fr += 1
    finally:
        cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--video-dir", default="/home/r4900g3/yhh/human-count/crosscam/in")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pad", type=float, default=0.25, help="人脸框方形扩展比例")
    ap.add_argument("--out-size", type=int, default=224)
    ap.add_argument("--dump-crops", type=int, default=0, help="额外导出 N 张裁剪图做目视校验")
    ap.add_argument("--dump-dir", default="/home/r4900g3/facecrop_check")
    args = ap.parse_args()

    from embed_fast import FastEncoder  # noqa: E402

    with open(args.manifest, encoding="utf-8") as f:
        man = json.load(f)
    items = man["items"]
    print(f"[manifest] tag={man.get('tag')} tids={len(items)} shots={man.get('n_shots')}",
          flush=True)

    # 按机位聚帧
    by_cam = {}
    for it in items:
        by_cam.setdefault(it["cam"], []).append(it)

    t0 = time.time()
    keys, vecs, n_used = [], [], []
    dump_left = args.dump_crops
    if dump_left:
        os.makedirs(args.dump_dir, exist_ok=True)
    pending = []      # (key, crop_rgb)
    for cam, its in sorted(by_cam.items()):
        frames = [s["frame"] for it in its for s in it["shots"]]
        fr_map = read_frames(cam, frames, args.video_dir)
        for it in its:
            crops = []
            for s in it["shots"]:
                img = fr_map.get(int(s["frame"]))
                if img is None:
                    continue
                cr = square_crop(img, s["face"], pad=args.pad, out_size=args.out_size)
                if cr is None:
                    continue
                if dump_left > 0:
                    p = os.path.join(args.dump_dir, f"{it['key'].replace('|', '_')}_{s['frame']}.jpg")
                    cv2.imwrite(p, cv2.cvtColor(cr, cv2.COLOR_RGB2BGR))
                    dump_left -= 1
                crops.append(cr)
            if crops:
                pending.append((it["key"], crops))
        print(f"[read] {cam}: items={len(its)} frames={len(fr_map)}  {time.time()-t0:.0f}s",
              flush=True)

    enc = FastEncoder(device=args.device, batch=args.batch)
    flat = [c for _, cs in pending for c in cs]
    print(f"[encode] {len(flat)} face crops ...", flush=True)
    te = time.time()
    E = enc.encode(flat) if flat else np.zeros((0, 768), np.float32)
    print(f"[encode] done in {time.time()-te:.1f}s "
          f"({1000*(time.time()-te)/max(1,len(flat)):.1f} ms/crop)", flush=True)

    i = 0
    for key, cs in pending:
        v = E[i:i + len(cs)]
        i += len(cs)
        m = v.mean(0)
        m = m / (np.linalg.norm(m) + 1e-9)
        keys.append(key); vecs.append(m); n_used.append(len(cs))

    np.savez_compressed(args.out, keys=np.array(keys),
                        vecs=np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 768), np.float32),
                        n_used=np.array(n_used), pad=np.array([args.pad]),
                        out_size=np.array([args.out_size]),
                        source=np.array([os.path.basename(args.manifest)]),
                        space=np.array(["cradio_scale_v3:radio_W_scale->EmbeddingGemma2(768)"]))
    print(f"[out] {len(keys)} face vectors -> {args.out}", flush=True)
    if args.dump_crops:
        print(f"[dump] crops n={args.dump_crops} -> {args.dump_dir}", flush=True)


if __name__ == "__main__":
    main()
