#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""人脸嵌入 × gperson 聚类可行性（E 路线，只读 GIL，不改任何表）

对每个有可用脸的 tid：取 q 最高 ≤3 帧 → 5 点对齐到 112 模板 → w600k_mbf(ONNX) 提嵌入
（L2 归一后取均值再归一）→ 统计 同 gperson / 跨 gperson 对的余弦分布 → 输出候选合并清单。

用法：python tools/gil_face_embed.py --tag p0b
输出：analysis/face/face_embed_{tag}.json + stdout 统计
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gil_face_capture import FACE_DIR  # noqa: E402
from gil_face_reassoc import iter_scanned  # noqa: E402

MODEL = os.path.join(ROOT, "models", "face", "w600k_mbf.onnx")
TPL112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def align112(img, kps5):
    M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), TPL112, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p0b")
    ap.add_argument("--per-tid", type=int, default=3, help="每 tid 取 q 最高 N 帧平均")
    ap.add_argument("--pack", default="synced10", choices=["synced10", "long35"],
                    help="素材包（必须与 gil_face_capture 一致，否则读错视频）")
    ap.add_argument("--ckpt", default="mbf", choices=["mbf", "r50"],
                    help="识别模型：mbf=w600k_mbf(13MB, 快) / r50=w600k_r50(174MB, 强)。"
                         "小脸（中位 52px）上 r50 明显更稳，做锚点首选 r50")
    ap.add_argument("--dump-vec", action="store_true",
                    help="★ 落盘 512d ArcFace 向量（face_embed_vec_{tag}.npz），"
                         "供双线融合/复核复用；不改变既有 json 产物")
    args = ap.parse_args()

    from gil_face_capture import set_pack
    set_pack(args.pack)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(4, (os.cpu_count() or 8) // 2)
    ckpt = MODEL if args.ckpt == "mbf" else os.path.join(ROOT, "models", "face", "w600k_r50.onnx")
    sess = ort.InferenceSession(ckpt, sess_options=so, providers=["CPUExecutionProvider"])
    print("ckpt:", os.path.basename(ckpt), flush=True)
    iname = sess.get_inputs()[0].name
    print("model:", os.path.basename(ckpt), "input:", sess.get_inputs()[0].shape,
          "output:", sess.get_outputs()[0].shape, flush=True)

    with open(os.path.join(FACE_DIR, f"face_shot_{args.tag}.json"), encoding="utf-8") as f:
        shots = json.load(f)["shots"]
    with open(os.path.join(FACE_DIR, f"face_best_{args.tag}.json"), encoding="utf-8") as f:
        best = json.load(f)
    gp_label = best.get("gp_label", {})

    groups = defaultdict(list)
    for s in shots:
        if s.get("usable") and s.get("tid") is not None:
            groups[(s["cam"], s["tid"])].append(s)

    t0 = time.time()
    embs, infos = {}, {}
    for (cam, tid), lst in sorted(groups.items()):
        lst.sort(key=lambda s: -s["q"])
        use = lst[: args.per_tid]
        img_by = dict(iter_scanned([s["frame"] for s in use], cam))
        vecs = []
        for s in use:
            img = img_by.get(s["frame"])
            if img is None:
                continue
            kps = np.array(s["kps"], np.float32).reshape(5, 2)
            ali = align112(img, kps)
            if ali is None:
                continue
            blob = ((ali[:, :, ::-1].astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
            v = sess.run(None, {iname: blob})[0][0]
            v = v / (np.linalg.norm(v) + 1e-9)
            vecs.append(v)
        if not vecs:
            continue
        e = np.mean(vecs, axis=0)
        e = e / (np.linalg.norm(e) + 1e-9)
        key = f"{cam}|{tid}"
        embs[key] = e
        s0 = lst[0]
        infos[key] = dict(cam=cam, tid=tid, q=s0["q"], n_used=len(vecs),
                          gperson_id=s0.get("gperson_id"),
                          gp_label=gp_label.get(str(s0.get("gperson_id"))) if s0.get("gperson_id") else None)
    print(f"embedded {len(embs)} tids in {time.time()-t0:.0f}s", flush=True)

    keys = sorted(embs)
    E = np.stack([embs[k] for k in keys])
    if args.dump_vec:
        vp = os.path.join(FACE_DIR, f"face_embed_vec_{args.tag}.npz")
        np.savez_compressed(vp, keys=np.array(keys), vecs=E.astype(np.float32),
                            model=np.array([os.path.basename(MODEL)]),
                            dim=np.array([E.shape[1]]))
        print(f"[dump-vec] {E.shape} -> {vp}", flush=True)
    S = E @ E.T
    same, cross = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            ga, gb = infos[a]["gperson_id"], infos[b]["gperson_id"]
            if ga is None or gb is None:
                continue
            rec = dict(a=a, b=b, cos=round(float(S[i, j]), 4),
                       ga=ga, gb=gb, la=infos[a]["gp_label"], lb=infos[b]["gp_label"])
            (same if ga == gb else cross).append(rec)

    def qs(v, name):
        v = np.array([x["cos"] for x in v]) if v else np.array([0.0])
        out = dict(n=len(v), mean=round(float(v.mean()), 4), p50=round(float(np.percentile(v, 50)), 4))
        for p in (5, 10, 25, 75, 90, 95, 99):
            out[f"p{p}"] = round(float(np.percentile(v, p)), 4)
        out["min"] = round(float(v.min()), 4)
        out["max"] = round(float(v.max()), 4)
        print(f"[{name}] n={out['n']} mean={out['mean']} p5={out['p5']} p50={out['p50']} "
              f"p95={out['p95']} max={out['max']}", flush=True)
        return out

    s_stat = qs(same, "same-gperson")
    c_stat = qs(cross, "cross-gperson")
    # 简易分离度：用 same 的 p5 和 cross 的 p95 观察重叠
    cross_sorted = sorted(cross, key=lambda x: -x["cos"])
    print("--- top-12 cross-gperson pairs (候选合并) ---")
    for r in cross_sorted[:12]:
        print(f"  {r['a']} <-> {r['b']} cos={r['cos']} gp={r['la']} vs {r['lb']}")
    same_sorted = sorted(same, key=lambda x: x["cos"])
    print("--- bottom-8 same-gperson pairs (可疑拆并) ---")
    for r in same_sorted[:8]:
        print(f"  {r['a']} <-> {r['b']} cos={r['cos']} gp={r['la']}")

    out = dict(meta=dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), tag=args.tag,
                         model=os.path.basename(MODEL), per_tid_frames=args.per_tid,
                         note="相似度仅为可行性证据，不做任何 GIL 变更；阈值待 val 校准"),
               n_tids=len(keys), stats=dict(same=s_stat, cross=c_stat),
               per_tid={k: infos[k] for k in keys},
               same_pairs=sorted(same, key=lambda x: x["cos"]),
               cross_top=[r for r in cross_sorted[:30]])
    with open(os.path.join(FACE_DIR, f"face_embed_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"-> face_embed_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
