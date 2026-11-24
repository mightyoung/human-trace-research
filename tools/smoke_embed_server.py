# -*- coding: utf-8 -*-
"""embed_server 冒烟测试（远程运行）：
  1) 从 318 视频准备 2 轨 × 8 帧 crops（用生产 prep_crop 同构逻辑）
  2) GET /health
  3) POST /encode_tracks（服务端编码）
  4) 直连 FastEncoder 对照（同 W 目录）→ 逐轨 cos 一致性
  5) 并发 2 请求（验证串行化不崩）
运行环境：远程 embed_venv；需先启动 embed_server（指定相同 --w-dir）
  ~/yhh/human-count/embed_venv/bin/python smoke_embed_server.py --w-dir cradio_scale_v4
"""
import argparse
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

ROOT = "/home/r4900g3/yhh/human-count/crosscam_run"
DATA = "/share/data4/yhh/data/human-count"
PORT = 8791
OUT = "/tmp/embed_smoke"


def post(path, obj, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path, timeout=30):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def prep_crop(frame, box, out_h=224, pad=0.08):
    """与生产 cradio_scale_remote.prep_crop 完全一致（BGR in → RGB out）"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad)); Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad)); Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
        return None
    s = out_h / c.shape[0]
    nw = max(1, int(round(c.shape[1] * s)))
    c = cv2.resize(c, (nw, out_h))
    if nw >= out_h:
        off = (nw - out_h) // 2
        c = c[:, off:off + out_h]
    else:
        padw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-dir", default="cradio_scale_v4")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    # ---------- 1) 准备 crops ----------
    os.makedirs(OUT, exist_ok=True)
    tracks = {}
    for line in open(f"{DATA}/inference-out-clean/318_detections.jsonl", encoding="utf-8"):
        r = json.loads(line)
        for t, b in zip(r["track_ids"], r["boxes"]):
            if t is not None and int(t) >= 0:
                tracks.setdefault(int(t), []).append((r["frame"], b))
    # 选 2 条 ≥8 帧的轨
    cand = sorted([t for t, v in tracks.items() if len(v) >= 8])[:2]
    assert len(cand) == 2, f"crops 候选不足: {cand}"
    cap = cv2.VideoCapture(f"{DATA}/video-6way/318.mp4")
    req_tracks = []
    for tid in cand:
        lst = sorted(tracks[tid])
        idxs = np.linspace(0, len(lst) - 1, 8).astype(int)
        paths = []
        for k, i in enumerate(idxs):
            fr, box = lst[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, frame = cap.read()
            assert ok, (tid, fr)
            crop = prep_crop(frame, box)
            assert crop is not None
            p = f"{OUT}/c{tid}_{k}.jpg"
            cv2.imwrite(p, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))   # 文件语义=BGR
            paths.append(p)
        req_tracks.append({"cam": "318", "tid": tid, "crops": paths})
    cap.release()
    print(f"[smoke] crops 就绪: {[(t['tid'], len(t['crops'])) for t in req_tracks]}", flush=True)

    # ---------- 2) health ----------
    h = get("/health")
    print(f"[health] {json.dumps(h, ensure_ascii=False)}", flush=True)

    # ---------- 3) encode_tracks ----------
    t0 = time.perf_counter()
    res = post("/encode_tracks", {"tracks": req_tracks, "batch": 8})
    dt = time.perf_counter() - t0
    embs_srv = {r["tid"]: np.array(r["emb"]) for r in res["tracks"]}
    print(f"[server] encode_tracks: {[(r['tid'], r['n_crops']) for r in res['tracks']]} "
          f"| {dt:.2f}s 总（含 HTTP）| ms_per_img={res['ms_per_img']}", flush=True)

    # ---------- 4) 直连对照 ----------
    from embed_fast import FastEncoder
    enc = FastEncoder(root=ROOT, device=args.device, w_dir=args.w_dir)
    embs_dir = {}
    for t in req_tracks:
        crops = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in t["crops"]]
        e = enc.encode(crops).mean(axis=0)
        e = e / (np.linalg.norm(e) + 1e-9)
        embs_dir[t["tid"]] = e
    cos = {}
    for tid in embs_srv:
        a, b = embs_srv[tid], embs_dir[tid]
        cos[tid] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"[对照] 直连 FastEncoder cos: { {k: round(v, 6) for k, v in cos.items()} }",
          flush=True)

    # ---------- 5) 并发 2 请求 ----------
    import threading
    errs = []

    def worker():
        try:
            r = post("/encode", {"images": req_tracks[0]["crops"][:4], "batch": 4})
            assert len(r["embeddings"]) == 4
        except Exception as ex:
            errs.append(f"{type(ex).__name__}: {ex}")

    ths = [threading.Thread(target=worker) for _ in range(2)]
    t0 = time.perf_counter()
    [t.start() for t in ths]
    [t.join() for t in ths]
    print(f"[并发] 2×4 图: {time.perf_counter()-t0:.2f}s errs={errs}", flush=True)

    # ---------- 6) store 接口（如果启用） ----------
    try:
        r = post("/store", {"items": [{
            "key": "smoke|318|1", "emb": embs_dir[cand[0]].tolist(),
            "cam": "318", "date": "smoke", "tid": 1}]})
        print(f"[store] {json.dumps(r)}", flush=True)
    except Exception as ex:
        print(f"[store] 跳过（{str(ex)[:80]}）", flush=True)

    # ---------- 汇总 ----------
    summary = {"health": h, "encode_ms_per_img": res["ms_per_img"],
               "cos_vs_direct": cos, "concurrent_errs": errs,
               "n_crops": sum(len(t["crops"]) for t in req_tracks)}
    json.dump(summary, open(f"{OUT}/smoke_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    ok = all(v > 0.999 for v in cos.values()) and not errs
    print(f"[smoke] {'PASS' if ok else 'CHECK'} -> {OUT}/smoke_summary.json", flush=True)


if __name__ == "__main__":
    main()
