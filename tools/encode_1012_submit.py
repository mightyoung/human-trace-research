#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1012 切段嵌入编码提交（运行在远程；须先启动 embed_server）。

对 12 段 × {a,b} 分片提交 /encode_video_tracks（cradio v4 FastEncoder，GPU1），
轮询至全部完成并打印汇总。产物：
  {BASE}/inference-out-1012-emb/{seg}_emb_{ab}.npz   键 = "{seg}#{tid}"
"""
import json
import os
import time
import urllib.request

BASE = "/share/data4/yhh/data/human-count"
VIDEO = f"{BASE}/video-1012"
AB_DIR = f"{BASE}/inference-out-1012-ab"
OUT = f"{BASE}/inference-out-1012-emb"
SERVER = "http://127.0.0.1:8791"

SEGS = [
    "320_train_514", "318_train_525", "323_train_510", "305_train_529",
    "320_val_519", "318_val_525", "323_val_538", "305_val_517",
    "320_test_518", "318_test_534", "323_test_527", "305_test_544",
]


def post(path, obj):
    req = urllib.request.Request(SERVER + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(path):
    return json.load(urllib.request.urlopen(SERVER + path, timeout=60))


def main():
    os.makedirs(OUT, exist_ok=True)
    # 健康检查
    h = get("/health")
    print(f"[health] {h}")

    job_ids = []
    for seg in SEGS:
        for ab in ("a", "b"):
            jp = os.path.join(AB_DIR, f"{seg}_{ab}.jsonl")
            if not os.path.exists(jp):
                print(f"[skip] {jp} missing")
                continue
            out_npz = os.path.join(OUT, f"{seg}_emb_{ab}.npz")
            if os.path.exists(out_npz):
                print(f"[skip] {out_npz} exists")
                continue
            r = post("/encode_video_tracks", {
                "cam": f"{seg}_{ab}",
                "video": os.path.join(VIDEO, f"{seg}.mp4"),
                "jsonl": jp,
                "prefix": f"{seg}#",
                "samples": 8, "min_frames": 3, "batch": 16, "chunk_tracks": 4,
                "out_npz": out_npz,
            })
            print(f"[submit] {seg}_{ab} -> {r}")
            job_ids.append(r.get("job_id") or r.get("id"))
            time.sleep(0.3)

    print(f"[submit] {len(job_ids)} jobs queued")
    # 轮询
    t0 = time.time()
    while True:
        st = {}
        for jid in job_ids:
            try:
                j = get(f"/jobs/{jid}")
                st[jid] = j
            except Exception as e:
                st[jid] = {"status": f"err:{e}"}
        done = sum(1 for j in st.values() if j.get("status") in ("done", "error"))
        running = [j for j in st.values() if j.get("status") == "running"]
        el = time.time() - t0
        print(f"[poll {el:.0f}s] done={done}/{len(job_ids)} running={len(running)}")
        if done >= len(job_ids):
            break
        time.sleep(25)

    print("\n===== SUMMARY =====")
    n_ok = 0
    for jid in job_ids:
        j = get(f"/jobs/{jid}")
        cam = j.get("cam")
        if j.get("status") == "done":
            res = j.get("result") or {}
            pr = j.get("progress") or {}
            print(f"OK   {cam}: {res.get('n_embs', '?')} embs, {pr.get('n_crops')} crops, "
                  f"{pr.get('ms_per_img')} ms/img")
            n_ok += 1
        else:
            print(f"FAIL {cam}: {j.get('status')} {str(j.get('error'))[:200]}")
    print(f"[done] {n_ok}/{len(job_ids)} ok")


if __name__ == "__main__":
    main()
