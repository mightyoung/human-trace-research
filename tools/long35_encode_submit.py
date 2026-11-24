# -*- coding: utf-8 -*-
"""长窗编码：向 embed_server 并发提交 3 路 /encode_video_tracks → 轮询直至完成。

用法（远程，服务器已启动）:
  cd ~/yhh/human-count/crosscam_run
  nohup ~/yhh/human-count/embed_venv/bin/python long35_encode_submit.py > long35_encode_submit.log 2>&1 &
"""
import argparse
import json
import time
import urllib.request

BASE_DATA = "/share/data4/yhh/data/human-count"
JOBS = [("316", "v316_"), ("318", "a"), ("320", "b")]


def post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8791")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    ids, t0 = [], time.time()
    for cam, prefix in JOBS:
        r = post(args.url + "/encode_video_tracks", {
            "cam": cam,
            "video": f"{BASE_DATA}/video-long35/{cam}_long35.mp4",
            "jsonl": f"{BASE_DATA}/inference-out-long35-clean/{cam}/{cam}_long35_detections.jsonl",
            "prefix": prefix, "samples": 8, "min_frames": 1,
            "batch": 16, "chunk_tracks": 4,
            "out_npz": f"{BASE_DATA}/long35-emb/{cam}_emb_long35.npz"})
        print(f"[submit] {cam} -> {r}", flush=True)
        ids.append((cam, r["job_id"]))

    done = {}
    while time.time() - t0 < args.timeout and len(done) < len(ids):
        time.sleep(20)
        st = get(args.url + "/health")
        print(f"[{time.strftime('%H:%M:%S')}] global n_img={st['n_img']} "
              f"avg={st['ms_per_img_avg']} queue={st['queue']} running={st['running']}",
              flush=True)
        for cam, jid in ids:
            if cam in done:
                continue
            j = get(args.url + f"/jobs/{jid}")
            pr = j.get("progress") or {}
            print(f"  {cam} {j['status']} {pr.get('done')}/{pr.get('total')} "
                  f"crops={pr.get('n_crops')} ms/img={pr.get('ms_per_img')} "
                  f"eta={pr.get('eta_s')}s", flush=True)
            if j["status"] in ("done", "error"):
                done[cam] = j
                print(f"[finish] {cam} {j['status']}: "
                      f"{json.dumps(j.get('result') or j.get('error'), ensure_ascii=False)}",
                      flush=True)
    print(f"[ALL] done={len(done)}/{len(ids)} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
