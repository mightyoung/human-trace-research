# -*- coding: utf-8 -*-
"""embed_server 并发吞吐基准：N 个客户端并发打 /encode_tracks（同步请求）。

用法（远程）:
  ~/yhh/human-count/embed_venv/bin/python svc_bench_remote.py --clients 1 3 6
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

import cv2
import numpy as np

CROPDIR = "/tmp/svc_bench"


def make_crops(n):
    os.makedirs(CROPDIR, exist_ok=True)
    paths = []
    rng = np.random.randint(0, 255, (n, 224, 224, 3), dtype=np.uint8)
    for i in range(n):
        p = os.path.join(CROPDIR, f"c{i}.jpg")
        if not os.path.exists(p):
            cv2.imwrite(p, rng[i])
        paths.append(p)
    return paths


def client_run(url, paths, n_req):
    lat = []
    for r in range(n_req):
        payload = {"tracks": [{"cam": "bench", "tid": r, "crops": paths}], "batch": 16}
        req = urllib.request.Request(
            url + "/encode_tracks", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        lat.append(time.perf_counter() - t0)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8791")
    ap.add_argument("--clients", nargs="+", type=int, default=[1, 3, 6])
    ap.add_argument("--req-per-client", type=int, default=8)
    ap.add_argument("--crops-per-req", type=int, default=32)
    args = ap.parse_args()

    paths = make_crops(args.crops_per_req)
    print(f"crops/req={len(paths)}  req/client={args.req_per_client}")
    out = []
    for nc in args.clients:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(nc) as ex:
            futs = [ex.submit(client_run, args.url, paths, args.req_per_client)
                    for _ in range(nc)]
            all_lat = []
            for f in futs:
                all_lat += f.result()
        dt = time.perf_counter() - t0
        n_imgs = nc * args.req_per_client * len(paths)
        rec = {"clients": nc, "imgs": n_imgs, "wall_s": round(dt, 1),
               "img_per_s": round(n_imgs / dt, 1),
               "lat_p50_ms": round(float(np.percentile(all_lat, 50)) * 1000),
               "lat_p90_ms": round(float(np.percentile(all_lat, 90)) * 1000)}
        out.append(rec)
        print(f"clients={nc}: {n_imgs} imgs in {dt:.1f}s -> {rec['img_per_s']} img/s | "
              f"lat p50={rec['lat_p50_ms']}ms p90={rec['lat_p90_ms']}ms", flush=True)
    with open("/tmp/svc_bench_result.json", "w") as f:
        json.dump(out, f, indent=1)
    print("-> /tmp/svc_bench_result.json")


if __name__ == "__main__":
    main()
