# -*- coding: utf-8 -*-
"""dl_gcs.py — GCS 多线程分块下载（RF-DETR 训练权重）。

用法：python dl_gcs.py
"""
import hashlib
import os
import threading
import time
import urllib.request

JOBS = [
    ("https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
     "rf-detr-small.pth", "fb37061c1af7bace359c91b723a8d5c1"),
    ("https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
     "rf-detr-medium.pth", "7223f764a87b863f02eb8d52bf0ce2ee"),
]


def download(url, out, md5_expected=None, nthreads=8, chunk_mb=16):
    if os.path.exists(out):
        print(f"[dl] {out} exists, skip")
        return
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers["Content-Length"])
    print(f"[dl] {out}: {total/1e6:.1f} MB, {nthreads} threads", flush=True)
    chunk = chunk_mb * 1_000_000
    ranges = [(i, min(i + chunk - 1, total - 1)) for i in range(0, total, chunk)]
    parts = {}
    lock = threading.Lock()
    state = {"done": 0}
    t0 = time.time()

    def worker(idx, start, end):
        for attempt in range(6):
            try:
                rq = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(rq, timeout=90) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} != {end-start+1}")
                with lock:
                    parts[idx] = data
                    state["done"] += len(data)
                return
            except Exception as e:
                print(f"[dl] part {idx} attempt {attempt}: {e}", flush=True)
                time.sleep(3)
        raise RuntimeError(f"part {idx} failed permanently")

    threads = []
    for idx, (s, e) in enumerate(ranges):
        t = threading.Thread(target=worker, args=(idx, s, e), daemon=True)
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        time.sleep(8)
        el = time.time() - t0
        print(f"  {state['done']/1e6:.0f}/{total/1e6:.0f} MB "
              f"{state['done']/el/1e3:.0f} KB/s", flush=True)
    for t in threads:
        t.join()

    tmp = out + ".part"
    with open(tmp, "wb") as f:
        for idx in range(len(ranges)):
            f.write(parts[idx])
    os.replace(tmp, out)
    el = time.time() - t0
    sz = os.path.getsize(out)
    print(f"[dl] done {out} {sz/1e6:.1f} MB in {el:.0f}s = {sz/el/1e3:.0f} KB/s", flush=True)
    if md5_expected:
        h = hashlib.md5()
        with open(out, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                h.update(b)
        got = h.hexdigest()
        print(f"[dl] md5 {got} vs {md5_expected} -> "
              f"{'OK' if got == md5_expected else 'MISMATCH!!!'}", flush=True)


if __name__ == "__main__":
    for url, out, md5 in JOBS:
        download(url, out, md5)
    print("[dl] ALL DONE", flush=True)
