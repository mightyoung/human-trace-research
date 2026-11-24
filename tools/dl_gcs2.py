# -*- coding: utf-8 -*-
"""dl_gcs2.py — GCS 多线程小块下载（2MB 短请求队列模式，16 线程）。

教训：16MB 大块长连接会停滞（观测累计 2MB 后卡死）；2MB 短请求可稳定 200-300KB/s/请求。
策略：2MB 块 + 16 线程动态队列 + 失败重试 + 断点续传（.parts 目录）。
"""
import hashlib
import os
import queue
import threading
import time
import urllib.request

JOBS = [
    ("https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
     "rf-detr-small.pth", "fb37061c1af7bace359c91b723a8d5c1"),
    ("https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
     "rf-detr-medium.pth", "7223f764a87b863f02eb8d52bf0ce2ee"),
    ("https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
     "rf-detr-large-2026.pth", "5cb72153541cbcb9aa6efa26222acc75"),
]

CHUNK = 2_000_000
NTHREADS = 16


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def download(url, out, md5_expected=None, nthreads=NTHREADS, chunk=CHUNK):
    if os.path.exists(out):
        print(f"[dl] {out} exists, skip", flush=True)
        return
    total = head_size(url)
    nchunks = (total + chunk - 1) // chunk
    pdir = out + ".parts"
    os.makedirs(pdir, exist_ok=True)
    print(f"[dl] {out}: {total/1e6:.1f} MB = {nchunks} chunks x {chunk/1e6:.0f} MB, "
          f"{nthreads} threads", flush=True)

    q = queue.Queue()
    for i in range(nchunks):
        pf = os.path.join(pdir, f"{i:05d}")
        if os.path.exists(pf) and os.path.getsize(pf) == min(chunk, total - i * chunk):
            continue  # 已有完整块（断点续传）
        q.put(i)

    t0 = time.time()
    lock = threading.Lock()
    state = {"done": 0, "fail": 0}

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            start = i * chunk
            end = min(start + chunk - 1, total - 1)
            pf = os.path.join(pdir, f"{i:05d}")
            for attempt in range(6):
                try:
                    rq = urllib.request.Request(url,
                                                headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(rq, timeout=25) as r:
                        data = r.read()
                    if len(data) != end - start + 1:
                        raise IOError(f"short {len(data)}")
                    with open(pf + ".tmp", "wb") as f:
                        f.write(data)
                    os.replace(pf + ".tmp", pf)
                    with lock:
                        state["done"] += 1
                    break
                except Exception as e:
                    if attempt == 5:
                        with lock:
                            state["fail"] += 1
                        print(f"[dl] chunk {i} FAILED: {e}", flush=True)
                    time.sleep(1.5)
            q.task_done()

    # 监控线程
    stop = threading.Event()

    def monitor():
        while not stop.wait(10):
            el = time.time() - t0
            d = state["done"]
            print(f"  {d}/{nchunks} chunks  {d*chunk/1e6:.0f}/{total/1e6:.0f} MB  "
                  f"{d*chunk/el/1e3:.0f} KB/s  fail={state['fail']}", flush=True)

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(nthreads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()

    if state["fail"] and state["fail"] > 0:
        print(f"[dl] {state['fail']} chunks failed; rerun to resume", flush=True)
        return

    # 拼接
    tmp = out + ".concat"
    with open(tmp, "wb") as fo:
        for i in range(nchunks):
            with open(os.path.join(pdir, f"{i:05d}"), "rb") as fi:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
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
