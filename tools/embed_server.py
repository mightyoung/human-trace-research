# -*- coding: utf-8 -*-
"""FastEncoder 常驻编码服务 v1.1（HTTP，内网）——多消费方共享一个编码器。

v1.0：模型只加载一次的常驻服务；/encode、/encode_tracks、/store、/health。
v1.1 新增（并行/服务深化）：
  - 异步长任务 POST /encode_video_tracks：服务端自行「视频抽帧→裁剪→编码→均值→落 npz」，
    调用方只给 (video, jsonl, cam, prefix)——多相机 worker 各自提交、服务端统一算力。
  - 任务状态 GET /jobs（列表）、GET /jobs/{id}（进度/ETA/结果）——nginx/断连不丢任务。
  - 多 worker 并发执行：解码（IO/CPU）与 GPU 编码重叠；GPU 串行由 gpu_lock 保证。
  - /health 扩展：队列深度、在跑数、累计 ms/图。

接口（JSON）：
  GET  /health
      → {"status":"ok","device":..,"n_req":..,"n_img":..,"ms_per_img_avg":..,
         "queue":N,"running":K,"workers":W,"jobs_total":M,"uptime_s":..,"store":bool}
  POST /encode          {"images":["/abs/a.jpg",...], "batch":16}
  POST /encode_tracks   {"tracks":[{"cam":"316","tid":12,"crops":[...]},...], "batch":16}
  POST /encode_video_tracks
      {"cam":"316","video":"/abs/316.mp4","jsonl":"/abs/316_det.jsonl",
       "prefix":"v316_","samples":8,"min_frames":2,"batch":16,"chunk_tracks":4,
       "prep":"legacy",
       "out_npz":"/abs/316_emb.npz"}
      → {"job_id":"j1","status":"queued","queue":N}
      prep: "legacy"（默认，行为不变） | "v2"（实验：质量选帧 + 2:1 原分辨率裁剪
            + 常量色 letterbox + 质量加权聚合；依赖 tools/reid_v2/，见
            docs/commit-015/PHASE_A_PLAN.md §6）
  GET  /jobs            → {"jobs":[{"id","cam","status","done","total","elapsed_s"}...]}
  GET  /jobs/j1         → {"id","cam","status","progress":{...},"result":{...},"error":...}
  POST /store           {"items":[{"key":..,"emb":[768],..}]}   （需以 --store-dir 启动）
  POST /shutdown

依赖：embed_fast.FastEncoder（同目录）；图像用 cv2 读（BGR→RGB）。
死重剥离（可选）：启动加 `--strip-dead` —— 剥离 EmbeddingGemma2 未使用的 audio/vision 塔
  （实测嵌入逐位不变、省 ~1.85GB 显存；见 docs/embgemma-strip-feasibility.md）。
启动（远程 GPU 例）：
  cd ~/yhh/human-count/crosscam_run
  nohup ~/yhh/human-count/embed_venv/bin/python embed_server.py \
      --port 8791 --device cuda:1 --w-dir cradio_scale_v4 --workers 3 \
      --jobs-log embed_server_jobs.jsonl > embed_server.log 2>&1 &
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GLOBAL = {
    "enc": None,
    "store": None,
    "store_lock": threading.Lock(),
    "gpu_lock": threading.Lock(),      # 单卡串行（FastEncoder 内部有实例状态）
    "n_req": 0, "n_img": 0, "ms_sum": 0.0,
    "t_start": time.time(),
    "device": None,
    # v1.1 任务系统
    "jobs": {},                        # job_id -> job dict
    "jobs_lock": threading.Lock(),
    "jobq": queue.Queue(),
    "workers": 0,
    "jobs_log": None,
    "job_seq": 0,
}


def _read_images(paths):
    import cv2
    imgs = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return imgs


# ---------------- 视频轨迹编码（与生产线一致的裁剪/采样） ----------------
def _prep_crop(frame, box, out_h=224, pad=0.08):
    """与 crosscam_reid / cradio_scale_remote 完全一致（BGR in → RGB out）"""
    import cv2
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad))
    Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad))
    Y2 = min(frame.shape[0], int(y2 + h * pad))
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
        c = cv2.copyMakeBorder(c, 0, 0, padw // 2, padw - padw // 2,
                               cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def _load_tracks(jsonl_path):
    """与 encode_tracks.load_tracks 完全一致"""
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((int(r["frame"]), b))
    for k in t:
        t[k].sort()
    return t


def _sample_crops(cap, tf, samples=8):
    import cv2
    import numpy as np
    idxs = np.linspace(0, len(tf) - 1, min(samples, len(tf))).astype(int)
    crops = []
    for k in idxs:
        fnum, box = tf[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = _prep_crop(frame, box)
        if rgb is not None:
            crops.append(rgb)
    return crops


# ---------------- v2 预处理（实验开关 prep=v2；默认 legacy 不启用） ----------------
def _sample_crops_v2(cap, tf, frame_boxes, samples=8, min_gap=10):
    """v2（评审 F1/F2/F4）：质量选帧 + 2:1 原分辨率裁剪 + 常量色 letterbox（喂 224² 模板编码器）。
    返回 (crops, weights)。tf: [(frame, box, score)]；frame_boxes: frame -> (N,4) 同帧全部框（算遮挡）。
    """
    import cv2
    import numpy as np
    from reid_v2.crop import border_touch, crop_person, square_letterbox
    from reid_v2.quality import QualityConfig, frame_quality, sharpness
    from reid_v2.aggregate import select_frames
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    qcfg = QualityConfig()
    geo = []
    for fnum, box, sc in tf:
        fb = frame_boxes.get(fnum)
        others = [b for b in (fb if fb is not None else []) if list(b) != list(box)]
        w, _ = frame_quality(box, np.asarray(others).reshape(-1, 4) if others else None,
                             border_touch(box, W, H), score=sc, cfg=qcfg)
        geo.append(w)
    idx = select_frames([t[0] for t in tf], geo, k=samples, min_gap=min_gap)
    crops, ws = [], []
    for k in idx:
        fnum, box, _ = tf[k]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if not ok:
            continue
        c, meta = crop_person(frame, box)
        if c is None:
            continue
        sharp = min(1.0, sharpness(c) / qcfg.sharp_ref)
        crops.append(square_letterbox(c, 224))
        ws.append(geo[k] * sharp * (1 - meta.fill_frac))
    return crops, ws


def _load_tracks_v2(jsonl_path):
    """同 _load_tracks，另返回每帧全部框（遮挡因子）与检测分。"""
    t, fb = {}, {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fr = int(r["frame"])
            boxes = r.get("boxes", []) or []
            fb[fr] = boxes
            tids = r.get("track_ids", []) or []
            scs = r.get("scores", []) or [None] * len(boxes)
            for i, b in enumerate(boxes):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((fr, b, scs[i] if i < len(scs) else None))
    for k in t:
        t[k].sort(key=lambda x: x[0])
    return t, fb


def _run_video_job(job):
    """worker 主流程：抽帧裁剪 → 分批编码 → 均值归一 → npz 落盘。"""
    import cv2
    import numpy as np

    job["status"] = "running"
    job["t_start"] = time.time()
    prog = job["progress"]

    prep = str(job.get("prep", "legacy"))
    v2 = prep == "v2"              # ★ v2 开关（默认 legacy，输出逐位不变）
    if v2:
        tracks, frame_boxes = _load_tracks_v2(job["jsonl"])
        weights_of = {}
    else:
        tracks = _load_tracks(job["jsonl"])
    tids = sorted(t for t, v in tracks.items() if len(v) >= job["min_frames"])
    prog["total"] = len(tids)

    cap = cv2.VideoCapture(job["video"])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {job['video']}")

    out = {}
    chunk = max(1, int(job.get("chunk_tracks", 4)))
    t0 = time.time()
    for i in range(0, len(tids), chunk):
        group = tids[i:i + chunk]
        all_crops, spans = [], []
        for tid in group:
            if v2:
                crops, ws = _sample_crops_v2(cap, tracks[tid], frame_boxes, job["samples"])
                weights_of[tid] = ws
            else:
                crops = _sample_crops(cap, tracks[tid], job["samples"])
            spans.append((tid, len(all_crops), len(all_crops) + len(crops)))
            all_crops += crops
        if all_crops:
            te = time.perf_counter()
            with GLOBAL["gpu_lock"]:
                embs = GLOBAL["enc"].encode(all_crops, batch=job["batch"])
            dt = time.perf_counter() - te
            GLOBAL["n_img"] += len(all_crops)
            GLOBAL["ms_sum"] += dt * 1000
            GLOBAL["n_req"] += 1
            prog["n_crops"] += len(all_crops)
            prog["_gpu_ms"] += dt * 1000
            for tid, a, b in spans:
                if b > a:
                    if v2 and sum(weights_of.get(tid, [])) > 0:
                        # 质量加权（与 C 包参考实现一致；尺度差异被后续 L2 归一吸收）
                        w_ = np.asarray(weights_of[tid], dtype=np.float32)[:, None]
                        m = (embs[a:b] * w_).sum(axis=0)
                    else:
                        m = embs[a:b].mean(axis=0)
                    m = m / (np.linalg.norm(m) + 1e-9)
                    out[f"{job['prefix']}{tid}"] = m.astype(np.float32)
                else:
                    prog["skipped"] += 1
        else:
            prog["skipped"] += len(group)
        prog["done"] += len(group)
        el = time.time() - t0
        prog["elapsed_s"] = round(el, 1)
        if prog["done"]:
            prog["eta_s"] = round(el / prog["done"] * (prog["total"] - prog["done"]), 1)
        if prog["n_crops"]:
            prog["ms_per_img"] = round(prog["_gpu_ms"] / prog["n_crops"], 2)
    cap.release()

    os.makedirs(os.path.dirname(job["out_npz"]) or ".", exist_ok=True)
    np.savez(job["out_npz"], **out)
    job["result"] = {
        "n_tracks": len(out), "total_tracks": prog["total"],
        "skipped": prog["skipped"], "out_npz": job["out_npz"],
        "elapsed_s": round(time.time() - job["t_start"], 1),
        "ms_per_img": prog.get("ms_per_img"),
        "prep": prep,             # 嵌入产出的预处理版本留痕（legacy / v2）
    }
    job["status"] = "done"
    job["t_end"] = time.time()


def _worker_loop(wid):
    while True:
        jid = GLOBAL["jobq"].get()
        job = None
        try:
            with GLOBAL["jobs_lock"]:
                job = GLOBAL["jobs"].get(jid)
            if job is not None:
                _run_video_job(job)
        except Exception as ex:
            if job is not None:
                job["status"] = "error"
                job["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
                job["t_end"] = time.time()
            traceback.print_exc()
        finally:
            GLOBAL["jobq"].task_done()
            if job is not None:
                job["t_end"] = job.get("t_end") or time.time()
                log = GLOBAL.get("jobs_log")
                if log:
                    try:
                        with open(log, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "id": job["id"], "cam": job.get("cam"),
                                "status": job["status"], "t_end": round(job["t_end"], 1),
                                "progress": job.get("progress"), "result": job.get("result"),
                                "error": job.get("error"),
                            }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "embed-server/1.1"

    def log_message(self, fmt, *args):   # 静音标准日志（避免刷屏）
        pass

    # ---------- 基础 ----------
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ---------- 路由 ----------
    def do_GET(self):
        if self.path == "/health":
            enc = GLOBAL["enc"]
            up = time.time() - GLOBAL["t_start"]
            avg = (GLOBAL["ms_sum"] / GLOBAL["n_img"]) if GLOBAL["n_img"] else None
            with GLOBAL["jobs_lock"]:
                running = sum(1 for j in GLOBAL["jobs"].values() if j["status"] == "running")
                n_jobs = len(GLOBAL["jobs"])
            self._send(200, {
                "status": "ok" if enc is not None else "loading",
                "device": GLOBAL["device"],
                "n_req": GLOBAL["n_req"], "n_img": GLOBAL["n_img"],
                "ms_per_img_avg": round(avg, 2) if avg else None,
                "queue": GLOBAL["jobq"].qsize(), "running": running,
                "workers": GLOBAL["workers"], "jobs_total": n_jobs,
                "uptime_s": round(up, 1),
                "store": GLOBAL["store"] is not None,
            })
        elif self.path == "/jobs":
            with GLOBAL["jobs_lock"]:
                lst = []
                for j in GLOBAL["jobs"].values():
                    pr = j.get("progress") or {}
                    lst.append({"id": j["id"], "cam": j.get("cam"),
                                "status": j["status"],
                                "done": pr.get("done"), "total": pr.get("total"),
                                "elapsed_s": pr.get("elapsed_s"),
                                "out_npz": (j.get("result") or {}).get("out_npz")})
            self._send(200, {"jobs": lst})
        elif self.path.startswith("/jobs/"):
            jid = self.path.split("/jobs/")[-1]
            with GLOBAL["jobs_lock"]:
                j = GLOBAL["jobs"].get(jid)
            if j is None:
                return self._send(404, {"error": f"no job {jid}"})
            out = {k: j.get(k) for k in ("id", "cam", "status", "progress", "result",
                                         "error", "created")}
            if j.get("t_start") and j["status"] == "running":
                out["progress"]["elapsed_s"] = round(time.time() - j["t_start"], 1)
            self._send(200, out)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == "/encode":
                self._handle_encode(body)
            elif self.path == "/encode_tracks":
                self._handle_encode_tracks(body)
            elif self.path == "/encode_video_tracks":
                self._handle_video_tracks(body)
            elif self.path == "/store":
                self._handle_store(body)
            elif self.path == "/shutdown":
                self._send(200, {"bye": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send(404, {"error": "not found"})
        except Exception as ex:
            self._send(500, {"error": f"{type(ex).__name__}: {str(ex)[:300]}",
                             "tb": traceback.format_exc()[-600:]})

    # ---------- 业务 ----------
    def _encode_imgs(self, imgs, batch):
        import numpy as np
        t0 = time.perf_counter()
        with GLOBAL["gpu_lock"]:
            embs = GLOBAL["enc"].encode(imgs, batch=batch)
        if not isinstance(embs, np.ndarray):
            embs = np.asarray(embs)
        dt = time.perf_counter() - t0
        GLOBAL["n_img"] += len(imgs)
        GLOBAL["ms_sum"] += dt * 1000
        GLOBAL["n_req"] += 1
        return embs, dt

    def _handle_encode(self, body):
        paths = body.get("images") or []
        if not paths:
            return self._send(400, {"error": "images 为空"})
        imgs = _read_images(paths)
        embs, dt = self._encode_imgs(imgs, int(body.get("batch", 16)))
        self._send(200, {"embeddings": embs.tolist(), "n": len(embs),
                         "ms_per_img": round(dt / len(imgs) * 1000, 2)})

    def _handle_encode_tracks(self, body):
        import numpy as np
        tracks = body.get("tracks") or []
        if not tracks:
            return self._send(400, {"error": "tracks 为空"})
        all_imgs, spans = [], []
        for t in tracks:
            n0 = len(all_imgs)
            all_imgs += _read_images(t.get("crops") or [])
            spans.append((n0, len(all_imgs)))
        embs, dt = self._encode_imgs(all_imgs, int(body.get("batch", 16)))
        out = []
        for t, (a, b) in zip(tracks, spans):
            if b <= a:
                out.append({"cam": t.get("cam"), "tid": t.get("tid"),
                            "emb": None, "n_crops": 0})
                continue
            e = embs[a:b].mean(axis=0)
            e = e / (np.linalg.norm(e) + 1e-9)
            out.append({"cam": t.get("cam"), "tid": t.get("tid"),
                        "emb": e.tolist(), "n_crops": b - a})
        self._send(200, {"tracks": out, "ms_per_img": round(dt / len(all_imgs) * 1000, 2)})

    def _handle_video_tracks(self, body):
        """异步提交：视频+检测 jsonl → 服务端抽帧裁剪编码 → out_npz。"""
        need = ["video", "jsonl", "out_npz"]
        miss = [k for k in need if not body.get(k)]
        if miss:
            return self._send(400, {"error": f"缺字段: {miss}"})
        for p in (body["video"], body["jsonl"]):
            if not os.path.exists(p):
                return self._send(400, {"error": f"路径不存在: {p}"})
        with GLOBAL["jobs_lock"]:
            GLOBAL["job_seq"] += 1
            jid = f"j{GLOBAL['job_seq']}"
            job = {
                "id": jid, "created": time.time(), "status": "queued",
                "cam": body.get("cam") or "?", "video": body["video"],
                "jsonl": body["jsonl"], "prefix": body.get("prefix", "t"),
                "samples": int(body.get("samples", 8)),
                "prep": str(body.get("prep", "legacy")),   # v2: 质量选帧+2:1 裁剪+letterbox
                "min_frames": int(body.get("min_frames", 2)),
                "batch": int(body.get("batch", 16)),
                "chunk_tracks": int(body.get("chunk_tracks", 4)),
                "out_npz": body["out_npz"],
                "progress": {"total": None, "done": 0, "skipped": 0,
                             "n_crops": 0, "_gpu_ms": 0.0, "ms_per_img": None,
                             "elapsed_s": 0.0, "eta_s": None},
            }
            GLOBAL["jobs"][jid] = job
        GLOBAL["jobq"].put(jid)
        self._send(200, {"job_id": jid, "status": "queued",
                         "queue": GLOBAL["jobq"].qsize()})

    def _handle_store(self, body):
        st = GLOBAL["store"]
        if st is None:
            return self._send(400, {"error": "store 未启用（启动时加 --store-dir）"})
        items = body.get("items") or []
        with GLOBAL["store_lock"]:
            for it in items:
                st.add(it["key"], it["emb"], cam=it.get("cam"), date=it.get("date"),
                       tid=it.get("tid"), t_first=it.get("t_first"),
                       t_last=it.get("t_last"), h_med=it.get("h_med"),
                       u=it.get("u"), v=it.get("v"))
            n = st.save()
        self._send(200, {"stored": len(items), "total": n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--store-dir", default=None, help="启用 /store 时的 store 目录")
    ap.add_argument("--store-version", default="cradio_gemma2_bridge_v4")
    ap.add_argument("--w-dir", default=None, help="桥接 W 所在目录（默认 FastEncoder DEFAULTS）")
    ap.add_argument("--w-name", default=None)
    ap.add_argument("--workers", type=int, default=3, help="并发任务 worker 数")
    ap.add_argument("--jobs-log", default=None, help="任务完成日志（jsonl 追加）")
    ap.add_argument("--strip-dead", action="store_true",
                    help="剥离 EmbeddingGemma2 死重塔（audio/vision，嵌入逐位不变，省 ~1.85GB 显存）")
    args = ap.parse_args()
    GLOBAL["device"] = args.device
    GLOBAL["jobs_log"] = args.jobs_log
    GLOBAL["workers"] = args.workers

    print(f"[server] 加载 FastEncoder（{args.device}）strip_dead={args.strip_dead} ...", flush=True)
    from embed_fast import FastEncoder
    GLOBAL["enc"] = FastEncoder(root=args.root, device=args.device,
                                w_dir=args.w_dir, w_name=args.w_name,
                                strip_dead=args.strip_dead)
    GLOBAL["enc"].encode([__import__("numpy").random.randint(
        0, 255, (224, 224, 3), dtype="uint8")])   # warmup
    print("[server] 编码器就绪", flush=True)

    if args.store_dir:
        from global_reid_store import GlobalReIDStore
        st = GlobalReIDStore(args.store_dir, args.store_version, 768, mrl_dim=256)
        n = st.load()
        GLOBAL["store"] = st
        print(f"[server] store 已载入 {n} 轨（{args.store_dir}）", flush=True)

    for w in range(args.workers):
        t = threading.Thread(target=_worker_loop, args=(w,), daemon=True)
        t.start()
    print(f"[server] workers={args.workers}", flush=True)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[server] listening :{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
