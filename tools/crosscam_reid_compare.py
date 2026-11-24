#!/usr/bin/env python
"""多后端外观 ReID 对比实验（2026-10-12）。

背景：跨画面去重第三判据"外观"当前用 DINOv2-small(ViT-S/14)。用户问：
DINOv3（Meta, 2025-08）或 C-RADIOv4（NVIDIA, 2026-01）会不会更好？

评估集：candidates_review.csv 已判读 30 对（8 CONF 正 / 22 REJ 负）。
协议：所有后端共享同一批轨迹裁剪帧（同帧同框同几何、同采样），
      逐帧 L2 归一化 → 轨迹均值 → L2 归一化 → 余弦 → AUC。

后端：
  dinov2  : DINOv2-S/14（torch.hub 本地缓存，现行基线）
  dinov3s : DINOv3-S/16（timm 权重，21M）
  dinov3b : DINOv3-B/16（timm 权重，86M）
  cradio  : C-RADIOv4-SO400M（transformers trust_remote_code，431M）

用法:
    python tools/crosscam_reid_compare.py --probe
    python tools/crosscam_reid_compare.py --backends dinov2,dinov3s,dinov3b,cradio
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

A_VIDEO = ROOT / "media-resource/multi-cemra/synced-10min/03_办公区2_318/318_085500-090500.mp4"
B_VIDEO = ROOT / "media-resource/multi-cemra/synced-10min/04_办公区球机_320/320_085500-090500.mp4"
A_JSONL = ROOT / "output_crosscam/318_detections.jsonl"
B_JSONL = ROOT / "output_crosscam/320_detections.jsonl"

MODELS_DIR = ROOT / "models"
CRADIO_DIR = MODELS_DIR / "C-RADIOv4-SO400M"
DINOV3S_WEIGHTS = MODELS_DIR / "dinov3_vits16_timm/model.safetensors"
DINOV3B_WEIGHTS = MODELS_DIR / "dinov3_vitb16_timm/model.safetensors"

OUT_DIR = ROOT / "output_crosscam/analysis/reid_compare"

# ---- 30 个已判读对（与 candidates_review.csv 一致）----
POS_PAIRS = [(14, 3), (120, 18), (37, 11), (120, 23), (69, 14), (139, 24),
             (1, 1), (146, 28)]
NEG_PAIRS = [(14, 1), (3, 1), (50, 11), (7, 1), (10, 1), (126, 18), (57, 11),
             (124, 18), (43, 11), (3, 4), (131, 24), (53, 11), (131, 28),
             (131, 23), (142, 26), (98, 26), (61, 14), (123, 19), (127, 19),
             (146, 21), (132, 21), (131, 21)]


def load_tracks(jsonl_path):
    """tid -> list[(frame, box)]"""
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
                    t.setdefault(int(tids[i]), []).append((r["frame"], b))
    for k in t:
        t[k].sort()
    return t


def prep_crop(frame, box, out_h=224, pad=0.08):
    """与 crosscam_reid.py 完全一致的裁剪逻辑。"""
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


def extract_crops(video_path, tracks, tids, samples):
    """tid -> list of RGB crops（同帧同框，供所有后端共享）。"""
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"cannot open {video_path}"
    out = {}
    for tid in sorted(tids):
        if tid not in tracks:
            print(f"  [warn] tid {tid} not in tracks", flush=True)
            continue
        tf = tracks[tid]
        idxs = np.linspace(0, len(tf) - 1, min(samples, len(tf))).astype(int)
        crops = []
        for i in idxs:
            fnum, box = tf[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = prep_crop(frame, box)
            if rgb is None:
                continue
            crops.append(rgb)
        out[tid] = crops
    cap.release()
    return out


# ---------------- 后端 ----------------

class BaseBackend:
    name = "base"

    def load(self):
        raise NotImplementedError

    def embed_batch(self, rgbs):
        """list[RGB np.ndarray] -> (N, D) L2 归一化向量"""
        raise NotImplementedError


class DinoV2Backend(BaseBackend):
    name = "dinov2_vits14"

    def load(self):
        import torch
        from torchvision import transforms
        hub_dir = os.path.expanduser("~/.cache/torch/hub")
        repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        torch.hub.set_dir(hub_dir)
        if os.path.isdir(repo):
            self.model = torch.hub.load(repo, "dinov2_vits14", source="local",
                                        trust_repo=True, verbose=False)
        else:
            self.model = torch.hub.load("facebookresearch/dinov2",
                                        "dinov2_vits14", trust_repo=True,
                                        verbose=False)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def embed_batch(self, rgbs):
        import torch
        from PIL import Image
        xs = torch.stack([self.tf(Image.fromarray(im)) for im in rgbs])
        with torch.inference_mode():
            f = self.model(xs)
        f = torch.nn.functional.normalize(f, dim=1)
        return f.numpy()


class DinoV3TimmBackend(BaseBackend):
    """timm DINOv3（Eva 架构实现）。

    要点（查证 timm 源码 eva.py）：
    - 官方用 CLS-token pooling；timm 默认 avg pooling → 必须 global_pool='token'
    - 官方默认输入 256（可插值），crop_pct=1.0、bicubic
    """

    def __init__(self, name, arch, weights_path, img_size=224):
        self.name = name
        self.arch = arch
        self.weights_path = weights_path
        self.img_size = img_size

    def load(self):
        import torch
        import timm
        from torchvision import transforms
        self.model = timm.create_model(self.arch, pretrained=False,
                                       num_classes=0, global_pool='token')
        timm.models.load_checkpoint(self.model, str(self.weights_path))
        self.model.eval()
        with torch.inference_mode():
            probe = self.model(torch.zeros(1, 3, self.img_size, self.img_size))
        print(f"  [{self.name}] out_dim={tuple(probe.shape)} "
              f"(global_pool=token, size={self.img_size})", flush=True)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def embed_batch(self, rgbs):
        import torch
        from PIL import Image
        items = []
        for im in rgbs:
            pim = Image.fromarray(im)
            if pim.size != (self.img_size, self.img_size):
                pim = pim.resize((self.img_size, self.img_size),
                                 Image.BICUBIC)
            items.append(self.tf(pim))
        xs = torch.stack(items)
        with torch.inference_mode():
            f = self.model(xs)
        f = torch.nn.functional.normalize(f.float(), dim=1)
        return f.numpy()


class CRadioBackend(BaseBackend):
    name = "cradio_so400m"

    def load(self):
        import torch
        from transformers import AutoModel, CLIPImageProcessor
        self.model = AutoModel.from_pretrained(
            str(CRADIO_DIR), trust_remote_code=True)
        self.model.eval()
        self.proc = CLIPImageProcessor.from_pretrained(str(CRADIO_DIR))
        print(f"  [cradio] processor: size={self.proc.size} "
              f"do_center_crop={self.proc.do_center_crop} "
              f"do_normalize={self.proc.do_normalize} "
              f"mean={self.proc.image_mean}", flush=True)

    def embed_batch(self, rgbs, normalize_input="default"):
        import torch
        from PIL import Image
        imgs = [Image.fromarray(x) for x in rgbs]
        if normalize_input == "default":
            pix = self.proc(images=imgs, return_tensors="pt").pixel_values
        else:  # raw: 只 resize，不归一化（RADIO 内部自带 normalizer）
            pix = self.proc(images=imgs, return_tensors="pt", do_resize=True,
                            do_center_crop=False,
                            do_normalize=False).pixel_values
        with torch.inference_mode():
            out = self.model(pix)
        if isinstance(out, tuple):
            summary = out[0]
        elif isinstance(out, dict):
            summary = out.get("summary")
            if summary is None:
                summary = next(v for v in out.values()
                               if isinstance(v, torch.Tensor) and v.ndim == 2)
        else:
            summary = getattr(out, "summary", None)
            if summary is None:  # 再兜底
                summary = out[0]
        f = torch.nn.functional.normalize(summary.float(), dim=1)
        return f.numpy()


# ---------------- 工具 ----------------

def aggregate(embed_fn, name, crop_map):
    """tid -> 均值向量（L2 归一化）；embed_fn 为批量编码函数。"""
    emb = {}
    t0 = time.time()
    n_imgs = 0
    for tid, crops in crop_map.items():
        if not crops:
            continue
        vecs = embed_fn(crops)
        n_imgs += len(crops)
        m = vecs.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-9)
        emb[tid] = m.astype(np.float32)
    dt = time.time() - t0
    print(f"  [{name}] encoded {n_imgs} crops in {dt:.1f}s "
          f"({dt/max(1,n_imgs)*1000:.0f} ms/img)", flush=True)
    return emb, dt, n_imgs


def eval_pairs(Ea, Eb, pairs):
    return [float(np.dot(Ea[a], Eb[b])) if (a in Ea and b in Eb) else None
            for a, b in pairs]


def auc(pv, nv):
    cnt = 0
    for p in pv:
        for n in nv:
            if p > n:
                cnt += 1
            elif p == n:
                cnt += 0.5
    return cnt / (len(pv) * len(nv)) if pv and nv else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="dinov2,dinov3s,dinov3b,cradio")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--dino3-size", type=int, default=224,
                    help="DINOv3 输入分辨率（官方默认 256）")
    ap.add_argument("--probe", action="store_true", help="只做加载自检")
    ap.add_argument("--cradio-input", default="both",
                    choices=["both", "default", "raw"],
                    help="C-RADIO 输入模式：default=224 原样 / raw=resize512 / both")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    which = [x.strip() for x in args.backends.split(",") if x.strip()]

    tracks_a = load_tracks(A_JSONL)
    tracks_b = load_tracks(B_JSONL)
    used_a = sorted({a for a, b in POS_PAIRS + NEG_PAIRS})
    used_b = sorted({b for a, b in POS_PAIRS + NEG_PAIRS})
    print(f"[data] A tracks: {len(used_a)}, B tracks: {len(used_b)}")

    # ---- crops（共享缓存）----
    cache = OUT_DIR / f"crops_s{args.samples}.pkl"
    if cache.exists():
        print(f"[crops] load cache {cache.name}")
        with open(cache, "rb") as f:
            ca, cb = pickle.load(f)
    else:
        t0 = time.time()
        print("[crops] extracting ...", flush=True)
        ca = extract_crops(A_VIDEO, tracks_a, used_a, args.samples)
        cb = extract_crops(B_VIDEO, tracks_b, used_b, args.samples)
        with open(cache, "wb") as f:
            pickle.dump((ca, cb), f)
        print(f"[crops] done in {time.time()-t0:.0f}s, "
              f"A={sum(len(v) for v in ca.values())} B={sum(len(v) for v in cb.values())} crops")

    # ---- 后端构建 ----
    def make(bk):
        if bk == "dinov2":
            return DinoV2Backend()
        if bk == "dinov3s":
            return DinoV3TimmBackend("dinov3_vits16",
                                     "vit_small_patch16_dinov3",
                                     DINOV3S_WEIGHTS,
                                     img_size=args.dino3_size)
        if bk == "dinov3b":
            return DinoV3TimmBackend("dinov3_vitb16",
                                     "vit_base_patch16_dinov3",
                                     DINOV3B_WEIGHTS,
                                     img_size=args.dino3_size)
        if bk == "cradio":
            return CRadioBackend()
        raise ValueError(bk)

    if args.probe:
        print("\n==== PROBE ====")
        for bk in which:
            b = make(bk)
            t0 = time.time()
            try:
                b.load()
                print(f"  [{bk}] loaded in {time.time()-t0:.0f}s")
                if bk == "cradio":
                    for mode in ("default", "raw"):
                        vecs = b.embed_batch([ca[used_a[0]][0]] * 2,
                                             normalize_input=mode)
                        print(f"  [{bk}] mode={mode} -> {vecs.shape} "
                              f"norm={np.linalg.norm(vecs[0]):.3f}")
                else:
                    vecs = b.embed_batch([ca[used_a[0]][0]] * 2)
                    print(f"  [{bk}] -> {vecs.shape} "
                          f"norm={np.linalg.norm(vecs[0]):.3f}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [{bk}] PROBE FAILED: {e}")
        return

    # ---- 正式评估 ----
    # 任务编排：cradio 可一次加载、按两种输入模式分别评估
    jobs = []  # (result_key, backend_key, mode_or_None)
    for bk in which:
        if bk == "cradio":
            if args.cradio_input == "both":
                jobs.append(("cradio224", "cradio", "default"))
                jobs.append(("cradio512", "cradio", "raw"))
            elif args.cradio_input == "raw":
                jobs.append(("cradio512", "cradio", "raw"))
            else:
                jobs.append(("cradio224", "cradio", "default"))
        elif bk in ("dinov3s", "dinov3b") and args.dino3_size != 224:
            jobs.append((f"{bk}@{args.dino3_size}", bk, None))
        else:
            jobs.append((bk, bk, None))

    backend_cache = {}
    results = {}
    for key, bk, mode in jobs:
        print(f"\n==== {key} ====")
        if bk not in backend_cache:
            b = make(bk)
            t0 = time.time()
            b.load()
            load_s = time.time() - t0
            print(f"  loaded in {load_s:.0f}s")
            backend_cache[bk] = (b, load_s)
        b, load_s = backend_cache[bk]
        if mode is not None:
            import functools
            embed_fn = functools.partial(b.embed_batch, normalize_input=mode)
        else:
            embed_fn = b.embed_batch
        Ea, dt_a, na = aggregate(embed_fn, key, ca)
        Eb, dt_b, nb = aggregate(embed_fn, key, cb)
        pv = eval_pairs(Ea, Eb, POS_PAIRS)
        nv = eval_pairs(Ea, Eb, NEG_PAIRS)
        pv_ok = [x for x in pv if x is not None]
        nv_ok = [x for x in nv if x is not None]
        a = auc(pv_ok, nv_ok)
        gap = min(pv_ok) - max(nv_ok)
        print(f"  min(pos)={min(pv_ok):.3f}  max(neg)={max(nv_ok):.3f}  "
              f"gap={gap:+.3f}  AUC={a:.3f}")
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
            tp = sum(1 for x in pv_ok if x >= thr)
            fp = sum(1 for x in nv_ok if x >= thr)
            print(f"    thr={thr:.1f}: recall={tp}/{len(pv_ok)}  "
                  f"false_pos={fp}/{len(nv_ok)}")
        results[key] = {
            "name": b.name + (f"@{mode}" if mode else ""),
            "load_s": round(load_s, 1),
            "encode_s": round(dt_a + dt_b, 1),
            "n_crops": na + nb,
            "ms_per_img": round((dt_a + dt_b) / max(1, na + nb) * 1000, 1),
            "pos": [[a_, b_, x] for (a_, b_), x in zip(POS_PAIRS, pv)],
            "neg": [[a_, b_, x] for (a_, b_), x in zip(NEG_PAIRS, nv)],
            "min_pos": min(pv_ok), "max_neg": max(nv_ok),
            "gap": gap, "auc": a,
        }

    # ---- 汇总输出（追加式：合并历史上已评估的后端）----
    print("\n" + "=" * 78)
    print(f"{'backend':<24} {'AUC':>6} {'gap':>8} {'min(pos)':>9} "
          f"{'max(neg)':>9} {'ms/img':>7} {'n_img':>6}")
    for bk, r in results.items():
        print(f"{r['name']:<24} {r['auc']:>6.3f} {r['gap']:>+8.3f} "
              f"{r['min_pos']:>9.3f} {r['max_neg']:>9.3f} "
              f"{r['ms_per_img']:>7.1f} {r['n_crops']:>6}")

    out_json = OUT_DIR / "result.json"
    old_results = {}
    if out_json.exists():
        try:
            old_results = json.loads(
                out_json.read_text(encoding="utf-8")).get("results", {})
        except Exception:  # noqa: BLE001
            old_results = {}
    merged = {**old_results, **results}

    # 明细 CSV（全部已评估后端）
    cols = list(merged.keys())
    lines = ["pair,label," + ",".join(merged[c]["name"] for c in cols)]
    for i, (a_, b_) in enumerate(POS_PAIRS):
        vals = ",".join(f"{merged[c]['pos'][i][2]:.4f}" if merged[c]['pos'][i][2] is not None else ""
                        for c in cols)
        lines.append(f"{a_}:{b_},POS,{vals}")
    for i, (a_, b_) in enumerate(NEG_PAIRS):
        vals = ",".join(f"{merged[c]['neg'][i][2]:.4f}" if merged[c]['neg'][i][2] is not None else ""
                        for c in cols)
        lines.append(f"{a_}:{b_},NEG,{vals}")
    (OUT_DIR / "scores.csv").write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(
        json.dumps({"samples": args.samples, "results": merged,
                    "pos_pairs": POS_PAIRS, "neg_pairs": NEG_PAIRS},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {OUT_DIR/'scores.csv'}")
    print(f"[out] {out_json}")


if __name__ == "__main__":
    main()
