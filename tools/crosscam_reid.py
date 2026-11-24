#!/usr/bin/env python
"""跨画面外观嵌入（ReID 基线）：DINOv2-small 对轨迹裁剪编码，余弦比对。

背景：免训练颜色指标无法分离同人/异人（见 REPORT §6），本脚本引入
自监督视觉底座 DINOv2-small（ViT-S/14, 384-d）作为外观嵌入基线，
在人工标注的确认对/否决对上做判别力评估。

用法:
    python tools/crosscam_reid.py \
        --a-jsonl .../318_detections.jsonl --b-jsonl .../320_detections.jsonl \
        --a-video ... --b-video ... \
        --out .../reid_scores.json [--save-emb .../reid_emb.npz]
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

# 人工标注（见 REPORT §4/§5）
POSITIVE_PAIRS = [(1, 1), (14, 3), (69, 14), (139, 24), (146, 28)]
NEGATIVE_PAIRS = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21),
                  (3, 1), (7, 1), (146, 21), (127, 19), (123, 19)]


def load_tracks(jsonl_path, W, H):
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
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1 = max(0, int(x1 - w * pad))
    Y1 = max(0, int(y1 - h * pad))
    X2 = min(frame.shape[1], int(x2 + w * pad))
    Y2 = min(frame.shape[0], int(y2 + h * pad))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < 12 or c.shape[1] < 8:
        return None
    # 保持长宽比：高固定 out_h，宽超过 out_h 则中心裁，不足则补边
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


class Embedder:
    """外观嵌入（多后端）。

    后端选择（2026-10-12 评估，30 判读对 / 本机 CPU）：
      dinov3b（默认）: DINOv3-B/16, AUC 0.983, ~188ms/图 —— 推荐
      dinov3s        : DINOv3-S/16, AUC 0.915, ~64ms/图  —— 轻量档
      dinov2         : DINOv2-S/14, AUC 0.869（旧基线，兼容保留）
    权重：timm 官方重托管（models/dinov3_*_timm/model.safetensors，本地加载、免网络）。
    """

    def __init__(self, device="cpu", backend="dinov3b"):
        self.device = device
        self.backend = backend
        print(f"[reid] backend={backend} device={device}", flush=True)
        self.model = self._load_model(backend)
        self.model.eval().to(device)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def _weights_path(backend):
        root = Path(__file__).resolve().parent.parent
        sub = ("dinov3_vitb16_timm" if backend == "dinov3b"
               else "dinov3_vits16_timm")
        return root / "models" / sub / "model.safetensors"

    def _load_model(self, backend):
        if backend == "dinov2":
            return self._load_dinov2()
        if backend in ("dinov3s", "dinov3b"):
            return self._load_dinov3(backend)
        raise ValueError(f"unknown backend: {backend}")

    @staticmethod
    def _load_dinov3(backend):
        """timm 本地权重加载（离线可用；timm DINOv3 需 global_pool='token'）。"""
        import timm
        arch = ("vit_base_patch16_dinov3" if backend == "dinov3b"
                else "vit_small_patch16_dinov3")
        w = Embedder._weights_path(backend)
        if not Path(w).exists():
            raise FileNotFoundError(f"DINOv3 权重缺失: {w}（见 tools/download_models.py）")
        print(f"[reid] loading {arch} (timm, local) ...", flush=True)
        m = timm.create_model(arch, pretrained=False, num_classes=0,
                              global_pool="token")
        timm.models.load_checkpoint(m, str(w))
        print("[reid] dinov3 loaded", flush=True)
        return m

    @staticmethod
    def _load_dinov2():
        """离线优先加载 DINOv2（2026-10-12 实测：网络窗口关闭时 torch.hub
        仓库名解析会挂起/失败（套2 卡 25min+）。本地缓存存在时 source='local'。"""
        hub_dir = os.path.expanduser("~/.cache/torch/hub")
        repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        torch.hub.set_dir(hub_dir)
        if os.path.isdir(repo):
            try:
                m = torch.hub.load(repo, "dinov2_vits14", source="local",
                                   trust_repo=True, verbose=False)
                print("[reid] loaded via local repo (no network)", flush=True)
                return m
            except Exception as e:  # noqa: BLE001
                print("[reid] local hub load failed:", str(e)[:200], flush=True)
        print("[reid] fallback to github spec ...", flush=True)
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                              trust_repo=True, verbose=False)

    @torch.no_grad()
    def embed_batch(self, rgb_list):
        xs = torch.stack([self.tf(im) for im in rgb_list]).to(self.device)
        f = self.model(xs)
        f = torch.nn.functional.normalize(f, dim=1)
        return f.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-jsonl", required=True)
    ap.add_argument("--b-jsonl", required=True)
    ap.add_argument("--a-video", required=True)
    ap.add_argument("--b-video", required=True)
    ap.add_argument("--a-wh", nargs=2, type=int, default=[1920, 1080])
    ap.add_argument("--b-wh", nargs=2, type=int, default=[2560, 1440])
    ap.add_argument("--samples", type=int, default=6, help="每条轨迹采样张数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-emb", default="")
    ap.add_argument("--backend", default="dinov3b",
                    choices=["dinov2", "dinov3s", "dinov3b"],
                    help="外观后端（默认 dinov3b，2026-10-12 评估推荐）")
    ap.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0 ...")
    args = ap.parse_args()

    tracks_a = load_tracks(args.a_jsonl, *args.a_wh)
    tracks_b = load_tracks(args.b_jsonl, *args.b_wh)

    emb = Embedder(device=args.device, backend=args.backend)

    def track_embedding(video_path, tid_frames, verbose_tag):
        cap = cv2.VideoCapture(str(video_path))
        fr = [p[0] for p in tid_frames]
        idxs = np.linspace(0, len(fr) - 1, min(args.samples, len(fr))).astype(int)
        vecs = []
        for i in idxs:
            fnum, box = tid_frames[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = prep_crop(frame, box)
            if rgb is None:
                continue
            v = emb.embed_batch([rgb])[0]
            vecs.append(v)
        cap.release()
        if not vecs:
            return None, 0
        M = np.stack(vecs)
        mean = M.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-9)
        return mean.astype(np.float32), len(vecs)

    # 只对"候选对涉及的轨迹"编码，控制算力
    used_a, used_b = set(), set()
    cand_path = Path(args.out).parent / "crosscam_candidates.json"
    if cand_path.exists():
        cj = json.loads(cand_path.read_text(encoding="utf-8"))
        for p in cj["pairs"]:
            used_a.add(p["tid_a"])
            used_b.add(p["tid_b"])
    for a, b in POSITIVE_PAIRS + NEGATIVE_PAIRS:
        used_a.add(a)
        used_b.add(b)
    used_a = sorted(x for x in used_a if x in tracks_a)
    used_b = sorted(x for x in used_b if x in tracks_b)
    print(f"[reid] encode A tracks: {len(used_a)}, B tracks: {len(used_b)}")

    Ea, Eb = {}, {}
    for i, tid in enumerate(used_a):
        v, n = track_embedding(args.a_video, tracks_a[tid], f"A{tid}")
        if v is not None:
            Ea[tid] = v
        print(f"  [A] tid={tid:>4} crops={n}", flush=True)
    for i, tid in enumerate(used_b):
        v, n = track_embedding(args.b_video, tracks_b[tid], f"B{tid}")
        if v is not None:
            Eb[tid] = v
        print(f"  [B] tid={tid:>4} crops={n}", flush=True)

    def sim(a, b):
        if a not in Ea or b not in Eb:
            return None
        return float(np.dot(Ea[a], Eb[b]))

    pos = sorted((sim(a, b) for a, b in POSITIVE_PAIRS), key=lambda x: -(x or -2))
    neg = sorted((sim(a, b) for a, b in NEGATIVE_PAIRS), key=lambda x: -(x or -2))
    print("\n=== 判别力评估（cosine 相似度）===")
    print(f"{'positives (确认同人)':<40} {'negatives (已否决)':<40}")
    rows = max(len(pos), len(neg))
    for i in range(rows):
        p = f"{POSITIVE_PAIRS[i]} {pos[i]:+.3f}" if i < len(pos) and pos[i] is not None else ""
        n = f"{NEGATIVE_PAIRS[i]} {neg[i]:+.3f}" if i < len(neg) and neg[i] is not None else ""
        print(f"{p:<40} {n:<40}")
    pv = [x for x in pos if x is not None]
    nv = [x for x in neg if x is not None]
    if pv and nv:
        print(f"\nmin(pos)={min(pv):.3f}  max(neg)={max(nv):.3f}  "
              f"gap={min(pv)-max(nv):+.3f}")
        # 简单阈值扫描
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            tp = sum(1 for x in pv if x >= thr)
            fp = sum(1 for x in nv if x >= thr)
            print(f"  thr={thr:.1f}: recall={tp}/{len(pv)}  false_pos={fp}/{len(nv)}")
        # AUC（简单秩统计）
        cnt = 0
        for p in pv:
            for n in nv:
                if p > n:
                    cnt += 1
                elif p == n:
                    cnt += 0.5
        print(f"  AUC = {cnt/(len(pv)*len(nv)):.3f}")

    # 每条 B 轨迹的 A 排名（正视 partner 是否 top）
    print("\n=== B 轨迹的 A-侧 top3（余弦）===")
    ranking = {}
    for b in used_b:
        scores = sorted(((sim(a, b), a) for a in used_a if sim(a, b) is not None),
                        key=lambda x: -x[0])
        ranking[b] = scores[:5]
        print(f"  B#{b}: " + ", ".join(f"A#{a}({s:+.3f})" for s, a in scores[:5]))

    out = {
        "samples_per_track": args.samples,
        "positive_pairs": [[a, b, sim(a, b)] for a, b in POSITIVE_PAIRS],
        "negative_pairs": [[a, b, sim(a, b)] for a, b in NEGATIVE_PAIRS],
        "ranking_b_to_a": {str(b): [[a, s] for s, a in v] for b, v in ranking.items()},
        "emb_tracks_a": sorted(Ea.keys()),
        "emb_tracks_b": sorted(Eb.keys()),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\n[out] {args.out}")
    if args.save_emb:
        np.savez(args.save_emb,
                 **{f"a{tid}": v for tid, v in Ea.items()},
                 **{f"b{tid}": v for tid, v in Eb.items()})
        print(f"[out] {args.save_emb}")


if __name__ == "__main__":
    main()
