#!/usr/bin/env python
"""OSNet vs DINOv3-B 同口径对照（冻结评测集 frozen_eval_v1）。

目的：回答「把 embedding 换成专用 ReID（OSNet）是否值得」——
      在同一批裁片、同一采样/聚合口径下比较判别力。

口径（对齐 docs/commit-015/verify_work/exp8_osnet）：
  - 评测集：frozen_eval_v1.json（109 对人工判定；same/diff 参与 AUC，unclear 单独报告）
  - 实体组装：pair 的 `cam:EID` → net_tracks_{cam}.json 的 entities[].tracks → cleanset_v2 逐框
  - 采样：每轨按位置均匀取 <=6 个框；跳过 w<24 或 h<48；有效框 <2 的轨道剔除
  - 裁片：每帧裁一次原始框（BGR uint8，无 pad）；OSNet 直接吃，DINOv3 走 prep_crop(pad 0.08, h=224)
  - 聚合：per-crop L2 → per-track 均值 → per-entity 均值 → L2
  - 指标：same vs diff 的 AUC（Mann-Whitney）、same/diff 中位、margin

输出：output_crosscam_local/osnet_eval/osnet_vs_d3b_frozen_eval.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "output_crosscam/analysis/relay_v2/frozen_eval_v1.json"
NET_TRACKS = str(ROOT / "output_crosscam/analysis/stitch_clean2/net_tracks_{cam}.json")
DET_DIR = ROOT / "output_crosscam/cleanset_v2"
VID_DIR = ROOT / "media-resource/multi-cemra/synced-10min"
VIDEO = {
    "316": "01_行政走廊通道_316/316_085500-090500.mp4",
    "305": "02_办公区1_305/305_085500-090500.mp4",
    "318": "03_办公区2_318/318_085500-090500.mp4",
    "320": "04_办公区球机_320/320_085500-090500.mp4",
    "323": "05_茶水间_323/323_085500-090500.mp4",
    "343": "06_电梯间_343/343_085500-090500.mp4",
}
OSNET_W = ("models/reid_osnet/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_"
           "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
DINO_W = "models/dinov3_vitb16_timm/model.safetensors"

SAMPLES = 6
MIN_W, MIN_H = 24, 48
MIN_VALID = 2


def load_entities():
    ent = {}
    for cam in VIDEO:
        d = json.load(open(NET_TRACKS.format(cam=cam), encoding="utf-8"))
        ent[cam] = {e["eid"]: e for e in d["entities"]}
    return ent


def load_dets():
    det = {}
    for cam in VIDEO:
        p = DET_DIR / f"{cam}_detections.jsonl"
        t = {}
        with open(p, encoding="utf-8") as f:
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
        det[cam] = t
    return det


def prep_crop_rgb(frame, box, out_h=224, pad=0.08):
    x1, y1, x2, y2 = [float(v) for v in box]
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
        pw = out_h - nw
        c = cv2.copyMakeBorder(c, 0, 0, pw // 2, pw - pw // 2, cv2.BORDER_REPLICATE)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def crop_raw(frame, box):
    x1, y1, x2, y2 = [int(float(v)) for v in box]
    X1, Y1 = max(0, x1), max(0, y1)
    X2, Y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < MIN_H or c.shape[1] < MIN_W:
        return None
    return c


class OSNetEnc:
    name = "osnet_msmt17"

    def __init__(self, device="auto"):
        from torchreid.reid.utils import FeatureExtractor
        dev = "cpu"
        if device != "cpu":
            try:
                import torch_directml  # noqa: F401
                dev = str(torch_directml.device())
            except Exception:
                if device == "dml":
                    raise
        if dev == "cpu":
            torch.set_flush_denormal(True)
        print(f"[osnet] device={dev}", flush=True)
        p = ROOT / OSNET_W
        self.fe = FeatureExtractor(model_name="osnet_x1_0", model_path=str(p),
                                   device=dev, verbose=False)
        self.dim = 512

    @torch.no_grad()
    def encode(self, bgr_list):
        v = self.fe(bgr_list).detach().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-9)


class DinoEnc:
    name = "dinov3b"
    dim = 768

    def __init__(self, device="auto"):
        import timm
        from safetensors.torch import load_file
        from torchvision import transforms
        dev = "cpu"
        if device != "cpu":
            try:
                import torch_directml
                dev = torch_directml.device()
            except Exception:
                if device == "dml":
                    raise
        print(f"[dinov3b] device={dev}", flush=True)
        m = timm.create_model("vit_base_patch16_dinov3", pretrained=False,
                              num_classes=0, global_pool="token")
        sd = load_file(str(ROOT / DINO_W))
        m.load_state_dict(sd, strict=False)
        self.device = dev
        self.m = m.eval().to(dev)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def encode(self, rgb_list):
        xs = torch.stack([self.tf(im) for im in rgb_list]).to(self.device)
        f = self.m(xs)
        f = torch.nn.functional.normalize(f, dim=1)
        return f.cpu().numpy().astype(np.float32)


def auc_mannwhitney(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    # tie 平均秩
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i + 1
        while j < len(sv) and sv[j] == sv[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output_crosscam_local/osnet_eval/osnet_vs_d3b_frozen_eval.json")
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--only-osnet", action="store_true", help="只跑 OSNet（跳过 DINO，加速）")
    ap.add_argument("--device", default="auto", choices=["auto", "dml", "cpu"])
    args = ap.parse_args()

    pairs = json.load(open(FROZEN, encoding="utf-8"))["pairs"]
    entities = load_entities()
    dets = load_dets()

    # 需要编码的实体（去重）
    need = {}
    for p in pairs:
        for side in ("a", "b"):
            cam, eid = p[side].split(":", 1)
            need[(cam, eid)] = True

    print(f"[eval] pairs={len(pairs)} entities={len(need)}", flush=True)
    encoders = [OSNetEnc(args.device)] if args.only_osnet else [OSNetEnc(args.device), DinoEnc(args.device)]
    res = {e.name: {} for e in encoders}

    cap_cache = {}
    n_crop = {e.name: 0 for e in encoders}
    t0 = time.time()
    done = 0
    for (cam, eid) in need:
        e = entities.get(cam, {}).get(eid)
        if not e:
            continue
        cap = cap_cache.get(cam)
        if cap is None:
            cap = cv2.VideoCapture(str(VID_DIR / VIDEO[cam]))
            cap_cache[cam] = cap
        # per-track 聚合（与 exp8 一致）：先轨均值，再实体均值
        per_track_vecs = {enc.name: [] for enc in encoders}
        for tid in e["tracks"]:
            tf = dets[cam].get(int(tid), [])
            if len(tf) < 1:
                continue
            idxs = np.linspace(0, len(tf) - 1, min(args.samples, len(tf))).astype(int)
            valid = []  # list[(raw_bgr, rgb_or_None)]
            for k in idxs:
                fnum, box = tf[k]
                cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                ok, frame = cap.read()
                if not ok:
                    continue
                rc = crop_raw(frame, box)
                if rc is None:
                    continue
                rgb = None if args.only_osnet else prep_crop_rgb(frame, box)
                valid.append((rc, rgb))
            if len(valid) < MIN_VALID:
                continue
            for enc in encoders:
                if enc.name == "osnet_msmt17":
                    crops = [rc for rc, _ in valid]
                else:
                    crops = [rgb for _, rgb in valid if rgb is not None]
                    if not crops:
                        continue
                V = enc.encode(crops)
                m = V.mean(axis=0)
                m = m / (np.linalg.norm(m) + 1e-9)
                per_track_vecs[enc.name].append(m)
                n_crop[enc.name] += V.shape[0]
        for enc in encoders:
            if not per_track_vecs[enc.name]:
                continue
            m = np.mean(per_track_vecs[enc.name], axis=0)
            m = m / (np.linalg.norm(m) + 1e-9)
            res[enc.name][f"{cam}:{eid}"] = m
        done += 1
        if done % 20 == 0:
            print(f"  [{done}/{len(need)}] {time.time()-t0:.0f}s "
                  f"crops={ {k: v for k, v in n_crop.items()} }", flush=True)
    for c in cap_cache.values():
        c.release()

    out = {}
    for name in res:
        same, diff, unclear = [], [], []
        for p in pairs:
            ka = p["a"]; kb = p["b"]
            if ka not in res[name] or kb not in res[name]:
                continue
            s = float(np.dot(res[name][ka], res[name][kb]))
            if p["label"] == "same":
                same.append(s)
            elif p["label"] == "diff":
                diff.append(s)
            else:
                unclear.append(s)
        auc = auc_mannwhitney(same, diff)
        out[name] = {
            "n_same": len(same), "n_diff": len(diff), "n_unclear": len(unclear),
            "auc": round(auc, 4) if not math.isnan(auc) else None,
            "same_median": round(float(np.median(same)), 4) if same else None,
            "diff_median": round(float(np.median(diff)), 4) if diff else None,
            "margin": round(float(np.min(same) - np.max(diff)), 4) if same and diff else None,
            "crops": n_crop[name],
            "unclear_median": round(float(np.median(unclear)), 4) if unclear else None,
        }

    out["_meta"] = {
        "frozen": str(FROZEN.relative_to(ROOT)),
        "samples_per_track": args.samples,
        "note": "entity-level 聚合；OSNet 吃原始框（无 pad），DINOv3 走 prep_crop(pad=0.08,h=224)；"
                "两者同裁片来源、同配对、同聚合口径。",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[out] {op}")


if __name__ == "__main__":
    main()
