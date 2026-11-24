#!/usr/bin/env python
"""病例对照实验（Stage B）：在 OSNet 认错的病例上，比较 DINOv3 与 OSNet 的排序能力。

设计要点（避免混淆）：
  * **同一批裁剪**同时喂给两个骨干 → 差异只来自模型，不来自采样；
  * 只在**有身份的簇**之间排名（无身份的簇不可能是"正确答案"）；
  * 簇得分 = 成员轨迹特征的**最大值**，与生产口径一致。

用法：
  PY tools/modal_case_encode.py --max-cases 200 --frames 6 \
      --backbone vit_base_patch16_dinov3 \
      --ckpt models/dinov3_vitb16_timm/model.safetensors \
      --out output_crosscam_local/selector_osnet_1012/modal_case_result.csv
"""
import argparse
import collections
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selector import _norm                      # noqa: E402
import session_selector_v1 as S                 # noqa: E402
import eval_mask_variants as M                  # noqa: E402
import cluster_camera_aware as CA               # noqa: E402
from case_control_modal import load_gt          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_W, MIN_H = 24, 48   # 与 encode_1012_osnet_seq.py 保持一致
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def track_frames(ab_dir, seg):
    """返回 {tid: [(frame, box), ...]}，合并 a/b 两侧。"""
    out = collections.defaultdict(list)
    for side in ("a", "b"):
        p = os.path.join(ab_dir, f"{seg}_{side}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            d = __import__("json").loads(line)
            for t, b in zip(d["track_ids"], d["boxes"]):
                out[t].append((d["frame"], b))
    for t in out:
        out[t].sort()
    return out


def pick(frames, n):
    if len(frames) <= n:
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def crop(frame, box):
    x1, y1, x2, y2 = [int(float(v)) for v in box]
    X1, Y1 = max(0, x1), max(0, y1)
    X2, Y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0 or c.shape[0] < MIN_H or c.shape[1] < MIN_W:
        return None
    return c


def prep_vit(bgr, size=224):
    im = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    im = im[:, :, ::-1].astype(np.float32) / 255.0
    return ((im - MEAN) / STD).transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="output_crosscam_local/1012_yolo8_dd_emb")
    ap.add_argument("--ab", default="output_crosscam_local/1012_yolo8_dd_ab")
    ap.add_argument("--videos", default="media-resource/1012/segments")
    ap.add_argument("--gt", default="deliverable/data-request-v2/gt")
    ap.add_argument("--backbone", default="vit_base_patch16_dinov3")
    ap.add_argument("--ckpt", default="models/dinov3_vitb16_timm/model.safetensors")
    ap.add_argument("--frames", type=int, default=6, help="每条轨迹采样帧数")
    ap.add_argument("--max-cases", type=int, default=200)
    ap.add_argument("--thr-in", type=float, default=0.15)
    ap.add_argument("--thr-out", type=float, default=0.25)
    ap.add_argument("--veto", type=float, default=0.30)
    ap.add_argument("--out", default="output_crosscam_local/selector_osnet_1012/modal_case_result.csv")
    a = ap.parse_args()

    gal, qry = load_gt(os.path.join(ROOT, a.gt))
    ktr, Va_tr, Vb_tr = M.load_split(os.path.join(ROOT, a.emb), S.TRAIN_SEGS)
    Vtr = _norm((Va_tr + Vb_tr) / 2.0)
    cams_tr = np.array([k[0].split("_")[0] for k in ktr])
    lab, kept = CA.camera_aware_labels(Vtr, cams_tr, a.thr_in, a.thr_out, S.MIN_CLUSTER)

    drop = []
    for c in kept:
        idx = np.where(lab == c)[0]
        if idx.size >= 2:
            Xc = Vtr[idx]
            dmax = float(1.0 - (Xc @ Xc.T)[np.triu_indices(idx.size, 1)].min())
        else:
            dmax = 0.0
        if dmax >= a.veto:
            drop.append(c)
    kept = [c for c in kept if c not in set(drop)]

    cid_pid, cid_members = {}, {}
    for ci, c in enumerate(kept):
        idx = np.where(lab == c)[0]
        cid_members[ci] = [ktr[i] for i in idx]
        votes = collections.Counter(gal[k] for k in cid_members[ci] if k in gal)
        if votes:
            cid_pid[ci] = votes.most_common(1)[0][0]
    named = [ci for ci in cid_pid]
    print(f"[cluster] K={len(kept)} 有身份 {len(named)}（成员合计 {sum(len(cid_members[c]) for c in named)} 条轨迹）")

    # 选病例：可评估、有身份、该身份存在有身份的簇
    cases = []
    for (seg, tid), (pid, ev) in qry.items():
        if not ev or not pid:
            continue
        corr = [ci for ci in named if cid_pid[ci] == pid]
        if corr:
            cases.append(((seg, tid), pid, corr))
    cases.sort()
    if len(cases) > a.max_cases:
        step = len(cases) / a.max_cases
        cases = [cases[int(i * step)] for i in range(a.max_cases)]
    print(f"[cases] {len(cases)} 条查询 × 对手簇 {len(named)} 个")

    # 需要裁剪的轨迹：病例查询 + 全部有身份簇的成员
    need = collections.defaultdict(set)
    for (seg, tid), _, _ in cases:
        need[seg].add(tid)
    for ci in named:
        for seg, tid in cid_members[ci]:
            need[seg].add(tid)

    # 组请求，按 (seg, frame) 归并，减少 seek
    req = collections.defaultdict(list)     # seg -> [(frame, tid, box)]
    for seg, tids in need.items():
        tf = track_frames(os.path.join(ROOT, a.ab), seg)
        for tid in tids:
            fr = tf.get(tid, [])
            for f, b in pick(fr, a.frames):
                req[seg].append((f, tid, b))

    crops, keys = [], []
    t0 = time.time()
    for seg, items in req.items():
        items.sort(key=lambda x: x[0])
        cap = cv2.VideoCapture(os.path.join(ROOT, a.videos, f"{seg}.mp4"))
        for f, tid, b in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            c = crop(img, b)
            if c is None:
                continue
            crops.append(c)
            keys.append((seg, tid))
        cap.release()
    print(f"[crop] {len(crops)} 张，用时 {time.time()-t0:.0f}s")

    # DINOv3
    print(f"[vit] 载入 {a.backbone}", flush=True)
    vm = timm_load(a.backbone, os.path.join(ROOT, a.ckpt))
    feat = {"vit": encode_vit(vm, crops), "osnet": encode_osnet(crops)}

    # 轨迹特征 = 裁剪特征均值
    def track_feats(V):
        agg = collections.defaultdict(list)
        for (seg, tid), v in zip(keys, V):
            agg[(seg, tid)].append(v)
        out = {}
        for k, vs in agg.items():
            out[k] = _norm(np.mean(vs, 0, keepdims=True))[0]
        return out

    res = {}
    for name, V in feat.items():
        res[name] = track_feats(V)

    rows = []
    for (seg, tid), pid, corr in cases:
        row = {"query": f"{seg}#{tid}", "gt_pid": pid, "n_correct": len(corr)}
        for name, tf in res.items():
            sc = []
            for ci in named:
                vals = [tf[k] @ tf[(seg, tid)] for k in cid_members[ci]
                        if k in tf and k != (seg, tid)]
                sc.append(max(vals) if vals else -1.0)
            sc = np.array(sc)
            order = np.argsort(-sc)
            rank = min(int(np.where(order == named.index(ci))[0][0]) + 1 for ci in corr if ci in named)
            row[f"{name}_rank"] = rank
            row[f"{name}_score"] = round(float(max(sc[named.index(ci)] for ci in corr if ci in named)), 3)
        rows.append(row)

    ok_v = sum(1 for r in rows if r.get("vit_rank") == 1)
    ok_o = sum(1 for r in rows if r.get("osnet_rank") == 1)
    both = sum(1 for r in rows if r.get("vit_rank") == 1 and r.get("osnet_rank") == 1)
    fixed = [r for r in rows if r.get("vit_rank") == 1 and r.get("osnet_rank") != 1]
    broke = [r for r in rows if r.get("osnet_rank") == 1 and r.get("vit_rank") != 1]
    print(f"\n=== 同裁剪、同对手簇下的 top-1 命中 ===")
    print(f"样本 {len(rows)} 条")
    print(f"osnet 命中 {ok_o} ({ok_o/max(1,len(rows))*100:.1f}%)")
    print(f"dinov3 命中 {ok_v} ({ok_v/max(1,len(rows))*100:.1f}%)")
    print(f"两者都对 {both}｜DINOv3 救回 {len(fixed)}｜DINOv3 弄错 {len(broke)}")
    if fixed:
        print(f"  救回样本按机位 {dict(collections.Counter(r['query'].split('_')[0] for r in fixed))}")

    p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {p}")


def timm_load(name, ckpt):
    import timm
    m = timm.create_model(name, pretrained=False, num_classes=0, checkpoint_path=ckpt)
    m.eval()
    return m


@torch.no_grad()
def encode_vit(m, crops, bs=16):
    outs = []
    for i in range(0, len(crops), bs):
        x = torch.from_numpy(np.stack([prep_vit(c) for c in crops[i:i + bs]]))
        f = m(x).numpy().astype(np.float32)
        outs.append(_norm(f))
    return np.concatenate(outs, 0)


def encode_osnet(crops, bs=32):
    from torchreid.reid.utils import FeatureExtractor
    # 必须用生产同一个权重（models/reid_osnet 下有 4 个 .pth，不能靠 glob 猜）
    path = os.path.join(ROOT, "models/reid_osnet",
                        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_"
                        "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
    fe = FeatureExtractor(model_name="osnet_x1_0", model_path=path, device="cpu", verbose=False)
    outs = []
    with torch.no_grad():
        for i in range(0, len(crops), bs):
            v = fe(crops[i:i + bs]).detach().cpu().numpy().astype(np.float32)
            outs.append(_norm(v))
    return np.concatenate(outs, 0)


if __name__ == "__main__":
    main()
