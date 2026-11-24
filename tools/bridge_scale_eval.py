# -*- coding: utf-8 -*-
"""规模化桥接评估：C-RADIOv4×(大样本 W) vs DINOv3-B，在两个协议上对比。

协议 A（30 对，跨机位 318↔320）：与历史严格可比
协议 B（labels_v2 全量：同机位 58 正/3 负 + 跨机位 8 正/24 负）
  - 同机位子集为"域内运行指标"（W 训练未排除同机位链接轨）；跨机位子集为留出泛化指标
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ["318", "320", "305", "316", "323", "343"]
PREFIX = {"318": "a", "320": "b", "305": "v305_", "316": "v316_",
          "323": "v323_", "343": "v343_"}

POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def load_embs(emb_dir, pat):
    embs = {}
    for cam in CAMS:
        p = os.path.join(emb_dir, pat.format(cam=cam))
        if not os.path.exists(p):
            print(f"[load] 缺 {p}")
            return None
        d = np.load(p)
        embs[cam] = {key_to_tid(k): d[k] for k in d.files}
    return embs


def sim(embs, c1, t1, c2, t2):
    if c1 not in embs or c2 not in embs:
        return None
    if t1 not in embs[c1] or t2 not in embs[c2]:
        return None
    a, b = embs[c1][t1], embs[c2][t2]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def auc(pv, nv):
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pv for n in nv]))


def report(tag, pv, nv):
    pv = [v for v in pv if v is not None]
    nv = [v for v in nv if v is not None]
    if not pv or not nv:
        return {"n_pos": len(pv), "n_neg": len(nv)}
    r = {"n_pos": len(pv), "n_neg": len(nv),
         "auc": round(auc(pv, nv), 4),
         "pos_min": round(min(pv), 4), "pos_mean": round(float(np.mean(pv)), 4),
         "neg_max": round(max(nv), 4), "neg_mean": round(float(np.mean(nv)), 4),
         "margin": round(min(pv) - max(nv), 4)}
    print(f"  [{tag:22s}] AUC={r['auc']:.4f} | POS {r['pos_min']:.3f}~ | "
          f"NEG ..{r['neg_max']:.3f} | margin {r['margin']:+.4f} "
          f"(pos n={r['n_pos']}, neg n={r['n_neg']})")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cradio-dir", default=os.path.join(
        ROOT, "output_crosscam", "analysis", "cradio_scale"))
    ap.add_argument("--cradio-pat", default="{cam}_emb_scale.npz")
    ap.add_argument("--d3b-dir", default=os.path.join(
        ROOT, "output_crosscam", "analysis", "relay", "remote_emb"))
    ap.add_argument("--d3b-pat", default="reid_emb_{cam}_d3b.npz")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "output_crosscam", "analysis", "bridge_scale", "eval_results.json"))
    args = ap.parse_args()

    embs_c = load_embs(args.cradio_dir, args.cradio_pat)
    embs_d = load_embs(args.d3b_dir, args.d3b_pat)
    if embs_c is None or embs_d is None:
        print("嵌入不全，退出")
        return

    lv = json.load(open(os.path.join(ROOT, "output_crosscam", "analysis",
                                     "stitch", "labels_v2.json"), encoding="utf-8"))
    same_pos = [(str(p["cam"]), int(p["a"]), str(p["cam"]), int(p["b"]))
                for p in lv["same_camera"]["positives"]]
    same_neg = [(str(p["cam"]), int(p["a"]), str(p["cam"]), int(p["b"]))
                for p in lv["same_camera"]["negatives"]]
    cross_pos = [("318", a, "320", b) for a, b in lv["cross_camera"]["positives"]]
    cross_neg = [("318", a, "320", b) for a, b in lv["cross_camera"]["negatives"]]

    results = {}
    for name, embs in [("d3b", embs_d), ("cradio_scale", embs_c)]:
        print(f"\n=== {name} ===")
        r = {}
        # 协议 A
        pv = [sim(embs, "318", a, "320", b) for a, b in POS30]
        nv = [sim(embs, "318", a, "320", b) for a, b in NEG30]
        r["protoA_30pair"] = report(f"{name} 协议A(30对)", pv, nv)
        # 协议 B
        sp = [sim(embs, *p) for p in same_pos]
        sn = [sim(embs, *p) for p in same_neg]
        cp = [sim(embs, *p) for p in cross_pos]
        cn = [sim(embs, *p) for p in cross_neg]
        r["B_same"] = report(f"{name} 同机位(58/3)", sp, sn)
        r["B_cross"] = report(f"{name} 跨机位(8/24)", cp, cn)
        r["B_all"] = report(f"{name} 全量(66/27)", sp + cp, sn + cn)
        results[name] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n[out] -> {args.out}")

    # 汇总表
    print("\n===== 汇总 =====")
    print(f"{'指标':<28}{'d3b':>12}{'cradio_scale':>14}")
    for k, label in [("protoA_30pair", "协议A AUC"), ("B_same", "同机位 AUC"),
                     ("B_cross", "跨机位 AUC"), ("B_all", "全量 AUC")]:
        a = results["d3b"][k].get("auc")
        c = results["cradio_scale"][k].get("auc")
        print(f"{label:<28}{a if a else '-':>12}{c if c else '-':>14}")


if __name__ == "__main__":
    main()
