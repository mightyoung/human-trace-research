#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO vs DINO 补充分析：每帧人数时序 + 框面积比 + 并集覆盖。

回答三个问题：
  1. 谁检的人更多？时序上谁更稳定（办公区 15min 内人数不应翻倍波动）？
  2. 匹配对的框面积比（YOLO/DINO）——DINO 的框是否系统性偏大（含椅子/邻座）？
  3. DINO ∪ YOLO 的并集有多大，YOLO 单独覆盖了 DINO 的多少？

用法: python tools/yolo_vs_dino_extra.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DINO_DIR = ROOT / "output_crosscam_local" / "1012_ab"
OUT_DIR = ROOT / "output_crosscam_local" / "yolo_vs_dino"
SEGS = ["305_test_544", "323_test_527"]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    aa = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    ba = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = aa + ba[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy(iou, thr):
    if iou.size == 0:
        return []
    idx = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    ua, ub, out = set(), set(), []
    for i, j in idx:
        v = float(iou[i, j])
        if v < thr:
            break
        if i in ua or j in ub:
            continue
        ua.add(int(i)); ub.add(int(j)); out.append((int(i), int(j), v))
    return out


def load_jsonl(p, with_score=False):
    out = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        b = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
        if with_score:
            s = np.asarray(r["scores"], np.float32).reshape(-1)
            out[int(r["frame"])] = (b, s)
        else:
            out[int(r["frame"])] = b
    return out


def main():
    report = {}
    for seg in SEGS:
        dino = load_jsonl(DINO_DIR / f"{seg}_a.jsonl")
        yolo = load_jsonl(OUT_DIR / f"{seg}_yolo.jsonl", with_score=True)
        frames = sorted(set(dino) & set(yolo))
        dcnt = np.array([len(dino[f]) for f in frames], np.float32)
        ycnt = {c: np.array([int(np.sum(yolo[f][1] >= c)) for f in frames], np.float32)
                for c in (0.1, 0.25, 0.4)}

        # 匹配对面积比
        ratios = []
        for f in frames:
            db = dino[f]
            yb, ys = yolo[f]
            keep = ys >= 0.1
            yb = yb[keep]
            for i, j, _ in greedy(iou_matrix(db, yb), 0.5):
                ad = (db[i][2] - db[i][0]) * (db[i][3] - db[i][1])
                ay = (yb[j][2] - yb[j][0]) * (yb[j][3] - yb[j][1])
                if ad > 0:
                    ratios.append(float(ay) / float(ad))
        ratios = np.array(ratios, np.float32)

        # 并集（IoU 去重，YOLO 为主）
        union_cnt = []
        for f in frames:
            db = dino[f]
            yb, ys = yolo[f]
            yb = yb[ys >= 0.1]
            if len(yb) == 0:
                union_cnt.append(len(db)); continue
            if len(db) == 0:
                union_cnt.append(len(yb)); continue
            iou = iou_matrix(yb, db)
            extra = len(db) - len(greedy(iou, 0.5))
            union_cnt.append(len(yb) + extra)
        union_cnt = np.array(union_cnt, np.float32)

        report[seg] = {
            "frames": len(frames),
            "dino_per_frame": {"mean": round(float(dcnt.mean()), 2), "std": round(float(dcnt.std()), 2),
                               "p10": float(np.percentile(dcnt, 10)), "p90": float(np.percentile(dcnt, 90)),
                               "cv": round(float(dcnt.std() / max(dcnt.mean(), 1e-6)), 3)},
            "yolo_per_frame": {f"conf{c}": {"mean": round(float(v.mean()), 2),
                                            "std": round(float(v.std()), 2),
                                            "p10": float(np.percentile(v, 10)),
                                            "p90": float(np.percentile(v, 90)),
                                            "cv": round(float(v.std() / max(v.mean(), 1e-6)), 3)}
                               for c, v in ycnt.items()},
            "union_per_frame_mean": round(float(union_cnt.mean()), 2),
            "area_ratio_yolo_over_dino": {
                "median": round(float(np.median(ratios)), 3) if len(ratios) else None,
                "p10": round(float(np.percentile(ratios, 10)), 3) if len(ratios) else None,
                "p90": round(float(np.percentile(ratios, 90)), 3) if len(ratios) else None,
                "pct_yolo_smaller": round(100.0 * float(np.mean(ratios < 0.9)), 1) if len(ratios) else None,
                "pct_yolo_larger": round(100.0 * float(np.mean(ratios > 1.1)), 1) if len(ratios) else None,
            },
            "frames_list": frames,
            "dino_counts": dcnt.tolist(),
            "yolo_counts_conf01": ycnt[0.1].tolist(),
            "yolo_counts_conf025": ycnt[0.25].tolist(),
        }
        print(f"[{seg}] dino={dcnt.mean():.2f}/f  yolo.1={ycnt[0.1].mean():.2f}/f "
              f"yolo.25={ycnt[0.25].mean():.2f}/f  union={union_cnt.mean():.2f}/f  "
              f"area_ratio_med={np.median(ratios):.3f}", flush=True)

    (OUT_DIR / "extra_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[out]", OUT_DIR / "extra_analysis.json")


if __name__ == "__main__":
    main()
