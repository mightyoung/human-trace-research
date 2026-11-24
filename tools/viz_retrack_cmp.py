#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""目检：同一帧上叠两套检测/跟踪结果，看策略到底改了哪个框。

用法:
  python tools/viz_retrack_cmp.py --video <305.mp4> --frames 4425,4435,4440 \
      --a base.jsonl --b hy.jsonl --out probe_out/viz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load(p: str) -> dict[int, tuple[list[int], list[list[float]]]]:
    m = {}
    with open(p, encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            m[int(d["frame"])] = (d.get("track_ids") or [], d.get("boxes") or [])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", default="4425,4435,4440")
    ap.add_argument("--a", required=True, help="基准（绿）")
    ap.add_argument("--b", required=True, help="实验（蓝）")
    ap.add_argument("--out", default="probe_out/viz")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()

    A, B = load(a.a), load(a.b)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")

    for fs in a.frames.split(","):
        fr = int(fs)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            print(f"[warn] {fr} read fail")
            continue
        panels = []
        for name, M, col in (("base", A, (80, 220, 80)), ("exp", B, (255, 160, 60))):
            vis = img.copy()
            tids, boxes = M.get(fr, ([], []))
            for t, b in zip(tids, boxes):
                x1, y1, x2, y2 = (int(v) for v in b)
                h = y2 - y1
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3)
                cv2.putText(vis, f"#{t} h{h}", (x1 + 4, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(vis, f"{name}  f{fr}  n={len(boxes)}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4)
            cv2.putText(vis, f"{name}  f{fr}  n={len(boxes)}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            panels.append(vis)
        cmp_img = np.hstack(panels)
        if a.scale != 1.0:
            cmp_img = cv2.resize(cmp_img, None, fx=a.scale, fy=a.scale,
                                 interpolation=cv2.INTER_AREA)
        p = out / f"cmp_f{fr}.png"
        cv2.imwrite(str(p), cmp_img)
        print(f"[out] {p}  base n={len(A.get(fr,([],[]))[1])} exp n={len(B.get(fr,([],[]))[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
