# -*- coding: utf-8 -*-
"""选择器实时增量：边跑边写池，会不会越跑越准（或越跑越偏）。

为什么单独测
------------
「让人脸把身份实时绑到簇上」这件事的机制部分，其实**不需要人脸**：
只要选择器能在流式处理中，把**已判定的轨**写进该身份的池，后面的查询就多一份模板。
人脸只是其中一种触发源。

所以要先把机制本身量出来，再决定要不要给它接人脸：
    history      只用历史库（t0 早于查询的轨，按簇归组）
    online       历史库 + 自学习池（每条轨判定后，若置信度够就写进所判身份的池）
    online_or    online 的**上限**：写池时用真值身份（等于「每次判定都对」）
    face         历史库 + 人脸绑定池（真实人脸识别，会错）

口径
----
- 严格因果：库与池都只用 t0 早于查询 t0 的条目。
- 历史库按「机位×簇」归组，形态加权（与生产一致）。
- 自学习触发：top-1 分数 ≥ `--tau` 且相对第二名分差 ≥ `--margin`。
- 评估用簇号（只做相对比较，不能当对外精度）。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

import selector as S  # noqa: E402

FPS = 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.85, help="自学习写池的判定阈值")
    ap.add_argument("--margin", type=float, default=0.05, help="自学习写池的分差要求")
    ap.add_argument("--qt", type=float, default=0.80, help="查询判定阈值")
    ap.add_argument("--face-gate", type=float, default=0.50)
    ap.add_argument("--face-margin", type=float, default=0.05)
    args = ap.parse_args()

    sel = S.init()
    if not sel.ready:
        print("选择器未就绪：", sel.err)
        return 2
    meta = sel.meta
    order = sorted(range(len(meta)),
                   key=lambda i: (float(meta[i].get("t0") or 0), meta[i]["cam"]))
    order = [i for i in order if float(meta[i].get("h_med") or 0) > 0]
    t0 = {i: float(meta[i].get("t0") or 0) for i in order}
    print(f"流式处理 {len(order)} 条轨")

    # 人脸绑定（真实）
    fb = []
    fp = S.REID / "face_suggest.json"
    if fp.exists():
        for s in json.loads(fp.read_text(encoding="utf-8"))["suggestions"]:
            i = sel.idx_of(s["cam"], int(s["tid"]))
            if i is None:
                continue
            r = meta[i]
            f, fr = s.get("frame"), r.get("f_rep")
            a = float(r.get("t1") or 0)
            if f is not None and fr is not None:
                a = min(max(float(r.get("t0") or 0) + (float(f) - float(fr)) / FPS,
                            float(r.get("t0") or 0)), a)
            fb.append({"i": i, "ent": s["target_entity"], "hit": bool(s.get("hit")),
                       "score": float(s["score"]), "margin": float(s.get("margin") or 0),
                       "t": a})

    res = {}
    for mode in ("history", "online", "online_or", "face"):
        pool = collections.defaultdict(list)     # (cam, ent) -> [(t, i)]
        seen = collections.defaultdict(list)     # (cam, ent) -> [(i, h)]
        n = hit = 0
        n0 = hit0 = 0
        drift = 0
        for qi in order:
            cam = meta[qi]["cam"]
            ent = sel.ent_of[qi]
            tn = t0[qi]
            # 历史库（因果）
            sel.groups = {}
            for (c, e), items in seen.items():
                if c == cam:
                    sel.groups[(c, e)] = list(items)
            # 池（因果）
            sel.pool = {}
            for (c, e), items in pool.items():
                if c != cam:
                    continue
                live = [{"i": i, "cam": c, "tid": int(meta[i]["tid"]), "src": mode,
                         "q": None, "at": t, "note": ""}
                        for (t, i) in items
                        if t < tn and i != qi]
                if live:
                    sel.pool[(c, e)] = live[-8:]
            try:
                r = sel.query(cam, int(meta[qi]["tid"]), scope="cam", tau=args.qt)
            except Exception:
                r = None
            if r and r["candidates"]:
                n += 1
                tp = r["top_both"]
                if tp and tp["accepted"]:
                    hit += int(tp["entity"] == ent)
                if mode != "history":
                    tl = r["top_lib"]
                    # 自学习条件：库侧判定要够硬
                    if tl and tl["accepted"] and tl["score"] >= args.tau:
                        cand = sorted([c for c in r["candidates"] if c["s_both"] is not None],
                                      key=lambda c: (-c["s_both"], c["entity"]))
                        mg = (cand[0]["s_both"] - cand[1]["s_both"]
                              if len(cand) > 1 else 1.0)
                        if mg >= args.margin:
                            write_ent = (ent if mode == "online_or" else cand[0]["entity"])
                            if mode == "online_or" or cand[0]["entity"] == ent:
                                pass
                            if mode != "online_or" and cand[0]["entity"] != ent:
                                drift += 1
                            pool[(cam, write_ent)].append((tn, qi))
            # 人脸绑定：在 query 之后写入（因果）
            if mode == "face":
                for b in fb:
                    if b["i"] == qi and b["score"] >= args.face_gate \
                            and b["margin"] >= args.face_margin:
                        pool[(cam, b["ent"])].append((b["t"], qi))
            # 历史库：query 之后才可见
            h = float(meta[qi].get("h_med") or 0)
            seen[(cam, ent)].append((qi, h))

        pr = 100.0 * hit / max(1, n)
        res[mode] = dict(n=n, hit=hit, prec=round(pr, 1), drift=drift,
                         n_pool=sum(len(v) for v in pool.values()))
        print(f"{mode:10} 查询 {n:4}  判对 {hit:4} = {pr:5.1f}%  "
              f"池条目 {res[mode]['n_pool']:4}  误写 {drift}")

    b = res["history"]["prec"]
    print(f"\n相对 history 的变化：online {res['online']['prec']-b:+.1f}pt，"
          f"online_or {res['online_or']['prec']-b:+.1f}pt，"
          f"face {res['face']['prec']-b:+.1f}pt")
    print("★ 评估分母是「库给出候选」的查询；两种模式分母不同时不可直接比，"
          "故此处打印各自 n。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
