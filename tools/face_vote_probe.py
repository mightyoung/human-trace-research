# -*- coding: utf-8 -*-
"""多帧人脸一致性：能不能把「高置信度」定义成与分数无关的东西。

背景
----
`face_suggest.json` 的 score 是**整轨人脸均值向量**对其它轨的余弦。实测它随阈值非单调
（0.50 附近 46%，0.55 以上掉到 0%）—— 也就是说「分数高」不是可靠性的证据。

这个探针换一个定义：**同一轨的多帧人脸是不是都指向同一个身份**。
如果逐帧投票一致（unanimous），那就是时间一致性，与分数无关，也不受单帧质量噪声影响。

口径
----
- 图库 = **其它轨**的逐帧人脸（按轨留一，不用 45 张卡图 —— 卡图与本包同源会泄题）
- 逐帧：对该帧在其它轨的全部帧里找最像的，取该帧的 top-1 身份
- 逐轨：对 n 帧的投票取多数；同时记录是否一致
- 真值用 GIL label
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

import faceauto as FA  # noqa: E402

REID = FA.REID


def load_shots():
    q = json.loads((REID / "face_query.json").read_text(encoding="utf-8"))
    crops = REID / "face_q"
    out = {}
    for key, r in q["tracks"].items():
        if not r.get("n_used"):
            continue
        vs = []
        for s in (r.get("shots") or []):
            p = crops / s["crop"]
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            if img.shape[0] != 112:
                img = cv2.resize(img, (112, 112))
            vs.append({"frame": s["frame"], "pd": s["pd"], "q": s["q"],
                       "v": FA.embed(img)})
        if vs:
            out[key] = {"cam": r["cam"], "tid": r["tid"], "h_med": r["h_med"],
                        "shots": vs}
    return out


def main():
    shots = load_shots()
    tpl, meta = FA.templates()
    truth = {t["key"]: t["ident"] for t in tpl}
    print(f"有脸轨 {len(shots)}（templates {len(tpl)}）")

    # 逐帧向量总表
    allsh = []
    for key, r in shots.items():
        for i, s in enumerate(r["shots"]):
            if truth.get(key):
                allsh.append({"key": key, "i": i, "v": s["v"],
                              "ident": truth[key], "pd": s["pd"]})
    print(f"逐帧样本 {len(allsh)}（有真值的轨）")

    rows = []
    for key, r in shots.items():
        if not truth.get(key):
            continue
        votes = []
        for i, s in enumerate(r["shots"]):
            best, bi = -9.0, None
            for x in allsh:
                if x["key"] == key:      # 按轨留一
                    continue
                c = float(s["v"] @ x["v"])
                if c > best:
                    best, bi = c, x["ident"]
            votes.append({"i": i, "ident": bi, "cos": round(best, 4), "pd": s["pd"]})
        cnt = collections.Counter(v["ident"] for v in votes)
        top, n_top = cnt.most_common(1)[0]
        rows.append({
            "key": key, "truth": truth[key], "n": len(votes),
            "votes": votes, "n_top": n_top,
            "unanimous": n_top == len(votes),
            "vote_ident": top,
            "vote_hit": top == truth[key],
            "cos_min": min(v["cos"] for v in votes),
            "cos_mean": round(float(np.mean([v["cos"] for v in votes])), 4),
        })

    def tab(name, sub):
        n = len(sub)
        k = sum(1 for r in sub if r["vote_hit"])
        print(f"  {name:26} n={n:3}  逐轨判对 {k:3}/{n:<3} = {100.0*k/max(1,n):5.1f}%")
        return n, k

    print("\n=== 按一致性分层（逐轨投票）===")
    una = [r for r in rows if r["unanimous"] and r["n"] >= 2]
    una3 = [r for r in rows if r["unanimous"] and r["n"] >= 3]
    non = [r for r in rows if not r["unanimous"]]
    tab("全量", rows)
    tab("一致（≥2 帧全同）", una)
    tab("一致（≥3 帧全同）", una3)
    tab("不一致", non)

    print("\n=== 与整轨均值口径对比 ===")
    sug = {s["key"]: s for s in FA.load_suggest()["suggestions"]}
    both = [r for r in rows if r["key"] in sug]
    k1 = sum(1 for r in both if sug[r["key"]]["hit"])
    k2 = sum(1 for r in both if r["vote_hit"])
    print(f"  同集合 n={len(both)}  整轨均值判对 {k1}  逐帧投票判对 {k2}")

    if una:
        print("\n=== 一致组逐条 ===")
        for r in sorted(una, key=lambda x: -x["cos_min"]):
            v = " ".join(f"{x['ident']}{'✓' if x['ident']==r['truth'] else '✗'}"
                         f"@{x['cos']:.2f}" for x in r["votes"])
            print(f"  {r['key']:>8} truth={r['truth']} n={r['n']} "
                  f"{'判对' if r['vote_hit'] else '判错'}  {v}")

    out = ROOT / "tmp" / "face_vote.json"
    out.write_text(json.dumps({"rows": rows, "n_track": len(rows),
                               "unanimous": len(una), "unanimous3": len(una3)},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写盘 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
