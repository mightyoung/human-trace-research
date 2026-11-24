# -*- coding: utf-8 -*-
"""组合检索（R2.3）：桥接嵌入 + 时空门控（通道模型）→ 全库 rank 改善（v4 clean 口径）。

背景：v4 全库检索 rank1=2/8 / avg 4.0（8 个 318 查询找 320 真配）——失败项被
"外观高分但时间/通道上不属于同一行程"的干扰轨占据（如 (1,1) 被 316|113 压到 rank16）。

门控模型（dt 窗来自 relay REPORT §四 通道拓扑实测；对称 ±D，单位秒）：
  318↔320 同楼面重叠视野（可共视）      ±60
  318↔316 走廊↔办公区2 门（观测 1.4–18.5s） ±60
  318↔305 办公区1↔办公区2 门（观测 5.8–23.9s）±90
  318↔323 经走廊（323↔316 15s + 316↔318 ≈10s）±90
  318↔343 经电梯厅（343→316 1.1s + …）    ±120
规则：候选轨 [s0,s1] 与查询 [t0,t1] 的区间必须在扩展窗 [-D, +D] 内有交集。

三种模式：
  A. baseline  : 全库暴力（复现 v4_compare）
  B. +时间门控  : 候选 = 通过区间门的其他机位轨
  C. +mutual-best: 在 B 基础上要求"候选对（反向门控后）的 318 侧最佳匹配 = 查询"

用法：python tools/combo_retrieval.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_reid_store import GlobalReIDStore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "output_crosscam", "analysis", "global_reid", "store_cradio_v4")
DATE = "2026-10-12"
QCAM = "318"
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
# 通道时间窗（±秒）
GATE_D = {"320": 60, "316": 60, "305": 90, "323": 90, "343": 120}


def sim_to(q_emb, E, keys):
    q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    return E @ q


def main():
    st = GlobalReIDStore(STORE, "cradio_gemma2_bridge_v4", 768, mrl_dim=256)
    n = st.load()
    print(f"[load] store v4: {n} tracks")
    E, keys = st._matrix()
    meta = st.meta

    # 校验时间字段齐全
    miss = [k for k in keys if meta[k].get("t_first") is None]
    assert not miss, f"{len(miss)} 轨缺时间: {miss[:5]}"

    rows_out = []
    agg = {"A": {"r1": 0, "rs": []}, "B": {"r1": 0, "rs": [], "gate_out": 0},
           "C": {"r1": 0, "rs": [], "mutual_out": 0}}
    for a, b in POS30:
        qk = f"{DATE}|{QCAM}|{a}"
        tk = f"{DATE}|320|{b}"
        q_emb = st.rows[qk]
        t0, t1 = meta[qk]["t_first"], meta[qk]["t_last"]
        s = sim_to(q_emb, E, keys)

        # ---- A: baseline 全库（排除查询机位） ----
        order = np.argsort(-s)
        candA = [keys[i] for i in order if meta[keys[i]]["cam"] != QCAM]
        rankA = candA.index(tk) + 1 if tk in candA else None
        topA = candA[:5]

        # ---- B: 时间门控 ----
        candB = []
        for i in order:
            k = keys[i]
            m = meta[k]
            if m["cam"] == QCAM:
                continue
            D = GATE_D.get(m["cam"])
            if D is None:
                continue
            s0, s1 = m["t_first"], m["t_last"]
            if s1 >= t0 - D and s0 <= t1 + D:
                candB.append(k)
        gate_out = len(candA) - len(candB)
        rankB = candB.index(tk) + 1 if tk in candB else None
        topB = candB[:5]

        # ---- C: B + mutual-best（候选反向最佳 318 匹配 == 查询） ----
        # 预计算：所有 318 轨嵌入
        k318 = [k for k in keys if meta[k]["cam"] == QCAM]
        E318 = np.stack([st.rows[k] for k in k318])
        E318 = E318 / (np.linalg.norm(E318, axis=1, keepdims=True) + 1e-9)
        candC = []
        for k in candB:
            m = meta[k]
            D = GATE_D[m["cam"]]
            s0, s1 = m["t_first"], m["t_last"]
            # 反向门控：318 侧哪些轨与该候选时间可达
            rev = [j for j, k2 in enumerate(k318)
                   if meta[k2]["t_last"] >= s0 - D and meta[k2]["t_first"] <= s1 + D]
            if not rev:
                continue
            c_emb = st.rows[k]
            sims = E318[rev] @ (c_emb / (np.linalg.norm(c_emb) + 1e-9))
            jbest = rev[int(np.argmax(sims))]
            if k318[jbest] == qk:
                candC.append(k)
        mutual_out = len(candB) - len(candC)
        rankC = candC.index(tk) + 1 if tk in candC else None
        topC = candC[:5]

        rows_out.append({
            "pair": [a, b], "q_interval": [round(t0, 1), round(t1, 1)],
            "A": {"rank": rankA, "top5": topA, "n": len(candA)},
            "B": {"rank": rankB, "top5": topB, "n": len(candB), "gate_out": gate_out},
            "C": {"rank": rankC, "top5": topC, "n": len(candC), "mutual_out": mutual_out},
        })
        for mode, r in (("A", rankA), ("B", rankB), ("C", rankC)):
            agg[mode]["rs"].append(r)
            if r == 1:
                agg[mode]["r1"] += 1
        if rankB is None:
            agg["B"]["gate_out"] += 1
        if rankC is None:
            agg["C"]["mutual_out"] += 1

    # ---- 输出 ----
    print(f"\n{'pair':>10} | {'A(baseline)':>22} | {'B(+时间门控)':>22} | {'C(+mutual)':>22}")
    for r in rows_out:
        def fmt(m):
            rk = m["rank"]
            return f"rank {str(rk):>4} (n={m['n']})"
        print(f"{str(r['pair']):>10} | {fmt(r['A']):>22} | {fmt(r['B']):>22} | {fmt(r['C']):>22}")

    def summ(mode):
        rs = [x for x in agg[mode]["rs"] if x]
        n_out = sum(1 for x in agg[mode]["rs"] if x is None)
        return (f"rank1 {agg[mode]['r1']}/8 | avg {np.mean(rs):.2f} ({len(rs)} 有效) | "
                f"被滤除 {n_out}")

    print(f"\n[A baseline   ] {summ('A')}")
    print(f"[B 时间门控    ] {summ('B')}  （门控共剔除候选见各对）")
    print(f"[C +mutual-best] {summ('C')}")

    print("\n---- 明细：各模式 top5 与命中 ----")
    for r in rows_out:
        print(f"\npair {r['pair']}  q[{r['q_interval'][0]}~{r['q_interval'][1]}]s")
        for mode in ("A", "B", "C"):
            m = r[mode]
            print(f"  {mode} rank={m['rank']} n={m['n']} top5={m['top5']}")

    out = os.path.join(ROOT, "output_crosscam", "analysis", "global_reid",
                       "combo_retrieval_v4.json")
    json.dump({"rows": rows_out,
               "summary": {m: {"rank1": agg[m]["r1"],
                               "avg": float(np.mean([x for x in agg[m]["rs"] if x]))
                               if any(agg[m]["rs"]) else None,
                               "ranks": agg[m]["rs"]}
                           for m in ("A", "B", "C")},
               "gate_D": GATE_D},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
