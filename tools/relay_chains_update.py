# -*- coding: utf-8 -*-
"""链拓扑更新：把人工否证（裁决台账 relay_conflict reject）从 relay 链边集中剔除并重建链。

背景：
  relay_chains_cradio_v5.json（47 边 / 32 链，cradio v5 mutual-best 口径）是链拓扑的
  正式产物（等价 relay_v2 primary 47 边的来源）。PRT 补口批人工否证 3 条（全部落在
  47 边内：323-E069->343-E040 / 323-E022->318-E036 / 316-E049->323-E092），
  链结构必须同步更新。

机制（与 relay_rescore.do_mutual 链构建逐位一致，仅输入边集不同）：
  边集 = v5 47 边 − 台账中 action=reject 且在该 47 边内的边；
  链构建 = succ/pred 图 → head（无前驱）沿唯一后继贪心延伸 → 剩余边再链 → 按长度排序。

安全：
  - 不覆盖 v5 产物；输出 relay_chains_cradio_v6.json（全新文件）
  - reject 边只在"确在 v5 边集内"时剔除；不在则忽略（v1 裁决 7 条全在 audit 池，不受影响）
  - 只读台账与 v5；不写 GIL。

用法：
  python tools/relay_chains_update.py            # 默认：v5 + human_adjudication_v2
  python tools/relay_chains_update.py --ver <台账路径> --out <输出路径>
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
CHAINS = os.path.join(ANA, "relay_clean2", "relay_chains_cradio_v5.json")
VER = os.path.join(ANA, "gil", "human_adjudication_v2.json")
OUT = os.path.join(ANA, "relay_clean2", "relay_chains_cradio_v6.json")


def build_chains(mut):
    """与 relay_rescore.do_mutual 完全一致的链构建（逐位保守）。"""
    succ, pred = defaultdict(list), defaultdict(list)
    for i, e in enumerate(mut):
        e["_i"] = i
        succ[e["a"]].append(e)
        pred[e["b"]].append(e)
    nodes = set()
    for e in mut:
        nodes.add(e["a"])
        nodes.add(e["b"])
    heads = sorted(n for n in nodes if not pred[n])
    chains, used = [], set()
    for h in heads:
        chain = [h]
        cur = h
        while True:
            if len(succ[cur]) != 1:
                break
            e = succ[cur][0]
            if e["_i"] in used:
                break
            used.add(e["_i"])
            chain.append(e["b"])
            cur = e["b"]
        if len(chain) >= 2:
            chains.append(chain)
    for e in mut:
        if e["_i"] in used:
            continue
        chain = [e["a"], e["b"]]
        used.add(e["_i"])
        cur = e["b"]
        while True:
            if len(succ[cur]) != 1:
                break
            e2 = succ[cur][0]
            if e2["_i"] in used:
                break
            used.add(e2["_i"])
            chain.append(e2["b"])
            cur = e2["b"]
        if len(chain) >= 2:
            chains.append(chain)
    chains.sort(key=lambda c: -len(c))
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default=VER)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    d = json.load(open(CHAINS, encoding="utf-8"))
    vj = json.load(open(a.ver, encoding="utf-8"))
    edges = d["edges"]
    eset = {(e["a"], e["b"]): e for e in edges}

    # 台账中 reject 且落在 47 边内的边
    rej_in = []
    rej_out = []
    for it in vj["items"]:
        if it.get("group") != "relay_conflict" or it.get("action") != "reject":
            continue
        ea, eb = it["edge"].split("->")
        if (ea, eb) in eset:
            rej_in.append((ea, eb, it))
        else:
            rej_out.append((ea, eb, it))

    print(f"[input] v5: {len(edges)} 边 / {d['n_chains']} 链")
    print(f"[台账] reject 共 {len(rej_in) + len(rej_out)}（在 47 边内 {len(rej_in)}；池外 {len(rej_out)} 忽略）")
    for ea, eb, it in rej_in:
        print(f"  剔边 {ea}->{eb}  (v={it['verdict']}, {it.get('rationale', '')[:40]})")

    keep = [dict(e) for e in edges if (e["a"], e["b"]) not in {(x[0], x[1]) for x in rej_in}]
    chains = build_chains(keep)

    n_multi = len([c for c in chains if len(c) >= 3])
    print(f"[chains] {len(edges)} -> {len(keep)} 边 | {d['n_chains']} -> {len(chains)} 链（多跳 {n_multi}）")
    print(f"  最长链 {len(chains[0])} 跳: {chains[0]}")

    # 链结构对照（哪些旧链变了）
    old_ch = [" -> ".join(c) for c in d["chains"]]
    new_ch = [" -> ".join(c) for c in chains]
    old_set, new_set = set(old_ch), set(new_ch)
    print(f"  旧链保留 {len(old_set & new_set)}；消失 {len(old_set - new_set)}；新增 {len(new_set - old_set)}")
    for c in sorted(old_set - new_set)[:8]:
        print(f"    - {c}")
    for c in sorted(new_set - old_set)[:8]:
        print(f"    + {c}")

    out = dict(
        version="relay_chains_cradio_v6",
        based_on="relay_chains_cradio_v5.json",
        created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        note=("v6 = v5 口径 − 人工否证边（human_adjudication_v2 relay_conflict reject，"
              "全部落在 47 边内）。链构建算法与 relay_rescore.do_mutual 一致。"),
        excluded_edges=[{"a": ea, "b": eb, "verdict": it["verdict"],
                         "src": it.get("src", "human_adjudication")} for ea, eb, it in rej_in],
        n_chains=len(chains),
        chains=chains,
        edges=keep,
    )
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[out] {a.out}")
    # 复算自检：边数守恒
    assert len(keep) == len(edges) - len(rej_in), "边数不守恒！"
    assert len(chains[0]) >= 2, "链异常"
    print("[selftest] 边数守恒 ✓ / 链非空 ✓")


if __name__ == "__main__":
    main()
