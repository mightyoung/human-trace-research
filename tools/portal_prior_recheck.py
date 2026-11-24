#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补口批复验：PRT 判读 × 先验排序 的定量验收（EVAL-4 未完成项②的部分实现）。

输入：
  labels_remote_v2.json（含 PRT_ 判读，远程快照）
  analysis/relay_v2/scored_v2_prior.json（rank_old/rank_new/w_portal）
  analysis/truth/v2portal/selection_v1.json（分组 G1–G4）
输出：
  analysis/relay_v2/portal_prior_recheck_v1.json + 控制台表

口径注意（防误读）：
  - G1/G3 为**条件采样**（按先验行为选择：被顶起/被压制），其绝对同人率不可外推；
  - G2（每 a 端 top-2）/G4（active 边）相对中立，可读绝对率；
  - 严格比较在 **per-a 端**内（rank 同集内可比）。
"""
import argparse
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "output_crosscam", "analysis")
LABELS = os.path.join(ROOT, "output_crosscam", "tmp_remote", "labels_remote_v2.json")
PRIOR = os.path.join(ANA, "relay_v2", "scored_v2_prior.json")
SEL = os.path.join(ANA, "truth", "v2portal", "selection_v1.json")
OUT = os.path.join(ANA, "relay_v2", "portal_prior_recheck_v1.json")
VNAME = {1: "同人", 0: "不是", 2: "看不清"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    lb = json.load(open(args.labels, encoding="utf-8"))["pair"]
    sel = json.load(open(SEL, encoding="utf-8"))["items"]
    prior = {r["a"].replace(":", "_E_") + "__" + r["b"].replace(":", "_E_"): r
             for r in json.load(open(PRIOR, encoding="utf-8"))["candidates"]}

    rows = []
    for x in sel:
        pid, key = x["pid"], x["pid"][4:]
        v = (lb.get(pid) or {}).get("v")
        pr = prior.get(key)
        if pr is None:
            continue
        rows.append(dict(pid=pid, grp=x["grp"], a=x["a"], b=x["b"], v=v,
                         ro=pr["rank_old"], rn=pr["rank_new"],
                         w=pr["w_portal"], sim=pr["sim"], st=pr.get("status"),
                         pre=pr.get("preselect")))

    out = {"version": "relay_v2/portal_recheck/1", "n": len(rows)}

    # 1) 分组统计
    print("=== 1. 分组判读分布 ===")
    grp_stat = {}
    for g in ("G1", "G2", "G3", "G4"):
        sub = [r for r in rows if r["grp"] == g]
        c = Counter(r["v"] for r in sub)
        grp_stat[g] = {VNAME.get(k, "未判"): v for k, v in c.items()}
        print(f"  {g}: n={len(sub)} | 同人 {c.get(1,0)} 不是 {c.get(0,0)} 看不清 {c.get(2,0)}")
    out["by_group"] = grp_stat

    eff = [r for r in rows if r["v"] in (0, 1)]      # 有效判读
    pos = [r for r in eff if r["v"] == 1]
    neg = [r for r in eff if r["v"] == 0]
    print(f"\n  有效判读 {len(eff)}（同人 {len(pos)} / 不是 {len(neg)}）")

    # 2) rank 分桶同人率（新 vs 旧）
    def bucketed(key, name):
        print(f"\n=== 2{'a' if name=='new' else 'b'}. rank_{name} 分桶同人率 ===")
        out_b = {}
        for lo, hi, lbl in ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+")):
            sub = [r for r in eff if lo <= r[key] <= hi]
            if not sub:
                continue
            c = Counter(r["v"] for r in sub)
            rate = c.get(1, 0) / len(sub)
            out_b[lbl] = dict(n=len(sub), same=c.get(1, 0), diff=c.get(0, 0),
                              same_rate=round(rate, 3))
            print(f"  rank {lbl:<4}: n={len(sub):<2} 同人 {c.get(1,0)} | 同人率 {rate:.2f}")
        return out_b

    out["by_rank_new"] = bucketed("rn", "new")
    out["by_rank_old"] = bucketed("ro", "old")

    # 3) per-a 端：pos/neg 的 rank（严格同集可比）
    print("\n=== 3. per-a 端：同人 vs 不是 的 rank 对比（新/旧）===")
    bya = defaultdict(list)
    for r in eff:
        bya[r["a"]].append(r)
    pa_stat = []
    for a, rs in sorted(bya.items()):
        ps = [r for r in rs if r["v"] == 1]
        ns = [r for r in rs if r["v"] == 0]
        row = dict(a=a, n=len(rs),
                   pos_rn=[r["rn"] for r in ps], pos_ro=[r["ro"] for r in ps],
                   neg_rn=[r["rn"] for r in ns], neg_ro=[r["ro"] for r in ns])
        pa_stat.append(row)
        print(f"  {a:<16} 同人 rank_new={row['pos_rn']} (old {row['pos_ro']}) | "
              f"不是 rank_new={row['neg_rn']} (old {row['neg_ro']})")
    out["per_a"] = pa_stat

    # 4) 先验损益（判读对维度）
    print("\n=== 4. 先验损益（rank_old → rank_new）===")
    gain = loss = 0
    mov = []
    for r in eff:
        d = r["ro"] - r["rn"]          # >0 = 上移
        if r["v"] == 1 and d > 0:
            gain += 1
        if r["v"] == 1 and d < 0:
            loss += 1
        if d != 0:
            mov.append(dict(pid=r["pid"], grp=r["grp"], v=VNAME[r["v"]],
                            ro=r["ro"], rn=r["rn"]))
    print(f"  同人对：被先验上移 {gain} / 下移 {loss}（净 {gain-loss:+d}）")
    print(f"  排序变化条数：{len(mov)}/{len(eff)}")
    for m in mov:
        print(f"    {m['pid'][4:]:<40} {m['v']} {m['ro']}->{m['rn']}")
    out["pos_moved_up"], out["pos_moved_down"] = gain, loss
    out["moved"] = mov

    # 5) top-1 精度（per-a，有判读）
    print("\n=== 5. top-1 / top-2 接续（per-a 端，有判读的 a）===")
    t1_ok = t1_all = 0
    t2_ok = t2_all = 0
    t1_detail = []
    for a, rs in bya.items():
        r1 = [r for r in rs if r["rn"] == 1]
        if not r1:
            continue
        t1_all += 1
        ok1 = any(r["v"] == 1 for r in r1)
        t1_ok += int(ok1)
        r2 = [r for r in rs if r["rn"] <= 2]
        t2_all += 1
        ok2 = any(r["v"] == 1 for r in r2)
        t2_ok += int(ok2)
        # 误接：top-1 纯"不是"判定
        t1_detail.append(dict(a=a, top1_same=ok1,
                              top1_v=[VNAME.get(r["v"]) for r in r1]))
    print(f"  top-1 为同人的 a 端：{t1_ok}/{t1_all}")
    print(f"  top-2 含同人的 a 端：{t2_ok}/{t2_all}")
    for d in t1_detail:
        print(f"    {d['a']:<16} top1={d['top1_v']} same={d['top1_same']}")
    out["top1"] = dict(ok=t1_ok, n=t1_all)
    out["top2"] = dict(ok=t2_ok, n=t2_all)

    # 6) G4 否证清单
    print("\n=== 6. relay active 边被否证清单（需处理）===")
    neg4 = [r for r in rows if r["grp"] == "G4" and r["v"] == 0]
    for r in neg4:
        print(f"  {r['a']} -> {r['b']}  status={r['st']} sim={r['sim']} w={r['w']}")
    out["g4_negated"] = [dict(a=r["a"], b=r["b"], sim=r["sim"], w=r["w"]) for r in neg4]

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
