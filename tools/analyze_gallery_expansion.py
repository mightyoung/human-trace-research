#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扩图库实验分解：新增图库轨迹去了哪、新增接受判对没有。

回答三个问题（单看 coverage 会误导）：
  1. 新增的图库轨迹被吸收到哪 —— 已有簇 / 全新簇 / 被 MIN_CLUSTER 丢弃
  2. 新增接受的 query 落在什么簇上 —— 有已标注成员（可核验）还是全未标注（只能记 U）
  3. 相对基线，C/W/U 的逐条转移（新增哪些判对、流失哪些原本判对）

用法：
  PY tools/analyze_gallery_expansion.py \
      --base deliverable/data-request-v2/results --cfg-base A_baseline \
      --exp  deliverable/data-request-v2/results_g4 --cfg-exp A_baseline
"""
import argparse
import collections
import csv
import os


def load_members(run, cfg):
    p = os.path.join(run, "members", f"{cfg}.csv")
    m = collections.defaultdict(list)
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        m[int(r["cluster_local"])].append(r["track_id"])
    return {k: v for k, v in m.items()}


def load_verdicts(run, cfg):
    p = os.path.join(run, "verdicts", f"{cfg}.csv")
    out = {}
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        out[r["query_id"]] = r
    return out


def load_queries(run, cfg):
    p = os.path.join(run, "queries", f"{cfg}.csv")
    return {r["query_id"]: r for r in csv.DictReader(open(p, encoding="utf-8-sig"))}


def load_summary(run, cfg):
    p = os.path.join(run, "summary.csv")
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        if r["config"] == cfg:
            return r
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cfg-base", default="A_baseline")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--cfg-exp", default="A_baseline")
    a = ap.parse_args()

    mb, me = load_members(a.base, a.cfg_base), load_members(a.exp, a.cfg_exp)
    tb, te = set(), set()
    for v in mb.values():
        tb |= set(v)
    for v in me.values():
        te |= set(v)
    sb, se = load_summary(a.base, a.cfg_base), load_summary(a.exp, a.cfg_exp)

    print("=" * 78)
    print(f"基线 {a.cfg_base}: K={sb.get('K')} θ={sb.get('theta')} "
          f"cov={sb.get('both_accept_rate')} same={sb.get('ab_same_rate')} "
          f"fm={sb.get('fm_raw')}")
    print(f"实验 {a.cfg_exp}: K={se.get('K')} θ={se.get('theta')} "
          f"cov={se.get('both_accept_rate')} same={se.get('ab_same_rate')} "
          f"fm={se.get('fm_raw')}")
    print("-" * 78)
    print(f"进簇轨迹 base={len(tb)} exp={len(te)} 新增={len(te - tb)} 流失={len(tb - te)}")

    # 1) 新增轨迹去向
    new_tracks = te - tb
    new_clusters = [c for c, mem in me.items() if mem and set(mem).isdisjoint(tb)]
    cluster_of = {t: c for c, mem in me.items() for t in mem}
    to_old = [t for t in new_tracks if cluster_of.get(t) not in new_clusters]
    to_new = [t for t in new_tracks if cluster_of.get(t) in new_clusters]
    print(f"\n[1] 新增轨迹去向：并入已有簇 {len(to_old)} 条 / 组成全新簇 {len(to_new)} 条"
          f"（全新簇 {len(new_clusters)} 个，成员 {sum(len(me[c]) for c in new_clusters)} 条）")
    per_old = collections.Counter(cluster_of[t] for t in to_old)
    if per_old:
        print("    受影响已有簇：" + "，".join(
            f"簇{c}(+{n})" for c, n in per_old.most_common(12)))

    # 2) 新簇是否可核验
    vb = load_verdicts(a.base, a.cfg_base)
    ve = load_verdicts(a.exp, a.cfg_exp)
    qb = load_queries(a.base, a.cfg_base)
    qe = load_queries(a.exp, a.cfg_exp)

    def nlab(qid, vmap):
        r = vmap.get(qid)
        return int(r["cluster_n_lab"]) if r and r.get("cluster_n_lab") else -1

    # 3) 逐 query 转移
    acc_b = {q for q, r in qb.items() if r["final_status"] == "accepted"}
    acc_e = {q for q, r in qe.items() if r["final_status"] == "accepted"}
    print(f"\n[2] 接受数 base={len(acc_b)} exp={len(acc_e)} "
          f"净 {len(acc_e) - len(acc_b):+d}")
    trans = collections.Counter()
    for q in qb:
        trans[("acc" if q in acc_b else "rej", "acc" if q in acc_e else "rej")] += 1
    print("    转移矩阵（base→exp）：", dict(trans))

    gained = sorted(acc_e - acc_b)
    lost = sorted(acc_b - acc_e)
    print(f"\n[3] 新增接受 {len(gained)} 条：")
    gv = collections.Counter()
    for q in gained:
        v = ve.get(q, {}).get("verdict", "?")
        gv[v] += 1
        print(f"    {q:18s} verdict={v} 簇已标注数={nlab(q, ve):3d} "
              f"真值={ve.get(q, {}).get('q_pid', ''):8s} 簇主体={ve.get(q, {}).get('c_pid', '')}")
    print(f"    小计 C={gv['C']} W={gv['W']} U={gv['U']} 其他={sum(v for k, v in gv.items() if k not in 'CWU')}")

    print(f"\n[4] 流失 {len(lost)} 条：")
    lv = collections.Counter()
    for q in lost:
        v = vb.get(q, {}).get("verdict", "?")
        lv[v] += 1
        print(f"    {q:18s} 原verdict={v} 簇已标注数={nlab(q, vb):3d}")
    print(f"    小计（原判决）C={lv['C']} W={lv['W']} U={lv['U']}")

    # 5) 净 C 变化
    def cw(vmap):
        c = sum(1 for r in vmap.values() if r.get("verdict") == "C")
        w = sum(1 for r in vmap.values() if r.get("verdict") == "W")
        u = sum(1 for r in vmap.values() if r.get("verdict") == "U")
        return c, w, u
    cb, wb, ub = cw(vb)
    ce, we, ue = cw(ve)
    print("\n" + "=" * 78)
    print(f"C/229  base={cb} → exp={ce} ({ce - cb:+d})")
    print(f"精度   base={cb/(cb+wb):.3f} → exp={ce/(ce+we):.3f}" if cb + wb and ce + we else "")
    print(f"W      base={wb} → exp={we} ({we - wb:+d})｜U base={ub} → exp={ue} ({ue - ub:+d})")
    gain_c = sum(1 for q in gained if ve.get(q, {}).get("verdict") == "C")
    lost_c = sum(1 for q in lost if vb.get(q, {}).get("verdict") == "C")
    print(f"分解：新增判对 {gain_c} 条 − 流失判对 {lost_c} 条 "
          f"= 净 {gain_c - lost_c:+d}；另有簇主体变化导致的改判（见上表）")


if __name__ == "__main__":
    main()
