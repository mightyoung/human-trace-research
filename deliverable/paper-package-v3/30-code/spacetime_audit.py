"""全局身份图的时空一致性审计（Global ReID v1 第一步）

背景：long35 身份图（tmp/full/long35_crosscam_attr8.json，443 组）是「外观 only」的合并，
没有任何时空约束。本脚本用 OSD 绝对时钟给每条轨算真实起止，检查三类物理矛盾：

  A. 非共视机位「同时在场」——同一人不可能同时出现在两个不共享视野的画面
  B. 跨机位 Δt 短于物理通行时间——需要门拓扑/转移时间分布才判，本轮只统计分布
  C. 组内时间线自相矛盾（同组两条轨互相覆盖且机位不同）

共视/同室先验（来自 docs/depth-ownership-and-camera-topology.md L149 与 d1-covis-model）：
  - 318 × 320 真共视（同室，单应映射 AUC 1.000）→ 重叠合理
  - 其余 14 对非共视；316 是走廊（星型拓扑的脊柱）

OSD 首帧（docs/memory-full-snapshot-2026-10-24.md L69，相对 318 08:55:20 的秒偏移）：
  318 0 / 320 1 / 323 2 / 316 16 / 305 25(秒位不详，±1) / 343 121
输出：tmp/full/spacetime_audit.json
"""
import json
import itertools
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "webapp/backend/data/tracks"
GRAPH = ROOT / "tmp/full/long35_crosscam_attr8.json"
GT = ROOT / "deliverable/gt_long35/gt_long35.json"
OUT = ROOT / "tmp/full/spacetime_audit.json"

OSD = {"318": 0.0, "320": 1.0, "323": 2.0, "316": 16.0, "305": 25.0, "343": 121.0}
OSD_UNCERTAIN = {"305": 1.0}          # 秒位读不到，留 ±1s
COVIS = {("318", "320")}              # 真共视：允许同时在场
FPS = 25.0
OVERLAP_HARD = 10.0                   # 判「同时在场」的缓冲，吸收 OSD ±1~2s 误差


def load_spans():
    spans = {}
    for cam in OSD:
        doc = json.loads((TRACKS / f"{cam}.json").read_text(encoding="utf-8"))
        tr = doc["tracks"]
        spans[cam] = {
            int(k): (OSD[cam] + v[0][0] / FPS, OSD[cam] + v[-1][0] / FPS)
            for k, v in tr.items() if v
        }
    return spans


def main():
    spans = load_spans()
    groups = json.loads(GRAPH.read_text(encoding="utf-8"))["groups"]
    gt = json.loads(GT.read_text(encoding="utf-8"))["items"]

    # 真值索引：(cam,tid) -> label
    gt_label = {}
    for it in gt:
        for side in ("a", "b"):
            cam, tid = it[side].split(":")
            gt_label.setdefault((cam, int(tid)), set()).add(it["label"])

    multi = [g for g in groups if len({c for c, _ in g}) > 1]
    viol_a, pairs_dt = [], []
    for gi, g in enumerate(multi):
        cams = sorted({c for c, _ in g})
        for (c1, t1), (c2, t2) in itertools.combinations(g, 2):
            if c1 == c2:
                continue
            s1, s2 = spans[c1][int(t1)], spans[c2][int(t2)]
            ov = min(s1[1], s2[1]) - max(s1[0], s2[0])
            pair = tuple(sorted((c1, c2)))
            if ov > OVERLAP_HARD and pair not in COVIS:
                viol_a.append({
                    "group": gi, "pair": "-".join(pair), "overlap_s": round(ov, 1),
                    "a": f"{c1}:{t1}", "b": f"{c2}:{t2}",
                    "a_span": [round(s1[0], 1), round(s1[1], 1)],
                    "b_span": [round(s2[0], 1), round(s2[1], 1)],
                    # 真值覆盖情况：True=GT 判同人，False=GT 判异人，None=无真值
                    "gt_a": sorted(gt_label.get((c1, int(t1)), [])) or None,
                    "gt_b": sorted(gt_label.get((c2, int(t2)), [])) or None,
                })
            elif ov <= 0:
                # 谁先谁后：正的 dt 表示 [c1 结束] → [c2 开始] 的间隔
                if s1[1] <= s2[0]:
                    dt, direction = s2[0] - s1[1], f"{c1}->{c2}"
                else:
                    dt, direction = s1[0] - s2[1], f"{c2}->{c1}"
                pairs_dt.append({"pair": "-".join(pair), "dir": direction,
                                 "dt": round(dt, 1), "group": gi})

    # 每个机位对 / 方向的 Δt 分布（这是转移时间模型的训练原料）
    dist = collections.defaultdict(list)
    for r in pairs_dt:
        dist[r["dir"]].append(r["dt"])
    transfer = {}
    for d, xs in dist.items():
        xs = sorted(xs)
        n = len(xs)
        q = lambda p: round(xs[min(n - 1, int(p * n))], 1)
        transfer[d] = {"n": n, "min": xs[0], "p05": q(0.05), "p25": q(0.25),
                       "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": xs[-1]}

    # GT 与违规的交集：GT 是否已经覆盖到这些矛盾
    gt_hit = collections.Counter()
    for v in viol_a:
        la, lb = v["gt_a"], v["gt_b"]
        if la and lb:
            same = bool(set(la) & set(lb))
            gt_hit["gt_says_same" if same else "gt_says_diff"] += 1
        elif la or lb:
            gt_hit["gt_partial"] += 1
        else:
            gt_hit["gt_blind"] += 1

    res = {
        "meta": {
            "graph": str(GRAPH.relative_to(ROOT)), "n_groups": len(groups),
            "n_multi_cam": len(multi), "overlap_hard_s": OVERLAP_HARD,
            "covis_pairs": ["-".join(p) for p in COVIS],
            "osd_first_frame_rel_318": OSD, "osd_uncertain": OSD_UNCERTAIN,
        },
        "summary": {
            "n_violation_A": len(viol_a),
            "n_groups_touched": len({v["group"] for v in viol_a}),
            "gt_coverage": dict(gt_hit),
            "n_pairs_dt": len(pairs_dt),
        },
        "violations": sorted(viol_a, key=lambda x: -x["overlap_s"]),
        "transfer_time": dict(sorted(transfer.items())),
        "dt_samples": pairs_dt,
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"组 {len(groups)}，跨机位组 {len(multi)}")
    print(f"A 类违规（非共视同时在场 >{OVERLAP_HARD:.0f}s）：{len(viol_a)} 处，涉及 {len({v['group'] for v in viol_a})} 组")
    print(f"  真值覆盖：{dict(gt_hit)}")
    for v in sorted(viol_a, key=lambda x: -x["overlap_s"])[:8]:
        print(f"  {v['overlap_s']:7.1f}s {v['pair']:9s} 组{v['group']:3d} {v['a']:9s} vs {v['b']:9s}"
              f"  gt={v['gt_a']}/{v['gt_b']}")
    print("\n转移时间（Δt 秒）按方向，样本最少的略：")
    for d, s in sorted(transfer.items(), key=lambda kv: -kv[1]["n"])[:10]:
        print(f"  {d:9s} n={s['n']:4d} min={s['min']:7.1f} p05={s['p05']:7.1f} "
              f"median={s['median']:7.1f} p95={s['p95']:7.1f} max={s['max']:7.1f}")
    print(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
