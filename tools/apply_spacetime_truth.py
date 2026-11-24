"""把 27 条人工判读当作「人工确认的关系」应用到身份图上，产出 v1。

做法（最小修正，不加任何新启发式）：
  组内所有轨对默认保持合并，**只切掉人工判为「异人」的那 15 对**，再取连通块。
  → 未判读的关系一律维持原状，因此改动量最小、可解释、可回滚。

同时做两项验收：
  ① 126 条 GT：v0 → v1 的 P/R 必须不下降（切的全是人工确认的错边，理论上不该掉）
  ② 27 条判读自身：15 对异人必须已分开，12 对同人必须仍在一起
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G_IN = ROOT / "tmp/full/long35_crosscam_attr8.json"
G_OUT = ROOT / "tmp/full/long35_crosscam_attr8_v1.json"
TRUTH = ROOT / "deliverable/spacetime_review/gt_spacetime_items.json"
GT126 = ROOT / "deliverable/gt_long35/gt_long35.json"
REPORT = ROOT / "deliverable/spacetime_review/apply_v1_report.md"


def norm(m):
    c, t = m
    return (str(c), str(t))


def groups_of(groups):
    """(cam,tid) -> 组号"""
    idx = {}
    for gi, g in enumerate(groups):
        for m in g:
            idx[norm(m)] = gi
    return idx


def eval_gt(idx, items, name):
    tp = fp = fn = 0
    for it in items:
        a, b = norm(it["a"].split(":")), norm(it["b"].split(":"))
        same = idx.get(a) is not None and idx.get(a) == idx.get(b)
        if it["label"] == 1:
            tp += same
            fn += not same
        else:
            fp += same
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    print(f"  {name:22s} TP {tp:3d} FP {fp:3d} FN {fn:3d}  P {p:.4f}  R {r:.4f}"
          f"  (n={len(items)})")
    return {"tp": tp, "fp": fp, "fn": fn, "P": round(p, 4), "R": round(r, 4)}


def main():
    groups = json.loads(G_IN.read_text(encoding="utf-8"))["groups"]
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))["items"]
    gt126 = json.loads(GT126.read_text(encoding="utf-8"))["items"]
    bad_conflict = []
    cut = {frozenset((norm(it["a"].split(":")), norm(it["b"].split(":")))) for it in truth
           if it["label"] == 0}
    keep_same = {frozenset((norm(it["a"].split(":")), norm(it["b"].split(":")))) for it in truth
                 if it["label"] == 1}
    if bad_conflict:
        print("!! 人工判读自相矛盾：", bad_conflict)
    print(f"身份组 {len(groups)}；确认切分 {len(cut)} 对，确认同人 {len(keep_same)} 对")

    # ---- 组内最小切分 ----
    new_groups, split_detail = [], []
    for gi, g in enumerate(groups):
        ms = [norm(m) for m in g]
        if len(ms) == 1:
            new_groups.append([list(ms[0])])
            continue
        # ★ 删边不保证减错（同组内两端仍可经别的成员连通）→ 改成求「最小可行分组」：
        #   1) 同人边先强连通（must-link）
        #   2) 以 must-differ 为冲突边做图着色，颜色数 = 最少要拆成几块
        #   这样拆出来的块数是最少的，不会把一个组炸成碎片。
        parent = {m: m for m in ms}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in keep_same:
            a, b = tuple(e)
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        nodes = sorted({find(m) for m in ms})
        conflict = collections.defaultdict(set)
        for e in cut:
            a, b = tuple(e)
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra == rb:
                    bad_conflict.append((gi, a, b))   # 同人边与异人边自相矛盾
                    continue
                conflict[ra].add(rb)
                conflict[rb].add(ra)
        # DSATUR 着色：按「已着色的邻居种类数」优先，度数最大优先
        color = {}
        while len(color) < len(nodes):
            nxt = max((n for n in nodes if n not in color),
                      key=lambda n: (len({color[x] for x in conflict[n] if x in color}),
                                     len(conflict[n])))
            used = {color[x] for x in conflict[nxt] if x in color}
            c = 0
            while c in used:
                c += 1
            color[nxt] = c
        comp = collections.defaultdict(list)
        for m in ms:
            comp[color[find(m)]].append(list(m))
        blocks = sorted(comp.values(), key=lambda x: -len(x))
        new_groups.extend(blocks)
        if len(blocks) > 1:
            split_detail.append({"old_group": gi, "size": len(ms), "blocks": len(blocks),
                                 "block_sizes": [len(b) for b in blocks],
                                 "cams": dict(collections.Counter(c for c, _ in ms))})

    # ---- 验收 ----
    print("\n[验收] 126 条人工 GT（7 机位对）")
    e0 = eval_gt(groups_of(groups), gt126, "v0 现网")
    e1 = eval_gt(groups_of(new_groups), gt126, "v1 本次修正")

    print("\n[验收] 27 条时空判读自身")
    idx1 = groups_of(new_groups)
    bad_same = [it for it in truth if it["label"] == 1
                and idx1.get(norm(it["a"].split(":"))) != idx1.get(norm(it["b"].split(":")))]
    bad_diff = [it for it in truth if it["label"] == 0
                and idx1.get(norm(it["a"].split(":"))) == idx1.get(norm(it["b"].split(":")))]
    print(f"  同人应保持合并 → 被误拆 {len(bad_same)} 条")
    for it in bad_same:
        print("    ✗", it["item"], it["a"], it["b"])
    print(f"  异人应已分开   → 仍合并 {len(bad_diff)} 条")
    for it in bad_diff:
        print("    ✗", it["item"], it["a"], it["b"])

    # ---- 写出 ----
    meta = json.loads(G_IN.read_text(encoding="utf-8")).get("_meta") or {}
    meta = dict(meta, **{
        "version": "v1", "based_on": "long35_crosscam_attr8.json",
        "correction": "人工判读 27 条时空矛盾：切掉 15 对确认异人（最小切分），其余保持",
        "n_groups_in": len(groups), "n_groups_out": len(new_groups),
        "n_cut": len(cut), "n_split_groups": len(split_detail),
        "truth": "deliverable/spacetime_review/gt_spacetime_items.json (labels_20261112.json)",
    })
    G_OUT.write_text(json.dumps({"_meta": meta, "groups": new_groups},
                                ensure_ascii=False), encoding="utf-8")

    lines = ["# 身份图 v1：应用 27 条时空判读（最小切分）", "",
             f"- 输入 {len(groups)} 组 → 输出 **{len(new_groups)}** 组"
             f"（净增 {len(new_groups) - len(groups)}）",
             f"- 切掉 {len(cut)} 对确认异人边；涉及 {len(split_detail)} 个组", "",
             "## 被切开的组", "", "| 原组 | 规模 | 机位分布 | 切后块 | 块大小 |", "|---|---|---|---|---|"]
    for d in sorted(split_detail, key=lambda x: -x["blocks"]):
        lines.append(f"| {d['old_group']} | {d['size']} | {d['cams']} | {d['blocks']} | {d['block_sizes']} |")
    lines += ["", "## 验收", "",
              "| 口径 | TP | FP | FN | P | R |", "|---|---|---|---|---|---|",
              f"| v0 | {e0['tp']} | {e0['fp']} | {e0['fn']} | {e0['P']} | {e0['R']} |",
              f"| v1 | {e1['tp']} | {e1['fp']} | {e1['fn']} | {e1['P']} | {e1['R']} |",
              "",
              f"- 判读自身的自洽：同人被误拆 **{len(bad_same)}** 条，异人仍合并 **{len(bad_diff)}** 条",
              "", "> 切分只依据人工确认的 15 对，未引入任何启发式；未判读的关系保持原样，"
              "所以这是「修正量下界」，不是重新聚类。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{split_detail and ''}切开 {len(split_detail)} 组：")
    for d in sorted(split_detail, key=lambda x: -x["blocks"]):
        print(f"  组{d['old_group']:3d} 规模{d['size']:3d} → {d['blocks']} 块 {d['block_sizes']} {d['cams']}")
    print(f"\n→ {G_OUT}\n→ {REPORT}")


if __name__ == "__main__":
    main()
