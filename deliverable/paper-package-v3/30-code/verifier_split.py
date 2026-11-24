"""把「去均值余弦」当事后切分器，量三个版本：

  v0  现网：raw≥.75 + attr8 贪心合并 → 443 组
  v1  人工判读修正：切掉 15 对确认异人（最小可行分组）
  v2  v1 + 验证器：组内所有跨机位对，off < T 一律当「异人」约束再着色

验收只看三个都真实存在的口径：
  ① 身份组数
  ② 126 条人工 GT 的 P/R（不能变差）
  ③ 27 条时空真值：12 对同人要保住几对、15 对异人要切开几对

★ 阈值 T 是在这 27 条上选的 → v2 在这 27 条上的数字有选择性偏差，
  必须靠第二批扩样才能当定论。这里只作为「方向性证据 + 上界估计」。
"""
import collections
import json

import numpy as np

CAMS = ["305", "316", "318", "320", "323", "343"]
G_IN = "tmp/full/long35_crosscam_attr8.json"
TRUTH = "deliverable/spacetime_review/gt_spacetime_items.json"
GT126 = "deliverable/gt_long35/gt_long35.json"


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def load_off():
    E = {}
    for c in CAMS:
        z = np.load(f"output_crosscam/long35/embs_osnet/{c}_emb_long35_osnet.npz", allow_pickle=True)
        for k in z.files:
            E[(k.split("_")[0].lstrip("v"), int(k.split("_")[1]))] = z[k].astype(np.float32)
    U = {k: unit(v) for k, v in E.items()}
    by = collections.defaultdict(list)
    for k, v in U.items():
        by[k[0]].append(v)
    mu = {c: np.mean(np.stack(vs), 0) for c, vs in by.items()}
    return {k: unit(v - mu[k[0]]) for k, v in U.items()}


def color_split(ms, must_differ):
    parent = {m: m for m in ms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nodes = ms
    conflict = collections.defaultdict(set)
    for a, b in must_differ:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            conflict[ra].add(rb)
            conflict[rb].add(ra)
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
    blocks = collections.defaultdict(list)
    for m in ms:
        blocks[color[find(m)]].append(list(m))
    return sorted(blocks.values(), key=lambda x: -len(x))


def evaluate(groups, gt126, truth, tag):
    idx = {}
    for gi, g in enumerate(groups):
        for m in g:
            idx[(str(m[0]), str(m[1]))] = gi
    tp = fp = fn = 0
    for it in gt126:
        a = tuple(it["a"].split(":"))
        b = tuple(it["b"].split(":"))
        same = idx.get(a) is not None and idx.get(a) == idx.get(b)
        if it["label"] == 1:
            tp += same
            fn += not same
        else:
            fp += same
    kept_same = sum(1 for it in truth if it["label"] == 1
                    and idx.get(tuple(it["a"].split(":"))) == idx.get(tuple(it["b"].split(":"))))
    cut_diff = sum(1 for it in truth if it["label"] == 0
                   and idx.get(tuple(it["a"].split(":"))) != idx.get(tuple(it["b"].split(":"))))
    n_same = sum(1 for it in truth if it["label"] == 1)
    n_diff = len(truth) - n_same
    print(f"  {tag:26s} 组 {len(groups):4d} | GT126 TP {tp:2d} FP {fp:2d} "
          f"P {tp / max(1, tp + fp):.3f} R {tp / max(1, tp + fn):.3f} "
          f"| 27真值 同人保住 {kept_same}/{n_same} 异人切开 {cut_diff}/{n_diff}")
    return {"groups": len(groups), "tp": tp, "fp": fp, "fn": fn,
            "kept_same": kept_same, "cut_diff": cut_diff}


def main():
    groups = json.loads(open(G_IN, encoding="utf-8").read())["groups"]
    truth = json.loads(open(TRUTH, encoding="utf-8").read())["items"]
    gt126 = json.loads(open(GT126, encoding="utf-8").read())["items"]
    Uoff = load_off()

    norm = lambda m: (str(m[0]), str(m[1]))
    hard = {frozenset((tuple(it["a"].split(":")), tuple(it["b"].split(":"))))
            for it in truth}

    v0 = [[norm(m) for m in g] for g in groups]

    manual = {frozenset((tuple(it["a"].split(":")), tuple(it["b"].split(":"))))
              for it in truth if it["label"] == 0}
    v1 = []
    for g in v0:
        v1.extend(color_split(g, [tuple(e) for e in manual if set(e) <= set(g)]))

    print("版本对比")
    evaluate(v0, gt126, truth, "v0 现网")
    evaluate(v1, gt126, truth, "v1 人工修正")

    results = {"v0": evaluate(v0, gt126, truth, "v0")}
    for T in (0.20, 0.25, 0.30, 0.35, 0.40):
        v2 = []
        for g in v1:
            g = [tuple(x) for x in g]        # color_split 返回 list，这里统一成 tuple
            md = []
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    a, b = g[i], g[j]
                    if a[0] == b[0]:
                        continue            # 同机位不归它管
                    A = (a[0], int(a[1]))
                    B = (b[0], int(b[1]))
                    if A in Uoff and B in Uoff and float(np.dot(Uoff[A], Uoff[B])) < T:
                        md.append((a, b))
            v2.extend(color_split(g, md))
        results[f"v2@{T}"] = evaluate(v2, gt126, truth, f"v2 验证器 off≥{T}")
        if T == 0.30:
            json.dump({"_meta": {"based_on": "long35_crosscam_attr8.json",
                                 "correction": "人工 27 条 + 去均值余弦事后切分 off<0.30",
                                 "note": "阈值在 27 条上选，需扩样复验"},
                       "groups": v2},
                      open("tmp/full/long35_crosscam_attr8_v2.json", "w", encoding="utf-8"),
                      ensure_ascii=False)
    json.dump(results, open("tmp/full/verifier_split.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n→ tmp/full/verifier_split.json（T=0.30 的图为 long35_crosscam_attr8_v2.json）")


if __name__ == "__main__":
    main()
