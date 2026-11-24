"""碎片合并的可分性：同身份不同簇 vs 不同身份（同机位、时间互斥）的簇原型余弦。

如果两组的余弦分得开，就说明「时间互斥 + 外观阈值」的合并规则可行；
如果重叠严重，就不该合并——那会把不同人并成一个，直接损坏计数与识别。

用法：PY tools/diag_merge_separability.py [--run results_gated] [--cam 305]
输出：results/merge_separability.json
"""
import argparse
import collections
import itertools
import json
import os

import numpy as np

import eval_query_fusion as F
import m3_counting as C

EMB_REL = "output_crosscam_local/1012_yolo8_dd_emb"


def cos(a, b):
    return float(np.dot(a, b))


def auc(pos, neg):
    """Mann-Whitney AUC：pos 的余弦高于 neg 的概率。"""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, s = {}, 0, 0.0
    # 用秩和法算 AUC
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rp = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return float((rp - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results_gated")
    ap.add_argument("--deliv", default="")
    ap.add_argument("--cfg", default="A_baseline")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    root = os.path.dirname(os.path.dirname(deliv))
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)
    emb_dir = os.path.join(root, EMB_REL)
    det_root = os.path.join(root, "output_crosscam_local/1012_yolo8_dd")

    gtruth = F.load_gallery_truth(gt)
    mem = collections.defaultdict(list)
    for r in F.rd(os.path.join(run, "members", a.cfg + ".csv")):
        mem[int(r["cluster_local"])].append(r["track_id"])

    # 载入逐轨迹嵌入（a/b 两半各一个，取均值再归一）
    per_track = {}
    for fn in os.listdir(emb_dir):
        if not fn.endswith("_emb_a.npz"):
            continue
        seg = fn[:-len("_emb_a.npz")]
        za = np.load(os.path.join(emb_dir, fn), allow_pickle=True)
        pb = os.path.join(emb_dir, seg + "_emb_b.npz")
        zb = np.load(pb, allow_pickle=True) if os.path.exists(pb) else {}
        for k in za.keys():
            v = np.asarray(za[k], dtype=np.float32)
            if k in zb:
                v = v / (np.linalg.norm(v) + 1e-9) + np.asarray(zb[k], dtype=np.float32) / (
                    np.linalg.norm(zb[k]) + 1e-9)
            v = v / (np.linalg.norm(v) + 1e-9)
            per_track[k] = v

    # 簇原型 + 时间跨度
    proto, span, ident, cam = {}, {}, {}, {}
    for cl, ts in mem.items():
        vs = [per_track[t] for t in ts if t in per_track]
        if not vs:
            continue
        p = np.mean(vs, axis=0)
        proto[cl] = p / (np.linalg.norm(p) + 1e-9)
        votes = collections.Counter(gtruth[t] for t in ts if gtruth.get(t))
        ident[cl] = votes.most_common(1)[0][0] if votes else ""
        cam[cl] = ts[0].split("_")[0]
        # 时间跨度：同段内取该段轨迹帧范围
        lo, hi = None, None
        for t in ts:
            seg, tid = t.split("#")
            sp = _spans(det_root, seg).get(int(tid))
            if not sp:
                continue
            lo = sp[0] if lo is None else min(lo, sp[0])
            hi = sp[1] if hi is None else max(hi, sp[1])
        span[cl] = (lo, hi)

    pairs = []
    for c1, c2 in itertools.combinations(sorted(proto), 2):
        if cam[c1] != cam[c2]:
            continue
        s = cos(proto[c1], proto[c2])
        a1, b1 = span[c1]
        a2, b2 = span[c2]
        disj = (a1 is None or b1 is None or a2 is None or b2 is None or b1 < a2 or b2 < a1)
        i1, i2 = ident[c1], ident[c2]
        if i1 and i2 and i1 == i2:
            kind = "same"
        elif i1 and i2:
            kind = "diff_disjoint" if disj else "diff_overlap"
        else:
            kind = "unknown"
        pairs.append(dict(c1=c1, c2=c2, cam=cam[c1], cos=round(s, 4), kind=kind,
                          disjoint=int(bool(disj))))

    out = {}
    print("%-14s %-5s %-8s %-8s %-8s %-8s" % ("组", "n", "min", "中位", "p90", "max"))
    for kind in ("same", "diff_disjoint", "diff_overlap"):
        v = sorted(p["cos"] for p in pairs if p["kind"] == kind)
        if not v:
            continue
        print("%-14s %-5d %-8.3f %-8.3f %-8.3f %-8.3f"
              % (kind, len(v), v[0], v[len(v) // 2], v[int(0.9 * (len(v) - 1))], v[-1]))
        out[kind] = dict(n=len(v), min=v[0], median=v[len(v) // 2],
                         p90=v[int(0.9 * (len(v) - 1))], max=v[-1])

    pos = [p["cos"] for p in pairs if p["kind"] == "same"]
    neg = [p["cos"] for p in pairs if p["kind"] in ("diff_disjoint", "diff_overlap")]
    print("\n同身份 vs 不同身份（同机位）：AUC = %s" % (round(auc(pos, neg), 3) if pos and neg else "NA"))
    print("\n阈值扫描（合并 = 余弦 ≥ τ）：")
    print("%-6s %-12s %-14s %-10s" % ("τ", "合并同身份对", "误并不同身份对", "精确率"))
    for tau in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        tp = sum(1 for v in pos if v >= tau)
        fp = sum(1 for v in neg if v >= tau)
        pr = tp / (tp + fp) if tp + fp else 0.0
        print("%-6.2f %-12s %-14s %-10.3f" % (tau, "%d/%d" % (tp, len(pos)),
                                             "%d/%d" % (fp, len(neg)), pr))

    # 分机位
    print("\n分机位：同身份中位 / 不同身份中位 / AUC")
    for c in sorted({p["cam"] for p in pairs}):
        p1 = [p["cos"] for p in pairs if p["cam"] == c and p["kind"] == "same"]
        n1 = [p["cos"] for p in pairs if p["cam"] == c and p["kind"].startswith("diff")]
        if not p1 or not n1:
            continue
        p1s, n1s = sorted(p1), sorted(n1)
        print("  %-4s %.3f(n=%d) / %.3f(n=%d) / AUC %.3f"
              % (c, p1s[len(p1s) // 2], len(p1), n1s[len(n1s) // 2], len(n1), auc(p1, n1)))

    json.dump(dict(cfg=a.cfg, summary=out, auc=auc(pos, neg) if pos and neg else None,
                   pairs=pairs),
              open(os.path.join(run, "merge_separability.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n→ results/merge_separability.json")


_SPAN_CACHE = {}


def _spans(det_root, seg):
    if seg in _SPAN_CACHE:
        return _SPAN_CACHE[seg]
    _SPAN_CACHE[seg] = C.load_spans(det_root, seg)
    return _SPAN_CACHE[seg]


if __name__ == "__main__":
    main()
