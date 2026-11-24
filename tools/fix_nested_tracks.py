"""同帧「同一人被检成多条轨迹」的嵌套合并，并量化它对计数的修复。

背景（实测）：305 的同身份同窗重复对 77 对，其中 52 对**时间重叠**、且较短那条的
帧级时间重叠中位数 **1.00**（完全被包含）——典型的「头框 + 半身框各自成一条轨迹」。
这类重复在计数上体现为「同一人多算一次」，而它们**时间上重叠**，所以聚类后合并
（时间互斥前提）根本管不到，必须在轨迹层先合。

规则（只用几何，不用身份真值）：
  两条轨迹 A、B，若
    ① 共同出现的帧数 ≥ 较短轨迹帧数的 50%，且
    ② 在这些共同帧里，A 的框有 ≥80% 面积落在 B 的框内（或反之）的帧占比 ≥ 80%
  则判为同一个人，合并。

输出：results/nested_merge.json（合并对、按机位统计）、
      合并后再算一遍计数（把被合并轨迹所属簇用并查集合并）。

用法：PY tools/fix_nested_tracks.py [--seg 305_test_544] [--window 60]
"""
import argparse
import collections
import json
import os

import numpy as np

import eval_query_fusion as F
import m3_counting as C

DET_REL = "output_crosscam_local/1012_yolo8_dd"
EMB_REL = "output_crosscam_local/1012_yolo8_dd_emb"
FPS = 25.0


def load_frames(det_root, seg):
    """frame -> {tid: box}"""
    out = {}
    for line in open(os.path.join(det_root, seg, "%s_detections.jsonl" % seg),
                     encoding="utf-8"):
        r = json.loads(line)
        out[r["frame"]] = {int(t): b for t, b in zip(r["track_ids"], r["boxes"])}
    return out


def load_emb(emb_dir, seg):
    """轨迹级嵌入（a/b 两半归一后相加再归一）；键为 seg#tid。"""
    za = np.load(os.path.join(emb_dir, seg + "_emb_a.npz"), allow_pickle=True)
    pb = os.path.join(emb_dir, seg + "_emb_b.npz")
    zb = np.load(pb, allow_pickle=True) if os.path.exists(pb) else {}
    out = {}
    for k in za.keys():
        v = np.asarray(za[k], np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        if k in zb:
            u = np.asarray(zb[k], np.float32)
            v = v + u / (np.linalg.norm(u) + 1e-9)
        out[k] = v / (np.linalg.norm(v) + 1e-9)
    return out


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def merge_pairs(frames, min_shared=0.5, min_contain=0.8):
    per = collections.defaultdict(dict)
    for f, m in frames.items():
        for t, b in m.items():
            per[t][f] = b
    tids = sorted(per)
    pairs = []
    for i, t1 in enumerate(tids):
        for t2 in tids[i + 1:]:
            f1, f2 = per[t1], per[t2]
            common = set(f1) & set(f2)
            if not common:
                continue
            shorter = min(len(f1), len(f2))
            if len(common) < min_shared * shorter:
                continue
            ok = 0
            for f in common:
                a, b = f1[f], f2[f]
                aa, ab = area(a), area(b)
                if aa <= 0 or ab <= 0:
                    continue
                r = inter(a, b) / min(aa, ab)   # 小框被大框包含的比例
                if r >= min_contain - 1e-9:
                    ok += 1
            if ok >= min_contain * len(common):
                pairs.append((t1, t2, len(common), shorter))
    return pairs, per


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
            return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deliv", default="")
    ap.add_argument("--run", default="results_gated")
    ap.add_argument("--cfg", default="A_baseline")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--appear", type=float, default=0.0,
                    help="外观余弦下限（0=只用几何）；建议 0.75（实测合并精度 93%%）")
    a = ap.parse_args()
    deliv = os.path.abspath(a.deliv) if a.deliv else F.DELIV
    root = os.path.dirname(os.path.dirname(deliv))
    det_root = os.path.join(root, DET_REL)
    emb_dir = os.path.join(root, EMB_REL)
    gt = os.path.join(deliv, "gt")
    run = os.path.join(deliv, a.run)

    gtruth = F.load_gallery_truth(gt)
    ident = F.build_ident(run, a.cfg, gtruth)
    qtruth = F.load_query_truth(gt)
    sysrows = {r["query_id"]: r for r in F.rd(os.path.join(run, "queries", a.cfg + ".csv"))}
    qrows = F.rd(os.path.join(gt, "gt_query_gt_FILLED.csv"))
    segs = sorted({r["seg"] for r in qrows})

    report = {}
    for seg in segs:
        frames = load_frames(det_root, seg)
        pairs, per = merge_pairs(frames)
        if a.appear > 0:
            E = load_emb(emb_dir, seg)
            keep = []
            for t1, t2, nc, sh in pairs:
                k1, k2 = "%s#%d" % (seg, t1), "%s#%d" % (seg, t2)
                if k1 in E and k2 in E and float(E[k1] @ E[k2]) >= a.appear:
                    keep.append((t1, t2, nc, sh))
            pairs = keep
        dsu = DSU()
        merged = 0
        for t1, t2, nc, sh in pairs:
            if dsu.union(t1, t2):
                merged += 1
        # 簇层面：被合并的轨迹若落在不同簇，把这些簇也并起来（仅计数口径）
        cl_union = DSU()
        n_cl_merge = 0
        for t in per:
            qid = "%s#%d" % (seg, t)
            s = sysrows.get(qid, {})
            if s.get("final_status") != "accepted":
                continue
            cl = int(s["final_cluster_local"])
            cl_union.find(cl)
        for t1, t2, _, _ in pairs:
            for t in (t1, t2):
                qid = "%s#%d" % (seg, t)
                s = sysrows.get(qid, {})
                if s.get("final_status") == "accepted":
                    cl_union.find(int(s["final_cluster_local"]))
        # 用并查集把簇号映射成代表簇
        cl2rep = {}
        for t1, t2, _, _ in pairs:
            q1, q2 = "%s#%d" % (seg, t1), "%s#%d" % (seg, t2)
            s1, s2 = sysrows.get(q1, {}), sysrows.get(q2, {})
            if s1.get("final_status") == "accepted" and s2.get("final_status") == "accepted":
                c1, c2 = int(s1["final_cluster_local"]), int(s2["final_cluster_local"])
                if c1 != c2 and cl_union.union(c1, c2):
                    n_cl_merge += 1
        for cl in list(cl_union.p):
            cl2rep[cl] = cl_union.find(cl)

        # 计数前后
        span = C.load_spans(det_root, seg)
        W = a.window * FPS
        lo = min(v[0] for v in span.values())
        hi = max(v[1] for v in span.values())
        nw = int((hi - lo) // W) + 1
        before = after = gtsum = 0
        mae_b = mae_a = 0.0
        for wi in range(nw):
            w0, w1 = lo + wi * W, lo + (wi + 1) * W
            present = []
            for r in qrows:
                if r["seg"] != seg:
                    continue
                tid = int(r["query_id"].split("#")[1])
                sp = span.get(tid)
                if not sp or sp[1] < w0 or sp[0] > w1:
                    continue
                pid = (r["person_id"] or "").strip()
                if C.is_nonperson(pid):
                    continue
                s = sysrows.get(r["query_id"], {})
                ok = s.get("final_status") == "accepted"
                present.append((tid, pid, ok, int(s["final_cluster_local"]) if ok else None))
            gt_keys = {(pid if pid else "unk:%d" % tid) for tid, pid, _, _ in present}
            cl_b = {cl for _, _, ok, cl in present if ok}
            cl_a = {cl2rep.get(cl, cl) for cl in cl_b}
            before += len(cl_b)
            after += len(cl_a)
            gtsum += len(gt_keys)
            mae_b += abs(len(cl_b) - len(gt_keys))
            mae_a += abs(len(cl_a) - len(gt_keys))
        n = max(1, nw)
        report[seg] = dict(pairs=len(pairs), tracks_merged=merged,
                           cluster_merges=n_cl_merge, windows=nw,
                           gt=gtsum / n, before=before / n, after=after / n,
                           mae_before=mae_b / n, mae_after=mae_a / n)
        print("%-16s 合并对 %2d（并簇 %2d）｜GT %.2f｜按簇 %.2f→%.2f｜MAE %.2f→%.2f"
              % (seg, len(pairs), n_cl_merge, gtsum / n, before / n, after / n,
                 mae_b / n, mae_a / n))

    tot = collections.Counter()
    for seg, r in report.items():
        for k in ("mae_before", "mae_after"):
            tot[k] += r[k]
    n = len(report)
    print("\n四段平均 MAE：%.2f → %.2f" % (tot["mae_before"] / n, tot["mae_after"] / n))
    json.dump(dict(appear=a.appear, report=report),
              open(os.path.join(run, "nested_merge.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ results/nested_merge.json")


if __name__ == "__main__":
    main()
