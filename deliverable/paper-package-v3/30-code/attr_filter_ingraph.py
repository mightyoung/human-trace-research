"""属性一致度作为合并条件：放进贪心循环，与 v0 对比。

每次合并都检查两侧身份的属性一致度，只否决「属性明显不一致」的那条边，
不会像事后拆组那样连带破坏已合好的部分。

指标两套（互相独立）：
  A. 126 条人工真值：同人接通 / 异人误合
  B. 63 条已判合并：这批合并还剩多少次、精度多少
"""
import collections
import json
import os

import numpy as np

ROOT = "."
WEB = os.path.join(ROOT, "webapp/backend/data")
EMB = os.path.join(ROOT, "output_crosscam/long35/embs_osnet")
S1 = os.path.join(ROOT, "tmp/full/long35_identity.json")
CAMS = ["305", "316", "318", "320", "323", "343"]
THR = float(os.environ.get("THR", "0.75"))
FIELDS = ["gender", "top_color", "top_type", "bottom_color", "bottom_type",
          "glasses", "hat", "backpack", "carrying"]


def main():
    groups = [[(str(c), int(t)) for c, t in g]
              for g in json.load(open(S1, encoding="utf-8"))["groups"]]
    n = len(groups)
    E = {}
    for cam in CAMS:
        z = np.load(os.path.join(EMB, f"{cam}_emb_long35_osnet.npz"), allow_pickle=True)
        for k in z.files:
            E[(cam, int(k.split("_")[1]))] = z[k]
    meta = {cam: json.load(open(os.path.join(WEB, "tracks", f"{cam}.json"), encoding="utf-8"))
            for cam in CAMS}

    # ---- 属性 ----
    idx = json.load(open(os.path.join(WEB, "attr_index.json"), encoding="utf-8"))["entities"]
    ent = json.load(open(os.path.join(WEB, "entities.json"), encoding="utf-8"))
    t2e = {}
    for cam, v in ent.items():
        for ek, e in v["entities"].items():
            for t in (e.get("tracks") or []):
                t2e[(cam, str(int(t)))] = ek
    ATTR = {}
    for cam in CAMS:
        for t in {x[1] for x in E if x[0] == cam}:
            ek = t2e.get((cam, str(t)))
            if ek:
                a = (idx.get(ek) or {}).get("attrs") or {}
                if a:
                    ATTR[(cam, t)] = a

    def agreepair(a, b):
        m = tot = 0
        for f in FIELDS:
            if a.get(f) and b.get(f):
                tot += 1
                m += (a[f] == b[f])
        return m, tot

    def span(cam, tid):
        tr = meta[cam]["tracks"].get(str(tid)) or []
        if not tr:
            return None
        o, fps = meta[cam]["offset"], meta[cam]["fps"]
        return o + tr[0][0] / fps, o + tr[-1][0] / fps

    C = np.zeros((n, 512), np.float32)
    for i, g in enumerate(groups):
        v = np.mean([E[t] for t in g if t in E], axis=0) if g else np.zeros(512, np.float32)
        C[i] = v / (np.linalg.norm(v) + 1e-9)
    S = np.clip(C @ C.T, -1, 1)

    cam_of0 = [{c for c, _ in g} for g in groups]
    spans0 = [collections.defaultdict(list) for _ in range(n)]
    for i, g in enumerate(groups):
        for t in g:
            sp = span(*t)
            if sp:
                spans0[i][t[0]].append(sp)
    events0 = [[(s0, s1, cam) for cam, lst in spans0[i].items() for s0, s1 in lst]
               for i in range(n)]

    def peak_cams(ev):
        cur, best = collections.Counter(), (0, set())
        for t, d, c in sorted([(a, 1, c) for a, b, c in ev] + [(b, -1, c) for a, b, c in ev]):
            cur[c] += d
            s = {k for k, v in cur.items() if v > 0}
            if len(s) > best[0]:
                best = (len(s), s)
        return best

    def conflict(si, sj):
        for cam in si.keys() & sj.keys():
            for a0, a1 in si[cam]:
                for b0, b1 in sj[cam]:
                    if min(a1, b1) - max(a0, b0) > 0.4:
                        return True
        return False

    # ---- 真值 ----
    gt = json.load(open("deliverable/gt_long35/gt_long35.json", encoding="utf-8"))["items"]

    def k2(s):
        c, t = s.split(":")
        return (c, int(t))

    idx_of = {(c, t): i for i, g in enumerate(groups) for c, t in g}
    pos, neg = [], []
    for it in gt:
        A, B = k2(it["a"]), k2(it["b"])
        if A not in idx_of or B not in idx_of:
            continue
        (pos if it["label"] == 1 else neg).append((idx_of[A], idx_of[B]))

    # ---- 63 条已判合并 ----
    qc = {x["r"]: x for x in json.load(
        open("deliverable/gt_long35/recheck_ai_qc.json", encoding="utf-8"))["items"]}
    items = {it["id"]: it for it in json.load(
        open("tmp/full/merge_recheck/items.json", encoding="utf-8"))["items"]}
    plan = {it["id"]: it for it in json.load(
        open("tmp/full/crosscam_review/plan.json", encoding="utf-8"))["items"]}
    rev = []
    for rid, q in qc.items():
        p = plan[items[rid]["m"]]
        ta = [idx_of[k2(s)] for s in p["sideA"] if k2(s) in idx_of]
        tb = [idx_of[k2(s)] for s in p["sideB"] if k2(s) in idx_of]
        if ta and tb:
            rev.append(dict(r=rid, ai=q["ai"], score=q["score"], ta=ta, tb=tb))

    def run(mode):
        need = {"base": 0, "attr6": 6, "attr7": 7, "attr8": 8, "attr9": 9}[mode]
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        cam_of = [set(s) for s in cam_of0]
        sp = [collections.defaultdict(list, {k: list(v) for k, v in s.items()}) for s in spans0]
        ev = [list(e) for e in events0]
        mem = {i: [i] for i in range(n)}
        order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
        merged, blocked = 0, collections.Counter()
        for i, j in order:
            i, j = int(i), int(j)
            if i >= j or S[i, j] < THR:
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if cam_of[ri] & cam_of[rj]:
                blocked["cam_overlap"] += 1
                continue
            if conflict(sp[ri], sp[rj]):
                blocked["conflict"] += 1
                continue
            nev = ev[ri] + ev[rj]
            pk, _ = peak_cams(nev)
            if pk > 2:
                blocked["peak"] += 1
                continue
            if need:
                best, seen = -1, False
                for a in mem[ri]:
                    for b in mem[rj]:
                        for ta in groups[a]:
                            aa = ATTR.get(ta)
                            if not aa:
                                continue
                            for tb in groups[b]:
                                ab = ATTR.get(tb)
                                if not ab:
                                    continue
                                m, tot = agreepair(aa, ab)
                                if tot >= 7:
                                    seen = True
                                    best = max(best, m)
                if seen and best < need:
                    blocked["attr"] += 1
                    continue
            par[rj] = ri
            cam_of[ri] |= cam_of[rj]
            for cam, v in sp[rj].items():
                sp[ri][cam].extend(v)
            ev[ri] = nev
            mem[ri].extend(mem[rj])
            mem.pop(rj, None)
            merged += 1

        roots = [find(i) for i in range(n)]
        n_after = len(set(roots))
        p_hit = sum(1 for a, b in pos if roots[a] == roots[b])
        q_hit = sum(1 for a, b in neg if roots[a] == roots[b])
        # 63 条已判合并：是否仍合在一起
        surv = []
        for x in rev:
            still = any(roots[ra] == roots[rb] for ra in x["ta"] for rb in x["tb"])
            if still:
                surv.append(x)
        judged = [x for x in surv if x["ai"] != "unsure"]
        good = sum(1 for x in judged if x["ai"] == "same")
        return dict(mode=mode, n_after=n_after, merged=merged, mem=mem,
                    blocked=dict(blocked),
                    pos_hit=p_hit, pos_n=len(pos), neg_hit=q_hit, neg_n=len(neg),
                    surv=len(surv), judged=len(judged), good=good,
                    prec=good / max(len(judged), 1))

    modes = os.environ.get("MODES", "base,attr6,attr7,attr8,attr9").split(",")
    rows = [run(m) for m in modes]
    print(f"阈值 {THR}")
    hdr = (f"{'方案':8s}{'身份':>6s}{'合并':>6s}{'GT同人':>9s}{'GT异人':>9s}"
           f"{'已判合并留存':>14s}{'其中对':>7s}{'精度':>8s}")
    print(hdr)
    print("-" * 76)
    for r in rows:
        print(f"{r['mode']:8s}{r['n_after']:6d}{r['merged']:6d}"
              f"{str(r['pos_hit'])+'/'+str(r['pos_n']):>9s}"
              f"{str(r['neg_hit'])+'/'+str(r['neg_n']):>9s}"
              f"{str(r['surv'])+'/'+str(len(rev)):>14s}{r['good']:7d}{r['prec']:8.3f}")
    print()
    print("被挡原因：", {r["mode"]: r["blocked"] for r in rows})

    dump = os.environ.get("DUMP", "")
    if dump:
        r = next(x for x in rows if x["mode"] == dump)
        fin = collections.defaultdict(list)
        for root, members in r["mem"].items():
            for m in members:
                fin[root].extend(groups[m])
        out = sorted([sorted(v) for v in fin.values()], key=lambda g: (-len(g), sorted(g)))
        fp = os.environ.get("DUMP_PATH", f"tmp/full/long35_crosscam_{dump}.json")
        json.dump(dict(_meta=dict(source="attr_filter_ingraph", mode=dump, thr=THR,
                                  n_stage1=n, n_groups=len(out)),
                       groups=[[[c, t] for c, t in g] for g in out]),
                  open(fp, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"写出 {fp}  {len(out)} 个身份")

    for x in rows:
        x.pop("mem", None)
    json.dump(dict(_meta=dict(thr=THR, n_groups=n, n_review=len(rev)), runs=rows),
              open("deliverable/gt_long35/attr_filter_ingraph.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n→ deliverable/gt_long35/attr_filter_ingraph.json")


if __name__ == "__main__":
    main()
