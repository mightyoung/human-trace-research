"""把「时空可达性」接进全局身份图的合并决策，验证它能否替代/加强现有的可见机位数闸门。

现状（tmp/full/attr_filter_ingraph.py，attr8 = 443 个身份）的漏洞：
  闸门是 peak_cams > 2 才拒，且时间轴用的是 tracks 的 offset（各机位 -5/4/20/19/18/19 秒，
  彼此不同基准）。于是「305 与 318 同时在场」这种非共视组合被放行，
  审计（tmp/full/spacetime_audit.py）实测出 27 处、20 处是 126 条真值的盲区。

本脚本做两件事：
  1. 时间轴换成 OSD 绝对时钟（318 起录为 0），与铁律一致
  2. 加 covis 约束：任意时刻同时在场的机位集合必须 ⊆ 某个共视组（{318,320}）
     单机位恒通过；{318,320} 通过；{305,318} / {305,320} / {316,323} 一律拒

配置（都跑 attr8 的属性门，只差时空层）：
  base   —— 复现现状：offset 时间轴 + peak_cams>2
  osd    —— 只换时间轴，规则不变（看时间轴本身修正带来多少变化）
  covis  —— OSD 时间轴 + 共视组约束（本轮主张）
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
# OSD 首帧（相对 318 08:55:20 的秒偏移），来源 docs/memory-full-snapshot-2026-10-24.md L69
OSD = {"318": 0.0, "320": 1.0, "323": 2.0, "316": 16.0, "305": 25.0, "343": 121.0}
COVIS_GROUPS = [{"318", "320"}]   # 真共视；其余 14 对均不共享视野


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

    def span_offset(cam, tid):
        tr = meta[cam]["tracks"].get(str(tid)) or []
        if not tr:
            return None
        o, fps = meta[cam]["offset"], meta[cam]["fps"]
        return o + tr[0][0] / fps, o + tr[-1][0] / fps

    def span_osd(cam, tid):
        tr = meta[cam]["tracks"].get(str(tid)) or []
        if not tr:
            return None
        fps = meta[cam]["fps"]
        return OSD[cam] + tr[0][0] / fps, OSD[cam] + tr[-1][0] / fps

    C = np.zeros((n, 512), np.float32)
    for i, g in enumerate(groups):
        v = np.mean([E[t] for t in g if t in E], axis=0) if g else np.zeros(512, np.float32)
        C[i] = v / (np.linalg.norm(v) + 1e-9)
    S = np.clip(C @ C.T, -1, 1)

    def build_spans(fn):
        out = [collections.defaultdict(list) for _ in range(n)]
        for i, g in enumerate(groups):
            for t in g:
                sp = fn(*t)
                if sp:
                    out[i][t[0]].append(sp)
        return out

    spans_off = build_spans(span_offset)
    spans_osd = build_spans(span_osd)

    def conflict(si, sj):
        """同机位时间重叠 >0.4s：两条轨是同一时刻画面里的两个框，不可能是同一人。"""
        for cam in si.keys() & sj.keys():
            for a0, a1 in si[cam]:
                for b0, b1 in sj[cam]:
                    if min(a1, b1) - max(a0, b0) > 0.4:
                        return True
        return False

    def scan(ev, ov_min=None):
        """扫描时间线，返回 (同时在场峰值, 违规段落)。

        违规 = 「不可共视的同时在场组合」单段持续 > ov_min。
        为什么不是一出现就判：305/318/320 有开放空间分区，人走过交界处会被两侧同时拍到
        十几秒；真正不可能的是「同一人同时坐在两个房间」这种长时重叠。
        """
        cur, peak, viol = {}, (0, set()), []
        pts = sorted([(a, 1, c) for a, b, c in ev] + [(b, -1, c) for a, b, c in ev])
        i, last_t = 0, None
        while i < len(pts):
            t = pts[i][0]
            if last_t is not None and t > last_t:
                s = {k for k, v in cur.items() if v > 0}
                if len(s) > peak[0]:
                    peak = (len(s), s)
                if ov_min is not None and len(s) > 1 and not any(s <= g for g in COVIS_GROUPS):
                    if t - last_t > ov_min:
                        viol.append((tuple(sorted(s)), round(t - last_t, 1)))
            while i < len(pts) and pts[i][0] == t:
                _, d, c = pts[i]
                cur[c] = cur.get(c, 0) + d
                i += 1
            last_t = t
        return peak, viol

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
        need = 8                       # attr8：每次合并要求属性一致 >= 8/9
        use_osd = mode != "base"
        # covisN：违规判据 = 不可共视同时在场单段 > N 秒；covis 不带数字等价于 10
        strict_ov = None
        if mode.startswith("covis"):
            strict_ov = float(mode[5:] or 10)
        spans0 = spans_osd if use_osd else spans_off
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        cam_of = [{c for c, _ in g} for g in groups]
        sp = [collections.defaultdict(list, {k: list(v) for k, v in s.items()}) for s in spans0]
        ev = [[(a, b, c) for c, lst in s.items() for a, b in lst] for s in spans0]
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
            pk, viol = scan(nev, strict_ov)
            if viol:
                blocked["viol_" + str(int(strict_ov))] += 1
                blocked["viol_cams:" + "/".join(viol[0][0])] += 1
                continue
            if pk[0] > 2:
                blocked["peak"] += 1
                continue
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
        p_hit = sum(1 for a, b in pos if roots[a] == roots[b])
        q_hit = sum(1 for a, b in neg if roots[a] == roots[b])
        surv = [x for x in rev
                if any(roots[ra] == roots[rb] for ra in x["ta"] for rb in x["tb"])]
        judged = [x for x in surv if x["ai"] != "unsure"]
        good = sum(1 for x in judged if x["ai"] == "same")
        return dict(mode=mode, n_after=len(set(roots)), merged=merged, mem=mem,
                    blocked=dict(blocked), pos_hit=p_hit, pos_n=len(pos),
                    neg_hit=q_hit, neg_n=len(neg), surv=len(surv), n_rev=len(rev),
                    judged=len(judged), good=good, prec=good / max(len(judged), 1))

    modes = os.environ.get("MODES", "base,osd,covis").split(",")
    rows = [run(m) for m in modes]
    print(f"阈值 {THR}   共视组 {[sorted(g) for g in COVIS_GROUPS]}")
    print(f"{'方案':8s}{'身份':>6s}{'合并':>6s}{'GT同人':>9s}{'GT异人':>9s}"
          f"{'已判合并留存':>14s}{'其中对':>7s}{'精度':>8s}")
    print("-" * 76)
    for r in rows:
        print(f"{r['mode']:8s}{r['n_after']:6d}{r['merged']:6d}"
              f"{str(r['pos_hit'])+'/'+str(r['pos_n']):>9s}"
              f"{str(r['neg_hit'])+'/'+str(r['neg_n']):>9s}"
              f"{str(r['surv'])+'/'+str(r['n_rev']):>14s}{r['good']:7d}{r['prec']:8.3f}")
    print()
    for r in rows:
        print(f"  {r['mode']} 被挡：{r['blocked']}")

    dump = os.environ.get("DUMP", "")
    if dump:
        r = next(x for x in rows if x["mode"] == dump)
        fin = collections.defaultdict(list)
        for root, members in r["mem"].items():
            for m in members:
                fin[root].extend(groups[m])
        out = sorted([sorted(v) for v in fin.values()], key=lambda g: (-len(g), sorted(g)))
        fp = os.environ.get("DUMP_PATH", f"tmp/full/long35_crosscam_{dump}.json")
        json.dump(dict(_meta=dict(source="spacetime_ingraph", mode=dump, thr=THR,
                                  n_stage1=n, n_groups=len(out),
                                  time_axis="osd" if dump != "base" else "offset",
                                  covis_groups=[sorted(g) for g in COVIS_GROUPS]),
                       groups=[[[c, t] for c, t in g] for g in out]),
                  open(fp, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"\n写出 {fp}  {len(out)} 个身份")

    for x in rows:
        x.pop("mem", None)
    json.dump(dict(_meta=dict(thr=THR, n_groups=n, time_axis="osd",
                              covis_groups=[sorted(g) for g in COVIS_GROUPS]),
                   runs=rows),
              open("tmp/full/spacetime_ingraph.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
