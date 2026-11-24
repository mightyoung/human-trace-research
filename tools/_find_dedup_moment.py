# -*- coding: utf-8 -*-
"""找出去重视图真正生效的时刻（同一 gperson 在同一抽样帧被 ≥2 机位画出）。
输出：若干候选前端时间 t + 期望的 未去重/去重后 框数。
"""
import json
import collections

ROOT = "."
pay = json.load(open("webapp/frontend/public/gil_owner.json", encoding="utf-8"))
cams = pay["cams"]
tid2g = pay["tid2g"]
# gid -> [(t0,t1,cam)]
own = collections.defaultdict(list)
for gid, cam, t0, t1, reason, n in pay["owners"]:
    own[gid].append((t0, t1, cam))
for g in own:
    own[g].sort()
min_offset = -5.0   # cams.json session.min_offset

tracks = {}
for c in cams:
    d = json.load(open(f"webapp/backend/data/tracks/{c}.json", encoding="utf-8"))
    idx = collections.defaultdict(list)
    for tid, pts in d["tracks"].items():
        for p in pts:
            idx[p[0]].append(int(tid))
    tracks[c] = idx
meta = {c: json.load(open(f"webapp/backend/data/tracks/{c}.json", encoding="utf-8")) for c in cams}
caminfo = {c["id"]: c for c in json.load(open("webapp/backend/data/cams.json", encoding="utf-8"))["cameras"]}
STRIDE = 5


def owner_at(gid, tg):
    arr = own.get(gid)
    if not arr:
        return None
    hit = None
    for t0, t1, cam in arr:
        if t0 <= tg <= t1:
            hit = cam
    return hit


best = []
for t_front in range(0, 600, 2):
    tg = t_front + min_offset
    raw = ded = 0
    per = collections.Counter()
    for c in cams:
        ci = caminfo[c]
        local = t_front - (ci["offset"] - min_offset)
        if local < 0:
            continue
        fk = round(round(local * ci["fps"]) / STRIDE) * STRIDE
        tids = tracks[c].get(fk, [])
        for tid in tids:
            raw += 1
            g = tid2g.get(f"{c}:{tid}")
            oc = owner_at(g, tg) if g is not None else None
            if oc and oc != c:
                per[(g, c)] += 1
                continue
            ded += 1
    if raw - ded > 0:
        best.append((t_front, raw, ded, raw - ded, dict(per)))

best.sort(key=lambda x: -x[3])
print(f"存在同人重复画框的时刻数（t 步长 2s）：{len(best)}")
for t, raw, ded, d, per in best[:12]:
    print(f"  front t={t:4d}s  raw {raw:2d} → dedup {ded:2d}（省 {d}） 详情 {per}")
if best:
    t, raw, ded, d, per = best[0]
    print(f"\n★ 推荐验证时刻 front t={t}（同一 gperson 被两个机位同时画出）")
