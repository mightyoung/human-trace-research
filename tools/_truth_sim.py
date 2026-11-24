# -*- coding: utf-8 -*-
"""为 truth 事件/配对候选计算外观相似度（预填建议）。一次性辅助脚本。

实体嵌入 = 该实体成员轨迹嵌入（store_cradio_v4, 539x768 f16）的均值归一化。
输入：output_crosscam/analysis/truth/{events_all.csv, pair_candidates.csv}
输出：output_crosscam/analysis/truth/_review/{event_emb.npz, pair_machine.csv}
"""
import csv, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = os.path.join(ROOT, "output_crosscam", "analysis", "truth")
OUT = os.path.join(TRUTH, "_review")
STORE = os.path.join(ROOT, "webapp", "backend", "data", "reid")
os.makedirs(OUT, exist_ok=True)


def rd(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


meta = json.load(open(os.path.join(STORE, "store_meta.json"), encoding="utf-8"))
idx = {(t["cam"], int(t["tid"])): t["i"] for t in meta["tracks"]}
emb = np.load(os.path.join(STORE, "store_emb.npy")).astype(np.float32)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
print("store", emb.shape, "model", meta["version"]["model_version"])

# 事件 -> 实体嵌入
seen, ev_emb, miss = {}, {}, []
for r in rd(os.path.join(TRUTH, "events_all.csv")):
    eid = r["event_id"]
    if eid in seen:
        continue
    seen[eid] = r
    cam = r["cam"]
    tids = [int(x) for x in (r["eid_tracks"] or "").replace('"', "").split(",") if x.strip()]
    ii = [idx[(cam, t)] for t in tids if (cam, t) in idx]
    if not ii:
        miss.append((eid, cam, tids))
        continue
    v = emb[ii].mean(0)
    v = v / (np.linalg.norm(v) + 1e-9)
    ev_emb[eid] = v.astype(np.float16)
print("events", len(seen), "with emb", len(ev_emb), "missing", len(miss), miss[:5])

np.savez_compressed(os.path.join(OUT, "event_emb.npz"),
                    ids=np.array(list(ev_emb.keys())),
                    emb=np.stack(list(ev_emb.values())))

# 配对相似度
rows = []
for r in rd(os.path.join(TRUTH, "pair_candidates.csv")):
    a, b = r["event_a"], r["event_b"]
    cos = float(np.dot(ev_emb[a].astype(np.float32), ev_emb[b].astype(np.float32))) if (a in ev_emb and b in ev_emb) else -9
    su = "same" if cos >= 0.95 else ("diff" if cos < 0.85 else "unclear")
    rows.append(dict(pair_id=r["pair_id"], cam_a=r["cam_a"], event_a=a, t_a=r["t_a"],
                     cam_b=r["cam_b"], event_b=b, t_b=r["t_b"], dt=r["dt"],
                     in_relay_v5=r["in_relay_v5"], cos_appear=round(cos, 4), machine=su,
                     thumb_a=r["thumb_a"], thumb_b=r["thumb_b"]))
rows.sort(key=lambda x: (-x["cos_appear"], float(x["dt"])))
with open(os.path.join(OUT, "pair_machine.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

import collections
print("pairs", len(rows), collections.Counter(x["machine"] for x in rows))
print("top15 by cos:")
for x in rows[:15]:
    print("  %.3f dt=%5s relay=%s %s | %s" % (x["cos_appear"], x["dt"], x["in_relay_v5"], x["event_a"], x["event_b"]))
