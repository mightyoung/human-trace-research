# -*- coding: utf-8 -*-
"""长窗（35min×3 路）全局 ReID 深化实验。

数据同源事实（已逐帧验证）：长窗前 15000 帧与 10 分钟检测**逐位一致**（tid/框相同）→
POS30/NEG30 标签可直接迁移（只需检查目标轨迹在 long35-clean 中是否仍存在）。

内容：
  S1 建 store_long35（316/318/320 clean 口径，v4 W 嵌入）
  S2 检索复现与扩展：8 对 318→320 查询
       R1 全量长窗库 rank（3.5× 更大、含后 25 分钟人群）
       R2 早段限制（t_first<650s，≈10 分钟组成）一致性检查
       R3 +时间门控（通道窗）对比
  S3 同机位共现负对挖掘（同帧共现>=5 帧 = 铁定不同人）：数量扩展 + 嵌入分布压力测试
  S4 跨时段检索机制演示（早期查询 → 仅后期 gallery）

用法：python tools/long35_reid_tests.py   （前置：long35-emb/*.npz 已下载）
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_reid_store import GlobalReIDStore, build_meta_from_detections, key_to_tid  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LR = os.path.join(ROOT, "output_crosscam", "long35")
EMB_DIR = os.path.join(LR, "embs")
CLEAN_JSONL = {
    "316": os.path.join(LR, "cleaning", "316_masked_cleaned.jsonl"),
    # 2026-10-13 晚修正：318 的 18 条运动学标记经复核全为假阳性 → clean 口径即 raw
    "318": os.path.join(LR, "318_long35_detections.jsonl"),
    "320": os.path.join(LR, "cleaning", "320_cleaned.jsonl"),
}
MIN_JSONL = {  # 10 分钟对照（318 用 raw：其 4 条标记已复核判废）
    "316": os.path.join(ROOT, "output_crosscam", "cleanset", "316_detections.jsonl"),
    "318": os.path.join(ROOT, "output_crosscam", "318_detections.jsonl"),
    "320": os.path.join(ROOT, "output_crosscam", "cleanset", "320_detections.jsonl"),
}
STORE_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "global_reid", "store_long35")
DATE = "2026-10-12"
QCAM = "318"
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]
NEG30 = [(14, 1), (131, 21), (61, 14), (50, 11), (132, 21), (3, 1), (7, 1), (146, 21),
         (127, 19), (123, 19), (57, 11), (131, 23), (124, 18), (131, 24), (3, 4),
         (126, 18), (43, 11), (10, 1), (131, 28), (98, 26), (142, 26), (53, 11)]
GATE_D = {"320": 60, "316": 60}
OUT_DIR = os.path.join(ROOT, "output_crosscam", "analysis", "long35_reid")
PREFIX = {"316": "v316_", "318": "a", "320": "b"}


def load_tracks(jsonl_path):
    t = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tids = r.get("track_ids", []) or []
            for i, b in enumerate(r.get("boxes", []) or []):
                if i < len(tids) and tids[i] is not None and tids[i] >= 0:
                    t.setdefault(int(tids[i]), []).append((int(r["frame"]), b))
    for k in t:
        t[k].sort()
    return t


def norm_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = {}

    # ---------- S1 建 store_long35 ----------
    st = GlobalReIDStore(STORE_DIR, "cradio_gemma2_bridge_v4_long35", 768, mrl_dim=256)
    st.prepare_rebuild()   # v2 代际库：已有发布代时先采用基线再清空（整库重建语义）
    n = 0
    for cam in ("316", "318", "320"):
        p = os.path.join(EMB_DIR, f"{cam}_emb_long35.npz")
        assert os.path.exists(p), f"缺 {p}（先下载嵌入）"
        meta = build_meta_from_detections(cam, CLEAN_JSONL[cam], date=DATE)
        n += st.add_npz(p, cam=cam, date=DATE, meta_fn=lambda tid, m=meta: m.get(tid, {}))
    st.save()
    bpt = st.stats()["bytes_per_track"]   # v2：单份 768d fp16 落盘（MRL 仅内存派生）
    # 检索延迟（两级 auto）
    import time as _t
    ks = list(st.rows.keys())
    t0 = _t.perf_counter()
    for k in ks[: min(len(ks), 200)]:
        st.search(st.rows[k], topk=10, exclude_key=k)
    lat = (_t.perf_counter() - t0) / min(len(ks), 200) * 1000
    print(f"[S1] store_long35: {n} 轨 | {bpt:.0f} B/轨（单份 768d fp16；MRL 内存派生）| 检索 {lat:.2f} ms/次")
    res["store"] = {"n": n, "bytes_per_track": round(bpt, 1), "search_ms": round(lat, 2)}

    E, keys = st._matrix()
    meta = st.meta
    K = {k: i for i, k in enumerate(keys)}

    def rank_of(qkey, tkey, gate=None, t_first_max=None):
        """qkey 查询 -> 其他机位 gallery 按相似度排序，返回 (target_rank, top5, n_gal)"""
        q = st.rows[qkey]
        s = E @ (q / (np.linalg.norm(q) + 1e-9))
        order = np.argsort(-s)
        qm = meta[qkey]
        t0, t1 = qm["t_first"], qm["t_last"]
        gal = []
        for i in order:
            k = keys[i]
            m = meta[k]
            if m["cam"] == qm["cam"]:
                continue
            if t_first_max is not None and (m["t_first"] or 0) >= t_first_max:
                continue
            if gate is not None:
                D = gate.get(m["cam"])
                if D is None:
                    continue
                if not (m["t_last"] >= t0 - D and m["t_first"] <= t1 + D):
                    continue
            gal.append(k)
        rk = gal.index(tkey) + 1 if tkey in gal else None
        return rk, gal[:5], len(gal)

    # ---------- S2 检索复现与扩展 ----------
    rows = []
    agg = {"R1": [], "R2": [], "R3": []}
    for a, b in POS30:
        qk = f"{DATE}|{QCAM}|{a}"
        tk = f"{DATE}|320|{b}"
        if qk not in st.rows or tk not in st.rows:
            print(f"  [skip] pair ({a},{b}): 查询存在={qk in st.rows} 目标存在={tk in st.rows}")
            rows.append({"pair": [a, b], "skip": True})
            continue
        r1 = rank_of(qk, tk)                                   # 全量长窗库
        r2 = rank_of(qk, tk, t_first_max=650.0)                # 早段库（≈10 分钟组成；650s 容边界对）
        r3 = rank_of(qk, tk, gate=GATE_D)                      # 全量 + 时间门控
        rows.append({"pair": [a, b],
                     "R1": {"rank": r1[0], "n": r1[2], "top5": r1[1]},
                     "R2": {"rank": r2[0], "n": r2[2]},
                     "R3": {"rank": r3[0], "n": r3[2], "top5": r3[1]}})
        for tag, r in (("R1", r1), ("R2", r2), ("R3", r3)):
            agg[tag].append(r[0])
        print(f"[S2] pair({a:>3},{b:>2})  R1全量 rank={r1[0]}(n={r1[2]})   "
              f"R2早段 rank={r2[0]}(n={r2[2]})   R3门控 rank={r3[0]}(n={r3[2]})")
    for tag in ("R1", "R2", "R3"):
        rs = [x for x in agg[tag] if x]
        print(f"[S2] {tag}: rank1 {sum(1 for x in rs if x == 1)}/{len(rs)} | avg {np.mean(rs):.2f}")
    res["retrieval"] = {"rows": rows,
                        "summary": {tag: {"ranks": agg[tag]} for tag in ("R1", "R2", "R3")}}

    # ---------- S3 同机位共现负对挖掘 ----------
    def cooc_negatives(cam, jsonl, embs, min_common=5, iou_max=0.30):
        """同机位共检负对：两轨同帧共检>=5 帧，且框不重叠（剔除换轨抖动重复）。"""
        tracks = load_tracks(jsonl)
        fmap = {}
        for tid, lst in tracks.items():
            if len(lst) < 2:
                continue
            for f, _ in lst:
                fmap.setdefault(f, []).append(tid)
        cnt = {}
        for f, tids in fmap.items():
            if len(tids) < 2:
                continue
            tids = sorted(set(tids))
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    key = (tids[i], tids[j])
                    cnt[key] = cnt.get(key, 0) + 1

        def iou(b1, b2):
            x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
            iw, ih = max(0, x2 - x1), max(0, y2 - y1)
            inter = iw * ih
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / (a1 + a2 - inter + 1e-9)

        pairs = []
        for (a, b), n in cnt.items():
            if n < min_common or a not in embs or b not in embs:
                continue
            ba = dict(tracks[a]); bb = dict(tracks[b])
            common = sorted(set(ba) & set(bb))
            sel = common[:12] if len(common) <= 12 else \
                [common[int(k)] for k in np.linspace(0, len(common) - 1, 12)]
            med_iou = float(np.median([iou(ba[f], bb[f]) for f in sel])) if sel else 0.0
            if med_iou > iou_max:      # 框重叠 = 同一物理目标的换轨/抖动重复
                continue
            pairs.append((a, b))
        sims = [norm_sim(embs[p[0]], embs[p[1]]) for p in pairs]
        return pairs, sims

    neg_res = {}
    for cam in ("316", "318", "320"):
        # long35
        z = np.load(os.path.join(EMB_DIR, f"{cam}_emb_long35.npz"))
        embs_l = {key_to_tid(k): z[k] for k in z.files}
        pairs_l, sims_l = cooc_negatives(cam, CLEAN_JSONL[cam], embs_l)
        # 10min 对照（v4 clean 嵌入）
        z2 = np.load(os.path.join(ROOT, "output_crosscam", "analysis",
                                  "cradio_scale_v4", f"{cam}_emb_scale.npz"))
        embs_m = {key_to_tid(k): z2[k] for k in z2.files}
        pairs_m, sims_m = cooc_negatives(cam, MIN_JSONL[cam], embs_m)
        neg_res[cam] = {
            "long35": {"n_pairs": len(pairs_l),
                       "sim_mean": round(float(np.mean(sims_l)), 4) if sims_l else None,
                       "sim_p95": round(float(np.percentile(sims_l, 95)), 4) if sims_l else None,
                       "sim_max": round(float(np.max(sims_l)), 4) if sims_l else None,
                       "n_ge090": int(sum(1 for s in sims_l if s >= 0.90)),
                       "n_ge093": int(sum(1 for s in sims_l if s >= 0.93))},
            "min10": {"n_pairs": len(pairs_m),
                      "sim_mean": round(float(np.mean(sims_m)), 4) if sims_m else None,
                      "sim_p95": round(float(np.percentile(sims_m, 95)), 4) if sims_m else None,
                      "sim_max": round(float(np.max(sims_m)), 4) if sims_m else None},
        }
        print(f"[S3] {cam}: long35 负对 {len(pairs_l)} 对 (sim max={neg_res[cam]['long35']['sim_max']}"
              f", >=0.9: {neg_res[cam]['long35']['n_ge090']}) | 10min 负对 {len(pairs_m)} 对")
    tot_l = sum(v["long35"]["n_pairs"] for v in neg_res.values())
    tot_m = sum(v["min10"]["n_pairs"] for v in neg_res.values())
    print(f"[S3] 三路合计: 10min {tot_m} 对 → long35 {tot_l} 对（{tot_l/max(1,tot_m):.1f}×）")
    res["negatives"] = {"per_cam": neg_res, "total_long35": tot_l, "total_min10": tot_m}

    # ---------- S4 跨时段检索机制演示 ----------
    demo = []
    demos = [("318", 1), ("320", 11), ("318", 120)]
    for cam, tid in demos:
        qk = f"{DATE}|{cam}|{tid}"
        if qk not in st.rows:
            demo.append({"q": [cam, tid], "missing": True})
            continue
        q = st.rows[qk]
        s = E @ (q / (np.linalg.norm(q) + 1e-9))
        order = np.argsort(-s)
        qm = meta[qk]
        t_end = qm["t_last"]
        hits = []
        for i in order:
            k = keys[i]
            m = meta[k]
            if k == qk or m["cam"] != cam:
                continue
            if (m["t_first"] or 0) >= t_end + 120:   # 只留"120 秒后"的 gallery
                hits.append({"key": k, "score": round(float(s[i]), 4),
                             "t_first": round(m["t_first"], 1)})
            if len(hits) >= 5:
                break
        demo.append({"q": [cam, tid], "q_time": [round(qm["t_first"], 1), round(t_end, 1)],
                     "hits_after": hits})
        print(f"[S4] 查询 {cam}#{tid} [t={qm['t_first']:.0f}~{t_end:.0f}s] → 120s 后 top: " +
              ", ".join(f"{h['key'].split('|')[2]}@{h['t_first']:.0f}s({h['score']:.3f})"
                        for h in hits[:3]))
    res["cross_time_demo"] = demo

    out = os.path.join(OUT_DIR, "results.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
