# -*- coding: utf-8 -*-
"""MRL 两级检索验证：单级（768d 暴力）vs 两级（256d 粗筛 + 768d 精排）。

验证三件事：
  1) 一致性：真实库（727 轨）上，两级检索 top-10 与单级是否一致（8 个查询）
  2) 规模延迟：复制放大到 ~10 万轨，对比单级/两级检索延迟
  3) 粗筛召回：单级 top-10 是否落在两级粗筛候选内（nprobe 敏感性）

用法：
  python tools/mrl_retrieval_check.py            # 重建 store + 全部验证
  python tools/mrl_retrieval_check.py --no-build  # 跳过重建（复用已有 store）
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from global_reid_store import GlobalReIDStore, build_meta_from_detections  # noqa: E402

CAMS = ["318", "320", "305", "316", "323", "343"]
DATE = "2026-10-12"
POS30 = [(1, 1), (14, 3), (69, 14), (120, 18), (120, 23), (139, 24), (146, 28), (37, 11)]


def key_to_tid(k):
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def build_store(store_dir):
    st = GlobalReIDStore(store_dir, "cradio_gemma2_bridge_v3", 768, mrl_dim=256)
    st.prepare_rebuild()   # v2 代际库：已有发布代时先采用基线再清空（整库重建语义）
    n = 0
    for cam in CAMS:
        p = os.path.join(ROOT, "output_crosscam", "analysis", "cradio_scale_v3",
                         f"{cam}_emb_scale.npz")
        meta = build_meta_from_detections(
            cam, os.path.join(ROOT, "output_crosscam", f"{cam}_detections.jsonl"))
        n += st.add_npz(p, cam=cam, date=DATE, meta_fn=lambda tid: meta.get(tid, {}))
    st.save()
    return st, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=os.path.join(
        ROOT, "output_crosscam", "analysis", "global_reid", "store_cradio_v3"))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "output_crosscam", "analysis", "mrl", "mrl_retrieval_check.json"))
    args = ap.parse_args()

    t0 = time.time()
    if args.no_build:
        st = GlobalReIDStore(args.store, "cradio_gemma2_bridge_v3", 768, mrl_dim=256)
        n = st.load()
    else:
        st, n = build_store(args.store)
    s = st.stats()
    print(f"[store] {n} 轨重建 {time.time()-t0:.1f}s | emb={s['bytes_emb']/1e6:.2f}MB "
          f"mrl={s['bytes_mrl']/1e6:.2f}MB | {s['bytes_per_track']:.0f} B/轨"
          f"（v2：MRL 仅内存派生，不落盘）")

    res = {"n": n, "bytes_per_track": s["bytes_per_track"],
           "bytes_emb": s["bytes_emb"], "bytes_mrl": s["bytes_mrl"]}

    # ---------- 1) 一致性（真实库） ----------
    queries = []
    for a, b in POS30:
        qk = f"{DATE}|318|{a}"
        tk = f"{DATE}|320|{b}"
        assert qk in st.rows, qk
        queries.append((qk, tk))
    # 再加 4 个额外查询（其他机位，防过拟合 318）
    extra = [(f"{DATE}|316|144", f"{DATE}|343|57"),   # 白帽链 316→343
             (f"{DATE}|343|57", f"{DATE}|316|144")]
    for qk, tk in extra:
        if qk in st.rows and tk in st.rows:
            queries.append((qk, tk))

    consistency = []
    hit_single = hit_two = 0
    for qk, tk in queries:
        q = st.rows[qk]
        r1 = st.search(q, topk=10, stage="single", exclude_key=qk)
        r2 = st.search(q, topk=10, stage="two", nprobe=64, exclude_key=qk)
        c1 = [h["key"] for h in r1]
        c2 = [h["key"] for h in r2]
        same_order = c1 == c2
        same_set = set(c1) == set(c2)
        rk1 = next((i + 1 for i, k in enumerate(c1) if k == tk), None)
        rk2 = next((i + 1 for i, k in enumerate(c2) if k == tk), None)
        hit_single += (rk1 == 1)
        hit_two += (rk2 == 1)
        consistency.append({"q": qk, "target": tk, "same_order": same_order,
                            "same_set": same_set, "rank_single": rk1, "rank_two": rk2})
        if not same_set:
            print(f"  [diff] {qk}: single {c1[:3]} vs two {c2[:3]}")
    print(f"[一致性] {len(queries)} 查询: top-10 同序 {sum(c['same_order'] for c in consistency)}/"
          f"{len(queries)}，同集合 {sum(c['same_set'] for c in consistency)}/{len(queries)}；"
          f" rank-1 单级 {hit_single} vs 两级 {hit_two}")
    res["consistency"] = consistency

    # nprobe 敏感性（放宽/收紧）
    sens = {}
    for npb in [8, 16, 32, 64, 128, 256]:
        ok = 0
        for qk, tk in queries:
            r2 = st.search(st.rows[qk], topk=10, stage="two", nprobe=npb, exclude_key=qk)
            c2 = [h["key"] for h in r2]
            r1 = st.search(st.rows[qk], topk=10, stage="single", exclude_key=qk)
            if c2 == [h["key"] for h in r1]:
                ok += 1
        sens[npb] = ok
    print(f"[nprobe 敏感性] 与单级同序的查询数: {json.dumps(sens)}")
    res["nprobe_sensitivity"] = {str(k): v for k, v in sens.items()}

    # ---------- 2) 规模延迟（复制放大） ----------
    E = st._matrix()[0]
    K = 137   # 727 * 137 ≈ 99,600 轨
    rng = np.random.default_rng(0)
    E_big = np.tile(E, (K, 1))
    E_big += rng.normal(0, 1e-4, E_big.shape).astype(np.float32)   # 打破完全重复
    E_big /= (np.linalg.norm(E_big, axis=1, keepdims=True) + 1e-9)
    M_big = E_big[:, :st.mrl_dim]
    M_big /= (np.linalg.norm(M_big, axis=1, keepdims=True) + 1e-9)
    N, D = E_big.shape
    print(f"[规模测试] {N} 轨 × {D}d（复制 {K}× + 噪声）")

    qs = [st.rows[qk] for qk, _ in queries]
    for qkq in qs:
        q = qkq / (np.linalg.norm(qkq) + 1e-9)
        _ = E_big @ q; _ = M_big @ q[:st.mrl_dim]          # warmup

    def bench(fn, nrep=30):
        fn()
        t0 = time.perf_counter()
        for _ in range(nrep):
            fn()
        return (time.perf_counter() - t0) / nrep * 1000

    q = qs[0] / (np.linalg.norm(qs[0]) + 1e-9)
    qm = q[:st.mrl_dim] / (np.linalg.norm(q[:st.mrl_dim]) + 1e-9)
    t_single = bench(lambda: np.argsort(-(E_big @ q))[:10])
    t_coarse = bench(lambda: np.argpartition(-(M_big @ qm), 63)[:64])
    cand = np.argpartition(-(M_big @ qm), 63)[:64]
    t_refine = bench(lambda: E_big[cand] @ q)
    lat = {"single_768d_ms": round(t_single, 3),
           "coarse_256d_ms": round(t_coarse, 3),
           "refine64_rows_ms": round(t_refine, 4),
           "two_stage_ms": round(t_coarse + t_refine, 3)}
    print(f"[延迟 {N} 轨] 单级 {lat['single_768d_ms']}ms | 粗筛 {lat['coarse_256d_ms']}ms + "
          f"精排64行 {lat['refine64_rows_ms']}ms = 两级 {lat['two_stage_ms']}ms")
    res["scale_test"] = {"N": N, "K": K, "latency": lat}

    # ---------- 3) 小库延迟（727 轨，真实使用口） ----------
    def bench_small(fn, nrep=200):
        fn()
        t0 = time.perf_counter()
        for _ in range(nrep):
            fn()
        return (time.perf_counter() - t0) / nrep * 1000

    q0 = st.rows[queries[0][0]]
    t0 = bench_small(lambda: st.search(q0, topk=10, stage="single"))
    t2 = bench_small(lambda: st.search(q0, topk=10, stage="two"))
    print(f"[延迟 727 轨] 单级 {t0:.3f}ms | 两级 {t2:.3f}ms")
    res["small_lib_latency_ms"] = {"single": round(t0, 3), "two": round(t2, 3)}

    # ---------- 4) 大库目标召回（真实语义：真实库 + 10 万随机干扰） ----------
    rng = np.random.default_rng(1)
    Noise = rng.normal(0, 1, (100000, 768)).astype(np.float32)
    Noise /= np.linalg.norm(Noise, axis=1, keepdims=True)
    E_mix = np.vstack([E, Noise])
    M_mix = E_mix[:, :st.mrl_dim]
    keys_list = st._matrix()[1]

    def target_rank(q, ti, nprobe=None):
        if nprobe is None:
            idx = np.argsort(-(E_mix @ q))
        else:
            qm = q[:st.mrl_dim]
            qm = qm / (np.linalg.norm(qm) + 1e-9)
            cand = np.argpartition(-(M_mix @ qm), nprobe - 1)[:nprobe]
            sf = E_mix[cand] @ q
            idx = cand[np.argsort(-sf)]
        hit = np.where(idx == ti)[0]
        return int(hit[0]) + 1 if len(hit) else None

    rk_s, rk_t = [], []
    for qk, tk in queries[:8]:
        q = st.rows[qk]
        q = q / (np.linalg.norm(q) + 1e-9)
        ti = keys_list.index(tk)
        rk_s.append(target_rank(q, ti, None))
        rk_t.append(target_rank(q, ti, 64))
    same = sum(1 for a, b in zip(rk_s, rk_t) if a == b)
    print(f"[大库目标召回] 真实库+10万干扰: 单级 vs 两级(64) rank 一致 {same}/{len(rk_s)}"
          f"（ranks {rk_s}）")
    res["mixed_gallery_rank"] = {"single": rk_s, "two_64": rk_t, "match": same}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()