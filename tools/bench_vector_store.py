# -*- coding: utf-8 -*-
"""向量存储选型基准 v2（GIL 存储层决策依据）。

测四件事，全在远程 48 核 CPU 上：
  A. 扩展性：numpy 精确扫描 vs sqlite-vec（539 / 10万 / 100万 × 768 维）
  B. 过滤形态：预过滤（掩码）vs post-filter，强选择率（时间窗 ~3%）
  C. dtype：float32 vs float16 存储（内存减半、速度代价）
  D. 写入模式：sqlite-vec 增量 INSERT vs numpy 整表 np.save 往返
另测 memmap 冷读成本（不常驻内存时的查询）。

用法（服务器，cd ~/human-count）：
  python _bench_vec_v2.py --real webapp/backend/data/reid/store_emb.npy
"""
import argparse
import json
import os
import sqlite3
import time

import numpy as np


def norm(E):
    E = np.asarray(E, dtype=np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def synth(n, d=768, seed=0):
    rng = np.random.RandomState(seed)
    return norm(rng.randn(n, d).astype(np.float32))


def pct(t, p=50):
    t = sorted(t)
    return t[min(len(t) - 1, int(len(t) * p / 100))]


def bench_query(E, Q, mask=None, topk=10):
    """预过滤形态：候选矩阵已构建好（掩码成本另行计）。"""
    sub = E if mask is None else E[mask]
    n = sub.shape[0]
    tt = []
    for q in Q:
        t0 = time.perf_counter()
        s = sub @ q
        k = min(topk, n)
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        tt.append((time.perf_counter() - t0) * 1000)
    return pct(tt, 50), pct(tt, 95), n


def bench_postfilter(E, Q, keep, topk=10):
    """post-filter：全量打分后按谓词筛（需 over-fetch，可能不够 k）。"""
    tt = []
    for q in Q:
        t0 = time.perf_counter()
        s = E @ q
        order = np.argsort(-s)
        out = 0
        for i in order:
            if keep[i]:
                out += 1
                if out >= topk:
                    break
        tt.append((time.perf_counter() - t0) * 1000)
    return pct(tt, 50), pct(tt, 95)


def sqlite_vec_run(E, cams, t_first, Q, reps=100, db="/tmp/_bv.sqlite", label=""):
    try:
        import sqlite_vec
    except Exception as e:  # noqa: BLE001
        return {"error": f"sqlite_vec 不可用: {e}"}
    for suf in ("", "-journal", "-wal", "-shm"):
        p = db + suf
        if os.path.exists(p):
            os.remove(p)
    con = sqlite3.connect(db)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    d = E.shape[1]
    con.execute(f"create virtual table v using vec0(id integer primary key, cam text partition key, kt integer, t_first float, emb float[{d}])")
    t0 = time.perf_counter()
    con.executemany("insert into v(id,cam,kt,t_first,emb) values (?,?,?,?,?)",
                    [(i, str(cams[i]), i % 7, float(t_first[i]), E[i].astype(np.float32).tobytes())
                     for i in range(E.shape[0])])
    con.commit()
    build_ms = (time.perf_counter() - t0) * 1000
    r = {"label": label, "n": int(E.shape[0]), "build_s": round(build_ms / 1000, 1),
         "db_mb": round(os.path.getsize(db) / 1e6, 1)}

    def timed(sql, params, reps):
        tt = []
        for _ in range(reps):
            t0 = time.perf_counter()
            con.execute(sql, params).fetchall()
            tt.append((time.perf_counter() - t0) * 1000)
        return pct(tt, 50), pct(tt, 95)

    q = Q[0].tobytes()
    r["q1_knn_ms"] = [round(x, 2) for x in timed("select id,distance from v where emb match ? and k=10", (q,), reps)]
    r["q2_notcam_ms"] = [round(x, 2) for x in timed(
        "select id,distance from v where cam != ? and emb match ? and k=10", ("316", q), reps)]
    tm = float(np.median(t_first))
    r["q3_timewin_ms"] = [round(x, 2) for x in timed(
        "select id,distance from v where t_first between ? and ? and emb match ? and k=10", (tm - 3, tm + 3, q), reps)]
    # 一致性
    s = E @ Q[0]
    ref = set(np.argsort(-s)[:10].tolist())
    got = set(x[0] for x in con.execute("select id,distance from v where emb match ? and k=10", (q,)).fetchall())
    r["top10_match"] = len(ref & got)
    # 增量写入：再插 1000 条，计时间
    base = E.shape[0]
    extra = E[:1000]
    t0 = time.perf_counter()
    con.executemany("insert into v(id,cam,kt,t_first,emb) values (?,?,?,?,?)",
                    [(base + i, "316", 0, 0.0, extra[i].astype(np.float32).tobytes()) for i in range(len(extra))])
    con.commit()
    r["insert_1k_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    con.close()
    for suf in ("", "-journal", "-wal", "-shm"):
        p = db + suf
        if os.path.exists(p):
            os.remove(p)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="")
    ap.add_argument("--sizes", default="100000,1000000")
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.RandomState(7)
    print("=" * 78)
    if args.real and os.path.exists(args.real):
        R = norm(np.load(args.real))
        print(f"[真实] {args.real}  n={R.shape[0]} d={R.shape[1]}")
        Q = R[rng.choice(R.shape[0], min(100, R.shape[0]), replace=False)]
        p50, p95, _ = bench_query(R, Q)
        print(f"  numpy 精确 Q1      : P50 {p50:.3f} / P95 {p95:.3f} ms")
        cams = np.array(["305", "316", "318", "320", "323", "343"])[rng.randint(0, 6, R.shape[0])]
        tf = rng.uniform(0, 600, R.shape[0])
        sq = sqlite_vec_run(R, cams, tf, Q, reps=100, db="/tmp/_bv_real.sqlite", label="real539")
        print("  sqlite-vec:", json.dumps(sq, ensure_ascii=False))
    else:
        print(f"[真实] 跳过（未找到 {args.real}）")

    for n in [int(x) for x in args.sizes.split(",") if x]:
        E = synth(n)
        Q = E[rng.choice(n, 100, replace=False)]
        print("-" * 78)
        print(f"[合成] n={n} d=768  f32={E.nbytes/1e6:.0f}MB  f16={E.nbytes/2e6:.0f}MB")
        p50, p95, _ = bench_query(E, Q)
        print(f"  numpy f32 Q1        : P50 {p50:.2f} / P95 {p95:.2f} ms")
        # f16 存储：转 f32 后检索（几乎零成本，因 E@q 需 f32）
        E16 = E.astype(np.float16)
        td = []
        for q in Q:
            t0 = time.perf_counter()
            s = E16.astype(np.float32) @ q
            np.argpartition(-s, 9)[:10]
            td.append((time.perf_counter() - t0) * 1000)
        print(f"  numpy f16→f32 每查   : P50 {pct(td):.2f} ms  （含每次 300MB 转换）")
        # f16 一次性转换后检索
        E16f = E16.astype(np.float32)
        del E16
        p50b, p95b, _ = bench_query(E16f, Q)
        print(f"  numpy f16 常驻检索   : P50 {p50b:.2f} ms")
        cams = np.array(["305", "316", "318", "320", "323", "343"])[rng.randint(0, 6, n)]
        tf = rng.uniform(0, 3600, n)
        m2 = np.array([c != "316" for c in cams])
        p50c, p95c, nc = bench_query(E, Q, m2)
        print(f"  numpy Q2 排除机位    : P50 {p50c:.2f} ms (候选 {nc})")
        tm = float(np.median(tf))
        m3 = (tf >= tm - 3) & (tf <= tm + 3)
        print(f"     时间窗±3s 选择率 {m3.mean():.2%} 候选 {int(m3.sum())}")
        if m3.sum() >= 10:
            p50d, p95d, _ = bench_query(E, Q, m3)
            print(f"  numpy Q3 时间窗(预过滤): P50 {p50d:.2f} ms")
        keep = (tf >= tm - 3) & (tf <= tm + 3)
        p50e, p95e = bench_postfilter(E, Q[:30], keep)
        print(f"  numpy Q3 时间窗(post)  : P50 {p50e:.2f} ms  ← ANN 在此形态会崩")
        mcost = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = E[m3]
            mcost.append((time.perf_counter() - t0) * 1000)
        print(f"  掩码构建成本 E[mask]   : P50 {pct(mcost):.2f} ms")
        if n <= 200000:
            sq = sqlite_vec_run(E, cams, tf, Q, reps=30, db="/tmp/_bv_syn.sqlite", label=f"syn{n}")
            print(f"  sqlite-vec: {json.dumps(sq, ensure_ascii=False)}")
        else:
            print("  sqlite-vec: 跳过（10^6 规模 brute force 已知不可用，且需 3GB 落盘）")
        del E, E16f

    # memmap 冷/热
    print("-" * 78)
    tmp = "/tmp/_bv_mmap.npy"
    M = synth(200000)
    np.save(tmp, M.astype(np.float16))
    del M
    mm = np.load(tmp, mmap_mode="r")
    q = np.asarray(mm[0], dtype=np.float32)
    t = []
    for _ in range(20):
        t0 = time.perf_counter()
        s = np.asarray(mm[:], dtype=np.float32) @ q
        np.argpartition(-s, 9)[:10]
        t.append((time.perf_counter() - t0) * 1000)
    print(f"  memmap 20万 f16（每查全量转换）: P50 {pct(t):.2f} ms  文件 {os.path.getsize(tmp)/1e6:.0f}MB")
    os.remove(tmp)
    print("=" * 78)


if __name__ == "__main__":
    main()
