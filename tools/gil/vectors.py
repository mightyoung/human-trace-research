# -*- coding: utf-8 -*-
"""GIL 向量缓存层：分块矩阵 + 常驻 f32 + 约束检索。

为什么不是向量数据库（实测依据，48 核 CPU / 768 维）：
    规模         numpy 精确扫描        sqlite-vec v0.1.9
    539 轨       0.10 ms               5.47 ms     (慢 57×)
    10 万        17.4 ms               190.6 ms    (慢 11×)
    100 万       136 ms                未测（brute force 已不可用）
  强过滤（时间窗 ±3s，候选 180）numpy 预过滤 **0.03 ms** —— 而 ANN 在强过滤下会遇
  "recall cliff"（文献：filtered-ANN 的困难本质是近似的代价，精确扫描下不存在）。
  因此在本项目规模（10^3–10^5）**不引入 ANN、也不引入向量数据库扩展**。

关键性能决策（实测）：
  - f16 落盘 → 启动时一次性转 **f32 常驻**：省 50% 磁盘、检索速度不变。
    ★ 绝不每查转换（10 万规模每查 f16→f32 = 400 ms；memmap 更差 660 ms）。
  - **按机位分块**：滤"排除同机位"= 不遍历该块，零拷贝。
    ★ 绝不整表 E[mask]（10^6 规模拷贝一次 940 ms；10^5 尚可 0.31 ms）。

升级触发条件（写死，避免过早上复杂度）：
  - 单块 > 3×10^5 → 该块换 faiss.IndexIVFPQ（pip install faiss-cpu 可装）
  - 需要跨进程/跨语言共享 + 高频写 → 才考虑常驻服务（Redis/Qdrant）
"""
import json
import os
from pathlib import Path

import numpy as np

DEFAULT_DIM = 768


def _l2(X):
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


class VectorBank:
    """分块向量库（磁盘真相 + 进程内 f32 缓存）。

    磁盘：<root>/<block>.npz  {"emb": f16[n,d] (已归一), "ids": int64[n]}
    内存：self._cache[block] = (E_f32[n,d], ids int64[n], id2row dict)
    """

    def __init__(self, root, dim=DEFAULT_DIM):
        self.root = Path(root)
        self.dim = dim
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        self._pending = {}     # block -> (list[emb], list[id]) 未落盘

    # ---------------- 写入 ----------------
    def add(self, block, ids, embs):
        """追加。embs: [n,d]（未归一化也可）；ids: [n] int64"""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        E = _l2(np.asarray(embs, dtype=np.float32))
        assert E.shape[1] == self.dim, (E.shape, self.dim)
        assert E.shape[0] == ids.shape[0]
        l1, l2 = self._pending.setdefault(block, ([], []))
        l1.append(E)
        l2.append(ids)
        self._cache.pop(block, None)

    def flush(self):
        """把 pending 合并到磁盘（每个块重写一次；批量写入后调用）。"""
        info = {}
        for block, (E_list, id_list) in self._pending.items():
            p = self.root / f"{block}.npz"
            old_e, old_i = None, None
            if p.exists():
                z = np.load(p)
                old_e, old_i = z["emb"], z["ids"]
            E = np.concatenate([old_e] + E_list) if old_e is not None else np.concatenate(E_list)
            I = np.concatenate([old_i] + id_list) if old_i is not None else np.concatenate(id_list)
            order = np.argsort(I)                       # 按 id 升序落盘（稳定行序）
            np.savez(p, emb=E[order].astype(np.float16), ids=I[order])
            info[block] = int(I.shape[0])
        self._pending.clear()
        return info

    # ---------------- 读取 ----------------
    def blocks(self):
        names = {p.stem for p in self.root.glob("*.npz")}
        return sorted(names | set(self._pending))

    def load(self, blocks=None, cache_dtype="float32"):
        """载入指定块到内存（默认全部）。cache_dtype: float32（推荐）| float16。"""
        used = []
        for b in (blocks or self.blocks()):
            if b in self._cache:
                used.append(b)
                continue
            p = self.root / f"{b}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            E = z["emb"].astype(cache_dtype).astype(np.float32) if cache_dtype != "float32" \
                else z["emb"].astype(np.float32)
            # 保证归一（f16 存储会带来微小偏差）
            E = _l2(E)
            ids = z["ids"].astype(np.int64)
            self._cache[b] = (E, ids, {int(v): i for i, v in enumerate(ids)})
            used.append(b)
        return used

    def n(self, block=None):
        if block:
            return len(self._cache.get(block, (np.empty(0),))[1]) if block in self._cache \
                else (int(np.load(self.root / f"{block}.npz")["ids"].shape[0])
                      if (self.root / f"{block}.npz").exists() else 0)
        return sum(len(v[1]) for v in self._cache.values())

    # ---------------- 检索 ----------------
    def search(self, q, blocks=None, allow_ids=None, exclude_ids=None,
               topk=10, oversample=1):
        """约束检索（预过滤形态，在候选子集里取精确 top-k）。

        q          : [d] 查询向量
        blocks     : 只在哪些块里找（None=全部已载入块）；"排除同机位"=传其他块名
        allow_ids  : 允许的 id 集合（None=不限）；典型来自 SQLite 的时间窗/可达查询
        exclude_ids: 排除的 id 集合
        oversample : 允许子集远大于 topk 时的候选放大（默认 1，精确扫描无需）
        返回 [{id, block, score}] 按 score 降序（最多 topk 条；子集不足则更少）
        """
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-9)
        use = [b for b in (blocks or self._cache.keys()) if b in self._cache]
        if not use:
            return []
        allow = None if allow_ids is None else np.asarray(sorted(allow_ids), dtype=np.int64)
        excl = None if exclude_ids is None else set(int(x) for x in exclude_ids)
        out = []
        for b in use:
            E, ids, id2row = self._cache[b]
            if allow is not None:
                rows = np.array([id2row[i] for i in allow if i in id2row], dtype=np.int64)
                if rows.size == 0:
                    continue
                s = E[rows] @ q
                cand_ids = ids[rows]
            else:
                s = E @ q
                cand_ids = ids
            k = min(topk, s.shape[0])
            if k <= 0:
                continue
            idx = np.argpartition(-s, k - 1)[:k]
            idx = idx[np.argsort(-s[idx])]
            for i in idx:
                i = int(i)
                gid = int(cand_ids[i])
                if (excl and gid in excl) or (allow is not None and gid not in id2row):
                    continue
                out.append({"id": gid, "block": b, "score": float(s[i])})
            out.sort(key=lambda x: -x["score"])
            out = out[:topk * max(1, oversample)]
        return out[:topk]

    def stats(self):
        s = {}
        for b in self.blocks():
            p = self.root / f"{b}.npz"
            n = self.n(b)
            s[b] = {"n": n,
                    "disk_mb": round(p.stat().st_size / 1e6, 2) if p.exists() else 0.0,
                    "loaded": b in self._cache}
        return s


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    vb = VectorBank(a.root)
    print("blocks:", json.dumps(vb.stats(), ensure_ascii=False, indent=1))
    if a.bench:
        import time
        vb.load()
        blocks = vb.blocks()
        n = vb.n()
        if not n:
            print("空库")
            return
        rng = np.random.RandomState(0)
        allids = np.concatenate([vb._cache[b][1] for b in blocks])
        q = vb._cache[blocks[0]][0][0]
        t = []
        for _ in range(200):
            t0 = time.perf_counter()
            vb.search(q, topk=10)
            t.append((time.perf_counter() - t0) * 1000)
        t.sort()
        print(f"n={n}  Q1 全库: P50 {t[100]:.3f} ms  P95 {t[190]:.3f} ms")
        if len(blocks) > 1:
            t = []
            for _ in range(200):
                t0 = time.perf_counter()
                vb.search(q, blocks=blocks[1:], topk=10)
                t.append((time.perf_counter() - t0) * 1000)
            t.sort()
            print(f"      排除首块: P50 {t[100]:.3f} ms")
        sub = set(int(x) for x in rng.choice(allids, max(10, n // 500)))
        t = []
        for _ in range(200):
            t0 = time.perf_counter()
            vb.search(q, allow_ids=sub, topk=10)
            t.append((time.perf_counter() - t0) * 1000)
        t.sort()
        print(f"      强过滤(候选 {len(sub)}): P50 {t[100]:.3f} ms")


if __name__ == "__main__":
    main()
