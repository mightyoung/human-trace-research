# -*- coding: utf-8 -*-
"""全局 ReID 嵌入缓存层 —— v2 代际格式（完整代际方案）。

存储布局（一个 store 目录 = 一个模型版本空间）：
  store/
    CURRENT.json                  # 唯一发布指针 {"format_version":2,"generation":"g<hex32>","manifest_sha256":…}
    .writer.lock                  # 跨进程写锁（本地单机）
    generations/g<uuid32hex>/     # 不可变代（发布后不再改写）
      emb.npy                     # [N, D] float16（dtype 记录于 manifest；读取时转 f32 常驻）
      meta.sqlite                 # row_index ↔ key 绑定 + 元数据（含完整 extra 证据）
      manifest.json               # 模型/维度/schema/父代/文件 sha256（内容哈希锚）
    operations/                   # 迁移/回退审计（仅有记录意义，非提交证明）
    _legacy_flat/                 # （经原地迁移后）旧 flat 文件存档，供对账/回退

发布序（LevelDB MANIFEST/CURRENT 模式）：
  staging 完整写（emb→sync→sqlite→sync→manifest）→ 改名不可变代 → fsync →
  **最后**原子写 CURRENT.json。崩溃任意点：CURRENT 未切 → 旧代继续可读、半成品不可见；
  已切 → 新代完整可读。stale writer（基于 _base_generation 比对）拒绝盲写覆盖。

读取（fail-closed）：校验 CURRENT → manifest(sha256) → 文件(sha256) → 行数/row_index/
  维度/模型，任一不符即 raise（半写/篡改不再静默错绑）。

旧 flat 库（emb.npy + meta.sqlite [+ emb_mrl.npy] + version.json，无 CURRENT.json）
**不静默迁移**；显式执行：
  python tools/store_admin.py migrate-in-place --root <dir> --acknowledge-unverified
（原地迁移：新代先发布、旧文件随后归档 _legacy_flat/，全程无不可读窗口）

与 G 包（chatgpt-human-count-core-improved）参考实现的四项规范偏离（理由见
docs/commit-015/GEN_STORE_V2_IMPLEMENTATION.md）：
  1. 落盘精度保持 float16（全链路数值连续；G 的 float32 属其重编码计划的一部分）；
  2. L2 归一化仍发生在检索端（_matrix），入库不归一（与现行行为逐位一致）；
  3. mrl_dim 不符仅告警（前缀为派生投影，不改变加载语义）；
  4. schema_id 记录于 manifest 供审计，不参与致命校验（model/dim 已分别校验）。

用法（CLI）：
  python tools/global_reid_store.py --stats <store_dir>   # 打印当前代统计
"""
import copy
import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid

import numpy as np

from artifact_io import (atomic_json, canonical_json, fsync_dir, sha256_file,
                         sync_file, writer_lock)

FORMAT_VERSION = 2
_GEN_RE = re.compile(r"^g[0-9a-f]{32}$")
_GEN_FILES = ("emb.npy", "meta.sqlite")
_LEGACY_FILES = ("emb.npy", "meta.sqlite", "emb_mrl.npy", "version.json")
_EMB_DTYPE = np.float16

TIME_OFFS = {"316": 4.0, "305": -5.0, "318": 20.0, "320": 19.0,
             "323": 18.0, "343": 19.0}   # 相对 08:55:00 的秒偏移（OSD 标定）


def key_to_tid(k):
    """'a1' -> 1, 'b12' -> 12, 'v323_45' -> 45（取尾部数字）"""
    num = ""
    for ch in reversed(k):
        if ch.isdigit():
            num = ch + num
        else:
            break
    return int(num)


def read_pointer(root):
    """读取 CURRENT.json 发布指针（模块级唯一实现）。

    不存在 → None；存在但损坏（格式/代号/哈希不合法）→ raise（fail-closed）。
    """
    p = os.path.join(str(root), "CURRENT.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            x = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"CURRENT.json 损坏（无法解析）：{p}｜{e}") from e
    if (not isinstance(x, dict) or x.get("format_version") != FORMAT_VERSION
            or not _GEN_RE.match(str(x.get("generation", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(x.get("manifest_sha256", "")))):
        raise ValueError(f"CURRENT.json 损坏（格式/代号/哈希不合法）：{x!r}")
    return x


class GlobalReIDStore:
    """轨道级嵌入库：add / search / stats（v2 代际格式）。

    - 键：f"{date}|{cam}|{tid}"
    - 检索：暴力余弦（<10 万轨规模毫秒级）；>100 万轨切换 faiss IVF
    - 两级检索（MRL 256d 粗筛 + 768d 精排）：前缀矩阵只从内存同代派生（不落盘、不跨代复用）
    """

    def __init__(self, root, model_version, dim, mrl_dim=256, provenance=None):
        if not root:
            raise ValueError("root 必填")
        if not model_version:
            raise ValueError("model_version 必填")
        if int(dim) != dim or dim <= 0:
            raise ValueError(f"dim 须为正整数：{dim!r}")
        self.root = str(root)
        self.model_version = str(model_version)
        self.dim = int(dim)
        self.mrl_dim = int(mrl_dim) if (mrl_dim and int(mrl_dim) < self.dim) else None
        self.schema = {"kind": "reid_track_embedding",
                       "manual_model_version": self.model_version,
                       "dim": self.dim, "mrl_dim": self.mrl_dim}
        import hashlib
        self.schema_id = "reid-v2:" + hashlib.sha256(
            canonical_json(self.schema)).hexdigest()[:24]
        self._provenance = copy.deepcopy(provenance)
        os.makedirs(self.root, exist_ok=True)
        self.rows = {}       # key -> np.ndarray [D] float32（内存常驻）
        self.meta = {}       # key -> dict
        self._E = None       # 检索用缓存矩阵（全维，已归一化）
        self._M = None       # 检索用缓存矩阵（mrl 前缀，已归一化）
        self._keys = None
        self._base_generation = None   # 本实例所见/发布的代（stale writer 判定）
        self._lock = threading.RLock()
        self.emb_path = ""       # 当前代 emb.npy（load/save 后可用）
        self.emb_mrl_path = ""   # v2 不再落盘 MRL 文件（恒空；保留属性兼容）
        self.db_path = ""        # 当前代 meta.sqlite

    # ---------- 指针与路径 ----------
    def _current(self):
        """读 CURRENT.json；不存在返回 None；存在但损坏 → raise（fail-closed）。"""
        return read_pointer(self.root)

    def _set_paths(self, gen_dir):
        if gen_dir is None:
            self.emb_path = self.db_path = self.emb_mrl_path = ""
        else:
            self.emb_path = os.path.join(str(gen_dir), "emb.npy")
            self.db_path = os.path.join(str(gen_dir), "meta.sqlite")
            self.emb_mrl_path = ""   # MRL 前缀只在内存派生

    def _invalidate(self):
        self._E = None
        self._M = None
        self._keys = None

    def _fault_hook(self, phase):
        """测试故障注入点（默认 no-op）：'after_vectors' / 'before_publish'。"""

    # ---------- 写入（内存态） ----------
    def add(self, key, emb, cam, date, tid, t_first=None, t_last=None,
            h_med=None, u=None, v=None, extra=None):
        if not isinstance(key, str) or not key:
            raise ValueError(f"key 须为非空字符串：{key!r}")
        if cam is None or date is None:
            raise ValueError("cam 与 date 必填")
        e = np.asarray(emb, dtype=np.float32).reshape(-1)
        if e.shape[0] != self.dim:
            raise ValueError(f"维度不符：{e.shape[0]} != {self.dim}（key={key}）")
        if not np.isfinite(e).all():
            raise ValueError(f"embedding 含 NaN/Inf（key={key}）")
        if not float(np.linalg.norm(e)) > 1e-12:
            raise ValueError(f"embedding 零范数（key={key}）")
        if tid is None or isinstance(tid, bool) or int(tid) != tid:
            raise ValueError(f"tid 须为整数：{tid!r}")
        for name, val in (("t_first", t_first), ("t_last", t_last),
                          ("h_med", h_med), ("u", u), ("v", v)):
            if val is not None and not np.isfinite(val):
                raise ValueError(f"{name} 非有限：{val!r}")
        if t_first is not None and t_last is not None and t_first > t_last:
            raise ValueError(f"t_first > t_last（{t_first} > {t_last}）")
        canonical_json(extra)        # 预校验可序列化（不落盘、不改状态）
        m = dict(cam=cam, date=date, tid=int(tid), t_first=t_first,
                 t_last=t_last, h_med=h_med, u=u, v=v, extra=copy.deepcopy(extra))
        with self._lock:
            self.rows[key] = e
            self.meta[key] = m
            self._invalidate()

    def add_many(self, items):
        """整批校验后一次性并入（任一项非法 → 全部不落，内存零变化）。"""
        staging = GlobalReIDStore.__new__(GlobalReIDStore)
        staging.dim = self.dim
        staging._lock = threading.RLock()
        staging._E = staging._M = staging._keys = None
        staging.rows, staging.meta = {}, {}
        for it in items:
            staging.add(**it)
        with self._lock:
            self.rows.update(staging.rows)
            self.meta.update(staging.meta)
            self._invalidate()
        return len(staging.rows)

    def apply_and_save(self, items):
        """整批写入并发布（全有或全无）。

        发布失败时：若 CURRENT 已被其他写者推移（base 不符），重载实际发布态
        （不假装回滚已提交的代）；否则恢复内存原状后抛出。
        """
        with self._lock:
            old_rows = dict(self.rows)
            old_meta = copy.deepcopy(self.meta)
            base = self._base_generation
            try:
                self.add_many(items)
                return self.save()
            except Exception:
                cur = None
                try:
                    cur = self._current()
                except Exception:
                    cur = None
                if cur is not None and cur["generation"] != base:
                    self.load()      # 以实际发布态为准
                else:
                    self.rows, self.meta = old_rows, old_meta
                    self._invalidate()
                raise

    def add_npz(self, npz_path, cam, date, meta_fn=None):
        """npz（键 = 轨迹键；值 = [D] 嵌入）批量入库。meta_fn(tid)->dict"""
        d = np.load(npz_path)
        n = 0
        for k in d.files:
            tid = key_to_tid(k)
            m = meta_fn(tid) if meta_fn else {}
            self.add(f"{date}|{cam}|{tid}", d[k], cam=cam, date=date, tid=tid, **m)
            n += 1
        return n

    def reset(self):
        """清空内存行（保留 _base_generation 与根目录状态）。

        用于"整库重建"流程（配合 prepare_rebuild）：save() 将发布内容为清空后
        累计的新代，父代 = 当前代（旧代仍可经 store_admin rollback 回退）。
        """
        with self._lock:
            self.rows, self.meta = {}, {}
            self._invalidate()

    def prepare_rebuild(self):
        """重建（replace-all）流程的准备：已有已发布代时先 load 采用基线，
        再清空内存行。避免"构造新实例直接 save"触发 stale-writer 拒绝。"""
        if self._current() is not None:
            self.load()
        self.reset()

    # ---------- 持久化 ----------
    def save(self, _allow_legacy_flat=False):
        """发布新代（返回行数）。CURRENT 是最后一步、原子切换。"""
        root = os.path.join(self.root)
        with self._lock, writer_lock(os.path.join(root, ".writer.lock")):
            cur = self._current()
            actual = cur["generation"] if cur else None
            if actual != self._base_generation:
                raise RuntimeError(
                    f"stale writer：CURRENT 已变化（base={self._base_generation!r}，"
                    f"current={actual!r}）；请先 load() 最新状态并重新应用改动，"
                    f"勿盲目重存覆盖")
            if (cur is None and not _allow_legacy_flat
                    and any(os.path.exists(os.path.join(root, p))
                            for p in _LEGACY_FILES)):
                raise ValueError(
                    "检测到旧 flat 库（无 CURRENT.json，且存在 emb.npy/meta.sqlite/"
                    "version.json）；v2 不静默覆盖迁移，请显式执行：\n"
                    f"  python tools/store_admin.py migrate-in-place --root {root} "
                    "--acknowledge-unverified")
            generation = "g" + uuid.uuid4().hex
            parent_dir = os.path.join(root, "generations")
            os.makedirs(parent_dir, exist_ok=True)
            stage = os.path.join(parent_dir, ".stage-" + generation)
            dest = os.path.join(parent_dir, generation)
            os.mkdir(stage)
            try:
                keys = sorted(self.rows)
                E = (np.stack([self.rows[k] for k in keys]).astype(_EMB_DTYPE)
                     if keys else np.empty((0, self.dim), _EMB_DTYPE))
                # ① 向量矩阵
                np.save(os.path.join(stage, "emb.npy"), E, allow_pickle=False)
                sync_file(os.path.join(stage, "emb.npy"))
                self._fault_hook("after_vectors")
                # ② 元数据（row_index ↔ key 显式绑定）
                con = sqlite3.connect(os.path.join(stage, "meta.sqlite"))
                try:
                    con.execute("PRAGMA synchronous=FULL")
                    con.execute("""CREATE TABLE tracks(
                        row_index INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL,
                        cam TEXT, date TEXT, tid INTEGER, t_first REAL, t_last REAL,
                        h_med REAL, u REAL, v REAL, model TEXT, extra TEXT)""")
                    for i, k in enumerate(keys):
                        m = self.meta[k]
                        con.execute(
                            "INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (i, k, m["cam"], m["date"], m["tid"], m["t_first"],
                             m["t_last"], m["h_med"], m["u"], m["v"],
                             self.model_version,
                             canonical_json(m.get("extra")).decode("utf-8")))
                    con.commit()
                finally:
                    con.close()
                sync_file(os.path.join(stage, "meta.sqlite"))
                # ③ manifest（文件哈希锚）
                manifest = {
                    "format_version": FORMAT_VERSION,
                    "generation": generation,
                    "parent": actual,
                    "model_version": self.model_version,
                    "schema_id": self.schema_id,
                    "schema": self.schema,
                    "dim": self.dim,
                    "mrl_dim": self.mrl_dim,
                    "emb_dtype": np.dtype(_EMB_DTYPE).name,
                    "n": len(keys),
                    "created_unix": time.time(),
                    "created_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "files": {name: sha256_file(os.path.join(stage, name))
                              for name in _GEN_FILES},
                }
                if self._provenance is not None:
                    manifest["provenance"] = self._provenance
                atomic_json(os.path.join(stage, "manifest.json"), manifest)
                fsync_dir(stage)
                # ④ 改名不可变代
                os.replace(stage, dest)
                fsync_dir(parent_dir)
                self._fault_hook("before_publish")
                # ⑤ 原子切换唯一发布指针（完成标志 = 最后一步）
                atomic_json(os.path.join(root, "CURRENT.json"),
                            {"format_version": FORMAT_VERSION,
                             "generation": generation,
                             "manifest_sha256": sha256_file(
                                 os.path.join(dest, "manifest.json"))})
                self._base_generation = generation
                self._set_paths(dest)
                return len(keys)
            finally:
                if os.path.exists(stage):
                    shutil.rmtree(stage, ignore_errors=True)

    def load(self, strict=True):
        """加载当前发布代（fail-closed）。

        strict=True（默认）：模型版本不符直接 raise；
        strict=False：仅把"模型版本/schema 不符"降级为告警（哈希/行数/维度校验
        永远生效，不设逃生门）。
        旧 flat 库（无 CURRENT.json）一律拒绝并给出迁移指引。
        """
        with self._lock:
            root = self.root
            cur = self._current()
            if cur is None:
                if any(os.path.exists(os.path.join(root, p)) for p in _LEGACY_FILES):
                    raise ValueError(
                        "旧 flat 库（无 CURRENT.json）：请显式迁移后再加载：\n"
                        f"  python tools/store_admin.py migrate-in-place --root {root} "
                        "--acknowledge-unverified")
                self.rows, self.meta = {}, {}
                self._base_generation = None
                self._set_paths(None)
                self._invalidate()
                return 0
            gen_dir = os.path.join(root, "generations", cur["generation"])
            mpath = os.path.join(gen_dir, "manifest.json")
            if not os.path.isfile(mpath) or sha256_file(mpath) != cur["manifest_sha256"]:
                raise ValueError(f"manifest 校验失败（sha256 不符或缺失）：{mpath}")
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            for field, expected in (("format_version", FORMAT_VERSION),
                                    ("generation", cur["generation"]),
                                    ("dim", self.dim)):
                if m.get(field) != expected:
                    raise ValueError(
                        f"manifest 字段不符：{field}={m.get(field)!r} != {expected!r}"
                        f"（库={gen_dir}）")
            if m.get("model_version") != self.model_version:
                msg = (f"store model_version={m.get('model_version')!r} != 实例 "
                       f"{self.model_version!r}（嵌入空间可能不可比）")
                if strict:
                    raise ValueError(msg + "；如确认无误读，显式 load(strict=False)")
                print(f"[warn] {msg}（strict=False 继续加载）")
            if m.get("mrl_dim") != self.mrl_dim:
                print(f"[warn] store mrl_dim={m.get('mrl_dim')!r} != 实例 "
                      f"{self.mrl_dim!r}（前缀为派生投影，仅提示配置差异）")
            emb_dtype = m.get("emb_dtype")
            if emb_dtype not in ("float16", "float32"):
                raise ValueError(f"manifest emb_dtype 非法：{emb_dtype!r}")
            if set(m.get("files", {})) != set(_GEN_FILES):
                raise ValueError(f"manifest 文件集非法：{sorted(m.get('files', {}))}")
            for name, hash_ in m["files"].items():
                fp = os.path.join(gen_dir, name)
                if not os.path.isfile(fp) or sha256_file(fp) != hash_:
                    raise ValueError(f"文件校验失败（sha256 不符）：{fp}")
            E = np.load(os.path.join(gen_dir, "emb.npy"), allow_pickle=False)
            if E.ndim != 2 or E.shape != (m["n"], self.dim):
                raise ValueError(
                    f"emb.npy 形状 {E.shape} != manifest 声明 {(m['n'], self.dim)}")
            if not np.isfinite(E).all():
                raise ValueError("emb.npy 含 NaN/Inf（拒绝加载）")
            con = sqlite3.connect(
                "file:" + os.path.abspath(
                    os.path.join(gen_dir, "meta.sqlite")).replace("\\", "/") + "?mode=ro",
                uri=True)
            try:
                recs = con.execute(
                    "SELECT row_index,key,cam,date,tid,t_first,t_last,h_med,u,v,"
                    "model,extra FROM tracks ORDER BY row_index").fetchall()
            finally:
                con.close()
            if len(recs) != len(E):
                raise ValueError(
                    f"行数不一致：sqlite={len(recs)} emb={len(E)}（拒绝加载，"
                    f"避免键↔向量错绑）")
            rows, meta = {}, {}
            for i, r in enumerate(recs):
                # 行标签一致性对比 manifest 自述值（而非实例值）：strict=False 时
                # 实例可有意放宽版本，但库内各行必须同标签（防混装）。
                if r[0] != i or r[1] in rows or r[10] != m.get("model_version"):
                    raise ValueError(
                        f"行绑定非法（row_index/重复键/model 标签）：第 {i} 行 {r[1]!r}")
                try:
                    extra = json.loads(r[11]) if r[11] is not None else None
                except json.JSONDecodeError as e:
                    raise ValueError(f"extra 非法 JSON（key={r[1]!r}）：{e}") from e
                rows[r[1]] = np.asarray(E[i], dtype=np.float32)
                meta[r[1]] = dict(cam=r[2], date=r[3], tid=r[4], t_first=r[5],
                                  t_last=r[6], h_med=r[7], u=r[8], v=r[9],
                                  extra=extra)
            self.rows, self.meta = rows, meta
            self._base_generation = cur["generation"]
            self._provenance = m.get("provenance")
            self._set_paths(gen_dir)
            self._invalidate()
            return len(rows)

    @classmethod
    def open(cls, root):
        """只读打开：由 CURRENT+manifest 自述参数构造并加载（供分析工具使用）。"""
        root = str(root)
        cp = os.path.join(root, "CURRENT.json")
        if not os.path.exists(cp):
            legacy = any(os.path.exists(os.path.join(root, p)) for p in _LEGACY_FILES)
            raise ValueError(
                ("旧 flat 库" if legacy else "目录内没有已发布代（缺 CURRENT.json）；"
                 "空目录请先构建") +
                (f"：请先迁移 python tools/store_admin.py migrate-in-place "
                 f"--root {root} --acknowledge-unverified" if legacy else ""))
        with open(cp, encoding="utf-8") as f:
            x = json.load(f)
        gen = str(x.get("generation", ""))
        if not _GEN_RE.match(gen):
            raise ValueError(f"CURRENT.json 代号非法：{gen!r}")
        mpath = os.path.join(root, "generations", gen, "manifest.json")
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        st = cls(root, m["model_version"], m["dim"], mrl_dim=m.get("mrl_dim"))
        st.load()
        return st

    # ---------- 检索 ----------
    def _matrix(self):
        if self._E is None:
            self._keys = sorted(self.rows.keys())
            if self._keys:
                E = np.stack([self.rows[k] for k in self._keys]).astype(np.float32)
                E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            else:
                E = np.empty((0, self.dim), dtype=np.float32)   # 空库安全
            self._E = E
        return self._E, self._keys

    def _matrix_mrl(self):
        """MRL 前缀矩阵：只从内存 rows 现算（不落盘、不跨代复用）。

        原因（评审 store-P1-4）：覆盖键后行数不变时，任何磁盘前缀缓存都是旧值，
        会造成"旧前缀粗筛 → 新全维精排"的候选错配（合法近邻被粗筛漏掉）。
        """
        if self.mrl_dim is None:
            return None, None
        if self._M is None:
            self._matrix()                       # 确保 _keys 已就绪
            if not self._keys:
                self._M = np.empty((0, self.mrl_dim), dtype=np.float32)
            else:
                M = np.stack([self.rows[k][:self.mrl_dim]
                              for k in self._keys]).astype(np.float32)
                M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
                self._M = M
        return self._M, self._keys

    def search(self, q, topk=10, cams=None, exclude_cam=None,
               t_range=None, exclude_key=None, stage="auto", nprobe=None):
        """q: [D] 查询向量；t_range=(t0,t1) 过滤首次出现时间（秒）。

        stage: "single" 全维暴力 | "two" MRL 粗筛+全维精排 | "auto"（有 MRL 则两级）
        nprobe: 两级检索粗筛保留数（默认 max(topk*8, 64)）

        先过滤后粗筛（评审 store-P1-3）：强约束（cam/t_range，选择率可低至 0.2–3%）
        先构造允许集合，再在集合内粗筛+精排；避免"全库 top-N 候选再过滤"在强约束下
        返回空/不足结果。
        """
        E, keys = self._matrix()
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-9)
        if topk < 0 or int(topk) != topk:
            raise ValueError(f"topk 须为非负整数：{topk!r}")
        if stage not in ("single", "two", "auto"):
            raise ValueError(f"stage 非法：{stage!r}（single/two/auto）")
        if topk == 0 or E.shape[0] == 0:
            return []
        allowed = []
        for i, k in enumerate(keys):
            m = self.meta[k]
            if cams is not None and m["cam"] not in cams:
                continue
            if exclude_cam is not None and m["cam"] == exclude_cam:
                continue
            if exclude_key is not None and k == exclude_key:
                continue
            if t_range is not None:
                t = m["t_first"] if m["t_first"] is not None else -1e9
                if not (t_range[0] <= t <= t_range[1]):
                    continue
            allowed.append(i)
        if not allowed:
            return []
        ai = np.asarray(allowed)
        # 无 MRL 时 two 档优雅降级为单级（而非 None 下标崩溃）
        two = self.mrl_dim is not None and stage in ("two", "auto")
        if two:
            M, _ = self._matrix_mrl()
            npb = int(nprobe or max(topk * 8, 64))
            npb = min(npb, len(ai))
            qm = q[:self.mrl_dim]
            qm = qm / (np.linalg.norm(qm) + 1e-9)
            sm = M[ai] @ qm
            if npb < len(ai):
                cand = ai[np.argpartition(-sm, npb - 1)[:npb]]
            else:
                cand = ai
            sf = E[cand] @ q
            oi = np.argsort(-sf)
            order = cand[oi]
            scores = sf[oi]
        else:
            s = E[ai] @ q
            oi = np.argsort(-s)
            order = ai[oi]
            scores = s[oi]

        out = []
        for i, sc in zip(order, scores):
            k = keys[i]
            out.append({"key": k, "score": float(sc), **self.meta[k]})
            if len(out) >= topk:
                break
        return out

    def stats(self):
        E, keys = self._matrix()
        n = len(keys)
        by_cam = {}
        for k in keys:
            c = self.meta[k]["cam"]
            by_cam[c] = by_cam.get(c, 0) + 1
        b_full = (os.path.getsize(self.emb_path)
                  if self.emb_path and os.path.exists(self.emb_path) else 0)
        return {"n": n, "dim": self.dim, "mrl_dim": self.mrl_dim,
                "bytes": b_full, "bytes_emb": b_full, "bytes_mrl": 0,
                "bytes_per_track": (b_full / n) if n else 0,
                "by_cam": by_cam,
                "model_version": self.model_version,
                "generation": self._base_generation,
                "schema_id": self.schema_id,
                "provenance": self._provenance}


def build_meta_from_detections(cam, det_jsonl, date="2026-10-12", fps=25.0):
    """从 detections.jsonl 提取 {tid: dict(t_first,t_last,h_med,u,v)}（时间=OSD 对齐秒）"""
    import json as _json
    acc = {}
    with open(det_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _json.loads(line)
            fr = int(r["frame"])
            tids = r.get("track_ids") or []
            boxes = r.get("boxes") or []
            for i, t in enumerate(tids):
                if t is None or t < 0 or i >= len(boxes):
                    continue
                b = boxes[i]
                h = b[3] - b[1]
                d = acc.setdefault(int(t), {"f0": fr, "f1": fr, "hs": [], "u": 0.0, "v": 0.0})
                d["f0"] = min(d["f0"], fr)
                d["f1"] = max(d["f1"], fr)
                d["hs"].append(h)
                d["u"], d["v"] = (b[0] + b[2]) / 2, b[3]
    off = TIME_OFFS.get(cam, 0.0)
    out = {}
    for t, d in acc.items():
        hs = np.array(d["hs"])
        out[t] = {"t_first": off + d["f0"] / fps,
                  "t_last": off + d["f1"] / fps,
                  "h_med": float(np.median(hs)),
                  "u": float(d["u"]), "v": float(d["v"])}
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="v2 代际库统计")
    ap.add_argument("--stats", default="output_crosscam/analysis/global_reid/store_cradio")
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.stats, "CURRENT.json")):
        st = GlobalReIDStore.open(args.stats)
        print(json.dumps(st.stats(), ensure_ascii=False, indent=1))
    else:
        print(f"（无 CURRENT.json：{args.stats}）")
