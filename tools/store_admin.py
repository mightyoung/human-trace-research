#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 代际库管理：inspect / rollback / migrate-in-place / migrate-legacy。

定位（与 G 包 store_admin.py 对齐，含本项目扩展）：
  - inspect         读发布代（或指定代）并做**全量校验**（哈希/行绑定/模型/维度）；
  - rollback        同空间快照回退（expected-current 防护 + intent 审计），只恢复
                    向量库本身，不撤销拓扑、模型训练或人数账本；
  - migrate-in-place  旧 flat 库**原地**升级：新代先发布、旧文件随后归档
                    `_legacy_flat/`（全程无不可读窗口、幂等）；本项目操作路径；
  - migrate-legacy  旧 flat 库 → **新根**档案模式（G 原语义），源库不动。

诚实边界：旧 flat 库的历史无法证明从未发生半写；迁移只保证"当下三文件一致"，
并在 manifest.provenance 显式记录 `source_identity_mapping_verified=false`。
运行迁移/回退前请确认没有其他进程在写该库（本工具使用 .writer.lock 互斥）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import uuid

import numpy as np

from artifact_io import atomic_json, sha256_file, writer_lock
from global_reid_store import (FORMAT_VERSION, GlobalReIDStore, _GEN_RE,
                               _LEGACY_FILES, read_pointer)

_LEGACY_NAMES = ("emb.npy", "meta.sqlite", "emb_mrl.npy", "version.json")
_SCHEMA_FIELDS = ("model_version", "dim", "mrl_dim")


# ---------------- 基础读取 ----------------
def _read_manifest(root, generation):
    mpath = os.path.join(str(root), "generations", generation, "manifest.json")
    if not os.path.isfile(mpath):
        raise ValueError(f"代不存在（缺 manifest）：{mpath}")
    with open(mpath, encoding="utf-8") as f:
        return json.load(f), mpath


def _snapshot_reader(root, pointer):
    """返回一个"固定指针"读取器类（读任意代而不看 CURRENT）。"""

    class _SnapshotReader(GlobalReIDStore):
        def _current(self):
            return pointer

    return _SnapshotReader


def snapshot(root, generation=None):
    """全量校验地打开一个代（默认当前发布代）。返回 (reader, manifest, pointer)。"""
    root = str(root)
    if generation is not None:
        if not _GEN_RE.match(generation):
            raise ValueError(f"代号非法：{generation!r}")
        m, mpath = _read_manifest(root, generation)
        pointer = {"format_version": FORMAT_VERSION, "generation": generation,
                   "manifest_sha256": sha256_file(mpath)}
    else:
        pointer = read_pointer(root)
        if pointer is None:
            legacy = any(os.path.isfile(os.path.join(root, p)) for p in _LEGACY_FILES)
            raise ValueError(
                ("旧 flat 库" if legacy else "目录内没有已发布代") +
                f"（{root}）；如为旧库请先 migrate-in-place")
        m, _ = _read_manifest(root, pointer["generation"])
    Reader = _snapshot_reader(root, pointer)
    reader = Reader(root, m["model_version"], m["dim"], mrl_dim=m.get("mrl_dim"),
                    provenance=m.get("provenance"))
    reader.load()   # 完整校验（哈希/行绑定/模型/维度）
    return reader, m, pointer


def read_legacy_flat(source):
    """严格读取旧 flat 库（迁移/对账共用；任何不一致即 raise，不做猜测修复）。"""
    source = str(source)
    for n in ("emb.npy", "meta.sqlite", "version.json"):
        if not os.path.isfile(os.path.join(source, n)):
            raise FileNotFoundError(f"旧库缺文件：{os.path.join(source, n)}")
    names = [n for n in _LEGACY_NAMES if os.path.isfile(os.path.join(source, n))]
    hashes = {n: sha256_file(os.path.join(source, n)) for n in names}
    with open(os.path.join(source, "version.json"), encoding="utf-8") as f:
        version = json.load(f)
    E = np.load(os.path.join(source, "emb.npy"), allow_pickle=False)
    con = sqlite3.connect(
        "file:" + os.path.abspath(os.path.join(source, "meta.sqlite"))
        .replace("\\", "/") + "?mode=ro", uri=True)
    try:
        recs = con.execute(
            "SELECT rowid,key,cam,date,tid,t_first,t_last,h_med,u,v,model,extra "
            "FROM tracks ORDER BY rowid").fetchall()
    finally:
        con.close()
    n = len(recs)
    if n != int(version.get("n", -1)):
        raise ValueError(f"旧库行数不一致：sqlite={n} version.json={version.get('n')}")
    if (E.ndim != 2 or E.shape != (n, int(version["dim"]))):
        raise ValueError(f"旧库 emb 形状 {E.shape} != 声明 {(n, version['dim'])}")
    keys = [r[1] for r in recs]
    if len(set(keys)) != n:
        raise ValueError("旧库存在重复键")
    if [r[0] for r in recs] != list(range(1, n + 1)):
        raise ValueError("旧库 rowid 不连续（行绑定可疑）")
    models = {r[10] for r in recs}
    if models != {version["model_version"]}:
        raise ValueError(f"旧库 model 列不统一：{models} != {version['model_version']!r}")
    Ef = E.astype(np.float32)
    if not np.isfinite(Ef).all():
        raise ValueError("旧库含 NaN/Inf（拒绝迁移，请人工检查来源）")
    norms = np.linalg.norm(Ef, axis=1)
    if n and float(norms.min()) <= 1e-12:
        raise ValueError(f"旧库存在零范数行（i={int(norms.argmin())}）；拒绝迁移")
    rows = []
    for r in recs:
        try:
            extra = json.loads(r[11]) if r[11] not in (None, "", "null") else None
        except (json.JSONDecodeError, TypeError):
            extra = None
        rows.append(dict(key=r[1], cam=r[2], date=r[3], tid=r[4], t_first=r[5],
                         t_last=r[6], h_med=r[7], u=r[8], v=r[9], extra_parsed=extra))
    if hashes != {n_: sha256_file(os.path.join(source, n_)) for n_ in names}:
        raise RuntimeError("读取期间旧库文件发生变化；请先停止旧写者后重试")
    return {"source": source, "keys": keys, "rows": rows, "E": E,
            "version": version, "hashes": hashes, "emb_dtype": str(E.dtype)}


# ---------------- 操作 ----------------
def inspect(root, generation=None, list_generations=False):
    reader, m, pointer = snapshot(root, generation)
    info = {"root": str(root), "pointer": pointer, "manifest": m,
            **reader.stats()}
    if list_generations:
        gdir = os.path.join(str(root), "generations")
        gens = []
        if os.path.isdir(gdir):
            for name in sorted(os.listdir(gdir)):
                if not _GEN_RE.match(name):
                    continue
                mpath = os.path.join(gdir, name, "manifest.json")
                size = 0
                for f_ in ("emb.npy", "meta.sqlite"):
                    fp = os.path.join(gdir, name, f_)
                    if os.path.isfile(fp):
                        size += os.path.getsize(fp)
                entry = {"generation": name, "is_current": name == pointer["generation"],
                         "bytes": size}
                if os.path.isfile(mpath):
                    try:
                        with open(mpath, encoding="utf-8") as f:
                            gm = json.load(f)
                        entry.update({"n": gm.get("n"), "parent": gm.get("parent"),
                                      "created_iso": gm.get("created_iso")})
                    except (json.JSONDecodeError, OSError):
                        entry["manifest"] = "损坏"
                gens.append(entry)
        info["generations"] = gens
    return info


def rollback(root, generation, expected_current):
    """同空间快照回退（只动 CURRENT；旧代保留）。"""
    root = str(root)
    if not _GEN_RE.match(generation):
        raise ValueError(f"目标代号非法：{generation!r}")
    with writer_lock(os.path.join(root, ".writer.lock")):
        cur = read_pointer(root)
        if cur is None:
            raise ValueError(f"无 CURRENT.json，无法回退：{root}")
        if cur["generation"] != expected_current:
            raise RuntimeError(
                f"CURRENT 已变化（当前 {cur['generation']} != 预期 {expected_current}）；"
                f"拒绝陈旧回退，请重新 inspect 后重试")
        if generation == cur["generation"]:
            raise ValueError("目标代即当前代，无需回退")
        m_cur, _ = _read_manifest(root, cur["generation"])
        m_t, mpath_t = _read_manifest(root, generation)
        for field in _SCHEMA_FIELDS:
            if m_cur.get(field) != m_t.get(field):
                raise ValueError(
                    f"跨 {field} 回退被禁止（{m_cur.get(field)!r} → {m_t.get(field)!r}）；"
                    f"请改用对应空间的另一 store 根")
        # 回退前对目标代做全量校验（含哈希/行绑定），避免切到损坏代
        snapshot(root, generation)
        audit = {"action": "rollback_store_snapshot", "from": cur["generation"],
                 "to": generation, "time_unix": time.time(),
                 "does_not_repair_other_derived_systems": True}
        atomic_json(os.path.join(root, "operations", uuid.uuid4().hex + ".intent.json"),
                    audit)
        atomic_json(os.path.join(root, "CURRENT.json"),
                    {"format_version": FORMAT_VERSION, "generation": generation,
                     "manifest_sha256": sha256_file(mpath_t)})
        return {**audit, "published": True}


def _archive_legacy(root):
    """把根目录下的旧 flat 文件移入 _legacy_flat/（内容一致的同名残留直接清理）。"""
    names = [n for n in _LEGACY_NAMES if os.path.isfile(os.path.join(root, n))]
    if not names:
        return []
    dest_dir = os.path.join(root, "_legacy_flat")
    os.makedirs(dest_dir, exist_ok=True)
    moved = []
    for n in names:
        src = os.path.join(root, n)
        dst = os.path.join(dest_dir, n)
        if os.path.exists(dst):
            if sha256_file(src) == sha256_file(dst):
                os.remove(src)          # 仅当字节一致（上次归档的重复完成）
                moved.append(n + "(dup-removed)")
                continue
            raise RuntimeError(f"归档冲突：{dst} 已存在且内容不同；请人工检查")
        os.replace(src, dst)
        moved.append(n)
    return moved


def _migration_provenance(leg, mode):
    return {"kind": "legacy_import_unverified",
            "tool": f"tools/store_admin.py {mode}",
            "migrated_unix": time.time(),
            "source_sha256": leg["hashes"],
            "source_identity_mapping_verified": False,
            "ordering_assumption": "旧库读按 rowid 序；迁移沿用该绑定（已核 rowid 连续且 = sorted(key)）",
            "numerical_change": None}


def _items_from_legacy(leg):
    return [dict(key=r["key"], emb=leg["E"][i], cam=r["cam"], date=r["date"],
                 tid=r["tid"], t_first=r["t_first"], t_last=r["t_last"],
                 h_med=r["h_med"], u=r["u"], v=r["v"], extra=r["extra_parsed"])
            for i, r in enumerate(leg["rows"])]


def migrate_in_place(root, acknowledge_unverified=False, _run_ts=None):
    """旧 flat 库原地升级（本项目路径）。

    顺序：严格读旧库 → 构造新代并用 `save(_allow_legacy_flat=True)` 发布
    （旧文件此时仍在，谁都能读）→ 独立全量校验 → 旧文件归档 _legacy_flat/ → 审计。
    崩溃任意点均不产生"新旧都不可读"窗口；重复运行幂等。
    """
    if not acknowledge_unverified:
        raise ValueError("需显式确认 acknowledge_unverified（旧库历史无法证明未发生半写；"
                         "迁移只保证当下三文件一致）")
    root = str(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"目录不存在：{root}")
    cur = read_pointer(root)
    if cur is not None:
        reader, _, _ = snapshot(root)          # 完整校验
        archived = _archive_legacy(root)
        return {"root": root, "already_migrated": True,
                "generation": cur["generation"], "n": reader.stats()["n"],
                "archived_files": archived}
    leg = read_legacy_flat(root)
    with writer_lock(os.path.join(root, ".writer.lock")):
        cur2 = read_pointer(root)              # 双检（防止并行迁移）
        if cur2 is not None:
            reader, _, _ = snapshot(root)
            archived = _archive_legacy(root)
            return {"root": root, "already_migrated": True,
                    "generation": cur2["generation"], "n": reader.stats()["n"],
                    "archived_files": archived}
        st = GlobalReIDStore(root, leg["version"]["model_version"],
                             leg["version"]["dim"],
                             mrl_dim=leg["version"].get("mrl_dim"),
                             provenance=_migration_provenance(leg, "migrate-in-place"))
        st.add_many(_items_from_legacy(leg))
        n = st.save(_allow_legacy_flat=True)   # 新代先发布（旧文件仍在）
        reader, _, _ = snapshot(root)          # 独立全量校验（经 CURRENT 读回）
        archived = _archive_legacy(root)       # 再归档旧文件
        result = {"root": root, "already_migrated": False,
                  "generation": reader._base_generation, "n": n,
                  "archived_files": archived,
                  "source_sha256": leg["hashes"],
                  "source_identity_mapping_verified": False}
        atomic_json(os.path.join(root, "operations", uuid.uuid4().hex + ".migrate.json"),
                    {**result, "action": "migrate_legacy_in_place",
                     "time_unix": time.time()})
        return result


def migrate_legacy(source, destination, acknowledge_unverified=False):
    """旧 flat 库 → 新根档案模式（G 原语义；源库不动）。"""
    if not acknowledge_unverified:
        raise ValueError("需显式确认 acknowledge_unverified（见 migrate-in-place 说明）")
    source = os.path.abspath(str(source))
    destination = os.path.abspath(str(destination))
    if (source == destination or source.startswith(destination + os.sep)
            or destination.startswith(source + os.sep)):
        raise ValueError("目标必须是独立的新根（不得位于源库之内/之上）")
    if os.path.exists(destination) and os.listdir(destination):
        raise ValueError(f"目标目录非空：{destination}")
    leg = read_legacy_flat(source)
    os.makedirs(destination, exist_ok=True)
    st = GlobalReIDStore(destination, leg["version"]["model_version"],
                         leg["version"]["dim"], mrl_dim=leg["version"].get("mrl_dim"),
                         provenance=_migration_provenance(leg, "migrate-legacy"))
    st.add_many(_items_from_legacy(leg))
    n = st.save()
    reader, _, _ = snapshot(destination)
    result = {"source": source, "destination": destination, "n": n,
              "generation": reader._base_generation,
              "source_untouched": True, "source_sha256": leg["hashes"],
              "source_identity_mapping_verified": False}
    atomic_json(os.path.join(destination, "operations",
                             uuid.uuid4().hex + ".migrate.json"),
                {**result, "action": "migrate_legacy_new_root",
                 "time_unix": time.time()})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect")
    p.add_argument("--root", required=True)
    p.add_argument("--generation", default=None)
    p.add_argument("--list-generations", action="store_true")
    p = sub.add_parser("rollback")
    p.add_argument("--root", required=True)
    p.add_argument("--generation", required=True)
    p.add_argument("--expected-current", required=True)
    p = sub.add_parser("migrate-in-place")
    p.add_argument("--root", required=True)
    p.add_argument("--acknowledge-unverified", action="store_true")
    p = sub.add_parser("migrate-legacy")
    p.add_argument("--source", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--acknowledge-unverified", action="store_true")
    args = ap.parse_args()
    if args.command == "inspect":
        result = inspect(args.root, args.generation, args.list_generations)
    elif args.command == "rollback":
        result = rollback(args.root, args.generation,
                          expected_current=args.expected_current)
    elif args.command == "migrate-in-place":
        result = migrate_in_place(args.root,
                                  acknowledge_unverified=args.acknowledge_unverified)
    else:
        result = migrate_legacy(args.source, args.destination,
                                acknowledge_unverified=args.acknowledge_unverified)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
