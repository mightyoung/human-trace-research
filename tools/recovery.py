#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""recovery point · 让「换个会话接着做」在技术上成立

## 为什么需要

此前只有零散的持久化：模板池存了参数、身份图是文件、嵌入是 npz、数据范围散在
各段 summary 里。**模板文件还在，不代表恢复后的算法行为与中断前相同**——这一点
已实测（sigma 0.42 重启后回 0.60）。本工具把它们收进一个可核对、可回退的点。

## 一个恢复点包含什么

| 项 | 来源 | 缺失时 |
|---|---|---|
| `code` | git rev + 脏文件数 + 跟踪文件数 | 必填 |
| `pack` | `webapp/backend/data/.pack_tag` | 必填 |
| `gallery` | `store_meta.json` 的 version 块（model/dim/n） | 必填 |
| `pool` | `selector_pool.json`（模板 + 行为参数） | 可缺席（空池也是合法状态） |
| `graph` | 生产身份图 + 属性闸门结果 | 必填 |
| `embeddings` | 逐 npz sha256 | 必填 |
| `tracks_range` | 各机位 summary 的 frames_processed/stride/fps | 必填 |
| `labels` | 真值集 + 复核记录 | 必填 |
| `training` | 优化器/调度器/采样进度 | **当前不存在**，如实记 absent |
| `ledger` | `ops/ops.db` sha + run 计数 | 必填 |

## 三种操作

    python tools/recovery.py snapshot --tag v1 --note "EXP-003b 跑完"
    python tools/recovery.py verify rp_20261125_094500_v1     # 与当前状态比，报漂移
    python tools/recovery.py restore rp_20261125_094500_v1    # 只回退可恢复项

`snapshot` 会把**小体积、可回退**的项复制到 `ops/recovery/<rp_id>/files/`；
大件（嵌入、轨迹、权重）只记 sha256，`restore` 明确拒绝回滚它们并提示重跑命令。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP_DIR = os.path.join(ROOT, "ops", "recovery")


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def sha256_file(p: str, chunk: int = 1 << 20) -> str | None:
    if not os.path.isfile(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return ""


# ------------------------------------------------------------------ 各项采集
def item_code() -> dict:
    rev = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain")
    tracked = git("ls-files")
    return dict(kind="ref", restorable=False,
                value={"git_rev": rev,
                       "dirty_files": len([x for x in dirty.splitlines() if x.strip()]),
                       "tracked_files": len([x for x in tracked.splitlines() if x.strip()])},
                note="代码版本；恢复需 git checkout <rev>，本工具不回滚代码")


def item_pack() -> dict:
    p = os.path.join(ROOT, "webapp/backend/data/.pack_tag")
    if not os.path.isfile(p):
        return dict(kind="state", restorable=True, missing=".pack_tag 不存在")
    txt = open(p, encoding="utf-8").read().strip()
    return dict(kind="state", restorable=True, value={"pack": txt}, files=[p],
                sha256=sha256_file(p), bytes=os.path.getsize(p))


def item_gallery() -> dict:
    p = os.path.join(ROOT, "webapp/backend/data/reid/store_meta.json")
    if not os.path.isfile(p):
        return dict(kind="state", restorable=False, missing="store_meta.json 不存在")
    d = json.load(open(p, encoding="utf-8"))
    v = d.get("version") or {}
    # 图库的「身份 → 轨」映射与嵌入分开存：meta 只给版本与规模，索引文件要一起核
    parts = {}
    for f in ("store_meta.json", "entity_index.json"):
        fp = os.path.join(ROOT, "webapp/backend/data/reid", f)
        if os.path.isfile(fp):
            parts[f"webapp/backend/data/reid/{f}"] = {
                "sha256": sha256_file(fp), "bytes": os.path.getsize(fp)}
    return dict(kind="state", restorable=False,
                value={"model_version": v.get("model_version"), "dim": v.get("dim"),
                       "n_tracks": v.get("n") or d.get("n"),
                       "updated": v.get("updated")},
                parts=parts,
                note="图库版本块 + 轨→实体索引；重建需重算嵌入，本工具只校验")


def item_pool() -> dict:
    p = os.path.join(ROOT, "webapp/backend/data/reid/selector_pool.json")
    if not os.path.isfile(p):
        return dict(kind="state", restorable=True, missing="尚无模板池（空池是合法状态）")
    d = json.load(open(p, encoding="utf-8"))
    return dict(kind="state", restorable=True,
                value={"params": d.get("params"), "n_entries": len(d.get("entries") or []),
                       "model_version": d.get("model_version"), "pack": d.get("pack")},
                files=[p], sha256=sha256_file(p), bytes=os.path.getsize(p),
                note="模板 + 行为参数；这是唯一能整体回退的运行时状态")


def item_graph() -> dict:
    files = [os.path.join(ROOT, p) for p in (
        "deliverable/derived-results-r3-2026-11-24/20-graph/graph_450_v1.json",
        "deliverable/gt_long35/attr_filter_ingraph.json",
        "deliverable/derived-results-r3-2026-11-24/20-graph/attr_gate_runs.json",
    )]
    have = [f for f in files if os.path.isfile(f)]
    if not have:
        return dict(kind="state", restorable=False, missing="身份图不在预期路径")
    return dict(kind="state", restorable=False,
                value={"files": [os.path.basename(f) for f in have]},
                parts={rel(f): {"sha256": sha256_file(f), "bytes": os.path.getsize(f)} for f in have},
                note="生产身份图 + 属性闸门；修改须走 diff+确认，本工具只校验")


def item_embeddings() -> dict:
    """嵌入：选择器的图库嵌入 + 各数据集的 per-cam 嵌入。"""
    singles = ["webapp/backend/data/reid/store_emb.npy"]
    dirs = ["output_crosscam/long35/embs_osnet"]
    parts, n, total = {}, 0, 0
    for s in singles:
        p = os.path.join(ROOT, s)
        if os.path.isfile(p):
            parts[s] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            n += 1
            total += os.path.getsize(p)
    for d in dirs:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        for fn in sorted(os.listdir(full)):
            if not fn.endswith(".npz"):
                continue
            p = os.path.join(full, fn)
            parts[rel(p)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            n += 1
            total += os.path.getsize(p)
    if not parts:
        return dict(kind="state", restorable=False, missing="嵌入目录为空或不存在")
    return dict(kind="state", restorable=False, value={"n_files": n, "bytes": total},
                parts=parts, note="嵌入；丢失须重跑编码，本工具不回滚")


def item_tracks_range() -> dict:
    """数据范围：各机位 summary 的处理水位（帧数/stride/fps），不哈希 40 MB 的 jsonl。"""
    d = os.path.join(ROOT, "output_crosscam/long35")
    rows = {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith("_summary.json"):
                continue
            try:
                s = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            except Exception:
                continue
            rows[fn.replace("_summary.json", "")] = {
                "frames_total": s.get("frames_total"),
                "frames_processed": s.get("frames_processed"),
                "stride": s.get("stride"),
                "fps": round(float(s.get("fps") or 0), 3),
                "resolution": s.get("resolution"),
            }
    if not rows:
        return dict(kind="state", restorable=False, missing="无 summary，数据范围不可知")
    return dict(kind="state", restorable=False, value={"per_camera": rows},
                note="已处理数据范围；重跑须用同一 stride/fps，否则口径变")


def item_labels() -> dict:
    base = "deliverable/gt_long35"
    files = ["gt_long35.json", "recheck_ai_qc.json"]
    parts, have = {}, []
    for f in files:
        p = os.path.join(ROOT, base, f)
        if os.path.isfile(p):
            parts[rel(p)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
            have.append(f)
    if not parts:
        return dict(kind="state", restorable=False, missing="真值集不存在")
    return dict(kind="state", restorable=False, value={"files": have}, parts=parts,
                note="人工真值；改动等于改评价基准，须走版本+复核记录")


def item_training() -> dict:
    """训练状态：有就记，没有就如实记 absent，不假装能恢复。"""
    cands = ["runs", "checkpoints", "output_train", "output_crosscam/train"]
    found = [c for c in cands if os.path.isdir(os.path.join(ROOT, c))]
    if not found:
        return dict(kind="state", restorable=False, absent=True,
                    note="当前没有训练态（无 optimizer/scheduler/rng/采样进度），"
                         "因此「周期训练中断续跑」暂不可验证；不要把它当成已验证能力")
    parts = {}
    for c in found:
        full = os.path.join(ROOT, c)
        for dp, dns, fns in os.walk(full):
            dns[:] = [x for x in dns if x not in ("__pycache__", ".git")]
            for fn in fns:
                if fn.endswith((".pt", ".pth", ".ckpt", ".state")):
                    p = os.path.join(dp, fn)
                    parts[rel(p)] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    return dict(kind="state", restorable=False, value={"dirs": found, "n_ckpt": len(parts)},
                parts=parts, note="训练态；续跑还缺 rng/采样进度，见 note")


def item_ledger() -> dict:
    p = os.path.join(ROOT, "ops/ops.db")
    if not os.path.isfile(p):
        return dict(kind="state", restorable=False, missing="台账库不存在")
    import sqlite3
    try:
        c = sqlite3.connect(p)
        runs = c.execute("select count(*) from run").fetchone()[0]
        claims = c.execute("select count(*) from claim").fetchone()[0]
        cards = c.execute("select count(*) from exp_card").fetchone()[0]
        c.close()
    except Exception:
        runs = claims = cards = None
    return dict(kind="state", restorable=False,
                value={"runs": runs, "claims": claims, "cards": cards},
                sha256=sha256_file(p), bytes=os.path.getsize(p),
                note="台账；记录了哪些 run 存在，但恢复实验状态不依赖它")


ITEMS = [
    ("code", item_code),
    ("pack", item_pack),
    ("gallery", item_gallery),
    ("pool", item_pool),
    ("graph", item_graph),
    ("embeddings", item_embeddings),
    ("tracks_range", item_tracks_range),
    ("labels", item_labels),
    ("training", item_training),
    ("ledger", item_ledger),
]


# ------------------------------------------------------------------ 操作
def cmd_snapshot(args) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rp_id = f"rp_{stamp}_{args.tag}" if args.tag else f"rp_{stamp}"
    out = os.path.join(RP_DIR, rp_id)
    os.makedirs(out, exist_ok=True)

    manifest = {"rp_id": rp_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": args.note or "", "root": ROOT, "items": {}}
    for name, fn in ITEMS:
        try:
            rec = fn()
        except Exception as e:  # noqa: BLE001
            rec = dict(kind="state", restorable=False, missing=f"采集失败：{e!r}")
        rec["name"] = name
        # 可恢复项：把原件复制进恢复点，否则 restore 无从下手
        if rec.get("restorable") and rec.get("files"):
            dst_dir = os.path.join(out, "files")
            os.makedirs(dst_dir, exist_ok=True)
            copies = []
            for src in rec["files"]:
                dst = os.path.join(dst_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                copies.append(rel(dst))
            rec["copies"] = copies
        manifest["items"][name] = rec

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"恢复点 {rp_id}")
    print("%-14s %-8s %-9s %s" % ("项", "状态", "可恢复", "摘要"))
    for name, rec in manifest["items"].items():
        st = "absent" if rec.get("absent") else ("missing" if rec.get("missing") else "ok")
        summary = rec.get("missing") or ""
        if not summary:
            v = rec.get("value") or {}
            if name == "code":
                summary = f"{v.get('git_rev')} 脏{v.get('dirty_files')} 跟踪{v.get('tracked_files')}"
            elif name == "pack":
                summary = v.get("pack", "")
            elif name == "gallery":
                summary = f"{v.get('model_version')} dim{v.get('dim')} n{v.get('n_tracks')}"
            elif name == "pool":
                p = v.get("params") or {}
                summary = f"{v.get('n_entries')} 条 σ{p.get('sigma')} k{p.get('pool_k')} minh{p.get('min_h')}"
            elif name == "embeddings":
                summary = f"{v.get('n_files')} 文件 {v.get('bytes', 0)/1048576:.1f} MB"
            elif name == "tracks_range":
                summary = f"{len(v.get('per_camera') or {})} 机位"
            elif name == "ledger":
                summary = f"runs={v.get('runs')} claims={v.get('claims')} cards={v.get('cards')}"
            else:
                summary = str(v)[:60]
        print("%-14s %-8s %-9s %s" % (name, st, "是" if rec.get("restorable") else "否", summary))
    print(f"\n-> {rel(out)}/manifest.json")
    return 0


def _load(rp_id: str) -> tuple[str, dict]:
    out = os.path.join(RP_DIR, rp_id)
    p = os.path.join(out, "manifest.json")
    if not os.path.isfile(p):
        raise SystemExit(f"找不到恢复点 {rp_id}（{rel(p)}）")
    return out, json.load(open(p, encoding="utf-8"))


def _current(name: str) -> dict:
    for n, fn in ITEMS:
        if n == name:
            return fn()
    return {}


def cmd_verify(args) -> int:
    out, man = _load(args.rp_id)
    print(f"核对 {args.rp_id}（建于 {man['created_at']}，{man.get('note') or '无备注'}）\n")
    print("%-14s %-10s %s" % ("项", "结果", "细节"))
    drift = 0
    for name, saved in man["items"].items():
        cur = _current(name)
        if saved.get("absent") or saved.get("missing"):
            if cur.get("absent") or cur.get("missing"):
                print("%-14s %-10s %s" % (name, "同前", saved.get("missing") or "absent"))
            else:
                print("%-14s %-10s 恢复点记 absent/missing，现在有了" % (name, "新增"))
                drift += 1
            continue
        # 比对：优先整体 sha，其次分件
        if saved.get("sha256") and cur.get("sha256"):
            ok = saved["sha256"] == cur["sha256"]
            d = "sha 一致" if ok else f"sha 变 {saved['sha256'][:8]}→{(cur.get('sha256') or '')[:8]}"
        elif saved.get("parts") and cur.get("parts"):
            a, b = saved["parts"], cur["parts"]
            miss = [k for k in a if k not in b]
            diff = [k for k in a if k in b and a[k]["sha256"] != b[k]["sha256"]]
            extra = [k for k in b if k not in a]
            ok = not (miss or diff or extra)
            d = "分件一致" if ok else f"缺{len(miss)} 变{len(diff)} 多{len(extra)}"
            if not ok:
                d += " " + str((miss + diff + extra)[:3])
        else:
            a = json.dumps(saved.get("value"), sort_keys=True, ensure_ascii=False)
            b = json.dumps(cur.get("value"), sort_keys=True, ensure_ascii=False)
            ok = a == b
            d = "值一致" if ok else "值变化"
            if not ok and name == "code":
                d += f" {saved['value'].get('git_rev')}→{cur['value'].get('git_rev')}"
        print("%-14s %-10s %s" % (name, "OK" if ok else "已变", d))
        if not ok:
            drift += 1
    print(f"\n漂移项 {drift} 个" + ("（与恢复点一致）" if drift == 0 else ""))
    return 0 if drift == 0 or args.tolerant else 1


def cmd_restore(args) -> int:
    out, man = _load(args.rp_id)
    only = set(x.strip() for x in (args.only or "").split(",") if x.strip())
    done, refused = [], []
    for name, rec in man["items"].items():
        if only and name not in only:
            continue
        if rec.get("missing") or rec.get("absent"):
            continue
        if not rec.get("restorable"):
            refused.append(name)
            continue
        for copy, src in zip(rec.get("copies") or [], rec.get("files") or []):
            cpath = os.path.join(ROOT, copy.replace("/", os.sep))
            if not os.path.isfile(cpath):
                print(f"  [{name}] 恢复点内副本缺失：{copy}", file=sys.stderr)
                continue
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.copy2(cpath, src)
            done.append(f"{name} → {rel(src)}")
    print(f"已恢复 {len(done)} 项")
    for d in done:
        print("  " + d)
    if refused:
        print(f"\n未恢复（需重跑或走版本流程）：{', '.join(refused)}")
        print("  嵌入 → python tools/reid_index.py（或对应编码脚本）")
        print("  身份图/真值 → 从 git 或交付包取回，并走 diff 确认")
        print("  训练态 → 当前不存在，无可恢复对象")
    return 0


def cmd_list(args) -> int:
    if not os.path.isdir(RP_DIR):
        print("（还没有恢复点）")
        return 0
    rows = []
    for name in sorted(os.listdir(RP_DIR)):
        p = os.path.join(RP_DIR, name, "manifest.json")
        if not os.path.isfile(p):
            continue
        m = json.load(open(p, encoding="utf-8"))
        bad = sum(1 for r in m["items"].values() if r.get("missing"))
        rows.append((m["created_at"], name, m.get("note") or "", bad))
    if not rows:
        print("（还没有恢复点）")
        return 0
    print("%-20s %-30s %-8s %s" % ("时间", "rp_id", "缺失项", "备注"))
    for t, rid, note, bad in rows:
        print("%-20s %-30s %-8d %s" % (t, rid, bad, note[:40]))
    return 0


def cmd_show(args) -> int:
    out, man = _load(args.rp_id)
    print(json.dumps(man, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="恢复点：快照 / 核对 / 回退")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="采集当前状态为一个恢复点")
    s.add_argument("--tag", default="", help="短标签，进 rp_id")
    s.add_argument("--note", default="", help="这次做了什么")
    s.set_defaults(fn=cmd_snapshot)

    v = sub.add_parser("verify", help="与当前状态比对，报漂移")
    v.add_argument("rp_id")
    v.add_argument("--tolerant", action="store_true", help="有漂移也返回 0")
    v.set_defaults(fn=cmd_verify)

    r = sub.add_parser("restore", help="回退可恢复项")
    r.add_argument("rp_id")
    r.add_argument("--only", default="", help="只恢复指定项，逗号分隔，如 pool,pack")
    r.set_defaults(fn=cmd_restore)

    l = sub.add_parser("list", help="列出所有恢复点")
    l.set_defaults(fn=cmd_list)

    sh = sub.add_parser("show", help="打印完整 manifest")
    sh.add_argument("rp_id")
    sh.set_defaults(fn=cmd_show)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
