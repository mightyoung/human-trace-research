#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验「疑似同人（换号）」候选：页面标注可以弱，但不能把两个人标成一个人。

检查项（任何一项非零即退出码 1）：
  1. 组内两两不得同帧共现 —— 共现即两人；这是硬约束，页面据此画虚线
  2. 每条 link 的时间必须顺（b 段起点 ≥ a 段终点）
  3. 组内所有 tid 必须在本机位 tracks 里存在，且同属一个机位
  4. gid 唯一、criteria 齐全
  5. link 的 a/b 实体必须在 entities.json 里，且不相等于同一实体（同实体已由实体图合并）

只读 webapp/backend/data（当前生效的数据包），不依赖原始 output_crosscam。

用法：
  python tools/check_suspects.py
  python tools/check_suspects.py --data webapp/backend/data_long35
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "webapp" / "backend" / "data"


def jload(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    data = Path(args.data)

    sus = jload(data / "suspects.json")
    if sus is None:
        print(f"[skip] {data/'suspects.json'} 不存在（老数据包或 --no-suspects 构建）")
        return 0
    ents = jload(data / "entities.json", {}) or {}

    errs: list[str] = []
    warn: list[str] = []

    # ---- 0. 结构 ----
    for k in ("criteria", "groups", "n_groups", "n_pairs"):
        if k not in sus:
            errs.append(f"缺少字段 {k}")
    groups = sus.get("groups") or []
    if sus.get("n_groups") != len(groups):
        errs.append(f"n_groups={sus.get('n_groups')} 与实际 {len(groups)} 不符")

    # ---- 1. 逐机位读 tracks（建帧集合）----
    tracks_cache: dict[str, dict] = {}

    def tracks_of(cam: str) -> dict:
        if cam not in tracks_cache:
            d = jload(data / "tracks" / f"{cam}.json", {}) or {}
            tracks_cache[cam] = d.get("tracks") or {}
        return tracks_cache[cam]

    seen_gid: set[str] = set()
    n_pairs = 0
    for g in groups:
        gid, cam = g.get("gid"), g.get("cam")
        if not gid or not cam:
            errs.append(f"组缺少 gid/cam: {g}")
            continue
        if gid in seen_gid:
            errs.append(f"gid 重复: {gid}")
        seen_gid.add(gid)

        tids = g.get("tids") or []
        tr = tracks_of(cam)
        for t in tids:
            if str(t) not in tr:
                errs.append(f"{gid}: tid {t} 不在 tracks/{cam}.json")
        # 组内两两共现
        frames = {}
        for t in tids:
            pts = tr.get(str(t)) or []
            frames[t] = {p[0] for p in pts}
        for a, b in itertools.combinations(tids, 2):
            ov = frames[a] & frames[b]
            if ov:
                errs.append(f"{gid}: t{a} 与 t{b} 同帧共现 {len(ov)} 帧（应为两人）")

        links = g.get("links") or []
        if not links:
            errs.append(f"{gid}: 无 links")
        n_pairs += len(links)
        for l in links:
            if l.get("b_t0", 0) < l.get("t_a_end", 0) - 1e-6:
                errs.append(f"{gid}: link {l.get('a_tid')}->{l.get('b_tid')} 时间倒挂")
            if l.get("cam") != cam:
                errs.append(f"{gid}: link cam={l.get('cam')} 与组 cam={cam} 不一致")
            ea, eb = l.get("a"), l.get("b")
            if ea == eb:
                errs.append(f"{gid}: link 两端是同一实体 {ea}（应已由实体图合并）")
            for e in (ea, eb):
                if e not in ((ents.get(cam) or {}).get("entities") or {}):
                    errs.append(f"{gid}: 实体 {e} 不在 entities.json[{cam}]")
            if l.get("grade") not in ("strong", "weak"):
                warn.append(f"{gid}: link grade={l.get('grade')} 非 strong/weak")
        if g.get("grade") not in ("strong", "weak"):
            errs.append(f"{gid}: grade={g.get('grade')}")

    if sus.get("n_pairs") != n_pairs:
        errs.append(f"n_pairs={sus.get('n_pairs')} 与实际 {n_pairs} 不符")

    # ---- 2. 汇总 ----
    cams: dict[str, int] = {}
    for g in groups:
        cams[g.get("cam")] = cams.get(g.get("cam"), 0) + 1
    by_grade: dict[str, int] = {}
    for g in groups:
        by_grade[g.get("grade")] = by_grade.get(g.get("grade"), 0) + 1
    max_chain = max((len(g.get("tids") or []) for g in groups), default=0)

    print(f"[data] {data}")
    print(f"[suspects] {len(groups)} 组 / {n_pairs} 段；"
          f"strong {by_grade.get('strong', 0)} · weak {by_grade.get('weak', 0)}；"
          f"最长链 {max_chain} 段")
    print(f"[cams] {cams}")
    print(f"[criteria] {json.dumps(sus.get('criteria'), ensure_ascii=False)}")
    if sus.get("n_rejected_by_overlap"):
        print(f"[链式传递拒绝] {sus['n_rejected_by_overlap']} 条（合并后组内会共现）")

    for w in warn[:10]:
        print(f"[warn] {w}")
    if errs:
        print(f"\n✗ {len(errs)} 个错误：")
        for e in errs[:30]:
            print("   -", e)
        return 1
    print("\n✓ 全部检查通过（组内无共现 / 时间不倒挂 / 实体存在 / 计数一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
